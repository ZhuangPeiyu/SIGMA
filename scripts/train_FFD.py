import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from highaccmaskgen.data import BRGenPairDataset, Sample, build_samples
from highaccmaskgen.metrics import f1, iou, precision, recall
from highaccmaskgen.sigma import (
    OfflineTextInputStore,
    build_ffd_net,
    causal_diff_loss,
    extract_instructions_from_batch as _extract_instructions_from_batch,
    extract_state_dict as _extract_state_dict,
    load_checkpoint_compatible as _load_checkpoint_compatible,
    resolve_hf_hub_cache_dir,
    state_dict_for_save as _state_dict_for_save,
)
from highaccmaskgen.utils import ensure_dir, load_yaml, set_seed


def _disable_inplace_activations(model: torch.nn.Module, log) -> int:
    changed = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.SiLU, torch.nn.ReLU)) and getattr(m, "inplace", False):
            m.inplace = False
            changed += 1
    if changed > 0:
        log(f"parallel: set inplace=False for {changed} activation layers")
    return changed


def _split_train_val(samples: list, val_ratio: float, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]
    return train_samples, val_samples


def _load_pairs_json_as_samples(
    json_path: Path, split_name: str, *, data_source: str = "deal300k"
) -> list[Sample]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must be a JSON list.")
    samples: list[Sample] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        real_p = row.get("real_image_path")
        fake_p = row.get("fake_image_path")
        mask_p = row.get("mask_image_path")
        if not real_p or not fake_p or not mask_p:
            continue
        real_path = Path(str(real_p))
        forged_path = Path(str(fake_p))
        mask_path = Path(str(mask_p))
        # if not (real_path.is_file() and forged_path.is_file() and mask_path.is_file()):
        #     continue
        s = Sample(
            method="pair_json",
            domain=str(row.get("split", split_name)),
            split=split_name,
            key=str(row.get("image_id", row.get("fake_row_uid", i))),
            real_path=real_path,
            forged_path=forged_path,
            mask_path=mask_path,
        )
        setattr(s, "instruction", str(row.get("instruction", "") or ""))
        setattr(s, "data_source", str(data_source))
        samples.append(s)
    return samples


@torch.no_grad()
def _evaluate_with_loss(
    model,
    loader,
    device: str,
    tcfg: dict,
    use_online_text: bool = False,
    offline_text_store: Optional[OfflineTextInputStore] = None,
) -> dict:
    model.eval()
    losses = []
    ious = []
    f1s = []
    ps = []
    rs = []
    for batch in loader:
        before = batch["real"].to(device, non_blocking=True)
        after = batch["forged"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        if offline_text_store is not None:
            text_inputs = offline_text_store.build_batch_text_inputs(batch, device=device)
            if text_inputs is not None:
                outputs = model(before, after, text_inputs=text_inputs)
            elif use_online_text:
                instructions = _extract_instructions_from_batch(batch, before.shape[0])
                outputs = model(before, after, instructions=instructions)
            else:
                outputs = model(before, after)
        elif use_online_text:
            instructions = _extract_instructions_from_batch(batch, before.shape[0])
            outputs = model(before, after, instructions=instructions)
        else:
            outputs = model(before, after)
        loss = causal_diff_loss(
            outputs,
            target,
            w_bce=float(tcfg.get("w_bce", 1.0)),
            w_dice=float(tcfg.get("w_dice", 1.0)),
            w_boundary=float(tcfg.get("w_boundary", 0.4)),
            w_smooth=float(tcfg.get("w_smooth", 0.02)),
        )
        losses.append(float(loss.item()))
        logits = outputs["mask_logits"]
        pred = (torch.sigmoid(logits).cpu().numpy()[:, 0] >= 0.5).astype(np.uint8)
        gt = target.cpu().numpy()[:, 0]
        for p, g in zip(pred, gt):
            ious.append(iou(p, g))
            f1s.append(f1(p, g))
            ps.append(precision(p, g))
            rs.append(recall(p, g))
    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "IoU": float(np.mean(ious)) if ious else 0.0,
        "F1": float(np.mean(f1s)) if f1s else 0.0,
        "Precision": float(np.mean(ps)) if ps else 0.0,
        "Recall": float(np.mean(rs)) if rs else 0.0,
    }


def _plot_curves(records: list[dict], outdir: Path) -> None:
    if not records:
        return
    epochs = [r["epoch"] for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_loss = [r["val_loss"] for r in records]
    ious = [r["val_iou"] for r in records]
    f1s = [r["val_f1"] for r in records]
    ps = [r["val_precision"] for r in records]
    rs = [r["val_recall"] for r in records]
    lrs = [r["lr"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, marker="o", label="train_loss")
    ax.plot(epochs, val_loss, marker="o", label="val_loss")
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, ious, marker="o", label="val_IoU")
    ax.plot(epochs, f1s, marker="o", label="val_F1")
    ax.set_title("Segmentation Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, ps, marker="o", label="val_Precision")
    ax.plot(epochs, rs, marker="o", label="val_Recall")
    ax.set_title("Precision / Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, lrs, marker="o", label="learning_rate")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(outdir / "curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ffd_local_ablations.yaml")
    parser.add_argument("--subset", type=int, default=0, help="0 means use all samples")
    parser.add_argument("--epochs", type=int, default=0, help="0 means use config value")
    parser.add_argument("--outdir", type=str, default="outputs/ffd")
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HuggingFace endpoint mirror used when downloading pretrained weights.",
    )
    parser.add_argument(
        "--encoder-pretrained",
        dest="encoder_pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to load pretrained encoder weights.",
    )
    parser.add_argument(
        "--hf-hub-cache",
        type=str,
        default="",
        help="HF hub cache directory for timm backbone weights.",
    )
    parser.add_argument(
        "--ffd-preset",
        type=str,
        default="",
        help="Named preset in FFD_PRESETS (e.g. ffd_baseline, ffd_sem_only, ffd_all_strategies).",
    )
    parser.add_argument(
        "--ablation-preset",
        type=str,
        default="",
        help="Alias of --ffd-preset for compatibility.",
    )
    parser.add_argument(
        "--multiscale-msdm",
        dest="multiscale_msdm",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--num-drb-blocks", type=int, default=-1)
    parser.add_argument(
        "--use-pixel-aux",
        dest="use_pixel_aux",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-sem-branch",
        dest="use_sem_branch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint (.pt): raw state_dict or dict with state_dict/model/net.",
    )
    parser.add_argument("--resume-non-strict", action="store_true")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--append-logs", action="store_true")
    parser.add_argument("--initial-best-iou", type=float, default=-1.0)
    parser.add_argument(
        "--data-parallel",
        dest="data_parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--val-every-iters",
        type=int,
        default=500,
        help="Print current train loss every N train iterations (0 = disabled).",
    )
    parser.add_argument(
        "--sample-build-workers",
        type=int,
        default=0,
        help=(
            "Worker threads for dataset sample discovery when not using pair-json "
            "(0 = auto, 1 = disable parallelism)."
        ),
    )
    parser.add_argument("--train-pairs-json", type=str, default="", help="Optional train pairs JSON file.")
    parser.add_argument(
        "--brgen-train-pairs-json",
        type=str,
        default="",
        help="Optional BRGen train pairs JSON (same schema as train-pairs). Merged into train only; no instruction is ok.",
    )
    parser.add_argument("--dev-pairs-json", type=str, default="", help="Optional dev pairs JSON file.")
    parser.add_argument(
        "--online-text-branch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable online parser+LangSAM forward path (requires text branch).",
    )
    parser.add_argument(
        "--acsf-batch-root",
        type=str,
        default="",
        help="Optional precomputed acsf_batch root for offline text_inputs loading.",
    )
    parser.add_argument(
        "--acsf-batch-train-root",
        type=str,
        default="",
        help="Optional train-only precomputed acsf_batch root for offline text_inputs loading.",
    )
    parser.add_argument(
        "--acsf-batch-dev-root",
        type=str,
        default="",
        help="Optional dev-only precomputed acsf_batch root for offline text_inputs loading.",
    )
    parser.add_argument(
        "--semantic-init-ckpt",
        type=str,
        default="",
        help="Optional semantic-only checkpoint used for initialization before training.",
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    outdir = ensure_dir(args.outdir)
    log_txt = outdir / "train.log"
    log_jsonl = outdir / "metrics.jsonl"
    log_csv = outdir / "metrics.csv"
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if not args.append_logs:
        for p in (log_txt, log_jsonl, log_csv):
            if p.exists():
                p.unlink()
        csv_mode = "w"
        csv_write_header = True
    else:
        csv_mode = "a" if log_csv.exists() else "w"
        csv_write_header = not (log_csv.exists() and log_csv.stat().st_size > 0)

    with open(log_csv, csv_mode, encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "event_type",
                "epoch",
                "iter_in_epoch",
                "global_step",
                "train_loss",
                "val_loss",
                "lr",
                "val_iou",
                "val_f1",
                "val_precision",
                "val_recall",
                "is_best",
            ],
        )
        if csv_write_header:
            writer.writeheader()

    paths = cfg["paths"]
    dcfg = cfg["dataset"]
    tcfg = cfg["train"]
    acfg = cfg.get("augment", {})
    forged_root = Path(paths["forged_root"])
    mask_root = Path(paths["mask_root"])
    real_root = Path(paths["real_root"])

    def _norm_to_list(v):
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return [str(v)]

    train_pairs_json = Path(args.train_pairs_json).expanduser() if str(args.train_pairs_json).strip() else None
    dev_pairs_json = Path(args.dev_pairs_json).expanduser() if str(args.dev_pairs_json).strip() else None
    if train_pairs_json is not None:
        if not train_pairs_json.is_file():
            raise FileNotFoundError(f"--train-pairs-json not found: {train_pairs_json}")
        n_deal = 0
        n_br = 0
        all_train_like_samples = _load_pairs_json_as_samples(
            train_pairs_json, split_name="train", data_source="deal300k"
        )
        n_deal = len(all_train_like_samples)
        brgen_json = Path(str(args.brgen_train_pairs_json).strip()).expanduser() if str(
            args.brgen_train_pairs_json
        ).strip() else None
        if brgen_json is not None:
            if not brgen_json.is_file():
                raise FileNotFoundError(f"--brgen-train-pairs-json not found: {brgen_json}")
            brgen_part = _load_pairs_json_as_samples(
                brgen_json, split_name="train", data_source="brgen"
            )
            n_br = len(brgen_part)
            all_train_like_samples = all_train_like_samples + brgen_part
            rng_mix = np.random.default_rng(int(cfg["seed"]) + 17)
            order = np.arange(len(all_train_like_samples))
            rng_mix.shuffle(order)
            all_train_like_samples = [all_train_like_samples[int(i)] for i in order.tolist()]

        if dev_pairs_json is not None:
            if not dev_pairs_json.is_file():
                raise FileNotFoundError(f"--dev-pairs-json not found: {dev_pairs_json}")
            train_samples = all_train_like_samples
            val_samples = _load_pairs_json_as_samples(
                dev_pairs_json, split_name="dev", data_source="deal300k"
            )
        else:
            # When no explicit dev json is provided (e.g. BRGen-only training),
            # split validation from the train list by configured val_ratio.
            train_samples, val_samples = _split_train_val(
                all_train_like_samples, float(dcfg["val_ratio"]), int(cfg["seed"])
            )
        if args.subset > 0:
            train_samples = train_samples[: args.subset]
            val_samples = val_samples[: max(1, min(args.subset, len(val_samples)))]
        if len(train_samples) < 4 or len(val_samples) < 1:
            raise RuntimeError(
                f"Not enough samples from pair json. train={len(train_samples)} val={len(val_samples)}"
            )
        if n_br > 0:
            log(
                f"dataset.source=pair_json mix=deal300k+brgen train_deal300k={n_deal} train_brgen={n_br} "
                f"train_total={len(train_samples)} val_deal300k={len(val_samples)}"
            )
        elif dev_pairs_json is None:
            log(
                f"dataset.source=pair_json split_from_train val_ratio={float(dcfg['val_ratio']):.4f} "
                f"train={len(train_samples)} val={len(val_samples)}"
            )
        else:
            log(f"dataset.source=pair_json train={len(train_samples)} val={len(val_samples)}")
    else:
        split_cfg = dcfg.get("split", "stuff")
        domain_cfg = dcfg.get("domain", "COCO")
        split_list = [s.lower().strip() for s in _norm_to_list(split_cfg)]
        if len(split_list) == 1 and split_list[0] == "all":
            split_list = ["stuff", "background"]
        domain_list = [d.strip() for d in _norm_to_list(domain_cfg)]
        if len(domain_list) == 1 and domain_list[0].lower() == "all":
            domain_list = sorted([p.name for p in real_root.iterdir() if p.is_dir()])

        log(f"dataset.scope: splits={split_list} domains={domain_list}")
        samples: list[Sample] = []
        seen: set[tuple[str, str, str, str]] = set()

        def _discover_one_scope(sp: str, dm: str) -> list[Sample]:
            return build_samples(
                forged_root=forged_root,
                mask_root=mask_root,
                real_root=real_root,
                methods=dcfg["methods"],
                domain=dm,
                split=sp,
            )

        scope_pairs = [(sp, dm) for sp in split_list for dm in domain_list]
        workers = int(args.sample_build_workers)
        if workers <= 0:
            workers = min(8, max(1, os.cpu_count() or 1))
        if workers == 1 or len(scope_pairs) <= 1:
            for sp, dm in scope_pairs:
                for s in _discover_one_scope(sp, dm):
                    key = (s.method, s.domain, s.split, s.key)
                    if key in seen:
                        continue
                    seen.add(key)
                    samples.append(s)
        else:
            log(f"dataset.discovery.parallel workers={workers} scopes={len(scope_pairs)}")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_discover_one_scope, sp, dm): (sp, dm) for sp, dm in scope_pairs}
                for fut in as_completed(futures):
                    part = fut.result()
                    for s in part:
                        key = (s.method, s.domain, s.split, s.key)
                        if key in seen:
                            continue
                        seen.add(key)
                        samples.append(s)
        if args.subset > 0:
            samples = samples[: args.subset]
        if len(samples) < 4:
            raise RuntimeError(f"Not enough samples to train. Found: {len(samples)}")
        log(f"samples_total={len(samples)}")
        train_samples, val_samples = _split_train_val(samples, float(dcfg["val_ratio"]), int(cfg["seed"]))

    acsf_train_root = str(args.acsf_batch_train_root).strip()
    acsf_dev_root = str(args.acsf_batch_dev_root).strip()
    acsf_legacy_root = str(args.acsf_batch_root).strip()
    # Backward compatibility: if split-specific roots are absent, reuse legacy root.
    if not acsf_train_root and acsf_legacy_root:
        acsf_train_root = acsf_legacy_root
    if not acsf_dev_root and acsf_legacy_root:
        acsf_dev_root = acsf_legacy_root

    offline_text_store_train: Optional[OfflineTextInputStore] = None
    offline_text_store_dev: Optional[OfflineTextInputStore] = None
    if acsf_train_root:
        offline_text_store_train = OfflineTextInputStore(Path(acsf_train_root).expanduser())
    if acsf_dev_root:
        offline_text_store_dev = OfflineTextInputStore(Path(acsf_dev_root).expanduser())

    if offline_text_store_train is not None or offline_text_store_dev is not None:
        n_train_before, n_val_before = len(train_samples), len(val_samples)
        if offline_text_store_train is not None:
            train_samples = [
                s
                for s in train_samples
                if getattr(s, "data_source", "deal300k") == "brgen" or offline_text_store_train.has_sample(s)
            ]
        if offline_text_store_dev is not None:
            val_samples = [s for s in val_samples if offline_text_store_dev.has_sample(s)]
        train_desc = acsf_train_root if offline_text_store_train is not None else "<disabled>"
        dev_desc = acsf_dev_root if offline_text_store_dev is not None else "<disabled>"
        train_key_n = len(offline_text_store_train.index) if offline_text_store_train is not None else 0
        train_path_n = len(offline_text_store_train.path_index) if offline_text_store_train is not None else 0
        dev_key_n = len(offline_text_store_dev.index) if offline_text_store_dev is not None else 0
        dev_path_n = len(offline_text_store_dev.path_index) if offline_text_store_dev is not None else 0
        log(
            "offline_text_inputs=enabled "
            f"train_root={train_desc} train_indexed_key={train_key_n} train_indexed_path={train_path_n} "
            f"dev_root={dev_desc} dev_indexed_key={dev_key_n} dev_indexed_path={dev_path_n} "
            f"filtered_train={len(train_samples)}/{n_train_before} filtered_val={len(val_samples)}/{n_val_before}"
        )
        if len(train_samples) < 4 or len(val_samples) < 1:
            raise RuntimeError(
                f"Not enough offline-matched samples after filtering. "
                f"train={len(train_samples)} val={len(val_samples)}"
            )

    train_ds = BRGenPairDataset(
        train_samples,
        image_size=int(dcfg["image_size"]),
        enable_augment=bool(acfg.get("enable_pixel_drift_aug", False)),
        augment_cfg=acfg,
    )
    val_ds = BRGenPairDataset(val_samples, image_size=int(dcfg["image_size"]), enable_augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(dcfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(dcfg["num_workers"]),
        pin_memory=True,
    )

    mcfg = dict(cfg.get("model", {}))
    if str(args.hf_hub_cache).strip():
        mcfg["hf_hub_cache"] = str(args.hf_hub_cache).strip()
    if args.ffd_preset:
        mcfg["ffd_preset"] = args.ffd_preset.strip()
    if args.ablation_preset:
        mcfg["ablation_preset"] = args.ablation_preset.strip()
    if args.multiscale_msdm is not None:
        mcfg["multiscale_msdm"] = args.multiscale_msdm
    if args.num_drb_blocks >= 0:
        mcfg["num_drb_blocks"] = int(args.num_drb_blocks)
    if args.use_pixel_aux is not None:
        mcfg["use_pixel_aux"] = args.use_pixel_aux
    if args.use_sem_branch is not None:
        mcfg["use_sem_branch"] = args.use_sem_branch

    encoder_pretrained = (
        bool(args.encoder_pretrained)
        if args.encoder_pretrained is not None
        else bool(mcfg.get("encoder_pretrained", True))
    )
    log(f"encoder_pretrained={encoder_pretrained}")
    _hub = resolve_hf_hub_cache_dir(str(mcfg.get("hf_hub_cache", "")).strip() or None)
    log(f"hf_hub_cache effective={_hub!r}")

    model = build_ffd_net(mcfg, encoder_pretrained=encoder_pretrained).to(device)
    log(
        f"ffd: preset={mcfg.get('ffd_preset', mcfg.get('ablation_preset', ''))!r} "
        f"multiscale_msdm={model.multiscale_msdm} num_drb_blocks={len(model.drb_blocks)} "
        f"use_pixel_aux={model.use_pixel_aux} "
        f"use_sem={model.use_sem_branch} "
        f"use_text={getattr(model, 'use_text_branch', False)}"
    )
    use_online_text = bool(
        args.online_text_branch
        if args.online_text_branch is not None
        else mcfg.get("enable_online_text_modules", False)
    )
    log(f"online_text_branch={use_online_text}")

    n_gpu = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    if bool(args.data_parallel) and n_gpu > 1:
        _disable_inplace_activations(model, log)
        model = torch.nn.DataParallel(model)
        log(f"parallel: enabled torch.nn.DataParallel on {n_gpu} GPUs")
    else:
        reason = "disabled by flag" if not bool(args.data_parallel) else "insufficient devices"
        log(f"parallel: single-device mode ({device}) [{reason}]")

    start_epoch = max(0, int(args.start_epoch))
    best_iou = float(args.initial_best_iou)
    global_step = 0

    def _load_checkpoint_with_report(ckpt_path: Path, *, strict: bool, tag: str) -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"{tag} not found: {ckpt_path}")
        try:
            ckpt_obj = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        except TypeError:
            ckpt_obj = torch.load(str(ckpt_path), map_location=device)
        sd = _extract_state_dict(ckpt_obj)
        stats = _load_checkpoint_compatible(model, sd, strict=strict, log=log)
        loaded = int(stats["loaded"])
        total = int(stats["from_ckpt"])
        status = "ok" if loaded > 0 else "empty"
        log(
            f"{tag}: checkpoint={ckpt_path} mode={stats['mode']} adapted={stats['adapted']} "
            f"loaded={loaded}/{total} missing={stats['missing']} unexpected={stats['unexpected']} "
            f"status={status}"
        )

    if str(args.semantic_init_ckpt).strip():
        sem_ckpt_path = Path(str(args.semantic_init_ckpt).strip()).expanduser()
        # semantic-only -> current branch typically needs non-strict compatible loading.
        _load_checkpoint_with_report(sem_ckpt_path, strict=False, tag="semantic_init")

    if str(args.resume).strip():
        ckpt_path = Path(str(args.resume).strip()).expanduser()
        _load_checkpoint_with_report(ckpt_path, strict=not args.resume_non_strict, tag="resume")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    base_lr = float(tcfg["lr"])
    lr_floor_cfg = float(tcfg.get("lr_floor", 1e-4))
    lr_floor = min(lr_floor_cfg, base_lr)
    epochs = int(args.epochs or tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, epochs),
        eta_min=lr_floor,
    )
    log(f"lr scheduler: cosine eta_min={lr_floor:.7f} (base_lr={base_lr:.7f})")

    all_records: list[dict] = []
    train_log_every_iters = int(args.val_every_iters or tcfg.get("log_every_iters", 0))

    def write_record(record: dict) -> None:
        if record.get("event_type") == "epoch_eval":
            all_records.append(record)
        with open(log_jsonl, "a", encoding="utf-8") as fj:
            fj.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(log_csv, "a", encoding="utf-8", newline="") as fcsv:
            writer = csv.DictWriter(
                fcsv,
                fieldnames=[
                    "event_type",
                    "epoch",
                    "iter_in_epoch",
                    "global_step",
                    "train_loss",
                    "val_loss",
                    "lr",
                    "val_iou",
                    "val_f1",
                    "val_precision",
                    "val_recall",
                    "is_best",
                ],
            )
            writer.writerow(record)

    def eval_and_log(
        *,
        event_type: str,
        epoch_idx: int,
        iter_in_epoch: int,
        train_loss_avg: float,
    ) -> tuple[dict, bool]:
        nonlocal best_iou
        val_metrics = _evaluate_with_loss(
            model,
            val_loader,
            device=device,
            tcfg=tcfg,
            use_online_text=use_online_text,
            offline_text_store=offline_text_store_dev,
        )
        is_best = val_metrics["IoU"] > best_iou
        log(
            f"{event_type} ep={epoch_idx:03d} iter={iter_in_epoch:05d} step={global_step:07d} "
            f"train_loss={train_loss_avg:.4f} val_loss={val_metrics['val_loss']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.7f} val_IoU={val_metrics['IoU']:.4f} "
            f"val_F1={val_metrics['F1']:.4f} val_P={val_metrics['Precision']:.4f} "
            f"val_R={val_metrics['Recall']:.4f}"
        )
        record = {
            "event_type": event_type,
            "epoch": epoch_idx,
            "iter_in_epoch": iter_in_epoch,
            "global_step": global_step,
            "train_loss": train_loss_avg,
            "val_loss": float(val_metrics["val_loss"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "val_iou": float(val_metrics["IoU"]),
            "val_f1": float(val_metrics["F1"]),
            "val_precision": float(val_metrics["Precision"]),
            "val_recall": float(val_metrics["Recall"]),
            "is_best": bool(is_best),
        }
        write_record(record)
        if is_best:
            best_iou = float(val_metrics["IoU"])
            prefix = f"ep{epoch_idx:03d}_f1{val_metrics['F1']:.4f}_iou{val_metrics['IoU']:.4f}"
            best_named = outdir / f"{prefix}_best.pt"
            sd_to_save = _state_dict_for_save(model)
            torch.save(sd_to_save, best_named)
            torch.save(sd_to_save, outdir / "best.pt")
            log(f"saved best checkpoint to: {best_named} (alias: {outdir / 'best.pt'})")
        return val_metrics, is_best

    for ep in range(epochs):
        ep_idx = start_epoch + ep + 1
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"train ep {ep_idx}/{start_epoch + epochs}")
        for it, batch in enumerate(pbar, start=1):
            before = batch["real"].to(device, non_blocking=True)
            after = batch["forged"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            if offline_text_store_train is not None:
                text_inputs = offline_text_store_train.build_batch_text_inputs(batch, device=device)
                if text_inputs is not None:
                    outputs = model(before, after, text_inputs=text_inputs)
                elif use_online_text:
                    instructions = _extract_instructions_from_batch(batch, before.shape[0])
                    outputs = model(before, after, instructions=instructions)
                else:
                    outputs = model(before, after)
            elif use_online_text:
                instructions = _extract_instructions_from_batch(batch, before.shape[0])
                outputs = model(before, after, instructions=instructions)
            else:
                outputs = model(before, after)
            loss = causal_diff_loss(
                outputs,
                target,
                w_bce=float(tcfg.get("w_bce", 1.0)),
                w_dice=float(tcfg.get("w_dice", 1.0)),
                w_boundary=float(tcfg.get("w_boundary", 0.4)),
                w_smooth=float(tcfg.get("w_smooth", 0.02)),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 1.0)))
            opt.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
            global_step += 1

            if train_log_every_iters > 0 and global_step % train_log_every_iters == 0:
                train_loss_avg = float(np.mean(losses)) if losses else 0.0
                log(
                    f"train_iter ep={ep_idx:03d} iter={it:05d} step={global_step:07d} "
                    f"train_loss={train_loss_avg:.4f} lr={scheduler.get_last_lr()[0]:.7f}"
                )
                write_record(
                    {
                        "event_type": "train_iter",
                        "epoch": ep_idx,
                        "iter_in_epoch": it,
                        "global_step": global_step,
                        "train_loss": train_loss_avg,
                        "val_loss": None,
                        "lr": float(scheduler.get_last_lr()[0]),
                        "val_iou": None,
                        "val_f1": None,
                        "val_precision": None,
                        "val_recall": None,
                        "is_best": False,
                    }
                )

        scheduler.step()
        avg_loss = float(np.mean(losses)) if losses else 0.0
        val_metrics, _ = eval_and_log(
            event_type="epoch_eval",
            epoch_idx=ep_idx,
            iter_in_epoch=len(train_loader),
            train_loss_avg=avg_loss,
        )

    last_prefix = f"ep{start_epoch + epochs:03d}_f1{val_metrics['F1']:.4f}_iou{val_metrics['IoU']:.4f}"
    last_named = outdir / f"{last_prefix}_last.pt"
    sd_to_save = _state_dict_for_save(model)
    torch.save(sd_to_save, last_named)
    torch.save(sd_to_save, outdir / "last.pt")
    log(f"saved last checkpoint to: {last_named} (alias: {outdir / 'last.pt'})")

    summary = {
        "best_iou": float(best_iou),
        "metrics_jsonl": str(log_jsonl),
        "metrics_csv": str(log_csv),
        "log_txt": str(log_txt),
        "train_log_every_iters": int(train_log_every_iters),
        "lr_floor": float(lr_floor),
        "model_variant": "ffd",
        "model_config": {
            k: mcfg.get(k)
            for k in (
                "ffd_preset",
                "ablation_preset",
                "multiscale_msdm",
                "num_drb_blocks",
                "use_pixel_aux",
                "use_sem_branch",
                "encoder_name",
                "hf_hub_cache",
            )
            if k in mcfg
        },
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as fs:
        json.dump(summary, fs, ensure_ascii=False, indent=2)
    _plot_curves(all_records, outdir)
    log(f"saved summary/curves to: {outdir}")


if __name__ == "__main__":
    main()
