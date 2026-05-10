#!/usr/bin/env python3
"""
Inference-only test on CrispEdit-style decoded folders **without GT masks**.

Index files ``records.jsonl`` are read from ``--records-root/<category>/`` (default: repo
``outputs/crip2m/``, synced from NAS via ``scripts/sync_crip2m_records_jsonl.sh``).
Relative paths inside each row are resolved against ``--decoded-root`` (the real decoded_data tree
where ``input_img`` / ``output_img`` files live — typically still on NAS).

Env / CLI patterns mirror ``eval_ffd_ablation_matrix.sh`` (``PARSER_MODEL_PATH``, ``ACSF_EVAL_ROOT``, …).

Example:
  ./scripts/sync_crip2m_records_jsonl.sh
  PYTHONPATH=src python scripts/test_without_gt.py \\
    --ckpt /path/to/best.pt \\
    --decoded-root /mnt/ugnas/upload/Text2Img/CrispEdit-2M/decoded_data \\
    --records-root outputs/crip2m \\
    --n 8 --seed 42 \\
    --ffd-preset ffd_msdm_drb2_textbranch --online-text-branch \\
    --parser-model-path \"$PARSER_MODEL_PATH\"
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from highaccmaskgen.data import CrispEditSample, build_crispedit_samples_from_jsonl
from highaccmaskgen.sigma import (
    OfflineTextInputStore,
    build_ffd_net,
    extract_instructions_from_batch as _extract_instructions_from_batch,
    load_checkpoint,
    resolve_hf_hub_cache_dir,
)
from highaccmaskgen.utils import ensure_dir, load_yaml, set_seed

_DEFAULT_CATEGORIES = ("add", "background_change", "color", "remove", "replace")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RECORDS_ROOT = _REPO_ROOT / "outputs" / "crip2m"
_DEFAULT_OUT_ROOT = Path("/mnt/ugnas/upload/Text2Img/SIGMA_preds")
_DATASET_DEFAULTS = {
    "anyedit": {
        "decoded_root": "/mnt/ugnas/upload/Text2Img/AnyEdit/decoded_data",
        "index_path": "/mnt/ugnas/upload/Text2Img/AnyEdit/decoded_data/all_records.jsonl",
        "instruction_field": "edit_instruction",
    },
    "crispedit2m": {
        "decoded_root": "/mnt/ugnas/upload/Text2Img/CrispEdit-2M/decoded_data",
        "index_path": "/mnt/ugnas/upload/Text2Img/CrispEdit-2M/decoded_data/all_records.jsonl",
        "instruction_field": "instruction",
    },
    "promptfixdata": {
        "decoded_root": "/mnt/ugnas/upload/Text2Img/PromptfixData/decoded_data",
        "index_path": "/mnt/ugnas/upload/Text2Img/PromptfixData/decoded_data/all_records.json",
        "instruction_field": "instruction",
    },
    "omnieditfiltered12m": {
        "decoded_root": "/mnt/ugnas/upload/Text2Img/OmniEdit-Filtered-1.2M_withInst/decode_data",
        "index_path": "/mnt/ugnas/upload/Text2Img/OmniEdit-Filtered-1.2M_withInst/decode_data/all_records.json",
        "instruction_field": "edited_prompt_list",
    },
}

def _log(msg: str) -> None:
    print(msg, flush=True)


def _to_repo_relative(path_like: str | Path) -> str:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except Exception:
        return str(p)


def _load_ffd_state(model: torch.nn.Module, ckpt_path: Path, device: str) -> None:
    stats = load_checkpoint(model, ckpt_path, device=device, strict=False, log=_log)
    _log(
        f"load ckpt: loaded={stats['loaded']}/{stats['from_ckpt']} "
        f"missing={stats['missing']} unexpected={stats['unexpected']}"
    )


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_rgb_to_hw(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _resize_rgb_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _safe_save_stem(sample: CrispEditSample, idx: int) -> str:
    raw = f"{sample.category}_{idx:04d}_{sample.key}"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)[:200]


def _samples_tensors_hw(
    sample: CrispEditSample,
    infer_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    forged_rgb = _read_rgb(sample.output_path)
    h, w = int(forged_rgb.shape[0]), int(forged_rgb.shape[1])
    real_rgb = _read_rgb(sample.input_path)
    real_hw = _resize_rgb_to_hw(real_rgb, w, h)
    real_s = _resize_rgb_square(real_hw, infer_size)
    forged_s = _resize_rgb_square(forged_rgb, infer_size)
    before_t = torch.from_numpy(real_s).permute(2, 0, 1).float().to(device) / 255.0
    after_t = torch.from_numpy(forged_s).permute(2, 0, 1).float().to(device) / 255.0
    return before_t.unsqueeze(0), after_t.unsqueeze(0), h, w


def _ffd_forward_one(
    model: torch.nn.Module,
    before: torch.Tensor,
    after: torch.Tensor,
    instruction: str,
    *,
    use_online_text: bool,
    offline_store: object | None,
    can_use_instruction_llm: bool,
) -> dict[str, torch.Tensor]:
    bsz = before.shape[0]
    batch = {
        "real": before,
        "forged": after,
        "meta": {
            "key": ["infer"],
            "real_path": ["dummy_real"],
            "forged_path": ["dummy_forged"],
            "instruction": [instruction],
            "data_source": ["deal300k"],
        },
    }

    def _any_instruction_text() -> bool:
        insts = batch["meta"].get("instruction")
        if not isinstance(insts, (list, tuple)):
            return bool(str(insts or "").strip())
        return any(str(x or "").strip() for x in insts)

    if offline_store is not None:
        text_inputs = offline_store.build_batch_text_inputs(batch, device=str(before.device))
        if text_inputs is not None:
            return model(before, after, text_inputs=text_inputs)
        if use_online_text and _any_instruction_text() and can_use_instruction_llm:
            inst = _extract_instructions_from_batch(batch, bsz)
            if inst is None:
                return model(before, after)
            return model(before, after, instructions=inst)
        return model(before, after)
    if use_online_text and _any_instruction_text() and can_use_instruction_llm:
        inst = _extract_instructions_from_batch(batch, bsz)
        if inst is None:
            return model(before, after)
        return model(before, after, instructions=inst)
    return model(before, after)


def _pred_prob_hw(
    logits: torch.Tensor,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    prob = torch.sigmoid(logits).float()[0, 0:1]
    prob_hw = F.interpolate(
        prob.unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return prob_hw.detach().cpu().numpy().astype(np.float32)


def _save_triplet_figure(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    pred_prob: np.ndarray,
    instruction: str,
    save_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(before_rgb)
    axes[0].set_title("original (input)")
    axes[1].imshow(after_rgb)
    axes[1].set_title("edited (output)")
    wrapped = textwrap.fill(instruction.strip() or "(no instruction)", width=48)
    axes[1].set_xlabel(wrapped, fontsize=9)
    axes[2].imshow(pred_prob, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("predicted mask (prob)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _pick_random_samples(samples: list[CrispEditSample], n: int, rng: np.random.Generator) -> list[CrispEditSample]:
    if n <= 0 or not samples:
        return []
    if len(samples) <= n:
        idx = np.arange(len(samples))
        rng.shuffle(idx)
        return [samples[int(i)] for i in idx]
    pick = rng.choice(len(samples), size=n, replace=False)
    return [samples[int(i)] for i in pick]


def _normalize_dataset_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _parse_categories(raw: str) -> list[str]:
    return [c.strip() for c in str(raw or "").split(",") if c.strip()]


def _discover_dataset_categories(image_root: Path) -> list[str]:
    out: list[str] = []
    try:
        for p in image_root.iterdir():
            if not p.is_dir():
                continue
            # CrispEdit-style: <cat>/{input_img,output_img}
            if (p / "input_img").is_dir() and (p / "output_img").is_dir():
                out.append(p.name)
                continue
            # Promptfix-like style: <cat>/{original_image,edited_image}
            if (p / "original_image").is_dir() and (p / "edited_image").is_dir():
                out.append(p.name)
                continue
    except Exception:
        return []
    return out


def _json_record_edit_pair_paths(rec: dict) -> tuple[str | None, str | None]:
    inp_obj = rec.get("input_img")
    out_obj = rec.get("output_img")
    if isinstance(inp_obj, dict) and isinstance(out_obj, dict):
        sp_in = inp_obj.get("saved_path")
        sp_out = out_obj.get("saved_path")
        if sp_in and sp_out:
            return str(sp_in), str(sp_out)
    orig_obj = rec.get("original_image")
    edit_obj = rec.get("edited_image")
    if isinstance(orig_obj, dict) and isinstance(edit_obj, dict):
        sp_in = orig_obj.get("saved_path")
        sp_out = edit_obj.get("saved_path")
        if sp_in and sp_out:
            return str(sp_in), str(sp_out)
    return None, None


def _instruction_from_record(rec: dict, preferred_key: str) -> str:
    keys = []
    pk = str(preferred_key or "").strip()
    if pk:
        keys.append(pk)
    keys.extend(["instruction", "edit_instruction", "edit_prompt", "prompt", "text", "caption", "task"])
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        v = rec.get(k)
        if isinstance(v, (list, tuple)):
            for item in v:
                if item is not None and str(item).strip():
                    return str(item).strip()
        elif v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _category_from_subdir_path(rel_input_path: str) -> str:
    parts = Path(rel_input_path).parts
    if not parts:
        return "unknown"
    anchors = ("input_img", "original_image", "edited_image", "output_img")
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            # Typical layout: <category>/input_img/xxx  or <category>/original_image/xxx
            if idx > 0:
                return parts[idx - 1]
            # Alternate layout: input_img/<category>/xxx
            if idx + 1 < len(parts):
                nxt = parts[idx + 1]
                if "." not in nxt:
                    return nxt
            break
    return parts[0]


def _category_from_record(rec: dict, rel_input_path: str) -> str:
    # For no-GT decoded datasets, N_PER_CAT is expected to cap samples per data subdirectory.
    # So we prioritize category inferred from relative path structure.
    cat_from_path = _category_from_subdir_path(rel_input_path)
    if cat_from_path and cat_from_path != "unknown":
        return cat_from_path
    td = rec.get("task_id_dir")
    if td is not None and str(td).strip():
        return str(td).strip()
    ty = rec.get("type") or rec.get("task_id")
    if ty is not None and str(ty).strip():
        return str(ty).strip()
    parts = Path(rel_input_path).parts
    if len(parts) >= 2 and parts[0] in ("original_image", "edited_image", "input_img", "output_img"):
        return parts[1]
    if parts:
        return parts[0]
    return "unknown"


def _iter_records(index_path: Path):
    if index_path.suffix.lower() == ".jsonl":
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    yield rec
        return
    obj = None
    json_ok = False
    try:
        with open(index_path, encoding="utf-8") as f:
            obj = json.load(f)
        json_ok = True
    except json.JSONDecodeError:
        json_ok = False
    if not json_ok:
        # Some datasets use ".json" filename but store JSONL records.
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    yield rec
        return
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("records"), list):
            rows = obj["records"]
        elif isinstance(obj.get("data"), list):
            rows = obj["data"]
        else:
            rows = [obj]
    else:
        rows = []
    for rec in rows:
        if isinstance(rec, dict):
            yield rec


def _build_samples_from_index(
    image_root: Path,
    index_path: Path,
    *,
    instruction_field: str,
    verify_paths: bool,
    categories: list[str] | None,
    n_per_cat_cap: int = 0,
) -> list[CrispEditSample]:
    cat_allow = None if categories is None else {str(c) for c in categories}
    out: list[CrispEditSample] = []
    per_cat_kept: dict[str, int] = defaultdict(int)
    cap = int(n_per_cat_cap)
    tracked_cats = list(cat_allow) if cat_allow is not None else []
    for rec in _iter_records(index_path):
        sp_in, sp_out = _json_record_edit_pair_paths(rec)
        if not sp_in or not sp_out:
            continue
        cat = _category_from_record(rec, sp_in)
        if cat_allow is not None and cat not in cat_allow:
            continue
        if cap > 0 and per_cat_kept.get(cat, 0) >= cap:
            if tracked_cats and all(per_cat_kept.get(c, 0) >= cap for c in tracked_cats):
                break
            continue
        input_path = (image_root / sp_in).resolve()
        output_path = (image_root / sp_out).resolve()
        if verify_paths and (not input_path.is_file() or not output_path.is_file()):
            continue
        stem = input_path.stem
        if stem.endswith("_input"):
            base = stem[: -len("_input")]
        elif stem.endswith("_orig"):
            base = stem[: -len("_orig")]
        else:
            base = stem
        key = str(rec.get("row_uid") or rec.get("id") or f"{cat}/{base}")
        out.append(
            CrispEditSample(
                category=cat,
                key=key,
                input_path=input_path,
                output_path=output_path,
                instruction=_instruction_from_record(rec, instruction_field),
            )
        )
        per_cat_kept[cat] = per_cat_kept.get(cat, 0) + 1
        if cap > 0 and tracked_cats and all(per_cat_kept.get(c, 0) >= cap for c in tracked_cats):
            break
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="FFD inference on decoded_data datasets without GT masks.")
    ap.add_argument(
        "--dataset",
        type=str,
        default="crispedit2m",
        help=(
            "Dataset preset: anyedit | crispedit2m | promptfixdata | omnieditfiltered12m | custom. "
            "custom = legacy records-root/<category>/records.jsonl flow."
        ),
    )
    ap.add_argument(
        "--decoded-root",
        type=str,
        default="",
        help="Root used to resolve ``input_img`` / ``output_img`` ``saved_path`` in jsonl (images on disk).",
    )
    ap.add_argument(
        "--index-path",
        type=str,
        default="",
        help="all_records index file path (.jsonl or .json). Used by preset datasets; optional for custom.",
    )
    ap.add_argument(
        "--instruction-field",
        type=str,
        default="",
        help="Instruction key in index records. Empty => preset default (or fallback keys for custom).",
    )
    ap.add_argument(
        "--records-root",
        type=str,
        default=str(_DEFAULT_RECORDS_ROOT),
        help=f"Root containing ``<category>/records.jsonl`` copies (default: {_DEFAULT_RECORDS_ROOT}).",
    )
    ap.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma-separated category folder names. Empty => all categories in index.",
    )
    ap.add_argument(
        "--records-name",
        type=str,
        default="records.jsonl",
        help="JSONL filename inside each category folder.",
    )
    ap.add_argument("-N", "--n", dest="n_per_cat", type=int, default=8, help="Random samples per category.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ckpt",
        type=str,
        default="weights/model.pt",
    )
    ap.add_argument("--config", type=str, default="configs/ffd_local_ablations.yaml")
    ap.add_argument("--ffd-preset", dest="ffd_preset", type=str, default="ffd_msdm_drb2_textbranch")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_ROOT))
    ap.add_argument("--infer-size", type=int, default=0, help="0 = use config dataset.image_size")
    ap.add_argument("--encoder-pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--online-text-branch", action="store_true")
    ap.add_argument("--parser-model-path", type=str, default="")
    ap.add_argument("--acsf-batch-eval-root", type=str, default="")
    ap.add_argument("--verify-paths", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hf-endpoint", type=str, default="")
    ap.add_argument("--mask-threshold", type=float, default=0.5, help="Also save hard mask PNG at this prob threshold.")
    ap.add_argument(
        "--save-vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save side-by-side comparison figures.",
    )
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    dataset_key = _normalize_dataset_name(args.dataset)
    if dataset_key == "crispedit":
        dataset_key = "crispedit2m"
    if dataset_key == "promptfix":
        dataset_key = "promptfixdata"
    if dataset_key in ("omniedit", "omnieditfiltered", "omnieditfiltered12", "omnieditfiltered12mwithinst"):
        dataset_key = "omnieditfiltered12m"
    if dataset_key not in set(_DATASET_DEFAULTS.keys()) | {"custom"}:
        raise ValueError(
            f"Unknown --dataset={args.dataset!r}; choose "
            "anyedit/crispedit2m/promptfixdata/omnieditfiltered12m/custom."
        )

    ds_defaults = _DATASET_DEFAULTS.get(dataset_key, {})
    decoded_root_s = str(args.decoded_root or ds_defaults.get("decoded_root") or "").strip()
    if not decoded_root_s:
        raise ValueError("--decoded-root is required for this dataset preset.")
    image_root = Path(decoded_root_s).expanduser().resolve()

    index_path_s = str(args.index_path or ds_defaults.get("index_path") or "").strip()
    instruction_field = str(args.instruction_field or ds_defaults.get("instruction_field") or "").strip()
    records_root = Path(args.records_root).expanduser().resolve()

    out_base = Path(args.out_dir).expanduser().resolve()
    out_root = ensure_dir(out_base if dataset_key == "custom" else (out_base / dataset_key))

    cats = _parse_categories(args.categories)
    if dataset_key == "custom" and not cats:
        cats = list(_DEFAULT_CATEGORIES)
    _log(f"dataset: {dataset_key}")
    _log(f"image path resolve root: {image_root}")
    if dataset_key == "custom":
        _log(f"records.jsonl root: {records_root}")
    else:
        _log(f"index file: {index_path_s}")
        _log(f"instruction field: {instruction_field or '(auto fallback)'}")

    cfg = load_yaml(args.config)
    dcfg = cfg.get("dataset", {})
    infer_size = int(args.infer_size) if int(args.infer_size) > 0 else int(dcfg.get("image_size", 518))
    device_s = (args.device or str(cfg.get("device", "cuda"))).strip()
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)

    mcfg_base = dict(cfg.get("model", {}))
    parser_cli = str(args.parser_model_path or "").strip() or str(os.environ.get("PARSER_MODEL_PATH", "") or "").strip()
    if parser_cli:
        mcfg_base["parser_model_path"] = parser_cli
    acsf_root = str(args.acsf_batch_eval_root or "").strip() or str(os.environ.get("ACSF_EVAL_ROOT", "") or "").strip()

    if str(mcfg_base.get("hf_hub_cache", "")).strip():
        resolve_hf_hub_cache_dir(str(mcfg_base.get("hf_hub_cache")).strip() or None)

    preset = str(args.ffd_preset or "").strip()
    if preset:
        mcfg_base["ffd_preset"] = preset
        mcfg_base["ablation_preset"] = ""
        mcfg_base["table1_preset"] = ""

    offline_store: object | None = None
    if acsf_root and OfflineTextInputStore is not None:
        try:
            offline_store = OfflineTextInputStore(Path(acsf_root).expanduser())
            _log(f"ACS offline store: {acsf_root}")
        except FileNotFoundError as e:
            _log(f"ACS offline disabled: {e}")

    use_online_text = bool(args.online_text_branch)
    can_use_instruction_llm = bool(str(mcfg_base.get("parser_model_path", "") or "").strip())
    if use_online_text and not can_use_instruction_llm and offline_store is None:
        _log(
            "warn: --online-text-branch without parser_model_path / PARSER_MODEL_PATH and no ACSF root → "
            "falling back to image-only forward where offline hits miss."
        )

    encoder_pre = bool(args.encoder_pretrained)
    model = build_ffd_net(mcfg_base, encoder_pretrained=encoder_pre).to(device)
    model.eval()
    _load_ffd_state(model, Path(args.ckpt).expanduser(), device_s)

    rng = np.random.default_rng(int(args.seed))
    set_seed(int(args.seed))

    summary_rows: list[dict] = []
    samples_by_cat: dict[str, list[CrispEditSample]] = defaultdict(list)

    if dataset_key == "custom":
        for cat in cats:
            jpath = records_root / cat / args.records_name
            if not jpath.is_file():
                _log(f"[skip] {cat}: missing {jpath}")
                continue
            try:
                all_samples = build_crispedit_samples_from_jsonl(
                    image_root,
                    jpath,
                    categories=[cat],
                    max_samples=0,
                    max_samples_per_category=0,
                    verify_paths_exist=bool(args.verify_paths),
                    progress_every=0,
                )
            except Exception as e:
                _log(f"[fail] {cat}: load jsonl: {e}")
                continue
            for s in all_samples:
                samples_by_cat[str(s.category)].append(s)
    else:
        if not index_path_s:
            raise ValueError("--index-path is required for preset datasets.")
        index_path = Path(index_path_s).expanduser().resolve()
        if not index_path.is_file():
            raise FileNotFoundError(f"Index not found: {index_path}")
        target_categories = cats or _discover_dataset_categories(image_root)
        if not cats and target_categories:
            _log(f"auto discovered categories from decoded root: {len(target_categories)}")
        try:
            all_samples = _build_samples_from_index(
                image_root,
                index_path,
                instruction_field=instruction_field,
                verify_paths=bool(args.verify_paths),
                categories=target_categories or None,
                n_per_cat_cap=int(args.n_per_cat),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset index {index_path}: {e}") from e
        for s in all_samples:
            samples_by_cat[str(s.category)].append(s)

    cat_iter = cats if cats else sorted(samples_by_cat.keys())
    for cat in cat_iter:
        all_samples = samples_by_cat.get(cat, [])
        if not all_samples:
            _log(f"[skip] {cat}: no samples after filter (verify_paths={args.verify_paths})")
            continue
        picked = _pick_random_samples(all_samples, int(args.n_per_cat), rng)
        _log(f"{cat}: total_indexed={len(all_samples)} picked={len(picked)}")

        mask_dir = ensure_dir(out_root / cat / "masks")
        hard_dir = ensure_dir(out_root / cat / "masks_binary")
        vis_dir = ensure_dir(out_root / cat / "vis") if bool(args.save_vis) else (out_root / cat / "vis")
        meta_dir = ensure_dir(out_root / cat / "meta")

        for i, s in enumerate(tqdm(picked, desc=f"{cat}", leave=False)):
            stem_save = _safe_save_stem(s, i)
            try:
                before_b, after_b, h, w = _samples_tensors_hw(s, infer_size, device)
                out = _ffd_forward_one(
                    model,
                    before_b,
                    after_b,
                    str(s.instruction or ""),
                    use_online_text=use_online_text,
                    offline_store=offline_store,
                    can_use_instruction_llm=can_use_instruction_llm,
                )
                prob_np = _pred_prob_hw(out["mask_logits"], h, w)
            except Exception as e:
                _log(f"[fail] {cat} {s.key}: {e}")
                continue

            prob_u8 = np.clip(prob_np * 255.0 + 0.5, 0, 255).astype(np.uint8)
            mask_png = mask_dir / f"{stem_save}_pred_prob.png"
            cv2.imwrite(str(mask_png), prob_u8)

            binary = (prob_np >= float(args.mask_threshold)).astype(np.uint8) * 255
            cv2.imwrite(str(hard_dir / f"{stem_save}_pred_bin.png"), binary)

            before_rgb = _read_rgb(s.input_path)
            after_rgb = _read_rgb(s.output_path)
            before_rgb = _resize_rgb_to_hw(before_rgb, w, h)
            after_rgb_u8 = after_rgb.astype(np.uint8) if after_rgb.dtype != np.uint8 else after_rgb

            vis_path = vis_dir / f"{stem_save}_compare.png"
            if bool(args.save_vis):
                _save_triplet_figure(
                    before_rgb.astype(np.uint8),
                    after_rgb_u8,
                    prob_np,
                    str(s.instruction or ""),
                    vis_path,
                    title=f"{cat} | {s.key}",
                )

            meta = {
                "category": cat,
                "key": s.key,
                "instruction": str(s.instruction or ""),
                "input_path": _to_repo_relative(s.input_path),
                "output_path": _to_repo_relative(s.output_path),
                "mask_prob_png": _to_repo_relative(mask_png),
                "vis_png": _to_repo_relative(vis_path) if bool(args.save_vis) else "",
                "dataset": dataset_key,
                "infer_size": infer_size,
                "ckpt": _to_repo_relative(Path(args.ckpt).expanduser()),
            }
            with open(meta_dir / f"{stem_save}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            summary_rows.append(meta)

    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "decoded_root": _to_repo_relative(image_root),
                "records_root": _to_repo_relative(records_root),
                "index_path": _to_repo_relative(index_path_s) if index_path_s else "",
                "dataset": dataset_key,
                "n_per_cat": int(args.n_per_cat),
                "seed": int(args.seed),
                "categories": cats,
                "records_name": args.records_name,
                "instruction_field": instruction_field,
                "save_vis": bool(args.save_vis),
                "samples": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _log(f"Wrote {len(summary_rows)} samples under {out_root} ; summary {summary_path}")


if __name__ == "__main__":
    main()
