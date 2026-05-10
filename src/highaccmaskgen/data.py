import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import pixel_drift_augment


@dataclass
class Sample:
    method: str
    domain: str
    split: str
    key: str
    real_path: Path
    forged_path: Path
    mask_path: Path


@dataclass
class CrispEditSample:
    """CrispEdit-2M ``data_decoded``: ``input_img`` (before) / ``output_img`` (after)."""

    category: str
    key: str
    input_path: Path
    output_path: Path
    instruction: str = ""


def _split_suffix(split: str) -> str:
    split = split.lower()
    if split == "stuff":
        return "_stuff"
    if split == "background":
        return "_background"
    raise ValueError(f"Unknown split: {split}")


def _find_real_image(real_domain_dir: Path, key: str) -> Path | None:
    # BRGen COCO/Places are typically .jpg while ImageNet often uses .JPEG.
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"):
        p = real_domain_dir / f"{key}{ext}"
        if p.exists():
            return p
    return None


def build_samples(
    forged_root: Path,
    mask_root: Path,
    real_root: Path,
    methods: Iterable[str],
    domain: str,
    split: str,
) -> list[Sample]:
    suffix = _split_suffix(split)
    samples: list[Sample] = []
    for method in methods:
        forged_dir = forged_root / method / split.capitalize() / domain
        mask_dir = mask_root / split.capitalize() / domain
        if not forged_dir.exists() or not mask_dir.exists():
            continue
        for forged_path in forged_dir.glob(f"*{suffix}.png"):
            key = forged_path.stem.replace(suffix, "")
            real_path = _find_real_image(real_root / domain, key)
            mask_path = mask_dir / forged_path.name
            if real_path is not None and mask_path.exists():
                samples.append(
                    Sample(
                        method=method,
                        domain=domain,
                        split=split,
                        key=key,
                        real_path=real_path,
                        forged_path=forged_path,
                        mask_path=mask_path,
                    )
                )
    return samples


def split_train_val_shuffle(samples: list, val_ratio: float, seed: int) -> tuple[list, list]:
    """Shuffle index order with RNG ``seed``, then split into train / val by ``val_ratio``."""
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = max(1, int(len(samples) * float(val_ratio)))
    val_set = set(idx[:n_val].tolist())
    train = [samples[i] for i in range(len(samples)) if i not in val_set]
    val = [samples[i] for i in range(len(samples)) if i in val_set]
    return train, val


def _discover_crispedit_categories(root: Path) -> list[str]:
    """Category dirs under ``root`` with ``input_img`` + ``output_img``; **os.scandir order (no sort)**."""
    out: list[str] = []
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            p = Path(entry.path)
            if (p / "input_img").is_dir() and (p / "output_img").is_dir():
                out.append(entry.name)
    return out


def _jsonl_record_edit_pair_paths(rec: dict) -> tuple[str | None, str | None]:
    """
    Return ``(rel_before, rel_after)`` paths relative to dataset root, or ``(None, None)``.

    Supports:

    - CrispEdit decoded: ``input_img.saved_path`` / ``output_img.saved_path``
    - Promptfix / OmniEdit-style: ``original_image.saved_path`` / ``edited_image.saved_path``
    """
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


def _instruction_from_jsonl_record(rec: dict) -> str:
    """Best-effort edit instruction / prompt text from a jsonl record (may be empty)."""
    for k in (
        "instruction",
        "edit_instruction",
        "edit_prompt",
        "prompt",
        "text",
        "caption",
        "task",
    ):
        v = rec.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _category_from_jsonl_record(rec: dict, sp_in: str) -> str:
    """Prefer ``task_id_dir`` / ``type``; else infer from path (CrispEdit or Promptfix layouts)."""
    td = rec.get("task_id_dir")
    if td is not None and str(td).strip():
        return str(td).strip()
    ty = rec.get("type") or rec.get("task_id")
    if ty is not None and str(ty).strip():
        return str(ty).strip()
    p_in_rel = Path(sp_in)
    parts = p_in_rel.parts
    if len(parts) >= 2 and parts[0] in ("original_image", "edited_image", "input_img", "output_img"):
        return parts[1]
    if parts:
        return parts[0]
    return "unknown"


def build_crispedit_samples_from_jsonl(
    data_root: Path | str,
    jsonl_path: Path | str,
    categories: Iterable[str] | None = None,
    max_samples: int = 0,
    max_samples_per_category: int = 0,
    *,
    verify_paths_exist: bool = True,
    progress_every: int = 100_000,
) -> list[CrispEditSample]:
    """
    Build sample list from a CrispEdit / Promptfix ``all_records*.json(l)`` index (one JSON object per line).

    Pair paths (relative to ``data_root``) are taken from either:

    - ``input_img.saved_path`` / ``output_img.saved_path`` (CrispEdit ``data_decoded`` index), or
    - ``original_image.saved_path`` / ``edited_image.saved_path`` (Promptfix, OmniEdit-Filtered, etc.).

    Category defaults to ``task_id_dir`` when present, else first path component (e.g. ``add`` under
    ``add/input_img/...``).

    ``verify_paths_exist``: if True (default), skip rows whose input/output files are missing.
    Set False on slow/network storage to avoid millions of ``stat`` calls (trust the index).

    ``progress_every``: print a line to stdout every N **non-empty** jsonl lines (0 = off).
    """
    root = Path(data_root).resolve()
    jpath = Path(jsonl_path)
    if not jpath.is_file():
        raise FileNotFoundError(f"CrispEdit index jsonl not found: {jpath}")
    cat_allow = None if categories is None else {str(c) for c in categories}
    per_cat: dict[str, int] = {}
    out: list[CrispEditSample] = []
    n_nonempty = 0

    with open(jpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_nonempty += 1
            if progress_every and n_nonempty % progress_every == 0:
                print(
                    f"[crispedit jsonl] lines≈{n_nonempty} kept={len(out)} ({jpath.name})",
                    flush=True,
                )
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sp_in, sp_out = _jsonl_record_edit_pair_paths(rec)
            if not sp_in or not sp_out:
                continue
            cat = _category_from_jsonl_record(rec, sp_in)
            if cat_allow is not None and cat not in cat_allow:
                continue
            if max_samples_per_category > 0 and per_cat.get(cat, 0) >= max_samples_per_category:
                continue

            inp_path = (root / sp_in).resolve()
            outp_path = (root / sp_out).resolve()
            if verify_paths_exist and (not inp_path.is_file() or not outp_path.is_file()):
                continue

            stem = inp_path.stem
            if stem.endswith("_input"):
                base = stem[: -len("_input")]
            elif stem.endswith("_orig"):
                base = stem[: -len("_orig")]
            else:
                base = stem
            key = str(rec.get("row_uid") or rec.get("id") or f"{cat}/{base}")
            instr = _instruction_from_jsonl_record(rec)
            out.append(
                CrispEditSample(
                    category=cat,
                    key=key,
                    input_path=inp_path,
                    output_path=outp_path,
                    instruction=instr,
                )
            )
            per_cat[cat] = per_cat.get(cat, 0) + 1
            if max_samples > 0 and len(out) >= max_samples:
                break
    return out


def _iter_crispedit_input_files(inp_dir: Path, *, sort_names: bool) -> Iterator[Path]:
    """List input_img files; ``sort_names=False`` uses ``os.scandir`` (faster on huge folders)."""
    if sort_names:
        for p in sorted(inp_dir.iterdir()):
            if p.is_file():
                yield p
        return
    with os.scandir(inp_dir) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)


def build_crispedit_samples(
    data_root: Path | str,
    categories: Iterable[str] | None = None,
    max_samples: int = 0,
    max_samples_per_category: int = 0,
    *,
    sort_input_files: bool = False,
    category_order_seed: int | None = None,
    index_jsonl: Path | str | None = None,
    jsonl_skip_exist_check: bool = False,
    jsonl_progress_every: int = 100_000,
) -> list[CrispEditSample]:
    """
    Scan CrispEdit ``data_decoded`` layout: ``<root>/<category>/input_img/*_input.*`` paired with
    ``output_img/<same-id>_output.<ext>`` (id = stem without ``_input``).

    If ``index_jsonl`` is set, **only** read that jsonl (no directory traversal); paths inside
    must be relative to ``data_root`` as in ``all_records.jsonl``.

    ``max_samples``: global cap on total pairs (scan stops as soon as this many are collected —
    with sorted categories this often fills only the first category). Prefer
    ``max_samples_per_category`` when you need a small run across **every** category.

    ``max_samples_per_category``: if > 0, take at most this many pairs **per** category, then
    continue to the next category.

    ``sort_input_files``: if ``False`` (default), iterate each ``input_img`` with ``os.scandir``
    without sorting (faster for large and full scans). If ``True``, use sorted ``iterdir`` for a
    stable, reproducible order.

    ``category_order_seed``: if set, shuffle the category list with this RNG before scanning
    (recommended with ``max_samples`` so caps are not biased toward alphabetical order).
    Auto-discovered categories are **not** sorted; order follows ``os.scandir`` on the root.
    Ignored when ``index_jsonl`` is set (order follows the jsonl file).

    ``jsonl_skip_exist_check``: if True, do not ``stat`` every file when reading ``index_jsonl``
    (much faster on NFS; default False).

    ``jsonl_progress_every``: print progress every N non-empty jsonl lines (0 = off).
    """
    if index_jsonl:
        return build_crispedit_samples_from_jsonl(
            data_root,
            index_jsonl,
            categories=categories,
            max_samples=max_samples,
            max_samples_per_category=max_samples_per_category,
            verify_paths_exist=not bool(jsonl_skip_exist_check),
            progress_every=int(jsonl_progress_every),
        )

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CrispEdit data root not found: {root}")

    if categories is None:
        cat_list = _discover_crispedit_categories(root)
    else:
        cat_list = [str(c) for c in categories]
    if category_order_seed is not None:
        rng = np.random.default_rng(int(category_order_seed))
        rng.shuffle(cat_list)
    out: list[CrispEditSample] = []
    for cat in cat_list:
        inp_dir = root / cat / "input_img"
        out_dir = root / cat / "output_img"
        if not inp_dir.is_dir() or not out_dir.is_dir():
            continue
        n_in_cat = 0
        for inp_path in _iter_crispedit_input_files(inp_dir, sort_names=sort_input_files):
            stem = inp_path.stem
            if not stem.endswith("_input"):
                continue
            base = stem[: -len("_input")]
            key = f"{cat}/{base}"
            outp = out_dir / f"{base}_output{inp_path.suffix}"
            if not outp.is_file():
                alt = list(out_dir.glob(f"{base}_output.*"))
                if not alt:
                    continue
                outp = alt[0]
            out.append(
                CrispEditSample(
                    category=cat,
                    key=key,
                    input_path=inp_path,
                    output_path=outp,
                )
            )
            n_in_cat += 1
            if max_samples_per_category > 0 and n_in_cat >= max_samples_per_category:
                break
            if max_samples > 0 and len(out) >= max_samples:
                return out
    return out


def build_pico_banana_samples(
    data_root: Path | str,
    max_samples: int,
    seed: int,
    *,
    mapping_path: Path | str | None = None,
) -> list[CrispEditSample]:
    """
    Pico-banana layout under ``data_root``: ``edited_source_mapping.json`` with ``records[]``;
    each record has ``source_image`` and ``edited_image`` paths relative to ``data_root``.

    ``mapping_path``: optional path to a subset JSON (same schema as ``edited_source_mapping.json``).
    If omitted, uses ``<data_root>/edited_source_mapping.json``.
    """
    root = Path(data_root).resolve()
    map_path = Path(mapping_path) if mapping_path else root / "edited_source_mapping.json"
    if not map_path.is_file():
        raise FileNotFoundError(f"pico-banana mapping not found: {map_path}")
    with open(map_path, encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records") or []
    if not records:
        return []
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(records))
    rng.shuffle(order)
    out: list[CrispEditSample] = []
    cap = int(max_samples)
    for j in order:
        if cap > 0 and len(out) >= cap:
            break
        rec = records[int(j)]
        src = root / str(rec["source_image"])
        edt = root / str(rec["edited_image"])
        if not src.is_file() or not edt.is_file():
            continue
        group = str(rec.get("group") or "unknown")
        stem_ed = Path(str(rec["edited_image"])).stem
        key = f"pico-banana/{group}/{stem_ed}"
        instr = ""
        for k in ("instruction", "edit_instruction", "edit_prompt", "prompt", "caption"):
            v = rec.get(k)
            if v is not None and str(v).strip():
                instr = str(v).strip()
                break
        out.append(
            CrispEditSample(
                category=f"pico-banana/{group}",
                key=key,
                input_path=src,
                output_path=edt,
                instruction=instr,
            )
        )
    return out


def build_text2img_mix_train_samples(
    *,
    sources: list[dict],
    seed: int,
    jsonl_skip_exist_check: bool = True,
    jsonl_progress_every: int = 100_000,
    log: Optional[Callable[[str], None]] = None,
) -> list[CrispEditSample]:
    """
    Mix several Text2Img-style edit pair sources into one ``CrispEditSample`` list (shuffled).

    Each source dict:

    - ``name`` / ``id``: label for logs (optional).
    - ``kind``: ``pico_banana`` | ``crispedit_decoded``.
    - ``root``: dataset root path.
    - ``max_samples``: cap for this source (> 0).
    - For ``crispedit_decoded``: optional ``index_jsonl``, ``categories``, ``max_samples_per_category``.
    """
    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)

    combined: list[CrispEditSample] = []
    for i, src in enumerate(sources):
        name = str(src.get("name") or src.get("id") or f"source_{i}")
        kind = str(src.get("kind") or "").strip().lower()
        root = Path(str(src["root"]))
        per_max = int(src.get("max_samples", 0))
        if per_max <= 0:
            raise ValueError(f"text2img_mix source {name!r}: max_samples must be > 0, got {per_max}")
        sub_seed = int(seed) + i * 9973

        if kind == "pico_banana":
            pico_map = (str(src.get("index_jsonl") or "").strip()) or None
            part = build_pico_banana_samples(
                root,
                per_max,
                sub_seed,
                mapping_path=Path(pico_map) if pico_map else None,
            )
            _emit(
                f"text2img_mix[{name}] kind=pico_banana root={root} pairs={len(part)} "
                f"(cap={per_max}, mapping={pico_map or 'edited_source_mapping.json'})"
            )
        elif kind in ("crispedit_decoded", "crispedit", "decoded"):
            index_jsonl = (str(src.get("index_jsonl") or "").strip()) or None
            cats = src.get("categories")
            if isinstance(cats, str):
                cats = [c.strip() for c in cats.split(",") if c.strip()]
            elif cats == []:
                cats = None
            mspc = int(src.get("max_samples_per_category", 0))
            part = build_crispedit_samples(
                root,
                categories=cats,
                max_samples=per_max,
                max_samples_per_category=mspc,
                category_order_seed=sub_seed,
                index_jsonl=index_jsonl,
                jsonl_skip_exist_check=jsonl_skip_exist_check,
                jsonl_progress_every=int(jsonl_progress_every),
            )
            _emit(
                f"text2img_mix[{name}] kind=crispedit_decoded root={root} pairs={len(part)} "
                f"(cap={per_max}, index={index_jsonl or 'dir_scan'})"
            )
        else:
            raise ValueError(f"Unknown text2img_mix kind {kind!r} for source {name!r}")

        combined.extend(part)

    rng = np.random.default_rng(int(seed))
    rng.shuffle(combined)
    _emit(f"text2img_mix total_pairs={len(combined)} (shuffled, seed={int(seed)})")
    return combined


def crispedit_categories_fingerprint(
    categories: Iterable[str] | None,
    *,
    index_jsonl: Path | str | None = None,
) -> str:
    """Stable short id for manifest filenames (``None`` = all categories under root)."""
    parts: list[str] = []
    if index_jsonl:
        parts.append(f"jsonl={Path(index_jsonl).resolve()}")
    if categories is None:
        parts.append("__ALL__")
    else:
        parts.append(",".join(sorted(str(c) for c in categories)))
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def crispedit_train_val_manifest_paths(
    data_root: Path | str,
    seed: int,
    val_ratio: float,
    categories: Iterable[str] | None,
    cache_dir: Path | str | None = None,
    *,
    max_samples: int = 0,
    max_samples_per_category: int = 0,
    subset: int = 0,
    index_jsonl: Path | str | None = None,
) -> tuple[Path, Path]:
    """
    Paths for cached train/val pair lists (tab-separated paths relative to ``data_root``).

    Default ``cache_dir`` is ``<data_root>/.highaccmaskgen_split_cache``.
    Tag includes ``max_samples`` / ``subset`` so different caps do not share the wrong split.
    """
    root = Path(data_root)
    fp = crispedit_categories_fingerprint(categories, index_jsonl=index_jsonl)
    d = Path(cache_dir) if cache_dir is not None else (root / ".highaccmaskgen_split_cache")
    tag = (
        f"seed{int(seed)}_vr{val_ratio:g}_ms{int(max_samples)}_mspc{int(max_samples_per_category)}"
        f"_sub{int(subset)}_{fp}"
    )
    return d / f"crispedit_{tag}_train.txt", d / f"crispedit_{tag}_val.txt"


def save_crispedit_train_val_manifests(
    data_root: Path | str,
    train: list[CrispEditSample],
    val: list[CrispEditSample],
    *,
    seed: int,
    val_ratio: float,
    categories: Iterable[str] | None,
    cache_dir: Path | str | None = None,
    max_samples: int = 0,
    max_samples_per_category: int = 0,
    subset: int = 0,
    index_jsonl: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write train/val manifests (relative paths, portable if ``data_root`` moves as a unit)."""
    root = Path(data_root).resolve()
    train_p, val_p = crispedit_train_val_manifest_paths(
        root,
        seed,
        val_ratio,
        categories,
        cache_dir,
        max_samples=max_samples,
        max_samples_per_category=max_samples_per_category,
        subset=subset,
        index_jsonl=index_jsonl,
    )
    train_p.parent.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, rows: list[CrispEditSample]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# crispedit_split_manifest v1\n")
            f.write(f"# data_root={root}\n")
            f.write(
                f"# seed={int(seed)} val_ratio={val_ratio} max_samples={int(max_samples)} "
                f"max_samples_per_category={int(max_samples_per_category)} subset={int(subset)} "
                f"index_jsonl={index_jsonl or ''} n={len(rows)}\n"
            )
            for s in rows:
                rel_in = s.input_path.resolve().relative_to(root)
                rel_out = s.output_path.resolve().relative_to(root)
                f.write(f"{rel_in.as_posix()}\t{rel_out.as_posix()}\n")

    _write(train_p, train)
    _write(val_p, val)
    return train_p, val_p


def load_crispedit_train_val_manifests(
    data_root: Path | str,
    train_path: Path | str,
    val_path: Path | str,
) -> tuple[list[CrispEditSample], list[CrispEditSample]]:
    """Load train/val lists written by :func:`save_crispedit_train_val_manifests`."""
    root = Path(data_root).resolve()

    def _read(path: Path) -> list[CrispEditSample]:
        out: list[CrispEditSample] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split("\t", 1)
                input_path = (root / a).resolve()
                output_path = (root / b).resolve()
                cat = Path(a).parts[0]
                stem = input_path.stem
                if not stem.endswith("_input"):
                    raise ValueError(f"Expected *_input stem, got {input_path}")
                base = stem[: -len("_input")]
                key = f"{cat}/{base}"
                out.append(
                    CrispEditSample(
                        category=str(cat),
                        key=key,
                        input_path=input_path,
                        output_path=output_path,
                    )
                )
        return out

    return _read(Path(train_path)), _read(Path(val_path))


def _load_rgb_path(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size > 0:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def _load_rgb(path: Path, size: int) -> np.ndarray:
    return _load_rgb_path(path, size)


def _load_mask(path: Path, size: int, *, gt_threshold: int = 127) -> np.ndarray:
    """Binary GT mask. ``gt_threshold <= 0`` → nonzero foreground (DEAL300K-style)."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if size > 0:
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    if int(gt_threshold) <= 0:
        return (m > 0).astype(np.float32)
    return (m > int(gt_threshold)).astype(np.float32)


class BRGenPairDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int = 256,
        enable_augment: bool = False,
        augment_cfg: dict | None = None,
        mask_gt_threshold: int = 127,
    ):
        self.samples = samples
        self.image_size = image_size
        self.enable_augment = enable_augment
        self.augment_cfg = augment_cfg or {}
        self.mask_gt_threshold = int(mask_gt_threshold)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        real = _load_rgb(s.real_path, self.image_size).astype(np.float32) / 255.0
        forged = _load_rgb(s.forged_path, self.image_size).astype(np.float32) / 255.0
        mask = _load_mask(s.mask_path, self.image_size, gt_threshold=self.mask_gt_threshold)
        if self.enable_augment:
            real_u8 = (np.clip(real, 0.0, 1.0) * 255.0).astype(np.uint8)
            forged_u8 = (np.clip(forged, 0.0, 1.0) * 255.0).astype(np.uint8)
            real_u8, forged_u8 = pixel_drift_augment(real_u8, forged_u8, self.augment_cfg)
            real = real_u8.astype(np.float32) / 255.0
            forged = forged_u8.astype(np.float32) / 255.0
        real_t = torch.from_numpy(real).permute(2, 0, 1).float()
        forged_t = torch.from_numpy(forged).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return {
            "real": real_t,
            "forged": forged_t,
            "mask": mask_t,
            "meta": {
                "method": s.method,
                "domain": s.domain,
                "key": s.key,
                "real_path": str(s.real_path),
                "forged_path": str(s.forged_path),
                "instruction": getattr(s, "instruction", "") or "",
                "data_source": getattr(s, "data_source", "deal300k") or "deal300k",
            },
        }


_EDITED_EXTS_DEAL300K = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _find_original_deal300k(original_dir: Path, sample_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = original_dir / f"{sample_id}{ext}"
        if p.is_file():
            return p
    return None


def iter_deal300k_test_pairs(test_root: Path | str) -> list[tuple[Path, Path, Path, str]]:
    """
    DEAL300K layout: ``test_root/{edited,mask,original}/``.

    Returns list of ``(edited_path, mask_path, original_path, stem)``.
    """
    test_root = Path(test_root)
    edited_dir = test_root / "edited"
    mask_dir = test_root / "mask"
    original_dir = test_root / "original"
    out: list[tuple[Path, Path, Path, str]] = []
    paths = sorted(
        p for p in edited_dir.iterdir() if p.is_file() and p.suffix.lower() in _EDITED_EXTS_DEAL300K
    )
    for edited_path in paths:
        stem = edited_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.is_file():
            mask_path = mask_dir / f"{stem}.jpg"
        if not mask_path.is_file():
            continue
        parts = stem.split("_", 1)
        sample_id = parts[0] if parts else stem
        op = _find_original_deal300k(original_dir, sample_id)
        if op is None:
            continue
        out.append((edited_path, mask_path, op, stem))
    return out


def build_deal300k_val_samples(test_root: Path | str, n: int, seed: int) -> list[Sample]:
    """Random ``n`` (or fewer if not enough pairs) DEAL300K test triplets for supervised val."""
    pairs = iter_deal300k_test_pairs(test_root)
    if not pairs:
        raise RuntimeError(f"No DEAL300K pairs under {test_root}")
    rng = np.random.default_rng(int(seed))
    take = min(int(n), len(pairs))
    idx = rng.choice(len(pairs), size=take, replace=False)
    out: list[Sample] = []
    for i in idx:
        edited_path, mask_path, orig_path, stem = pairs[i]
        out.append(
            Sample(
                method="deal300k",
                domain="test",
                split="test",
                key=str(stem),
                real_path=orig_path,
                forged_path=edited_path,
                mask_path=mask_path,
            )
        )
    return out


def build_cocoglide_samples(root: Path | str) -> list[Sample]:
    """
    Load CocoGlide pairs from ``<root>/table.csv``.

    CSV columns: ``real``, ``fake``, ``mask``, ``prompt`` (paths relative to ``root``).
    Only rows where all three image files exist are included.
    """
    root = Path(root)
    csv_path = root / "table.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CocoGlide table.csv not found: {csv_path}")
    samples: list[Sample] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            real_path = root / row["real"]
            forged_path = root / row["fake"]
            mask_path = root / row["mask"]
            if not real_path.is_file() or not forged_path.is_file() or not mask_path.is_file():
                continue
            prompt = row.get("prompt", "")
            samples.append(
                Sample(
                    method="cocoglide",
                    domain=prompt,
                    split="all",
                    key=Path(row["real"]).stem,
                    real_path=real_path,
                    forged_path=forged_path,
                    mask_path=mask_path,
                )
            )
    return samples


def filter_crispedit_samples_existing(
    samples: list[CrispEditSample],
) -> tuple[list[CrispEditSample], int]:
    """Keep only pairs whose ``input_path`` and ``output_path`` exist on disk. Returns (kept, n_dropped)."""
    kept: list[CrispEditSample] = []
    for s in samples:
        if s.input_path.is_file() and s.output_path.is_file():
            kept.append(s)
    return kept, len(samples) - len(kept)


class CrispEditPairDataset(Dataset):
    """
    Edit pairs without ground-truth masks: ``real`` = input (before), ``forged`` = output (after).
    ``mask`` is all zeros with same spatial size; ``meta.has_real_mask`` is False.
    Use teacher pseudo-label training or inference-only pipelines.
    """

    def __init__(
        self,
        samples: list[CrispEditSample],
        image_size: int = 256,
        enable_augment: bool = False,
        augment_cfg: dict | None = None,
    ):
        self.samples = samples
        self.image_size = image_size
        self.enable_augment = enable_augment
        self.augment_cfg = augment_cfg or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        real = _load_rgb_path(s.input_path, self.image_size).astype(np.float32) / 255.0
        forged = _load_rgb_path(s.output_path, self.image_size).astype(np.float32) / 255.0
        if self.enable_augment:
            real_u8 = (np.clip(real, 0.0, 1.0) * 255.0).astype(np.uint8)
            forged_u8 = (np.clip(forged, 0.0, 1.0) * 255.0).astype(np.uint8)
            real_u8, forged_u8 = pixel_drift_augment(real_u8, forged_u8, self.augment_cfg)
            real = real_u8.astype(np.float32) / 255.0
            forged = forged_u8.astype(np.float32) / 255.0
        h, w = real.shape[0], real.shape[1]
        mask = np.zeros((h, w), dtype=np.float32)
        real_t = torch.from_numpy(real).permute(2, 0, 1).float()
        forged_t = torch.from_numpy(forged).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return {
            "real": real_t,
            "forged": forged_t,
            "mask": mask_t,
            "meta": {
                "category": s.category,
                "key": s.key,
                "has_real_mask": False,
                "instruction": getattr(s, "instruction", "") or "",
            },
        }
