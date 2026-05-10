#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")
    out = []
    for row in obj:
        if isinstance(row, dict) and row.get("status") == "ok":
            out.append(row)
    return out


def _build_manifest_from_examples(examples_root: Path) -> list[dict]:
    rows: list[dict] = []
    for d in sorted(examples_root.iterdir()):
        if not d.is_dir():
            continue
        p_orig = d / "original.jpg"
        p_edit = d / "edited.jpg"
        if not p_orig.is_file() or not p_edit.is_file():
            continue
        inst_path = d / "instruction.txt"
        instruction = ""
        if inst_path.is_file():
            instruction = inst_path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "requested_name": d.name,
                "row_uid": d.name,
                "instruction": instruction,
                "original_image": str(Path("examples/AnyEdit_selected") / d.name / "original.jpg"),
                "edited_image": str(Path("examples/AnyEdit_selected") / d.name / "edited.jpg"),
                "status": "ok",
            }
        )
    return rows


def _ensure_records(
    examples_root: Path,
    decoded_root: Path,
    records_root: Path,
    category: str,
    rows: list[dict],
) -> Path:
    # Build a CrispEdit-like layout so custom category filtering works:
    # <decoded_root>/<category>/{input_img,output_img}/*.jpg
    in_dir = decoded_root / category / "input_img"
    out_dir = decoded_root / category / "output_img"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_dir = records_root / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    records_path = cat_dir / "records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            req_name = str(r["requested_name"])
            p_orig = examples_root / req_name / "original.jpg"
            p_edit = examples_root / req_name / "edited.jpg"
            if not p_orig.is_file() or not p_edit.is_file():
                continue
            in_name = f"{req_name}_{i:03d}_input.jpg"
            out_name = f"{req_name}_{i:03d}_output.jpg"
            dst_in = in_dir / in_name
            dst_out = out_dir / out_name
            if not dst_in.exists():
                os.symlink(p_orig, dst_in)
            if not dst_out.exists():
                os.symlink(p_edit, dst_out)
            rec = {
                "row_uid": str(r.get("row_uid") or req_name),
                "instruction": str(r.get("instruction") or ""),
                "input_img": {"saved_path": f"{category}/input_img/{in_name}"},
                "output_img": {"saved_path": f"{category}/output_img/{out_name}"},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Run test_without_gt.py on examples/AnyEdit_selected via custom jsonl records."
    )
    ap.add_argument(
        "--examples-root",
        type=str,
        default="examples/AnyEdit_selected",
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Manifest JSON path; default <examples-root>/manifest.json",
    )
    ap.add_argument(
        "--records-root",
        type=str,
        default="outputs/anyedit_selected_records",
    )
    ap.add_argument(
        "--decoded-root",
        type=str,
        default="outputs/anyedit_selected_decoded",
        help="Generated decoded-style root used by test_without_gt custom mode.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="outputs/anyedit_selected_test",
    )
    ap.add_argument(
        "--category",
        type=str,
        default="anyedit_selected",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="configs/ffd_local_ablations.yaml",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="weights/model.pt",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ffd-preset", type=str, default="ffd_msdm_drb2_textbranch")
    ap.add_argument(
        "--parser-model-path",
        type=str,
        default="resource/models--Qwen--Qwen2.5-0.5B-Instruct",
        help="Parser model path for online text branch.",
    )
    ap.add_argument(
        "--online-text-branch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--save-vis",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--verify-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = ap.parse_args()

    examples_root = (repo_root / Path(args.examples_root).expanduser()).resolve()
    if not examples_root.is_dir():
        raise FileNotFoundError(f"Examples root not found: {examples_root}")
    if str(args.manifest).strip():
        manifest_path = (repo_root / Path(args.manifest).expanduser()).resolve()
    else:
        manifest_path = examples_root / "manifest.json"
    if manifest_path.is_file():
        rows = _load_manifest(manifest_path)
    else:
        rows = _build_manifest_from_examples(examples_root)
        if rows:
            manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[info] Generated manifest: {manifest_path}", flush=True)
        else:
            raise FileNotFoundError(
                f"Manifest not found and no valid samples discovered under: {examples_root}"
            )
    if not rows:
        raise RuntimeError(f"No usable samples in manifest: {manifest_path}")
    n_per_cat = len(rows)

    records_root = Path(args.records_root)
    decoded_root = Path(args.decoded_root)
    _ensure_records(examples_root, decoded_root, records_root, args.category, rows)

    cmd = [
        "python",
        "scripts/test_without_gt.py",
        "--dataset",
        "custom",
        "--decoded-root",
        str(decoded_root),
        "--records-root",
        str(records_root),
        "--categories",
        args.category,
        "--config",
        str(Path(args.config)),
        "--ckpt",
        str(Path(args.ckpt)),
        "--out-dir",
        str(Path(args.out_dir)),
        "-N",
        str(n_per_cat),
        "--seed",
        str(int(args.seed)),
        "--ffd-preset",
        str(args.ffd_preset),
    ]
    if args.online_text_branch:
        cmd.append("--online-text-branch")
        parser_model_path = str(args.parser_model_path).strip()
        if parser_model_path:
            cmd.extend(["--parser-model-path", parser_model_path])
    cmd.append("--save-vis" if args.save_vis else "--no-save-vis")
    cmd.append("--verify-paths" if args.verify_paths else "--no-verify-paths")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}"

    print("Running command:")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))


if __name__ == "__main__":
    main()
