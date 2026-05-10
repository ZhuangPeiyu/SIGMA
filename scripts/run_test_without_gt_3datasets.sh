#!/usr/bin/env bash
# One-click sequential inference on multiple no-GT datasets.
# Default output root: /mnt/ugnas/upload/Text2Img/SIGMA_preds

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1) activate conda env
source "${CONDA_HOME:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-InContextMLLM}"

# 2) runtime env
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PARSER_MODEL_PATH="${PARSER_MODEL_PATH:-${ROOT}/resource/models--Qwen--Qwen2.5-0.5B-Instruct}"
export HF_HOME="${HF_HOME:-${ROOT}/resource}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${ROOT}/resource}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${ROOT}/resource}"

# 3) defaults (can be overridden by env vars)
CKPT="${CKPT:-${ROOT}/weights/model.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/SIGMA_preds}"
N_PER_CAT="${N_PER_CAT:-10}"  # each subdirectory/category under a dataset root
SEED="${SEED:-42}"
FFD_PRESET="${FFD_PRESET:-ffd_msdm_drb2_textbranch}"
ONLINE_TEXT_BRANCH="${ONLINE_TEXT_BRANCH:-1}"
SAVE_VIS="${SAVE_VIS:-1}"  # 1: --save-vis ; 0: --no-save-vis
VERIFY_PATHS="${VERIFY_PATHS:-0}"  # 1: verify pair paths exist; 0: skip path stat for faster index scan

# declare -a DATASETS=("anyedit" "promptfixdata" "omnieditfiltered12m")
# declare -a DATASETS=("promptfixdata")
# declare -a DATASETS=("anyedit")
# declare -a DATASETS=("crispedit2m")
declare -a DATASETS=("omnieditfiltered12m")

echo "[INFO] Conda env       : ${CONDA_ENV_NAME:-InContextMLLM}"
echo "[INFO] Checkpoint      : ${CKPT}"
echo "[INFO] Output root     : ${OUT_DIR}"
echo "[INFO] N per category  : ${N_PER_CAT}"
echo "[INFO] Seed            : ${SEED}"
echo "[INFO] FFD preset      : ${FFD_PRESET}"
echo "[INFO] Online text     : ${ONLINE_TEXT_BRANCH}"
echo "[INFO] Save vis        : ${SAVE_VIS}"
echo "[INFO] Verify paths    : ${VERIFY_PATHS}"
echo

for ds in "${DATASETS[@]}"; do
  echo "========== Running dataset: ${ds} =========="
  CMD=(
    python scripts/test_without_gt.py
    --dataset "${ds}"
    --ckpt "${CKPT}"
    -N "${N_PER_CAT}"
    --seed "${SEED}"
    --ffd-preset "${FFD_PRESET}"
    --parser-model-path "${PARSER_MODEL_PATH}"
    --out-dir "${OUT_DIR}"
  )

  if [[ "${ONLINE_TEXT_BRANCH}" == "1" ]]; then
    CMD+=(--online-text-branch)
  fi
  if [[ "${SAVE_VIS}" == "1" ]]; then
    CMD+=(--save-vis)
  else
    CMD+=(--no-save-vis)
  fi
  if [[ "${VERIFY_PATHS}" == "1" ]]; then
    CMD+=(--verify-paths)
  else
    CMD+=(--no-verify-paths)
  fi

  "${CMD[@]}"
  echo "========== Done dataset: ${ds} =========="
  echo
done

echo "[DONE] All datasets finished. Results saved under: ${OUT_DIR}"
