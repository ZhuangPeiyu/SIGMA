# SIGMA

Minimal runnable SIGMA/FFD project with local `resource/` assets and relative-path configuration.

## 1) Environment Setup

### 1.1 Create and activate Conda environment

```bash
conda create -n sigma python=3.10 -y
conda activate sigma
```

### 1.2 Install dependencies

Install dependencies from the exported environment snapshot:

```bash
pip install -r requirements.txt
```

---

## 2) Required Models and Resource Paths (`resource/`)

Before running inference, download the required public model weights and place them under `resource/`.

Default local resources (relative to repository root):

- `resource/models--Qwen--Qwen2.5-0.5B-Instruct`: parser model for the online text branch (HF repo: `Qwen/Qwen2.5-0.5B-Instruct`)
- `resource/models--IDEA-Research--grounding-dino-base`: GroundingDINO (HF repo: `IDEA-Research/grounding-dino-base`)
- `resource/models--timm--vit_base_patch14_dinov2.lvd142m`: timm/DINOv2 backbone cache (HF repo: `timm/vit_base_patch14_dinov2.lvd142m`)
- `resource/sam2.1_hiera_small.pt`: SAM2.1 checkpoint (HF repo: `facebook/sam2.1-hiera-small`)

Optional environment variables (scripts already have defaults):

```bash
export PARSER_MODEL_PATH="resource/models--Qwen--Qwen2.5-0.5B-Instruct"
export HF_HOME="resource"
export HUGGINGFACE_HUB_CACHE="resource"
export HF_HUB_CACHE="resource"
```

---

## 3) Test Scripts

### 3.1 `scripts/run_anyedit_selected_test.py`

Purpose: run inference on samples under `examples/AnyEdit_selected` and save outputs to `outputs/anyedit_selected_test`.

- Automatically reads or generates `examples/AnyEdit_selected/manifest.json`
- Automatically builds custom records and decoded staging directories

Example:

```bash
python scripts/run_anyedit_selected_test.py
```

If your test samples do not contain instructions, disable the online text branch:

```bash
python scripts/run_anyedit_selected_test.py --no-online-text-branch
```

### 3.2 `scripts/test_without_gt.py`

Purpose: main no-GT inference script; supports preset datasets and custom records input.

Custom example:

```bash
PYTHONPATH=src python scripts/test_without_gt.py \
  --dataset custom \
  --decoded-root outputs/anyedit_selected_decoded \
  --records-root outputs/anyedit_selected_records \
  --categories anyedit_selected \
  --ckpt weights/model.pt \
  --config configs/ffd_local_ablations.yaml \
  --ffd-preset ffd_msdm_drb2_textbranch \
  --out-dir outputs/anyedit_selected_test
```

---

## 4) Model Weights

Default main checkpoint:

- `weights/model.pt`

To use a different checkpoint, override via CLI:

```bash
python scripts/run_anyedit_selected_test.py --ckpt weights/your_model.pt
```

