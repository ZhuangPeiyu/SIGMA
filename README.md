# SIGMA

SIGMA/FFD 最小可运行工程（已适配本地 `resource/` 资源目录与相对路径）。

## 1) 环境安装

### 1.1 创建并激活 Conda 环境

```bash
source "${CONDA_HOME:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda create -n InContextMLLM python=3.10 -y
conda activate InContextMLLM
```

### 1.2 安装依赖

当前仓库根目录的 `requirements.txt` 由 `InContextMLLM` 环境导出，可直接安装：

```bash
pip install -r requirements.txt
```

如需重新导出当前环境依赖：

```bash
pip freeze > requirements.txt
```

---

## 2) 依赖模型与资源路径（`resource/`）

项目默认依赖以下本地资源（相对仓库根目录）：

- `resource/models--Qwen--Qwen2.5-0.5B-Instruct`：在线文本分支 parser 模型
- `resource/models--IDEA-Research--grounding-dino-base`：GroundingDINO
- `resource/models--timm--vit_base_patch14_dinov2.lvd142m`：timm/DINOv2 backbone 缓存
- `resource/sam2.1_hiera_small.pt`：SAM2.1 checkpoint

常用环境变量（可选，脚本已有默认值）：

```bash
export PARSER_MODEL_PATH="resource/models--Qwen--Qwen2.5-0.5B-Instruct"
export HF_HOME="resource"
export HUGGINGFACE_HUB_CACHE="resource"
export HF_HUB_CACHE="resource"
```

---

## 3) 测试脚本介绍

### 3.1 `scripts/run_anyedit_selected_test.py`

用途：对 `examples/AnyEdit_selected` 下样本执行推理测试，并输出到 `outputs/anyedit_selected_test`。

- 自动读取/生成 `examples/AnyEdit_selected/manifest.json`
- 自动构建 custom records 与 decoded staging

示例：

```bash
python scripts/run_anyedit_selected_test.py
```

关闭在线文本分支：

```bash
python scripts/run_anyedit_selected_test.py --no-online-text-branch
```

### 3.2 `scripts/test_without_gt.py`

用途：无 GT 推理主脚本，支持 preset 数据集与 custom records 输入。

示例（custom）：

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

### 3.3 `scripts/run_test_without_gt_3datasets.sh`

用途：多数据集顺序推理（默认读取脚本中的数据集列表）。

示例：

```bash
bash scripts/run_test_without_gt_3datasets.sh
```

---

## 4) 其他常用脚本

- `scripts/eval_finet2i_mask_methods.py`：有 GT 的方法对比评估（IoU/F1）
- `scripts/prepare_examples_and_eval.py`：样本筛选 + 评估流程脚本
- `scripts/train_FFD.py`：训练脚本

---

## 5) 权重文件约定

默认主权重文件名：

- `weights/model.pt`

若使用其他权重，传参覆盖即可：

```bash
python scripts/run_anyedit_selected_test.py --ckpt weights/your_model.pt
```

