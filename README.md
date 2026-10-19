# SIGMA

**SIGMA: Semantic-Difference Instruction-Grounding Mask Annotator for Text-Driven Image Manipulation Localization**

🎉🏆✨ This paper has been accepted as a **Spotlight** at **NeurIPS 2026**.

<p align="center">
  <a href="https://arxiv.org/abs/2605.27924"><img src="https://img.shields.io/badge/arXiv-2605.27924-b31b1b" alt="arXiv"></a>
  <a href="https://zhuangpeiyu.github.io/projects/SIGMA/"><img src="https://img.shields.io/badge/Project-Page-2ea44f" alt="Project Page"></a>
  <a href="https://huggingface.co/PatrickZZZ/SIGMA"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/datasets/PatrickZZZ/SIGMA"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face Dataset"></a>
  <a href="https://www.modelscope.ai/datasets/PeiyuZhuang/SIGMA_small"><img src="https://img.shields.io/badge/ModelScope-10K%20subset-624aff" alt="ModelScope 10K subset"></a>
  <a href="https://www.modelscope.ai/datasets/PeiyuZhuang/SIGMA_partial"><img src="https://img.shields.io/badge/ModelScope-100K%20subset-624aff" alt="ModelScope 100K subset"></a>
</p>

## Abstract

Text-driven image editing has advanced rapidly, but reliably localizing these manipulations requires image manipulation localization (IML) models trained on large pixel-annotated datasets, and there is still no low-cost way to obtain such training data at scale. We observe that these data already exist in disguise: public editing datasets contain millions of structurally identical *(original, edited)* pairs to IML training samples, lacking only pixel-level masks. Recovering these masks automatically is non-trivial: pixel differencing is overwhelmed by diffusion-induced perturbations across all pixels, and instruction-only grounding localizes only what the prompt describes, missing unintended editor side-effects.

We propose **SIGMA** (**S**emantic-difference **I**nstruction-**G**rounding **M**ask **A**nnotator), which performs semantic-feature differencing in a vision foundation backbone and injects an instruction-derived spatial prior into this visual stream via bidirectional cross-modal refinement, amplifying the difference signal at intended-edit regions when the editor faithfully realizes user intent. SIGMA is trained in two complementary stages: Stage I supervises on inpainting masks; Stage II closes the diffusion-domain shift via VAE-roundtrip noise calibration, EMA self-training, and an edit-noise disentanglement loss. SIGMA outperforms existing automatic mask generators on five benchmarks (**+12.20%** F1, **+11.16%** IoU). When applied to public editing corpora, it produces a ~1.1M IML training set that improves four diverse detectors by **+20.98%** F1 across five datasets, turning previously unused editing data into a model-agnostic supervisory resource for IML.

![Overview of the SIGMA architecture.](assets/sigma_framework.png)

*Overview of the SIGMA architecture.*

## Results

Quantitative comparison across different datasets. F1 and IoU scores (%) are reported. The best score in each column is in **bold**.

| Method | CoCoGlide F1 | CoCoGlide IoU | AutoSplice F1 | AutoSplice IoU | MagicBrush F1 | MagicBrush IoU | DEAL-300K F1 | DEAL-300K IoU | OpenSDI F1 | OpenSDI IoU | Average F1 | Average IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PixDiff-Fixed (τ=0) | 35.46 | 25.22 | 53.90 | 40.94 | 23.48 | 14.79 | 16.38 | 10.33 | 24.65 | 16.17 | 30.77 | 21.49 |
| PixDiff-Fixed (τ=10) | 92.26 | 86.22 | 92.18 | 86.03 | 74.89 | 62.93 | 28.04 | 19.11 | 68.11 | 56.18 | 71.10 | 62.09 |
| PixDiff-Fixed (τ=20) | 82.01 | 71.00 | 86.15 | 76.57 | 69.11 | 55.52 | 33.44 | 23.28 | 63.85 | 49.65 | 66.91 | 55.20 |
| PixDiff-Fixed (τ=30) | 72.27 | 58.62 | 79.76 | 67.50 | 62.10 | 47.66 | 35.24 | 24.41 | 55.50 | 40.61 | 60.97 | 47.76 |
| PixDiff-Fixed (τ=40) | 63.34 | 48.59 | 73.46 | 59.37 | 55.45 | 40.83 | 35.24 | 24.10 | 47.23 | 32.81 | 54.94 | 41.14 |
| PixDiff-Fixed-morph | 59.92 | 46.12 | 77.67 | 65.22 | 64.06 | 50.03 | 46.33 | 34.44 | 51.91 | 37.45 | 59.98 | 46.65 |
| PixDiff-Otsu | 62.40 | 46.54 | 62.97 | 47.07 | 51.88 | 36.84 | 34.21 | 23.17 | 50.07 | 34.95 | 52.31 | 37.71 |
| Qwen2.5VL-LangSAM | 84.54 | 77.66 | 86.19 | 80.30 | 38.95 | 29.09 | 43.13 | 33.68 | 43.71 | 36.19 | 59.31 | 51.38 |
| Meta-CD (TGRS 2025) | 73.50 | 62.40 | 86.90 | 77.80 | 76.00 | 64.10 | 45.80 | 33.80 | 74.28 | 62.19 | 71.30 | 60.06 |
| DDPM-CD (WACV 2025) | **97.80** | **96.00** | 96.20 | 92.80 | **89.90** | **83.10** | 20.90 | 13.80 | 83.35 | 76.14 | 77.63 | 72.37 |
| SIGMA | 95.62 | 91.84 | **96.69** | **93.74** | 89.03 | 81.91 | **76.43** | **65.31** | **91.37** | **84.82** | **89.83** | **83.52** |

![Qualitative comparison of SIGMA and other methods.](assets/qualitative_comparison.png)

*Qualitative comparison of SIGMA and other methods.*

![Robustness to JPEG compression, Gaussian noise, and Gaussian blur on AutoSplice and OpenSDI.](assets/robustness.png)

*Robustness to JPEG compression, Gaussian noise, and Gaussian blur on AutoSplice and OpenSDI.*

Cross-dataset localization after fine-tuning on automatically annotated editing data. For each detector we report publicly released weights, fine-tuning on the same image pairs annotated by DDPM-CD, and fine-tuning on those pairs annotated by SIGMA (Finetune w/ SA). F1 and IoU scores (%) are reported. The best result within each method is in **bold**.

| Method | Weights | AutoSplice F1 | AutoSplice IoU | CoCoGlide F1 | CoCoGlide IoU | DEAL-300K F1 | DEAL-300K IoU | MagicBrush F1 | MagicBrush IoU | OpenSDI F1 | OpenSDI IoU | Average F1 | Average IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAT-Net | Public Weight | 41.49 | 31.79 | 40.97 | 33.69 | 12.14 | 8.75 | 5.33 | 3.29 | 7.32 | 5.27 | 21.45 | 16.56 |
| CAT-Net | Finetune w/ DDPM-CD | 44.27 | 30.48 | 32.83 | 21.99 | 13.83 | 8.40 | 59.19 | 42.99 | 20.56 | 12.68 | 34.14 | 23.31 |
| CAT-Net | Finetune w/ SA | **60.82** | **47.23** | **45.53** | **34.42** | **29.99** | **20.98** | **61.02** | **50.05** | **27.02** | **18.14** | **44.88** | **34.16** |
| MVSSNet | Public Weight | 18.04 | 12.22 | 32.80 | 25.30 | 5.43 | 3.61 | 1.79 | 1.01 | 1.46 | 0.91 | 11.90 | 8.61 |
| MVSSNet | Finetune w/ DDPM-CD | 53.98 | 41.03 | 35.51 | 25.27 | 16.40 | 10.34 | **91.20** | **85.10** | **25.06** | **16.46** | **44.43** | **35.64** |
| MVSSNet | Finetune w/ SA | **54.14** | **41.43** | **35.63** | **25.38** | **20.05** | **13.29** | 81.95 | 73.72 | 21.69 | 14.51 | 42.69 | 33.67 |
| PSCC-Net | Public Weight | 56.36 | 43.83 | **50.85** | **40.60** | 20.19 | 13.69 | 14.87 | 9.37 | 14.50 | 9.64 | 31.35 | 23.43 |
| PSCC-Net | Finetune w/ DDPM-CD | 54.19 | 41.33 | 36.16 | 27.46 | 16.22 | 10.77 | **41.15** | **32.71** | 18.54 | 12.10 | 33.25 | 24.87 |
| PSCC-Net | Finetune w/ SA | **57.06** | **45.46** | 42.30 | 31.45 | **21.16** | **14.20** | 36.69 | 26.67 | **22.44** | **15.25** | **35.93** | **26.61** |
| TruFor | Public Weight | 32.69 | 25.52 | 28.33 | 24.33 | 9.26 | 6.77 | 2.51 | 1.48 | 6.93 | 5.22 | 15.94 | 12.66 |
| TruFor | Finetune w/ DDPM-CD | 57.06 | 41.97 | 55.33 | 42.31 | 22.49 | 14.36 | 20.63 | 12.36 | 20.13 | 12.78 | 35.13 | 24.76 |
| TruFor | Finetune w/ SA | **63.42** | **50.63** | **58.72** | **47.34** | **27.85** | **18.92** | **28.35** | **20.06** | **27.02** | **19.20** | **41.07** | **31.23** |

![Overview of SIGMA-converted IML training data from four generative image editing corpora and their editing operation taxonomy.](assets/dataset_distribution.png)

*Overview of SIGMA-converted IML training data from four generative image editing corpora and their editing operation taxonomy.*

The full annotated set is released on Hugging Face at [PatrickZZZ/SIGMA](https://huggingface.co/datasets/PatrickZZZ/SIGMA). Two smaller splits of the same annotations are also available on ModelScope:

- **10,000** samples: [PeiyuZhuang/SIGMA_small](https://www.modelscope.ai/datasets/PeiyuZhuang/SIGMA_small)
- **100,000** samples: [PeiyuZhuang/SIGMA_partial](https://www.modelscope.ai/datasets/PeiyuZhuang/SIGMA_partial)

## Data Use and Citation

The SIGMA-annotated data released with this project are derived from public image-editing corpora. **They may be used for academic research only.** Any use must also follow the license and terms of each source dataset, and must cite both that dataset and SIGMA.

| Source | Link | License / terms |
| --- | --- | --- |
| PromptfixData | [yeates/PromptfixData](https://huggingface.co/datasets/yeates/PromptfixData) | Research use only. The dataset and any model trained on it may not be used for commercial purposes. Users must also comply with the licenses of its source data. |
| AnyEdit | [Bin1117/AnyEdit](https://huggingface.co/datasets/Bin1117/AnyEdit) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its primary intended use is research on text-to-image and instruction-based image editing. |
| CrispEdit-2M | [WeiChow/CrispEdit-2M](https://huggingface.co/datasets/WeiChow/CrispEdit-2M) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). |
| OmniEdit-Filtered-1.2M | [TIGER-Lab/OmniEdit-Filtered-1.2M](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M) | [MIT](https://opensource.org/license/mit). |

Please cite SIGMA together with the source papers:

```bibtex
@misc{zhuang2026sigmasemanticdifferenceinstructiongroundingmask,
  title={SIGMA: Semantic-Difference Instruction-Grounding Mask Annotator for Text-Driven Image Manipulation Localization},
  author={Peiyu Zhuang and Jianquan Yang and Haodong Li and Zhuoying Cai and Ruitao Xie and Jishen Zeng and Baoying Chen and Jiwu Huang and Xiaochun Cao},
  year={2026},
  eprint={2605.27924},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.27924}
}

@inproceedings{yu2025anyedit,
  title={AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea},
  author={Yu, Qifan and Chow, Wei and Yue, Zhongqi and Pan, Kaihang and Wu, Yang and Wan, Xiaoyang and Li, Juncheng and Tang, Siliang and Zhang, Hanwang and Zhuang, Yueting},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26125--26135},
  year={2025}
}

@article{chow2025editmgt,
  title={EditMGT: Unleashing Potentials of Masked Generative Transformers in Image Editing},
  author={Chow, Wei and Li, Linfeng and Kong, Lingdong and Li, Zefeng and Xu, Qi and Song, Hang and Ye, Tian and Wang, Xian and Bai, Jinbin and Xu, Shilin and others},
  journal={arXiv preprint arXiv:2512.11715},
  year={2025}
}

@inproceedings{wei2025omniedit,
  title={OmniEdit: Building Image Editing Generalist Models Through Specialist Supervision},
  author={Wei, Cong and Xiong, Zheyang and Ren, Weiming and Du, Xinrun and Zhang, Ge and Chen, Wenhu},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}

@article{yu2024promptfix,
  title={PromptFix: You Prompt and We Fix the Photo},
  author={Yu, Yongsheng and Zeng, Ziyun and Hua, Hang and Fu, Jianlong and Luo, Jiebo},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={40000--40031},
  year={2024}
}
```

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

Download the SIGMA checkpoint and the bundled runtime resources from the Hugging Face model repository [PatrickZZZ/SIGMA](https://huggingface.co/PatrickZZZ/SIGMA):

```bash
hf download PatrickZZZ/SIGMA model.pt --local-dir weights
hf download PatrickZZZ/SIGMA --include "resource/**" --local-dir .
```

The first command saves the main checkpoint to `weights/model.pt`. The second command places the following resources under `resource/`:

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

The default main checkpoint is `weights/model.pt`, released at [PatrickZZZ/SIGMA](https://huggingface.co/PatrickZZZ/SIGMA). Download it with the command in Section 2.

To use a different checkpoint, override via CLI:

```bash
python scripts/run_anyedit_selected_test.py --ckpt weights/your_model.pt
```
