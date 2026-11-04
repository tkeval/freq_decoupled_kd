# FreqKD: Frequency-Decoupled Cross-Modal Knowledge Distillation

Anonymous code release for the BMVC submission *"FreqKD: Frequency-Decoupled
Cross-Modal Knowledge Distillation for Infrared Object Detection."*

This repository is built on top of MMDetection 3.x. Only the components below
are specific to FreqKD; the remainder is the standard MMDetection codebase.

## Method code
- `mmdet/models/distillers/freq_decoupled_distiller.py` — Stage-1 frequency-decoupled distiller (ViT→ViT).
- `mmdet/models/distillers/cross_arch_freq_distiller.py` — cross-architecture variant (ViT→ResNet-50).
- `mmdet/models/distillers/freq_decoupled_cnn.py` — CNN distiller / Stage-2 Faster R-CNN helper.
- `mmdet/models/distillers/stage2_guided_detector.py` — Stage-2 detector with LoRA merge.
- `mmdet/models/utils/lora.py` — LoRA injection and merge utilities.
- `mmdet/datasets/kaist.py`, `mmdet/datasets/transforms/load_multi_images.py` — paired RGB–IR loading.

## Configs
- `configs/kaist/distillation/stage1_freq_decoupled.py` — Stage 1 (FreqKD pre-training).
- `configs/kaist/distillation/stage2_freq_decoupled_det.py` — Stage 2 (detection fine-tuning).
- `configs/kaist/distillation/stage1_freq_low_only.py`, `stage1_freq_high_only.py` — band ablations.
- `configs/kaist/distillation/stage1_freq_decoupled_freq_cutoff_*.py` — cut-off sweep.
- `configs/kaist/distillation/lora_rank_ablation/` — LoRA rank / full-finetune ablation (Stage 1 + 2).
- `configs/kaist/distillation/lora_alpha_merge_hyperparameter/` — LoRA merge-scale ablation (Stage 2).
- `configs/kaist/frcnn/` — cross-architecture (ResNet-50) configs.
- `configs/flir/distillation/` — cross-dataset transfer (FLIR).

## Analysis scripts (post-hoc, no training)
- `tools/spectral_divergence_analysis.py` — per-band RGB–IR feature divergence (Table 2).
- `tools/cka_analysis.py` — teacher–student CKA, full / low / high bands (Table 6).
- `tools/mmd_freq_analysis.py` — frequency-aware RBF-MMD diagnostic (Table 8).

## Paths
Configs and scripts reference two locations via relative paths:
- `./data/kaist-rgbt/` — KAIST multispectral dataset (RGB+thermal, COCO-style annotations).
- `./work_dirs/` — training outputs and checkpoints.

Set these to your local data/output locations (symlinks work). Other datasets
(FLIR ADAS, MFNet) follow the same convention under `./data/`.

## Training

Stage 1 (frequency-decoupled backbone pre-training):
```bash
bash ./tools/dist_train.sh configs/kaist/distillation/stage1_freq_decoupled.py 4 \
    --work-dir ./work_dirs/stage1_freq_decoupled
```

Stage 2 (detection fine-tuning from the Stage-1 checkpoint):
```bash
bash ./tools/dist_train.sh configs/kaist/distillation/stage2_freq_decoupled_det.py 4 \
    --work-dir ./work_dirs/stage2_freq_decoupled
```
Set `teacher_checkpoint` in the Stage-2 config to the Stage-1 `epoch_12.pth`.

## Environment
Python 3.8, PyTorch 2.x, MMEngine, MMCV, MMDetection 3.x, and MMPretrain
(for the DINOv2 ViT-L backbone via `TIMMBackbone`). See the standard
MMDetection installation instructions.
