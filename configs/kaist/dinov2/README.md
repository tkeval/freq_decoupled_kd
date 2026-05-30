# DINOv2 Experiments on KAIST Dataset

This directory contains configurations for object detection on the KAIST pedestrian dataset using DINOv2 backbones.

## 📁 Configurations

### 1. **faster-rcnn_dinov2-base-p14_fpn_8xb2-12e_kaist.py**
- **Detector**: Faster R-CNN (traditional two-stage detector)
- **Backbone**: DINOv2-Base (ViT-B/14)
- **Neck**: FPN (Feature Pyramid Network)
- **Training**: 12 epochs, AdamW optimizer
- **Use case**: Baseline comparison with traditional detector

### 2. **dino-4scale_dinov2-base-p14_8xb2-12e_kaist.py**
- **Detector**: DINO (transformer-based detector)
- **Backbone**: DINOv2-Base (ViT-B/14)
- **Features**: 4-scale multi-level features
- **Training**: 12 epochs, AdamW optimizer with denoising
- **Use case**: Modern transformer detector for better performance

### 3. **dino-4scale_dinov2-base-p14-reg4_8xb2-12e_kaist.py** ⭐ **Recommended**
- **Detector**: DINO (transformer-based detector)
- **Backbone**: DINOv2-Base with 4 register tokens (improved variant)
- **Features**: 4-scale multi-level features
- **Training**: 12 epochs, AdamW optimizer with denoising
- **Use case**: Best expected performance (~2-3% better than standard DINOv2)

## 🚀 Quick Start

### Run All Experiments Sequentially
```bash
# Option 1: Python script
python run_dinov2_experiments.py

# Option 2: Bash script
bash run_dinov2_experiments.sh
```

### Run Individual Experiments
```bash
# Experiment 1: Faster R-CNN
bash ./tools/dist_train.sh configs/kaist/dinov2/faster-rcnn_dinov2-base-p14_fpn_8xb2-12e_kaist.py 4 \
    --work-dir work_dirs/kaist_dinov2_frcnn

# Experiment 2: DINO detector
bash ./tools/dist_train.sh configs/kaist/dinov2/dino-4scale_dinov2-base-p14_8xb2-12e_kaist.py 4 \
    --work-dir work_dirs/kaist_dinov2_dino_head

# Experiment 3: DINO + registers (best)
bash ./tools/dist_train.sh configs/kaist/dinov2/dino-4scale_dinov2-base-p14-reg4_8xb2-12e_kaist.py 4 \
    --work-dir work_dirs/kaist_dinov2_reg_dino_head
```

## ⚙️ Hardware Configuration

**Optimized for: 4x NVIDIA A40 GPUs (48GB each)**

### Training Settings
- **Batch size**: 2 per GPU (8 total) - Conservative for fine-tuning
- **Learning rate**: 
  - Faster R-CNN: 1e-5 (conservative for pretrained backbone)
  - DINO: 5e-5 (with 0.1× multiplier for backbone = 5e-6)
- **Workers**: 4 per GPU
- **Mixed precision**: Enabled (automatic)
- **Gradient clipping**: Yes

### Memory Usage (Estimated)
- Faster R-CNN: ~8-12GB per GPU
- DINO detector: ~12-16GB per GPU
- DINO + registers: ~12-16GB per GPU

**Note**: Batch size kept small (2 per GPU) for stable fine-tuning of pretrained models.

## 📊 Expected Performance

Based on typical DINOv2 performance on pedestrian detection:

| Model | Expected mAP@0.5:0.95 | Training Time (12 epochs) |
|-------|----------------------|---------------------------|
| Faster R-CNN + DINOv2 | 35-40% | ~3-4 hours |
| DINO + DINOv2 | 40-45% | ~4-5 hours |
| DINO + DINOv2-reg ⭐ | 42-47% | ~4-5 hours |

*Note: Actual performance depends on dataset quality and training conditions*

## 🔧 Key Hyperparameters

### Learning Rate Schedule
- **Warmup**: Linear warmup for 500 iterations (start_factor=0.001)
- **Main schedule**: 
  - Faster R-CNN: Cosine annealing (eta_min=1e-6)
  - DINO: Multi-step LR decay at epoch 11 (gamma=0.1)
- **Backbone LR multiplier**: 0.1× (lower LR for pretrained backbone)

### Data Augmentation
- **Resize**: (640, 512) with aspect ratio preservation
- **Random flip**: 50% probability
- **Pad divisor**: 14 (matches DINOv2 patch size)
- **Filter empty GT**: True (removes images without annotations)

### Optimizer
- **Type**: AdamW
- **Learning Rate**:
  - Faster R-CNN: 1e-5 (conservative for fine-tuning)
  - DINO: 5e-5 (backbone gets 0.1× = 5e-6)
- **Weight decay**: 
  - Faster R-CNN: 0.05
  - DINO: 1e-4
- **Gradient clipping**: 
  - Faster R-CNN: max_norm=35
  - DINO: max_norm=0.1

**Important**: These LRs are optimized for **fine-tuning pretrained DINOv2**. Do NOT scale linearly with batch size!

## 📈 Monitoring Training

### View Logs
```bash
# Real-time training log
tail -f work_dirs/kaist_dinov2_*/*/log.txt

# TensorBoard
tensorboard --logdir work_dirs/
```

### Check Results
```bash
# Latest checkpoint
ls -lh work_dirs/kaist_dinov2_*/latest.pth

# Best checkpoint (auto-saved)
ls -lh work_dirs/kaist_dinov2_*/best_*.pth
```

## 🧪 Evaluation

### Evaluate Trained Model
```bash
python tools/test.py \
    configs/kaist/dinov2/dino-4scale_dinov2-base-p14-reg4_8xb2-12e_kaist.py \
    work_dirs/kaist_dinov2_reg_dino_head/latest.pth \
    --work-dir work_dirs/kaist_dinov2_reg_dino_head/eval
```

### Visualize Predictions
```bash
python tools/test.py \
    configs/kaist/dinov2/dino-4scale_dinov2-base-p14-reg4_8xb2-12e_kaist.py \
    work_dirs/kaist_dinov2_reg_dino_head/latest.pth \
    --show-dir work_dirs/kaist_dinov2_reg_dino_head/vis
```

## 🔍 Troubleshooting

### Out of Memory (OOM)
Reduce batch size in config:
```python
train_dataloader = dict(
    batch_size=2,  # Reduce from 4 to 2
    num_workers=2)
```

### Slow Training
- Check if `persistent_workers=True` in dataloader
- Increase `num_workers` if CPU is underutilized
- Verify GPU utilization with `nvidia-smi`

### Poor Performance
- Ensure `filter_empty_gt=True` in dataset config
- Check learning rate (should be ~2e-4 for batch_size=16)
- Verify backbone is loading pretrained weights
- Train for more epochs (24-36 for better convergence)

## 📚 References

- **DINOv2**: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **DINO Detector**: [DINO: DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605)
- **KAIST Dataset**: [Multispectral Pedestrian Detection Benchmark](https://soonminhwang.github.io/rgbt-ped-detection/)
- **MMDetection**: [OpenMMLab Detection Toolbox](https://github.com/open-mmlab/mmdetection)

## 📝 Notes

- All configs use **RGB images only** (not thermal)
- Dataset class: `person` (single class detection)
- Pretrained weights: Automatically downloaded from TIMM
- Config naming follows MMDetection conventions
- `filter_empty_gt=True` is recommended for KAIST (many empty frames)
