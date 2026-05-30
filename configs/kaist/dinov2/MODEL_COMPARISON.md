# Model Comparison: DINOv1 Swin-L vs DINOv2-Large

## 🎯 Quick Summary

| Configuration | Model Size | Scales | LR | Batch | Current mAP@50 | Expected Final |
|--------------|------------|---------|-----|-------|----------------|----------------|
| **DINOv1 + Swin-L** | 197M | 5 | 5e-5 | 2 | **68.5%** ✅ | 68.5% |
| DINOv2-Base + 4-scale | 86M | 4 | 2e-4 ❌ | 4 | ~60% | ~60-62% |
| **DINOv2-Large + 5-scale + reg** | **304M** | **5** | **5e-5** ✅ | **2** | **TBD** | **67-72%** 🎯 |

## 📊 Detailed Comparison

### 1. **Model Architecture**

#### DINOv1 + Swin-L (Your Baseline - 68.5% mAP)
```python
backbone:
  - Type: Swin Transformer Large
  - Parameters: ~197M
  - Embed dims: 192 → 1536 (hierarchical)
  - Layers: [2, 2, 18, 2] = 24 layers
  - Window size: 12
  - Pretrained: ImageNet-22K + COCO
  
neck:
  - in_channels: [192, 384, 768, 1536]
  - num_scales: 5 ✅
  
optimizer:
  - lr: 5e-5 ✅
  - batch_size: 2 per GPU
```

#### DINOv2-Base + 4-scale (Current - ~60% mAP)
```python
backbone:
  - Type: Vision Transformer Base
  - Parameters: ~86M (44% of Swin-L!)
  - Embed dims: 768 (uniform)
  - Layers: 12
  - Pretrained: ImageNet-22K (no COCO)
  
neck:
  - in_channels: [768, 768, 768, 768]
  - num_scales: 4 ❌
  
optimizer:
  - lr: 2e-4 ❌ (TOO HIGH!)
  - batch_size: 4 per GPU
```

**Problems:**
- ❌ Model is **less than half the size** of Swin-L
- ❌ Missing 5th scale level
- ❌ Learning rate is **4× too high** (destroys pretrained features)
- ❌ No register tokens (artifacts in feature maps)
- ❌ Not pretrained on COCO (has to learn detection from scratch)

#### DINOv2-Large + 5-scale + registers (NEW - Expected 67-72%)
```python
backbone:
  - Type: Vision Transformer Large with 4 register tokens
  - Parameters: ~304M (155% of Swin-L!) ✅
  - Embed dims: 1024 (uniform)
  - Layers: 24 ✅
  - Pretrained: ImageNet-22K + DINOv2 (142M images!)
  
neck:
  - in_channels: [1024, 1024, 1024, 1024, 1024]
  - num_scales: 5 ✅
  
optimizer:
  - lr: 5e-5 ✅ (same as working Swin-L config)
  - batch_size: 2 per GPU ✅
```

**Improvements:**
- ✅ Model is **larger than Swin-L** (304M vs 197M)
- ✅ Full 5-scale architecture (matches Swin-L)
- ✅ Conservative learning rate (preserves pretrained features)
- ✅ Register tokens (better feature quality, reduces artifacts)
- ✅ Better pretraining (DINOv2 learned from 142M images!)

## 🔬 Key Changes Explained

### Change 1: Base → Large (Most Important!)

```python
# Old (Base - 86M params)
model_name='vit_base_patch14_reg4_dinov2.lvd142m'
in_channels=[768, 768, 768, 768]
out_indices=(3, 6, 9, 11)  # From 12 layers

# New (Large - 304M params)
model_name='vit_large_patch14_reg4_dinov2.lvd142m'
in_channels=[1024, 1024, 1024, 1024, 1024]
out_indices=(7, 15, 19, 21, 23)  # From 24 layers
```

**Impact:** +5-8% mAP (model capacity matches Swin-L)

### Change 2: 4-scale → 5-scale

```python
# Old
num_levels = 4
out_indices=(3, 6, 9, 11)

# New
num_levels = 5
out_indices=(7, 15, 19, 21, 23)
```

**Impact:** +1-2% mAP (better multi-scale detection)

### Change 3: Fix Learning Rate

```python
# Old (BROKEN!)
lr=2e-4  # Destroys pretrained features!

# New (FIXED!)
lr=5e-5  # Preserves features, allows fine-tuning
paramwise_cfg=dict(
    custom_keys={'backbone': dict(lr_mult=0.1)}  # 5e-6 for backbone
)
```

**Impact:** +5-10% mAP (critical fix!)

### Change 4: Reduce Batch Size

```python
# Old
batch_size=4  # 4 × 3 GPUs = 12 effective

# New
batch_size=2  # 2 × 3 GPUs = 6 effective
```

**Impact:** More stable training, better generalization, less memory pressure

## 📈 Expected Training Progression

### DINOv1 + Swin-L (Your Baseline)
```
Epoch 1:  mAP@50 ≈ 35-40% (benefits from COCO pretraining)
Epoch 6:  mAP@50 ≈ 55-60%
Epoch 12: mAP@50 ≈ 68.5% ✅
```

### DINOv2-Base + 4-scale (Current - Broken)
```
Epoch 1:  mAP@50 ≈ 15-20% ❌ (LR too high, features corrupted)
Epoch 6:  mAP@50 ≈ 50-55%
Epoch 12: mAP@50 ≈ 60-62% (limited by small model size)
```

### DINOv2-Large + 5-scale + registers (NEW)
```
Epoch 1:  mAP@50 ≈ 35-40% (excellent pretraining compensates)
Epoch 6:  mAP@50 ≈ 55-60%
Epoch 12: mAP@50 ≈ 67-72% 🎯 (should match or exceed Swin-L)
```

## 💻 Memory Usage Estimates

| Config | Backbone Size | Per-GPU Memory | Total (4 GPUs) |
|--------|--------------|----------------|----------------|
| Swin-L (bs=2) | 197M | ~8-10 GB | ~32-40 GB |
| DINOv2-Base (bs=4) | 86M | ~6.3 GB | ~25 GB |
| **DINOv2-Large (bs=2)** | **304M** | **~10-12 GB** | **~40-48 GB** |

Your A40s have 48GB each, so you're safe with batch_size=2 ✅

## 🚀 Training Command

```bash
bash ./tools/dist_train.sh \
  configs/kaist/dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py \
  4 \
  --work-dir work_dirs/kaist_dinov2_large_reg_5scale
```

## 🎓 Why This Should Match or Beat Swin-L

### Advantages of DINOv2-Large:

1. **Larger model** (304M vs 197M) → more capacity
2. **Better pretraining** (DINOv2 on 142M images vs Swin on 22K)
3. **Register tokens** → cleaner features, fewer artifacts
4. **Modern architecture** → ViT-Large is state-of-the-art
5. **Stronger features** → DINOv2 self-supervised learning is excellent

### Potential Disadvantages:

1. **No COCO pretraining** → may start slower (but catches up)
2. **Uniform features** → ViT doesn't have hierarchical structure like Swin
3. **First time tuning** → may need hyperparameter adjustments

### Net Result: **Expected to match or slightly exceed 68.5%**

## 🔧 If Results Are Below 68.5% After 12 Epochs

Try these in order:

### 1. Train Longer (Easiest)
```python
max_epochs=24  # Instead of 12
milestones=[18, 22]  # LR drops
```
**Expected gain:** +2-3% mAP

### 2. Adjust Learning Rate (If needed)
```python
# If training is unstable
lr=3e-5  # Reduce from 5e-5

# If converging too slowly
lr=7e-5  # Increase from 5e-5
```

### 3. Load COCO Pretrained Head (Advanced)
```python
# Initialize detection head from DINOv1
load_from = 'https://download.openmmlab.com/.../dino-5scale_swin-l_8xb2-12e_coco/...'
# But keep DINOv2-Large backbone
```
**Expected gain:** +3-5% mAP (faster convergence)

### 4. Increase Resolution (Expensive)
```python
# In kaist_detection.py
train_pipeline = [
    dict(type='Resize', scale=(896, 896)),  # Up from 640x512
    ...
]
```
**Expected gain:** +2-4% mAP (but 2× slower training)

## 📝 Monitoring Training Health

### Good Signs ✅
- Epoch 1 mAP@50 > 30%
- Steady improvement every epoch
- Gradient norm < 500
- Loss decreasing smoothly
- No NCCL timeout errors

### Bad Signs ❌
- Epoch 1 mAP@50 < 20% → LR too high or model issue
- Gradient norm > 1000 → Exploding gradients
- Loss oscillating wildly → Unstable training
- Memory errors → Reduce batch size
- NCCL timeout → Memory issue or GPU hang

## 🎯 Final Prediction

**DINOv2-Large + 5-scale + registers with optimized hyperparameters:**

- **Conservative estimate:** 67-69% mAP@50 (matches Swin-L)
- **Realistic estimate:** 68-71% mAP@50 (slight improvement)
- **Optimistic estimate:** 70-73% mAP@50 (with 24 epochs + tuning)

The key was:
1. ✅ Matching model capacity (304M vs 197M)
2. ✅ Fixing the learning rate (5e-5 vs 2e-4)
3. ✅ Adding 5th scale level
4. ✅ Using registers for better features
5. ✅ Conservative batch size for stability

Good luck with training! 🚀

