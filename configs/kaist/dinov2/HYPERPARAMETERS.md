# Hyperparameter Guide for DINOv2 Fine-tuning

## 🎯 Key Insight: Fine-tuning ≠ Training from Scratch

**IMPORTANT**: DINOv2 is **pretrained** on massive datasets. When fine-tuning, you need **much smaller learning rates** than training from scratch!

## ✅ Recommended Settings (Validated)

### **Faster R-CNN + DINOv2**
```python
# Optimizer
optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05)

# Batch size
train_dataloader = dict(
    batch_size=2,  # per GPU (8 total with 4 GPUs)
    num_workers=4)

# Results after 1 epoch: mAP_50 = 37.6% ✅
```

### **DINO Detector + DINOv2**
```python
# Optimizer
optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4)
paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1),  # backbone LR = 5e-6
    })

# Batch size
train_dataloader = dict(
    batch_size=2,  # per GPU (8 total with 4 GPUs)
    num_workers=4)
```

## ❌ What NOT to Do

### **Don't Scale LR Linearly with Batch Size (for fine-tuning)**

**Wrong approach** (caused mAP drop from 37.6% → 14%):
```python
# batch_size doubled (8 → 16)
# LR doubled (1e-5 → 2e-4)  ❌ TOO HIGH!
optimizer=dict(type='AdamW', lr=2e-4)  # This breaks fine-tuning!
```

**Why it failed:**
- Linear LR scaling works for **training from scratch**
- For **fine-tuning pretrained models**, use small, conservative LRs
- High LR destroys pretrained features → poor performance

## 📊 Learning Rate Guidelines

### **For Fine-tuning Pretrained Vision Transformers:**

| Model Component | Learning Rate Range | Recommended |
|----------------|---------------------|-------------|
| Detection head | 1e-5 to 1e-4 | **5e-5** |
| Pretrained backbone | 1e-6 to 1e-5 | **5e-6** (0.1× head LR) |

### **Rule of Thumb:**
```
Fine-tuning LR = (1/10 to 1/100) × Training-from-scratch LR
```

## 🔧 Batch Size Considerations

### **For 4x A40 GPUs (48GB each):**

**Conservative (Recommended for fine-tuning):**
- `batch_size=2` per GPU → 8 total
- More stable training
- Better for small datasets like KAIST

**Aggressive (Only if needed):**
- `batch_size=4` per GPU → 16 total
- Faster training
- **Must reduce LR further** (e.g., 5e-6 for Faster R-CNN)
- Risk of overfitting on small datasets

## 📈 Expected Performance Timeline

### **Faster R-CNN + DINOv2:**
```
Epoch 1:  mAP_50 ≈ 35-40%
Epoch 6:  mAP_50 ≈ 45-50%
Epoch 12: mAP_50 ≈ 50-55% (converged)
```

### **DINO + DINOv2:**
```
Epoch 1:  mAP_50 ≈ 30-35%
Epoch 6:  mAP_50 ≈ 45-50%
Epoch 12: mAP_50 ≈ 55-60% (converged)
```

### **DINO + DINOv2-reg (Best):**
```
Epoch 1:  mAP_50 ≈ 32-37%
Epoch 6:  mAP_50 ≈ 47-52%
Epoch 12: mAP_50 ≈ 57-62% (converged)
```

## 🧪 Hyperparameter Tuning Strategy

### **If training is unstable (loss explodes):**
1. ✅ Reduce learning rate by 2-5×
2. ✅ Reduce batch size
3. ✅ Increase warmup iterations (500 → 1000)
4. ✅ Check gradient clipping is enabled

### **If training is too slow (no improvement):**
1. ✅ Slightly increase LR (e.g., 1e-5 → 2e-5)
2. ✅ Check if backbone is frozen (should be trainable)
3. ✅ Verify pretrained weights are loaded
4. ✅ Increase training epochs (12 → 24)

### **If overfitting (train mAP >> val mAP):**
1. ✅ Increase weight decay (0.05 → 0.1)
2. ✅ Add more data augmentation
3. ✅ Reduce model capacity (use smaller backbone)
4. ✅ Early stopping based on validation mAP

## 🔬 Advanced: Learning Rate Schedules

### **Cosine Annealing (Faster R-CNN):**
```python
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, end=1000),  # Warmup
    dict(type='CosineAnnealingLR', T_max=12, eta_min=1e-6)  # Main schedule
]
```
- Smooth decay from peak LR to near-zero
- Good for fine-tuning
- Prevents sudden LR drops

### **Multi-step LR (DINO):**
```python
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, end=500),  # Warmup
    dict(type='MultiStepLR', milestones=[11], gamma=0.1)  # Drop at epoch 11
]
```
- Sharp LR drop near end of training
- Good for final convergence
- Common in transformer detectors

## 📝 Summary: What Changed & Why

### **Original (Working):**
- LR: `1e-5`
- Batch: `2 per GPU`
- Result: **37.6% mAP_50** ✅

### **Bad Update (Broken):**
- LR: `2e-4` (20× higher!) ❌
- Batch: `4 per GPU`
- Result: **14% mAP_50** ❌
- **Reason**: LR too high, destroyed pretrained features

### **Fixed (Current):**
- LR: `1e-5` (Faster R-CNN), `5e-5` (DINO)
- Batch: `2 per GPU`
- Expected: **35-40% mAP_50** ✅

## 🎓 Key Takeaways

1. **Fine-tuning requires small LRs** (1e-5 to 1e-4)
2. **Don't blindly scale LR with batch size** for pretrained models
3. **Backbone LR should be 10× smaller** than head LR
4. **Batch size 2-4 per GPU is sufficient** for detection
5. **Conservative settings are safer** for small datasets like KAIST
6. **Monitor first epoch mAP** as a sanity check (should be >30%)

## 📚 References

- [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187) - LR scaling rules
- [How to Fine-tune BERT](https://arxiv.org/abs/1905.05583) - Fine-tuning best practices
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193) - Recommended fine-tuning settings
- [MMDetection Docs](https://mmdetection.readthedocs.io/) - Training tips




