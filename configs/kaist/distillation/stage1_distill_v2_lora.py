# -------------------------------------------------------------------------
# Stage 1 v2 + LoRA: Multi-Teacher Feature Distillation with LoRA Adapters
# -------------------------------------------------------------------------
# Inherits the v2 config (cosine sim, 12 layer pairs, CLIP ViT-L/14) and
# adds LoRA adapters to the student backbone.
#
# Key difference from v2:
#   - Student backbone is FROZEN (original DINOv2 weights preserved 100%)
#   - LoRA adapters (rank=16) injected into all attention QKV projections
#   - Only LoRA params + projectors are trained
#   - Higher base LR (1e-4) since LoRA/projectors are randomly initialized
#
# LoRA parameter count for ViT-Large (24 blocks, QKV only):
#   Per block: A=(1024, 16) + B=(16, 3072) = 16384 + 49152 = 65536 params
#   24 blocks: 24 * 65536 = 1,572,864 params (~0.5% of 304M backbone)
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_distill_v2_lora.py 4 --work-dir ./work_dirs/stage1_distill_v2_lora
# -------------------------------------------------------------------------
_base_ = [
    './stage1_distill_v2.py'
]

# -------------------------------------------------------------------------
# 1. Add LoRA config to the model
# -------------------------------------------------------------------------
# Override model to add lora_cfg (everything else inherited from v2)
model = dict(
    lora_cfg=dict(
        rank=16,                        # Low-rank dimension
        alpha=16.0,                     # Scaling factor (alpha/rank = 1.0)
        dropout=0.05,                   # Light regularization
        target_modules=['attn.qkv'],    # Inject into QKV projections only
    )
)

# -------------------------------------------------------------------------
# 2. Learning Rate: higher for LoRA + projectors (randomly initialized)
# -------------------------------------------------------------------------
# Backbone is frozen via requires_grad=False, so backbone lr_mult is
# irrelevant. LoRA params and projectors both get the base LR.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=1e-4),
    # Must use _delete_=True to remove inherited paramwise_cfg from base.
    # Without it, LoRA params inside backbone get lr_mult=0.1 (10x too low).
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# -------------------------------------------------------------------------
# 3. Work directory
# -------------------------------------------------------------------------
work_dir = './work_dirs/stage1_distill_v2_lora'
