# -------------------------------------------------------------------------
# Stage 1 v2 + Strong LoRA: Multi-Teacher KD with High-Capacity LoRA
# -------------------------------------------------------------------------
# Stronger LoRA config vs v2_lora:
#   - Rank 64 (vs 16): 4x more capacity per adapter
#   - Target QKV + MLP (vs QKV only): covers full transformer block
#   - Correct LR 2e-4 for batch_size=8 (vs 1e-4 which was undertrained)
#   - ~7% trainable backbone params (vs 0.5%)
#
# LoRA parameter count for ViT-Large (24 blocks, QKV + MLP):
#   attn.qkv: 24 * (1024*64 + 64*3072) = 6,291,456
#   mlp.fc1:  24 * (1024*64 + 64*4096) = 7,864,320
#   mlp.fc2:  24 * (4096*64 + 64*1024) = 7,864,320
#   Total: ~22M params (~7.2% of 304M backbone)
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_distill_v2_lora_strong.py 4 \
#       --work-dir ./work_dirs/stage1_v2_lora_strong
# -------------------------------------------------------------------------
_base_ = [
    './stage1_distill_v2.py'
]

# -------------------------------------------------------------------------
# 1. Strong LoRA config
# -------------------------------------------------------------------------
model = dict(
    lora_cfg=dict(
        rank=64,
        alpha=64.0,                     # alpha=rank → scaling=1.0
        dropout=0.05,
        target_modules=['attn.qkv', 'mlp.fc1', 'mlp.fc2'],
    )
)

# -------------------------------------------------------------------------
# 2. Learning Rate: 2e-4 for batch_size=8
# -------------------------------------------------------------------------
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
# 3. Batch size 8 (fits in 46GB A40)
# -------------------------------------------------------------------------
train_dataloader = dict(batch_size=8)

# -------------------------------------------------------------------------
# 4. Work directory
# -------------------------------------------------------------------------
work_dir = './work_dirs/stage1_v2_lora_strong'
