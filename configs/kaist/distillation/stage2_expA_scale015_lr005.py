# -------------------------------------------------------------------------
# Experiment A: Partial LoRA Merge (scaling=0.15) + Lower Backbone LR
# -------------------------------------------------------------------------
# Based on stage2_guided_det.py with two changes:
#   1. lora_merge_scaling: 0.1 → 0.15 (slightly more Stage 1 knowledge)
#   2. backbone lr_mult: 0.1 → 0.05 (preserve Stage 1 features longer)
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage2_expA_scale015_lr005.py 4 \
#       --work-dir ./work_dirs/stage2_expA_scale015_lr005
# -------------------------------------------------------------------------
_base_ = [
    './stage2_guided_det.py'
]

# Override LoRA merge scaling: 0.1 → 0.15
model = dict(lora_merge_scaling=0.15)

# Override backbone lr_mult: 0.1 → 0.05
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.05),       # Was 0.1, now 0.05
            'teacher_backbone': dict(lr_mult=0.0), # Frozen (unchanged)
            'sampling_offsets': dict(lr_mult=0.1),  # Unchanged
            'reference_points': dict(lr_mult=0.1),  # Unchanged
        }))

work_dir = './work_dirs/stage2_expA_scale015_lr005'
