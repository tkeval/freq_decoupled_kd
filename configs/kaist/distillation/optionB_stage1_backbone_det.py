# -------------------------------------------------------------------------
# Option B: Detection fine-tuning with Stage 1 backbone initialization
# -------------------------------------------------------------------------
# This is identical to the baseline DINOv2 detector, except the backbone
# is initialized from Stage 1 weights instead of DINOv2 pretrained.
# No distillation loss -- pure detection training.
#
# Purpose: Diagnostic experiment to test if the Stage 1 backbone is
# better than vanilla DINOv2 for IR detection.
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/optionB_stage1_backbone_det.py 4 \
#       --work-dir ./work_dirs/optionB_stage1_backbone
# -------------------------------------------------------------------------
_base_ = [
    '../dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py'
]

# -------------------------------------------------------------------------
# Stage 1 checkpoint path (update after Stage 1 finishes)
# -------------------------------------------------------------------------
stage1_checkpoint = './work_dirs/stage1_distill_bs4/epoch_12.pth'

# -------------------------------------------------------------------------
# Custom hook to load Stage 1 backbone weights into the detector
# -------------------------------------------------------------------------
# We use custom_hooks with a checkpoint loader to initialize only the
# backbone from Stage 1. The neck/encoder/decoder/head use default init.
#
# The Stage 1 checkpoint has keys like "student.backbone.xxx".
# The detector expects keys like "backbone.xxx".
# We handle this mapping in a custom init hook below.
#
# Alternative approach: set load_from and handle key mapping.
# For simplicity, we override the init_cfg of the backbone.

# -------------------------------------------------------------------------
# Load Stage 1 backbone weights
# -------------------------------------------------------------------------
# First, extract the backbone with:
#   python tools/extract_stage1_backbone.py \
#       ./work_dirs/stage1_distill/epoch_12.pth \
#       ./work_dirs/stage1_distill/stage1_backbone.pth
#
# Then this load_from will initialize the backbone from Stage 1.
# The neck/encoder/decoder/head will use default init.
# MMEngine loads with strict=False by default, so missing keys (neck, head)
# are silently skipped.
load_from = './work_dirs/stage1_distill_bs4/stage1_backbone.pth'

# Override train pipeline (fix the keys issue from base config)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline))

# Work directory
work_dir = './work_dirs/optionB_stage1_backbone'
