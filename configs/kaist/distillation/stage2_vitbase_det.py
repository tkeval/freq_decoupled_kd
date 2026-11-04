# -------------------------------------------------------------------------
# Stage 2: Detection Fine-tuning with Cross-Modal Pretrained ViT-Base
# -------------------------------------------------------------------------
# Uses the ViT-Base backbone from Stage 1 cross-modal feature distillation
# as initialization. Otherwise identical to the vanilla ViT-Base baseline.
#
# Before running, extract the backbone from Stage 1:
#   python tools/extract_stage1_backbone.py \
#       ./work_dirs/stage1_crossmodal_vitbase/epoch_12.pth \
#       ./work_dirs/stage1_crossmodal_vitbase/backbone_vitbase.pth
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/distillation/stage2_vitbase_det.py 4 \
#       --work-dir ./work_dirs/stage2_vitbase_det
# -------------------------------------------------------------------------
_base_ = [
    '../../kaist/dinov2/dino-5scale_dinov2-base-p14-reg4_8xb2-12e_kaist.py'
]

# Load the distilled backbone from Stage 1
# UPDATE THIS PATH after running extract_stage1_backbone.py
load_from = './work_dirs/stage1_crossmodal_vitbase/backbone_vitbase.pth'
