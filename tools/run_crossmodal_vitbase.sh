#!/bin/bash
# -------------------------------------------------------------------------
# Full pipeline: Cross-Modal Feature Distillation → Detection Fine-tuning
#   Stage 1: RGB DINOv2-Large → IR ViT-Base (feature KD, 12 epochs)
#   Extract: Pull backbone weights from Stage 1 checkpoint
#   Stage 2: Fine-tune ViT-Base DINO detector on KAIST IR (12 epochs)
# -------------------------------------------------------------------------
set -e  # Exit on any error

WORK_DIR="./work_dirs"
STAGE1_DIR="${WORK_DIR}/stage1_crossmodal_vitbase"
STAGE2_DIR="${WORK_DIR}/stage2_vitbase_det"
NUM_GPUS=4

echo "=============================================="
echo "Stage 1: Cross-Modal Feature Distillation"
echo "=============================================="
bash ./tools/dist_train.sh \
    configs/kaist/distillation/stage1_crossmodal_vitbase.py ${NUM_GPUS} \
    --work-dir ${STAGE1_DIR}

echo ""
echo "=============================================="
echo "Extract: Backbone from Stage 1 checkpoint"
echo "=============================================="
# Find the last epoch checkpoint
STAGE1_CKPT=$(ls -t ${STAGE1_DIR}/epoch_*.pth 2>/dev/null | head -1)
if [ -z "$STAGE1_CKPT" ]; then
    echo "ERROR: No Stage 1 checkpoint found in ${STAGE1_DIR}"
    exit 1
fi
echo "Using checkpoint: ${STAGE1_CKPT}"

python tools/extract_stage1_backbone.py \
    ${STAGE1_CKPT} \
    ${STAGE1_DIR}/backbone_vitbase.pth

echo ""
echo "=============================================="
echo "Stage 2: Detection Fine-tuning"
echo "=============================================="
bash ./tools/dist_train.sh \
    configs/kaist/distillation/stage2_vitbase_det.py ${NUM_GPUS} \
    --work-dir ${STAGE2_DIR}

echo ""
echo "=============================================="
echo "Done! Check results in: ${STAGE2_DIR}"
echo "=============================================="
