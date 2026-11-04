#!/usr/bin/env bash
# ----------------------------------------------------------------
# Few-Shot Stage 2: Distilled IR backbone (LoRA merged) on KAIST
# Runs 10%, 25%, 50%, 70%, 100% subsets sequentially.
# Uses scaling=0.1 (best detection result from full-data experiments).
# ----------------------------------------------------------------

CONFIG="configs/kaist/distillation/stage2_guided_det.py"
GPUS=4
WORK_DIR_BASE="./work_dirs/fewshot_stage2_scale01"
ANN_DIR="annotations/few_shot"

for SPLIT in 10p 25p 50p 70p 100p; do
    ANN_FILE="${ANN_DIR}/instancesonly_filtered_all-02_train_subset_${SPLIT}.json"
    WORK_DIR="${WORK_DIR_BASE}/${SPLIT}"

    echo "============================================"
    echo "Running few-shot Stage 2 (distilled): ${SPLIT}"
    echo "  ann_file: ${ANN_FILE}"
    echo "  work_dir: ${WORK_DIR}"
    echo "============================================"

    bash ./tools/dist_train.sh ${CONFIG} ${GPUS} \
        --work-dir ${WORK_DIR} \
        --cfg-options train_dataloader.dataset.ann_file="${ANN_FILE}"

    echo "Finished ${SPLIT}"
    echo ""
done

echo "All few-shot Stage 2 runs complete."
