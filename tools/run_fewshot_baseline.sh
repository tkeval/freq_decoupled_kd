#!/usr/bin/env bash
# ----------------------------------------------------------------
# Few-Shot Baseline: DINOv2-Base IR on KAIST with varying annotations
# Runs 10%, 25%, 50%, 70%, 100% subsets sequentially.
# ----------------------------------------------------------------

CONFIG="configs/kaist/dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py"
GPUS=4
WORK_DIR_BASE="./work_dirs/fewshot_baseline_vitlarge"
ANN_DIR="annotations/few_shot"

for SPLIT in 25p 50p 70p 100p; do
    ANN_FILE="${ANN_DIR}/instancesonly_filtered_all-02_train_subset_${SPLIT}.json"
    WORK_DIR="${WORK_DIR_BASE}/${SPLIT}"

    echo "============================================"
    echo "Running few-shot baseline: ${SPLIT}"
    echo "  ann_file: ${ANN_FILE}"
    echo "  work_dir: ${WORK_DIR}"
    echo "============================================"

    bash ./tools/dist_train.sh ${CONFIG} ${GPUS} \
        --work-dir ${WORK_DIR} \
        --cfg-options train_dataloader.dataset.ann_file="${ANN_FILE}"

    echo "Finished ${SPLIT}"
    echo ""
done

echo "All few-shot baseline runs complete."
