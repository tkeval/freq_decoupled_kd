#!/bin/bash
# -------------------------------------------------------------------------
# Run all LoRA merge scale ablation experiments sequentially.
#
# Each run uses 4 GPUs and trains a Stage 2 DINO detector with a different
# lora_merge_scaling value. Logs are tee'd to per-run log files for later
# inspection. Failures do not abort the remaining runs.
#
# Usage:
#   bash configs/kaist/distillation/lora_alpha_merge_hyperparameter/run_all.sh
# -------------------------------------------------------------------------
set -u  # error on unset variables

CONFIG_DIR="configs/kaist/distillation/lora_alpha_merge_hyperparameter"
WORK_ROOT="./work_dirs/stage2/fft"
NUM_GPUS=4

# (config_name, work_dir_suffix) pairs
SCALES=("0.25" "0.75" "1.0")

mkdir -p "${WORK_ROOT}"
SUMMARY_LOG="${WORK_ROOT}/merge_ablation_summary.log"
echo "=== Merge scale ablation started: $(date) ===" | tee -a "${SUMMARY_LOG}"

for SCALE in "${SCALES[@]}"; do
    CONFIG="${CONFIG_DIR}/stage2_freq_decoupled_det_merge_${SCALE}.py"
    WORK_DIR="${WORK_ROOT}/stage2_freq_decoupled_merge_${SCALE}"
    LOG_FILE="${WORK_DIR}/run.log"

    mkdir -p "${WORK_DIR}"

    echo "" | tee -a "${SUMMARY_LOG}"
    echo "------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"
    echo "[$(date)] Starting α = ${SCALE}" | tee -a "${SUMMARY_LOG}"
    echo "  Config:   ${CONFIG}" | tee -a "${SUMMARY_LOG}"
    echo "  Work dir: ${WORK_DIR}" | tee -a "${SUMMARY_LOG}"
    echo "------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"

    if [[ ! -f "${CONFIG}" ]]; then
        echo "  ERROR: config not found, skipping" | tee -a "${SUMMARY_LOG}"
        continue
    fi

    START_TS=$(date +%s)

    bash ./tools/dist_train.sh "${CONFIG}" "${NUM_GPUS}" \
        --work-dir "${WORK_DIR}" 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[$(date)] α = ${SCALE} finished OK (elapsed ${ELAPSED}s)" | tee -a "${SUMMARY_LOG}"
    else
        echo "[$(date)] α = ${SCALE} FAILED (exit ${EXIT_CODE}, elapsed ${ELAPSED}s)" | tee -a "${SUMMARY_LOG}"
    fi
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "=== Merge scale ablation finished: $(date) ===" | tee -a "${SUMMARY_LOG}"
echo "Summary log: ${SUMMARY_LOG}"
