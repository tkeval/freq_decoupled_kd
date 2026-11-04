#!/bin/bash
# -------------------------------------------------------------------------
# Run Stage 2 detection fine-tuning for the LoRA rank ablation.
#
# Each run consumes the matching Stage 1 checkpoint (rank 16/128/full-FT)
# and trains a DINO detector. Run AFTER the Stage 1 sweep (run_all.sh) has
# produced the epoch_12.pth checkpoints.
#
# Failures do not abort the remaining runs.
#
# Usage:
#   bash configs/kaist/distillation/lora_rank_ablation/run_all_stage2.sh
# -------------------------------------------------------------------------
set -u

CONFIG_DIR="configs/kaist/distillation/lora_rank_ablation"
WORK_ROOT="./work_dirs/stage2/fft"
STAGE1_ROOT="./work_dirs/stage1/fft"
NUM_GPUS=4

mkdir -p "${WORK_ROOT}"
SUMMARY_LOG="${WORK_ROOT}/lora_rank_ablation_stage2_summary.log"
echo "=== LoRA rank ablation (Stage 2) started: $(date) ===" | tee -a "${SUMMARY_LOG}"

run_one () {
    local LABEL=$1
    local CONFIG=$2
    local WORK_DIR=$3
    local STAGE1_CKPT=$4

    local LOG_FILE="${WORK_DIR}/run.log"
    mkdir -p "${WORK_DIR}"

    echo "" | tee -a "${SUMMARY_LOG}"
    echo "------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"
    echo "[$(date)] Starting ${LABEL}" | tee -a "${SUMMARY_LOG}"
    echo "  Config:   ${CONFIG}" | tee -a "${SUMMARY_LOG}"
    echo "  Work dir: ${WORK_DIR}" | tee -a "${SUMMARY_LOG}"
    echo "------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"

    if [[ ! -f "${CONFIG}" ]]; then
        echo "  ERROR: config not found, skipping" | tee -a "${SUMMARY_LOG}"
        return
    fi
    if [[ ! -f "${STAGE1_CKPT}" ]]; then
        echo "  ERROR: Stage 1 checkpoint not found (${STAGE1_CKPT}), skipping" | tee -a "${SUMMARY_LOG}"
        echo "         (run the Stage 1 sweep first: run_all.sh)" | tee -a "${SUMMARY_LOG}"
        return
    fi

    local START_TS=$(date +%s)

    bash ./tools/dist_train.sh "${CONFIG}" "${NUM_GPUS}" \
        --work-dir "${WORK_DIR}" 2>&1 | tee "${LOG_FILE}"
    local EXIT_CODE=${PIPESTATUS[0]}

    local END_TS=$(date +%s)
    local ELAPSED=$((END_TS - START_TS))

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[$(date)] ${LABEL} finished OK (elapsed ${ELAPSED}s)" | tee -a "${SUMMARY_LOG}"
    else
        echo "[$(date)] ${LABEL} FAILED (exit ${EXIT_CODE}, elapsed ${ELAPSED}s)" | tee -a "${SUMMARY_LOG}"
    fi
}

# --- r = 16 --------------------------------------------------------------
run_one \
    "stage2 r=16" \
    "${CONFIG_DIR}/stage2_freq_decoupled_det_rank_16.py" \
    "${WORK_ROOT}/stage2_freq_decoupled_rank_16" \
    "${STAGE1_ROOT}/stage1_freq_decoupled_rank_16/epoch_12.pth"

# --- r = 128 -------------------------------------------------------------
run_one \
    "stage2 r=128" \
    "${CONFIG_DIR}/stage2_freq_decoupled_det_rank_128.py" \
    "${WORK_ROOT}/stage2_freq_decoupled_rank_128" \
    "${STAGE1_ROOT}/stage1_freq_decoupled_rank_128/epoch_12.pth"

# --- full fine-tune ------------------------------------------------------
run_one \
    "stage2 full fine-tune" \
    "${CONFIG_DIR}/stage2_freq_decoupled_det_full_finetune.py" \
    "${WORK_ROOT}/stage2_freq_decoupled_full_finetune" \
    "${STAGE1_ROOT}/stage1_freq_decoupled_full_finetune/epoch_12.pth"

echo "" | tee -a "${SUMMARY_LOG}"
echo "=== LoRA rank ablation (Stage 2) finished: $(date) ===" | tee -a "${SUMMARY_LOG}"
echo "Summary log: ${SUMMARY_LOG}"
