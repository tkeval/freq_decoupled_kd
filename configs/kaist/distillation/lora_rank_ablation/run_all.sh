#!/bin/bash
# -------------------------------------------------------------------------
# Run LoRA rank ablation experiments sequentially.
#
# Each run uses 4 GPUs and trains a Stage 1 FreqKD distiller with a
# different LoRA rank. Logs are tee'd to per-run log files for later
# inspection. Failures do not abort the remaining runs.
#
#   r=16:           resumed from existing checkpoint (--resume)
#   r=128:          fresh start
#   full fine-tune: fresh start (lora_cfg=None, 307M trainable backbone params)
#
# Usage:
#   bash configs/kaist/distillation/lora_rank_ablation/run_all.sh
# -------------------------------------------------------------------------
set -u

CONFIG_DIR="configs/kaist/distillation/lora_rank_ablation"
WORK_ROOT="./work_dirs/stage1/fft"
NUM_GPUS=4

mkdir -p "${WORK_ROOT}"
SUMMARY_LOG="${WORK_ROOT}/lora_rank_ablation_summary.log"
echo "=== LoRA rank ablation started: $(date) ===" | tee -a "${SUMMARY_LOG}"

run_one () {
    local SCALE_LABEL=$1     # e.g. "r=16"
    local CONFIG=$2
    local WORK_DIR=$3
    local EXTRA_FLAGS=$4     # e.g. "--resume" or ""

    local LOG_FILE="${WORK_DIR}/run.log"
    mkdir -p "${WORK_DIR}"

    echo "" | tee -a "${SUMMARY_LOG}"
    echo "------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"
    echo "[$(date)] Starting ${SCALE_LABEL}" | tee -a "${SUMMARY_LOG}"
    echo "  Config:   ${CONFIG}" | tee -a "${SUMMARY_LOG}"
    echo "  Work dir: ${WORK_DIR}" | tee -a "${SUMMARY_LOG}"
    echo "  Extra:    ${EXTRA_FLAGS:-(none)}" | tee -a "${SUMMARY_LOG}"
    echo "------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"

    if [[ ! -f "${CONFIG}" ]]; then
        echo "  ERROR: config not found, skipping" | tee -a "${SUMMARY_LOG}"
        return
    fi

    local START_TS=$(date +%s)

    bash ./tools/dist_train.sh "${CONFIG}" "${NUM_GPUS}" \
        --work-dir "${WORK_DIR}" ${EXTRA_FLAGS} 2>&1 | tee "${LOG_FILE}"
    local EXIT_CODE=${PIPESTATUS[0]}

    local END_TS=$(date +%s)
    local ELAPSED=$((END_TS - START_TS))

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[$(date)] ${SCALE_LABEL} finished OK (elapsed ${ELAPSED}s)" | tee -a "${SUMMARY_LOG}"
    else
        echo "[$(date)] ${SCALE_LABEL} FAILED (exit ${EXIT_CODE}, elapsed ${ELAPSED}s)" | tee -a "${SUMMARY_LOG}"
    fi
}

# --- r = 16 (resume) -----------------------------------------------------
run_one \
    "r=16 (resume)" \
    "${CONFIG_DIR}/stage1_freq_decoupled_rank_16.py" \
    "${WORK_ROOT}/stage1_freq_decoupled_rank_16" \
    "--resume"

# --- r = 128 (fresh) -----------------------------------------------------
run_one \
    "r=128" \
    "${CONFIG_DIR}/stage1_freq_decoupled_rank_128.py" \
    "${WORK_ROOT}/stage1_freq_decoupled_rank_128" \
    ""

# --- full fine-tune (no LoRA) --------------------------------------------
run_one \
    "full fine-tune" \
    "${CONFIG_DIR}/stage1_freq_decoupled_full_finetune.py" \
    "${WORK_ROOT}/stage1_freq_decoupled_full_finetune" \
    ""

echo "" | tee -a "${SUMMARY_LOG}"
echo "=== LoRA rank ablation finished: $(date) ===" | tee -a "${SUMMARY_LOG}"
echo "Summary log: ${SUMMARY_LOG}"
