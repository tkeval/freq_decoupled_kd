# -------------------------------------------------------------------------
# Ablation: High-Frequency Only KD
# -------------------------------------------------------------------------
# Identical to stage1_freq_decoupled.py, except low_freq_weight=0.0.
# No MSE on low-freq — only relaxed logMSE on high-freq (texture) content.
#
# Expected: No gain or slight regression vs. baseline, since high-freq
# features are modality-specific and don't transfer well RGB → IR.
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_freq_high_only.py 4 \
#       --work-dir ./work_dirs/stage1/fft/stage1_freq_high_only
# -------------------------------------------------------------------------
_base_ = ['stage1_freq_decoupled.py']

model = dict(
    low_freq_weight=0.0,    # disable low-freq loss entirely
    high_freq_weight=1.0,   # full weight on high-freq (not 0.1 — it's now the only signal)
)

work_dir = './work_dirs/stage1/fft/stage1_freq_high_only'
