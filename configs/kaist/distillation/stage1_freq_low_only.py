# -------------------------------------------------------------------------
# Ablation: Low-Frequency Only KD
# -------------------------------------------------------------------------
# Identical to stage1_freq_decoupled.py, except high_freq_weight=0.0.
# No logMSE on high-freq — only strong MSE on low-freq (structural) content.
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_freq_low_only.py 4 \
#       --work-dir ./work_dirs/stage1/fft/stage1_freq_low_only
# -------------------------------------------------------------------------
_base_ = ['stage1_freq_decoupled.py']

model = dict(
    high_freq_weight=0.0,   # disable high-freq loss entirely
    # low_freq_weight stays at 1.0 (default)
)

work_dir = './work_dirs/stage1/fft/stage1_freq_low_only'
