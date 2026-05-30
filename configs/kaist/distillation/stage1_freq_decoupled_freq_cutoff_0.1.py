# -------------------------------------------------------------------------
# Sensitivity Sweep: freq_cutoff = 0.1
# -------------------------------------------------------------------------
# Identical to stage1_freq_decoupled.py, except freq_cutoff=0.1.
# Low-freq band is now the center 10% of each spatial dimension
# (vs. the default 50%). Smaller low-freq region → stricter "structural"
# alignment, more content pushed into the relaxed high-freq band.
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_freq_decoupled_freq_cutoff_0.1.py 4 \
#       --work-dir ./work_dirs/stage1/fft/stage1_freq_decoupled_freq_cutoff_0.1
# bash ./tools/dist_train.sh configs/kaist/distillation/stage1_freq_decoupled_freq_cutoff_0.1.py 1 --work-dir work-dirs/stage1/fft/stage1_freq_decoupled_freq_cutoff_0.1
# -------------------------------------------------------------------------
_base_ = ['stage1_freq_decoupled.py']

model = dict(
    freq_cutoff=0.1,        # smaller low-freq region (vs. 0.5 default)
    low_freq_weight=1.0,    # unchanged from main method
    high_freq_weight=0.1,   # unchanged from main method
)

work_dir = 'work-dirs/stage1/fft/stage1_freq_decoupled_freq_cutoff_0.1'
