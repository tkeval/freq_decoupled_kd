# -------------------------------------------------------------------------
# Sensitivity Sweep: freq_cutoff = 0.75
# -------------------------------------------------------------------------
# Identical to stage1_freq_decoupled.py, except freq_cutoff=0.75.
# Low-freq band is now the center 75% of each spatial dimension
# (vs. the default 50%). Larger low-freq region → more content gets
# strong MSE supervision, less is left for the relaxed high-freq band.
# Risk: low-freq band may start including modality-specific texture.
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_freq_decoupled_freq_cutoff_0.75.py 4 \
#       --work-dir ./work_dirs/stage1/fft/stage1_freq_decoupled_freq_cutoff_0.75
# bash ./tools/dist_train.sh configs/kaist/distillation/stage1_freq_decoupled_freq_cutoff_0.75.py 4 --work-dir work-dirs/stage1/fft/stage1_freq_decoupled_freq_cutoff_0.75 
# -------------------------------------------------------------------------
_base_ = ['stage1_freq_decoupled.py']

model = dict(
    freq_cutoff=0.75,       # larger low-freq region (vs. 0.5 default)
    low_freq_weight=1.0,    # unchanged from main method
    high_freq_weight=0.1,   # unchanged from main method
)

work_dir = './work_dirs/stage1/fft/stage1_freq_decoupled_freq_cutoff_0.75'
