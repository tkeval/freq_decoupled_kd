# -------------------------------------------------------------------------
# Stage 2: Detection Fine-Tuning — Low-Freq Only Ablation
# -------------------------------------------------------------------------
# Uses backbone from stage1_freq_low_only (MSE on low-freq only, no high-freq).
# Everything else identical to stage2_freq_decoupled_det.py.
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage2_freq_cutoff_0.1.py 4 \
#       --work-dir ./work_dirs/stage2/fft/stage2_freq_cutoff_0.1
# -------------------------------------------------------------------------
_base_ = ['stage2_freq_decoupled_det.py']

stage1_checkpoint = './work_dirs/stage1/fft/stage1_freq_decoupled_freq_cutoff_0.1/epoch_12.pth'

model = dict(
    teacher_checkpoint=stage1_checkpoint,
)

work_dir = './work_dirs/stage2/fft/stage2_freq_cutoff_0.1'
