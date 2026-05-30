# -------------------------------------------------------------------------
# Stage 2: Detection Fine-Tuning — High-Freq Only Ablation
# -------------------------------------------------------------------------
# Uses backbone from stage1_freq_high_only (logMSE on high-freq, no low-freq).
# Everything else identical to stage2_freq_decoupled_det.py.
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage2_freq_high_only_det.py 4 \
#       --work-dir ./work_dirs/stage2/fft/stage2_freq_high_only
# -------------------------------------------------------------------------
_base_ = ['stage2_freq_decoupled_det.py']

stage1_checkpoint = './work_dirs/stage1/fft/stage1_freq_high_only/epoch_12.pth'

model = dict(
    teacher_checkpoint=stage1_checkpoint,
)

work_dir = './work_dirs/stage2/fft/stage2_freq_high_only'
