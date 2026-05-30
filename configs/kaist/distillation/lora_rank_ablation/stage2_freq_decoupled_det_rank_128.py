# -------------------------------------------------------------------------
# Stage 2: Detection Fine-Tuning — LoRA Rank Ablation (rank = 128)
# -------------------------------------------------------------------------
# Consumes the Stage 1 backbone trained with LoRA rank=128.
# Everything else identical to stage2_freq_decoupled_det.py
# (lora_merge_scaling=0.5 inherited from base).
#
# The LoRA adapters in the checkpoint encode rank=128 in their B@A shapes,
# so the merge requires no rank-specific config — only the checkpoint path.
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/distillation/lora_rank_ablation/stage2_freq_decoupled_det_rank_128.py 4 \
#       --work-dir ./work_dirs/stage2/fft/stage2_freq_decoupled_rank_128
# -------------------------------------------------------------------------
_base_ = ['../stage2_freq_decoupled_det.py']

stage1_checkpoint = './work_dirs/stage1/fft/stage1_freq_decoupled_rank_128/epoch_12.pth'

model = dict(
    teacher_checkpoint=stage1_checkpoint,
)

work_dir = './work_dirs/stage2/fft/stage2_freq_decoupled_rank_128'
