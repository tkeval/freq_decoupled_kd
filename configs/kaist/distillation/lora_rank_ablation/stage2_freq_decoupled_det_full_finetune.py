# -------------------------------------------------------------------------
# Stage 2: Detection Fine-Tuning — Full Fine-Tune Ablation (no LoRA)
# -------------------------------------------------------------------------
# Consumes the Stage 1 backbone trained WITHOUT LoRA (full fine-tune).
#
# IMPORTANT: this checkpoint has no LoRA adapters — merge_lora_state_dict()
# finds no lora_A/lora_B keys and returns the backbone weights unchanged,
# so the fully-trained backbone loads directly. The inherited
# lora_merge_scaling value is therefore a no-op for this run (there is
# nothing to scale), and the backbone is effectively used at full strength.
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/distillation/lora_rank_ablation/stage2_freq_decoupled_det_full_finetune.py 4 \
#       --work-dir ./work_dirs/stage2/fft/stage2_freq_decoupled_full_finetune
# -------------------------------------------------------------------------
_base_ = ['../stage2_freq_decoupled_det.py']

stage1_checkpoint = './work_dirs/stage1/fft/stage1_freq_decoupled_full_finetune/epoch_12.pth'

model = dict(
    teacher_checkpoint=stage1_checkpoint,
)

work_dir = './work_dirs/stage2/fft/stage2_freq_decoupled_full_finetune'
