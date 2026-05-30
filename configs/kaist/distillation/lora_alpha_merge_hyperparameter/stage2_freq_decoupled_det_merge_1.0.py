# -------------------------------------------------------------------------
# Merge Scale Ablation: lora_merge_scaling = 1.0
# -------------------------------------------------------------------------
# Identical to stage2_freq_decoupled_det.py, except merge α = 1.0.
# Full Stage 1 LoRA adaptation applied to the backbone before
# detection fine-tuning (max KD imprint).
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/lora_alpha_merge_hyperparameter/stage2_freq_decoupled_det_merge_1.0.py 4 \
#       --work-dir ./work_dirs/stage2/fft/stage2_freq_decoupled_merge_1.0
# -------------------------------------------------------------------------
_base_ = ['../stage2_freq_decoupled_det.py']

model = dict(
    lora_merge_scaling=1.0,
)

work_dir = './work_dirs/stage2/fft/stage2_freq_decoupled_merge_1.0'
