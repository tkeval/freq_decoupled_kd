# -------------------------------------------------------------------------
# Merge Scale Ablation: lora_merge_scaling = 0.75
# -------------------------------------------------------------------------
# Identical to stage2_freq_decoupled_det.py, except merge α = 0.75.
# 75% of the Stage 1 LoRA adaptation is applied to the backbone
# before detection fine-tuning (heavier KD imprint than default 0.5).
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/lora_alpha_merge_hyperparameter/stage2_freq_decoupled_det_merge_0.75.py 4 \
#       --work-dir ./work_dirs/stage2/fft/stage2_freq_decoupled_merge_0.75
# -------------------------------------------------------------------------
_base_ = ['../stage2_freq_decoupled_det.py']

model = dict(
    lora_merge_scaling=0.75,
)

work_dir = './work_dirs/stage2/fft/stage2_freq_decoupled_merge_0.75'
