# -------------------------------------------------------------------------
# Merge Scale Ablation: lora_merge_scaling = 0.25
# -------------------------------------------------------------------------
# Identical to stage2_freq_decoupled_det.py, except merge α = 0.25.
# Only 25% of the Stage 1 LoRA adaptation is applied to the backbone
# before detection fine-tuning (more pretrained, less KD).
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/lora_alpha_merge_hyperparameter/stage2_freq_decoupled_det_merge_0.25.py 4 \
#       --work-dir ./work_dirs/stage2/fft/stage2_freq_decoupled_merge_0.25
# -------------------------------------------------------------------------
_base_ = ['../stage2_freq_decoupled_det.py']

model = dict(
    lora_merge_scaling=0.25,
)

work_dir = './work_dirs/stage2/fft/stage2_freq_decoupled_merge_0.25'
