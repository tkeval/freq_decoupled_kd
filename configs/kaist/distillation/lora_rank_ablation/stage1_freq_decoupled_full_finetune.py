# -------------------------------------------------------------------------
# LoRA Rank Ablation: Full Backbone Fine-Tune (no LoRA)
# -------------------------------------------------------------------------
# Identical to stage1_freq_decoupled.py, except lora_cfg=None.
# Setting lora_cfg=None disables LoRA injection entirely; the FreqDecoupled
# distiller skips the freeze + adapter-injection step, so the entire student
# backbone trains end-to-end via the FFT KD loss.
#
# Note: With 307M trainable backbone params at lr=2e-4, this can be
# unstable. If it diverges, consider lowering lr to ~5e-5 or adding a
# backbone lr_mult — but for a fair ablation we keep all other hyperparams
# unchanged.
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/distillation/lora_rank_ablation/stage1_freq_decoupled_full_finetune.py 4 \
#       --work-dir ./work_dirs/stage1/fft/stage1_freq_decoupled_full_finetune
# -------------------------------------------------------------------------
_base_ = ['../stage1_freq_decoupled.py']

model = dict(
    lora_cfg=None,   # disable LoRA → full backbone fine-tune
)

work_dir = './work_dirs/stage1/fft/stage1_freq_decoupled_full_finetune'
