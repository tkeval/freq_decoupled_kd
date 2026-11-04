# -------------------------------------------------------------------------
# LoRA Rank Ablation: rank = 128
# -------------------------------------------------------------------------
# Identical to stage1_freq_decoupled.py, except LoRA rank=128 (vs. default 64).
# Higher-capacity adapters — more trainable params, may over-fit.
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/distillation/lora_rank_ablation/stage1_freq_decoupled_rank_128.py 4 \
#       --work-dir ./work_dirs/stage1/fft/stage1_freq_decoupled_rank_128
# -------------------------------------------------------------------------
_base_ = ['../stage1_freq_decoupled.py']

model = dict(
    lora_cfg=dict(
        rank=128,
        alpha=128.0,                                          # alpha = rank (scaling 1.0)
        dropout=0.05,
        target_modules=['attn.qkv', 'mlp.fc1', 'mlp.fc2'],
    ),
)

work_dir = './work_dirs/stage1/fft/stage1_freq_decoupled_rank_128'
