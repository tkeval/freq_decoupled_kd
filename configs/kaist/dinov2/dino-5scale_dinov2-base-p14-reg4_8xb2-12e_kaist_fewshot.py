# ------------------------------------------------------------
# DINOv2-BASE Few-Shot Baseline on KAIST IR
# Same as the full baseline but uses few-shot annotation subsets.
# Override ann_file via --cfg-options for each split.
# ------------------------------------------------------------
_base_ = [
    './dino-5scale_dinov2-base-p14-reg4_8xb2-12e_kaist.py'
]

# Placeholder — override via --cfg-options at launch time:
#   --cfg-options train_dataloader.dataset.ann_file=annotations/few_shot/instancesonly_filtered_all-02_train_subset_10p.json
