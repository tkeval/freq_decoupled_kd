# -------------------------------------------------------------------------
# Stage 1 v2 + Strong LoRA + ConcatDataset
# -------------------------------------------------------------------------
# Combines:
#   - stage1_distill_v2_lora_strong: Strong LoRA (rank=64, QKV+MLP, ~7% params)
#   - ConcatDataset: KAIST (~25k) + FLIR (~4.1k) + MFNet (~784) ≈ 30k pairs
#   - CLIP ViT-L/14 teacher (1024-dim, from v2 base) — stronger than ViT-B/32
#
# More diverse training data should improve cross-modal feature alignment,
# since the student sees IR/RGB pairs from multiple domains (surveillance,
# driving, segmentation scenes).
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_distill_v2_lora_strong_concat.py 4 \
#       --work-dir ./work_dirs/stage1_v2_lora_strong_concat
# -------------------------------------------------------------------------
_base_ = [
    './stage1_distill_v2_lora_strong.py'
]

# -------------------------------------------------------------------------
# 1. Dataset: ConcatDataset (KAIST + FLIR + MFNet)
# -------------------------------------------------------------------------
# Stage 1 only needs paired IR/RGB images (no detection annotations used).
# KAIST: ~25k paired images  |  FLIR Aligned: ~4.1k  |  MFNet: ~784

kaist_data_root = './data/kaist-rgbt/'
flir_data_root = './data/flir_aligned/'
mfnet_data_root = './data/mfnet_ir_seg_dataset/'

train_pipeline = [
    dict(type='LoadPairedImagesFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PairedResize', scale=(640, 512), keep_ratio=True),
    dict(type='PairedRandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'img2_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'img_rgb'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='KAISTDataset',
                data_root=kaist_data_root,
                ann_file='annotations/instancesonly_filtered_all-02_train.json',
                data_prefix=dict(img='images/'),
                filter_cfg=dict(filter_empty_gt=False),
                pipeline=train_pipeline,
                metainfo=dict(classes=('person',)),
                include_empty_images=True),
            dict(
                type='KAISTDataset',
                data_root=flir_data_root,
                ann_file='flir_aligned_train.json',
                data_prefix=dict(img=''),
                filter_cfg=dict(filter_empty_gt=False),
                pipeline=train_pipeline,
                metainfo=dict(classes=('person', 'car', 'bicycle', 'dog')),
                include_empty_images=True),
            dict(
                type='KAISTDataset',
                data_root=mfnet_data_root,
                ann_file='annotations/mfnet_paired_train.json',
                data_prefix=dict(img=''),
                filter_cfg=dict(filter_empty_gt=False),
                pipeline=train_pipeline,
                metainfo=dict(classes=()),
                include_empty_images=True),
        ]))

# -------------------------------------------------------------------------
# 2. Work directory
# -------------------------------------------------------------------------
work_dir = './work_dirs/stage1_v2_lora_strong_concat'
