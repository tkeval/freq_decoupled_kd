# MFNet paired (IR + RGB) config for cross-modal distillation.
#
# MFNet stores paired data as 4-channel RGBA PNGs (RGB in ch 0-2, thermal
# in ch 3).  The conversion script (tools/dataset_converters/mfnet_to_paired_coco.py)
# splits these into separate rgb/ and ir/ directories and produces COCO JSON
# with file_name / file_name2 pairs, reusing KAISTDataset.
#
# NOTE: MFNet is a semantic segmentation dataset — the COCO JSONs have empty
# annotations.  This config is only useful for Stage 1 feature distillation
# (which doesn't need bounding-box labels).

dataset_type = 'KAISTDataset'
data_root = 'data/mfnet_ir_seg_dataset/'

backend_args = None

train_pipeline = [
    dict(type='LoadPairedImagesFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PairedResize', scale=(640, 512), keep_ratio=True),
    dict(type='PairedRandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'img2_path', 'ori_shape', 'img_shape',
        'scale_factor', 'flip', 'flip_direction', 'img_rgb'))
]

mfnet_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/mfnet_paired_train.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    metainfo=dict(classes=()),
    include_empty_images=True)

mfnet_all_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/mfnet_paired_all.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    metainfo=dict(classes=()),
    include_empty_images=True)
