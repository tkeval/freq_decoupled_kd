# FLIR Aligned paired (IR + RGB) detection / distillation config.
# Uses the same pipeline as KAIST paired detection.

dataset_type = 'KAISTDataset'
data_root = 'data/flir_aligned/'

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

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'img2_path', 'ori_shape',
                   'img_shape', 'scale_factor'))
]

flir_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/flir_aligned_train.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    metainfo=dict(classes=('person', 'car', 'bicycle', 'dog')))

flir_val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/flir_aligned_val.json',
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=test_pipeline,
    metainfo=dict(classes=('person', 'car', 'bicycle', 'dog')))
