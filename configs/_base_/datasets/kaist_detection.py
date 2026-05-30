 # Detection Config for single detection
# Setup common/shared configuration parameters

# Dataset settings
dataset_type = 'KAISTDataset'
# data_root =  r"F:/Datasets/kaist-rgbt/"
# orig:
data_root = './data/kaist-rgbt/'
# data_root = './data/kaist-rgbt/'

backend_args = None

train_pipeline = [
    dict(type='LoadPairedImagesFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True, keys=['img', 'img_rgb']),
    dict(type='RandomFlip', prob=0.5, keys=['img', 'img_rgb']),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'img2_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'img_rgb'))
]
# In this version, we revert to a standard `LoadImageFromFile` pipeline
# because the distiller now handles the paired image loading internally.
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'img2_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instancesonly_filtered_all-02_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False), # MIGHT NEED TO BE CHANGE TO FALSE
        pipeline=train_pipeline,
        metainfo=dict(classes=('person',))))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instancesonly_filtered_all-20_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('person',))))

test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = dict(
    type='CocoMetric',
    # orig: ann_file='./data/kaist-rgbt/instancesonly_filtered_all-20_val.json',
    ann_file= data_root + 'annotations/instancesonly_filtered_all-20_val.json',
    metric='bbox',
    classwise=True,  # This will show per-class performance
    format_only=False)

test_evaluator = val_evaluator