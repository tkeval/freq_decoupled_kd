# -------------------------------------------------------------------------
# Baseline: Faster R-CNN with ResNet-50 + FPN on KAIST IR
# -------------------------------------------------------------------------
# Standard IR-only detection (no KD). This is the baseline to compare
# against the frequency-decoupled KD approach on CNN backbones.
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/frcnn/faster_rcnn_r50_fpn_12e_kaist_ir.py 4 \
#       --work-dir ./work_dirs/frcnn/baseline_ir
# -------------------------------------------------------------------------
_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
]

# -------------------------------------------------------------------------
# 1. Model: Faster R-CNN, 1 class (person)
# -------------------------------------------------------------------------
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

# -------------------------------------------------------------------------
# 2. Dataset: KAIST IR (single-image pipeline, no pairing needed)
# -------------------------------------------------------------------------
dataset_type = 'KAISTDataset'
data_root = './data/kaist-rgbt/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
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
        filter_cfg=dict(filter_empty_gt=False),
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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instancesonly_filtered_all-20_val.json',
    metric='bbox',
    classwise=True,
    format_only=False)

test_evaluator = val_evaluator

# -------------------------------------------------------------------------
# 3. Training Schedule: 12 epochs, SGD with lower LR for small dataset
# -------------------------------------------------------------------------
# KAIST (~9.5K images) is ~12x smaller than COCO — lr=0.02 overfits by
# epoch 2. Use lr=0.005 with cosine annealing for smooth convergence.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=12,
        by_epoch=True,
        eta_min=1e-5)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# -------------------------------------------------------------------------
# 4. Checkpointing
# -------------------------------------------------------------------------
work_dir = './work_dirs/frcnn/baseline_ir_v2'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP_50',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50))
