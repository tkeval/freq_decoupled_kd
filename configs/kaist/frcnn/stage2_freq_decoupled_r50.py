# -------------------------------------------------------------------------
# Stage 2: Faster R-CNN Detection with Stage 1 Adapted Backbone
# -------------------------------------------------------------------------
# Loads ResNet-50 backbone weights from Stage 1 (freq-decoupled KD) into
# a standard Faster R-CNN detector. No teacher or KD during Stage 2 —
# just standard detection training on IR.
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/frcnn/stage2_freq_decoupled_r50.py 4 \
#       --work-dir ./work_dirs/frcnn/stage2_freq_decoupled_r50
# -------------------------------------------------------------------------
_base_ = [
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmdet.models.distillers'],
    allow_failed_imports=False)

# -------------------------------------------------------------------------
# 1. Stage 1 checkpoint path (update after Stage 1 training)
# -------------------------------------------------------------------------
stage1_checkpoint = './work_dirs/frcnn/stage1_freq_decoupled_r50/epoch_12.pth'
# -------------------------------------------------------------------------
# 2. Model: Stage2FasterRCNN wrapper
# -------------------------------------------------------------------------
model = dict(
    type='Stage2FasterRCNN',
    stage1_checkpoint=stage1_checkpoint,
    backbone_merge_scale=0.5,  # Blend: 50% pretrained + 50% Stage 1 (like LoRA merge)
    detector_cfg=dict(
        type='FasterRCNN',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32),
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            # No init_cfg — pretrained loaded explicitly in _load_stage1_backbone
            # and merged with Stage 1 weights. _is_init=True prevents overwrite.
            init_cfg=None),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))))

# -------------------------------------------------------------------------
# 3. Dataset: KAIST IR (single-image, standard detection)
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
# 4. Training Schedule: SGD, 12 epochs (same as baseline)
# -------------------------------------------------------------------------
# Inherits from schedule_1x: SGD lr=0.02, MultiStepLR [8,11]

# -------------------------------------------------------------------------
# 5. Checkpointing
# -------------------------------------------------------------------------
work_dir = './work_dirs/frcnn/stage2_freq_decoupled_r50'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_50',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50))
