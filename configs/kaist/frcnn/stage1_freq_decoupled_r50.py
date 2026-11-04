# -------------------------------------------------------------------------
# Stage 1: Frequency-Decoupled Cross-Modal KD for ResNet-50
# -------------------------------------------------------------------------
# RGB teacher (ImageNet ResNet-50) → IR student (ResNet-50) backbone KD.
# Same FFT decomposition as the ViT version but adapted for CNN:
#   - No LoRA needed (ResNet is small enough to fine-tune directly)
#   - 4 layer pairs matching ResNet stages (vs 5 for ViT)
#   - Same architecture teacher/student → no projectors needed
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/frcnn/stage1_freq_decoupled_r50.py 4 \
#       --work-dir ./work_dirs/frcnn/stage1_freq_decoupled_r50
# -------------------------------------------------------------------------

custom_imports = dict(
    imports=['mmdet.models.distillers'],
    allow_failed_imports=False)

# -------------------------------------------------------------------------
# 1. Teacher: RGB ResNet-50 backbone (ImageNet pretrained, frozen)
# -------------------------------------------------------------------------
teacher_backbone_cfg = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=4,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

# -------------------------------------------------------------------------
# 2. Student: Faster R-CNN with ResNet-50 + FPN (full detector, but only
#    backbone is trained via frequency loss in Stage 1)
# -------------------------------------------------------------------------
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32)

student_cfg = dict(
    type='FasterRCNN',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # Train all stages
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
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
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
            max_per_img=100)))

# -------------------------------------------------------------------------
# 3. Model: FreqDecoupledDistillerCNN
# -------------------------------------------------------------------------
model = dict(
    type='FreqDecoupledDistillerCNN',
    student_cfg=student_cfg,
    teacher_backbone_cfg=teacher_backbone_cfg,
    data_preprocessor=data_preprocessor,

    # --- Frequency decomposition params (same as ViT version) ---
    freq_cutoff=0.5,
    high_freq_weight=0.1,

    # --- 4 layer pairs (ResNet stages 1-4) ---
    # Both teacher and student are ResNet-50 → same channels, no projectors
    distill_cfg=[
        dict(
            name='loss_res2',
            student_feature_index=0,
            teacher_feature_index=0,
            student_channels=256,
            teacher_channels=256,
            loss_weight=1.0),
        dict(
            name='loss_res3',
            student_feature_index=1,
            teacher_feature_index=1,
            student_channels=512,
            teacher_channels=512,
            loss_weight=1.0),
        dict(
            name='loss_res4',
            student_feature_index=2,
            teacher_feature_index=2,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0),
        dict(
            name='loss_res5',
            student_feature_index=3,
            teacher_feature_index=3,
            student_channels=2048,
            teacher_channels=2048,
            loss_weight=1.0),
    ],
)

# -------------------------------------------------------------------------
# 4. Dataset: KAIST Paired (IR + RGB) for cross-modal KD
# -------------------------------------------------------------------------
dataset_type = 'KAISTDataset'
data_root = './data/kaist-rgbt/'

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

train_dataloader = dict(
    batch_size=8,
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
        include_empty_images=True,
        metainfo=dict(classes=('person',))))

# No val/test during Stage 1 (backbone-only, no detection eval)
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None

# -------------------------------------------------------------------------
# 5. Training Schedule
# -------------------------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=999)

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
        eta_min=1e-7)
]

# AdamW: student backbone gets full LR (no LoRA, direct fine-tune)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2))

# -------------------------------------------------------------------------
# 6. DDP: Teacher + unused detection head → find_unused_parameters
# -------------------------------------------------------------------------
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

# -------------------------------------------------------------------------
# 7. Runtime
# -------------------------------------------------------------------------
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best=None),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

work_dir = './work_dirs/frcnn/stage1_freq_decoupled_r50'
