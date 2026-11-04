# ------------------------------------------------------------
# DINOv2 + Faster R-CNN fine-tuning on KAIST dataset
# ------------------------------------------------------------
# usage: bash ./tools/dist_train.sh configs/kaist/dinov2/faster-rcnn_dinov2-base-p14_fpn_8xb2-12e_kaist.py 4 --work-dir work_dirs/kaist_dinov2_frcnn
_base_ = [
    '../../_base_/datasets/kaist_detection.py',              # import your dataset setup
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_1x.py'
]

custom_imports = dict(
    imports=['mmpretrain.models'],
    allow_failed_imports=False)


# --------------------
# Model settings
# --------------------
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14),  # Match DINOv2 patch size
    backbone=dict(
        type='mmpretrain.TIMMBackbone',  # Use TIMMBackbone from MMPretrain
        model_name='vit_base_patch14_dinov2.lvd142m',  # DINOv2 ViT-B/14
        pretrained=True,
        features_only=True,
        out_indices=(11,),  # Use only the last layer
        dynamic_img_size=True,  # Allow flexible input sizes
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[768],  # Single feature from last ViT layer
        out_channels=256,
        num_outs=5,  # FPN will create pyramid
        add_extra_convs='on_input'),  # Add extra convs for more scales
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[14, 28, 56, 112, 224]),  # Based on patch size 14
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
            featmap_strides=[14, 28, 56, 112, 224]),  # Match the RPN strides
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # single class: 'person'
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
            allowed_border=0,
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
                match_low_quality=True,
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
            max_per_img=100))
)

# --------------------
# Optimization & training
# --------------------
optim_wrapper = dict(
    _delete_=True,  # Delete the entire base optim_wrapper
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05),  # Conservative LR for fine-tuning pretrained DINOv2
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=1e-6,
        by_epoch=True,
        begin=1)
]

train_cfg = dict(max_epochs=12, val_interval=1)

# --------------------
# Runtime settings
# --------------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=50)
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = None
resume = False
