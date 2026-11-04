# -------------------------------------------------------------------------
# Stage 1: Cross-Modal Feature Distillation (RGB DINOv2-Large → IR ViT-Base)
# -------------------------------------------------------------------------
# Feature-only distillation using cosine similarity. No detection loss.
# Goal: teach ViT-Base to produce IR features that are semantically
# similar to what DINOv2-Large sees from RGB images.
#
# After this stage, extract the backbone and fine-tune for detection
# using the ViT-Base detection config.
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/distillation/stage1_crossmodal_vitbase.py 4 \
#       --work-dir ./work_dirs/stage1_crossmodal_vitbase
# -------------------------------------------------------------------------
_base_ = [
    '../../_base_/datasets/kaist_paired_detection.py',
    '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmpretrain.models', 'mmdet.models.distillers'],
    allow_failed_imports=False)

# -------------------------------------------------------------------------
# 1. Teacher: DINOv2-Large RGB backbone (frozen)
# -------------------------------------------------------------------------
# 24 blocks, 1024-dim. We select 4 evenly-spaced layers for alignment.
teachers = dict(
    dino=dict(
        source='mmpretrain',
        backbone=dict(
            type='mmpretrain.TIMMBackbone',
            model_name='vit_large_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            features_only=True,
            out_indices=(5, 11, 17, 23),
            dynamic_img_size=True,
        )
    ),
)

# -------------------------------------------------------------------------
# 2. Student: ViT-Base IR backbone (12 blocks, 768-dim)
# -------------------------------------------------------------------------
# out_indices for distillation alignment + detection:
#   Distillation: blocks 2, 5, 8, 11 (4 evenly-spaced from 12 blocks)
#   Detection:    blocks 2, 5, 8, 10, 11 (5-scale for DINO)
# Combined:       blocks 2, 5, 8, 10, 11 (indices 0-4)
# Distillation uses indices 0, 1, 2, 4 (blocks 2, 5, 8, 11)
student_out_indices = (2, 5, 8, 10, 11)

# All 5 outputs are used for detection (same as baseline)
detection_feature_indices = [0, 1, 2, 3, 4]

num_levels = 5

model = dict(
    type='Stage1FeatureDistiller',

    student_cfg=dict(
        type='DINO',
        num_feature_levels=num_levels,
        num_queries=900,
        with_box_refine=True,
        as_two_stage=True,
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=14),
        backbone=dict(
            type='mmpretrain.TIMMBackbone',
            model_name='vit_base_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            features_only=True,
            out_indices=student_out_indices,
            dynamic_img_size=True,
            init_cfg=None),
        neck=dict(
            type='ChannelMapper',
            in_channels=[768, 768, 768, 768, 768],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=num_levels),
        encoder=dict(
            num_layers=6,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_levels=num_levels,
                    num_heads=8, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
        decoder=dict(
            num_layers=6,
            return_intermediate=True,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_heads=8, dropout=0.0),
                cross_attn_cfg=dict(
                    embed_dims=256, num_levels=num_levels, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
            post_norm_cfg=None),
        positional_encoding=dict(
            num_feats=128, normalize=True, offset=0.0, temperature=20),
        bbox_head=dict(
            type='DINOHead',
            num_classes=1,
            sync_cls_avg_factor=True,
            loss_cls=dict(
                type='FocalLoss', use_sigmoid=True, gamma=2.0,
                alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(
                dynamic=True, num_groups=None, num_dn_queries=100)),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
        test_cfg=dict(max_per_img=100)),

    detection_feature_indices=detection_feature_indices,

    teacher_cfgs=teachers,

    # -------------------------------------------------------------------------
    # 3. Distillation: 4 layer pairs, cosine similarity
    # -------------------------------------------------------------------------
    # Student ViT-Base (12 blocks) → Teacher DINOv2-Large (24 blocks)
    #   Student block 2  (out idx 0) → Teacher block 5  (out idx 0)
    #   Student block 5  (out idx 1) → Teacher block 11 (out idx 1)
    #   Student block 8  (out idx 2) → Teacher block 17 (out idx 2)
    #   Student block 11 (out idx 4) → Teacher block 23 (out idx 3)
    distill_cfg=[
        dict(
            name='loss_feat_l1',
            teacher_name='dino',
            student_feature_index=0,   # student block 2
            teacher_feature_index=0,   # teacher block 5
            student_channels=768,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0),
        dict(
            name='loss_feat_l2',
            teacher_name='dino',
            student_feature_index=1,   # student block 5
            teacher_feature_index=1,   # teacher block 11
            student_channels=768,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0),
        dict(
            name='loss_feat_l3',
            teacher_name='dino',
            student_feature_index=2,   # student block 8
            teacher_feature_index=2,   # teacher block 17
            student_channels=768,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0),
        dict(
            name='loss_feat_l4',
            teacher_name='dino',
            student_feature_index=4,   # student block 11
            teacher_feature_index=3,   # teacher block 23
            student_channels=768,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0),
    ])

# -------------------------------------------------------------------------
# 4. Training pipeline (paired RGB + IR)
# -------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadPairedImagesFromFile', backend_args=None),
    dict(type='PairedResize', scale=(640, 512), keep_ratio=True),
    dict(type='PairedRandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'img2_path', 'ori_shape', 'img_shape',
        'scale_factor', 'flip', 'flip_direction', 'img_rgb'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        include_empty_images=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

# -------------------------------------------------------------------------
# 5. Optimizer
# -------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01,
         by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, end=12,
         by_epoch=True, eta_min=1e-7)]

# -------------------------------------------------------------------------
# 6. Training loop (no validation — feature distillation only)
# -------------------------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best=None),
    logger=dict(type='LoggerHook', interval=50))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
