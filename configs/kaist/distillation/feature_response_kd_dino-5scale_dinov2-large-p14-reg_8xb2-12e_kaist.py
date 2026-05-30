# Config for HYBRID FEATURE-RESPONSE Cross-Modal Knowledge Distillation
# It uses the FeatureResponseKDDINO distiller.
# usage: bash ./tools/dist_train.sh configs/kaist/distillation/feature_response_kd_dino-5scale_dinov2-large-p14-reg_8xb2-12e_kaist.py 4 --work-dir work_dirs/feature_response_kd_dino

_base_ = [
    '../../_base_/datasets/kaist_detection.py',
    '../../_base_/default_runtime.py'
]

# For DDP training with a frozen teacher model, we need to tell the wrapper
# to find parameters that are not used in the backward pass.
find_unused_parameters = True

custom_imports = dict(
    imports=['mmdet.models.distillers'],
    allow_failed_imports=False)

teacher_checkpoint = 'work_dirs/kaist_dinov2_large_reg_5scale/best_coco_person_precision_epoch_10.pth'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'img2_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]

# Set batch size for dataloaders
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

# DINO 5-scale model configuration, used for both teacher and student
model_cfg = dict(
    type='DINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=5,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14),
    backbone=dict(
        type='mmpretrain.TIMMBackbone',
        model_name='vit_large_patch14_reg4_dinov2.lvd142m',
        features_only=True,
        pretrained=True,
        out_indices=(7, 15, 19, 21, 23),
        dynamic_img_size=True,
        init_cfg=None),
    neck=dict(
        type='ChannelMapper',
        in_channels=[1024, 1024, 1024, 1024, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=5, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=5, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100)
)

# Distiller model configuration
model = model_cfg.copy()
model['type'] = 'FeatureResponseKDDINO' # Use the new hybrid distiller
model['teacher_checkpoint'] = teacher_checkpoint
model['teacher_cfg'] = model_cfg
model['distill_cfg'] = dict(
    loss_weight_cls=0.25,
    loss_weight_bbox=0.25,
    loss_weight_feat=0.5, # Emphasize feature distillation
    temperature=2.0)

# Optimizer and Schedule from the working response-based KD config
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1)
        }),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[11], gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

