# ------------------------------------------------------------
# DINOv2 + DINO detector on KAIST dataset
# This combines DINOv2 backbone with DINO transformer detector head
# ------------------------------------------------------------
# usage: bash ./tools/dist_train.sh configs/kaist/dinov2/dino-4scale_dinov2-base-p14_8xb2-12e_kaist.py 4 --work-dir work_dirs/kaist_dinov2_dino_head
_base_ = [
    '../../_base_/datasets/kaist_detection.py',
    '../../_base_/default_runtime.py'
]

# Import MMPretrain models to register DINOv2 backbones
custom_imports = dict(
    imports=['mmpretrain.models'],
    allow_failed_imports=False
)

# --------------------
# Model settings
# --------------------
model = dict(
    type='DINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14),  # Match DINOv2 patch size
    
    # DINOv2 backbone
    backbone=dict(
        type='mmpretrain.TIMMBackbone',
        model_name='vit_base_patch14_dinov2.lvd142m',  # DINOv2 ViT-B/14
        pretrained=True,
        features_only=True,
        out_indices=(3, 6, 9, 11),  # Multi-scale features from ViT layers
        dynamic_img_size=True,
        init_cfg=None),
    
    # Neck to adapt ViT features
    neck=dict(
        type='ChannelMapper',
        in_channels=[768, 768, 768, 768],  # DINOv2-Base outputs
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    
    # DINO encoder
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, 
                num_levels=4,
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    
    # DINO decoder
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, 
                num_heads=8,
                dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256, 
                num_levels=4,
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    
    # Positional encoding
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    
    # DINO head
    bbox_head=dict(
        type='DINOHead',
        num_classes=1,  # KAIST: single class (person)
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    
    # Denoising config for DINO
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(
            dynamic=True, 
            num_groups=None, 
            num_dn_queries=100)),
    
    # Training config
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    
    # Testing config
    test_cfg=dict(max_per_img=100))

# --------------------
# Optimization & training
# --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,  # Scaled for 4 GPUs × 4 batch_size = 16 total
        weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Lower LR for pretrained backbone
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Training dataloader
train_dataloader = dict(
    batch_size=4,  # 4 per GPU × 4 GPUs = 16 total batch size
    num_workers=4)
val_dataloader = dict(
    batch_size=2,  
    num_workers=4)
test_dataloader = dict(
    batch_size=2,
    num_workers=4)

# --------------------
# Runtime settings
# --------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_level = 'INFO'
load_from = None
resume = False
