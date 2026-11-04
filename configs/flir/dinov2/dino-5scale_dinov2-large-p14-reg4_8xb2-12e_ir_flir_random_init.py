# ------------------------------------------------------------
# DINOv2-LARGE with Registers + 5-scale DINO detector on flir
# Optimized configuration for best performance (matches Swin-L capacity)
# ------------------------------------------------------------
# usage: bash ./tools/dist_train.sh flir/dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_ir_flir_random_init 4 --work-dir work_dirs/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_ir_flir_random_init
_base_ = [
    '../../_base_/datasets/flir_detection.py',
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
num_levels = 5  # 5-scale for better multi-scale detection

model = dict(
    type='DINO',
    num_feature_levels=num_levels,  # CRITICAL: Must match num_levels=5
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14),  # Match DINOv2 patch size
    
    # DINOv2-Large with registers backbone
    backbone=dict(
        type='mmpretrain.TIMMBackbone',
        model_name='vit_large_patch14_reg4_dinov2.lvd142m',  # DINOv2-Large with 4 registers (304M params)
        pretrained=False,
        features_only=True,
        out_indices=(7, 15, 19, 21, 23),  # 5 layers from 24-layer ViT-Large for multi-scale features
        dynamic_img_size=True,
        init_cfg=None),
    
    # Neck to adapt ViT features - 5 scales
    neck=dict(
        type='ChannelMapper',
        in_channels=[1024, 1024, 1024, 1024, 1024],  # DINOv2-Large outputs 1024 dims
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels),  # 5 output scales
    
    # DINO encoder - 5 scale support
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, 
                num_levels=num_levels,  # 5 scales
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    
    # DINO decoder - 5 scale support
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
                num_levels=num_levels,  # 5 scales
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
        num_classes=3,   # person, car, bicycle
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
# IR-only train pipeline (overrides base which loads paired RGB+IR)
# --------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),  # loads img_path = IR image (swapped in prepare_data)
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# --------------------
# Optimization & training
# --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,          # standard from-scratch LR
        weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=12, eta_min=1e-6)
]

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=2)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Training dataloader — override base dataset to use IR-only pipeline
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline))
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
        max_keep_ckpts=2,  # Keep 2 checkpoints
        save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_level = 'INFO'
load_from = None
resume = False

# --------------------
# Configuration Notes
# --------------------
# Random init (pretrained=False): lr=1e-4, no backbone lr_mult needed.
# IR-only training: KAISTDataset.prepare_data swaps img_path → lwir,
#   so LoadImageFromFile always loads the IR image.
# 24 epochs with cosine annealing: 1e-4 → 1e-6 over full training duration.

