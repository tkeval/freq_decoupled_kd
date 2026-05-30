# ------------------------------------------------------------
# DINOv3-LARGE + 5-scale DINO detector on KAIST
# Latest generation DINOv3 with DINO detection head
# ------------------------------------------------------------
# usage: bash ./tools/dist_train.sh configs/kaist/dinov3/dino-5scale_dinov3-large-p16_8xb2-12e_kaist.py 4 --work-dir work_dirs/kaist_dinov3_large_5scale
_base_ = [
    '../../_base_/datasets/kaist_detection.py',
    '../../_base_/default_runtime.py'
]

# Import MMPretrain models to register DINOv3 backbones
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
        pad_size_divisor=16),  # DINOv3 uses patch16 (not patch14!)
    
    # DINOv3-Large backbone
    backbone=dict(
        type='mmpretrain.TIMMBackbone',
        model_name='vit_large_patch16_dinov3',  # DINOv3-Large (1024 dims, 24 layers)
        pretrained=True,
        features_only=True,
        out_indices=(7, 15, 19, 21, 23),  # 5 layers from 24-layer ViT-Large for multi-scale features
        dynamic_img_size=True,
        init_cfg=None),
    
    # Neck to adapt ViT features - 5 scales
    neck=dict(
        type='ChannelMapper',
        in_channels=[1024, 1024, 1024, 1024, 1024],  # DINOv3-Large outputs 1024 dims
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
# Start with same hyperparameters as DINOv2
# May need adjustment based on DINOv3's characteristics
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,  # Conservative LR for fine-tuning pretrained DINOv3
        weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Backbone gets 5e-6 (10x smaller)
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,  # Start from 5e-8, warmup to 5e-5
        by_epoch=False,
        begin=0,
        end=500),  # 500 iterations warmup
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[11],  # Drop LR by 10x at epoch 11
        gamma=0.1)
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Training dataloader - batch_size=2 for stability
train_dataloader = dict(
    batch_size=2,  # Conservative batch size for stable fine-tuning
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
        max_keep_ckpts=1,  # Keep 2 checkpoints
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
# DINOv3 vs DINOv2 Differences
# --------------------
# 1. Patch size: 16 (vs 14 in DINOv2)
#    - pad_size_divisor changed to 16
#    - Slightly different resolution handling
#
# 2. Architecture: Uses Eva-style architecture
#    - Same 1024 dims and 24 layers as DINOv2-Large
#    - Potentially improved feature quality
#
# 3. Pretraining: DINOv3 may have different/improved pretraining
#    - Could lead to better downstream performance
#    - May converge faster or reach higher mAP
#
# Expected performance (DINOv3-Large):
# - Epoch 1: mAP@50 ≈ 35-40% (baseline check)
# - Epoch 6: mAP@50 ≈ 55-60%
# - Epoch 12: mAP@50 ≈ 67-72% (similar to DINOv2-Large)
# - Epoch 24: mAP@50 ≈ 69-73% (with extended training)
#
# Model comparison:
# - DINOv1 + Swin-L: 197M params → 68.5% mAP@50
# - DINOv2-Large + reg: 304M params → ~67% mAP@50 (12 epochs)
# - DINOv3-Large: 304M params → 67-72% mAP@50 (expected, 12 epochs)
#
# Notes:
# 1. DINOv3 is the latest version - may have quality improvements
# 2. Start with same hyperparameters as DINOv2
# 3. Monitor first epoch - if significantly different from DINOv2, adjust LR
# 4. Consider training for 24 epochs to reach full potential


