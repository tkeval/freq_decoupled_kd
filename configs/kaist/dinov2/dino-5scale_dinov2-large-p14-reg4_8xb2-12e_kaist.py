# ------------------------------------------------------------
# DINOv2-LARGE with Registers + 5-scale DINO detector on KAIST
# Optimized configuration for best performance (matches Swin-L capacity)
# ------------------------------------------------------------
# usage: bash ./tools/dist_train.sh configs/kaist/dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py 4 --work-dir work_dirs/kaist_dinov2_large_reg_5scale_ir
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
        pretrained=True,
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
# CRITICAL: Fine-tuning pretrained models requires SMALL learning rates!
# See explanation below for why 5e-5 is chosen.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,  # Conservative LR for fine-tuning pretrained DINOv2
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

# Override train pipeline: use standard single-image transforms
# (base dataset config has paired-image transforms with keys= that break here)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

# Training dataloader - batch_size=2 for stability
# See explanation below for batch size choice
train_dataloader = dict(
    batch_size=2,  # Conservative batch size for stable fine-tuning
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
# WHY learning_rate = 5e-5?
# 1. DINOv2 is PRETRAINED on 142M images - features are already excellent
# 2. High LR (like 2e-4) DESTROYS pretrained features → poor performance
# 3. Fine-tuning rule: Use 1/10 to 1/100 of training-from-scratch LR
# 4. DINO trains from scratch with ~1e-4, so fine-tuning uses 5e-5
# 5. Backbone gets 10x smaller LR (5e-6) to preserve pretrained features
#
# WHY batch_size = 2?
# 1. STABILITY: Smaller batches = more gradient updates = more stable for fine-tuning
# 2. KAIST is SMALL (7,601 train images): Large batches → overfitting
# 3. Effective batch size = 2 × 4 GPUs = 8 (good for detection)
# 4. Batch size 4 can work but may need even smaller LR (e.g., 3e-5)
# 5. For small datasets, conservative settings (batch=2, lr=5e-5) are safer
#
# Expected performance (with ViT-Large):
# - Epoch 1: mAP@50 ≈ 35-40%
# - Epoch 6: mAP@50 ≈ 55-60%
# - Epoch 12: mAP@50 ≈ 67-72% (should match or exceed Swin-L at 68.5%)
#
# Model comparison:
# - DINOv1 + Swin-L: 197M params → 68.5% mAP@50
# - DINOv2-Large + registers: 304M params → 67-72% mAP@50 (expected)
#
# To improve further:
# 1. Train for 24 epochs instead of 12 (+2-3% mAP)
# 2. Add stronger augmentation (multiscale, mixup)
# 3. Increase image resolution (640x512 → 896x896)

