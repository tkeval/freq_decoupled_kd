_base_ = [
    '../dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py'
]

# -------------------------------------------------------------------------
# 1. Define Teachers (using mmpretrain configs)
# -------------------------------------------------------------------------
teachers = dict(
    # SAM Teacher (Structure)
    # Using TIMM implementation of SAM-ViT-Large
    sam=dict(
        source='mmpretrain',
        backbone=dict(
            type='mmpretrain.TIMMBackbone',
            model_name='samvit_large_patch16', # SAM1 ViT-Large
            pretrained=True,
            features_only=True,
            out_indices=(3, 7, 11, 15), # 4 stages
            img_size=1024, # SAM handles variable sizes natively (windowed attention)
        )
    ),
    
    # DINOv2 Teacher (Semantics)
    # Using the same DINOv2 backbone as the student, but frozen & RGB
    dino=dict(
        source='mmpretrain',
        backbone=dict(
            type='mmpretrain.TIMMBackbone',
            model_name='vit_large_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            features_only=True,
            out_indices=(7, 15, 19, 23), # Extract features from blocks 8, 16, 20, 24
            dynamic_img_size=True
        )
    ),
    
    # CLIP Teacher (Abstract)
    # Using TIMM implementation of CLIP-ViT-L/14 (matches student capacity)
    # ViT-L/14: 24 blocks, embed_dim=1024, patch_size=14
    clip=dict(
        source='mmpretrain',
        backbone=dict(
            type='mmpretrain.TIMMBackbone',
            model_name='vit_large_patch14_clip_224.openai',  # CLIP ViT-L/14
            pretrained=True,
            features_only=True,
            out_indices=(5, 11, 17, 23),  # 4 evenly-spaced stages from 24 blocks
            dynamic_img_size=True,
        )
    )
)

# -------------------------------------------------------------------------
# 2. Stage 1 Distiller Configuration
# -------------------------------------------------------------------------
model = dict(
    _delete_=True,
    type='Stage1FeatureDistiller',
    
    # Student Config (Inherit from base detector)
    student_cfg={{_base_.model}},
    
    # Teachers Config
    teacher_cfgs=teachers,
    
    # Distillation Losses
    distill_cfg=[
        # --- SAM Layers (Early Layers: 1-4) ---
        # Student Layer 1 (Block 6 approx) -> SAM Layer 1 (Block 4)
        dict(
            name='loss_sam_early',
            teacher_name='sam',
            student_feature_index=0, # Corresponds to out_indices[0] of student
            teacher_feature_index=0, # Corresponds to out_indices[0] of SAM
            student_channels=1024,   # ViT-Large
            teacher_channels=1024,   # SAM-Large
            loss_type='MSELoss',
            loss_weight=1.0
        ),
        
        # --- DINO Layers (Middle Layers: 5-8) ---
        # Student Layer 2 (Block 12 approx) -> DINO Layer 2 (Block 16)
        dict(
            name='loss_dino_mid',
            teacher_name='dino',
            student_feature_index=1,
            teacher_feature_index=1,
            student_channels=1024,
            teacher_channels=1024,
            loss_type='MSELoss',
            loss_weight=1.0
        ),
        
        # --- CLIP Layers (Deep Layers: 9-12) ---
        # Student Layer 4 (Block 21 approx) -> CLIP Layer 4 (Block 23)
        dict(
            name='loss_clip_deep',
            teacher_name='clip',
            student_feature_index=3,
            teacher_feature_index=3,
            student_channels=1024,
            teacher_channels=1024,   # CLIP ViT-L/14 (embed_dim=1024)
            loss_type='MSELoss',
            loss_weight=1.0
        ),
    ]
)

# -------------------------------------------------------------------------
# 3. Override train pipeline to use PairedResize / PairedRandomFlip
# -------------------------------------------------------------------------
# The base config (kaist_detection.py) uses Resize/RandomFlip with keys=[...]
# which is not supported. We override to use PairedResize/PairedRandomFlip.
train_pipeline = [
    dict(type='LoadPairedImagesFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PairedResize', scale=(640, 512), keep_ratio=True),
    dict(type='PairedRandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'img2_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'img_rgb'))
]

# -------------------------------------------------------------------------
# 5. DDP: Allow unused parameters
# -------------------------------------------------------------------------
# In Stage 1, only the student backbone + projectors are used.
# The student's neck/encoder/decoder/head and all teacher params are unused.
# We must tell DDP to allow this.
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

# -------------------------------------------------------------------------
# 6. Training Loop & Checkpointing
# -------------------------------------------------------------------------
# Disable validation (no detection head trained, mAP will be 0)
# Save only the last checkpoint to conserve disk space
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=999)  # Effectively disable validation
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Work directory on scratch (more space)
work_dir = './work_dirs/stage1_distill'

# Only keep 1 checkpoint (the latest)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,         # Save every 4 epochs (epochs 4, 8, 12)
        max_keep_ckpts=1,   # Only keep the latest checkpoint
        save_best=None),    # No "best" tracking (validation disabled)
    logger=dict(type='LoggerHook', interval=50))

# -------------------------------------------------------------------------
# 6. Batch Size & Learning Rate for 4x 48GB GPUs
# -------------------------------------------------------------------------
# Memory breakdown at batch_size=2: ~14.2 GB
# With 48GB GPUs, batch_size=4 is safe (~28 GB), leaving headroom for peaks
# Effective batch size = 4 per GPU × 4 GPUs = 16
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        pipeline=train_pipeline,
        include_empty_images=True))  # Use ALL images for feature distillation

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
        milestones=[8, 11],
        gamma=0.1)
]

# Conservative LR to preserve pretrained DINOv2 features.
# Backbone gets 0.1x multiplier (5e-6 effective) to avoid feature destruction.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)
