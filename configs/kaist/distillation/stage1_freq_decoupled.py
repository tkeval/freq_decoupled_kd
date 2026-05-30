# -------------------------------------------------------------------------
# Stage 1: Frequency-Decoupled Cross-Modal KD
# -------------------------------------------------------------------------
# Single DINOv2 teacher (RGB) → DINOv2 student (IR) with LoRA.
#
# Key differences from stage1_distill_v2 (old approach):
#   1. Single DINOv2 teacher (no SAM/CLIP — no conflicting gradients)
#   2. Same architecture → no projection layers needed
#   3. Frequency-decoupled loss instead of cosine similarity:
#      - MSE on low-frequency (structural, modality-general)
#      - logMSE on high-frequency (texture, modality-specific — relaxed)
#   4. 5 layer pairs matching detection out_indices (not 12)
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_freq_decoupled.py 4 \
#       --work-dir ./work_dirs/stage1/fft/stage1_freq_decoupled
# -------------------------------------------------------------------------
_base_ = [
    '../dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py'
]

custom_imports = dict(
    imports=['mmpretrain.models', 'mmdet.models.distillers'],
    allow_failed_imports=False)

# -------------------------------------------------------------------------
# 1. Teacher: RGB DINOv2 ViT-Large (same architecture as student)
# -------------------------------------------------------------------------
teacher_backbone_cfg = dict(
    type='mmpretrain.TIMMBackbone',
    model_name='vit_large_patch14_reg4_dinov2.lvd142m',
    pretrained=True,
    features_only=True,
    out_indices=(7, 15, 19, 21, 23),  # Same 5 layers as student
    dynamic_img_size=True,
)

# -------------------------------------------------------------------------
# 2. Student: DINOv2 ViT-Large IR backbone (same out_indices as teacher)
# -------------------------------------------------------------------------
# Using the same 5 out_indices as detection baseline.
# No need for detection_feature_indices since backbone outputs match neck.
student_out_indices = (7, 15, 19, 21, 23)

# -------------------------------------------------------------------------
# 3. Model: FreqDecoupledDistiller
# -------------------------------------------------------------------------
model = dict(
    _delete_=True,
    type='FreqDecoupledDistiller',

    student_cfg=dict(
        type='DINO',
        num_feature_levels={{_base_.num_levels}},
        num_queries=900,
        with_box_refine=True,
        as_two_stage=True,
        data_preprocessor={{_base_.model.data_preprocessor}},
        backbone=dict(
            type='mmpretrain.TIMMBackbone',
            model_name='vit_large_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            features_only=True,
            out_indices=student_out_indices,
            dynamic_img_size=True,
            init_cfg=None),
        neck={{_base_.model.neck}},
        encoder={{_base_.model.encoder}},
        decoder={{_base_.model.decoder}},
        positional_encoding={{_base_.model.positional_encoding}},
        bbox_head={{_base_.model.bbox_head}},
        dn_cfg={{_base_.model.dn_cfg}},
        train_cfg={{_base_.model.train_cfg}},
        test_cfg={{_base_.model.test_cfg}},
    ),

    teacher_backbone_cfg=teacher_backbone_cfg,

    # --- LoRA config (same as stage1_v2_lora_strong) ---
    lora_cfg=dict(
        rank=64,
        alpha=64.0,
        dropout=0.05,
        target_modules=['attn.qkv', 'mlp.fc1', 'mlp.fc2'],
    ),

    # --- Frequency decomposition params ---
    freq_cutoff=0.5,         # Center 50% of spectrum = low-freq
    high_freq_weight=0.1,    # logMSE weight (10x weaker than MSE)

    # --- Distillation: 5 layer pairs (same layers, same channels) ---
    # Both teacher and student use out_indices=(7, 15, 19, 21, 23)
    # with 1024-dim features. No projectors needed.
    distill_cfg=[
        dict(
            name='loss_layer7',
            student_feature_index=0,   # out_indices[0] = block 7
            teacher_feature_index=0,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_layer15',
            student_feature_index=1,   # out_indices[1] = block 15
            teacher_feature_index=1,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_layer19',
            student_feature_index=2,   # out_indices[2] = block 19
            teacher_feature_index=2,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_layer21',
            student_feature_index=3,   # out_indices[3] = block 21
            teacher_feature_index=3,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_layer23',
            student_feature_index=4,   # out_indices[4] = block 23
            teacher_feature_index=4,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
    ],
)

# -------------------------------------------------------------------------
# 4. Training Pipeline (paired RGB-IR images)
# -------------------------------------------------------------------------
# Override base single-image pipeline with paired pipeline for KD
train_pipeline = [
    dict(type='LoadPairedImagesFromFile', backend_args=None),
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
    dataset=dict(
        pipeline=train_pipeline,
        include_empty_images=True))

# -------------------------------------------------------------------------
# 5. Training Schedule
# -------------------------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=999)  # No detection eval during Stage 1
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Cosine annealing for smooth convergence
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

# LoRA params get full LR; no backbone lr_mult since backbone is frozen
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# -------------------------------------------------------------------------
# 6. DDP: Teacher backbone is frozen, may have unused params
# -------------------------------------------------------------------------
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

# -------------------------------------------------------------------------
# 7. Checkpointing & Work Dir
# -------------------------------------------------------------------------
work_dir = './work_dirs/stage1/fft/stage1_freq_decoupled'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best=None),
    logger=dict(type='LoggerHook', interval=50))
