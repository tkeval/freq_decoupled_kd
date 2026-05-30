# -------------------------------------------------------------------------
# Stage 2: Detection Fine-Tuning with Freq-Decoupled Stage 1 Backbone
# -------------------------------------------------------------------------
# Uses the backbone from Stage 1 (frequency-decoupled cross-modal KD).
# The LoRA adapters learned modality-general (low-freq) adaptations,
# which should be safe to merge at higher scaling than the old cosine-sim
# Stage 1 (which learned harmful modality-specific mimicry).
#
# Inherits from baseline DINO config and uses Stage2GuidedDetector to
# handle LoRA merge and optional feature KD from frozen teacher.
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage2_freq_decoupled_det.py 4 \
#       --work-dir ./work_dirs/stage2_freq_decoupled
# -------------------------------------------------------------------------
_base_ = [
    '../dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py'
]

# -------------------------------------------------------------------------
# 1. Stage 1 Checkpoint (frequency-decoupled KD)
# -------------------------------------------------------------------------
stage1_checkpoint = './work_dirs/stage1/fft/stage1_freq_decoupled/epoch_12.pth'

# -------------------------------------------------------------------------
# 2. Teacher Backbone Config (same architecture as student)
# -------------------------------------------------------------------------
teacher_backbone_cfg = dict(
    type='mmpretrain.TIMMBackbone',
    model_name='vit_large_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    features_only=True,
    out_indices=(7, 15, 19, 21, 23),
    dynamic_img_size=True,
)

# -------------------------------------------------------------------------
# 3. Model: Stage2GuidedDetector
# -------------------------------------------------------------------------
model = dict(
    _delete_=True,
    type='Stage2GuidedDetector',

    # --- Teacher (frozen IR-ViT from Stage 1) ---
    teacher_backbone_cfg=teacher_backbone_cfg,
    teacher_checkpoint=stage1_checkpoint,

    # --- LoRA Merge Scaling ---
    # With freq-decoupled Stage 1, the LoRA learned structural (low-freq)
    # adaptations. Full merge should be safe (unlike old cosine-sim Stage 1
    # where scale=1.0 was catastrophic).
    lora_merge_scaling=0.5,

    # --- Distillation Config ---
    # Keep light feature KD (same as old Stage 2 best: 0.1)
    distill_weight=0.1,

    distill_cfg=[
        dict(
            name='loss_feat_layer7',
            student_feature_index=0,
            teacher_feature_index=0,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_feat_layer15',
            student_feature_index=1,
            teacher_feature_index=1,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_feat_layer19',
            student_feature_index=2,
            teacher_feature_index=2,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_feat_layer21',
            student_feature_index=3,
            teacher_feature_index=3,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_feat_layer23',
            student_feature_index=4,
            teacher_feature_index=4,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
    ],

    # --- Student Detector (inherited from base DINO config) ---
    num_feature_levels={{_base_.num_levels}},
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor={{_base_.model.data_preprocessor}},
    backbone={{_base_.model.backbone}},
    neck={{_base_.model.neck}},
    encoder={{_base_.model.encoder}},
    decoder={{_base_.model.decoder}},
    positional_encoding={{_base_.model.positional_encoding}},
    bbox_head={{_base_.model.bbox_head}},
    dn_cfg={{_base_.model.dn_cfg}},
    train_cfg={{_base_.model.train_cfg}},
    test_cfg={{_base_.model.test_cfg}},
)

# -------------------------------------------------------------------------
# 4. Training Schedule (matches baseline exactly)
# -------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'teacher_backbone': dict(lr_mult=0.0),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=0.1, norm_type=2))

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

# -------------------------------------------------------------------------
# 5. Training Loop
# -------------------------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# -------------------------------------------------------------------------
# 6. Data: Standard IR-only pipeline (matches baseline)
# -------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline))

# -------------------------------------------------------------------------
# 7. DDP: Allow unused parameters (teacher backbone is frozen)
# -------------------------------------------------------------------------
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

# -------------------------------------------------------------------------
# 8. Checkpointing & Work Dir
# -------------------------------------------------------------------------
work_dir = './work_dirs/stage2_freq_decoupled'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))
