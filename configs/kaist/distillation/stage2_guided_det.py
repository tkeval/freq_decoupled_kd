# -------------------------------------------------------------------------
# Stage 2: Guided Detection Fine-Tuning
# -------------------------------------------------------------------------
# The student is a full DINO detector with IR-ViT backbone (initialized
# from Stage 1). A frozen copy of the Stage 1 IR-ViT acts as a teacher,
# providing feature-level guidance during detection training.
#
# Total Loss = Detection Loss + λ * Feature Alignment Loss (cosine sim)
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage2_guided_det.py 4 \
#       --work-dir ./work_dirs/stage2_guided_det_fixed
# -------------------------------------------------------------------------
_base_ = [
    '../dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py'
]

# -------------------------------------------------------------------------
# 1. Stage 1 Checkpoint (update this path after Stage 1 finishes)
# -------------------------------------------------------------------------
stage1_checkpoint = './work_dirs/stage1_v2_lora_strong_fixed/epoch_12.pth'

# -------------------------------------------------------------------------
# 2. Teacher Backbone Config (same architecture as student backbone)
# -------------------------------------------------------------------------
# This must match the student backbone architecture exactly so we can
# load the Stage 1 weights into it.
teacher_backbone_cfg = dict(
    type='mmpretrain.TIMMBackbone',
    model_name='vit_large_patch14_reg4_dinov2.lvd142m',
    pretrained=False,  # We load from Stage 1 checkpoint, not pretrained
    features_only=True,
    out_indices=(7, 15, 19, 21, 23),  # Same as student backbone
    dynamic_img_size=True,
)

# -------------------------------------------------------------------------
# 3. Model: Stage2GuidedDetector (extends DINO)
# -------------------------------------------------------------------------
model = dict(
    _delete_=True,
    type='Stage2GuidedDetector',

    # --- Teacher (frozen IR-ViT from Stage 1) ---
    teacher_backbone_cfg=teacher_backbone_cfg,
    teacher_checkpoint=stage1_checkpoint,

    # --- LoRA Merge Scaling ---
    # Controls how much Stage 1 cross-modal knowledge is applied:
    #   0.0 = pure DINOv2 (baseline), 1.0 = full merge (current)
    lora_merge_scaling=0.1,

    # --- Distillation Config ---
    # λ: global weight for all distillation losses relative to detection loss
    distill_weight=0.1,

    # Feature alignment pairs: Student backbone layer ↔ Teacher backbone layer
    # Both use out_indices=(7, 15, 19, 21, 23), so indices 0-4 map to the
    # same ViT blocks. Cosine similarity loss per pair.
    distill_cfg=[
        dict(
            name='loss_feat_layer7',
            student_feature_index=0,   # out_indices[0] = block 7
            teacher_feature_index=0,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_feat_layer15',
            student_feature_index=1,   # out_indices[1] = block 15
            teacher_feature_index=1,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_feat_layer19',
            student_feature_index=2,   # out_indices[2] = block 19
            teacher_feature_index=2,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_feat_layer21',
            student_feature_index=3,   # out_indices[3] = block 21
            teacher_feature_index=3,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
        dict(
            name='loss_feat_layer23',
            student_feature_index=4,   # out_indices[4] = block 23
            teacher_feature_index=4,
            student_channels=1024,
            teacher_channels=1024,
            loss_weight=1.0,
        ),
    ],

    # --- Student Detector (inherited from base DINO config) ---
    # All remaining kwargs are passed to DINO.__init__
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
# 4. Student Backbone Initialization
# -------------------------------------------------------------------------
# init_student_from_stage1=True (default) loads Stage 1 backbone weights
# into both the student and the frozen teacher. The student backbone is
# first initialized with DINOv2 pretrained weights (pretrained=True in
# base config), then overridden with Stage 1 weights.
# Set init_student_from_stage1=False to keep vanilla DINOv2 init for
# the student (only the teacher gets Stage 1 weights).

# -------------------------------------------------------------------------
# 5. Training Schedule
# -------------------------------------------------------------------------
# Use the same conservative schedule as baseline detection training,
# since we now have detection losses that need careful LR management.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,            # Conservative for fine-tuning
        weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Backbone gets 5e-6
            'teacher_backbone': dict(lr_mult=0.0),  # Teacher: no LR (frozen)
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
# 6. Training Loop
# -------------------------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1)  # Validate every epoch (mAP is meaningful now)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# -------------------------------------------------------------------------
# 7. Data: Standard IR-only pipeline (no paired images needed)
# -------------------------------------------------------------------------
# Stage 2 only uses IR images. Both teacher and student see the same IR input.
# Override the base pipeline which has broken Resize(..., keys=...) for paired images.
# Stage 2 uses a standard single-image pipeline.
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
# No need for filter_cfg or val_dataloader override.
# KAISTDataset now defaults to only loading images with annotations.
# Stage 1 uses include_empty_images=True to load all images.

# -------------------------------------------------------------------------
# 8. DDP: Allow unused parameters (teacher backbone is frozen)
# -------------------------------------------------------------------------
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

# -------------------------------------------------------------------------
# 9. Checkpointing & Work Dir
# -------------------------------------------------------------------------
work_dir = './work_dirs/stage2_guided_det_fixed'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))
