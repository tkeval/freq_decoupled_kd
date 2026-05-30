# -------------------------------------------------------------------------
# Stage 1 v2: Improved Multi-Teacher Feature Distillation
# -------------------------------------------------------------------------
# Key changes from v1:
#   1. Cosine similarity loss instead of MSE (direction alignment, not magnitude)
#   2. Much lower LR (1e-5) to preserve DINOv2 pretrained features
#   3. 24 epochs with cosine LR schedule for smoother convergence
#   4. More layer pairs (4 per teacher = 12 total, matching arch diagram)
#   5. Student out_indices expanded to cover all 12 alignment layers
#
# Usage:
#   bash ./tools/dist_train.sh configs/kaist/distillation/stage1_distill_v2.py 4 \
#       --work-dir ./work_dirs/stage1_distill_v2_fixes
# -------------------------------------------------------------------------
_base_ = [
    '../dinov2/dino-5scale_dinov2-large-p14-reg4_8xb2-12e_kaist.py'
]

# -------------------------------------------------------------------------
# 1. Define Teachers
# -------------------------------------------------------------------------
teachers = dict(
    # SAM Teacher (Structure) - Layers 1-4 of student
    sam=dict(
        source='mmpretrain',
        backbone=dict(
            type='mmpretrain.TIMMBackbone',
            model_name='samvit_large_patch16',
            pretrained=True,
            features_only=True,
            out_indices=(3, 7, 11, 15),  # SAM blocks 4, 8, 12, 16
            img_size=1024,
        )
    ),

    # DINOv2 Teacher (Semantics) - Layers 5-8 of student
    dino=dict(
        source='mmpretrain',
        backbone=dict(
            type='mmpretrain.TIMMBackbone',
            model_name='vit_large_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            features_only=True,
            out_indices=(7, 11, 15, 19),  # DINOv2 blocks 8, 12, 16, 20
            dynamic_img_size=True
        )
    ),

    # CLIP Teacher (Abstract) - Layers 9-12 of student
    # ViT-L/14: 24 blocks, embed_dim=1024, patch_size=14 (matches student capacity)
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
# 2. Student backbone with expanded out_indices
# -------------------------------------------------------------------------
# Original: out_indices=(7, 15, 19, 21, 23) -- 5 layers for detection
# Expanded: 12 layers to align with all teacher outputs
# ViT-Large has 24 blocks (0-23). We select 12 evenly spaced:
#   SAM alignment:  blocks 1, 3, 5, 7     (early layers)
#   DINO alignment: blocks 9, 11, 13, 15  (middle layers)
#   CLIP alignment: blocks 17, 19, 21, 23 (deep layers)
student_out_indices = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)

# Neck for detection: takes 5 features (selected by detection_feature_indices
# from the 12 backbone outputs). Matches the base DINO 5-scale config.
student_neck = dict(
    type='ChannelMapper',
    in_channels=[1024] * 5,  # 5 detection features, all 1024-dim
    kernel_size=1,
    out_channels=256,
    act_cfg=None,
    norm_cfg=dict(type='GN', num_groups=32),
    num_outs={{_base_.num_levels}})

# Map from 12 backbone outputs to 5 detection features:
#   backbone out_indices: (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)
#   detection blocks:     7=idx3, 15=idx7, 19=idx9, 21=idx10, 23=idx11
detection_feature_indices = [3, 7, 9, 10, 11]

# Override the student backbone to output all 12 layers
model = dict(
    _delete_=True,
    type='Stage1FeatureDistiller',

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
            out_indices=student_out_indices,  # 12 layers for distillation
            dynamic_img_size=True,
            init_cfg=None),
        neck=student_neck,
        encoder={{_base_.model.encoder}},
        decoder={{_base_.model.decoder}},
        positional_encoding={{_base_.model.positional_encoding}},
        bbox_head={{_base_.model.bbox_head}},
        dn_cfg={{_base_.model.dn_cfg}},
        train_cfg={{_base_.model.train_cfg}},
        test_cfg={{_base_.model.test_cfg}},
    ),

    detection_feature_indices=detection_feature_indices,

    teacher_cfgs=teachers,

    # -------------------------------------------------------------------------
    # 3. Distillation Config: 12 layer pairs with Cosine Similarity
    # -------------------------------------------------------------------------
    # Per architecture diagram:
    #   Student layers 1-4  (blocks 1,3,5,7)   → match SAM features
    #   Student layers 5-8  (blocks 9,11,13,15) → match DINOv2 features
    #   Student layers 9-12 (blocks 17,19,21,23) → match CLIP features
    distill_cfg=[
        # --- SAM: Early layers (structure) ---
        dict(
            name='loss_sam_l1',
            teacher_name='sam',
            student_feature_index=0,   # student block 1
            teacher_feature_index=0,   # SAM block 4
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_sam_l2',
            teacher_name='sam',
            student_feature_index=1,   # student block 3
            teacher_feature_index=1,   # SAM block 8
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_sam_l3',
            teacher_name='sam',
            student_feature_index=2,   # student block 5
            teacher_feature_index=2,   # SAM block 12
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_sam_l4',
            teacher_name='sam',
            student_feature_index=3,   # student block 7
            teacher_feature_index=3,   # SAM block 16
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),

        # --- DINOv2: Middle layers (semantics) ---
        dict(
            name='loss_dino_l1',
            teacher_name='dino',
            student_feature_index=4,   # student block 9
            teacher_feature_index=0,   # DINOv2 block 8
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_dino_l2',
            teacher_name='dino',
            student_feature_index=5,   # student block 11
            teacher_feature_index=1,   # DINOv2 block 12
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_dino_l3',
            teacher_name='dino',
            student_feature_index=6,   # student block 13
            teacher_feature_index=2,   # DINOv2 block 16
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_dino_l4',
            teacher_name='dino',
            student_feature_index=7,   # student block 15
            teacher_feature_index=3,   # DINOv2 block 20
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),

        # --- CLIP ViT-L/14: Deep layers (abstract) ---
        dict(
            name='loss_clip_l1',
            teacher_name='clip',
            student_feature_index=8,   # student block 17
            teacher_feature_index=0,   # CLIP block 6
            student_channels=1024,
            teacher_channels=1024,     # CLIP ViT-L/14 = 1024 dim
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_clip_l2',
            teacher_name='clip',
            student_feature_index=9,   # student block 19
            teacher_feature_index=1,   # CLIP block 12
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_clip_l3',
            teacher_name='clip',
            student_feature_index=10,  # student block 21
            teacher_feature_index=2,   # CLIP block 18
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
        dict(
            name='loss_clip_l4',
            teacher_name='clip',
            student_feature_index=11,  # student block 23
            teacher_feature_index=3,   # CLIP block 24
            student_channels=1024,
            teacher_channels=1024,
            loss_type='CosineSimilarity',
            loss_weight=1.0
        ),
    ]
)

# -------------------------------------------------------------------------
# 4. Override train pipeline
# -------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadPairedImagesFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PairedResize', scale=(640, 512), keep_ratio=True),
    dict(type='PairedRandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'img2_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'img_rgb'))
]

# -------------------------------------------------------------------------
# 5. DDP
# -------------------------------------------------------------------------
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

# -------------------------------------------------------------------------
# 6. Training Loop & Checkpointing
# -------------------------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

work_dir = './work_dirs/stage1_distill_v2'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,         # Save at epochs 4, 8, 12
        max_keep_ckpts=2,
        save_best=None),
    logger=dict(type='LoggerHook', interval=50))

# -------------------------------------------------------------------------
# 7. Batch Size & Learning Rate
# -------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        pipeline=train_pipeline,
        include_empty_images=True))

# Cosine annealing schedule for smooth convergence
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

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # 5e-6 for backbone (preserve pretrained)
        }),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
