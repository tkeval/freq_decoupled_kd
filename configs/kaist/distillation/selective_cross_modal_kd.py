# -------------------------------------------------------------------------
# GT-Mediated Cross-Modal KD: RGB Teacher → IR Student
# -------------------------------------------------------------------------
# Key insight: In two-stage DINO, query indices are NOT aligned between
# teacher (RGB) and student (IR) because queries come from encoder top-k
# proposals which are data-dependent. Index-wise response KD compares
# misaligned queries → noise, not signal.
#
# Solution: GT-mediated query matching. For each GT object:
#   1. Find the teacher query with highest IoU to this GT
#   2. Find the student query with highest IoU to this GT
#   3. Distill only between these semantically matched pairs
#
# Usage:
#   bash ./tools/dist_train.sh \
#       configs/kaist/distillation/selective_cross_modal_kd.py 4 \
#       --work-dir ./work_dirs/gt_mediated_kd
# -------------------------------------------------------------------------
_base_ = [
    '../../_base_/datasets/kaist_paired_detection.py',
    '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmpretrain.models', 'mmdet.models.distillers'],
    allow_failed_imports=False)

# -------------------------------------------------------------------------
# 1. RGB Teacher Checkpoint
# -------------------------------------------------------------------------
teacher_checkpoint = 'work_dirs/kaist_dinov2_large_reg_5scale/best_coco_person_precision_epoch_10.pth'

# -------------------------------------------------------------------------
# 2. Shared DINO architecture (ViT-Large, same for teacher and student)
# -------------------------------------------------------------------------
num_levels = 5

dino_model_cfg = dict(
    type='DINO',
    num_feature_levels=num_levels,
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14),
    backbone=dict(
        type='mmpretrain.TIMMBackbone',
        model_name='vit_large_patch14_reg4_dinov2.lvd142m',
        pretrained=True,
        features_only=True,
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
        num_outs=num_levels),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_levels=num_levels,
                num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0,
            alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(
            dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
    test_cfg=dict(max_per_img=100))

# -------------------------------------------------------------------------
# 3. Student: SelectiveCrossModalKD
# -------------------------------------------------------------------------
model = dino_model_cfg.copy()
model['type'] = 'SelectiveCrossModalKD'
model['teacher_cfg'] = dino_model_cfg
model['teacher_checkpoint'] = teacher_checkpoint
model['distill_cfg'] = dict(
    loss_weight_feat=0.01,
    loss_weight_cls=0.1,
    loss_weight_bbox=0.1,
    temperature=2.0,
    teacher_iou_threshold=0.3,  # Only teacher must overlap GT; no student threshold
)

# -------------------------------------------------------------------------
# 4. Optimizer (matching ViT-Large IR baseline settings)
# -------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
            'teacher': dict(lr_mult=0.0),
        }),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001,
         by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12,
         by_epoch=True, milestones=[11], gamma=0.1)]

# -------------------------------------------------------------------------
# 5. Training
# -------------------------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=2, num_workers=4)
test_dataloader = dict(batch_size=2, num_workers=4)

# -------------------------------------------------------------------------
# 6. DDP + Checkpointing
# -------------------------------------------------------------------------
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

work_dir = './work_dirs/gt_mediated_kd'
