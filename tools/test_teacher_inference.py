"""Quick sanity check: does the RGB teacher produce high-confidence detections?

Usage:
    conda run -n mmpretrain_git python tools/test_teacher_inference.py
"""
import torch
import mmcv
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

register_all_modules()

# --- Build teacher model (same config as in selective_cross_modal_kd.py) ---
num_levels = 5
teacher_cfg = dict(
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

teacher_checkpoint = 'work_dirs/kaist_dinov2_large_reg_5scale/best_coco_person_precision_epoch_10.pth'

print("Building teacher model...")
model = MODELS.build(teacher_cfg)
print("Loading checkpoint...")
ckpt_info = load_checkpoint(model, teacher_checkpoint, map_location='cpu')
model.eval()
model.cuda()

# --- Load a sample RGB image from KAIST ---
import glob
import os

# Find a sample RGB image
rgb_dir = './data/kaist-rgbt/images/'
rgb_images = glob.glob(os.path.join(rgb_dir, '**/visible/*.jpg'), recursive=True)
if not rgb_images:
    rgb_images = glob.glob(os.path.join(rgb_dir, '**/*.jpg'), recursive=True)

if not rgb_images:
    print("ERROR: No RGB images found!")
    exit(1)

sample_path = rgb_images[0]
print(f"Loading sample image: {sample_path}")
img = mmcv.imread(sample_path, channel_order='bgr')
print(f"Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")

# Resize to match training pipeline
img = mmcv.imresize(img, (640, 512))
print(f"After resize: {img.shape}")

# Preprocess
img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
print(f"Tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")

# Use data_preprocessor
preprocessed = model.data_preprocessor({'inputs': [img_tensor]}, training=False)
batch_inputs = preprocessed['inputs'].cuda()
print(f"Preprocessed shape: {batch_inputs.shape}, dtype: {batch_inputs.dtype}")
print(f"Preprocessed range: [{batch_inputs.min():.2f}, {batch_inputs.max():.2f}]")

# Forward through backbone + neck
with torch.no_grad():
    feats = model.extract_feat(batch_inputs)
    print(f"Feature shapes: {[f.shape for f in feats]}")

    # We need dummy batch_data_samples for forward_transformer
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    ds = DetDataSample()
    ds.set_metainfo({
        'img_shape': (512, 640),
        'batch_input_shape': tuple(batch_inputs.shape[-2:]),
        'scale_factor': (1.0, 1.0),
        'ori_shape': (512, 640),
    })
    ds.gt_instances = InstanceData()
    ds.gt_instances.bboxes = torch.zeros((0, 4))
    ds.gt_instances.labels = torch.zeros((0,), dtype=torch.long)

    head_inputs = model.forward_transformer(feats, [ds])
    cls_scores, bbox_preds = model.bbox_head.forward(
        head_inputs['hidden_states'], head_inputs['references'])

    # Final layer predictions
    cls_final = cls_scores[-1]  # (1, 900, 1)
    bbox_final = bbox_preds[-1]  # (1, 900, 4)

    scores = cls_final[0].sigmoid()
    print(f"\n=== Teacher Prediction Stats ===")
    print(f"Cls logits range: [{cls_final.min():.4f}, {cls_final.max():.4f}]")
    print(f"Sigmoid scores range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Top-10 scores: {scores.squeeze().topk(10).values.tolist()}")
    print(f"Scores > 0.5: {(scores > 0.5).sum().item()}")
    print(f"Scores > 0.3: {(scores > 0.3).sum().item()}")
    print(f"Scores > 0.1: {(scores > 0.1).sum().item()}")
    print(f"Scores > 0.05: {(scores > 0.05).sum().item()}")
    print(f"BBox range: [{bbox_final.min():.4f}, {bbox_final.max():.4f}]")
