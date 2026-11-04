_base_ = [
    '../../../configs/kaist/dinov1/dino-4scale_r50_8xb2-12e_coco.py',
]
# usage: bash ./tools/dist_train.sh configs/kaist/dinov1/kaist_dino-5scale_swin-l_8xb2-12e_coco.py 4 --work-dir work_dirs/kaist_dino-5scale_swin-l_8xb2-12e_coco
# The official DINO-SwinL COCO checkpoint for fine-tuning
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))),
    bbox_head=dict(
        num_classes=1, # Override for KAIST dataset
    ))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        # CORRECTED: LR scaled for a new total batch size of 8
        lr=0.00005,
    ))
