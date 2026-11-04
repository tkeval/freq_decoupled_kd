# ------------------------------------------------------------
# DINOv2 with Registers + DINO detector on KAIST dataset
# Uses improved DINOv2-reg variant for better feature quality
# ------------------------------------------------------------
# usage: bash ./tools/dist_train.sh configs/kaist/dinov2/dino-4scale_dinov2-base-p14-reg4_8xb2-12e_kaist.py 4 --work-dir work_dirs/kaist_dinov2_reg_dino_head
_base_ = './dino-4scale_dinov2-base-p14_8xb2-12e_kaist.py'

# Only override the backbone to use register variant
model = dict(
    backbone=dict(
        model_name='vit_base_patch14_reg4_dinov2.lvd142m',  # DINOv2 with 4 register tokens
        # All other settings inherited from base config
    )
)

# Note: Register tokens improve feature quality and reduce artifacts
# Expected to give ~2-3% better mAP than standard DINOv2
