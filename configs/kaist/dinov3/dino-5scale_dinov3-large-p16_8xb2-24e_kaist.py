# ------------------------------------------------------------
# DINOv3-LARGE + 5-scale DINO detector on KAIST (24 epochs)
# Extended training for better convergence
# ------------------------------------------------------------
# usage: bash ./tools/dist_train.sh configs/kaist/dinov3/dino-5scale_dinov3-large-p16_8xb2-24e_kaist.py 4 --work-dir work_dirs/kaist_dinov3_large_5scale_24ep
_base_ = './dino-5scale_dinov3-large-p16_8xb2-12e_kaist.py'

# Override training configuration for 24 epochs
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,  # Extended from 12 to 24
    val_interval=1)

# Update learning rate schedule for 24 epochs
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,  # Warmup from 5e-8 to 5e-5
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[18, 22],  # Drop LR at epoch 18 and 22
        gamma=0.1)
]

# Expected performance with 24 epochs:
# - Epoch 1:  mAP@50 ≈ 35-40%
# - Epoch 6:  mAP@50 ≈ 55-60%
# - Epoch 12: mAP@50 ≈ 67-70%
# - Epoch 18: mAP@50 ≈ 69-72% (after LR drop)
# - Epoch 24: mAP@50 ≈ 70-73% (fully converged)
#
# This should exceed the DINOv1 + Swin-L baseline of 68.5%


