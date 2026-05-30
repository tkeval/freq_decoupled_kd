# usage: bash ./tools/dist_train.sh configs/kaist/distillation/response_kd_dino_lw-1.0_config.py 4 --work-dir work_dirs/response_kd_dino_lw-1.0_config
_base_ = './kd_dino-5scale_dinov2-large-p14-reg_8xb2-12e_kaist.py'

custom_imports = dict(imports=['mmdet.models.losses.distill_losses'], allow_failed_imports=False)

# In the original file, the loss_weight was 0.25 and the loss was KLDivLoss.
# Here we increase the weight to 1.0 and use the correct BCE-based loss.
distill_cfg = [
    dict(
        type='BCEWithLogitsDistillationLoss',
        name='loss_distill_cls',
        loss_weight=0.5,
        T=2.0),
    dict(
        type='L1Loss',
        name='loss_distill_bbox',
        loss_weight=0.5,
    )
]

# The base config now defines a complete distiller model.
# We only need to override the `distill_cfg` part of that model.
# We also add initialization from the teacher to boost performance.
model = dict(
    distill_cfg=distill_cfg,
    init_cfg=dict(type='Pretrained', checkpoint=teacher_checkpoint)
)
