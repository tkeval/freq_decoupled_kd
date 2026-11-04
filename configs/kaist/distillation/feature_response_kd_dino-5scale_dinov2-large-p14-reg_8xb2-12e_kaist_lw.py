_base_ = [
    './feature_response_kd_dino-5scale_dinov2-large-p14-reg_8xb2-12e_kaist.py'
]

distill_cfg = [
    dict(
        type='FeatLoss',
        loss_weight=0.1,
        student_channels=256,
        teacher_channels=256)
]

model = dict(
    type='FeatureResponseKDDINO',
    distill_cfg=distill_cfg,
)



