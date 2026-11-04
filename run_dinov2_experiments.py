#!/usr/bin/env python3
"""Run DINOv2 experiments sequentially."""

import subprocess

experiments = [
    ('configs/kaist/dinov2/faster-rcnn_dinov2-base-p14_fpn_8xb2-12e_kaist.py', 'work_dirs/kaist_dinov2_frcnn'),
    ('configs/kaist/dinov2/dino-4scale_dinov2-base-p14_8xb2-12e_kaist.py', 'work_dirs/kaist_dinov2_dino_head'),
    ('configs/kaist/dinov2/dino-4scale_dinov2-base-p14-reg4_8xb2-12e_kaist.py', 'work_dirs/kaist_dinov2_reg_dino_head'),
]

for config, work_dir in experiments:
    print(f"\n{'='*80}")
    print(f"Running: {config}")
    print(f"{'='*80}\n")
    subprocess.run([
        'bash', './tools/dist_train.sh', config, '4', '--work-dir', work_dir
    ], check=True)
    print(f"\n✓ Completed: {work_dir}\n")

print("\n✓ All experiments done!")
