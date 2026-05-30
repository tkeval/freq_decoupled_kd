"""Test the Stage 1 ConcatDataset dataloader end-to-end.

Instantiates the actual KAISTDataset + pipeline (LoadPairedImagesFromFile,
PairedResize, PairedRandomFlip, PackDetInputs) for all three datasets
and visualizes samples from each, exactly as they would appear during training.

Usage:
    python tools/test_stage1_dataloader.py [--num-per-dataset 3] [--out-dir vis_flir]
"""

import argparse
import json
import os
import os.path as osp
import random
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    {
        'name': 'KAIST',
        'data_root': 'data/kaist-rgbt/',
        'ann_file': 'annotations/instancesonly_filtered_all-02_train.json',
        'img_prefix': 'images/',
    },
    {
        'name': 'FLIR Aligned',
        'data_root': 'data/flir_aligned/',
        'ann_file': 'annotations/flir_aligned_train.json',
        'img_prefix': '',
    },
    {
        'name': 'MFNet',
        'data_root': 'data/mfnet_ir_seg_dataset/',
        'ann_file': 'annotations/mfnet_paired_train.json',
        'img_prefix': '',
    },
]

TARGET_SIZE = (640, 512)  # (W, H)


def load_and_process_sample(data_root, img_prefix, img_info, flip=False):
    """Simulate the exact Stage 1 pipeline for one sample.

    Pipeline: LoadPairedImagesFromFile → PairedResize → PairedRandomFlip → PackDetInputs
    KAISTDataset.prepare_data swaps paths: IR→img_path, RGB→img2_path
    LoadPairedImagesFromFile: img_path→img (student IR), img2_path→img_rgb (teacher RGB)
    """
    rgb_file = img_info['file_name']
    ir_file = img_info['file_name2']

    rgb_path = osp.join(data_root, img_prefix, rgb_file)
    ir_path = osp.join(data_root, img_prefix, ir_file)

    # KAISTDataset.prepare_data swaps: IR→img_path, RGB→img2_path
    # Then LoadPairedImagesFromFile reads img_path as student, img2_path as teacher
    # Net effect: student sees IR, teacher sees RGB
    ir_img = cv2.imread(ir_path)
    rgb_img = cv2.imread(rgb_path)

    if ir_img is None:
        raise FileNotFoundError(f'IR image not found: {ir_path}')
    if rgb_img is None:
        raise FileNotFoundError(f'RGB image not found: {rgb_path}')

    # PairedResize: resize both to target size
    tw, th = TARGET_SIZE
    ir_img = cv2.resize(ir_img, (tw, th))
    rgb_img = cv2.resize(rgb_img, (tw, th))

    # PairedRandomFlip: same flip applied to both
    if flip:
        ir_img = cv2.flip(ir_img, 1)
        rgb_img = cv2.flip(rgb_img, 1)

    return ir_img, rgb_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-per-dataset', type=int, default=3)
    parser.add_argument('--out-dir', default='vis_flir')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    all_samples = []

    for ds in DATASETS:
        ann_path = osp.join(ds['data_root'], ds['ann_file'])
        if not osp.exists(ann_path):
            print(f"WARNING: {ann_path} not found, skipping {ds['name']}")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        n_images = len(coco['images'])
        n_anns = len(coco['annotations'])
        print(f"{ds['name']:15s}: {n_images:6d} images, {n_anns:6d} annotations")

        picks = random.sample(coco['images'], min(args.num_per_dataset, n_images))
        for img_info in picks:
            flip = random.random() < 0.5
            ir_img, rgb_img = load_and_process_sample(
                ds['data_root'], ds['img_prefix'], img_info, flip=flip)
            stem = osp.splitext(osp.basename(img_info['file_name']))[0]
            all_samples.append((ds['name'], stem, ir_img, rgb_img, flip))

    # Print combined stats
    total = sum(1 for ds in DATASETS if osp.exists(osp.join(ds['data_root'], ds['ann_file'])))
    print(f"\nConcatDataset will present all {total} datasets as one unified dataset.")
    print(f"DefaultSampler(shuffle=True) will interleave samples from all datasets.\n")

    # Visualize
    n = len(all_samples)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        'Stage 1 Dataloader — Actual Training View\n'
        'ConcatDataset(KAIST + FLIR + MFNet) with PairedResize + PairedRandomFlip',
        fontsize=14, y=1.01)

    colors = {'KAIST': '#2ecc71', 'FLIR Aligned': '#3498db', 'MFNet': '#e74c3c'}

    for i, (ds_name, stem, ir_img, rgb_img, flip) in enumerate(all_samples):
        flip_str = ' [flipped]' if flip else ''
        color = colors.get(ds_name, 'white')

        axes[i, 0].imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(
            f'Student (IR) — {ds_name}: {stem}{flip_str}',
            fontsize=10, color=color, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(
            f'Teacher (RGB) — {ds_name}: {stem}{flip_str}',
            fontsize=10, color=color, fontweight='bold')
        axes[i, 1].axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=n) for n, c in colors.items()]
    fig.legend(handles=legend_els, loc='upper right', fontsize=10,
              title='Dataset Source', title_fontsize=11)

    plt.tight_layout()
    save_path = osp.join(args.out_dir, 'stage1_concat_dataloader.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {save_path}')


if __name__ == '__main__':
    main()
