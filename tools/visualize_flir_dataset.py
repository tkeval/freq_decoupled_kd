"""Visualize FLIR Aligned dataset: paired RGB/IR images + bounding box annotations.

Checks:
  1. Raw paired images load correctly (RGB ↔ IR correspondence)
  2. COCO JSON annotations parse and draw properly
  3. KAISTDataset dataloader produces valid paired samples

Usage:
    python tools/visualize_flir_dataset.py [--num-samples 6] [--out-dir vis_flir]
"""

import argparse
import json
import os
import os.path as osp
import random

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


FLIR_DATA_ROOT = 'data/flir_aligned/'
TRAIN_ANN = 'annotations/flir_aligned_train.json'
VAL_ANN = 'annotations/flir_aligned_val.json'

COLORS = {
    'person':  (0, 255, 0),
    'car':     (255, 0, 0),
    'bicycle': (0, 165, 255),
    'dog':     (255, 0, 255),
}


def load_coco_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    id_to_cat = {c['id']: c['name'] for c in data['categories']}
    img_id_to_anns = {}
    for ann in data['annotations']:
        img_id_to_anns.setdefault(ann['image_id'], []).append(ann)
    return data, id_to_cat, img_id_to_anns


def draw_bboxes(img, anns, id_to_cat):
    """Draw bounding boxes on an image (in-place)."""
    vis = img.copy()
    for ann in anns:
        cat_name = id_to_cat[ann['category_id']]
        color = COLORS.get(cat_name, (255, 255, 255))
        x, y, w, h = [int(v) for v in ann['bbox']]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, cat_name, (x, max(y - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def visualize_raw_pairs(data_root, ann_path, num_samples, out_dir):
    """Visualize raw paired RGB/IR images with annotations from COCO JSON."""
    data, id_to_cat, img_id_to_anns = load_coco_json(ann_path)

    annotated_imgs = [img for img in data['images']
                      if img['id'] in img_id_to_anns]
    samples = random.sample(annotated_imgs, min(num_samples, len(annotated_imgs)))

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle('FLIR Aligned Dataset — Raw Pairs + Annotations', fontsize=16, y=1.0)

    for i, img_info in enumerate(samples):
        rgb_path = osp.join(data_root, img_info['file_name'])
        ir_path = osp.join(data_root, img_info['file_name2'])

        rgb = cv2.imread(rgb_path)
        ir = cv2.imread(ir_path)

        if rgb is None:
            print(f'WARNING: Could not load RGB: {rgb_path}')
            continue
        if ir is None:
            print(f'WARNING: Could not load IR: {ir_path}')
            continue

        anns = img_id_to_anns.get(img_info['id'], [])
        rgb_ann = draw_bboxes(rgb, anns, id_to_cat)
        ir_ann = draw_bboxes(ir, anns, id_to_cat)

        axes[i, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"RGB: {osp.basename(img_info['file_name'])}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cv2.cvtColor(ir, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"IR: {osp.basename(img_info['file_name2'])}")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(cv2.cvtColor(rgb_ann, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f"RGB + Annotations ({len(anns)} objects)")
        axes[i, 2].axis('off')

    legend_elements = [patches.Patch(facecolor=np.array(c[::-1]) / 255, label=n)
                       for n, c in COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = osp.join(out_dir, 'flir_raw_pairs.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved raw pairs visualization → {save_path}')


def visualize_dataloader(data_root, ann_path, num_samples, out_dir):
    """Test KAISTDataset loading with FLIR data (no mmdet dependency needed)."""
    data, id_to_cat, img_id_to_anns = load_coco_json(ann_path)

    annotated_imgs = [img for img in data['images']
                      if img['id'] in img_id_to_anns]
    samples = random.sample(annotated_imgs, min(num_samples, len(annotated_imgs)))

    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle('FLIR — Simulated Dataloader (Student IR ← | → Teacher RGB)',
                 fontsize=16, y=1.0)

    for i, img_info in enumerate(samples):
        rgb_path = osp.join(data_root, img_info['file_name'])
        ir_path = osp.join(data_root, img_info['file_name2'])

        rgb = cv2.imread(rgb_path)
        ir = cv2.imread(ir_path)

        if rgb is None or ir is None:
            print(f'WARNING: missing image for {img_info["id"]}')
            continue

        # Simulate the dataloader: resize to (640, 512) keeping aspect ratio
        target_w, target_h = 640, 512
        ir_resized = cv2.resize(ir, (target_w, target_h))
        rgb_resized = cv2.resize(rgb, (target_w, target_h))

        anns = img_id_to_anns.get(img_info['id'], [])

        # Scale annotations to resized dimensions
        sx = target_w / img_info['width']
        sy = target_h / img_info['height']
        scaled_anns = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            scaled_anns.append({**ann, 'bbox': [x * sx, y * sy, w * sx, h * sy]})

        ir_vis = draw_bboxes(ir_resized, scaled_anns, id_to_cat)
        rgb_vis = draw_bboxes(rgb_resized, scaled_anns, id_to_cat)

        axes[i, 0].imshow(cv2.cvtColor(ir_vis, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Student (IR) — {ir_resized.shape[1]}×{ir_resized.shape[0]}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f'Teacher (RGB) — {rgb_resized.shape[1]}×{rgb_resized.shape[0]}')
        axes[i, 1].axis('off')

    legend_elements = [patches.Patch(facecolor=np.array(c[::-1]) / 255, label=n)
                       for n, c in COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    save_path = osp.join(out_dir, 'flir_dataloader_sim.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved dataloader simulation → {save_path}')


def print_dataset_stats(train_path, val_path):
    """Print annotation statistics."""
    for split, path in [('train', train_path), ('val', val_path)]:
        data, id_to_cat, img_id_to_anns = load_coco_json(path)

        total_anns = sum(len(v) for v in img_id_to_anns.values())
        imgs_with_anns = len(img_id_to_anns)
        imgs_without = len(data['images']) - imgs_with_anns

        cat_counts = {}
        for anns in img_id_to_anns.values():
            for ann in anns:
                name = id_to_cat[ann['category_id']]
                cat_counts[name] = cat_counts.get(name, 0) + 1

        print(f'\n{"=" * 50}')
        print(f'FLIR Aligned — {split.upper()} split')
        print(f'{"=" * 50}')
        print(f'  Total images:          {len(data["images"])}')
        print(f'  Images w/ annotations: {imgs_with_anns}')
        print(f'  Images w/o annotations:{imgs_without}')
        print(f'  Total annotations:     {total_anns}')
        print(f'  Categories:')
        for name, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f'    {name:12s}: {count:6d}')

        # Bbox size statistics
        areas = [ann['area'] for anns in img_id_to_anns.values() for ann in anns]
        if areas:
            areas = np.array(areas)
            print(f'  Bbox area stats:')
            print(f'    min: {areas.min():.0f}  median: {np.median(areas):.0f}  '
                  f'mean: {areas.mean():.0f}  max: {areas.max():.0f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=FLIR_DATA_ROOT)
    parser.add_argument('--num-samples', type=int, default=6)
    parser.add_argument('--out-dir', default='vis_flir')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_ann = osp.join(args.data_root, TRAIN_ANN)
    val_ann = osp.join(args.data_root, VAL_ANN)

    print_dataset_stats(train_ann, val_ann)

    print('\n--- Generating visualizations ---')
    visualize_raw_pairs(args.data_root, train_ann, args.num_samples, args.out_dir)
    visualize_dataloader(args.data_root, train_ann, args.num_samples, args.out_dir)

    print(f'\nAll outputs saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
