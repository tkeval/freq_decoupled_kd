"""Convert MFNet dataset for cross-modal distillation.

MFNet stores paired RGB+thermal data as 4-channel RGBA PNGs:
  - Channels 0-2: RGB (visible)
  - Channel 3:    Thermal / IR (single-channel, saved as 3-channel for pipeline)

This script:
  1. Splits the 4-channel images into separate RGB and IR files
  2. Generates a COCO-style JSON (file_name / file_name2) compatible with
     KAISTDataset, with empty annotations (Stage 1 only needs paired images)

Output layout:
    data/mfnet_ir_seg_dataset/
      rgb/         <-- extracted RGB images  (XXXXX.png)
      ir/          <-- extracted IR images   (XXXXX.png, 3-channel grayscale)
      annotations/ <-- COCO JSON files

Usage:
    python tools/dataset_converters/mfnet_to_paired_coco.py \
        --data-root data/mfnet_ir_seg_dataset
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
from PIL import Image


def split_images(data_root, stems):
    """Split 4-channel RGBA PNGs into separate RGB and IR directories."""
    src_dir = osp.join(data_root, 'images')
    rgb_dir = osp.join(data_root, 'rgb')
    ir_dir = osp.join(data_root, 'ir')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)

    skipped = 0
    processed = 0
    for stem in stems:
        src_path = osp.join(src_dir, f'{stem}.png')
        rgb_path = osp.join(rgb_dir, f'{stem}.png')
        ir_path = osp.join(ir_dir, f'{stem}.png')

        if osp.exists(rgb_path) and osp.exists(ir_path):
            skipped += 1
            continue

        if not osp.exists(src_path):
            print(f'  Warning: {src_path} not found, skipping')
            continue

        img = np.array(Image.open(src_path))  # (H, W, 4) RGBA
        if img.ndim != 3 or img.shape[2] < 4:
            print(f'  Warning: {src_path} is not 4-channel ({img.shape}), skipping')
            continue

        rgb = img[:, :, :3]
        thermal = img[:, :, 3]

        Image.fromarray(rgb).save(rgb_path)
        # Save thermal as 3-channel (grayscale replicated) for pipeline compatibility
        thermal_3ch = np.stack([thermal, thermal, thermal], axis=2)
        Image.fromarray(thermal_3ch).save(ir_path)
        processed += 1

    print(f'  Split images: {processed} new, {skipped} already exist')


def build_coco_json(data_root, stems, out_path):
    """Build a COCO-format JSON with paired paths and empty annotations."""
    images = []
    for img_id, stem in enumerate(stems):
        src_path = osp.join(data_root, 'images', f'{stem}.png')
        if not osp.exists(src_path):
            continue
        img = Image.open(src_path)
        w, h = img.size

        images.append(dict(
            id=img_id,
            file_name=f'rgb/{stem}.png',
            file_name2=f'ir/{stem}.png',
            width=w,
            height=h,
        ))

    coco = dict(
        images=images,
        categories=[],
        annotations=[],
    )

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(coco, f)

    print(f'  Saved {out_path}  ({len(images)} images)')


def read_split(data_root, split_file):
    """Read split file, filtering out _flip augmented entries."""
    path = osp.join(data_root, split_file)
    with open(path) as f:
        stems = [line.strip() for line in f if line.strip()]
    # Filter out flipped augmentations (only use originals)
    originals = [s for s in stems if '_flip' not in s]
    return originals


def main():
    parser = argparse.ArgumentParser(description='MFNet → paired RGB/IR + COCO JSON')
    parser.add_argument('--data-root', default='data/mfnet_ir_seg_dataset')
    args = parser.parse_args()

    ann_dir = osp.join(args.data_root, 'annotations')

    # Read blacklist
    bl_path = osp.join(args.data_root, 'black_list.txt')
    blacklist = set()
    if osp.exists(bl_path):
        with open(bl_path) as f:
            blacklist = {line.strip() for line in f if line.strip()}
        print(f'Blacklist: {len(blacklist)} images')

    for split_name, split_file in [('train', 'train.txt'), ('val', 'val.txt'), ('test', 'test.txt')]:
        print(f'\n--- {split_name} ---')
        stems = read_split(args.data_root, split_file)
        stems = [s for s in stems if s not in blacklist]
        print(f'  Stems (no flip, no blacklist): {len(stems)}')

        split_images(args.data_root, stems)
        build_coco_json(
            args.data_root, stems,
            osp.join(ann_dir, f'mfnet_paired_{split_name}.json'),
        )

    # Also build an "all" set (train + val + test, no duplicates)
    all_stems = set()
    for sf in ['train.txt', 'val.txt', 'test.txt']:
        all_stems.update(read_split(args.data_root, sf))
    all_stems -= blacklist
    all_stems = sorted(all_stems)
    print(f'\n--- all (combined) ---')
    print(f'  Stems: {len(all_stems)}')
    split_images(args.data_root, all_stems)
    build_coco_json(
        args.data_root, all_stems,
        osp.join(ann_dir, 'mfnet_paired_all.json'),
    )


if __name__ == '__main__':
    main()
