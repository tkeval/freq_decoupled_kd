"""Convert FLIR Aligned dataset from VOC XML to COCO JSON format.

Produces a COCO-style JSON with paired image paths (file_name / file_name2)
matching the convention used by KAISTDataset:
  - file_name  : RGB image   (e.g. JPEGImages/FLIR_00002_RGB.jpg)
  - file_name2 : IR/thermal  (e.g. JPEGImages/FLIR_00002_PreviewData.jpeg)

Usage:
    python tools/dataset_converters/flir_voc_to_coco.py \
        --data-root data/flir_aligned \
        --out-dir data/flir_aligned/annotations
"""

import argparse
import json
import os
import os.path as osp
import xml.etree.ElementTree as ET


FLIR_CLASSES = ('person', 'car', 'bicycle', 'dog')

CLASS_TO_ID = {name: i for i, name in enumerate(FLIR_CLASSES)}


def parse_voc_xml(xml_path):
    """Parse a single FLIR VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find('size')
    width = int(size_el.find('width').text)
    height = int(size_el.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_TO_ID:
            continue
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
        objects.append(dict(
            name=name,
            bbox=[xmin, ymin, w, h],
            area=w * h,
            difficult=difficult,
        ))
    return width, height, objects


def convert_split(data_root, split_file, out_path):
    """Convert one split (train or val) to COCO JSON."""
    split_path = osp.join(data_root, split_file)
    with open(split_path) as f:
        stems = [line.strip() for line in f if line.strip()]

    categories = [dict(id=i, name=name) for i, name in enumerate(FLIR_CLASSES)]
    images = []
    annotations = []
    ann_id = 0

    for img_id, stem in enumerate(stems):
        xml_name = f'{stem}.xml'
        xml_path = osp.join(data_root, 'Annotations', xml_name)

        if not osp.exists(xml_path):
            print(f'Warning: annotation not found for {stem}, skipping')
            continue

        width, height, objects = parse_voc_xml(xml_path)

        # Derive paired filenames from the stem.
        # stem example: FLIR_00002_PreviewData
        base_id = stem.replace('_PreviewData', '')  # e.g. FLIR_00002
        rgb_filename = f'JPEGImages/{base_id}_RGB.jpg'
        ir_filename = f'JPEGImages/{base_id}_PreviewData.jpeg'

        images.append(dict(
            id=img_id,
            file_name=rgb_filename,
            file_name2=ir_filename,
            width=width,
            height=height,
        ))

        for obj in objects:
            annotations.append(dict(
                id=ann_id,
                image_id=img_id,
                category_id=CLASS_TO_ID[obj['name']],
                bbox=obj['bbox'],
                area=obj['area'],
                iscrowd=0,
                ignore=bool(obj['difficult']),
            ))
            ann_id += 1

    coco = dict(
        images=images,
        categories=categories,
        annotations=annotations,
    )

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(coco, f)

    n_with_ann = sum(1 for img in images
                     if any(a['image_id'] == img['id'] for a in annotations))
    print(f'Saved {out_path}')
    print(f'  images: {len(images)}  (with annotations: {n_with_ann})')
    print(f'  annotations: {len(annotations)}')


def main():
    parser = argparse.ArgumentParser(description='FLIR VOC XML → COCO JSON')
    parser.add_argument('--data-root', default='data/flir_aligned',
                        help='Root directory of flir_aligned dataset')
    parser.add_argument('--out-dir', default=None,
                        help='Output dir for JSON files (default: <data-root>/annotations)')
    args = parser.parse_args()

    out_dir = args.out_dir or osp.join(args.data_root, 'annotations')

    convert_split(
        args.data_root,
        'align_train.txt',
        osp.join(out_dir, 'flir_aligned_train.json'),
    )
    convert_split(
        args.data_root,
        'align_validation.txt',
        osp.join(out_dir, 'flir_aligned_val.json'),
    )


if __name__ == '__main__':
    main()
