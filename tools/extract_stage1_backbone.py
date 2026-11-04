"""
Extract the student backbone weights from a Stage 1 checkpoint
and save them as a standalone checkpoint that can be loaded
directly into a DINO detector via load_from.

Automatically detects and merges LoRA weights if present.

Usage:
    python tools/extract_stage1_backbone.py  ./work_dirs/stage1_distill_v2_lora/epoch_11.pth ./work_dirs/stage1_distill_v2_lora/stage1_epoch11.pth

    # Override LoRA scaling (default: 1.0, i.e. alpha == rank):
    python tools/extract_stage1_backbone.py \\
        /path/to/stage1_lora/epoch_12.pth \\
        /path/to/output/stage1_backbone.pth \\
        --lora-scaling 1.0

The output checkpoint can be used with:
    load_from = '/path/to/output/stage1_backbone.pth'
in any detector config.
"""
import argparse
import sys
import os
import torch

# Add project root to path so we can import mmdet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mmdet.models.utils.lora import merge_lora_state_dict


def main():
    parser = argparse.ArgumentParser(
        description='Extract Stage 1 backbone (auto-merges LoRA if present)')
    parser.add_argument('input', help='Stage 1 checkpoint path')
    parser.add_argument('output', help='Output backbone checkpoint path')
    parser.add_argument('--lora-scaling', type=float, default=1.0,
                        help='LoRA scaling factor (alpha/rank). '
                             'Default 1.0 assumes alpha == rank.')
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location='cpu', weights_only=False, mmap=True)
    state_dict = ckpt.get('state_dict', ckpt)

    # Stage 1 keys: "student.backbone.xxx" → "backbone.xxx"
    backbone_dict = {}
    prefix = 'student.backbone.'
    for key, value in state_dict.items():
        if key.startswith(prefix):
            backbone_dict[key[len(prefix):]] = value

    if not backbone_dict:
        prefixes = set(k.split('.')[0] for k in state_dict.keys())
        raise RuntimeError(
            f"No keys with prefix '{prefix}' found. "
            f"Available top-level prefixes: {prefixes}")

    # Merge LoRA weights if present
    backbone_dict = merge_lora_state_dict(
        backbone_dict, scaling=args.lora_scaling)

    # Add 'backbone.' prefix for detector loading
    new_state_dict = {f'backbone.{k}': v for k, v in backbone_dict.items()}

    print(f"Extracted {len(new_state_dict)} backbone keys")
    print(f"Sample keys: {list(new_state_dict.keys())[:5]}")

    output_ckpt = {'state_dict': new_state_dict}
    torch.save(output_ckpt, args.output)
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
