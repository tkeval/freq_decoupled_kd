"""
Linear Probing for Semantic Segmentation on MFNet IR Dataset.

Freezes a DINOv2 ViT-Large backbone and trains only a lightweight
segmentation head to measure backbone feature quality.

Usage:
  # Baseline: pretrained DINOv2 (ImageNet)
  python tools/eval_seg_linear_probe.py --pretrained

  # FFT KD: Stage 2 checkpoint
  python tools/eval_seg_linear_probe.py \
      --checkpoint /path/to/stage2/best_checkpoint.pth

  # Custom options
  python tools/eval_seg_linear_probe.py --pretrained --epochs 30 --lr 1e-3
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -------------------------------------------------------------------------
# MFNet Dataset
# -------------------------------------------------------------------------
CLASSES = [
    'unlabeled', 'car', 'person', 'bike', 'curve',
    'car_stop', 'guardrail', 'color_cone', 'bump'
]
NUM_CLASSES = 9  # including unlabeled (0)
IGNORE_INDEX = 255  # only truly invalid pixels (none in MFNet)
MIOU_IGNORE = 0     # exclude 'unlabeled' (background) from mIoU, but train on it


class MFNetSegDataset(Dataset):
    """MFNet IR segmentation dataset."""

    def __init__(self, data_root, split='train', img_size=(480, 640)):
        self.data_root = data_root
        self.img_size = img_size  # (H, W)

        split_file = os.path.join(data_root, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.filenames = [
                line.strip() for line in f
                if line.strip() and '_flip' not in line.strip()]

        # ImageNet normalization
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.data_root, 'ir', f'{name}.png')
        lbl_path = os.path.join(self.data_root, 'labels', f'{name}.png')

        img = np.array(Image.open(img_path))  # (H, W, 3) BGR uint8
        lbl = np.array(Image.open(lbl_path))  # (H, W) uint8

        # Normalize: BGR→RGB, then ImageNet normalization
        img = img[:, :, ::-1].copy()  # BGR→RGB
        img = (img.astype(np.float32) - self.mean) / self.std

        # HWC → CHW
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


# -------------------------------------------------------------------------
# Linear Segmentation Head
# -------------------------------------------------------------------------
class LinearSegHead(nn.Module):
    """Simple linear segmentation head for multi-scale features.

    For each feature level: 1×1 conv → upsample to input resolution.
    Then concatenate all levels and classify with a final 1×1 conv.
    """

    def __init__(self, in_channels_list, num_classes, img_size=(480, 640)):
        super().__init__()
        self.img_size = img_size
        embed_dim = 256

        # Per-level projection: in_ch → embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Final classifier: (embed_dim * num_levels) → num_classes
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, features):
        """
        Args:
            features: list of (B, C, H_i, W_i) feature tensors.
        Returns:
            logits: (B, num_classes, H, W) at input resolution.
        """
        H, W = self.img_size
        projected = []
        for feat, proj in zip(features, self.projections):
            x = proj(feat)
            x = F.interpolate(x, size=(H, W), mode='bilinear',
                              align_corners=False)
            projected.append(x)

        fused = torch.cat(projected, dim=1)
        return self.classifier(fused)


# -------------------------------------------------------------------------
# Backbone loading
# -------------------------------------------------------------------------
def build_backbone(pretrained=True):
    """Build DINOv2 ViT-Large backbone via mmpretrain's TIMMBackbone."""
    from mmpretrain.models import build_backbone as mmpretrain_build_backbone

    cfg = dict(
        type='mmpretrain.TIMMBackbone',
        model_name='vit_large_patch14_reg4_dinov2.lvd142m',
        pretrained=pretrained,
        features_only=True,
        out_indices=(7, 15, 19, 21, 23),
        dynamic_img_size=True,
    )
    backbone = mmpretrain_build_backbone(cfg)
    return backbone


def load_checkpoint_backbone(backbone, checkpoint_path):
    """Load backbone weights from Stage 1 or Stage 2 checkpoint.

    Handles three key formats:
      - Stage 1: 'student.backbone.xxx'   (FreqDecoupledDistiller)
      - Stage 2: 'backbone.xxx'           (DINO / Faster R-CNN detector)
      - Stage 2 wrapped: 'detector.backbone.xxx' (Stage2GuidedDetector)
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    backbone_weights = {}
    for prefix in ['student.backbone.', 'backbone.', 'detector.backbone.']:
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                if new_key not in backbone_weights:
                    backbone_weights[new_key] = v
        if backbone_weights:
            print(f"  Found backbone weights under prefix '{prefix}'")
            break

    if not backbone_weights:
        available = sorted(set(k.split('.')[0] for k in state_dict.keys()))
        raise RuntimeError(
            f"No backbone weights found in '{checkpoint_path}'.\n"
            f"Top-level keys: {available}")

    missing, unexpected = backbone.load_state_dict(
        backbone_weights, strict=False)
    print(f"Loaded {len(backbone_weights)} backbone weights from checkpoint")
    if missing:
        print(f"  Missing: {len(missing)} keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)} keys")


def pad_to_patch_size(img, patch_size=14):
    """Pad image tensor to be divisible by patch_size."""
    _, _, H, W = img.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return img


# -------------------------------------------------------------------------
# Evaluation: mIoU
# -------------------------------------------------------------------------
def compute_miou(pred, target, num_classes, ignore_index=0):
    """Compute per-class IoU and mIoU."""
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue  # Class not present in this batch
        ious.append(intersection / union)
    return ious


class IoUAccumulator:
    """Accumulate intersection and union across batches for accurate mIoU."""

    def __init__(self, num_classes, ignore_class=0):
        self.num_classes = num_classes
        self.ignore_class = ignore_class  # Exclude from mIoU (background)
        self.intersection = np.zeros(num_classes)
        self.union = np.zeros(num_classes)

    def update(self, pred, target):
        """pred, target: numpy arrays of shape (H, W) or (B, H, W)."""
        for cls in range(self.num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            self.intersection[cls] += (pred_mask & target_mask).sum()
            self.union[cls] += (pred_mask | target_mask).sum()

    def get_miou(self):
        ious = {}
        valid_ious = []
        for cls in range(self.num_classes):
            if self.union[cls] == 0:
                ious[CLASSES[cls]] = float('nan')
            else:
                iou = self.intersection[cls] / self.union[cls]
                ious[CLASSES[cls]] = iou
                if cls != self.ignore_class:
                    valid_ious.append(iou)
        miou = np.mean(valid_ious) if valid_ious else 0.0
        return miou, ious


# -------------------------------------------------------------------------
# Training & Evaluation
# -------------------------------------------------------------------------
def train_one_epoch(backbone, seg_head, dataloader, optimizer, device):
    seg_head.train()
    total_loss = 0
    num_batches = 0

    for imgs, lbls in dataloader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        # Pad to DINOv2 patch size
        imgs_padded = pad_to_patch_size(imgs, patch_size=14)

        with torch.no_grad():
            features = backbone(imgs_padded)

        logits = seg_head(features)
        loss = F.cross_entropy(logits, lbls, ignore_index=IGNORE_INDEX)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(backbone, seg_head, dataloader, device):
    seg_head.eval()
    accumulator = IoUAccumulator(NUM_CLASSES, ignore_class=MIOU_IGNORE)

    for imgs, lbls in dataloader:
        imgs = imgs.to(device)
        imgs_padded = pad_to_patch_size(imgs, patch_size=14)

        features = backbone(imgs_padded)
        logits = seg_head(features)

        preds = logits.argmax(dim=1).cpu().numpy()
        targets = lbls.numpy()
        accumulator.update(preds, targets)

    miou, per_class = accumulator.get_miou()
    return miou, per_class


def main():
    parser = argparse.ArgumentParser(
        description='Linear probing for segmentation on MFNet IR')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained DINOv2 (baseline)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to Stage 2 checkpoint for backbone')
    parser.add_argument('--data-root', type=str,
                        default='datasets/mfnet_ir_seg_dataset',
                        help='MFNet dataset root')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--eval-split', type=str, default='test',
                        choices=['val', 'test'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if not args.pretrained and args.checkpoint is None:
        parser.error("Must specify either --pretrained or --checkpoint")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    # ----- Build backbone -----
    print("Building DINOv2 ViT-Large backbone...")
    backbone = build_backbone(pretrained=True)  # Always load pretrained first

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint_backbone(backbone, args.checkpoint)
    else:
        print("Using pretrained DINOv2 (ImageNet baseline)")

    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # ----- Probe feature dimensions -----
    print("Probing feature dimensions...")
    with torch.no_grad():
        dummy = torch.randn(1, 3, 490, 644).to(device)  # 480→490, 640→644 (mult of 14)
        feats = backbone(dummy)
        in_channels = [f.shape[1] for f in feats]
        print(f"  Feature levels: {len(feats)}")
        for i, f in enumerate(feats):
            print(f"  Level {i}: {f.shape}")
    del dummy

    # ----- Build segmentation head -----
    seg_head = LinearSegHead(
        in_channels_list=in_channels,
        num_classes=NUM_CLASSES,
        img_size=(480, 640)
    ).to(device)

    num_params = sum(p.numel() for p in seg_head.parameters())
    print(f"Segmentation head params: {num_params:,}")

    # ----- Datasets -----
    train_dataset = MFNetSegDataset(args.data_root, split='train')
    eval_dataset = MFNetSegDataset(args.data_root, split=args.eval_split)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} images, "
          f"Eval ({args.eval_split}): {len(eval_dataset)} images")

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        seg_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ----- Training loop -----
    best_miou = 0
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(backbone, seg_head, train_loader,
                               optimizer, device)
        miou, per_class = evaluate(backbone, seg_head, eval_loader, device)
        scheduler.step()

        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        lr = optimizer.param_groups[0]['lr']
        marker = " *" if is_best else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss: {loss:.4f} | lr: {lr:.2e} | "
              f"mIoU: {miou:.4f}{marker}")

    # ----- Final results -----
    print("-" * 70)
    print(f"\nBest mIoU: {best_miou:.4f}")

    # Run final eval and print per-class results
    miou, per_class = evaluate(backbone, seg_head, eval_loader, device)
    print(f"\nFinal mIoU: {miou:.4f}")
    print(f"\nPer-class IoU:")
    print(f"  {'Class':<15} {'IoU':>8}")
    print(f"  {'-'*24}")
    for cls_name, iou in per_class.items():
        if np.isnan(iou):
            print(f"  {cls_name:<15} {'N/A':>8}")
        else:
            print(f"  {cls_name:<15} {iou:>8.4f}")

    # ----- Summary -----
    mode = "pretrained (baseline)" if args.pretrained and not args.checkpoint \
        else f"checkpoint: {os.path.basename(args.checkpoint)}"
    print(f"\n{'='*70}")
    print(f"Backbone: {mode}")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
