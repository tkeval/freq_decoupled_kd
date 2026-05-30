"""
CKA Analysis (Table 6) — teacher(RGB) vs student(IR) representational similarity.

Computes linear CKA between the frozen RGB DINOv2 teacher and an IR student
backbone on 500 KAIST val pairs, reported on the FULL feature map and on the
LOW / HIGH frequency bands separately (band split matches the model's
FreqDecoupledDistiller: rectangular centred mask, cutoff r_c).

Run once per backbone:

  # Row 1: No-KD baseline (student = pretrained DINOv2 fed IR)
  python tools/cka_analysis.py --label "No KD" --n 500 --gpu 0

  # Row 2: Uniform feature KD
  python tools/cka_analysis.py --label "Uniform KD" \
      --checkpoint /path/to/uniform_kd/epoch_12.pth --n 500 --gpu 0

  # Row 3: FreqKD
  python tools/cka_analysis.py --label "FreqKD" \
      --checkpoint ./work_dirs/stage1/fft/stage1_freq_decoupled/epoch_12.pth \
      --n 500 --gpu 0

GPU: 1 device, < 8 GB, a few minutes for 500 pairs.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Make the local mmdet package importable when run directly (no dist wrapper)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = './data/kaist-rgbt'
VAL_JSON  = f'{DATA_ROOT}/annotations/instancesonly_filtered_all-20_val.json'
IMG_ROOT  = f'{DATA_ROOT}/images'
BLOCKS    = (7, 15, 19, 21, 23)


# --------------------------------------------------------------------------
# Model logic (matches FreqDecoupledDistiller)
# --------------------------------------------------------------------------
def standardize(feat):
    feat = feat - feat.mean(dim=(2, 3), keepdim=True)
    feat = F.normalize(feat, p=2, dim=1)
    return feat


def freq_decompose(feat, cutoff_ratio=0.5):
    """Rectangular centred mask, identical to the training loss."""
    B, C, H, W = feat.shape
    fft = torch.fft.fft2(feat, norm='ortho')
    fs = torch.fft.fftshift(fft, dim=(-2, -1))
    mask = torch.zeros(1, 1, H, W, device=feat.device, dtype=feat.dtype)
    h_c, w_c = H // 2, W // 2
    h_r = max(1, int(H * cutoff_ratio / 2))
    w_r = max(1, int(W * cutoff_ratio / 2))
    mask[:, :, h_c - h_r:h_c + h_r, w_c - w_r:w_c + w_r] = 1.0
    low  = torch.fft.ifft2(torch.fft.ifftshift(fs * mask,       dim=(-2,-1)), norm='ortho').real
    high = torch.fft.ifft2(torch.fft.ifftshift(fs * (1 - mask), dim=(-2,-1)), norm='ortho').real
    return low, high


# --------------------------------------------------------------------------
# Linear CKA
# --------------------------------------------------------------------------
def linear_cka(X, Y):
    """X, Y: [n, d] (n samples/tokens, d features). Returns scalar CKA in [0,1]."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    yx = Y.t() @ X                       # [d, d]
    num = (yx ** 2).sum()
    xx = X.t() @ X
    yy = Y.t() @ Y
    den = torch.sqrt((xx ** 2).sum()) * torch.sqrt((yy ** 2).sum())
    return (num / (den + 1e-12)).item()


def feat_to_tokens(f):
    """[1,C,H,W] -> [H*W, C]."""
    return f.squeeze(0).flatten(1).t().contiguous()


# --------------------------------------------------------------------------
# Backbone build / load
# --------------------------------------------------------------------------
def build_backbone(device):
    from mmpretrain.models import build_backbone as mm_build
    cfg = dict(type='mmpretrain.TIMMBackbone',
               model_name='vit_large_patch14_reg4_dinov2.lvd142m',
               pretrained=True, features_only=True,
               out_indices=BLOCKS, dynamic_img_size=True)
    b = mm_build(cfg).to(device)
    b.eval()  # TIMMBackbone.eval() returns None — don't chain
    for p in b.parameters():
        p.requires_grad = False
    return b


def load_student_checkpoint(backbone, ckpt_path, lora_scale=1.0):
    """Load student backbone weights, trying several key prefixes.

    Handles both Stage-1 distiller checkpoints ('student.backbone.*', with LoRA)
    and single-stage detector+KD checkpoints ('backbone.*' / 'detector.backbone.*').
    LoRA factors, if present, are merged; otherwise merge is a no-op.
    """
    from mmdet.models.utils.lora import merge_lora_state_dict
    ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True, weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    bw = {}
    for prefix in ('student.backbone.', 'backbone.', 'detector.backbone.'):
        cand = {k[len(prefix):]: v for k, v in sd.items()
                if k.startswith(prefix) and not k.startswith('teacher')}
        if cand:
            bw = cand
            print(f"  using prefix '{prefix}'  ({len(cand)} tensors)")
            break
    if not bw:
        raise RuntimeError(f"No backbone keys in {ckpt_path}. "
                           f"Top keys: {sorted(set(k.split('.')[0] for k in sd))}")
    bw = merge_lora_state_dict(bw, scaling=lora_scale)
    missing, unexpected = backbone.load_state_dict(bw, strict=False)
    print(f"  loaded {len(bw)} student-backbone tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")


def load_img(path, device):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std  = np.array([58.395, 57.12, 57.375], np.float32)
    rgb = (rgb - mean) / std
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    _, _, H, W = t.shape
    ph, pw = (14 - H % 14) % 14, (14 - W % 14) % 14
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='backbone')
    ap.add_argument('--checkpoint', default=None,
                    help='Stage-1 student checkpoint; omit for No-KD pretrained baseline')
    ap.add_argument('--lora-scale', type=float, default=1.0)
    ap.add_argument('--cutoff', type=float, default=0.5)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  label: {args.label}")

    # Teacher: always pretrained DINOv2 fed RGB
    teacher = build_backbone(device)
    # Student: pretrained (No-KD) or loaded from checkpoint
    student = build_backbone(device)
    if args.checkpoint:
        print(f"Loading student from: {args.checkpoint}")
        load_student_checkpoint(student, args.checkpoint, args.lora_scale)
    else:
        print("No checkpoint → student = pretrained DINOv2 (No-KD baseline)")

    images = json.load(open(VAL_JSON))['images'][:args.n]
    print(f"Computing CKA over {len(images)} pairs (cutoff={args.cutoff})\n")

    # per-block accumulators of per-image CKA
    cka_full = {b: [] for b in BLOCKS}
    cka_low  = {b: [] for b in BLOCKS}
    cka_high = {b: [] for b in BLOCKS}

    for i, im in enumerate(images):
        rgb = load_img(os.path.join(IMG_ROOT, im['file_name']),  device)  # teacher RGB
        ir  = load_img(os.path.join(IMG_ROOT, im['file_name2']), device)  # student IR
        with torch.no_grad():
            tf = teacher(rgb)
            sf = student(ir)
        for bi, blk in enumerate(BLOCKS):
            t = standardize(tf[bi].float())
            s = standardize(sf[bi].float())
            # full
            cka_full[blk].append(linear_cka(feat_to_tokens(t), feat_to_tokens(s)))
            # bands
            tl, th = freq_decompose(t, args.cutoff)
            sl, sh = freq_decompose(s, args.cutoff)
            cka_low[blk].append(linear_cka(feat_to_tokens(tl), feat_to_tokens(sl)))
            cka_high[blk].append(linear_cka(feat_to_tokens(th), feat_to_tokens(sh)))
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(images)}")

    full = np.mean([np.mean(cka_full[b]) for b in BLOCKS])
    low  = np.mean([np.mean(cka_low[b])  for b in BLOCKS])
    high = np.mean([np.mean(cka_high[b]) for b in BLOCKS])

    print(f"\n{'='*52}")
    print(f"  CKA results for: {args.label}")
    print(f"{'='*52}")
    print(f"  {'Block':>6} {'full':>8} {'low':>8} {'high':>8}")
    for b in BLOCKS:
        print(f"  {b:>6} {np.mean(cka_full[b]):>8.3f} "
              f"{np.mean(cka_low[b]):>8.3f} {np.mean(cka_high[b]):>8.3f}")
    print(f"  {'-'*32}")
    print(f"  {'MEAN':>6} {full:>8.3f} {low:>8.3f} {high:>8.3f}")
    print(f"\n  LaTeX row:")
    print(f"  {args.label} & {full:.2f} & {low:.2f} & {high:.2f} \\\\")


if __name__ == '__main__':
    main()
