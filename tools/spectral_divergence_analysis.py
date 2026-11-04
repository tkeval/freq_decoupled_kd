"""
Spectral Divergence Analysis (Table 2) — go/no-go diagnostic.

Measures RGB-teacher vs IR-student feature divergence per frequency band on
the PRETRAINED DINOv2 backbone (before distillation), at blocks {7,15,19,21,23}.

Reports THREE candidate divergence metrics per band so we can pick a principled
one and confirm the hypothesis direction (high-freq divergence > low-freq):
  1. raw MSE                : ||T_band - S_band||^2            (magnitude-dominated)
  2. normalized MSE (NMSE)  : ||T_band - S_band||^2 / ||T_band||^2
  3. cosine distance        : 1 - <T_band,S_band>/(||T||·||S||)

Uses the EXACT standardize + FFT logic from FreqDecoupledDistiller.
Bands: inner `r_low` radius = low; outer `r_high` radius = high (radial mask).

Usage:
  conda run -n mmpretrain_git python tools/spectral_divergence_analysis.py --n 500
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from collections import defaultdict

DATA_ROOT = './data/kaist-rgbt'
VAL_JSON  = f'{DATA_ROOT}/annotations/instancesonly_filtered_all-20_val.json'
IMG_ROOT  = f'{DATA_ROOT}/images'
BLOCKS    = (7, 15, 19, 21, 23)


# --------------------------------------------------------------------------
# Model logic (matches FreqDecoupledDistiller)
# --------------------------------------------------------------------------
def standardize(feat):
    """[B,C,H,W] -> per-channel mean-sub + L2 norm."""
    feat = feat - feat.mean(dim=(2, 3), keepdim=True)
    feat = F.normalize(feat, p=2, dim=1)
    return feat


def radial_masks(H, W, r_low=0.25, r_high=0.25, device='cpu'):
    """Build inner (low) and outer (high) radial masks.

    r_low : fraction of max-radius for the inner low-freq disk.
    r_high: fraction of max-radius for the outer high-freq ring (from edge).
    """
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    r = torch.sqrt((yy - cy).float()**2 + (xx - cx).float()**2)
    r_max = float(min(cy, cx))
    rn = r / r_max  # normalized radius in [0, ~1]
    low_mask  = (rn <= r_low).float().to(device)
    high_mask = (rn >= (1.0 - r_high)).float().to(device)
    return low_mask[None, None], high_mask[None, None]


def band_components(feat, low_mask, high_mask):
    """Return spatial-domain low/high band features via masked FFT."""
    fft = torch.fft.fft2(feat, norm='ortho')
    fs = torch.fft.fftshift(fft, dim=(-2, -1))
    low  = torch.fft.ifft2(torch.fft.ifftshift(fs * low_mask,  dim=(-2,-1)), norm='ortho').real
    high = torch.fft.ifft2(torch.fft.ifftshift(fs * high_mask, dim=(-2,-1)), norm='ortho').real
    return low, high


# --------------------------------------------------------------------------
# Divergence metrics (per sample, per band)
# --------------------------------------------------------------------------
def metrics(t, s):
    """t, s: [1,C,H,W] band features. Returns (mse, nmse, cosdist)."""
    diff2 = (t - s).pow(2).sum().item()
    tnorm2 = t.pow(2).sum().item()
    mse = diff2 / t.numel()
    nmse = diff2 / (tnorm2 + 1e-12)
    tf, sf = t.flatten(), s.flatten()
    cos = torch.dot(tf, sf) / (tf.norm() * sf.norm() + 1e-12)
    cosdist = (1.0 - cos).item()
    return mse, nmse, cosdist


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------
def build_backbone():
    from mmpretrain.models import build_backbone as mm_build
    cfg = dict(type='mmpretrain.TIMMBackbone',
               model_name='vit_large_patch14_reg4_dinov2.lvd142m',
               pretrained=True, features_only=True,
               out_indices=BLOCKS, dynamic_img_size=True)
    return mm_build(cfg)


def load_img(path, device):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std  = np.array([58.395, 57.12, 57.375], np.float32)
    rgb = (rgb - mean) / std
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None]  # [1,3,H,W]
    # pad to multiple of 14
    _, _, H, W = t.shape
    ph, pw = (14 - H % 14) % 14, (14 - W % 14) % 14
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--r-low', type=float, default=0.25)
    ap.add_argument('--r-high', type=float, default=0.25)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    backbone = build_backbone().to(device)
    backbone.eval()  # TIMMBackbone.eval() returns None — don't chain/assign
    for p in backbone.parameters():
        p.requires_grad = False

    images = json.load(open(VAL_JSON))['images'][:args.n]
    print(f"Analysing {len(images)} paired samples; bands r_low={args.r_low}, r_high={args.r_high}\n")

    # accumulators: agg[block][metric][band] = list
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for i, im in enumerate(images):
        rgb = load_img(os.path.join(IMG_ROOT, im['file_name']),  device)  # RGB teacher
        ir  = load_img(os.path.join(IMG_ROOT, im['file_name2']), device)  # IR student
        with torch.no_grad():
            tfeats = backbone(rgb)
            sfeats = backbone(ir)
        for bi, blk in enumerate(BLOCKS):
            t = standardize(tfeats[bi].float())
            s = standardize(sfeats[bi].float())
            H, W = t.shape[-2:]
            lm, hm = radial_masks(H, W, args.r_low, args.r_high, device)
            t_low, t_high = band_components(t, lm, hm)
            s_low, s_high = band_components(s, lm, hm)
            for name, (tb, sb) in [('low', (t_low, s_low)), ('high', (t_high, s_high))]:
                mse, nmse, cosd = metrics(tb, sb)
                agg[blk]['mse'][name].append(mse)
                agg[blk]['nmse'][name].append(nmse)
                agg[blk]['cos'][name].append(cosd)
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(images)}")

    # ---- Report ----
    for metric_key, metric_name in [('mse', 'Raw MSE'),
                                     ('nmse', 'Normalized MSE'),
                                     ('cos', 'Cosine distance')]:
        print(f"\n{'='*64}\n  METRIC: {metric_name}\n{'='*64}")
        print(f"{'Block':>6} {'D_low':>10} {'D_high':>10} {'ratio (high/low)':>18}")
        print('-' * 48)
        all_low, all_high = [], []
        for blk in BLOCKS:
            dl = np.mean(agg[blk][metric_key]['low'])
            dh = np.mean(agg[blk][metric_key]['high'])
            all_low.append(dl); all_high.append(dh)
            print(f"{blk:>6} {dl:>10.4f} {dh:>10.4f} {dh/(dl+1e-12):>17.2f}x")
        ml, mh = np.mean(all_low), np.mean(all_high)
        print('-' * 48)
        print(f"{'Mean':>6} {ml:>10.4f} {mh:>10.4f} {mh/(ml+1e-12):>17.2f}x")
        verdict = "HIGH > LOW  ✓ (supports hypothesis)" if mh > ml else "LOW > HIGH  ✗ (contradicts!)"
        print(f"  → {verdict}")


if __name__ == '__main__':
    main()
