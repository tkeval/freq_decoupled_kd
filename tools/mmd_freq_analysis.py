"""
Frequency-aware MMD diagnostic (Table 8) — no training required.

Post-hoc measurement of teacher(RGB) vs student(IR) Maximum Mean Discrepancy
per frequency band, on the PRETRAINED DINOv2 backbone (No-KD). Argues that a
global distribution-matching objective applies comparable pressure to both
bands (MMD_low ~ MMD_high), unlike FreqKD's asymmetric 1.0 / 0.1 weighting.

RBF kernel (linear-kernel MMD would collapse the ~zero-mean high band to 0),
token subsampling for tractable O(N^2) kernels, and a per-band median-heuristic
bandwidth so the two bands are compared on a shape (not magnitude) basis.

Usage:
  python tools/mmd_freq_analysis.py --n 500 --gpu 0
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

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


def feat_to_tokens(f):
    return f.squeeze(0).flatten(1).t().contiguous()   # [H*W, C]


# --------------------------------------------------------------------------
# RBF MMD
# --------------------------------------------------------------------------
def _sqdist(A, B):
    a2 = (A * A).sum(1, keepdim=True)
    b2 = (B * B).sum(1, keepdim=True).t()
    return (a2 + b2 - 2.0 * (A @ B.t())).clamp_min_(0)


def median_sigma2(Z, cap=1000):
    """Median-heuristic bandwidth (sigma^2) from a subsample of Z."""
    n = Z.shape[0]
    idx = torch.randperm(n, device=Z.device)[:min(cap, n)]
    S = Z[idx]
    d2 = _sqdist(S, S)
    med = d2[d2 > 0].median()
    return 0.5 * med + 1e-12


def rbf_mmd2(X, Y, sigma2):
    """Unbiased RBF MMD^2 between sample sets X [m,d], Y [n,d]."""
    Kxx = torch.exp(-_sqdist(X, X) / (2 * sigma2))
    Kyy = torch.exp(-_sqdist(Y, Y) / (2 * sigma2))
    Kxy = torch.exp(-_sqdist(X, Y) / (2 * sigma2))
    m, n = X.shape[0], Y.shape[0]
    s_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
    s_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    s_xy = Kxy.mean()
    return (s_xx + s_yy - 2 * s_xy).item()


# --------------------------------------------------------------------------
# Backbone / IO
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
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--tokens-per-img', type=int, default=10)
    ap.add_argument('--cutoff', type=float, default=0.5)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    backbone = build_backbone(device)  # pretrained = No-KD
    images = json.load(open(VAL_JSON))['images'][:args.n]
    print(f"Collecting tokens from {len(images)} pairs "
          f"({args.tokens_per_img} tokens/img, cutoff={args.cutoff})\n")

    # token buffers: [block][band]['t'|'s'] -> list of [k, C]
    buf = {b: {'low': {'t': [], 's': []}, 'high': {'t': [], 's': []}} for b in BLOCKS}

    for i, im in enumerate(images):
        rgb = load_img(os.path.join(IMG_ROOT, im['file_name']),  device)
        ir  = load_img(os.path.join(IMG_ROOT, im['file_name2']), device)
        with torch.no_grad():
            tf = backbone(rgb)
            sf = backbone(ir)
        for bi, blk in enumerate(BLOCKS):
            t = standardize(tf[bi].float())
            s = standardize(sf[bi].float())
            tl, th = freq_decompose(t, args.cutoff)
            sl, sh = freq_decompose(s, args.cutoff)
            for band, (tb, sb) in [('low', (tl, sl)), ('high', (th, sh))]:
                tt, st = feat_to_tokens(tb), feat_to_tokens(sb)
                ntok = tt.shape[0]
                idx = torch.randperm(ntok, device=device)[:args.tokens_per_img]
                buf[blk][band]['t'].append(tt[idx].cpu())
                buf[blk][band]['s'].append(st[idx].cpu())
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(images)}")

    print("\nComputing per-band RBF-MMD ...")
    print(f"{'Block':>6} {'MMD_low':>12} {'MMD_high':>12} {'ratio (high/low)':>18}")
    print('-' * 50)
    lows, highs = [], []
    for blk in BLOCKS:
        res = {}
        for band in ('low', 'high'):
            X = torch.cat(buf[blk][band]['t']).to(device)
            Y = torch.cat(buf[blk][band]['s']).to(device)
            sig2 = median_sigma2(torch.cat([X, Y]))
            res[band] = rbf_mmd2(X, Y, sig2)
        lows.append(res['low']); highs.append(res['high'])
        print(f"{blk:>6} {res['low']:>12.5f} {res['high']:>12.5f} "
              f"{res['high']/(res['low']+1e-12):>17.2f}x")
    ml, mh = np.mean(lows), np.mean(highs)
    print('-' * 50)
    print(f"{'Mean':>6} {ml:>12.5f} {mh:>12.5f} {mh/(ml+1e-12):>17.2f}x")
    print(f"\n  → MMD_high / MMD_low ratio = {mh/(ml+1e-12):.2f}  "
          f"(near 1.0 ⇒ uniform pressure across bands)")
    print(f"\n  LaTeX row:")
    print(f"  Teacher--student MMD (no KD) & {ml:.3f} & {mh:.3f} & {mh/(ml+1e-12):.2f} \\\\")


if __name__ == '__main__':
    main()
