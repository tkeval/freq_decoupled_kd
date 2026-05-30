"""
Visualize FFT-based frequency decomposition on a KAIST RGB image.

Replicates exactly the _freq_decompose() and _standardize() logic from
FreqDecoupledDistiller (mmdet/models/distillers/freq_decoupled_distiller.py).

Outputs a matplotlib figure showing:
  - Original RGB image
  - FFT magnitude spectrum (log scale)
  - Low-frequency component (spatial domain)
  - High-frequency component (spatial domain)
  - Low/high-pass masks used in the decomposition

Usage:
    python tools/visualize_fft_decomposition.py
    python tools/visualize_fft_decomposition.py --img path/to/image.jpg
    python tools/visualize_fft_decomposition.py --cutoff 0.3 --no-standardize
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ---------------------------------------------------------------------------
# Replicated logic from FreqDecoupledDistiller
# ---------------------------------------------------------------------------

def standardize(feat: torch.Tensor) -> torch.Tensor:
    """Mean subtraction + L2 norm along channel dim (per-sample)."""
    feat = feat - feat.mean(dim=(2, 3), keepdim=True)
    feat = F.normalize(feat, p=2, dim=1)
    return feat


def freq_decompose(feat: torch.Tensor, cutoff_ratio: float = 0.5):
    """Decompose [B, C, H, W] into low/high frequency via 2D FFT.

    Returns:
        low_freq:  [B, C, H, W] spatial low-frequency component
        high_freq: [B, C, H, W] spatial high-frequency component
        fft_shifted: [B, C, H, W] complex shifted spectrum (for visualization)
        mask: [1, 1, H, W] low-pass mask
    """
    B, C, H, W = feat.shape

    fft = torch.fft.fft2(feat, norm='ortho')
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Low-pass mask: centered rectangle
    mask = torch.zeros(1, 1, H, W, dtype=feat.dtype)
    h_center, w_center = H // 2, W // 2
    h_radius = max(1, int(H * cutoff_ratio / 2))
    w_radius = max(1, int(W * cutoff_ratio / 2))
    mask[:, :,
         h_center - h_radius:h_center + h_radius,
         w_center - w_radius:w_center + w_radius] = 1.0

    low_fft  = fft_shifted * mask
    high_fft = fft_shifted * (1 - mask)

    low_freq  = torch.fft.ifft2(
        torch.fft.ifftshift(low_fft,  dim=(-2, -1)), norm='ortho').real
    high_freq = torch.fft.ifft2(
        torch.fft.ifftshift(high_fft, dim=(-2, -1)), norm='ortho').real

    return low_freq, high_freq, fft_shifted, mask


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def to_display(tensor: torch.Tensor, normalize: bool = True) -> np.ndarray:
    """Convert [1, C, H, W] tensor → [H, W, 3] uint8 for imshow.

    Works for both single-channel (C=1) and RGB (C=3).
    """
    img = tensor.squeeze(0)             # [C, H, W]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.permute(1, 2, 0).numpy()  # [H, W, C]

    if normalize:
        img = img - img.min()
        rng = img.max() - img.min()
        if rng > 1e-8:
            img = img / rng

    return np.clip(img, 0, 1)


def spectrum_display(fft_shifted: torch.Tensor) -> np.ndarray:
    """Log-magnitude spectrum averaged over channels → [H, W] float [0,1]."""
    mag = fft_shifted.abs().mean(dim=1).squeeze(0)  # [H, W]
    mag = torch.log1p(mag).numpy()
    mag = mag - mag.min()
    mag = mag / (mag.max() + 1e-8)
    return mag


def draw_mask_outline(ax, mask: np.ndarray, color='cyan', lw=2):
    """Draw rectangle border of the low-pass mask region on an axis."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    from matplotlib.patches import Rectangle
    rect = Rectangle((c0 - 0.5, r0 - 0.5),
                     c1 - c0 + 1, r1 - r0 + 1,
                     linewidth=lw, edgecolor=color,
                     facecolor='none')
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualize FFT frequency decomposition on a KAIST RGB image')
    parser.add_argument(
        '--img',
        default='./data/kaist-rgbt/'
                'images/set00/V000/visible/I00000.jpg',
        help='Path to a KAIST visible (RGB) image')
    parser.add_argument(
        '--cutoff', type=float, default=0.5,
        help='Low-frequency cutoff ratio (default: 0.5 = center 50%% of spectrum)')
    parser.add_argument(
        '--no-standardize', action='store_true',
        help='Skip feature standardization (mean-sub + L2-norm)')
    parser.add_argument(
        '--out', default=None,
        help='Optional output path to save the figure (e.g. fft_vis.png)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load image  (H, W, 3) BGR
    # ------------------------------------------------------------------
    import cv2
    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.img}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = img_rgb.shape[:2]
    print(f"Loaded: {args.img}  ({W_orig}x{H_orig})")

    # ------------------------------------------------------------------
    # 2. Convert to tensor [1, 3, H, W] float, ImageNet-normalized
    # ------------------------------------------------------------------
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395,  57.12,  57.375], dtype=np.float32)

    img_norm = (img_rgb.astype(np.float32) - mean) / std  # [H, W, 3]
    feat = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # ------------------------------------------------------------------
    # 3. Optionally standardize (matches distiller behaviour)
    # ------------------------------------------------------------------
    standardize_applied = not args.no_standardize
    if standardize_applied:
        feat_proc = standardize(feat)
    else:
        feat_proc = feat

    # ------------------------------------------------------------------
    # 4. FFT decomposition
    # ------------------------------------------------------------------
    low_freq, high_freq, fft_shifted, mask = freq_decompose(
        feat_proc, cutoff_ratio=args.cutoff)

    mask_np = mask.squeeze().numpy()  # [H, W]

    # logMSE transform applied to high-freq (as in distiller)
    high_freq_log = torch.log1p(high_freq.abs())

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f'FFT Frequency Decomposition  |  cutoff={args.cutoff}  '
        f'standardize={standardize_applied}\n'
        f'{Path(args.img).name}',
        fontsize=13, y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.35, wspace=0.25,
                           top=0.90, bottom=0.04,
                           left=0.04, right=0.97)

    # Row 0: image + spectrum + masks
    ax_orig    = fig.add_subplot(gs[0, 0])
    ax_spec    = fig.add_subplot(gs[0, 1])
    ax_lowmask = fig.add_subplot(gs[0, 2])
    ax_highmask = fig.add_subplot(gs[0, 3])

    # Row 1: spatial components
    ax_low     = fig.add_subplot(gs[1, 0])
    ax_high    = fig.add_subplot(gs[1, 1])
    ax_high_log = fig.add_subplot(gs[1, 2])
    ax_recon   = fig.add_subplot(gs[1, 3])

    # --- Original image ---
    ax_orig.imshow(img_rgb)
    ax_orig.set_title('Original RGB', fontsize=11)
    ax_orig.axis('off')

    # --- Log-magnitude spectrum with mask outline ---
    spec = spectrum_display(fft_shifted)
    ax_spec.imshow(spec, cmap='inferno')
    draw_mask_outline(ax_spec, mask_np, color='cyan')
    ax_spec.set_title(f'FFT Magnitude (log scale)\ncyan box = low-pass region',
                      fontsize=10)
    ax_spec.axis('off')

    # --- Low-pass mask ---
    ax_lowmask.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
    ax_lowmask.set_title(f'Low-pass mask\n(center {args.cutoff*100:.0f}% of spectrum)',
                         fontsize=10)
    ax_lowmask.axis('off')

    # --- High-pass mask ---
    ax_highmask.imshow(1 - mask_np, cmap='gray', vmin=0, vmax=1)
    ax_highmask.set_title('High-pass mask\n(remaining spectrum)', fontsize=10)
    ax_highmask.axis('off')

    # --- Low-frequency spatial component ---
    ax_low.imshow(to_display(low_freq))
    ax_low.set_title('Low-freq (spatial)\nMSE loss target', fontsize=10)
    ax_low.axis('off')

    # --- High-frequency spatial component ---
    ax_high.imshow(to_display(high_freq))
    ax_high.set_title('High-freq (spatial)\nraw', fontsize=10)
    ax_high.axis('off')

    # --- High-freq after log1p transform ---
    ax_high_log.imshow(to_display(high_freq_log))
    ax_high_log.set_title('High-freq log(1+|x|)\nlogMSE loss target', fontsize=10)
    ax_high_log.axis('off')

    # --- Reconstruction = low + high (should match original) ---
    recon = low_freq + high_freq
    ax_recon.imshow(to_display(recon))
    ax_recon.set_title('Reconstruction\nlow + high', fontsize=10)
    ax_recon.axis('off')

    # ------------------------------------------------------------------
    # 6. Print energy stats
    # ------------------------------------------------------------------
    total_energy  = feat_proc.pow(2).sum().item()
    low_energy    = low_freq.pow(2).sum().item()
    high_energy   = high_freq.pow(2).sum().item()
    print(f"\nEnergy breakdown (cutoff={args.cutoff}):")
    print(f"  Total   : {total_energy:.4f}")
    print(f"  Low-freq: {low_energy:.4f}  ({100*low_energy/total_energy:.1f}%)")
    print(f"  High-freq:{high_energy:.4f}  ({100*high_energy/total_energy:.1f}%)")

    recon_err = (recon - feat_proc).abs().max().item()
    print(f"  Recon max-abs error: {recon_err:.2e}  (should be ~0)")

    # ------------------------------------------------------------------
    # 7. Save / show
    # ------------------------------------------------------------------
    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {args.out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
