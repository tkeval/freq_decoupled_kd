"""
Cross-Architecture Frequency-Decoupled Cross-Modal KD.

DINOv2 ViT-Large teacher (RGB) → ResNet-50 student (IR) via FFT loss.
Bridges the architecture gap with 1×1 conv projectors that map student
channels (256/512/1024/2048) to the teacher's uniform 1024-dim space.

Key differences from FreqDecoupledDistillerCNN (same-arch ResNet→ResNet):
  - Teacher: DINOv2 ViT-Large (mmpretrain), not ResNet-50
  - Teacher produces fixed-resolution features (H/14 × W/14), student
    produces multi-scale pyramid (strides 4/8/16/32) → spatial interpolation
  - Every layer pair needs a projector (channels never match)
  - Teacher input padded to patch_size=14, not 32

Stage 1: Backbone-only KD (no detection loss). Student backbone fine-tuned
directly (no LoRA — ResNet is small enough).
Stage 2: Merge adapted backbone with pretrained (via Stage2FasterRCNN with
backbone_merge_scale) and train standard Faster R-CNN on IR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from typing import Dict, List, Optional, Union


@MODELS.register_module()
class CrossArchFreqDistiller(BaseDetector):
    """Stage 1: Cross-Architecture Frequency-Decoupled KD.

    DINOv2 ViT-Large (RGB) teacher → ResNet-50 (IR) student backbone.

    Args:
        student_cfg: Config for the student detector (e.g., Faster R-CNN).
        teacher_backbone_cfg: Config for the DINOv2 ViT-Large teacher backbone.
            Built via mmpretrain's build_backbone (TIMMBackbone).
        distill_cfg: Layer pairs for distillation. Each entry:
            - name (str): Loss name for logging
            - student_feature_index (int): Index into student backbone outputs
            - teacher_feature_index (int): Index into teacher backbone outputs
            - student_channels (int): Student feature channels (e.g., 256)
            - teacher_channels (int): Teacher feature channels (1024 for ViT-L)
            - loss_weight (float): Weight for this layer pair
        freq_cutoff: Fraction of spectrum for low-frequency band (default 0.5).
        high_freq_weight: Multiplier for high-freq logMSE loss (default 0.1).
        teacher_pad_size: Pad teacher inputs to multiple of this (14 for DINOv2).
        data_preprocessor: Data preprocessor config.
        init_cfg: Initialization config.
    """

    def __init__(self,
                 student_cfg: dict,
                 teacher_backbone_cfg: dict,
                 distill_cfg: List[dict],
                 freq_cutoff: float = 0.5,
                 high_freq_weight: float = 0.1,
                 teacher_pad_size: int = 14,
                 data_preprocessor: dict = None,
                 init_cfg: dict = None):
        if data_preprocessor is None:
            data_preprocessor = student_cfg.get('data_preprocessor', None)

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.freq_cutoff = freq_cutoff
        self.high_freq_weight = high_freq_weight
        self.teacher_pad_size = teacher_pad_size

        # ----- Student detector (Faster R-CNN with ResNet-50) -----
        self.student = MODELS.build(student_cfg)

        # ----- Teacher backbone (DINOv2 ViT-Large, frozen) -----
        from mmpretrain.models import build_backbone
        self.teacher_backbone = build_backbone(teacher_backbone_cfg)
        self.teacher_backbone._is_init = True
        self.teacher_backbone.eval()
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False

        # ----- Projectors (1×1 conv: student_ch → teacher_ch) -----
        # Every pair needs a projector since architectures differ
        self.projectors = nn.ModuleDict()
        self.distill_configs = distill_cfg
        for d_cfg in distill_cfg:
            name = d_cfg['name']
            s_ch = d_cfg['student_channels']
            t_ch = d_cfg['teacher_channels']
            if s_ch != t_ch:
                self.projectors[name] = nn.Conv2d(s_ch, t_ch, kernel_size=1)

    # -----------------------------------------------------------------
    # Overrides for teacher freezing / checkpoint exclusion
    # -----------------------------------------------------------------
    def train(self, mode=True):
        """Keep teacher in eval mode."""
        super().train(mode)
        self.teacher_backbone.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Exclude frozen teacher from checkpoints."""
        sd = super().state_dict(*args, **kwargs)
        filtered = type(sd)(
            {k: v for k, v in sd.items()
             if not k.startswith('teacher_backbone.')})
        if hasattr(sd, '_metadata'):
            filtered._metadata = sd._metadata
        return filtered

    def load_state_dict(self, state_dict, strict=True):
        """Load student-only checkpoint (teacher excluded from saves)."""
        student_sd = {k: v for k, v in state_dict.items()
                      if not k.startswith('teacher_backbone.')}
        return super().load_state_dict(student_sd, strict=False)

    # -----------------------------------------------------------------
    # Frequency decomposition (same logic as FreqDecoupledDistiller)
    # -----------------------------------------------------------------
    @staticmethod
    def _freq_decompose(feat, cutoff_ratio=0.5):
        """Decompose feature map into low/high frequency via 2D FFT."""
        B, C, H, W = feat.shape

        fft = torch.fft.fft2(feat, norm='ortho')
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        mask = torch.zeros(1, 1, H, W, device=feat.device, dtype=feat.dtype)
        h_center, w_center = H // 2, W // 2
        h_radius = max(1, int(H * cutoff_ratio / 2))
        w_radius = max(1, int(W * cutoff_ratio / 2))
        mask[:, :,
             h_center - h_radius:h_center + h_radius,
             w_center - w_radius:w_center + w_radius] = 1.0

        low_fft = fft_shifted * mask
        high_fft = fft_shifted * (1 - mask)

        low_freq = torch.fft.ifft2(
            torch.fft.ifftshift(low_fft, dim=(-2, -1)),
            norm='ortho').real
        high_freq = torch.fft.ifft2(
            torch.fft.ifftshift(high_fft, dim=(-2, -1)),
            norm='ortho').real

        return low_freq, high_freq

    @staticmethod
    def _standardize(feat):
        """Feature standardization: spatial mean subtraction + L2 norm."""
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    def _compute_freq_loss(self, s_feat, t_feat):
        """Compute frequency-decoupled distillation loss for one layer."""
        s_feat = self._standardize(s_feat)
        t_feat = self._standardize(t_feat)

        s_low, s_high = self._freq_decompose(s_feat, self.freq_cutoff)
        t_low, t_high = self._freq_decompose(t_feat, self.freq_cutoff)

        loss_low = F.mse_loss(s_low, t_low)

        s_high_log = torch.log1p(s_high.abs())
        t_high_log = torch.log1p(t_high.abs())
        loss_high = F.mse_loss(s_high_log, t_high_log)

        return loss_low, loss_high

    # -----------------------------------------------------------------
    # Forward methods
    # -----------------------------------------------------------------
    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Compute frequency-decoupled distillation losses."""
        losses = {}

        # 1. Student backbone forward (IR)
        student_feats = self.student.backbone(batch_inputs)

        # 2. Teacher backbone forward (RGB) — frozen
        self.teacher_backbone.eval()
        img_rgbs = [s.metainfo['img_rgb'] for s in batch_data_samples]
        rgb_inputs = self._prepare_teacher_inputs(
            img_rgbs, batch_inputs.device)

        with torch.no_grad():
            teacher_feats = self.teacher_backbone(rgb_inputs)

        # 3. Frequency-decoupled loss per layer pair
        for d_cfg in self.distill_configs:
            name = d_cfg['name']
            s_idx = d_cfg['student_feature_index']
            t_idx = d_cfg['teacher_feature_index']
            weight = d_cfg.get('loss_weight', 1.0)

            s_feat = student_feats[s_idx]
            t_feat = teacher_feats[t_idx]

            # Project student channels → teacher channels (1×1 conv)
            if name in self.projectors:
                s_feat = self.projectors[name](s_feat)

            # Resize student features to match teacher spatial dims
            # (ViT: H/14 × W/14 fixed; ResNet: multi-scale pyramid)
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(
                    s_feat, size=t_feat.shape[-2:],
                    mode='bilinear', align_corners=False)

            loss_low, loss_high = self._compute_freq_loss(s_feat, t_feat)

            losses[f'{name}_low'] = loss_low * weight
            losses[f'{name}_high'] = loss_high * weight * self.high_freq_weight

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        """Forward prediction using the student detector."""
        return self.student.predict(
            batch_inputs, batch_data_samples, rescale)

    def _forward(self, batch_inputs, batch_data_samples):
        return self.student._forward(batch_inputs, batch_data_samples)

    def extract_feat(self, batch_inputs):
        return self.student.extract_feat(batch_inputs)

    # -----------------------------------------------------------------
    # Teacher input preparation (DINOv2 — pad to patch_size=14)
    # -----------------------------------------------------------------
    def _prepare_teacher_inputs(self, img_rgbs, device):
        """Stack, normalize, and pad RGB images for the DINOv2 teacher.

        Args:
            img_rgbs: List of BGR numpy arrays from metainfo.
            device: Target device.
        """
        processed = []
        mean = torch.tensor(
            [123.675, 116.28, 103.53], device=device).view(3, 1, 1)
        std = torch.tensor(
            [58.395, 57.12, 57.375], device=device).view(3, 1, 1)

        for img in img_rgbs:
            if isinstance(img, torch.Tensor):
                t = img.float()
            else:
                t = torch.from_numpy(img).permute(2, 0, 1).float()
            # BGR → RGB
            t = t[[2, 1, 0], ...]
            t = (t.to(device) - mean) / std
            processed.append(t)

        max_h = max(t.shape[1] for t in processed)
        max_w = max(t.shape[2] for t in processed)

        # Pad to multiple of teacher patch size (14 for DINOv2)
        pad_div = self.teacher_pad_size
        max_h += (pad_div - max_h % pad_div) % pad_div
        max_w += (pad_div - max_w % pad_div) % pad_div

        padded = []
        for t in processed:
            pad_h = max_h - t.shape[1]
            pad_w = max_w - t.shape[2]
            if pad_h > 0 or pad_w > 0:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded.append(t)

        return torch.stack(padded)
