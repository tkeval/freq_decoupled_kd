"""
Stage 1: Frequency-Decoupled Cross-Modal Knowledge Distillation.

Transfers modality-general (low-frequency) knowledge from an RGB DINOv2
teacher to an IR DINOv2 student, while relaxing constraints on
modality-specific (high-frequency) features.

Key insight: RGB and IR share structural/spatial information (low-freq)
but differ in appearance/texture (high-freq). Standard cosine similarity
or MSE forces the student to match both, which is impossible and harmful.
Frequency decomposition via 2D FFT separates these components so we can
distill only what transfers.

Loss design (from FD-CMKD):
  - Low-frequency:  MSE (strong alignment — modality-general)
  - High-frequency: logMSE via log(1+|x|) (relaxed — modality-specific)
  - Feature standardization: mean subtraction + L2 norm before FFT

Reference:
  "Frequency-Decoupled Cross-Modal Knowledge Distillation" (arXiv 2025)
  FD2-Net (AAAI 2025) — confirms RGB=high-freq, IR=low-freq for thermal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.models.utils.lora import inject_lora
from typing import Dict, List, Optional, Union


@MODELS.register_module()
class FreqDecoupledDistiller(BaseDetector):
    """Stage 1: Frequency-Decoupled Cross-Modal KD for IR backbone.

    Single DINOv2 teacher (RGB) → DINOv2 student (IR) with LoRA.
    Same architecture for both, so no projection layers needed.

    Args:
        student_cfg: Config for the student DINO detector.
        teacher_backbone_cfg: Config for teacher DINOv2 backbone.
        distill_cfg: Layer pairs for distillation. Each entry:
            - name (str): Loss name for logging
            - student_feature_index (int): Index into student backbone outputs
            - teacher_feature_index (int): Index into teacher backbone outputs
            - student_channels (int): Student feature channels
            - teacher_channels (int): Teacher feature channels
            - loss_weight (float): Weight for this layer pair
        lora_cfg: LoRA config for student backbone. Keys:
            rank, alpha, dropout, target_modules.
        freq_cutoff: Fraction of spectrum for low-frequency band.
            0.5 means the center 50% of each spatial dimension is low-freq.
        low_freq_weight: Multiplier for low-freq MSE loss. Default 1.0.
            Set to 0.0 to ablate (high-freq only).
        high_freq_weight: Multiplier for high-freq logMSE loss relative
            to low-freq MSE loss. Default 0.1 (10x weaker).
            Set to 0.0 to ablate (low-freq only).
        detection_feature_indices: Indices to select from backbone outputs
            for the detection neck during predict(). Needed when backbone
            outputs more layers for distillation than the neck expects.
        data_preprocessor: Data preprocessor config.
        init_cfg: Initialization config.
    """

    def __init__(self,
                 student_cfg: dict,
                 teacher_backbone_cfg: dict,
                 distill_cfg: List[dict],
                 lora_cfg: Optional[dict] = None,
                 freq_cutoff: float = 0.5,
                 low_freq_weight: float = 1.0,
                 high_freq_weight: float = 0.1,
                 detection_feature_indices: Optional[List[int]] = None,
                 data_preprocessor: dict = None,
                 init_cfg: dict = None):
        if data_preprocessor is None:
            data_preprocessor = student_cfg.get('data_preprocessor', None)

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.freq_cutoff = freq_cutoff
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
        self.detection_feature_indices = detection_feature_indices

        # ----- Student (DINO detector with IR backbone) -----
        self.student = MODELS.build(student_cfg)

        # ----- LoRA injection (optional) -----
        if lora_cfg is not None:
            self._inject_lora(lora_cfg)

        # ----- Teacher backbone (frozen RGB DINOv2) -----
        from mmpretrain.models import build_backbone
        self.teacher_backbone = build_backbone(teacher_backbone_cfg)
        self.teacher_backbone.eval()
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False

        # ----- Projectors (only if channel dims differ) -----
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
        """Load student-only checkpoint."""
        student_sd = {k: v for k, v in state_dict.items()
                      if not k.startswith('teacher_backbone.')}
        return super().load_state_dict(student_sd, strict=False)

    # -----------------------------------------------------------------
    # LoRA injection
    # -----------------------------------------------------------------
    def _inject_lora(self, lora_cfg: dict):
        """Inject LoRA adapters into the student backbone."""
        rank = lora_cfg.get('rank', 16)
        alpha = lora_cfg.get('alpha', 16.0)
        dropout = lora_cfg.get('dropout', 0.05)
        target_modules = lora_cfg.get('target_modules', ['attn.qkv'])

        # Freeze student backbone
        for param in self.student.backbone.parameters():
            param.requires_grad = False

        # Inject LoRA
        num_replaced = inject_lora(
            self.student.backbone,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout)

        lora_params = sum(
            p.numel() for p in self.student.backbone.parameters()
            if p.requires_grad)
        total_params = sum(
            p.numel() for p in self.student.backbone.parameters())

        print(f"[LoRA] Injected into {num_replaced} modules "
              f"(rank={rank}, alpha={alpha}, dropout={dropout})")
        print(f"[LoRA] Target modules: {target_modules}")
        print(f"[LoRA] Trainable backbone params: {lora_params:,} / "
              f"{total_params:,} ({100*lora_params/total_params:.2f}%)")

    # -----------------------------------------------------------------
    # Frequency decomposition
    # -----------------------------------------------------------------
    @staticmethod
    def _freq_decompose(feat, cutoff_ratio=0.5):
        """Decompose feature map into low/high frequency via 2D FFT.

        Args:
            feat: [B, C, H, W] feature tensor (should be standardized).
            cutoff_ratio: Fraction of each spatial dimension to include
                in the low-frequency band. 0.5 = center 50%.

        Returns:
            low_freq: [B, C, H, W] low-frequency component.
            high_freq: [B, C, H, W] high-frequency component.
        """
        B, C, H, W = feat.shape

        # 2D FFT (orthonormal to preserve energy)
        fft = torch.fft.fft2(feat, norm='ortho')
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        # Low-pass mask: centered rectangle covering cutoff_ratio of each dim
        mask = torch.zeros(1, 1, H, W, device=feat.device, dtype=feat.dtype)
        h_center, w_center = H // 2, W // 2
        h_radius = max(1, int(H * cutoff_ratio / 2))
        w_radius = max(1, int(W * cutoff_ratio / 2))
        mask[:, :,
             h_center - h_radius:h_center + h_radius,
             w_center - w_radius:w_center + w_radius] = 1.0

        # Split spectrum
        low_fft = fft_shifted * mask
        high_fft = fft_shifted * (1 - mask)

        # Inverse FFT back to spatial domain
        low_freq = torch.fft.ifft2(
            torch.fft.ifftshift(low_fft, dim=(-2, -1)),
            norm='ortho').real
        high_freq = torch.fft.ifft2(
            torch.fft.ifftshift(high_fft, dim=(-2, -1)),
            norm='ortho').real

        return low_freq, high_freq

    @staticmethod
    def _standardize(feat):
        """Feature standardization before frequency decomposition.

        1. Subtract spatial mean (per channel) — removes DC bias
        2. L2 normalize along channel dim — removes scale differences

        This addresses distributional shift between RGB and IR features.
        """
        # Subtract spatial mean per channel
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        # L2 normalize along channel dimension
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    def _compute_freq_loss(self, s_feat, t_feat):
        """Compute frequency-decoupled distillation loss for one layer pair.

        Args:
            s_feat: [B, C, H, W] student feature (gradient flows through).
            t_feat: [B, C, H, W] teacher feature (detached/no_grad).

        Returns:
            loss_low: MSE on low-frequency components.
            loss_high: logMSE on high-frequency components.
        """
        # Standardize (student keeps gradients, teacher is detached)
        s_feat = self._standardize(s_feat)
        t_feat = self._standardize(t_feat)

        # Frequency decomposition
        s_low, s_high = self._freq_decompose(s_feat, self.freq_cutoff)
        t_low, t_high = self._freq_decompose(t_feat, self.freq_cutoff)

        # Low-frequency: MSE (strong alignment for modality-general features)
        loss_low = F.mse_loss(s_low, t_low)

        # High-frequency: logMSE (relaxed for modality-specific features)
        # log(1+|x|) compresses the range, preventing high-freq noise
        # from dominating the gradient
        s_high_log = torch.log1p(s_high.abs())
        t_high_log = torch.log1p(t_high.abs())
        loss_high = F.mse_loss(s_high_log, t_high_log)

        return loss_low, loss_high

    # -----------------------------------------------------------------
    # Forward methods
    # -----------------------------------------------------------------
    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Compute frequency-decoupled distillation losses.

        Args:
            batch_inputs: [B, C, H, W] IR images (preprocessed).
            batch_data_samples: Contains 'img_rgb' in metainfo.
        """
        losses = {}

        # 1. Student backbone forward (IR) — gradients flow
        student_feats = self.student.backbone(batch_inputs)

        # 2. Teacher backbone forward (RGB) — no gradients
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

            # Project if channels differ
            if name in self.projectors:
                s_feat = self.projectors[name](s_feat)

            # Resize if spatial dimensions mismatch
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(
                    s_feat, size=t_feat.shape[-2:],
                    mode='bilinear', align_corners=False)

            loss_low, loss_high = self._compute_freq_loss(s_feat, t_feat)

            losses[f'{name}_low'] = loss_low * weight * self.low_freq_weight
            losses[f'{name}_high'] = loss_high * weight * self.high_freq_weight

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        """Forward prediction using the student model."""
        if self.detection_feature_indices is None:
            return self.student.predict(
                batch_inputs, batch_data_samples, rescale)

        # Select subset of backbone features for detection neck
        all_feats = self.student.backbone(batch_inputs)
        det_feats = tuple(
            all_feats[i] for i in self.detection_feature_indices)

        if self.student.with_neck:
            det_feats = self.student.neck(det_feats)

        head_inputs_dict = self.student.forward_transformer(
            det_feats, batch_data_samples)
        results_list = self.student.bbox_head.predict(
            **head_inputs_dict, rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.student.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs, batch_data_samples):
        return self.student._forward(batch_inputs, batch_data_samples)

    def extract_feat(self, batch_inputs):
        return self.student.extract_feat(batch_inputs)

    # -----------------------------------------------------------------
    # Teacher input preparation
    # -----------------------------------------------------------------
    def _prepare_teacher_inputs(self, img_rgbs, device, pad_size_divisor=14):
        """Stack, normalize, and pad RGB images for the DINOv2 teacher.

        Args:
            img_rgbs: List of BGR numpy arrays from metainfo (loaded by
                mmcv.imread). Converted to RGB before normalization.
            device: Target device.
            pad_size_divisor: Pad to multiple of patch_size (14 for DINOv2).
        """
        processed = []
        # ImageNet normalization (RGB order)
        mean = torch.tensor(
            [123.675, 116.28, 103.53], device=device).view(3, 1, 1)
        std = torch.tensor(
            [58.395, 57.12, 57.375], device=device).view(3, 1, 1)

        for img in img_rgbs:
            if isinstance(img, torch.Tensor):
                t = img.float()
            else:
                # HWC BGR numpy → CHW tensor
                t = torch.from_numpy(img).permute(2, 0, 1).float()
            # BGR → RGB
            t = t[[2, 1, 0], ...]
            t = (t.to(device) - mean) / std
            processed.append(t)

        # Pad to uniform size (batch may have different sizes)
        max_h = max(t.shape[1] for t in processed)
        max_w = max(t.shape[2] for t in processed)

        # Round up to nearest multiple of pad_size_divisor
        max_h += (pad_size_divisor - max_h % pad_size_divisor) % pad_size_divisor
        max_w += (pad_size_divisor - max_w % pad_size_divisor) % pad_size_divisor

        padded = []
        for t in processed:
            pad_h = max_h - t.shape[1]
            pad_w = max_w - t.shape[2]
            if pad_h > 0 or pad_w > 0:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded.append(t)

        return torch.stack(padded)
