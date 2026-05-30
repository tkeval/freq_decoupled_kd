"""
Frequency-Decoupled Cross-Modal KD for CNN backbones (ResNet, etc.).

Same FFT decomposition as FreqDecoupledDistiller but designed for
CNN backbones where:
  - No LoRA needed (backbone fine-tuned directly with reduced LR)
  - Teacher backbone built via mmdet's MODELS registry
  - Multi-scale features with different spatial sizes and channels

Stage 1: RGB teacher (ImageNet-pretrained ResNet-50) provides structural
guidance to IR student (ResNet-50) via frequency-decoupled loss.
Stage 2: Standard Faster R-CNN training with the adapted backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from typing import Dict, List, Optional, Union


@MODELS.register_module()
class FreqDecoupledDistillerCNN(BaseDetector):
    """Stage 1: Frequency-Decoupled Cross-Modal KD for CNN backbones.

    Args:
        student_cfg: Config for the student detector (e.g., Faster R-CNN).
        teacher_backbone_cfg: Config for the teacher backbone (e.g., ResNet-50).
            Built via mmdet's MODELS registry.
        distill_cfg: Layer pairs for distillation. Each entry:
            - name (str): Loss name for logging
            - student_feature_index (int): Index into student backbone outputs
            - teacher_feature_index (int): Index into teacher backbone outputs
            - student_channels (int): Student feature channels
            - teacher_channels (int): Teacher feature channels
            - loss_weight (float): Weight for this layer pair
        freq_cutoff: Fraction of spectrum for low-frequency band (default 0.5).
        high_freq_weight: Multiplier for high-freq logMSE loss (default 0.1).
        data_preprocessor: Data preprocessor config.
        init_cfg: Initialization config.
    """

    def __init__(self,
                 student_cfg: dict,
                 teacher_backbone_cfg: dict,
                 distill_cfg: List[dict],
                 freq_cutoff: float = 0.5,
                 high_freq_weight: float = 0.1,
                 data_preprocessor: dict = None,
                 init_cfg: dict = None):
        if data_preprocessor is None:
            data_preprocessor = student_cfg.get('data_preprocessor', None)

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.freq_cutoff = freq_cutoff
        self.high_freq_weight = high_freq_weight

        # ----- Student detector (Faster R-CNN, etc.) -----
        self.student = MODELS.build(student_cfg)

        # ----- Teacher backbone (frozen, pretrained on ImageNet RGB) -----
        self.teacher_backbone = MODELS.build(teacher_backbone_cfg)
        self.teacher_backbone._is_init = True
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
    # Frequency decomposition (same as FreqDecoupledDistiller)
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

            # Project if channels differ
            if name in self.projectors:
                s_feat = self.projectors[name](s_feat)

            # Resize if spatial dimensions mismatch
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
    # Teacher input preparation
    # -----------------------------------------------------------------
    def _prepare_teacher_inputs(self, img_rgbs, device, pad_size_divisor=32):
        """Stack, normalize, and pad RGB images for the teacher backbone.

        Args:
            img_rgbs: List of BGR numpy arrays from metainfo.
            device: Target device.
            pad_size_divisor: Pad to multiple of this (32 for ResNet/FPN).
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


@MODELS.register_module()
class Stage2FasterRCNN(BaseDetector):
    """Stage 2: Faster R-CNN with backbone initialized from Stage 1.

    Loads Stage 1 backbone weights (from FreqDecoupledDistillerCNN checkpoint)
    into a standard Faster R-CNN detector. No teacher or KD during Stage 2 —
    just standard detection training.

    Args:
        stage1_checkpoint: Path to Stage 1 checkpoint.
        detector_cfg: Full config for the Faster R-CNN detector.
    """

    def __init__(self,
                 stage1_checkpoint: str,
                 detector_cfg: dict,
                 backbone_merge_scale: float = 0.5,
                 data_preprocessor: dict = None,
                 init_cfg: dict = None):
        """
        Args:
            stage1_checkpoint: Path to Stage 1 checkpoint.
            detector_cfg: Full config for the Faster R-CNN detector.
            backbone_merge_scale: Blending factor for Stage 1 vs pretrained
                weights: final = (1 - scale) * pretrained + scale * stage1.
                This is analogous to LoRA merge scaling. Default 0.5.
        """
        if data_preprocessor is None:
            data_preprocessor = detector_cfg.get('data_preprocessor', None)

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build the full detector
        self.detector = MODELS.build(detector_cfg)

        # Load Stage 1 backbone weights (merged with pretrained)
        self._load_stage1_backbone(stage1_checkpoint, backbone_merge_scale)

    def _load_stage1_backbone(self, checkpoint_path, merge_scale=0.5):
        """Load backbone weights from Stage 1 checkpoint, blended with pretrained.

        final_weight = (1 - merge_scale) * pretrained + merge_scale * stage1

        This is conceptually identical to LoRA merge scaling: full fine-tuning
        in Stage 1 drifts the backbone far from pretrained, so we interpolate
        to preserve detection-useful features while incorporating the cross-modal
        adaptation.
        """
        import logging
        logger = logging.getLogger('mmengine')

        # 1. Load ImageNet pretrained ResNet-50 weights explicitly
        #    (init_cfg is NOT applied during __init__, only later in
        #     init_weights(), so backbone.state_dict() would be random here)
        from mmengine.runner import load_checkpoint
        pretrained_sd = load_checkpoint(
            self.detector.backbone,
            'torchvision://resnet50',
            map_location='cpu')
        # load_checkpoint loads into the model; now save the state
        pretrained_sd = {
            k: v.clone() for k, v in
            self.detector.backbone.state_dict().items()}
        logger.info("[Stage2] Loaded ImageNet pretrained ResNet-50 as base")

        # 2. Extract Stage 1 backbone weights
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        backbone_weights = {}
        prefix = 'student.backbone.'
        for k, v in state_dict.items():
            if k.startswith(prefix):
                backbone_weights[k[len(prefix):]] = v

        if not backbone_weights:
            raise RuntimeError(
                f"No backbone weights found with prefix '{prefix}' "
                f"in checkpoint '{checkpoint_path}'")

        # 3. Blend: (1 - scale) * pretrained + scale * stage1
        merged_weights = {}
        for k in pretrained_sd:
            if k in backbone_weights:
                merged_weights[k] = (
                    (1 - merge_scale) * pretrained_sd[k] +
                    merge_scale * backbone_weights[k].to(pretrained_sd[k].dtype))
            else:
                merged_weights[k] = pretrained_sd[k]

        missing, unexpected = self.detector.backbone.load_state_dict(
            merged_weights, strict=False)

        # Prevent mmengine from re-initializing the backbone
        self.detector.backbone._is_init = True

        logger.info(
            f"[Stage2] Merged {len(backbone_weights)} Stage 1 backbone "
            f"weights with pretrained (scale={merge_scale})")
        logger.info(
            f"[Stage2]   Missing: {len(missing)}, "
            f"Unexpected: {len(unexpected)}")
        if missing:
            logger.warning(f"[Stage2]   Missing keys: {missing[:5]}...")

    def loss(self, batch_inputs, batch_data_samples):
        return self.detector.loss(batch_inputs, batch_data_samples)

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        return self.detector.predict(
            batch_inputs, batch_data_samples, rescale)

    def _forward(self, batch_inputs, batch_data_samples):
        return self.detector._forward(batch_inputs, batch_data_samples)

    def extract_feat(self, batch_inputs):
        return self.detector.extract_feat(batch_inputs)
