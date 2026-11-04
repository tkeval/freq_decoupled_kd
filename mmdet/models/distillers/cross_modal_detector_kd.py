import copy
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.model import ModuleList
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from ..detectors.dino import DINO


@MODELS.register_module()
class CrossModalDetectorKD(DINO):
    """Single-stage cross-modal knowledge distillation for DINO detector.

    A pre-trained RGB DINO detector acts as teacher, guiding an IR DINO
    student with combined detection + feature + response distillation losses.

    Loss = Det_Loss(IR_student, GT)
         + lambda_feat * Feature_KD(student_neck, teacher_neck)
         + lambda_cls  * Response_KD_Cls(student_logits, teacher_logits)
         + lambda_bbox * Response_KD_BBox(student_boxes, teacher_boxes)

    The teacher processes paired RGB images (from metainfo['img_rgb']),
    while the student processes IR images. Both share the same architecture.
    """

    def __init__(self,
                 teacher_cfg: dict,
                 teacher_checkpoint: str,
                 distill_cfg: dict,
                 **kwargs):
        """
        Args:
            teacher_cfg: Full DINO model config for the RGB teacher.
            teacher_checkpoint: Path to the trained RGB DINO checkpoint.
            distill_cfg: Dict with distillation hyperparameters:
                - loss_weight_feat (float): Weight for feature KD. Default: 0.5
                - loss_weight_cls (float): Weight for classification KD. Default: 0.25
                - loss_weight_bbox (float): Weight for bbox KD. Default: 0.25
                - temperature (float): Temperature for soft labels. Default: 2.0
            **kwargs: All remaining args passed to DINO.__init__().
        """
        # Capture neck config before super().__init__ consumes it
        self._neck_cfg = kwargs.get('neck')
        super().__init__(**kwargs)

        # --- Build & freeze teacher ---
        self.teacher = MODELS.build(copy.deepcopy(teacher_cfg))
        load_checkpoint(self.teacher, teacher_checkpoint, map_location='cpu')
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        # CRITICAL: Prevent runner's init_weights() from reinitializing
        # the teacher. mmengine checks `is_init` (no underscore) but the
        # attribute is `_is_init`, so we must set BOTH to be safe.
        self.teacher._is_init = True
        self.teacher.is_init = True

        # --- Distillation config ---
        self.loss_weight_feat = distill_cfg.get('loss_weight_feat', 0.5)
        self.loss_weight_cls = distill_cfg.get('loss_weight_cls', 0.25)
        self.loss_weight_bbox = distill_cfg.get('loss_weight_bbox', 0.25)
        self.temperature = distill_cfg.get('temperature', 2.0)

        # --- Feature adaptation layers (student neck → teacher neck) ---
        student_channels = self._neck_cfg['out_channels']
        teacher_channels = teacher_cfg['neck']['out_channels']
        num_outs = self._neck_cfg['num_outs']
        self.adaption_layers = ModuleList([
            nn.Conv2d(student_channels, teacher_channels,
                      kernel_size=1, stride=1)
            for _ in range(num_outs)
        ])
        self._num_neck_outs = num_outs

        # Store teacher checkpoint path for reloading after checkpoint restore
        self._teacher_cfg = copy.deepcopy(teacher_cfg)
        self._teacher_checkpoint = teacher_checkpoint

        print(f"[CrossModalDetectorKD] Teacher loaded from: {teacher_checkpoint}")
        print(f"[CrossModalDetectorKD] loss_weight_feat={self.loss_weight_feat}, "
              f"loss_weight_cls={self.loss_weight_cls}, "
              f"loss_weight_bbox={self.loss_weight_bbox}, "
              f"temperature={self.temperature}")

    def state_dict(self, *args, **kwargs):
        """Exclude frozen teacher from saved checkpoints to save memory."""
        sd = super().state_dict(*args, **kwargs)
        filtered = type(sd)({k: v for k, v in sd.items()
                             if not k.startswith('teacher.')})
        if hasattr(sd, '_metadata'):
            filtered._metadata = sd._metadata
        return filtered

    def load_state_dict(self, state_dict, strict=True):
        """Load student-only checkpoint; teacher is loaded from its own ckpt."""
        # Filter out any teacher keys and load student parts only
        student_sd = {k: v for k, v in state_dict.items()
                      if not k.startswith('teacher.')}
        return super().load_state_dict(student_sd, strict=False)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Compute detection loss + cross-modal KD losses.

        Args:
            batch_inputs: (B, C, H, W) preprocessed IR images.
            batch_data_samples: List[DetDataSample] with GT + metainfo['img_rgb'].
        """
        # =============================================================
        # 1. Student forward (single pass) + detection loss
        # =============================================================
        student_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            student_feats, batch_data_samples)

        # Detection losses (cls + bbox + giou + dn losses)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        # Skip all KD computation when all weights are zero
        # (gives identical training to baseline DINO)
        kd_active = (self.loss_weight_feat > 0 or
                     self.loss_weight_cls > 0 or
                     self.loss_weight_bbox > 0)
        if not kd_active:
            losses['loss_distill_feat'] = torch.tensor(
                0.0, device=batch_inputs.device)
            losses['loss_distill_cls'] = torch.tensor(
                0.0, device=batch_inputs.device)
            losses['loss_distill_bbox'] = torch.tensor(
                0.0, device=batch_inputs.device)
            return losses

        # Student raw predictions for response KD
        student_cls_scores, student_bbox_preds = self.bbox_head.forward(
            head_inputs_dict['hidden_states'],
            head_inputs_dict['references'])

        # =============================================================
        # 2. Teacher forward on paired RGB images
        # =============================================================
        # CRITICAL: runner.train() sets ALL submodules to training mode,
        # including the frozen teacher. We must ensure teacher is in eval
        # mode so it behaves like inference (no DN queries, no dropout).
        self.teacher.eval()

        teacher_rgb_inputs = self._prepare_teacher_rgb(
            batch_data_samples, batch_inputs.device)

        with torch.no_grad():
            teacher_feats = self.teacher.extract_feat(teacher_rgb_inputs)
            teacher_head_inputs = self.teacher.forward_transformer(
                teacher_feats, batch_data_samples)
            teacher_cls_scores, teacher_bbox_preds = \
                self.teacher.bbox_head.forward(
                    teacher_head_inputs['hidden_states'],
                    teacher_head_inputs['references'])

        # =============================================================
        # 3. Feature KD loss (neck-level)
        # =============================================================
        loss_feat = torch.tensor(0.0, device=batch_inputs.device)
        for i in range(self._num_neck_outs):
            adapted = self.adaption_layers[i](student_feats[i])
            # Spatially align if sizes differ
            if adapted.shape[-2:] != teacher_feats[i].shape[-2:]:
                adapted = F.interpolate(
                    adapted, size=teacher_feats[i].shape[-2:],
                    mode='bilinear', align_corners=False)
            loss_feat = loss_feat + F.l1_loss(
                adapted, teacher_feats[i], reduction='mean')
        losses['loss_distill_feat'] = loss_feat * self.loss_weight_feat

        # =============================================================
        # 4. Response KD losses (head-level, final decoder layer only)
        # =============================================================
        # Strip DN queries (prepended by DINO during training)
        num_dn_student = head_inputs_dict.get('dn_meta', {}).get(
            'num_denoising_queries', 0) if head_inputs_dict.get('dn_meta') else 0
        num_dn_teacher = teacher_head_inputs.get('dn_meta', {}).get(
            'num_denoising_queries', 0) if teacher_head_inputs.get('dn_meta') else 0

        s_cls = student_cls_scores[-1][:, num_dn_student:, :]
        s_bbox = student_bbox_preds[-1][:, num_dn_student:, :]
        t_cls = teacher_cls_scores[-1][:, num_dn_teacher:, :]
        t_bbox = teacher_bbox_preds[-1][:, num_dn_teacher:, :]

        # Classification KD: temperature-scaled BCE
        with torch.no_grad():
            teacher_probs = torch.sigmoid(t_cls / self.temperature)
        loss_cls_kd = F.binary_cross_entropy_with_logits(
            s_cls / self.temperature,
            teacher_probs,
            reduction='mean')
        losses['loss_distill_cls'] = (
            loss_cls_kd * self.loss_weight_cls * self.temperature ** 2)

        # BBox KD: L1 on normalized coordinates
        loss_bbox_kd = F.l1_loss(s_bbox, t_bbox, reduction='mean')
        losses['loss_distill_bbox'] = loss_bbox_kd * self.loss_weight_bbox

        return losses

    def _prepare_teacher_rgb(self, batch_data_samples, device):
        """Prepare paired RGB images for the teacher.

        Extracts img_rgb from metainfo, converts to tensors, and runs
        through the teacher's data_preprocessor for normalization/padding.
        """
        img_rgbs = [s.metainfo['img_rgb'] for s in batch_data_samples]

        tensors = []
        for img in img_rgbs:
            if isinstance(img, np.ndarray):
                t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            elif isinstance(img, torch.Tensor):
                t = img
            else:
                raise TypeError(f"Unsupported img_rgb type: {type(img)}")
            tensors.append(t)

        # Use teacher's data_preprocessor for normalization + padding
        teacher_data = self.teacher.data_preprocessor(
            {'inputs': tensors}, training=False)
        return teacher_data['inputs'].to(device)

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        """Prediction uses student only (no teacher needed at inference)."""
        return super().predict(batch_inputs, batch_data_samples, rescale)
