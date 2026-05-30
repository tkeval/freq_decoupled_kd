import copy
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from ..detectors.dino import DINO


@MODELS.register_module()
class GTMatchedResponseKD(DINO):
    """Response-Based KD for DINO with GT-Mediated Query Matching.

    Addresses the query misalignment problem in DINO-based cross-modal KD:
    - DINO generates queries from encoder top-k proposals (data-dependent)
    - Teacher (RGB) and student (IR) produce different proposals
    - Index-wise matching compares unrelated queries → noise

    Fix: Use GT boxes to mediate the matching:
    1. Hungarian-match teacher queries → GT → teacher_matched[gt_i] = query_j
    2. Hungarian-match student queries → GT → student_matched[gt_i] = query_k
    3. For each GT, distill teacher query_j → student query_k

    This ensures we always compare predictions for the SAME real object.
    """

    def __init__(self,
                 teacher_cfg: dict,
                 teacher_checkpoint: str,
                 kd_cls_weight: float = 1.0,
                 kd_bbox_weight: float = 2.0,
                 temperature: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)

        # --- Build & freeze teacher ---
        self.teacher = MODELS.build(copy.deepcopy(teacher_cfg))
        load_checkpoint(self.teacher, teacher_checkpoint, map_location='cpu')
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher._is_init = True
        self.teacher.is_init = True

        self.kd_cls_weight = kd_cls_weight
        self.kd_bbox_weight = kd_bbox_weight
        self.temperature = temperature

        print(f"[GTMatchedResponseKD] Teacher loaded from: {teacher_checkpoint}")
        print(f"[GTMatchedResponseKD] kd_cls_weight={kd_cls_weight}, "
              f"kd_bbox_weight={kd_bbox_weight}, temperature={temperature}")

    def train(self, mode=True):
        """Keep teacher in eval mode always."""
        super().train(mode)
        self.teacher.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Exclude frozen teacher from saved checkpoints."""
        sd = super().state_dict(*args, **kwargs)
        filtered = type(sd)({k: v for k, v in sd.items()
                             if not k.startswith('teacher.')})
        if hasattr(sd, '_metadata'):
            filtered._metadata = sd._metadata
        return filtered

    def load_state_dict(self, state_dict, strict=True):
        """Load student-only checkpoint; teacher loaded from its own ckpt."""
        student_sd = {k: v for k, v in state_dict.items()
                      if not k.startswith('teacher.')}
        return super().load_state_dict(student_sd, strict=False)

    def _prepare_teacher_rgb(self, batch_data_samples, device):
        """Prepare paired RGB images for the teacher."""
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

        teacher_data = self.teacher.data_preprocessor(
            {'inputs': tensors}, training=False)
        return teacher_data['inputs'].to(device)

    def _match_queries_to_gt(self, cls_scores, bbox_preds, gt_instances,
                             img_meta):
        """Hungarian-match predicted queries to GT boxes.

        Uses the same assigner as DINO's own loss function.

        Args:
            cls_scores: (num_queries, num_classes) raw logits
            bbox_preds: (num_queries, 4) normalized cxcywh (sigmoid output)
            gt_instances: InstanceData with .bboxes (xyxy abs) and .labels
            img_meta: dict with 'img_shape'

        Returns:
            gt_to_query: dict mapping gt_idx (0-based) → query_idx
        """
        gt_bboxes = gt_instances.bboxes
        if len(gt_bboxes) == 0:
            return {}

        # Convert predictions to xyxy pixel coords (matching DINO's convention)
        img_h, img_w = img_meta['img_shape']
        factor = bbox_preds.new_tensor([img_w, img_h, img_w, img_h])
        bbox_preds_xyxy = bbox_cxcywh_to_xyxy(bbox_preds) * factor

        pred_instances = InstanceData(
            scores=cls_scores, bboxes=bbox_preds_xyxy)

        # Use DINO's own assigner (HungarianAssigner)
        assign_result = self.bbox_head.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        # Extract GT → query mapping (gt_inds is 1-indexed, 0 = unmatched)
        gt_to_query = {}
        for gt_idx in range(len(gt_bboxes)):
            matched = (assign_result.gt_inds == gt_idx + 1).nonzero(
                as_tuple=True)[0]
            if len(matched) > 0:
                gt_to_query[gt_idx] = matched[0].item()

        return gt_to_query

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Compute detection loss + GT-matched response KD loss."""
        # =============================================================
        # 1. Student forward + detection loss
        # =============================================================
        student_feats = self.extract_feat(batch_inputs)
        student_head_inputs = self.forward_transformer(
            student_feats, batch_data_samples)
        det_losses = self.bbox_head.loss(
            **student_head_inputs, batch_data_samples=batch_data_samples)

        # Get student raw predictions (last decoder layer)
        student_outputs = self.bbox_head.forward(
            student_head_inputs['hidden_states'],
            student_head_inputs['references'])
        student_cls_all = student_outputs[0][-1]   # (bs, num_q+dn, C)
        student_bbox_all = student_outputs[1][-1]  # (bs, num_q+dn, 4)

        # Strip DN queries (prepended at beginning during training)
        dn_meta = student_head_inputs.get('dn_meta')
        num_dn = dn_meta.get('num_denoising_queries', 0) if dn_meta else 0
        student_cls = student_cls_all[:, num_dn:, :]
        student_bbox = student_bbox_all[:, num_dn:, :]

        # =============================================================
        # 2. Teacher forward on paired RGB images
        # =============================================================
        self.teacher.eval()
        teacher_rgb_inputs = self._prepare_teacher_rgb(
            batch_data_samples, batch_inputs.device)

        with torch.no_grad():
            teacher_feats = self.teacher.extract_feat(teacher_rgb_inputs)
            teacher_head_inputs = self.teacher.forward_transformer(
                teacher_feats, batch_data_samples)
            teacher_outputs = self.teacher.bbox_head.forward(
                teacher_head_inputs['hidden_states'],
                teacher_head_inputs['references'])
            teacher_cls = teacher_outputs[0][-1]   # (bs, num_q, C)
            teacher_bbox = teacher_outputs[1][-1]  # (bs, num_q, 4)

        # =============================================================
        # 3. GT-mediated matching + KD losses
        # =============================================================
        total_kd_cls = batch_inputs.new_tensor(0.0)
        total_kd_bbox = batch_inputs.new_tensor(0.0)
        num_matched = 0

        for b in range(len(batch_data_samples)):
            gt_instances = batch_data_samples[b].gt_instances
            img_meta = batch_data_samples[b].metainfo

            if len(gt_instances.bboxes) == 0:
                continue

            # Match teacher queries to GT (detached, no grad through matching)
            teacher_gt_match = self._match_queries_to_gt(
                teacher_cls[b].detach(),
                teacher_bbox[b].detach(),
                gt_instances, img_meta)

            # Match student queries to GT (detached for matching only)
            student_gt_match = self._match_queries_to_gt(
                student_cls[b].detach(),
                student_bbox[b].detach(),
                gt_instances, img_meta)

            # For each GT with both a teacher and student match, compute KD
            for gt_idx in teacher_gt_match:
                if gt_idx not in student_gt_match:
                    continue

                t_q = teacher_gt_match[gt_idx]
                s_q = student_gt_match[gt_idx]

                # Classification KD: temperature-scaled BCE
                t_logit = teacher_cls[b, t_q].detach()
                s_logit = student_cls[b, s_q]

                t_prob = torch.sigmoid(t_logit / self.temperature)
                kd_cls = F.binary_cross_entropy_with_logits(
                    s_logit / self.temperature, t_prob,
                    reduction='mean') * self.temperature ** 2
                total_kd_cls = total_kd_cls + kd_cls

                # Bounding box KD: L1 on normalized cxcywh
                t_box = teacher_bbox[b, t_q].detach()
                s_box = student_bbox[b, s_q]
                kd_bbox = F.l1_loss(s_box, t_box, reduction='mean')
                total_kd_bbox = total_kd_bbox + kd_bbox

                num_matched += 1

        # Average over matched pairs
        if num_matched > 0:
            det_losses['kd_loss_cls'] = (
                total_kd_cls / num_matched * self.kd_cls_weight)
            det_losses['kd_loss_bbox'] = (
                total_kd_bbox / num_matched * self.kd_bbox_weight)
        else:
            # Use sum of student predictions * 0 to keep gradient graph alive
            # (prevents DDP unused parameter warnings)
            dummy = (student_cls.sum() + student_bbox.sum()) * 0
            det_losses['kd_loss_cls'] = dummy
            det_losses['kd_loss_bbox'] = dummy.clone()

        return det_losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        """Prediction uses student only (no teacher at inference)."""
        return super().predict(batch_inputs, batch_data_samples, rescale)
