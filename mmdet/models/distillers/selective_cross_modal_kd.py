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
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.structures.bbox import BaseBoxes
from ..detectors.dino import DINO


@MODELS.register_module()
class SelectiveCrossModalKD(DINO):
    """GT-mediated cross-modal KD for DINO detector.

    Key insight: In two-stage DINO, teacher and student queries are
    data-dependent (generated from encoder top-k proposals). Since teacher
    sees RGB and student sees IR, query index i in teacher does NOT
    correspond to the same object as query index i in student.

    Solution: GT-mediated query matching. For each GT object:
      1. Find the teacher query with highest IoU to this GT
      2. Find the student query with highest IoU to this GT
      3. Distill only between these semantically matched pairs

    This ensures "teacher knowledge about person k" flows to
    "student representation of person k".

    Loss = Det_Loss(IR_student, GT)
         + lambda_feat * Feature_KD(student_neck, teacher_neck)
         + lambda_cls  * GT_Matched_KD_Cls(student_logits, teacher_logits)
         + lambda_bbox * GT_Matched_KD_BBox(student_boxes, teacher_boxes)
    """

    def __init__(self,
                 teacher_cfg: dict,
                 teacher_checkpoint: str,
                 distill_cfg: dict,
                 **kwargs):
        self._neck_cfg = kwargs.get('neck')
        super().__init__(**kwargs)

        # --- Build & freeze teacher ---
        self.teacher = MODELS.build(copy.deepcopy(teacher_cfg))
        load_checkpoint(self.teacher, teacher_checkpoint, map_location='cpu')
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher._is_init = True
        self.teacher.is_init = True

        # --- Distillation config ---
        self.loss_weight_feat = distill_cfg.get('loss_weight_feat', 0.05)
        self.loss_weight_cls = distill_cfg.get('loss_weight_cls', 0.1)
        self.loss_weight_bbox = distill_cfg.get('loss_weight_bbox', 0.1)
        self.temperature = distill_cfg.get('temperature', 2.0)
        self.teacher_iou_threshold = distill_cfg.get('teacher_iou_threshold', 0.3)

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

        # Store for potential reloading
        self._teacher_cfg = copy.deepcopy(teacher_cfg)
        self._teacher_checkpoint = teacher_checkpoint

        print(f"[SelectiveCrossModalKD] Teacher loaded from: {teacher_checkpoint}")
        print(f"[SelectiveCrossModalKD] GT-mediated query matching enabled")
        print(f"[SelectiveCrossModalKD] loss_weight_feat={self.loss_weight_feat}, "
              f"loss_weight_cls={self.loss_weight_cls}, "
              f"loss_weight_bbox={self.loss_weight_bbox}, "
              f"temperature={self.temperature}, "
              f"teacher_iou_threshold={self.teacher_iou_threshold}")

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
        student_sd = {k: v for k, v in state_dict.items()
                      if not k.startswith('teacher.')}
        return super().load_state_dict(student_sd, strict=False)

    def _gt_mediated_matching(self, t_cls, t_bbox, s_cls, s_bbox,
                              batch_data_samples):
        """Find GT-mediated query pairs between teacher and student.

        For each GT object, find the best teacher query and best student
        query (by IoU), then pair them for distillation.

        Args:
            t_cls: (bs, num_queries, C) teacher classification logits
            t_bbox: (bs, num_queries, 4) teacher bbox predictions (normalized cxcywh)
            s_cls: (bs, num_queries, C) student classification logits
            s_bbox: (bs, num_queries, 4) student bbox predictions (normalized cxcywh)
            batch_data_samples: GT instances + metadata

        Returns:
            matched_t_cls: (N_matched, C) teacher cls logits for matched pairs
            matched_s_cls: (N_matched, C) student cls logits for matched pairs
            matched_t_bbox: (N_matched, 4) teacher bbox for matched pairs
            matched_s_bbox: (N_matched, 4) student bbox for matched pairs
            matched_weights: (N_matched,) soft weight per pair (teacher IoU * conf)
            num_matched: int, total matched pairs
        """
        device = t_cls.device
        bs = t_cls.shape[0]

        all_t_cls, all_s_cls = [], []
        all_t_bbox, all_s_bbox = [], []
        all_weights = []

        for i in range(bs):
            gt_instances = batch_data_samples[i].gt_instances
            if len(gt_instances) == 0:
                continue

            gt_boxes = gt_instances.bboxes
            if isinstance(gt_boxes, BaseBoxes):
                gt_boxes = gt_boxes.tensor  # (num_gt, 4) absolute xyxy

            img_h, img_w = batch_data_samples[i].metainfo['img_shape'][:2]
            scale = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])

            # Convert teacher predictions to absolute xyxy
            t_boxes_abs = bbox_cxcywh_to_xyxy(t_bbox[i]) * scale
            # Convert student predictions to absolute xyxy
            s_boxes_abs = bbox_cxcywh_to_xyxy(s_bbox[i]) * scale

            # IoU: (num_queries, num_gt)
            t_ious = bbox_overlaps(t_boxes_abs, gt_boxes)
            s_ious = bbox_overlaps(s_boxes_abs, gt_boxes)

            # Teacher confidence per query
            t_conf = t_cls[i].sigmoid().max(dim=-1).values  # (num_queries,)

            num_gt = gt_boxes.shape[0]
            for g in range(num_gt):
                # Best teacher query for GT g
                t_iou_g = t_ious[:, g]  # (num_queries,)
                t_best_idx = t_iou_g.argmax()
                t_best_iou = t_iou_g[t_best_idx]

                # Best student query for GT g (no threshold)
                s_best_idx = s_ious[:, g].argmax()

                # Only require teacher to have good overlap with GT.
                # No student threshold — the student needs KD most when
                # it hasn't found the object yet. The argmax still picks
                # the closest student query.
                if t_best_iou < self.teacher_iou_threshold:
                    continue

                # Soft weight: teacher's quality for this GT
                weight = t_best_iou * t_conf[t_best_idx]

                all_t_cls.append(t_cls[i, t_best_idx])
                all_s_cls.append(s_cls[i, s_best_idx])
                all_t_bbox.append(t_bbox[i, t_best_idx])
                all_s_bbox.append(s_bbox[i, s_best_idx])
                all_weights.append(weight)

        if len(all_t_cls) == 0:
            # No matched pairs — return empty tensors
            C = t_cls.shape[-1]
            return (torch.zeros(0, C, device=device),
                    torch.zeros(0, C, device=device),
                    torch.zeros(0, 4, device=device),
                    torch.zeros(0, 4, device=device),
                    torch.zeros(0, device=device),
                    0)

        matched_t_cls = torch.stack(all_t_cls)    # (N, C)
        matched_s_cls = torch.stack(all_s_cls)    # (N, C)
        matched_t_bbox = torch.stack(all_t_bbox)  # (N, 4)
        matched_s_bbox = torch.stack(all_s_bbox)  # (N, 4)
        matched_weights = torch.stack(all_weights) # (N,)

        return (matched_t_cls, matched_s_cls,
                matched_t_bbox, matched_s_bbox,
                matched_weights, len(all_t_cls))

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Compute detection loss + GT-mediated cross-modal KD losses."""

        # =============================================================
        # 1. Student forward + detection loss
        # =============================================================
        student_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            student_feats, batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        # Student raw predictions for response KD
        student_cls_scores, student_bbox_preds = self.bbox_head.forward(
            head_inputs_dict['hidden_states'],
            head_inputs_dict['references'])

        # =============================================================
        # 2. Teacher forward on paired RGB
        # =============================================================
        self.teacher.eval()

        teacher_rgb_inputs = self._prepare_teacher_rgb(
            batch_data_samples, batch_inputs.device)

        with torch.no_grad():
            teacher_feats = self.teacher.extract_feat(teacher_rgb_inputs)
            teacher_head_inputs = self.teacher.forward_transformer(
                teacher_feats, batch_data_samples)
            teacher_cls_scores_all, teacher_bbox_preds_all = \
                self.teacher.bbox_head.forward(
                    teacher_head_inputs['hidden_states'],
                    teacher_head_inputs['references'])

        # =============================================================
        # 3. Strip DN queries from both teacher and student
        # =============================================================
        num_dn_student = head_inputs_dict.get('dn_meta', {}).get(
            'num_denoising_queries', 0) if head_inputs_dict.get('dn_meta') else 0
        num_dn_teacher = teacher_head_inputs.get('dn_meta', {}).get(
            'num_denoising_queries', 0) if teacher_head_inputs.get('dn_meta') else 0

        s_cls = student_cls_scores[-1][:, num_dn_student:, :]
        s_bbox = student_bbox_preds[-1][:, num_dn_student:, :]
        t_cls = teacher_cls_scores_all[-1][:, num_dn_teacher:, :]
        t_bbox = teacher_bbox_preds_all[-1][:, num_dn_teacher:, :]

        assert s_cls.shape[1] == t_cls.shape[1], (
            f"Query count mismatch after DN stripping: "
            f"student={s_cls.shape[1]}, teacher={t_cls.shape[1]}")

        # =============================================================
        # 4. Feature KD loss (spatially aligned — always applied)
        # =============================================================
        loss_feat = torch.tensor(0.0, device=batch_inputs.device)
        for i in range(self._num_neck_outs):
            adapted = self.adaption_layers[i](student_feats[i])
            if adapted.shape[-2:] != teacher_feats[i].shape[-2:]:
                adapted = F.interpolate(
                    adapted, size=teacher_feats[i].shape[-2:],
                    mode='bilinear', align_corners=False)
            loss_feat = loss_feat + F.l1_loss(
                adapted, teacher_feats[i], reduction='mean')
        losses['loss_distill_feat'] = loss_feat * self.loss_weight_feat

        # =============================================================
        # 5. GT-mediated Response KD (only matched query pairs)
        # =============================================================
        (matched_t_cls, matched_s_cls,
         matched_t_bbox, matched_s_bbox,
         matched_weights, num_matched) = self._gt_mediated_matching(
            t_cls, t_bbox, s_cls, s_bbox, batch_data_samples)

        if num_matched > 0:
            # --- Classification KD on matched pairs ---
            with torch.no_grad():
                teacher_probs = torch.sigmoid(
                    matched_t_cls / self.temperature)

            cls_loss = F.binary_cross_entropy_with_logits(
                matched_s_cls / self.temperature,
                teacher_probs,
                reduction='none')  # (N_matched, C)
            cls_loss_per_pair = cls_loss.mean(dim=-1)  # (N_matched,)

            # Weighted average over matched pairs
            weighted_cls = (cls_loss_per_pair * matched_weights).sum()
            loss_cls_kd = weighted_cls / matched_weights.sum().clamp(min=1.0)

            losses['loss_distill_cls'] = (
                loss_cls_kd * self.loss_weight_cls * self.temperature ** 2)

            # --- BBox KD on matched pairs ---
            bbox_loss = F.l1_loss(
                matched_s_bbox, matched_t_bbox,
                reduction='none')  # (N_matched, 4)
            bbox_loss_per_pair = bbox_loss.mean(dim=-1)  # (N_matched,)

            weighted_bbox = (bbox_loss_per_pair * matched_weights).sum()
            loss_bbox_kd = weighted_bbox / matched_weights.sum().clamp(min=1.0)

            losses['loss_distill_bbox'] = loss_bbox_kd * self.loss_weight_bbox
        else:
            # No GT matches — zero KD loss for this batch
            losses['loss_distill_cls'] = torch.tensor(
                0.0, device=batch_inputs.device)
            losses['loss_distill_bbox'] = torch.tensor(
                0.0, device=batch_inputs.device)

        return losses

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

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        """Prediction uses student only (no teacher needed at inference)."""
        return super().predict(batch_inputs, batch_data_samples, rescale)
