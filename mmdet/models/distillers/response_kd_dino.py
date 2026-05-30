import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from mmdet.models.utils import multi_apply
import numpy as np
from PIL import Image
import cv2

from mmdet.models.utils.misc import unpack_gt_instances
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor


@MODELS.register_module()
class ResponseKDDINO(DINO):

    def __init__(self,
                 teacher_cfg,
                 teacher_checkpoint,
                 distill_cfg,
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = MODELS.build(teacher_cfg)
        self.load_checkpoint(teacher_checkpoint, self.teacher_model)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        self.teacher_model._is_init = True
        self.teacher_model.is_init = True

        self.distill_cfg = distill_cfg
        self.distill_losses = nn.ModuleDict()
        for cfg in self.distill_cfg:
            # Make a copy to avoid modifying the original config
            cfg = cfg.copy()
            # Extract the name (or loss_name) to use as the key
            loss_name = cfg.pop('name', cfg.pop('loss_name', None))
            if loss_name is None:
                raise ValueError("Distillation loss config must have a 'name' or 'loss_name' field.")
            
            # Build the loss module with the remaining config
            self.distill_losses[loss_name] = MODELS.build(cfg)

    def load_checkpoint(self, checkpoint, model):
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
        print(f"Teacher checkpoint loaded from {checkpoint}")

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        
        # -------------------------
        # 1. Student Forward & Loss
        # -------------------------
        # We need to manually run the student's forward pass to get the raw
        # outputs for distillation. calling super().loss() gives us the scalar
        # losses but not the predictions.
        
        student_feats = self.extract_feat(batch_inputs)
        student_head_inputs_dict = self.forward_transformer(student_feats, batch_data_samples)
        
        # Compute student's supervised loss
        student_losses = self.bbox_head.loss(
            **student_head_inputs_dict, batch_data_samples=batch_data_samples)
        
        # Get student's raw predictions for distillation
        # The forward method returns a tuple (all_cls_scores, all_bbox_preds)
        # DINOHead.forward only accepts hidden_states and references
        student_preds_tuple = self.bbox_head.forward(
            hidden_states=student_head_inputs_dict['hidden_states'],
            references=student_head_inputs_dict['references'])

        # -------------------------
        # 2. Teacher Forward
        # -------------------------
        # Extract RGB images from metainfo and stack them
        img_rgbs = [data_sample.metainfo['img_rgb'] for data_sample in batch_data_samples]
        
        teacher_batch_inputs = []
        for img in img_rgbs:
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            elif isinstance(img, torch.Tensor):
                img_tensor = img
            else:
                raise TypeError(f"Unsupported type for img_rgb: {type(img)}")
            teacher_batch_inputs.append(img_tensor)
            
        teacher_batch_inputs = torch.stack(teacher_batch_inputs)
        teacher_batch_inputs = teacher_batch_inputs.to(batch_inputs.device)

        with torch.no_grad():
            # Preprocess: normalization etc.
            teacher_data = self.teacher_model.data_preprocessor(
                {'inputs': teacher_batch_inputs}, training=False)
            
            # Extract features
            teacher_feats = self.teacher_model.extract_feat(teacher_data['inputs'])
            
            # Run Transformer
            # Important: For teacher (eval mode), forward_transformer handles DN queries correctly
            # (skips them if not training) or we should assume it does.
            # DINO forward_transformer uses batch_data_samples for DN.
            # Since teacher_model is in eval(), DINO logic should skip DN.
            teacher_head_inputs_dict = self.teacher_model.forward_transformer(
                teacher_feats, batch_data_samples)
            
            # Get predictions
            # DINOHead.forward only accepts hidden_states and references
            teacher_preds_tuple = self.teacher_model.bbox_head.forward(
                hidden_states=teacher_head_inputs_dict['hidden_states'],
                references=teacher_head_inputs_dict['references'])

        # -------------------------
        # 3. Distillation Loss
        # -------------------------
        distill_losses = self.calculate_distillation_losses(
            student_preds_tuple, teacher_preds_tuple)
        
        student_losses.update(distill_losses)
        return student_losses

    def calculate_distillation_losses(self, student_preds, teacher_preds):
        """Calculate distillation losses.
        
        Args:
            student_preds (tuple): (all_cls_scores, all_bbox_preds) from DINOHead
            teacher_preds (tuple): (all_cls_scores, all_bbox_preds) from DINOHead
        """
        # Unpack predictions
        # DINOHead returns (all_layers_cls_scores, all_layers_bbox_preds)
        student_cls_scores, student_bbox_preds = student_preds
        teacher_cls_scores, teacher_bbox_preds = teacher_preds

        # Filter out DN queries from student predictions if necessary
        # Student predictions shape: (num_layers, bs, num_queries + num_dn, ...)
        # Teacher predictions shape: (num_layers, bs, num_queries, ...)
        # We assume DN queries are appended at the end or prepended.
        # In standard DINO, DN queries are usually concatenated to the learnable content queries.
        # But DINO has two-stage queries.
        
        # Let's check shapes to be sure.
        # If student has more queries than teacher, we assume the matching queries 
        # correspond to the teacher's queries.
        # In DINO, DN queries are usually concatenated *before* the content queries (or mixed?).
        # Actually, `pre_decoder` concatenates `query = torch.cat([dn_query, query])`.
        # So DN queries are at the BEGINNING.
        
        if student_cls_scores.shape[2] > teacher_cls_scores.shape[2]:
            # Student has DN queries. Teacher does not (eval mode).
            # We need to slice the student outputs to match the teacher.
            # Assuming DN queries are at the beginning (indices 0 to N_dn-1).
            # And matching queries are at the end.
            num_teacher_queries = teacher_cls_scores.shape[2]
            # Take the last N queries
            student_cls_scores = student_cls_scores[:, :, -num_teacher_queries:, :]
            student_bbox_preds = student_bbox_preds[:, :, -num_teacher_queries:, :]
            
        elif student_cls_scores.shape[2] < teacher_cls_scores.shape[2]:
             # This shouldn't happen normally unless config mismatch
             pass

        distill_losses = {}
        
        # Distill classification scores
        loss_distill_cls = self.distill_losses['loss_distill_cls'](
            student_cls_scores, teacher_cls_scores)
        distill_losses['loss_distill_cls'] = loss_distill_cls
        
        # Distill bounding box predictions
        loss_distill_bbox = self.distill_losses['loss_distill_bbox'](
            student_bbox_preds, teacher_bbox_preds)
        distill_losses['loss_distill_bbox'] = loss_distill_bbox
        
        return distill_losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        # For simplicity, we only use the student for predictions
        return super().predict(batch_inputs, batch_data_samples, rescale)
