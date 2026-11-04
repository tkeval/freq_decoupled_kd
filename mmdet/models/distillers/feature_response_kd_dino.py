import copy
from typing import List, Union

import torch
import torch.nn as nn
import mmcv
from torch import Tensor
import torch.nn.functional as F
from mmengine.model import BaseModule, ModuleList
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS
from .. import DINO


@MODELS.register_module()
class FeatureResponseKDDINO(DINO):
    """Hybrid Knowledge Distillation wrapper for DINO.

    This class implements a knowledge distillation strategy that combines both
    feature-based and response-based distillation.
    - Feature-based: Uses adaption layers to align student's neck features
      with the teacher's, teaching the student to form similar internal
      representations.
    - Response-based: Matches the student's final classification logits and
      bounding box predictions to the teacher's, teaching the student to
      mimic the final output.
    """

    def __init__(self,
                 teacher_cfg,
                 teacher_checkpoint,
                 distill_cfg,
                 **kwargs):
        # The DINO base class __init__ uses the 'neck' kwarg but does not
        # save the config dict. We need it to determine the number of
        # channels for our feature adaption layers, so we capture it here.
        self.neck_cfg = kwargs.get('neck')

        super().__init__(**kwargs)
        self.teacher = MODELS.build(copy.deepcopy(teacher_cfg))
        load_checkpoint(self.teacher, teacher_checkpoint, map_location='cpu')

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher._is_init = True
        self.teacher.is_init = True

        self.distill_cfg = distill_cfg
        self.distill_loss_weight_cls = self.distill_cfg.get('loss_weight_cls', 1.0)
        self.distill_loss_weight_bbox = self.distill_cfg.get('loss_weight_bbox', 1.0)
        self.distill_loss_weight_feat = self.distill_cfg.get('loss_weight_feat', 1.0)
        self.temperature = self.distill_cfg.get('temperature', 2.0)

        # Create adaptation layers for feature distillation
        # The 'ChannelMapper' neck does not have an 'out_channels' attribute.
        # We access it from the neck_cfg dictionary instead.
        student_channels = self.neck_cfg['out_channels']
        teacher_channels = teacher_cfg['neck']['out_channels']
        self.adaption_layers = ModuleList([
            nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, stride=1)
            for i in range(self.neck_cfg['num_outs'])
        ])

        print(f"Distiller initialized with the following settings:")
        print(f"  - Teacher Checkpoint: {teacher_checkpoint}")
        print(f"  - Class Distill Loss Weight: {self.distill_loss_weight_cls}")
        print(f"  - BBox Distill Loss Weight: {self.distill_loss_weight_bbox}")
        print(f"  - Feature Distill Loss Weight: {self.distill_loss_weight_feat}")
        print(f"  - Temperature: {self.temperature}")

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: List['DetDataSample']) -> Union[dict, list]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # --- Step 1: Standard Student Loss on IR images ---
        losses = super().loss(batch_inputs, batch_data_samples)

        # --- Step 2: Manually Load, Augment, and Preprocess Teacher RGB Images ---
        teacher_augmented_imgs = []
        for sample in batch_data_samples:
            # Load the RGB image corresponding to the IR image
            rgb_path = sample.metainfo['img2_path']
            img = mmcv.imread(rgb_path, channel_order='rgb')

            # Apply the same spatial augmentations as the student
            target_shape = sample.metainfo['img_shape']
            img_resized = mmcv.imresize(img, (target_shape[1], target_shape[0]))

            if sample.metainfo.get('flip', False):
                img_flipped = mmcv.imflip(
                    img_resized, direction=sample.metainfo['flip_direction'])
                teacher_augmented_imgs.append(img_flipped.copy())
            else:
                teacher_augmented_imgs.append(img_resized.copy())

        # Convert augmented images to tensors
        teacher_tensors = [
            torch.from_numpy(img).permute(2, 0, 1)
            for img in teacher_augmented_imgs
        ]

        # Preprocess the batch of teacher images
        teacher_data_samples = copy.deepcopy(batch_data_samples)
        teacher_data_batch = {
            'inputs': teacher_tensors,
            'data_samples': teacher_data_samples
        }
        processed_teacher_data = self.teacher.data_preprocessor(
            teacher_data_batch, training=False)
        teacher_rgb_inputs = processed_teacher_data['inputs']
        teacher_data_samples = processed_teacher_data['data_samples']

        # --- Step 3: Get Teacher Features & Predictions ---
        with torch.no_grad():
            # Get teacher features from the neck
            teacher_feats = self.teacher.extract_feat(teacher_rgb_inputs)
            # Get teacher predictions from the head
            teacher_transformer_outputs = self.teacher.forward_transformer(
                teacher_feats, teacher_data_samples)
            teacher_head_outputs = self.teacher.bbox_head.forward(
                teacher_transformer_outputs['hidden_states'],
                teacher_transformer_outputs['references'])

        # --- Step 4: Get Student Features & Predictions ---
        student_feats = self.extract_feat(batch_inputs)
        student_transformer_outputs = self.forward_transformer(
            student_feats, batch_data_samples)
        student_head_outputs = self.bbox_head.forward(
            student_transformer_outputs['hidden_states'],
            student_transformer_outputs['references'])

        # --- Step 5: Calculate Distillation Losses ---

        # 5.1: Feature Distillation
        loss_distill_feat = 0.
        for i in range(self.neck_cfg['num_outs']):
            student_feat_adapted = self.adaption_layers[i](student_feats[i])
            loss_distill_feat += F.l1_loss(
                student_feat_adapted,
                teacher_feats[i],
                reduction='mean')
        loss_distill_feat = loss_distill_feat * self.distill_loss_weight_feat


        # 5.2: Response Distillation (Classification)
        student_logits = student_head_outputs[0][-1]
        teacher_logits = teacher_head_outputs[0][-1]

        student_logits = student_logits.squeeze(-1)
        teacher_logits = teacher_logits.squeeze(-1)

        with torch.no_grad():
            teacher_probs = torch.sigmoid(teacher_logits / self.temperature)

        loss_distill_cls = F.binary_cross_entropy_with_logits(
            student_logits / self.temperature,
            teacher_probs,
            reduction='mean')

        loss_distill_cls = loss_distill_cls * (
            self.distill_loss_weight_cls * self.temperature**2)

        # 5.3: Response Distillation (Bounding Box)
        student_bboxes = student_head_outputs[1][-1]
        teacher_bboxes = teacher_head_outputs[1][-1]

        loss_distill_bbox = F.l1_loss(
            student_bboxes,
            teacher_bboxes,
            reduction='mean'
        ) * self.distill_loss_weight_bbox

        losses['loss_distill_feat'] = loss_distill_feat
        losses['loss_distill_cls'] = loss_distill_cls
        losses['loss_distill_bbox'] = loss_distill_bbox

        return losses

