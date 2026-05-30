import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.detectors.dino import DINO
from mmdet.models.utils.lora import merge_lora_state_dict
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from typing import Dict, List, Union, Tuple


@MODELS.register_module()
class Stage2GuidedDetector(DINO):
    """
    Stage 2: Guided Detection Fine-Tuning.
    
    Extends the DINO detector with feature-level distillation from a frozen
    IR-ViT teacher (trained in Stage 1). The student is the full DINO detector
    and receives both detection losses and feature alignment losses.
    
    Architecture:
        IR image → Frozen IR-ViT Teacher → Teacher features (guidance)
        IR image → Student DINO Detector → Detection losses + Feature alignment
    
    Total Loss = Detection Loss + λ * Feature Alignment Loss
    """

    def __init__(self,
                 teacher_backbone_cfg: dict,
                 teacher_checkpoint: str,
                 distill_cfg: List[dict],
                 distill_weight: float = 0.5,
                 init_student_from_stage1: bool = True,
                 lora_merge_scaling: float = 1.0,
                 **kwargs):
        """
        Args:
            teacher_backbone_cfg: Config for the IR-ViT teacher backbone.
            teacher_checkpoint: Path to Stage 1 checkpoint.
            distill_cfg: List of feature alignment configs, each with:
                - name: Loss name for logging
                - student_feature_index: Index into student backbone output tuple
                - teacher_feature_index: Index into teacher backbone output tuple
                - student_channels: Channel dim of student feature
                - teacher_channels: Channel dim of teacher feature
                - loss_weight: Weight for this specific pair
            distill_weight: Global weight λ for all distillation losses.
            init_student_from_stage1: If True, initialize the student backbone
                from the Stage 1 checkpoint (overriding pretrained weights).
            **kwargs: All remaining args passed to DINO detector.
        """
        super().__init__(**kwargs)

        # -----------------------------------------------------------
        # 1. Load Stage 1 checkpoint (shared between teacher and student)
        # -----------------------------------------------------------
        ckpt = torch.load(teacher_checkpoint, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Extract student backbone weights from Stage 1 checkpoint
        # Stage 1 keys look like: "student.backbone.xxx"
        stage1_backbone_weights = {}
        prefix = 'student.backbone.'
        for key, value in state_dict.items():
            if key.startswith(prefix):
                stage1_backbone_weights[key[len(prefix):]] = value

        if not stage1_backbone_weights:
            raise RuntimeError(
                f"No keys found with prefix '{prefix}' in checkpoint "
                f"'{teacher_checkpoint}'. Available prefixes: "
                f"{set(k.split('.')[0] for k in state_dict.keys())}")

        # Auto-merge LoRA weights if present (from Stage 1 LoRA training)
        # scaling < 1.0 applies partial merge: W = W_orig + scaling * (B @ A)
        # This keeps features closer to pretrained DINOv2 while adding
        # a fraction of the cross-modal knowledge from Stage 1.
        stage1_backbone_weights = merge_lora_state_dict(
            stage1_backbone_weights, scaling=lora_merge_scaling)

        # -----------------------------------------------------------
        # 2. Initialize Student Backbone from Stage 1
        # -----------------------------------------------------------
        if init_student_from_stage1:
            missing, unexpected = self.backbone.load_state_dict(
                stage1_backbone_weights, strict=False)
            print(f"[Stage2] Student backbone initialized from Stage 1")
            print(f"[Stage2]   Missing: {len(missing)}, "
                  f"Unexpected: {len(unexpected)}")
            if missing:
                print(f"[Stage2]   Missing keys: {missing[:5]}...")
            if unexpected:
                print(f"[Stage2]   Unexpected keys: {unexpected[:5]}...")

        # -----------------------------------------------------------
        # 3. Build & Load Teacher Backbone (frozen IR-ViT from Stage 1)
        # -----------------------------------------------------------
        self.teacher_backbone = self._build_teacher_backbone(
            teacher_backbone_cfg, stage1_backbone_weights)

        # Freeze teacher
        self.teacher_backbone.eval()
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False

        # -----------------------------------------------------------
        # 2. Build Projectors & Distillation Losses
        # -----------------------------------------------------------
        self.distill_weight = distill_weight
        self.distill_projectors = nn.ModuleDict()
        self.distill_losses = nn.ModuleDict()
        self.distill_configs = []

        for i, d_cfg in enumerate(distill_cfg):
            name = d_cfg.get('name', f'loss_distill_{i}')
            s_dim = d_cfg['student_channels']
            t_dim = d_cfg['teacher_channels']

            # Projector: align student channels → teacher channels
            # Only needed if dimensions differ; identity if same
            if s_dim != t_dim:
                self.distill_projectors[name] = nn.Conv2d(
                    s_dim, t_dim, kernel_size=1)
            else:
                self.distill_projectors[name] = nn.Identity()

            # Loss: Cosine similarity (as per architecture diagram)
            self.distill_losses[name] = nn.CosineSimilarity(dim=1)

            # Store config with resolved name
            cfg_copy = d_cfg.copy()
            cfg_copy['loss_module_name'] = name
            self.distill_configs.append(cfg_copy)

    def _build_teacher_backbone(self, backbone_cfg, stage1_backbone_weights):
        """
        Build the teacher backbone and load Stage 1 weights.

        Args:
            backbone_cfg: Config dict for the teacher backbone architecture.
            stage1_backbone_weights: Pre-extracted state dict with backbone
                weights from the Stage 1 checkpoint.
        """
        backbone = MODELS.build(backbone_cfg)

        missing, unexpected = backbone.load_state_dict(
            stage1_backbone_weights, strict=False)
        print(f"[Stage2] Teacher backbone loaded from Stage 1 weights")
        print(f"[Stage2]   Missing: {len(missing)}, "
              f"Unexpected: {len(unexpected)}")
        if missing:
            print(f"[Stage2]   Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"[Stage2]   Unexpected keys: {unexpected[:5]}...")

        return backbone

    def train(self, mode=True):
        """Override to keep frozen teacher backbone in eval mode."""
        super().train(mode)
        self.teacher_backbone.eval()
        return self

    def extract_feat(self, batch_inputs):
        """Override to capture backbone features before neck.
        
        Stores raw backbone features in self._cached_backbone_feats
        so they can be reused for distillation without a second forward pass.
        """
        self._cached_backbone_feats = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(self._cached_backbone_feats)
        else:
            x = self._cached_backbone_feats
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """
        Compute detection loss + feature alignment loss.
        
        Single backbone forward pass is shared between detection and
        distillation via the cached backbone features from extract_feat.
        
        Args:
            batch_inputs: Tensor (B, C, H, W) - Preprocessed IR images
            batch_data_samples: List[DetDataSample] - GT annotations
        """
        # -----------------------------------------------------------
        # 1. Student Forward: Full DINO pipeline (single backbone pass)
        # -----------------------------------------------------------
        # extract_feat caches backbone features and passes through neck
        student_feats = self.extract_feat(batch_inputs)

        # Run DINO transformer (encoder + decoder)
        head_inputs_dict = self.forward_transformer(
            student_feats, batch_data_samples)

        # Compute detection losses (cls + bbox + giou)
        det_losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        # -----------------------------------------------------------
        # 2. Get backbone features for distillation
        # -----------------------------------------------------------
        # Reuse cached backbone features from extract_feat (no second pass!)
        student_backbone_feats = self._cached_backbone_feats

        # Teacher forward (frozen, no grad)
        with torch.no_grad():
            teacher_backbone_feats = self.teacher_backbone(batch_inputs)
            if isinstance(teacher_backbone_feats, torch.Tensor):
                teacher_backbone_feats = [teacher_backbone_feats]

        # -----------------------------------------------------------
        # 3. Feature Alignment Losses
        # -----------------------------------------------------------
        for d_cfg in self.distill_configs:
            name = d_cfg['loss_module_name']
            s_idx = d_cfg['student_feature_index']
            t_idx = d_cfg['teacher_feature_index']
            pair_weight = d_cfg.get('loss_weight', 1.0)

            s_feat = student_backbone_feats[s_idx]
            t_feat = teacher_backbone_feats[t_idx]

            # Project student features if needed
            s_feat_proj = self.distill_projectors[name](s_feat)

            # Spatial alignment if sizes differ
            if s_feat_proj.shape[-2:] != t_feat.shape[-2:]:
                s_feat_proj = F.interpolate(
                    s_feat_proj,
                    size=t_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False)

            # Cosine similarity loss: maximize similarity → minimize (1 - sim)
            cos_sim = self.distill_losses[name](s_feat_proj, t_feat)
            loss_val = (1 - cos_sim).mean()

            # Apply per-pair weight and global distill weight
            det_losses[name] = loss_val * pair_weight * self.distill_weight

        return det_losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        """Prediction uses the full student DINO detector (no teacher needed)."""
        return super().predict(batch_inputs, batch_data_samples, rescale)
