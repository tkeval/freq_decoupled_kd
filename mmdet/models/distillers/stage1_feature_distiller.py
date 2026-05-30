import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmengine.runner import load_checkpoint
from mmdet.structures import SampleList
from typing import Dict, List, Optional, Union, Tuple
from mmdet.models.utils.lora import inject_lora

@MODELS.register_module()
class Stage1FeatureDistiller(BaseDetector):
    """
    Stage 1 Distiller: Multi-Teacher Knowledge Distillation.

    Optionally injects LoRA adapters into the student backbone to preserve
    pretrained features while learning modality-specific adaptations.
    When lora_cfg is provided, the student backbone is frozen and only
    LoRA parameters + projectors are trained.
    """
    def __init__(self,
                 student_cfg: dict,
                 teacher_cfgs: Dict[str, dict],
                 distill_cfg: List[dict] = None,
                 lora_cfg: Optional[dict] = None,
                 detection_feature_indices: Optional[List[int]] = None,
                 data_preprocessor: dict = None,
                 init_cfg: dict = None):
        """
        Args:
            student_cfg: Config for the student detector.
            teacher_cfgs: Dict of teacher configs keyed by name.
            distill_cfg: List of distillation loss configs.
            lora_cfg: Optional LoRA config. If provided, injects LoRA
                adapters into the student backbone and freezes original
                weights. Keys:
                - rank (int): LoRA rank. Default: 16.
                - alpha (float): Scaling factor. Default: 16.0.
                - dropout (float): LoRA dropout. Default: 0.05.
                - target_modules (list): Module name suffixes to target.
                    Default: ['attn.qkv'].
            detection_feature_indices: Indices into backbone output tuple
                to select for the detection neck/head during predict().
                Needed when backbone outputs more features (for distillation)
                than the detector neck expects. E.g., [3, 7, 9, 10, 11]
                selects 5 of 12 features. If None, all features are used.
            data_preprocessor: Data preprocessor config.
            init_cfg: Initialization config.
        """
        # If data_preprocessor is not provided, we try to use the student's
        if data_preprocessor is None:
            data_preprocessor = student_cfg.get('data_preprocessor', None)

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.detection_feature_indices = detection_feature_indices

        # -----------------------------------------------------------
        # 1. Build Student (The Detection Model)
        # -----------------------------------------------------------
        self.student = MODELS.build(student_cfg)

        # -----------------------------------------------------------
        # 1b. Inject LoRA adapters (if configured)
        # -----------------------------------------------------------
        if lora_cfg is not None:
            self._inject_lora(lora_cfg)

        # -----------------------------------------------------------
        # 2. Build Teachers
        # -----------------------------------------------------------
        self.teachers = nn.ModuleDict()
        self.teacher_types = {}

        for name, cfg in teacher_cfgs.items():
            model, t_type = self._build_teacher(cfg)
            self.teachers[name] = model
            self.teacher_types[name] = t_type

            # Freeze teacher
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        # -----------------------------------------------------------
        # 3. Build Projectors & Losses
        # -----------------------------------------------------------
        self.distill_losses = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        if distill_cfg:
            self._init_distill_losses(distill_cfg)

    def train(self, mode=True):
        """Override to keep frozen teachers in eval mode."""
        super().train(mode)
        for model in self.teachers.values():
            model.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Exclude frozen teachers from checkpoints to prevent OOM."""
        sd = super().state_dict(*args, **kwargs)
        filtered = type(sd)({k: v for k, v in sd.items()
                             if not k.startswith('teachers.')})
        if hasattr(sd, '_metadata'):
            filtered._metadata = sd._metadata
        return filtered

    def load_state_dict(self, state_dict, strict=True):
        """Load student-only checkpoint; teachers are loaded from pretrained."""
        student_sd = {k: v for k, v in state_dict.items()
                      if not k.startswith('teachers.')}
        return super().load_state_dict(student_sd, strict=False)

    def _inject_lora(self, lora_cfg: dict):
        """Inject LoRA adapters into the student backbone.

        Freezes all original backbone parameters and injects trainable
        LoRA adapters into the specified target modules.
        """
        rank = lora_cfg.get('rank', 16)
        alpha = lora_cfg.get('alpha', 16.0)
        dropout = lora_cfg.get('dropout', 0.05)
        target_modules = lora_cfg.get('target_modules', ['attn.qkv'])

        # Freeze the entire student backbone first
        for param in self.student.backbone.parameters():
            param.requires_grad = False

        # Inject LoRA into the backbone
        num_replaced = inject_lora(
            self.student.backbone,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout)

        # Count trainable vs total params
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

    def _forward(self, batch_inputs, batch_data_samples):
        """
        Implementation of the abstract method _forward from BaseDetector.
        This is used for 'tensor' mode (returning raw features/outputs).
        """
        return self.student._forward(batch_inputs, batch_data_samples)

    def extract_feat(self, batch_inputs):
        """
        Implementation of the abstract method extract_feat from BaseDetector.
        """
        return self.student.extract_feat(batch_inputs)

    def _build_teacher(self, cfg):
        """Build teacher model. Supports mmpretrain and custom wrappers."""
        if cfg.get('source', 'mmpretrain') == 'mmpretrain':
            # Use mmpretrain models
            # We usually just need the backbone
            from mmpretrain.models import build_backbone
            if 'backbone' in cfg:
                model = build_backbone(cfg['backbone'])
            else:
                # Try building as a classifier and strip head? 
                # Or just build whatever is passed
                # For CLIP, it might be a full model
                from mmpretrain.models import build_model
                model = build_model(cfg)
            return model, 'mmpretrain'
        else:
            # Fallback for external models (e.g. SAM2 from torch.hub)
            # This is a placeholder for external loading logic
            raise NotImplementedError("Only mmpretrain source currently supported in this snippet.")

    def _init_distill_losses(self, distill_configs):
        """
        Initialize projectors and loss functions.
        """
        for i, d_cfg in enumerate(distill_configs):
            name = d_cfg.get('name', f'loss_{i}')
            s_dim = d_cfg['student_channels']
            t_dim = d_cfg['teacher_channels']
            
            # 1. Projector (Linear 1x1 conv)
            # Maps Student (B, C_s, H, W) -> Teacher (B, C_t, H, W)
            self.projectors[name] = nn.Conv2d(s_dim, t_dim, kernel_size=1)
            
            # 2. Loss Module
            loss_type = d_cfg.get('loss_type', 'MSELoss')
            weight = d_cfg.get('loss_weight', 1.0)
            
            if loss_type == 'MSELoss':
                self.distill_losses[name] = nn.MSELoss()
            elif loss_type == 'CosineSimilarity':
                self.distill_losses[name] = nn.CosineSimilarity(dim=1)
            
            # Store metadata for forward pass
            d_cfg['loss_module_name'] = name
            
        self.distill_configs = distill_configs

    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """
        Args:
            batch_inputs: Tensor (B, C, H, W) - IR Images (Preprocessed)
            batch_data_samples: List[DetDataSample] - Contains 'img_rgb' in metainfo
        """
        losses = {}
        
        # -----------------------------------------------------------
        # 1. Forward Student (IR)
        # -----------------------------------------------------------
        # Extract features from student backbone
        # Note: self.student is a Detector. We access its backbone directly.
        student_backbone_feats = self.student.backbone(batch_inputs)
        
        # -----------------------------------------------------------
        # 2. Forward Teachers (RGB)
        # -----------------------------------------------------------
        # Ensure teachers stay in eval mode (model.train() propagates)
        for model in self.teachers.values():
            model.eval()

        # Prepare RGB inputs
        img_rgbs = [sample.metainfo['img_rgb'] for sample in batch_data_samples]
        rgb_inputs = self._prepare_teacher_inputs(img_rgbs, batch_inputs.device)

        teacher_features = {}
        with torch.no_grad():
            for name, model in self.teachers.items():
                # This assumes the teacher returns a tuple/list of features
                # matching the requested out_indices in its config
                feats = model(rgb_inputs)
                if isinstance(feats, tuple) or isinstance(feats, list):
                    teacher_features[name] = feats
                else:
                    # Handle cases where model returns single tensor or dict
                    teacher_features[name] = [feats]

        # -----------------------------------------------------------
        # 3. Compute Distillation Losses
        # -----------------------------------------------------------
        for d_cfg in self.distill_configs:
            name = d_cfg['loss_module_name']
            t_name = d_cfg['teacher_name']
            
            # Get features (Indices are 0-based in the list)
            s_idx = d_cfg['student_feature_index']
            t_idx = d_cfg['teacher_feature_index']
            
            s_feat = student_backbone_feats[s_idx]
            t_feat = teacher_features[t_name][t_idx]
            
            # Project Student Feature
            s_feat_proj = self.projectors[name](s_feat)
            
            # Resize if spatial dimensions mismatch
            if s_feat_proj.shape[-2:] != t_feat.shape[-2:]:
                s_feat_proj = F.interpolate(
                    s_feat_proj, 
                    size=t_feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Compute Loss
            loss_func = self.distill_losses[name]
            loss_val = loss_func(s_feat_proj, t_feat)
            
            # Handle Cosine Similarity (maximize similarity = minimize 1 - sim)
            if isinstance(loss_func, nn.CosineSimilarity):
                loss_val = (1 - loss_val).mean()
            
            losses[name] = loss_val * d_cfg['loss_weight']
        
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        """Forward prediction using the student model.

        When detection_feature_indices is set, selects a subset of backbone
        features before passing to the neck (e.g., 5 of 12 for DINO 5-scale).
        """
        if self.detection_feature_indices is None:
            return self.student.predict(
                batch_inputs, batch_data_samples, rescale)

        # Backbone outputs all features (e.g., 12 for distillation)
        all_feats = self.student.backbone(batch_inputs)
        # Select only the detection-relevant features for the neck
        det_feats = tuple(all_feats[i] for i in self.detection_feature_indices)

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

    def _prepare_teacher_inputs(self, img_rgbs, device, pad_size_divisor=224):
        """Stack, normalize, and pad RGB images for teachers.

        Args:
            img_rgbs: List of numpy arrays or tensors (H, W, 3) or (3, H, W).
                Images from metainfo are in BGR channel order (loaded by
                mmcv.imread). They are converted to RGB here before
                normalization, since TIMM teacher models expect RGB input.
            device: Target device.
            pad_size_divisor: Pad H and W to nearest multiple of this value.
                LCM(14, 16) = 112 ensures compatibility with all teachers:
                - DINOv2 (patch_size=14)
                - SAM (patch_size=16)
                - CLIP ViT-L/14 (patch_size=14)
        """
        processed = []
        # Standard ImageNet normalization (RGB order)
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1)

        for img in img_rgbs:
            if isinstance(img, torch.Tensor):
                t = img.float()
            else:
                # img is HWC BGR numpy array from mmcv.imread
                t = torch.from_numpy(img).permute(2, 0, 1).float()
            # Convert BGR → RGB (channel dim is 0 after permute)
            t = t[[2, 1, 0], ...]
            # Normalize per-image (before stacking, since sizes may differ)
            t = (t.to(device) - mean) / std
            processed.append(t)

        # Find max H, W across the batch (images may have different sizes
        # when mixing datasets, e.g. KAIST 512x640 vs FLIR 480x640)
        max_h = max(t.shape[1] for t in processed)
        max_w = max(t.shape[2] for t in processed)

        # Round up to nearest multiple of pad_size_divisor
        max_h = max_h + (pad_size_divisor - max_h % pad_size_divisor) % pad_size_divisor
        max_w = max_w + (pad_size_divisor - max_w % pad_size_divisor) % pad_size_divisor

        # Pad each image to (max_h, max_w) and stack
        padded = []
        for t in processed:
            pad_h = max_h - t.shape[1]
            pad_w = max_w - t.shape[2]
            if pad_h > 0 or pad_w > 0:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded.append(t)

        batch = torch.stack(padded)
        return batch
