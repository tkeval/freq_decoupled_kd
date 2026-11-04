import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .utils import weighted_loss

@weighted_loss
def bce_with_logits_distillation_loss(student_logits, teacher_logits, T):
    """BCEWithLogitsLoss for single-class knowledge distillation."""
    # Squeeze the last dimension which is 1 for single-class problems
    student_logits = student_logits.squeeze(-1)
    teacher_logits = teacher_logits.squeeze(-1)

    # Apply temperature scaling
    student_logits_t = student_logits / T
    teacher_logits_t = teacher_logits / T

    # The teacher's logits act as soft labels, so we apply sigmoid to get probabilities
    teacher_probs_t = torch.sigmoid(teacher_logits_t)
    
    # Calculate BCE with logits loss
    loss = F.binary_cross_entropy_with_logits(
        student_logits_t,
        teacher_probs_t,
        reduction='mean'
    )
    return loss

@MODELS.register_module()
class BCEWithLogitsDistillationLoss(nn.Module):
    """Wrapper for bce_with_logits_distillation_loss.

    Args:
        T (float): Temperature for knowledge distillation.
        reduction (str): The method used to reduce the loss.
        loss_weight (float): The weight of the loss.
    """

    def __init__(self, T=2.0, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.T = T
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                student_logits,
                teacher_logits,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss_cls = self.loss_weight * bce_with_logits_distillation_loss(
            student_logits,
            teacher_logits,
            T=self.T
        )
        return loss_cls



