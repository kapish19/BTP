"""
Visibility loss (L_occ) for synthetic-mask-supervised part visibility learning.
Computes soft binary cross-entropy between predicted visibility v_ij and target visibility m_ij:
L_occ = - (1 / 3B) * sum_{i=1}^B sum_{j=1}^3 [m_ij * log(v_ij) + (1 - m_ij) * log(1 - v_ij)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibilityLoss(nn.Module):
    """
    Supervises part visibility predictions using known synthetic occlusion masks.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, v_pred: torch.Tensor, m_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_pred: Predicted visibility in (0, 1) of shape (B, 3)
            m_target: Target visibility ratio in [0, 1] of shape (B, 3)
        Returns:
            loss: Mean visibility cross-entropy loss scalar
        """
        v_pred = torch.clamp(v_pred, min=self.eps, max=1.0 - self.eps)
        m_target = torch.clamp(m_target, min=0.0, max=1.0)

        # BCE with soft targets
        bce = -(m_target * torch.log(v_pred) + (1.0 - m_target) * torch.log(1.0 - v_pred))
        return bce.mean()
