"""
Label Smoothing Cross-Entropy Loss for Person Re-Identification.
Applies uniform smoothing across C classes:
q_ik = (1 - epsilon) * 1(k == y_i) + epsilon / C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    """
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits of shape (B, num_classes)
            targets: Ground-truth class indices of shape (B,)
        Returns:
            loss: Label-smoothed scalar cross-entropy loss
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth one-hot targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.epsilon)

        loss = torch.sum(-true_dist * log_probs, dim=-1).mean()
        return loss
