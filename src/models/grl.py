"""
Gradient Reversal Layer (GRL) for domain-adversarial learning.
Implements forward identity mapping and reversed backpropagation gradients.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Function:
    Forward:  R_lambda(x) = x
    Backward: dR_lambda / dx = -lambda * I
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse gradient sign and scale by alpha
        reversed_grad = -ctx.alpha * grad_output
        return reversed_grad, None


class GradientReversalLayer(nn.Module):
    """
    Module wrapper for Gradient Reversal Layer with scheduled reversal factor alpha(e).
    
    Schedule:
        alpha(e) = min(1.0, (e - 30) / 10) for e >= 31
        alpha(e) = 0.0 for e < 31
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, epoch: int, start_epoch: int = 31, ramp_epochs: int = 10):
        """
        Update the adversarial gradient multiplier according to current training epoch.
        """
        if epoch < start_epoch:
            self.alpha = 0.0
        else:
            self.alpha = float(min(1.0, (epoch - (start_epoch - 1)) / float(ramp_epochs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0 or not self.training:
            return x
        return GradientReversalFunction.apply(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.4f}"
