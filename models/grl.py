"""
Gradient Reversal Layer (GRL) for Visual-Side Domain Erasure.
Implements the standard Ganin & Lempitsky (ICML 2015) formulation:
  Forward:  R_lambda(x) = x
  Backward: d(R_lambda)/dx = -lambda * I
Also includes dynamic GRL schedule alpha(e) from the BTP report:
  alpha(e) = min(1.0, (e - 30) / 10) for e >= 31 (Stage 3).
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradients flowing back to the feature extractor are reversed
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReverseFunction.apply(x, alpha)


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.
    Holds dynamic alpha value that can be updated per epoch.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def forward(self, x):
        return grad_reverse(x, self.alpha)

    def extra_repr(self):
        return f"alpha={self.alpha:.4f}"


def compute_grl_alpha(epoch, stage3_start=31, warmup_epochs=10):
    """
    Exact GRL warmup schedule from Section 5.5 of the BTP report:
      alpha(e) = min(1.0, (e - 30) / 10), e >= 31
    For e < 31, alpha = 0.0 (adversarial signal inactive in Stage 1 & 2).
    """
    if epoch < stage3_start:
        return 0.0
    return min(1.0, float(epoch - (stage3_start - 1)) / float(warmup_epochs))
