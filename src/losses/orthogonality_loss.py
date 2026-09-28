"""
Orthogonality penalty loss for latent domain factor diversity:
L_ortho = || G_hat^T G_hat - I_K ||_F^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalityLoss(nn.Module):
    """
    Computes Frobenius norm squared of (G_hat^T G_hat - I_K).
    """
    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            factors: Learnable domain factors of shape (feat_dim, K)
        Returns:
            loss: Orthogonality penalty scalar
        """
        K = factors.size(1)
        G_hat = F.normalize(factors, p=2, dim=0)
        gram = torch.matmul(G_hat.t(), G_hat)
        eye = torch.eye(K, device=factors.device, dtype=factors.dtype)
        diff = gram - eye
        return torch.norm(diff, p='fro') ** 2
