"""
Orthogonal Domain Factors auxiliary regularization module.
Maintains K learnable domain factor vectors and calculates the orthogonality penalty:
L_ortho = || G_hat^T G_hat - I_K ||_F^2
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalDomainFactors(nn.Module):
    """
    K = 4 learnable latent domain factor vectors in R^{512 x K}.
    """
    def __init__(self, feat_dim: int = 512, num_factors: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_factors = num_factors

        # G in R^{feat_dim x num_factors}
        self.factors = nn.Parameter(torch.empty(feat_dim, num_factors))
        self._init_weights()

    def _init_weights(self):
        # Initialize with orthogonal vectors
        nn.init.orthogonal_(self.factors)

    def get_normalized_factors(self) -> torch.Tensor:
        """
        Column-wise L2 normalized factors: G_hat in R^{feat_dim x K}
        """
        return F.normalize(self.factors, p=2, dim=0)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        L_ortho = || G_hat^T G_hat - I_K ||_F^2
        """
        G_hat = self.get_normalized_factors()  # (512, K)
        gram = torch.matmul(G_hat.t(), G_hat)  # (K, K)
        eye = torch.eye(self.num_factors, device=self.factors.device, dtype=self.factors.dtype)
        diff = gram - eye
        loss_ortho = torch.norm(diff, p='fro') ** 2
        return loss_ortho

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        G_hat = self.get_normalized_factors()
        loss = self.compute_orthogonality_loss()
        return G_hat, loss
