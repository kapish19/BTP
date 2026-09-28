r"""
Orthogonal Domain Factors Module.
Formulation from Section 5.11 of the BTP report:
  Learnable domain-factor vectors G = [g_1, ..., g_K] in R^(512 x K) (K = 4)
  Column-wise normalized: \hat{G} = [g_1 / ||g_1||_2, ..., g_K / ||g_K||_2]
  Orthogonality penalty: L_ortho = ||\hat{G}^T \hat{G} - I_K||_F^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalDomainFactors(nn.Module):
    def __init__(self, num_factors=4, dim=512):
        super().__init__()
        self.num_factors = num_factors
        self.dim = dim
        # Latent factor matrix G of shape [dim, num_factors]
        self.factors = nn.Parameter(torch.empty(dim, num_factors))
        nn.init.orthogonal_(self.factors)

    def get_orthogonality_loss(self):
        r"""
        Computes L_ortho = ||\hat{G}^T \hat{G} - I_K||_F^2
        """
        # Normalize columns (dim=0 has length `dim`, so norm across dim 0)
        g_norm = F.normalize(self.factors, p=2, dim=0) # [dim, K]
        gram = torch.matmul(g_norm.t(), g_norm)         # [K, K]
        identity = torch.eye(self.num_factors, device=self.factors.device, dtype=self.factors.dtype)
        diff = gram - identity
        # Squared Frobenius norm
        loss = torch.sum(diff ** 2)
        return loss

    def forward(self):
        return self.factors
