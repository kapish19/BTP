"""
Batch-Hard Triplet Loss for metric learning in Person Re-Identification.
Mines the hardest positive and hardest negative for each sample in the mini-batch.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes pairwise Euclidean distance matrix between x and y:
    dist[i, j] = ||x_i - y_j||_2
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2.0 * torch.matmul(x, y.t())
    dist = dist.clamp(min=eps).sqrt()
    return dist


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Loss:
    L_tri = (1 / B) * sum_{i=1}^B [m + max_{p} d(x_i, x_p) - min_{n} d(x_i, x_n)]_+
    """
    def __init__(self, margin: float = 0.3, normalize_feature: bool = False):
        super().__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Embeddings of shape (B, D)
            targets: Identity labels of shape (B,)
        Returns:
            loss: Batch-hard triplet loss scalar
            dist_ap: Mean anchor-positive distance
            dist_an: Mean anchor-negative distance
        """
        if self.normalize_feature:
            features = F.normalize(features, p=2, dim=-1)

        # Pairwise distance matrix: (B, B)
        dist_mat = euclidean_dist(features, features)
        N = dist_mat.size(0)

        # Masks for positive and negative pairs
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        # For each anchor, find hardest positive (largest distance among positives)
        dist_ap = []
        for i in range(N):
            pos_dists = dist_mat[i][is_pos[i]]
            if pos_dists.numel() > 1:
                # Exclude self-distance (which is 0)
                pos_dists_no_self = pos_dists[pos_dists > 1e-6]
                if pos_dists_no_self.numel() > 0:
                    dist_ap.append(pos_dists_no_self.max().unsqueeze(0))
                else:
                    dist_ap.append(pos_dists.max().unsqueeze(0))
            else:
                dist_ap.append(pos_dists.max().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)

        # For each anchor, find hardest negative (smallest distance among negatives)
        dist_an = []
        for i in range(N):
            neg_dists = dist_mat[i][is_neg[i]]
            if neg_dists.numel() > 0:
                dist_an.append(neg_dists.min().unsqueeze(0))
            else:
                dist_an.append(torch.tensor(0.0, device=features.device, dtype=features.dtype).unsqueeze(0))
        dist_an = torch.cat(dist_an)

        # Compute hinge loss: max(0, dist_ap - dist_an + margin)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss, dist_ap.mean(), dist_an.mean()
