"""
Batch Normalization Neck (BNNeck) following the Strong Baseline for Person Re-ID (Luo et al., 2020).
Decouples metric learning (triplet loss on pre-BN feature) from classification (CE on post-BN feature).
"""

from typing import Tuple
import torch
import torch.nn as nn


class BNNeck(nn.Module):
    """
    BNNeck Module:
    Maps pre-BN embedding h -> BatchNorm1d -> g -> Linear classifier -> logits.
    """
    def __init__(self, in_features: int = 512, num_classes: int = 1000):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.bn = nn.BatchNorm1d(in_features)
        self.bn.bias.requires_grad_(False)  # Luo et al. recommendation

        self.classifier = nn.Linear(in_features, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Feature before BN of shape (B, 512)
        Returns:
            g: Feature after BN of shape (B, 512)
            logits: Classification logits of shape (B, num_classes)
        """
        g = self.bn(h)
        logits = self.classifier(g)
        return g, logits
