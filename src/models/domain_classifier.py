"""
Source-domain classifier MLP for visual domain erasure.
Architecture: 768 -> 384 -> M (BatchNorm1d + ReLU).
"""

import torch
import torch.nn as nn


class DomainClassifier(nn.Module):
    """
    Domain Classifier MLP operating on the visual class token.
    Maps: 768 -> Linear -> 384 -> BatchNorm1d -> ReLU -> Linear -> num_domains (M)
    """
    def __init__(self, in_features: int = 768, hidden_dim: int = 384, num_domains: int = 3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_domains)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Class token feature (typically after GRL) of shape (B, 768)
        Returns:
            domain_logits: Logits over source domains of shape (B, num_domains)
        """
        return self.net(u)
