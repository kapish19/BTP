"""
Domain Classifier MLP for Visual-Side Domain Adversarial Learning.
Architecture from Section 5.5 of the BTP report:
  Linear(768 -> 384) -> BatchNorm1d(384) -> ReLU() -> Linear(384 -> M)
where M is the number of source training domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    def __init__(self, in_features=768, hidden_dim=384, num_domains=4):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_domains)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, std=0.01)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, in_features], usually passed after GRL.
        Returns:
            logits: Tensor of shape [B, num_domains].
        """
        # Handle batch size = 1 edge case (BatchNorm requires B > 1)
        if x.shape[0] == 1:
            h = self.fc1(x)
            # Repeat to allow BN running stats, take first
            h_rep = h.repeat(2, 1)
            h_bn = self.bn1(h_rep)[:1]
            h_act = self.relu(h_bn)
            return self.fc2(h_act)

        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        logits = self.fc2(h)
        return logits
