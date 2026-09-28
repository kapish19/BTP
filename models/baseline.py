"""
Baseline CLIP Model for Person Re-Identification.
Standard visual baseline (No GRL domain erasure, No PartVisibilityGAT branch).
Corresponds to 'CILP-FGDI Baseline' in Table 7 of the BTP report:
  - CLIP ViT-B/16 visual encoder.
  - Global projection 768 -> 512 + BNNeck.
  - Identity classifier with label-smoothing and batch-hard triplet loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgreid_model import DGReID


class BaselineCLIP(nn.Module):
    def __init__(self, num_classes, backbone_name="ViT-B-16", img_size=(256, 128), feat_dim=512, pretrained=True):
        super().__init__()
        # Wrapped around DGReID with GRL and Part branch disabled
        self.model = DGReID(
            num_classes=num_classes,
            num_domains=1,
            backbone_name=backbone_name,
            img_size=img_size,
            feat_dim=feat_dim,
            pretrained=pretrained
        )

    def forward(self, x, label=None):
        return self.model(
            x,
            label=label,
            disable_grl=True,
            disable_part_branch=True,
            eval_feat="global"
        )
