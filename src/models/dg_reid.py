"""
Complete DG-ReID Architecture:
Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification.
Combines:
  - CLIP ViT-B/16 visual backbone (256x128)
  - Global Identity Branch (Linear + BNNeck)
  - Visual-side GRL + Domain Classifier (768 -> 384 -> M)
  - Part-Token Construction + Graph Attention Network (Head, Torso, Legs)
  - Visibility Predictor and Visibility Modulation
  - Local Identity Branch (Linear + BNNeck)
  - Orthogonal Domain Factors Regularization
  - Global-Local Embedding Fusion for Retrieval
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_vit import CLIPVisionTransformer
from .grl import GradientReversalLayer
from .domain_classifier import DomainClassifier
from .part_gat import PartVisibilityGAT
from .bnneck import BNNeck
from .orthogonal_factors import OrthogonalDomainFactors


class DGReID(nn.Module):
    """
    Unified DG-ReID model implementing the full dual-branch architecture.
    """
    def __init__(
        self,
        num_classes: int,
        num_domains: int = 3,
        feat_dim: int = 512,
        clip_dim: int = 768,
        num_factors: int = 4,
        depth: int = 12,
        unfreeze_blocks: int = 2,
        enable_grl: bool = True,
        enable_part_branch: bool = True,
        enable_gat: bool = True,
        enable_vis_gating: bool = True,
        enable_ortho: bool = True,
        enable_reconstruction_gate: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feat_dim = feat_dim
        self.clip_dim = clip_dim
        self.depth = depth
        self.enable_grl = enable_grl
        self.enable_part_branch = enable_part_branch
        self.enable_gat = enable_gat
        self.enable_vis_gating = enable_vis_gating
        self.enable_ortho = enable_ortho

        # 1. CLIP ViT-B/16 Visual Backbone (256x128 resolution)
        self.backbone = CLIPVisionTransformer(
            img_size=(256, 128),
            patch_size=16,
            embed_dim=clip_dim,
            depth=depth,
            num_heads=12,
            unfreeze_blocks=unfreeze_blocks
        )

        # 2. Global Branch
        # h_g = W_g * f_cls + b_g
        self.proj_global = nn.Linear(clip_dim, feat_dim)
        nn.init.kaiming_normal_(self.proj_global.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.proj_global.bias, 0.0)
        self.global_bnneck = BNNeck(in_features=feat_dim, num_classes=num_classes)

        # 3. Domain-Adversarial Global Branch
        self.grl = GradientReversalLayer(alpha=0.0)
        self.domain_classifier = DomainClassifier(
            in_features=clip_dim,
            hidden_dim=384,
            num_domains=num_domains
        )

        # 4. Visibility-Aware Part Branch
        if self.enable_part_branch:
            self.part_gat = PartVisibilityGAT(
                embed_dim=clip_dim,
                attn_dim=clip_dim,
                vis_hidden_dim=256,
                out_dim=feat_dim,
                grid_size=(16, 8),
                part_splits=(5, 6, 5),
                use_gat=enable_gat,
                use_vis_gating=enable_vis_gating,
                use_reconstruction_gate=enable_reconstruction_gate
            )
            self.local_bnneck = BNNeck(in_features=feat_dim, num_classes=num_classes)
        else:
            self.part_gat = None
            self.local_bnneck = None

        # 5. Orthogonal Domain Factors
        if self.enable_ortho:
            self.domain_factors = OrthogonalDomainFactors(feat_dim=feat_dim, num_factors=num_factors)
        else:
            self.domain_factors = None

    def set_stage(self, stage: int, epoch: int):
        """
        Configure model parameters and GRL schedule according to training stage:
        - Stage 1 (Epochs 1-10): Backbone frozen, GRL alpha = 0.
        - Stage 2 (Epochs 11-30): Final 2 transformer blocks unfrozen, GRL alpha = 0.
        - Stage 3 (Epochs 31-60): Final 2 blocks unfrozen, GRL alpha warm-up min(1, (e-30)/10).
        """
        self.backbone.set_stage(stage)

        if stage in (1, 2) or not self.enable_grl:
            self.grl.alpha = 0.0
            for p in self.domain_classifier.parameters():
                p.requires_grad = (stage == 3 and self.enable_grl)
        elif stage == 3 and self.enable_grl:
            self.grl.set_alpha(epoch=epoch, start_epoch=31, ramp_epochs=10)
            for p in self.domain_classifier.parameters():
                p.requires_grad = True

    def forward_train(
        self,
        x: torch.Tensor,
        stage: int = 1
    ) -> Dict[str, Union[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass during training.
        Args:
            x: Tensor of shape (B, 3, 256, 128)
            stage: Current training stage (1, 2, or 3)
        Returns:
            Dictionary containing:
                - h_g: Global metric feature before BN (B, 512)
                - g: Global normalized feature after BN (B, 512)
                - logits_g: Global identity logits (B, num_classes)
                - h_l: Local metric feature before BN (B, 512) or None
                - l: Local normalized feature after BN (B, 512) or None
                - logits_l: Local identity logits (B, num_classes) or None
                - v_pred: Predicted part visibility scores (B, 3) or None
                - domain_logits: Source domain prediction logits (B, M) or None
                - loss_ortho: Domain factor orthogonality penalty (scalar) or None
        """
        # 1. Backbone forward
        f_cls, F_patch = self.backbone(x)  # (B, 768), (B, 128, 768)

        # 2. Global Branch
        h_g = self.proj_global(f_cls)  # (B, 512)
        g, logits_g = self.global_bnneck(h_g)  # (B, 512), (B, num_classes)

        # 3. Domain-Adversarial Branch (Stage 3)
        domain_logits = None
        if self.enable_grl and stage == 3:
            u = self.grl(f_cls)
            domain_logits = self.domain_classifier(u)

        # 4. Visibility-Aware Part Branch
        h_l, l, logits_l, v_pred = None, None, None, None
        if self.enable_part_branch:
            h_l, v_pred, _ = self.part_gat(F_patch)  # (B, 512), (B, 3)
            l, logits_l = self.local_bnneck(h_l)  # (B, 512), (B, num_classes)

        # 5. Orthogonal Domain Factors
        loss_ortho = None
        if self.enable_ortho:
            _, loss_ortho = self.domain_factors()

        return {
            'h_g': h_g,
            'g': g,
            'logits_g': logits_g,
            'h_l': h_l,
            'l': l,
            'logits_l': logits_l,
            'v_pred': v_pred,
            'domain_logits': domain_logits,
            'loss_ortho': loss_ortho,
            'f_cls': f_cls
        }

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during testing / retrieval.
        Equation 5.12:
            f = (g + l) / || g + l ||_2
        """
        f_cls, F_patch = self.backbone(x)
        h_g = self.proj_global(f_cls)
        g = self.global_bnneck.bn(h_g)

        if self.enable_part_branch:
            h_l, _, _ = self.part_gat(F_patch)
            l = self.local_bnneck.bn(h_l)
            f = g + l
        else:
            f = g

        # L2 normalize final retrieval embedding
        return F.normalize(f, p=2, dim=1)

    def forward(self, x: torch.Tensor, stage: int = 1) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.training:
            return self.forward_train(x, stage=stage)
        else:
            return self.forward_inference(x)
