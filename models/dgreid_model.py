"""
DG-ReID: Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification.
Complete unified model architecture implementing the shared-backbone, dual-evidence framework
from Section 5 and Figure 1 of the BTP report:
  - CLIP ViT-B/16 Visual Transformer Backbone (256x128 input, 16x8 patch grid).
  - Global Identity Branch with projection, BNNeck, and identity classifier.
  - Visual-side Domain-Adversarial Branch with GRL and DomainClassifier MLP (768 -> 384 -> M).
  - Visibility-Aware Part Branch with PartVisibilityGAT, reconstruction gate, visibility head, and local classifier.
  - Orthogonal Domain Factors (K=4, dim=512).
  - Multi-granular L2-normalized feature fusion for zero-shot cross-domain retrieval.
  - Full support for 3-stage training parameter freezing/unfreezing.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import clip
from .grl import GradientReversalLayer, compute_grl_alpha
from .domain_classifier import DomainClassifier
from .part_gat import PartVisibilityGAT
from .domain_factors import OrthogonalDomainFactors


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def build_clip_visual(backbone_name="ViT-B-16", h_resolution=16, w_resolution=8, vision_stride_size=16, pretrained=True):
    """
    Builds the CLIP visual transformer with bicubic interpolation of positional embeddings
    from pretrained grid (14x14) to target grid (h_resolution x w_resolution = 16x8).
    """
    if pretrained and backbone_name in clip._MODELS:
        url = clip._MODELS[backbone_name]
        try:
            model_path = clip._download(url)
            state_dict = torch.load(model_path, map_location="cpu")
        except Exception as e:
            print(f"[Warning] Could not download pretrained CLIP ({e}). Initializing randomly for offline/test mode.")
            state_dict = None
    else:
        state_dict = None

    if state_dict is not None:
        # Build model from state dict
        clip_model = clip.build_model(state_dict, h_resolution, w_resolution, vision_stride_size)
    else:
        # Construct synthetic state dict with proper shapes for offline/test mode
        embed_dim = 512
        vision_layers = 12
        vision_width = 768
        vision_patch_size = 16
        image_resolution = 224
        context_length = 77
        vocab_size = 49408
        transformer_width = 512
        transformer_heads = 8
        transformer_layers = 12

        from .clip.model import CLIP
        clip_model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            h_resolution, w_resolution
        )

    return clip_model.visual


class DGReID(nn.Module):
    def __init__(
        self,
        num_classes,
        num_domains=4,
        backbone_name="ViT-B-16",
        img_size=(256, 128),
        feat_dim=512,
        num_factors=4,
        pretrained=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feat_dim = feat_dim
        self.img_h, self.img_w = img_size

        # Grid resolution: 256 / 16 = 16, 128 / 16 = 8
        self.h_resolution = int((self.img_h - 16) // 16 + 1)
        self.w_resolution = int((self.img_w - 16) // 16 + 1)
        self.vision_stride = 16
        self.in_planes = 768  # ViT-B/16 width

        # 1. CLIP ViT-B/16 visual encoder
        self.image_encoder = build_clip_visual(
            backbone_name=backbone_name,
            h_resolution=self.h_resolution,
            w_resolution=self.w_resolution,
            vision_stride_size=self.vision_stride,
            pretrained=pretrained
        )

        # 2. Global Identity Branch (Section 5.4)
        # Linear projection W_g: 768 -> 512
        self.global_proj = nn.Linear(self.in_planes, self.feat_dim)
        # BNNeck
        self.global_bn = nn.BatchNorm1d(self.feat_dim)
        self.global_bn.bias.requires_grad_(False)
        # Global identity classifier
        self.global_classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)

        # 3. Domain-Adversarial Global Branch (Section 5.5)
        self.grl = GradientReversalLayer(alpha=0.0)
        self.domain_classifier = DomainClassifier(
            in_features=self.in_planes,
            hidden_dim=384,
            num_domains=self.num_domains
        )

        # 4. Visibility-Aware Part GAT (Sections 5.6 - 5.10)
        self.part_gat = PartVisibilityGAT(
            in_features=self.in_planes,
            proj_dim=self.feat_dim,
            num_parts=3,
            grid_hw=(self.h_resolution, self.w_resolution)
        )
        # Local identity classifier
        self.local_classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)

        # 5. Orthogonal Domain Factors (Section 5.11)
        self.domain_factors = OrthogonalDomainFactors(num_factors=num_factors, dim=self.feat_dim)

        self._init_heads()

    def _init_heads(self):
        self.global_proj.apply(weights_init_kaiming)
        self.global_bn.apply(weights_init_kaiming)
        self.global_classifier.apply(weights_init_classifier)
        self.local_classifier.apply(weights_init_classifier)

    def set_stage(self, stage=1):
        """
        Implements the 3-stage training schedule from Table 3 of the BTP report:
          Stage 1 (Epochs 1-10): Keep CLIP visual backbone completely frozen.
                                 Train projection heads, BNNecks, classifiers, PartVisibilityGAT,
                                 visibility head, and domain factors.
          Stage 2 (Epochs 11-30): Stage-1 modules + unfreeze final 2 ViT transformer blocks.
          Stage 3 (Epochs 31-60): Stage-2 modules + activate domain classifier with GRL warm-up.
        """
        if stage == 1:
            # Freeze entire image encoder
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Ensure heads, GAT, visibility, and domain factors are trainable
            for m in [self.global_proj, self.global_bn, self.global_classifier,
                      self.part_gat, self.local_classifier, self.domain_factors]:
                for param in m.parameters():
                    param.requires_grad = True
            # Domain classifier can be frozen or idle in Stage 1
            for param in self.domain_classifier.parameters():
                param.requires_grad = False
            self.grl.set_alpha(0.0)

        elif stage == 2:
            # Freeze patch projection and early transformer blocks
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Unfreeze final 2 transformer blocks
            if hasattr(self.image_encoder, 'transformer') and hasattr(self.image_encoder.transformer, 'resblocks'):
                for blk in self.image_encoder.transformer.resblocks[-2:]:
                    for param in blk.parameters():
                        param.requires_grad = True
            # Unfreeze ln_post if present
            if hasattr(self.image_encoder, 'ln_post'):
                for param in self.image_encoder.ln_post.parameters():
                    param.requires_grad = True

            # Train heads and GAT
            for m in [self.global_proj, self.global_bn, self.global_classifier,
                      self.part_gat, self.local_classifier, self.domain_factors]:
                for param in m.parameters():
                    param.requires_grad = True
            for param in self.domain_classifier.parameters():
                param.requires_grad = False
            self.grl.set_alpha(0.0)

        elif stage == 3:
            # Freeze patch projection and early transformer blocks (Section 5.3)
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Unfreeze final 2 transformer blocks + ln_post
            if hasattr(self.image_encoder, 'transformer') and hasattr(self.image_encoder.transformer, 'resblocks'):
                for blk in self.image_encoder.transformer.resblocks[-2:]:
                    for param in blk.parameters():
                        param.requires_grad = True
            if hasattr(self.image_encoder, 'ln_post'):
                for param in self.image_encoder.ln_post.parameters():
                    param.requires_grad = True

            for m in [self.global_proj, self.global_bn, self.global_classifier,
                      self.part_gat, self.local_classifier, self.domain_factors,
                      self.domain_classifier]:
                for param in m.parameters():
                    param.requires_grad = True

    def set_grl_alpha(self, alpha):
        """Sets the gradient reversal factor alpha(e)."""
        self.grl.set_alpha(alpha)

    def extract_tokens(self, x):
        """
        Passes image through CLIP ViT-B/16 visual encoder.
        Returns:
          f_cls: [B, 768] (class token)
          F_patch: [B, 128, 768] (patch tokens)
        """
        _, image_features, _ = self.image_encoder(x, None)
        f_cls = image_features[:, 0].contiguous()        # Class token
        F_patch = image_features[:, 1:].contiguous()     # 128 patch tokens
        return f_cls, F_patch

    def forward(
        self,
        x,
        label=None,
        grl_alpha=None,
        disable_grl=False,
        disable_part_branch=False,
        disable_vis_gating=False,
        eval_feat="combined"
    ):
        """
        Args:
            x: Input images [B, 3, 256, 128]
            label: Ground truth identity labels [B]
            grl_alpha: Current GRL factor (if None, uses self.grl.alpha)
            disable_grl: Ablation flag to turn off visual domain classifier
            disable_part_branch: Ablation flag to turn off PartVisibilityGAT
            disable_vis_gating: Ablation flag to turn off visibility weighting in GAT
            eval_feat: 'combined' (f = (g+l)/||g+l||), 'global' (g/||g||), or 'local' (l/||l||)
        """
        if grl_alpha is not None:
            self.set_grl_alpha(grl_alpha)

        B = x.shape[0]

        # 1. Visual Backbone forward pass
        f_cls, F_patch = self.extract_tokens(x)

        # 2. Global Identity Branch (Section 5.4)
        h_g = self.global_proj(f_cls)  # [B, 512]
        if B == 1:
            g = self.global_bn(h_g.repeat(2, 1))[:1]
        else:
            g = self.global_bn(h_g)    # [B, 512]

        # 3. Domain-Adversarial Global Branch (Section 5.5)
        if disable_grl:
            domain_logits = torch.zeros(B, self.num_domains, device=x.device)
        else:
            u = self.grl(f_cls)
            domain_logits = self.domain_classifier(u)

        # 4. Visibility-Aware Part Branch (Section 5.6 - 5.10)
        if disable_part_branch:
            local_feat = torch.zeros_like(g)
            local_feat_pre_bn = torch.zeros_like(h_g)
            vis_scores = torch.zeros(B, 3, 1, device=x.device)
            vis_logits = torch.zeros(B, 3, 1, device=x.device)
            attn_weights = torch.zeros(B, 3, 3, device=x.device)
        else:
            part_out = self.part_gat(F_patch, disable_vis_gating=disable_vis_gating)
            local_feat = part_out['local_feat']                 # [B, 512] (l)
            local_feat_pre_bn = part_out['local_feat_pre_bn']   # [B, 512] (h_l)
            vis_scores = part_out['vis_scores']                 # [B, 3, 1] (v_ij)
            vis_logits = part_out['vis_logits']                 # [B, 3, 1] (z_ij)
            attn_weights = part_out['attn_weights']             # [B, 3, 3]

        # 5. Training vs Inference Forward Pass
        if self.training:
            # Classification logits
            global_id_logits = self.global_classifier(g)
            local_id_logits = self.local_classifier(local_feat) if not disable_part_branch else global_id_logits
            ortho_loss = self.domain_factors.get_orthogonality_loss()

            return {
                'id_logits': global_id_logits,
                'global_feat': g,
                'global_feat_pre_bn': h_g,
                'local_logits': local_id_logits,
                'local_feat': local_feat,
                'local_feat_pre_bn': local_feat_pre_bn,
                'vis_scores': vis_scores,
                'vis_logits': vis_logits,
                'attn_weights': attn_weights,
                'domain_logits': domain_logits,
                'ortho_loss': ortho_loss,
                'f_cls': f_cls
            }
        else:
            # Zero-shot cross-domain retrieval inference (Section 5.12):
            # f = (g + l) / ||g + l||_2
            norm_g = F.normalize(g, p=2, dim=-1)
            norm_l = F.normalize(local_feat, p=2, dim=-1)

            if disable_part_branch or eval_feat == 'global':
                return norm_g
            elif eval_feat == 'local':
                return norm_l
            else:
                # Combined representation
                return F.normalize(norm_g + norm_l, p=2, dim=-1)

    def load_param(self, trained_path):
        """Loads weights from checkpoint file."""
        state_dict = torch.load(trained_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        # Clean potential module. prefix
        cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(cleaned, strict=False)
        print(f"Loaded weights from {trained_path}")
