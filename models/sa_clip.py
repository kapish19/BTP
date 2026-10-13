"""
SA-CLIP: Style-Agnostic Feature-Invariant Vision Transformer for Domain Generalization Person Re-ID.
Combines:
  1. OpenAI CLIP ViT-B/16 visual backbone with parameter-efficient fine-tuning (last 2 blocks trainable).
  2. PatchMixStyle: In-transformer feature statistics perturbation across batch instances.
  3. BNNeck for decoupled metric learning (Triplet Loss) and classification (CrossEntropy).
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchMixStyle(nn.Module):
    """MixStyle applied to spatial patch tokens in Vision Transformers.
    
    Synthesizes unseen virtual camera styles during training by mixing feature 
    statistics (channel mean and standard deviation) across batch instances.
    At evaluation time, this is an identity pass-through.
    
    Reference:
        Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: [L, B, D] sequence tensor (CLIP transformer layout), where:
               x[0] is the CLS token, and x[1:] are spatial patch tokens.
        """
        if not self.training or random.random() > self.p:
            return x

        # Transpose from [L, B, D] -> [B, L, D]
        x_bld = x.permute(1, 0, 2)
        B, L, D = x_bld.shape

        cls_token = x_bld[:, :1, :]
        patches = x_bld[:, 1:, :]  # [B, L-1, D]

        # Calculate instance mean and variance across spatial patches
        mu = patches.mean(dim=1, keepdim=True)  # [B, 1, D]
        var = patches.var(dim=1, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()

        # Normalized style-agnostic spatial structure
        patches_normed = (patches - mu) / sig

        # Sample mixing weights from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)

        # Random permutation for cross-instance style pairing
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]

        # Synthesize mixed camera style statistics
        mu_mixed = lam * mu + (1.0 - lam) * mu2
        sig_mixed = lam * sig + (1.0 - lam) * sig2

        # Re-stylize patch tokens
        patches_stylized = patches_normed * sig_mixed + mu_mixed

        # Recombine CLS token with stylized patches
        out = torch.cat([cls_token, patches_stylized], dim=1)
        return out.permute(1, 0, 2)  # Return [L, B, D]


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
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


class SACLIP(nn.Module):
    """Style-Agnostic CLIP Model for Domain Generalization Person Re-ID."""

    def __init__(self, num_classes, backbone='ViT-B/16', img_size=(256, 128),
                 use_mixstyle=True, mixstyle_p=0.5, mixstyle_alpha=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.use_mixstyle = use_mixstyle
        self.feat_dim = 512

        # Load CLIP model
        from .clip import clip
        clip_model, _ = clip.load(backbone, device='cpu')
        visual = clip_model.visual

        self.conv1 = visual.conv1
        self.class_embedding = visual.class_embedding
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer
        self.ln_post = visual.ln_post
        self.proj = visual.proj  # [768, 512]

        # Interpolate positional embeddings to target Re-ID resolution (256, 128)
        # 256 // 16 = 16 rows, 128 // 16 = 8 cols -> 128 patches
        patch_size = 16
        h_patches = img_size[0] // patch_size  # 16
        w_patches = img_size[1] // patch_size  # 8
        num_target_patches = h_patches * w_patches  # 128

        orig_pos = visual.positional_embedding  # [197, 768] for 224x224 (14x14 grid)
        cls_pos = orig_pos[:1, :]
        patch_pos = orig_pos[1:, :]  # [196, 768]
        dim = orig_pos.shape[-1]

        orig_grid = int(math.sqrt(patch_pos.shape[0]))  # 14
        patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, dim).permute(0, 3, 1, 2)
        # Bicubic interpolation from 14x14 -> 16x8
        new_patch_pos = F.interpolate(patch_pos, size=(h_patches, w_patches), mode='bicubic', align_corners=False)
        new_patch_pos = new_patch_pos.permute(0, 2, 3, 1).reshape(num_target_patches, dim)
        self.positional_embedding = nn.Parameter(torch.cat([cls_pos, new_patch_pos], dim=0))

        # Freeze early transformer blocks (blocks 0-9), train only blocks 10-11
        for param in self.conv1.parameters():
            param.requires_grad = False
        self.class_embedding.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.ln_pre.requires_grad = False

        for i in range(10):
            for param in self.transformer.resblocks[i].parameters():
                param.requires_grad = False

        for i in [10, 11]:
            for param in self.transformer.resblocks[i].parameters():
                param.requires_grad = True

        for param in self.ln_post.parameters():
            param.requires_grad = True
        self.proj.requires_grad = True

        # In-Transformer MixStyle module
        self.mixstyle = PatchMixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else nn.Identity()

        # BNNeck Layer (Luo et al., CVPR 2019 BoT)
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # Classification Head
        self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # 1. Patch projection
        x = self.conv1(x)  # [B, 768, 16, 8]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, 128, 768]

        # 2. Add CLS token
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 129, 768]

        # 3. Add positional embedding & pre-norm
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # 4. Transpose to [L, B, D] for CLIP Transformer
        x = x.permute(1, 0, 2)  # [129, B, 768]

        # 5. Pass through frozen blocks (0-9)
        x = self.transformer.resblocks[:10](x)

        # 6. Apply In-Transformer MixStyle
        x = self.mixstyle(x)

        # 7. Pass through trainable fine-tuning blocks (10-11)
        x = self.transformer.resblocks[10:](x)

        # 8. Post-norm and extract CLS representation
        x = x.permute(1, 0, 2)  # [B, 129, 768]
        cls_token = self.ln_post(x[:, 0, :])  # [B, 768]

        # 9. Projection to embedding space
        global_feat = cls_token @ self.proj  # [B, 512]

        # 10. BNNeck
        bn_feat = self.bottleneck(global_feat)  # [B, 512]

        if self.training:
            cls_score = self.classifier(bn_feat)  # [B, num_classes]
            # global_feat used for TripletLoss, cls_score for CrossEntropy
            return cls_score, global_feat
        else:
            # Inference: L2-normalized feature representation
            return F.normalize(bn_feat, p=2, dim=-1)


def build_sa_clip(num_classes, args=None, use_mixstyle=True):
    """Builder helper for SA-CLIP model."""
    img_size = (256, 128)
    backbone = getattr(args, 'backbone', 'ViT-B/16') if args else 'ViT-B/16'
    mixstyle_p = getattr(args, 'mixstyle_p', 0.5) if args else 0.5
    mixstyle_alpha = getattr(args, 'mixstyle_alpha', 0.1) if args else 0.1

    model = SACLIP(
        num_classes=num_classes,
        backbone=backbone,
        img_size=img_size,
        use_mixstyle=use_mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha
    )
    return model
