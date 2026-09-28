"""
CLIP ViT-B/16 visual backbone with 256x128 resolution support and bicubic
positional embedding interpolation.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    """
    QuickGELU activation function as used in OpenAI CLIP:
    QuickGELU(x) = x * sigmoid(1.702 * x)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Transformer Residual Attention Block with pre-LayerNorm.
    """
    def __init__(self, d_model: int = 768, n_head: int = 12, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention with residual
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        # Pre-LN MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPVisionTransformer(nn.Module):
    """
    CLIP ViT-B/16 visual transformer adapted for Person Re-ID (256x128 resolution).

    Input:
        x: Tensor of shape (B, 3, 256, 128)
    Output:
        f_cls: Class token feature of shape (B, 768)
        F_patch: Spatial patch tokens of shape (B, 128, 768), corresponding to 16x8 grid
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 128),
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout: float = 0.0,
        unfreeze_blocks: int = 2
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.unfreeze_blocks = unfreeze_blocks

        # Grid dimensions: 256 // 16 = 16, 128 // 16 = 8 -> 128 patches
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.total_tokens = self.num_patches + 1  # 1 CLS token + 128 patch tokens = 129

        # Conv2d patch projection
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        scale = embed_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.total_tokens, embed_dim))
        self.ln_pre = nn.LayerNorm(embed_dim)

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(d_model=embed_dim, n_head=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_post = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization
        nn.init.normal_(self.positional_embedding, std=0.02)
        nn.init.normal_(self.class_embedding, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.no_grad()
    def interpolate_pos_encoding(self, orig_pos_embed: torch.Tensor, orig_grid: Tuple[int, int] = (14, 14)):
        """
        Bicubic interpolation of CLIP positional embedding from original resolution (e.g. 14x14)
        to target resolution (16x8).
        
        Args:
            orig_pos_embed: Tensor of shape (1 + orig_h * orig_w, embed_dim)
            orig_grid: (orig_h, orig_w), default (14, 14) for 224x224 input
        """
        cls_pos = orig_pos_embed[0:1, :]  # (1, 768)
        patch_pos = orig_pos_embed[1:, :]  # (196, 768)

        orig_h, orig_w = orig_grid
        patch_pos = patch_pos.reshape(1, orig_h, orig_w, self.embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)  # (1, 768, 14, 14)

        # Bicubic interpolation to (16, 8)
        new_patch_pos = F.interpolate(
            patch_pos,
            size=(self.grid_h, self.grid_w),
            mode='bicubic',
            align_corners=False
        )  # (1, 768, 16, 8)

        new_patch_pos = new_patch_pos.permute(0, 2, 3, 1).reshape(self.num_patches, self.embed_dim)  # (128, 768)
        new_pos_embed = torch.cat([cls_pos, new_patch_pos], dim=0)  # (129, 768)

        self.positional_embedding.copy_(new_pos_embed)

    def load_clip_weights(self, clip_state_dict: dict):
        """
        Loads official CLIP visual weights and applies bicubic positional embedding interpolation.
        Supports both raw OpenAI state_dict and prefixed state_dict.
        """
        # Determine prefix if present
        prefix = ""
        if any(k.startswith("visual.") for k in clip_state_dict.keys()):
            prefix = "visual."

        # Load positional embedding with bicubic interpolation
        pos_key = f"{prefix}positional_embedding"
        if pos_key in clip_state_dict:
            pos_weight = clip_state_dict[pos_key]
            if pos_weight.shape[0] == 197:  # 1 + 14x14
                self.interpolate_pos_encoding(pos_weight, orig_grid=(14, 14))
            elif pos_weight.shape[0] == self.total_tokens:
                self.positional_embedding.copy_(pos_weight)

        # Load class embedding
        cls_key = f"{prefix}class_embedding"
        if cls_key in clip_state_dict:
            self.class_embedding.copy_(clip_state_dict[cls_key])

        # Load patch projection conv
        conv_key = f"{prefix}conv1.weight"
        if conv_key in clip_state_dict:
            self.conv1.weight.copy_(clip_state_dict[conv_key])

        # Load pre/post LN
        ln_pre_w = f"{prefix}ln_pre.weight"
        ln_pre_b = f"{prefix}ln_pre.bias"
        if ln_pre_w in clip_state_dict:
            self.ln_pre.weight.copy_(clip_state_dict[ln_pre_w])
            self.ln_pre.bias.copy_(clip_state_dict[ln_pre_b])

        ln_post_w = f"{prefix}ln_post.weight"
        ln_post_b = f"{prefix}ln_post.bias"
        if ln_post_w in clip_state_dict:
            self.ln_post.weight.copy_(clip_state_dict[ln_post_w])
            self.ln_post.bias.copy_(clip_state_dict[ln_post_b])

        # Load transformer blocks
        for i in range(self.depth):
            block_prefix = f"{prefix}transformer.resblocks.{i}."
            block = self.resblocks[i]
            
            # Map weights
            for name, param in block.named_parameters():
                # Handling OpenAI CLIP key naming differences if any
                clip_name = block_prefix + name
                # In OpenAI CLIP, multihead attention weights are in in_proj_weight / in_proj_bias
                if clip_name in clip_state_dict:
                    param.copy_(clip_state_dict[clip_name])

    def set_stage(self, stage: int):
        """
        Configure parameter trainability per project training schedule:
        Stage 1 (Epochs 1-10): All visual backbone parameters frozen.
        Stage 2 (Epochs 11-30): Final two transformer blocks unfrozen; early blocks frozen.
        Stage 3 (Epochs 31-60): Final two transformer blocks unfrozen; early blocks frozen.
        """
        if stage == 1:
            # Freeze entire backbone
            for param in self.parameters():
                param.requires_grad = False
        elif stage in (2, 3):
            # Freeze patch projection and initial blocks (0 to depth - unfreeze_blocks - 1)
            self.conv1.requires_grad_(False)
            self.class_embedding.requires_grad = False
            self.positional_embedding.requires_grad = False
            self.ln_pre.requires_grad_(False)
            
            frozen_depth = self.depth - self.unfreeze_blocks  # 12 - 2 = 10
            for i in range(frozen_depth):
                for p in self.resblocks[i].parameters():
                    p.requires_grad = False
                    
            # Unfreeze final transformer blocks
            for i in range(frozen_depth, self.depth):
                for p in self.resblocks[i].parameters():
                    p.requires_grad = True
                    
            self.ln_post.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, 256, 128)
        Returns:
            f_cls: Class token (B, 768)
            F_patch: Spatial patch tokens (B, 128, 768)
        """
        B = x.shape[0]

        # Patch projection: (B, 3, 256, 128) -> (B, 768, 16, 8) -> (B, 128, 768)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(B, self.embed_dim, -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] (B, 128, 768)

        # Prepend class token: (B, 1, 768)
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(B, 1, self.embed_dim, dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tokens, x], dim=1)  # shape = [*, grid ** 2 + 1, width] (B, 129, 768)

        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # Transformer resblocks
        for block in self.resblocks:
            x = block(x)

        x = self.ln_post(x)

        # Separate class token and patch tokens
        f_cls = x[:, 0, :]               # (B, 768)
        F_patch = x[:, 1:, :]            # (B, 128, 768)

        return f_cls, F_patch
