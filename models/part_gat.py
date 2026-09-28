r"""
Visibility-Aware Part Graph Attention Network (PartVisibilityGAT).
Implements the local body-part pathway described in Sections 5.6-5.10 of the BTP report:
  1. Patch-token part construction from 16x8 spatial grid into 3 vertical regions:
       R_1 (Head/Shoulder): rows 0..4
       R_2 (Torso):         rows 5..10
       R_3 (Legs):          rows 11..15
  2. Graph Attention across body parts:
       e_jk = (W_Q p_j)^T (W_K p_k) / sqrt(d_a)
       a_jk = softmax_k(e_jk)
       p'_j = sum_k a_jk W_V p_k
  3. Internal Reconstruction Gate (Section 5.10):
       gamma_j = sigmoid(w_r^T p_j + b_r)
       \tilde{p}_j = gamma_j p'_j + (1 - gamma_j) p_j
  4. Visibility Prediction (Section 5.8):
       v_ij = sigmoid(w_2^T ReLU(W_1 p'_{ij} + b_1) + b_2)
  5. Visibility-Aware Weighting & Aggregation (Section 5.9):
       \hat{p}_j = v_j * \tilde{p}_j
       Local feature = sum_{j=1}^3 \hat{p}_j
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartVisibilityGAT(nn.Module):
    def __init__(self, in_features=768, proj_dim=512, num_parts=3, grid_hw=(16, 8)):
        super().__init__()
        self.in_features = in_features
        self.proj_dim = proj_dim
        self.num_parts = num_parts
        self.grid_h, self.grid_w = grid_hw

        # Graph Attention projections (d_a = in_features)
        self.w_q = nn.Linear(in_features, in_features, bias=False)
        self.w_k = nn.Linear(in_features, in_features, bias=False)
        self.w_v = nn.Linear(in_features, in_features, bias=False)
        self.scale = 1.0 / math.sqrt(in_features)

        # Internal Reconstruction Gate (Section 5.10): gamma_j = sigmoid(w_r^T p_j + b_r)
        self.recon_gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )

        # Visibility Predictor (Section 5.8): 2-layer MLP with ReLU and Sigmoid
        # Produces both raw logits (for stable BCEWithLogitsLoss) and probabilities v_ij
        hidden_vis = in_features // 2
        self.vis_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_vis),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_vis, 1)
        )

        # Local projection W_l: maps aggregated part features to Re-ID feature dimension (512)
        self.local_proj = nn.Linear(in_features, proj_dim)
        # BNNeck for local branch
        self.local_bn = nn.BatchNorm1d(proj_dim)
        self.local_bn.bias.requires_grad_(False)

        self._init_weights()

    def _init_weights(self):
        for m in [self.w_q, self.w_k, self.w_v, self.local_proj]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        for m in self.vis_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for m in self.recon_gate:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.constant_(self.local_bn.weight, 1.0)
        nn.init.constant_(self.local_bn.bias, 0.0)

    def pool_coarse_parts(self, patch_tokens):
        """
        Pools patch tokens [B, 128, 768] into 3 coarse vertical regions:
          Upper (rows 0..4):  Head / Shoulder
          Middle (rows 5..10): Torso
          Lower (rows 11..15): Legs
        """
        B, N, D = patch_tokens.shape
        H, W = self.grid_h, self.grid_w
        if N != H * W:
            # Dynamic fallback if grid dimensions differ
            H = round(math.sqrt(N * 2))
            W = N // H if H > 0 else 1
            if H * W != N:
                # Sequential chunking fallback
                part_size = N // self.num_parts
                parts = []
                for i in range(self.num_parts):
                    s = i * part_size
                    e = s + part_size if i < self.num_parts - 1 else N
                    parts.append(patch_tokens[:, s:e, :].mean(dim=1))
                return torch.stack(parts, dim=1)

        grid = patch_tokens.reshape(B, H, W, D)

        # 3 coarse vertical regions on 16-row grid:
        # Rows 0..4 (5 rows, 40 patches)
        # Rows 5..10 (6 rows, 48 patches)
        # Rows 11..15 (5 rows, 40 patches)
        r1_end = int(round(H * 5.0 / 16.0))
        r2_end = int(round(H * 11.0 / 16.0))

        p1 = grid[:, 0:r1_end, :, :].reshape(B, -1, D).mean(dim=1)
        p2 = grid[:, r1_end:r2_end, :, :].reshape(B, -1, D).mean(dim=1)
        p3 = grid[:, r2_end:H, :, :].reshape(B, -1, D).mean(dim=1)

        return torch.stack([p1, p2, p3], dim=1) # [B, 3, D]

    def forward(self, patch_tokens, disable_vis_gating=False):
        """
        Args:
            patch_tokens: Tensor [B, 128, 768]
            disable_vis_gating: If True (ablation mode), skip visibility weighting (v_j = 1)
        Returns:
            dict containing:
              'local_feat': [B, proj_dim] (after BNNeck)
              'local_feat_pre_bn': [B, proj_dim] (before BNNeck)
              'vis_scores': [B, 3, 1] (probabilities v_ij in (0, 1))
              'vis_logits': [B, 3, 1] (raw logits z_ij for BCEWithLogitsLoss)
              'attn_weights': [B, 3, 3] (attention coefficients a_jk)
              'part_nodes': [B, 3, D] (pooled part vectors p_j)
        """
        B = patch_tokens.shape[0]

        # 1. Coarse part pooling: [B, 3, D]
        part_nodes = self.pool_coarse_parts(patch_tokens)

        # 2. Graph Attention Across Body Parts (Section 5.7)
        Q = self.w_q(part_nodes) # [B, 3, D]
        K = self.w_k(part_nodes) # [B, 3, D]
        V = self.w_v(part_nodes) # [B, 3, D]

        # Attention coefficients: e_jk = (W_Q p_j)^T (W_K p_k) / sqrt(d_a)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale # [B, 3, 3]
        attn_weights = F.softmax(scores, dim=-1)                   # [B, 3, 3]
        context_parts = torch.matmul(attn_weights, V)               # [B, 3, D] (p'_j)

        # 3. Internal Reconstruction Gate (Section 5.10):
        #    \tilde{p}_j = gamma_j * p'_j + (1 - gamma_j) * p_j
        gate = self.recon_gate(part_nodes)                          # [B, 3, D]
        recon_parts = gate * context_parts + (1.0 - gate) * part_nodes

        # 4. Visibility Prediction (Section 5.8):
        #    z_ij = w_2^T ReLU(W_1 p'_{ij} + b_1) + b_2
        vis_logits = self.vis_mlp(context_parts)                    # [B, 3, 1]
        vis_scores = torch.sigmoid(vis_logits)                      # [B, 3, 1]

        # 5. Visibility-Aware Weighting (Section 5.9):
        #    \hat{p}_j = v_j * \tilde{p}_j
        if disable_vis_gating:
            weighted_parts = recon_parts
        else:
            weighted_parts = recon_parts * vis_scores

        # Aggregation: sum_{j=1}^3 \hat{p}_j
        part_sum = weighted_parts.sum(dim=1)                        # [B, D]

        # Projection & BNNeck
        h_l = self.local_proj(part_sum)                             # [B, proj_dim]

        if h_l.shape[0] == 1:
            h_rep = h_l.repeat(2, 1)
            l = self.local_bn(h_rep)[:1]
        else:
            l = self.local_bn(h_l)

        return {
            'local_feat': l,
            'local_feat_pre_bn': h_l,
            'vis_scores': vis_scores,
            'vis_logits': vis_logits,
            'attn_weights': attn_weights,
            'part_nodes': part_nodes
        }
