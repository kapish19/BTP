"""
Visibility-Aware Part Graph Attention Network (PartVisibilityGAT).
Performs vertical coarse part pooling (head, torso, legs), graph attention exchange,
learned internal reconstruction gating, and visibility-prediction gating.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartVisibilityGAT(nn.Module):
    """
    Part-based feature extraction with Graph Attention Network and Visibility Gating.

    Part Construction:
        Grid size: 16x8 (128 patches)
        R1 (Head/Shoulder): Rows 0 to 4 (40 patches)
        R2 (Torso):         Rows 5 to 10 (48 patches)
        R3 (Legs):          Rows 11 to 15 (40 patches)

    Graph Attention:
        3 nodes interacting via multi-head/single-head scaled dot-product attention.
    
    Reconstruction Gate:
        p_tilde_j = gamma_j * p'_j + (1 - gamma_j) * p_j,  gamma_j = sigmoid(w_r^T p_j + b_r)

    Visibility Predictor:
        v_j = sigmoid(w_2^T * relu(W_1 * p'_j + b1) + b2)
    """
    def __init__(
        self,
        embed_dim: int = 768,
        attn_dim: int = 768,
        vis_hidden_dim: int = 256,
        out_dim: int = 512,
        grid_size: Tuple[int, int] = (16, 8),
        part_splits: Tuple[int, int, int] = (5, 6, 5),  # rows for upper, middle, lower
        use_gat: bool = True,
        use_vis_gating: bool = True,
        use_reconstruction_gate: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.grid_h, self.grid_w = grid_size
        self.part_splits = part_splits
        self.use_gat = use_gat
        self.use_vis_gating = use_vis_gating
        self.use_reconstruction_gate = use_reconstruction_gate

        # Region row slices
        r1_end = part_splits[0]
        r2_end = r1_end + part_splits[1]
        self.region_slices = [
            (0, r1_end),
            (r1_end, r2_end),
            (r2_end, self.grid_h)
        ]

        # Graph Attention projections: W_Q, W_K, W_V
        self.wq = nn.Linear(embed_dim, attn_dim, bias=False)
        self.wk = nn.Linear(embed_dim, attn_dim, bias=False)
        self.wv = nn.Linear(embed_dim, attn_dim, bias=False)
        self.scale = 1.0 / (attn_dim ** 0.5)

        # Internal reconstruction gate: gamma_j = sigmoid(w_r^T p_j + b_r)
        self.gate_linear = nn.Linear(embed_dim, 1)

        # Visibility predictor: v_ij = sigmoid(w_2^T * relu(W_1 * p'_ij + b_1) + b_2)
        self.vis_predictor = nn.Sequential(
            nn.Linear(attn_dim, vis_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vis_hidden_dim, 1),
            nn.Sigmoid()
        )

        # Local projection head: W_l * (\sum p_hat_j) + b_l
        self.proj_local = nn.Linear(attn_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.wq, self.wk, self.wv, self.proj_local]:
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, 0.0)

        for m in self.vis_predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def pool_coarse_parts(self, F_patch: torch.Tensor) -> torch.Tensor:
        """
        Pool 128 patch tokens into 3 coarse vertical parts.
        Args:
            F_patch: Tensor of shape (B, 128, 768)
        Returns:
            parts: Tensor of shape (B, 3, 768)
        """
        B = F_patch.shape[0]
        # Reshape to spatial grid: (B, 16, 8, 768)
        grid = F_patch.view(B, self.grid_h, self.grid_w, self.embed_dim)

        parts = []
        for r_start, r_end in self.region_slices:
            # Slice vertical rows: (B, num_rows, 8, 768)
            part_tokens = grid[:, r_start:r_end, :, :]
            # Mean pool over spatial patch locations within region
            part_feat = part_tokens.reshape(B, -1, self.embed_dim).mean(dim=1)  # (B, 768)
            parts.append(part_feat)

        return torch.stack(parts, dim=1)  # (B, 3, 768)

    def forward(
        self,
        F_patch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            F_patch: Spatial patch tokens of shape (B, 128, 768)
        Returns:
            h_l: Local aggregated feature before BNNeck (B, 512)
            v_pred: Predicted visibility scores for 3 parts (B, 3)
            p_hat: Visibility-modulated contextual parts (B, 3, 768)
        """
        B = F_patch.shape[0]

        # 1. Coarse part pooling: (B, 3, 768)
        p = self.pool_coarse_parts(F_patch)  # [p1, p2, p3]

        # 2. Graph Attention across body parts
        if self.use_gat:
            Q = self.wq(p)  # (B, 3, attn_dim)
            K = self.wk(p)  # (B, 3, attn_dim)
            V = self.wv(p)  # (B, 3, attn_dim)

            # Scaled dot-product attention scores
            # e_jk = (W_Q p_j)^T (W_K p_k) / sqrt(d_a)
            e = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, 3, 3)
            a = F.softmax(e, dim=-1)  # (B, 3, 3)

            # Context-enhanced part representations: p'_j = \sum a_jk V p_k
            p_prime = torch.bmm(a, V)  # (B, 3, attn_dim)
        else:
            p_prime = p

        # 3. Internal Reconstruction Gate:
        # p_tilde_j = gamma_j * p'_j + (1 - gamma_j) * p_j
        if self.use_reconstruction_gate and self.use_gat:
            gamma = torch.sigmoid(self.gate_linear(p))  # (B, 3, 1)
            p_tilde = gamma * p_prime + (1.0 - gamma) * p
        else:
            p_tilde = p_prime

        # 4. Visibility Predictor:
        # v_ij = sigmoid(w_2^T * relu(W_1 * p'_ij + b_1) + b_2)
        v_pred = self.vis_predictor(p_prime).squeeze(-1)  # (B, 3)

        # 5. Visibility-Aware Local Modulation:
        # p_hat_j = v_j * p_tilde_j
        if self.use_vis_gating:
            v_expanded = v_pred.unsqueeze(-1)  # (B, 3, 1)
            p_hat = v_expanded * p_tilde
        else:
            p_hat = p_tilde

        # 6. Aggregation and projection:
        # h_l = W_l (\sum_{j=1}^3 p_hat_j) + b_l
        p_agg = p_hat.sum(dim=1)  # (B, attn_dim)
        h_l = self.proj_local(p_agg)  # (B, out_dim = 512)

        return h_l, v_pred, p_hat
