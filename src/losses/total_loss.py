"""
Total composite loss implementation combining all 7 training objectives:
L_total = lambda_id * L_id^g + lambda_tri * L_tri^g
        + lambda_id_loc * L_id^l + lambda_tri_loc * L_tri^l
        + lambda_occ * L_occ + lambda_adv * L_dom + lambda_ortho * L_ortho
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_entropy import LabelSmoothingCrossEntropy
from .triplet import BatchHardTripletLoss
from .visibility_loss import VisibilityLoss


class TotalLoss(nn.Module):
    """
    Composite loss manager for DG-ReID.
    Automatically manages loss weights and active components across Stage 1, 2, and 3.
    """
    def __init__(
        self,
        lambda_id: float = 1.0,
        lambda_tri: float = 1.0,
        lambda_id_loc: float = 0.5,
        lambda_tri_loc: float = 0.5,
        lambda_occ: float = 0.3,
        lambda_adv: float = 0.1,
        lambda_ortho: float = 0.05,
        triplet_margin: float = 0.3,
        label_smooth_eps: float = 0.1
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.lambda_tri = lambda_tri
        self.lambda_id_loc = lambda_id_loc
        self.lambda_tri_loc = lambda_tri_loc
        self.lambda_occ = lambda_occ
        self.lambda_adv = lambda_adv
        self.lambda_ortho = lambda_ortho

        self.ce_loss = LabelSmoothingCrossEntropy(epsilon=label_smooth_eps)
        self.triplet_loss = BatchHardTripletLoss(margin=triplet_margin)
        self.vis_loss = VisibilityLoss()
        self.dom_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        pids: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        vis_targets: Optional[torch.Tensor] = None,
        stage: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dictionary of forward outputs from DGReID model
            pids: Person identity labels (B,)
            domain_labels: Domain labels (B,)
            vis_targets: Soft visibility target ratios (B, 3)
            stage: Current training stage (1, 2, or 3)
        Returns:
            total_loss: Total scalar loss
            loss_dict: Dictionary of individual loss values for logging
        """
        device = pids.device
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        # 1. Global Identity Loss (L_id^g)
        if outputs.get('logits_g') is not None and self.lambda_id > 0:
            l_id_g = self.ce_loss(outputs['logits_g'], pids)
            total_loss = total_loss + self.lambda_id * l_id_g
            loss_dict['loss_id_g'] = l_id_g.item()

        # 2. Global Triplet Loss (L_tri^g)
        if outputs.get('h_g') is not None and self.lambda_tri > 0:
            l_tri_g, dist_ap_g, dist_an_g = self.triplet_loss(outputs['h_g'], pids)
            total_loss = total_loss + self.lambda_tri * l_tri_g
            loss_dict['loss_tri_g'] = l_tri_g.item()
            loss_dict['dist_ap_g'] = dist_ap_g.item()
            loss_dict['dist_an_g'] = dist_an_g.item()

        # 3. Local Identity Loss (L_id^l)
        if outputs.get('logits_l') is not None and self.lambda_id_loc > 0:
            l_id_l = self.ce_loss(outputs['logits_l'], pids)
            total_loss = total_loss + self.lambda_id_loc * l_id_l
            loss_dict['loss_id_l'] = l_id_l.item()

        # 4. Local Triplet Loss (L_tri^l)
        if outputs.get('h_l') is not None and self.lambda_tri_loc > 0:
            l_tri_l, dist_ap_l, dist_an_l = self.triplet_loss(outputs['h_l'], pids)
            total_loss = total_loss + self.lambda_tri_loc * l_tri_l
            loss_dict['loss_tri_l'] = l_tri_l.item()
            loss_dict['dist_ap_l'] = dist_ap_l.item()
            loss_dict['dist_an_l'] = dist_an_l.item()

        # 5. Visibility Loss (L_occ)
        if outputs.get('v_pred') is not None and vis_targets is not None and self.lambda_occ > 0:
            l_occ = self.vis_loss(outputs['v_pred'], vis_targets)
            total_loss = total_loss + self.lambda_occ * l_occ
            loss_dict['loss_occ'] = l_occ.item()

        # 6. Domain-Invariance Loss (L_dom) - active only in Stage 3
        if stage == 3 and outputs.get('domain_logits') is not None and domain_labels is not None and self.lambda_adv > 0:
            l_dom = self.dom_loss_fn(outputs['domain_logits'], domain_labels)
            total_loss = total_loss + self.lambda_adv * l_dom
            loss_dict['loss_dom'] = l_dom.item()

        # 7. Orthogonality Loss (L_ortho)
        if outputs.get('loss_ortho') is not None and self.lambda_ortho > 0:
            l_ortho = outputs['loss_ortho']
            total_loss = total_loss + self.lambda_ortho * l_ortho
            loss_dict['loss_ortho'] = l_ortho.item()

        loss_dict['loss_total'] = total_loss.item()
        return total_loss, loss_dict
