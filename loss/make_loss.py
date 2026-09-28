"""
DG-ReID Loss Function Module.
Implements the complete 7-component joint optimization objective from Eq. (1)
and Table 2 of the BTP report:
  L_total = lambda_id   * L_id^g   + lambda_tri      * L_tri^g
          + lambda_id^l * L_id^l   + lambda_tri^l    * L_tri^l
          + lambda_occ  * L_occ    + lambda_adv      * L_dom
          + lambda_ortho* L_ortho

Default Weights (Table 2):
  lambda_id       = 1.0  (Global identity classification)
  lambda_tri      = 1.0  (Global triplet metric separation)
  lambda_id_local = 0.5  (Local identity classification)
  lambda_tri_local= 0.5  (Local triplet metric separation)
  lambda_occ      = 0.3  (Synthetic-mask visibility supervision)
  lambda_adv      = 0.1  (Visual-side domain confusion through GRL)
  lambda_ortho    = 0.05 (Latent domain-factor diversity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet_loss import TripletLoss
from .softmax_loss import CrossEntropyLabelSmooth


class TotalLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        lambda_id=1.0,
        lambda_tri=1.0,
        lambda_id_local=0.5,
        lambda_tri_local=0.5,
        lambda_occ=0.3,
        lambda_adv=0.1,
        lambda_ortho=0.05,
        triplet_margin=0.3,
        label_smooth=0.1
    ):
        super().__init__()
        self.num_classes = num_classes

        # Loss components
        self.xent_global = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=label_smooth)
        self.xent_local = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=label_smooth)
        self.triplet_global = TripletLoss(margin=triplet_margin)
        self.triplet_local = TripletLoss(margin=triplet_margin)
        self.bce_occ = nn.BCEWithLogitsLoss()
        self.xent_domain = nn.CrossEntropyLoss()

        # Loss weights
        self.lambda_id = float(lambda_id)
        self.lambda_tri = float(lambda_tri)
        self.lambda_id_local = float(lambda_id_local)
        self.lambda_tri_local = float(lambda_tri_local)
        self.lambda_occ = float(lambda_occ)
        self.lambda_adv = float(lambda_adv)
        self.lambda_ortho = float(lambda_ortho)

    def forward(
        self,
        outputs,
        targets,
        domain_labels=None,
        occ_targets=None,
        stage=3,
        grl_alpha=1.0,
        disable_grl=False,
        disable_part_branch=False
    ):
        """
        Args:
            outputs: Dictionary returned by DGReID.forward:
              - 'id_logits': [B, C]
              - 'global_feat': [B, 512]
              - 'local_logits': [B, C]
              - 'local_feat': [B, 512]
              - 'vis_logits': [B, 3, 1]
              - 'vis_scores': [B, 3, 1]
              - 'domain_logits': [B, M]
              - 'ortho_loss': scalar
            targets: [B] identity labels
            domain_labels: [B] source domain indices
            occ_targets: [B, 3, 1] soft visibility ratios in [0, 1]
            stage: Training stage (1, 2, or 3)
            grl_alpha: Current GRL factor alpha(e)
            disable_grl: If True, omit domain loss
            disable_part_branch: If True, omit local ID, local triplet, and visibility losses
        Returns:
            total_loss: scalar tensor for backpropagation
            loss_dict: dictionary of float values for detailed logging
        """
        device = targets.device

        # 1. Global Identity Loss (L_id^g)
        L_id_g = self.xent_global(outputs['id_logits'], targets)

        # 2. Global Triplet Loss (L_tri^g)
        L_tri_g = self.triplet_global(outputs['global_feat'], targets)[0]

        # 3. Local Identity and Triplet Losses (L_id^l, L_tri^l)
        if disable_part_branch:
            L_id_l = torch.tensor(0.0, device=device)
            L_tri_l = torch.tensor(0.0, device=device)
            L_occ = torch.tensor(0.0, device=device)
        else:
            L_id_l = self.xent_local(outputs['local_logits'], targets)
            L_tri_l = self.triplet_local(outputs['local_feat'], targets)[0]

            # 4. Visibility Loss (L_occ) - Section 4.5 & 5.8
            if occ_targets is not None:
                # Shape matching: ensure occ_targets is [B, 3, 1] float
                if occ_targets.dim() == 2:
                    occ_targets = occ_targets.unsqueeze(-1)
                occ_targets = occ_targets.float().to(device)

                if 'vis_logits' in outputs and outputs['vis_logits'] is not None:
                    L_occ = self.bce_occ(outputs['vis_logits'], occ_targets)
                else:
                    vis_s = outputs['vis_scores'].clamp(1e-6, 1.0 - 1e-6)
                    L_occ = F.binary_cross_entropy(vis_s, occ_targets)
            else:
                L_occ = torch.tensor(0.0, device=device)

        # 5. Domain Loss (L_dom) - Section 4.4 & 5.5
        # Only active in Stage 3 when GRL warmup is non-zero and domain labels are provided
        if stage == 3 and not disable_grl and domain_labels is not None and grl_alpha > 0.0:
            L_dom = self.xent_domain(outputs['domain_logits'], domain_labels)
        else:
            L_dom = torch.tensor(0.0, device=device)

        # 6. Orthogonal Domain Factors Loss (L_ortho) - Section 5.11
        L_ortho = outputs.get('ortho_loss', torch.tensor(0.0, device=device))

        # Joint optimization objective (Eq. 1)
        total_loss = (
            self.lambda_id * L_id_g +
            self.lambda_tri * L_tri_g +
            self.lambda_id_local * L_id_l +
            self.lambda_tri_local * L_tri_l +
            self.lambda_occ * L_occ +
            self.lambda_adv * L_dom +
            self.lambda_ortho * L_ortho
        )

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_id_global': L_id_g.item(),
            'loss_tri_global': L_tri_g.item(),
            'loss_id_local': L_id_l.item() if not disable_part_branch else 0.0,
            'loss_tri_local': L_tri_l.item() if not disable_part_branch else 0.0,
            'loss_occ': L_occ.item() if not disable_part_branch else 0.0,
            'loss_dom': L_dom.item() if (stage == 3 and not disable_grl) else 0.0,
            'loss_ortho': L_ortho.item()
        }

        return total_loss, loss_dict


def make_loss(num_classes, args=None):
    """Factory function for TotalLoss."""
    lambda_id = getattr(args, 'lambda_id', 1.0) if args else 1.0
    lambda_tri = getattr(args, 'lambda_tri', 1.0) if args else 1.0
    lambda_id_loc = getattr(args, 'lambda_id_local', 0.5) if args else 0.5
    lambda_tri_loc = getattr(args, 'lambda_tri_local', 0.5) if args else 0.5
    lambda_occ = getattr(args, 'lambda_occ', 0.3) if args else 0.3
    lambda_adv = getattr(args, 'lambda_adv', 0.1) if args else 0.1
    lambda_ortho = getattr(args, 'lambda_ortho', 0.05) if args else 0.05

    return TotalLoss(
        num_classes=num_classes,
        lambda_id=lambda_id,
        lambda_tri=lambda_tri,
        lambda_id_local=lambda_id_loc,
        lambda_tri_local=lambda_tri_loc,
        lambda_occ=lambda_occ,
        lambda_adv=lambda_adv,
        lambda_ortho=lambda_ortho
    )
