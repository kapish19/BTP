"""
Stage-wise Training Manager for DG-ReID.
Implements the 3-stage training protocol from Table 3 and Algorithm 1 of the BTP report:
  - Stage 1 (Epochs 1-10): Keep CLIP visual backbone frozen.
                           Train projection heads, BNNecks, identity heads, part GAT,
                           visibility head, and domain factors.
  - Stage 2 (Epochs 11-30): Stage-1 modules + unfreeze final 2 ViT transformer blocks.
  - Stage 3 (Epochs 31-60): Stage-2 modules + domain classifier with GRL warm-up:
                           alpha(e) = min(1.0, (e - 30) / 10), e >= 31.
"""

import torch
from .make_optimizer import build_optimizer


def get_current_stage(epoch, stage1_epochs=10, stage2_epochs=30):
    """
    Returns the stage index (1, 2, or 3) for the given epoch.
    """
    if epoch <= stage1_epochs:
        return 1
    elif epoch <= stage2_epochs:
        return 2
    else:
        return 3


class StageManager:
    def __init__(
        self,
        model,
        cfg,
        stage1_epochs=10,
        stage2_epochs=30,
        total_epochs=60
    ):
        self.model = model
        self.cfg = cfg
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.total_epochs = total_epochs
        self.current_stage = 0

    def get_stage(self, epoch):
        return get_current_stage(epoch, self.stage1_epochs, self.stage2_epochs)

    def get_grl_alpha(self, epoch):
        if epoch <= self.stage2_epochs:
            return 0.0
        warmup = max(1, self.total_epochs - self.stage2_epochs)
        warmup = 10 # exact formulation from section 5.5: min(1, (e - 30)/10)
        return min(1.0, float(epoch - self.stage2_epochs) / float(warmup))

    def update_stage(self, epoch, optimizer=None, logger=None):
        """
        Checks if the stage has changed, updates model parameter freezing,
        and optionally rebuilds optimizer parameter groups.
        """
        stage = self.get_stage(epoch)
        stage_changed = (stage != self.current_stage)

        if stage_changed:
            self.current_stage = stage
            if hasattr(self.model, 'set_stage'):
                self.model.set_stage(stage)

            msg = f"\n>>> [StageManager] Transitioning to Stage {stage} at Epoch {epoch} <<<"
            if stage == 1:
                msg += "\n    Components: Heads, BNNecks, GAT, Visibility, Domain Factors (Backbone Frozen)"
            elif stage == 2:
                msg += "\n    Components: Stage 1 + Unfreezing final 2 ViT transformer blocks"
            elif stage == 3:
                msg += "\n    Components: Stage 2 + Domain Classifier with GRL Warm-up"

            if logger:
                logger.info(msg)
            else:
                print(msg)

            # Rebuild optimizer parameter groups if stage transitioned
            if optimizer is not None:
                new_opt = build_optimizer(self.model, self.cfg)
                # Transfer state if compatible
                optimizer.param_groups = new_opt.param_groups

        # Update GRL alpha
        grl_alpha = self.get_grl_alpha(epoch)
        if hasattr(self.model, 'set_grl_alpha'):
            self.model.set_grl_alpha(grl_alpha)

        return stage, grl_alpha
