"""
Stage manager governing parameter trainability and GRL schedules across:
- Stage 1: Epochs 1-10 (Backbone frozen)
- Stage 2: Epochs 11-30 (Final 2 ViT blocks unfrozen)
- Stage 3: Epochs 31-60 (Domain classifier active with GRL warmup)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class StageManager:
    """
    Manages the 3-stage training protocol defined in Section 5.14 and Table 3.
    """
    def __init__(
        self,
        stage1_epochs: int = 10,
        stage2_epochs: int = 30,
        stage3_epochs: int = 60,
        grl_ramp_epochs: int = 10
    ):
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.grl_ramp_epochs = grl_ramp_epochs

    def get_stage_for_epoch(self, epoch: int) -> int:
        """
        Maps 1-indexed epoch to stage number (1, 2, or 3).
        """
        if epoch <= self.stage1_epochs:
            return 1
        elif epoch <= self.stage2_epochs:
            return 2
        else:
            return 3

    def configure_model_for_epoch(self, model: nn.Module, epoch: int) -> int:
        """
        Updates model parameter trainability and GRL alpha for current epoch.
        Returns the active stage.
        """
        stage = self.get_stage_for_epoch(epoch)
        if hasattr(model, 'set_stage'):
            model.set_stage(stage=stage, epoch=epoch)
        return stage

    def get_stage_summary(self, epoch: int) -> Dict[str, str]:
        stage = self.get_stage_for_epoch(epoch)
        if stage == 1:
            return {
                "stage": "Stage 1 (Epochs 1-10)",
                "trainable": "Projection heads, BNNecks, identity heads, GAT, visibility, domain factors",
                "frozen": "Entire CLIP ViT-B/16 visual backbone",
                "adversarial": "Inactive (GRL alpha = 0)"
            }
        elif stage == 2:
            return {
                "stage": "Stage 2 (Epochs 11-30)",
                "trainable": "Stage 1 modules + final 2 ViT transformer blocks (blocks 10 & 11)",
                "frozen": "ViT blocks 0-9, patch conv, position/class embeddings",
                "adversarial": "Inactive (GRL alpha = 0)"
            }
        else:
            alpha = max(0.0, min(1.0, (epoch - 30) / float(self.grl_ramp_epochs)))
            return {
                "stage": f"Stage 3 (Epoch {epoch})",
                "trainable": "Stage 2 modules + Domain classifier",
                "frozen": "ViT blocks 0-9, patch conv, position/class embeddings",
                "adversarial": f"Active with GRL alpha = {alpha:.3f}"
            }
