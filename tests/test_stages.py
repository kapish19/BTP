"""
Unit test for Stage-wise training transitions and parameter trainability:
- Verifies Stage 1 freezing of the entire CLIP backbone.
- Verifies Stage 2 unfreezing of final 2 ViT transformer blocks only.
- Verifies Stage 3 activation of domain classifier and GRL warmup.
"""

import unittest
import torch
from src.models.dg_reid import DGReID
from src.training.stage_manager import StageManager


class TestStages(unittest.TestCase):
    def setUp(self):
        self.model = DGReID(
            num_classes=10,
            num_domains=3,
            clip_dim=768,
            feat_dim=512,
            unfreeze_blocks=2
        )
        self.stage_mgr = StageManager(stage1_epochs=10, stage2_epochs=30, stage3_epochs=60)

    def test_stage_1_trainability(self):
        stage = self.stage_mgr.configure_model_for_epoch(self.model, epoch=5)
        self.assertEqual(stage, 1)

        # Entire backbone must be frozen
        for p in self.model.backbone.parameters():
            self.assertFalse(p.requires_grad)

        # Heads must be trainable
        self.assertTrue(self.model.proj_global.weight.requires_grad)
        self.assertTrue(self.model.part_gat.proj_local.weight.requires_grad)

        # GRL alpha must be 0
        self.assertEqual(self.model.grl.alpha, 0.0)

    def test_stage_2_trainability(self):
        stage = self.stage_mgr.configure_model_for_epoch(self.model, epoch=15)
        self.assertEqual(stage, 2)

        # Blocks 0-9 must be frozen
        for i in range(10):
            for p in self.model.backbone.resblocks[i].parameters():
                self.assertFalse(p.requires_grad)

        # Blocks 10 and 11 must be unfrozen
        for i in [10, 11]:
            for p in self.model.backbone.resblocks[i].parameters():
                self.assertTrue(p.requires_grad)

        # GRL alpha must still be 0
        self.assertEqual(self.model.grl.alpha, 0.0)

    def test_stage_3_trainability(self):
        stage = self.stage_mgr.configure_model_for_epoch(self.model, epoch=35)
        self.assertEqual(stage, 3)

        # Final 2 blocks unfrozen
        for i in [10, 11]:
            for p in self.model.backbone.resblocks[i].parameters():
                self.assertTrue(p.requires_grad)

        # Domain classifier must be trainable
        for p in self.model.domain_classifier.parameters():
            self.assertTrue(p.requires_grad)

        # GRL alpha should be warmed up: (35 - 30) / 10 = 0.5
        self.assertAlmostEqual(self.model.grl.alpha, 0.5)


if __name__ == '__main__':
    unittest.main()
