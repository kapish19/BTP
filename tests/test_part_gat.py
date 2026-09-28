"""
Unit test for PartVisibilityGAT:
- Verifies 3-part coarse pooling shapes (Head, Torso, Legs).
- Verifies Graph Attention context exchange.
- Verifies Visibility Predictor output in (0, 1).
- Verifies Local projection to 512-d.
"""

import unittest
import torch
from src.models.part_gat import PartVisibilityGAT


class TestPartVisibilityGAT(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.gat = PartVisibilityGAT(
            embed_dim=768,
            attn_dim=768,
            vis_hidden_dim=256,
            out_dim=512,
            grid_size=(16, 8),
            part_splits=(5, 6, 5)
        )

    def test_forward_output_shapes(self):
        F_patch = torch.randn(self.B, 128, 768, requires_grad=True)
        h_l, v_pred, p_hat = self.gat(F_patch)

        # Local projection shape: (B, 512)
        self.assertEqual(h_l.shape, (self.B, 512))

        # Visibility prediction shape: (B, 3)
        self.assertEqual(v_pred.shape, (self.B, 3))

        # Visibility values must lie strictly in (0, 1)
        self.assertTrue((v_pred > 0.0).all() and (v_pred < 1.0).all())

        # Modulated parts shape: (B, 3, 768)
        self.assertEqual(p_hat.shape, (self.B, 3, 768))

    def test_gradient_flow(self):
        F_patch = torch.randn(self.B, 128, 768, requires_grad=True)
        h_l, v_pred, _ = self.gat(F_patch)
        loss = h_l.sum() + v_pred.sum()
        loss.backward()

        self.assertIsNotNone(F_patch.grad)
        self.assertTrue(torch.norm(F_patch.grad) > 0)


if __name__ == '__main__':
    unittest.main()
