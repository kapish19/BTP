"""
Unit test for Gradient Reversal Layer (GRL):
- Verifies forward identity mapping.
- Verifies exact reversed gradient sign: dR_lambda / dx = -lambda * I.
- Verifies warmup schedule: alpha(e) = min(1, (e - 30) / 10).
"""

import unittest
import torch
from src.models.grl import GradientReversalLayer, GradientReversalFunction


class TestGRL(unittest.TestCase):
    def test_forward_identity(self):
        x = torch.randn(4, 768, requires_grad=True)
        grl = GradientReversalLayer(alpha=1.0)
        grl.train()
        out = grl(x)
        self.assertTrue(torch.allclose(out, x))

    def test_reversed_gradient_sign(self):
        alpha = 0.5
        x = torch.randn(4, 768, requires_grad=True)
        # Dummy linear operation
        w = torch.randn(768, 3, requires_grad=True)

        grl = GradientReversalLayer(alpha=alpha)
        grl.train()
        u = grl(x)
        loss = (u @ w).sum()
        loss.backward()

        # The analytical gradient w.r.t u is: sum_c w[:, c]
        expected_grad_u = w.sum(dim=1).unsqueeze(0).expand(4, 768)
        # Gradient w.r.t x must be exactly -alpha * expected_grad_u
        expected_grad_x = -alpha * expected_grad_u

        self.assertTrue(torch.allclose(x.grad, expected_grad_x, atol=1e-5))

    def test_alpha_schedule(self):
        grl = GradientReversalLayer()
        # Before stage 3: alpha = 0
        grl.set_alpha(epoch=10)
        self.assertEqual(grl.alpha, 0.0)
        grl.set_alpha(epoch=30)
        self.assertEqual(grl.alpha, 0.0)

        # Stage 3 warm-up:
        # epoch 31: (31 - 30) / 10 = 0.1
        grl.set_alpha(epoch=31)
        self.assertAlmostEqual(grl.alpha, 0.1)

        # epoch 35: (35 - 30) / 10 = 0.5
        grl.set_alpha(epoch=35)
        self.assertAlmostEqual(grl.alpha, 0.5)

        # epoch 40: (40 - 30) / 10 = 1.0
        grl.set_alpha(epoch=40)
        self.assertAlmostEqual(grl.alpha, 1.0)

        # epoch 50: min(1, ...) = 1.0
        grl.set_alpha(epoch=50)
        self.assertAlmostEqual(grl.alpha, 1.0)


if __name__ == '__main__':
    unittest.main()
