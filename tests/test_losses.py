"""
Unit test for loss functions and backpropagation:
- Verifies label-smoothed cross entropy loss.
- Verifies batch-hard triplet loss with hardest mining.
- Verifies visibility BCE loss.
- Verifies orthogonality loss.
- Verifies non-zero gradients and numerical finiteness.
"""

import unittest
import torch
from src.losses.cross_entropy import LabelSmoothingCrossEntropy
from src.losses.triplet import BatchHardTripletLoss
from src.losses.visibility_loss import VisibilityLoss
from src.losses.orthogonality_loss import OrthogonalityLoss
from src.losses.total_loss import TotalLoss


class TestLosses(unittest.TestCase):
    def setUp(self):
        self.B = 8
        self.C = 10
        self.D = 512

    def test_label_smoothing_ce(self):
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (self.B,))
        loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)

        loss = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.item() > 0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_triplet_loss(self):
        features = torch.randn(self.B, self.D, requires_grad=True)
        # 4 identities, 2 samples each
        targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss_fn = BatchHardTripletLoss(margin=0.3)

        loss, ap, an = loss_fn(features, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(ap))
        self.assertTrue(torch.isfinite(an))

        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_visibility_loss(self):
        v_pred = torch.sigmoid(torch.randn(self.B, 3)).requires_grad_(True)
        m_target = torch.rand(self.B, 3)
        loss_fn = VisibilityLoss()

        loss = loss_fn(v_pred, m_target)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.item() > 0)

        loss.backward()
        self.assertIsNotNone(v_pred.grad)
        self.assertTrue(torch.isfinite(v_pred.grad).all())

    def test_orthogonality_loss(self):
        factors = torch.randn(self.D, 4, requires_grad=True)
        loss_fn = OrthogonalityLoss()

        loss = loss_fn(factors)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(factors.grad)

    def test_total_loss_integration(self):
        criterion = TotalLoss()
        outputs = {
            'logits_g': torch.randn(self.B, self.C, requires_grad=True),
            'h_g': torch.randn(self.B, self.D, requires_grad=True),
            'logits_l': torch.randn(self.B, self.C, requires_grad=True),
            'h_l': torch.randn(self.B, self.D, requires_grad=True),
            'v_pred': torch.sigmoid(torch.randn(self.B, 3)).requires_grad_(True),
            'domain_logits': torch.randn(self.B, 3, requires_grad=True),
            'loss_ortho': torch.tensor(0.01, requires_grad=True)
        }
        pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        domains = torch.randint(0, 3, (self.B,))
        vis_targets = torch.rand(self.B, 3)

        total_loss, loss_dict = criterion(
            outputs=outputs,
            pids=pids,
            domain_labels=domains,
            vis_targets=vis_targets,
            stage=3
        )

        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in loss_dict.values()))

        total_loss.backward()
        self.assertIsNotNone(outputs['logits_g'].grad)
        self.assertIsNotNone(outputs['logits_l'].grad)


if __name__ == '__main__':
    unittest.main()
