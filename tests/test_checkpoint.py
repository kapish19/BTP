"""
Unit test for checkpoint saving and loading:
- Verifies that saving and loading state dict produces exact identical inference embeddings.
"""

import os
import tempfile
import unittest
import torch
from src.models.dg_reid import DGReID
from src.utils.checkpoint import save_checkpoint, load_checkpoint


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_roundtrip_inference(self):
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            # Model 1 with depth=2 for fast test
            model1 = DGReID(num_classes=20, num_domains=3, depth=2)
            model1.eval()

            # Input batch
            x = torch.randn(2, 3, 256, 128)
            with torch.no_grad():
                feat1 = model1(x)

            # Save state
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
            save_checkpoint({'model_state_dict': model1.state_dict(), 'epoch': 5}, tmpdir, filename="test_ckpt.pth")

            # Model 2
            model2 = DGReID(num_classes=20, num_domains=3, depth=2)
            load_checkpoint(ckpt_path, model2)
            model2.eval()

            with torch.no_grad():
                feat2 = model2(x)

            # Assert embeddings are numerically identical
            self.assertTrue(torch.allclose(feat1, feat2, atol=1e-6))

    def test_checkpoint_shape_mismatch_resilience(self):
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            # Model 1 trained with 20 classes and 2 domains (e.g. synthetic)
            model1 = DGReID(num_classes=20, num_domains=2, depth=2)
            model1.eval()

            x = torch.randn(2, 3, 256, 128)
            with torch.no_grad():
                feat1 = model1(x)

            ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
            save_checkpoint({'model_state_dict': model1.state_dict(), 'epoch': 5}, tmpdir, filename="test_ckpt.pth")

            # Model 2 initialized with 100 classes and 4 domains (e.g. evaluating on benchmark)
            model2 = DGReID(num_classes=100, num_domains=4, depth=2)
            # Should not raise RuntimeError: size mismatch
            load_checkpoint(ckpt_path, model2, strict=False)
            model2.eval()

            with torch.no_grad():
                feat2 = model2(x)

            # Feature extractor embeddings should still match exactly
            self.assertTrue(torch.allclose(feat1, feat2, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
