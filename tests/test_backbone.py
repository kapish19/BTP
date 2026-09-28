"""
Unit test for CLIP ViT-B/16 visual backbone:
- Verifies output tensor shapes for 256x128 input.
- Verifies 16x8 (128) patch token reconstruction.
- Verifies bicubic positional embedding interpolation.
"""

import unittest
import torch
from src.models.clip_vit import CLIPVisionTransformer


class TestCLIPBackbone(unittest.TestCase):
    def setUp(self):
        self.img_size = (256, 128)
        self.batch_size = 2
        self.model = CLIPVisionTransformer(
            img_size=self.img_size,
            patch_size=16,
            embed_dim=768,
            depth=4,  # Small depth for fast unit test
            num_heads=8,
            unfreeze_blocks=2
        )

    def test_forward_shapes(self):
        x = torch.randn(self.batch_size, 3, *self.img_size)
        f_cls, F_patch = self.model(x)

        # Check CLS token shape: (B, 768)
        self.assertEqual(f_cls.shape, (self.batch_size, 768))

        # Check patch tokens shape: (B, 128, 768)
        self.assertEqual(F_patch.shape, (self.batch_size, 128, 768))

    def test_bicubic_pos_interpolation(self):
        # Original 14x14 grid CLIP pos embedding (197, 768)
        orig_pos = torch.randn(197, 768)
        self.model.interpolate_pos_encoding(orig_pos, orig_grid=(14, 14))

        # Target positional embedding should now be 1 + 16x8 = 129 tokens
        self.assertEqual(self.model.positional_embedding.shape, (129, 768))


if __name__ == '__main__':
    unittest.main()
