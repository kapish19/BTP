"""
Unit test for Synthetic Occlusion Generator:
- Verifies exact 16x8 patch-grid occlusion mask derivation.
- Verifies visibility targets for fully visible (1.0) and fully occluded (0.0) regions.
- Verifies deterministic controlled occlusion levels (clean, mild, moderate, severe).
"""

import unittest
import torch
from src.data.occlusion import SyntheticOcclusionGenerator


class TestSyntheticOcclusion(unittest.TestCase):
    def setUp(self):
        self.occ_gen = SyntheticOcclusionGenerator(
            img_size=(256, 128),
            patch_size=16,
            part_splits=(5, 6, 5)
        )

    def test_clean_image_targets(self):
        img = torch.zeros(3, 256, 128)
        corr_img, mask, vis = self.occ_gen.apply_controlled_occlusion(img, severity="clean")

        # Mask should be all zeros
        self.assertEqual(mask.shape, (16, 8))
        self.assertEqual(mask.sum().item(), 0.0)

        # Visibility target should be [1.0, 1.0, 1.0]
        self.assertEqual(vis.shape, (3,))
        self.assertTrue(torch.allclose(vis, torch.tensor([1.0, 1.0, 1.0])))

    def test_controlled_occlusion_severities(self):
        img = torch.ones(3, 256, 128)

        # Test Mild (~20%), Moderate (~35%), Severe (~55%)
        severities = ["mild", "moderate", "severe"]
        for sev in severities:
            corr_img, mask, vis = self.occ_gen.apply_controlled_occlusion(img, severity=sev, seed=123)
            self.assertEqual(mask.shape, (16, 8))
            self.assertEqual(vis.shape, (3,))

            # All visibility targets must be between 0.0 and 1.0
            self.assertTrue((vis >= 0.0).all() and (vis <= 1.0).all())

            # Occlusion mask has at least some occluded patches
            self.assertTrue(mask.sum().item() > 0)

    def test_part_visibility_derivation(self):
        # Create an artificial mask where region 1 (rows 0..4) is 100% occluded
        mask = torch.zeros(16, 8)
        mask[0:5, :] = 1.0  # 100% occluded in head

        vis = self.occ_gen._compute_part_visibilities(mask.numpy())
        self.assertAlmostEqual(vis[0], 0.0)  # Head is 0% visible
        self.assertAlmostEqual(vis[1], 1.0)  # Torso is 100% visible
        self.assertAlmostEqual(vis[2], 1.0)  # Legs are 100% visible


if __name__ == '__main__':
    unittest.main()
