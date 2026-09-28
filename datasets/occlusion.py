"""
Synthetic Occlusion Generation and Visibility Target Extraction.
Implements the exact formulation from Sections 4.5, 5.8, and Table 8 of the BTP report:
  1. Selected training images undergo synthetic occlusion with probability p=0.5.
  2. Produces corrupted image \tilde{x}_i and exact patch-grid visibility mask O_i in {0, 1}^(16 x 8).
  3. Computes continuous soft visibility targets m_{ij} in [0, 1] for 3 coarse body parts:
       R_1 (Upper: Head/Shoulders, rows 0..4)
       R_2 (Middle: Torso,         rows 5..10)
       R_3 (Lower: Legs,           rows 11..15)
     where m_{ij} = (visible patches in R_j) / (total patches in R_j).
     Fully visible:  m_{ij} = 1.0
     Fully occluded: m_{ij} = 0.0
  4. Provides deterministic target-query occlusion generators for Clean, Mild, Moderate, Severe
     evaluation as reported in Table 8.
"""

import math
import random
import torch
import numpy as np


class SyntheticOcclusionTransform:
    def __init__(
        self,
        p=0.5,
        grid_hw=(16, 8),
        area_range=(0.1, 0.45),
        aspect_ratio_range=(0.5, 2.0),
        erase_value=0.0
    ):
        self.p = p
        self.grid_h, self.grid_w = grid_hw
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.erase_value = erase_value

        # Part row boundaries on 16-row grid:
        # R1: rows 0..4 (5 rows, 40 patches)
        # R2: rows 5..10 (6 rows, 48 patches)
        # R3: rows 11..15 (5 rows, 40 patches)
        self.r1_end = int(round(self.grid_h * 5.0 / 16.0))  # 5
        self.r2_end = int(round(self.grid_h * 11.0 / 16.0)) # 11

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: Tensor of shape [C, H, W] (e.g. [3, 256, 128])
        Returns:
            corrupted_img: Tensor [C, H, W]
            visibility_targets: Tensor [3, 1] in [0.0, 1.0] (1.0 = fully visible, 0.0 = fully occluded)
            grid_mask: Tensor [grid_h, grid_w] in {0.0, 1.0}
        """
        C, H, W = img_tensor.shape
        grid_mask = torch.ones((self.grid_h, self.grid_w), dtype=torch.float32)
        vis_targets = torch.ones((3, 1), dtype=torch.float32)

        if random.random() > self.p:
            return img_tensor, vis_targets, grid_mask

        corrupted_img = img_tensor.clone()

        # Target occlusion area fraction
        target_area = random.uniform(self.area_range[0], self.area_range[1]) * (H * W)
        aspect_ratio = random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])

        occ_h = int(round(math.sqrt(target_area * aspect_ratio)))
        occ_w = int(round(math.sqrt(target_area / aspect_ratio)))

        occ_h = min(max(1, occ_h), H - 1)
        occ_w = min(max(1, occ_w), W - 1)

        y0 = random.randint(0, H - occ_h)
        x0 = random.randint(0, W - occ_w)
        y1 = y0 + occ_h
        x1 = x0 + occ_w

        # Corrupt region
        corrupted_img[:, y0:y1, x0:x1] = self.erase_value

        # Project occlusion onto patch grid (16 x 8)
        patch_h = H / float(self.grid_h)
        patch_w = W / float(self.grid_w)

        for r in range(self.grid_h):
            py0 = r * patch_h
            py1 = (r + 1) * patch_h
            inter_y = max(0.0, min(y1, py1) - max(y0, py0))
            if inter_y <= 0:
                continue

            for c in range(self.grid_w):
                px0 = c * patch_w
                px1 = (c + 1) * patch_w
                inter_x = max(0.0, min(x1, px1) - max(x0, px0))

                if inter_x > 0:
                    inter_area = inter_y * inter_x
                    patch_area = patch_h * patch_w
                    # If more than 30% of the patch is occluded, mark patch as occluded
                    if inter_area / patch_area > 0.3:
                        grid_mask[r, c] = 0.0

        # Calculate exact continuous visibility proportion for the 3 coarse parts
        # R_1: Upper (Head / Shoulders)
        r1_patches = grid_mask[0:self.r1_end, :]
        vis_targets[0, 0] = r1_patches.mean()

        # R_2: Middle (Torso)
        r2_patches = grid_mask[self.r1_end:self.r2_end, :]
        vis_targets[1, 0] = r2_patches.mean()

        # R_3: Lower (Legs)
        r3_patches = grid_mask[self.r2_end:self.grid_h, :]
        vis_targets[2, 0] = r3_patches.mean()

        return corrupted_img, vis_targets, grid_mask


def apply_target_occlusion(img_tensor, severity="clean", seed=None):
    """
    Applies deterministic occlusion to target query images as specified in Table 8:
      - 'clean':    No occlusion (0%)
      - 'mild':     ~15% area occlusion (e.g. bottom of legs or partial arm)
      - 'moderate': ~30% area occlusion (e.g. lower body or backpack)
      - 'severe':   ~50% area occlusion (e.g. torso and lower body)
    """
    if severity == "clean":
        return img_tensor.clone()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    C, H, W = img_tensor.shape
    corrupted = img_tensor.clone()

    severity_ratios = {
        'mild': 0.15,
        'moderate': 0.30,
        'severe': 0.50
    }
    area_frac = severity_ratios.get(severity, 0.0)
    if area_frac <= 0:
        return corrupted

    target_area = area_frac * (H * W)
    # Realistic pedestrian occluders are typically horizontal or vertical barriers
    aspect_ratio = random.uniform(0.7, 1.5)
    occ_h = int(round(math.sqrt(target_area * aspect_ratio)))
    occ_w = int(round(math.sqrt(target_area / aspect_ratio)))

    occ_h = min(max(1, occ_h), H - 1)
    occ_w = min(max(1, occ_w), W - 1)

    # Position occluder realistically: bottom (pedestrian legs), center, or side
    y0 = random.randint(0, H - occ_h)
    x0 = random.randint(0, W - occ_w)

    corrupted[:, y0:y0+occ_h, x0:x0+occ_w] = 0.0
    return corrupted
