"""
Synthetic Occlusion Generator and Visibility Target Calculator.
Generates corrupted images x_tilde together with exact patch-grid occlusion masks O_i (16x8),
and computes coarse part visibility ratios m_ij in [0, 1] for Head, Torso, and Legs.
"""

import math
import random
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch


class SyntheticOcclusionGenerator:
    """
    Applies synthetic occlusion to pedestrian images and derives exact patch-grid masks.

    Grid: 16 vertical x 8 horizontal = 128 patches (each 16x16 pixels on 256x128 image).
    Coarse Body Regions:
      - R1: Head / Shoulders: Rows 0 to 4 (5 rows, 40 patches)
      - R2: Torso:           Rows 5 to 10 (6 rows, 48 patches)
      - R3: Legs:            Rows 11 to 15 (5 rows, 40 patches)
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 128),
        patch_size: int = 16,
        part_splits: Tuple[int, int, int] = (5, 6, 5),
        prob: float = 0.5,
        area_range: Tuple[float, float] = (0.1, 0.5),
        aspect_range: Tuple[float, float] = (0.3, 3.3)
    ):
        self.img_h, self.img_w = img_size
        self.patch_size = patch_size
        self.grid_h = self.img_h // patch_size  # 16
        self.grid_w = self.img_w // patch_size  # 8
        self.part_splits = part_splits
        self.prob = prob
        self.area_range = area_range
        self.aspect_range = aspect_range

        # Part row boundaries
        r1_end = part_splits[0]
        r2_end = r1_end + part_splits[1]
        self.part_slices = [
            (0, r1_end),
            (r1_end, r2_end),
            (r2_end, self.grid_h)
        ]

    def _compute_part_visibilities(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute visibility ratio m_j in [0, 1] for each part from occlusion mask.
        mask: (16, 8) with 1 = occluded, 0 = visible
        """
        vis_targets = np.zeros(3, dtype=np.float32)
        for j, (r_start, r_end) in enumerate(self.part_slices):
            part_mask = mask[r_start:r_end, :]
            total_patches = part_mask.size
            occluded_patches = np.sum(part_mask)
            # Visibility ratio = fraction of unoccluded patches
            vis_targets[j] = float(np.clip(1.0 - (occluded_patches / total_patches), 0.0, 1.0))
        return vis_targets

    def apply_controlled_occlusion(
        self,
        tensor_img: torch.Tensor,
        severity: str = "clean",
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply deterministic or controlled occlusion for evaluation protocol (Table 8):
        - 'clean': 0% occlusion
        - 'mild': ~20% area occlusion
        - 'moderate': ~35% area occlusion
        - 'severe': ~55% area occlusion
        """
        if seed is not None:
            rng = random.Random(seed)
            np_rng = np.random.RandomState(seed)
        else:
            rng = random
            np_rng = np.random

        if severity == "clean":
            mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            vis = np.ones(3, dtype=np.float32)
            return tensor_img.clone(), torch.from_numpy(mask), torch.from_numpy(vis)

        target_areas = {
            "mild": 0.20,
            "moderate": 0.35,
            "severe": 0.55
        }
        target_area_ratio = target_areas.get(severity, 0.20)

        total_area = self.img_h * self.img_w
        occ_area = total_area * target_area_ratio

        # Determine box dimensions maintaining person-like occluder aspect ratio
        aspect = rng.uniform(0.5, 2.0)
        h = int(round(math.sqrt(occ_area * aspect)))
        w = int(round(math.sqrt(occ_area / aspect)))
        h = min(self.img_h, max(16, h))
        w = min(self.img_w, max(16, w))

        # Random position
        top = rng.randint(0, self.img_h - h)
        left = rng.randint(0, self.img_w - w)

        corrupted = tensor_img.clone()
        # Fill with random mean/noise or gray
        noise_val = torch.from_numpy(np_rng.uniform(0.0, 1.0, size=(3, h, w)).astype(np.float32)).to(tensor_img.device)
        corrupted[:, top:top+h, left:left+w] = noise_val

        # Project occlusion onto 16x8 patch grid
        # A patch is occluded if more than 30% of its area is covered
        mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for gh in range(self.grid_h):
            for gw in range(self.grid_w):
                p_top = gh * self.patch_size
                p_bottom = p_top + self.patch_size
                p_left = gw * self.patch_size
                p_right = p_left + self.patch_size

                # Compute intersection
                inter_top = max(top, p_top)
                inter_bottom = min(top + h, p_bottom)
                inter_left = max(left, p_left)
                inter_right = min(left + w, p_right)

                if inter_bottom > inter_top and inter_right > inter_left:
                    inter_area = (inter_bottom - inter_top) * (inter_right - inter_left)
                    if (inter_area / (self.patch_size * self.patch_size)) >= 0.30:
                        mask[gh, gw] = 1.0

        vis = self._compute_part_visibilities(mask)
        return corrupted, torch.from_numpy(mask), torch.from_numpy(vis)

    def __call__(
        self,
        tensor_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stochastic training occlusion.
        Args:
            tensor_img: Normalized image tensor of shape (C, H, W)
        Returns:
            corrupted_img: Corrupted image tensor (C, H, W)
            mask: Patch grid mask (16, 8), 1 = occluded, 0 = visible
            vis_targets: Part visibility ratios (3,), values in [0, 1]
        """
        if random.random() > self.prob:
            mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            vis = np.ones(3, dtype=np.float32)
            return tensor_img, torch.from_numpy(mask), torch.from_numpy(vis)

        # Random area and aspect ratio
        total_area = self.img_h * self.img_w
        target_area = random.uniform(self.area_range[0], self.area_range[1]) * total_area
        aspect = random.uniform(self.aspect_range[0], self.aspect_range[1])

        h = int(round(math.sqrt(target_area * aspect)))
        w = int(round(math.sqrt(target_area / aspect)))
        h = min(self.img_h, max(16, h))
        w = min(self.img_w, max(16, w))

        top = random.randint(0, self.img_h - h)
        left = random.randint(0, self.img_w - w)

        corrupted = tensor_img.clone()
        # Random fill: random uniform noise, random solid color, or Gaussian noise
        fill_type = random.choice(['noise', 'mean', 'random_solid'])
        if fill_type == 'noise':
            patch_fill = torch.rand(3, h, w, dtype=tensor_img.dtype, device=tensor_img.device)
        elif fill_type == 'mean':
            patch_fill = tensor_img.mean(dim=(1, 2), keepdim=True).expand(3, h, w)
        else:
            solid_c = torch.rand(3, 1, 1, dtype=tensor_img.dtype, device=tensor_img.device).expand(3, h, w)
            patch_fill = solid_c

        corrupted[:, top:top+h, left:left+w] = patch_fill

        # Map to 16x8 grid
        mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for gh in range(self.grid_h):
            for gw in range(self.grid_w):
                p_top = gh * self.patch_size
                p_bottom = p_top + self.patch_size
                p_left = gw * self.patch_size
                p_right = p_left + self.patch_size

                inter_top = max(top, p_top)
                inter_bottom = min(top + h, p_bottom)
                inter_left = max(left, p_left)
                inter_right = min(left + w, p_right)

                if inter_bottom > inter_top and inter_right > inter_left:
                    inter_area = (inter_bottom - inter_top) * (inter_right - inter_left)
                    if (inter_area / (self.patch_size * self.patch_size)) >= 0.30:
                        mask[gh, gw] = 1.0

        vis = self._compute_part_visibilities(mask)
        return corrupted, torch.from_numpy(mask), torch.from_numpy(vis)
