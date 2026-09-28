"""
Synthetic Re-ID dataset generator for smoke testing, CI pipelines, and shape verification.
Creates valid pedestrian crop images and standard folder hierarchies for Market-1501 and MSMT17.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw


def generate_synthetic_reid_dataset(
    output_dir: str,
    num_identities: int = 24,
    images_per_identity: int = 6,
    num_cameras: int = 4,
    num_domains: int = 3,
    img_size: Tuple[int, int] = (256, 128)
) -> Dict[str, str]:
    """
    Generates a synthetic multi-domain dataset on disk with standard Re-ID folder structures.
    
    Structure:
      output_dir/
        domain_0/ (e.g. Market-1501 style)
          bounding_box_train/
          query/
          bounding_box_test/
        domain_1/ (e.g. MSMT17 style)
          bounding_box_train/
          query/
          bounding_box_test/
        domain_2/ (e.g. CUHK03 style)
          bounding_box_train/
          query/
          bounding_box_test/
    """
    os.makedirs(output_dir, exist_ok=True)
    domain_paths = {}

    h, w = img_size
    rng = np.random.RandomState(42)

    for d_idx in range(num_domains):
        domain_name = f"domain_{d_idx}"
        d_dir = os.path.join(output_dir, domain_name)
        train_dir = os.path.join(d_dir, "bounding_box_train")
        query_dir = os.path.join(d_dir, "query")
        gallery_dir = os.path.join(d_dir, "bounding_box_test")

        for p in [train_dir, query_dir, gallery_dir]:
            os.makedirs(p, exist_ok=True)

        # Generate training images
        for pid in range(num_identities):
            # Base color for identity
            base_color = rng.randint(50, 220, size=3)

            for img_idx in range(images_per_identity):
                camid = (img_idx % num_cameras) + 1
                # Create synthetic person silhouette
                img_arr = np.zeros((h, w, 3), dtype=np.uint8)
                # Background
                bg_color = (d_idx * 50 + 20, d_idx * 30 + 40, d_idx * 20 + 60)
                img_arr[:, :] = bg_color

                # Person torso / head / legs
                person_color = np.clip(base_color + rng.randint(-20, 20, size=3), 0, 255).astype(np.uint8)
                img_arr[30:220, 25:103] = person_color
                # Head
                img_arr[30:70, 44:84] = [220, 180, 150]

                pil_img = Image.fromarray(img_arr)

                # Filename: <pid>_c<cam>s1_<frame>_<idx>.jpg
                fname = f"{pid:04d}_c{camid}s1_{img_idx:06d}_{img_idx:02d}.jpg"
                pil_img.save(os.path.join(train_dir, fname))

        # Generate query and gallery images for testing
        test_pids = range(num_identities, num_identities + 8)
        for pid in test_pids:
            base_color = rng.randint(50, 220, size=3)
            # Query (cam 1)
            q_arr = np.zeros((h, w, 3), dtype=np.uint8)
            q_arr[:, :] = (d_idx * 40 + 30, 80, 80)
            q_arr[30:220, 25:103] = base_color
            Image.fromarray(q_arr).save(os.path.join(query_dir, f"{pid:04d}_c1s1_000001_00.jpg"))

            # Gallery (cam 2 and cam 3)
            for g_cam in [2, 3]:
                g_arr = np.zeros((h, w, 3), dtype=np.uint8)
                g_arr[:, :] = (d_idx * 40 + 30, 80, 80)
                g_arr[30:220, 25:103] = np.clip(base_color + rng.randint(-15, 15, size=3), 0, 255).astype(np.uint8)
                Image.fromarray(g_arr).save(os.path.join(gallery_dir, f"{pid:04d}_c{g_cam}s1_000002_00.jpg"))

        domain_paths[domain_name] = d_dir

    return domain_paths
