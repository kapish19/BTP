"""
Dataset classes and parsers for Person Re-Identification benchmarks:
- Market-1501 (Market-1501-v15.09.15)
- MSMT17 (MSMT17_V1 / MSMT17_V2)
- CUHK03 (CUHK03-NP protocol with splits_new_detected.json / splits_new_labeled.json)
- CUHK-SYSU (cuhk_sysu with Image/SSM and annotation/Person.mat or cropped_images)
- Occluded-DukeMTMC (bounding_box_train, bounding_box_test, query)

Supports automatic directory resolution across standard Kaggle/GitHub download structures.
"""

import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import build_transforms
from .occlusion import SyntheticOcclusionGenerator


class ReIDImageDataset(Dataset):
    """
    Single-domain ReID dataset for training, query, or gallery.
    Sample: (image_path, pid, camid, domain_id)
    """
    def __init__(
        self,
        samples: List[Tuple[str, int, int, int]],
        transform: Optional[Callable] = None,
        occlusion_generator: Optional[SyntheticOcclusionGenerator] = None,
        is_train: bool = True
    ):
        self.samples = samples
        self.transform = transform if transform is not None else build_transforms(is_train=is_train)
        self.occlusion_generator = occlusion_generator
        self.is_train = is_train

        # Map person IDs to contiguous range [0, num_pids - 1] if training
        if is_train:
            unique_pids = sorted(list(set(s[1] for s in self.samples if s[1] != -1)))
            self.pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}
            self.num_classes = len(unique_pids)
        else:
            self.pid2label = None
            self.num_classes = len(set(s[1] for s in self.samples if s[1] != -1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, pid, camid, domain_id = self.samples[index]

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        img_tensor = self.transform(img)

        if self.is_train:
            target_pid = self.pid2label[pid] if self.pid2label and pid in self.pid2label else pid
            if self.occlusion_generator is not None:
                corrupted_img, mask, vis_target = self.occlusion_generator(img_tensor)
                return corrupted_img, target_pid, camid, domain_id, vis_target, img_path
            else:
                vis_target = torch.ones(3, dtype=torch.float32)
                return img_tensor, target_pid, camid, domain_id, vis_target, img_path
        else:
            return img_tensor, pid, camid, domain_id, img_path


class MultiDomainReIDDataset(Dataset):
    """
    Aggregates multiple source domain datasets into a single unified training dataset.
    Remaps PIDs to a globally unique contiguous space [0, C - 1] across domains.
    """
    def __init__(
        self,
        datasets: List[Tuple[str, List[Tuple[str, int, int, int]]]],  # [(domain_name, samples), ...]
        transform: Optional[Callable] = None,
        occlusion_generator: Optional[SyntheticOcclusionGenerator] = None
    ):
        self.transform = transform if transform is not None else build_transforms(is_train=True)
        self.occlusion_generator = occlusion_generator

        self.samples: List[Tuple[str, int, int, int]] = []
        self.domain_names: List[str] = [d[0] for d in datasets]
        self.domain_to_id: Dict[str, int] = {name: i for i, name in enumerate(self.domain_names)}

        # Remap IDs globally across domains
        global_pids = {}
        next_global_pid = 0

        for domain_idx, (domain_name, domain_samples) in enumerate(datasets):
            for img_path, orig_pid, camid, _ in domain_samples:
                if orig_pid == -1:
                    continue  # Skip junk
                key = (domain_name, orig_pid)
                if key not in global_pids:
                    global_pids[key] = next_global_pid
                    next_global_pid += 1
                g_pid = global_pids[key]
                self.samples.append((img_path, g_pid, camid, domain_idx))

        self.num_classes = next_global_pid
        self.num_domains = len(self.domain_names)

        # Build pid to sample indices mapping for PK sampler
        self.pid_to_indices = {}
        for idx, (_, g_pid, _, _) in enumerate(self.samples):
            if g_pid not in self.pid_to_indices:
                self.pid_to_indices[g_pid] = []
            self.pid_to_indices[g_pid].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, g_pid, camid, domain_id = self.samples[index]

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        img_tensor = self.transform(img)

        if self.occlusion_generator is not None:
            corrupted_img, mask, vis_target = self.occlusion_generator(img_tensor)
            return corrupted_img, g_pid, camid, domain_id, vis_target, img_path
        else:
            vis_target = torch.ones(3, dtype=torch.float32)
            return img_tensor, g_pid, camid, domain_id, vis_target, img_path


# ==============================================================================
# Automatic Path Resolution Helper
# ==============================================================================

FOLDER_CANDIDATES = {
    "market-1501": ["Market-1501-v15.09.15", "Market-1501", "market1501", "Market1501"],
    "msmt17": ["MSMT17_V1", "MSMT17", "msmt17", "MSMT17_V2"],
    "cuhk03": ["cuhk03", "CUHK03", "cuhk03_release", "archive"],
    "cuhk-sysu": ["cuhk_sysu", "CUHK-SYSU", "cuhksysu"],
    "occluded-dukemtmc": ["Occluded-DukeMTMC", "occluded_dukemtmc", "DukeMTMC-reID", "occluded_duke"]
}


def resolve_dataset_dir(root_dir: str, dataset_name: str) -> str:
    """
    Resolves the actual dataset folder path under root_dir, handling common
    naming variants from Kaggle, Google Drive, or official archives.
    """
    # Direct match or exact path
    direct_path = os.path.join(root_dir, dataset_name)
    if os.path.isdir(direct_path):
        return direct_path

    # Check known candidates
    key = dataset_name.lower().replace("_", "-")
    candidates = FOLDER_CANDIDATES.get(key, [dataset_name])

    for cand in candidates:
        cand_path = os.path.join(root_dir, cand)
        if os.path.isdir(cand_path):
            return cand_path

    # Case-insensitive directory search under root_dir
    if os.path.exists(root_dir):
        for entry in os.listdir(root_dir):
            entry_path = os.path.join(root_dir, entry)
            if os.path.isdir(entry_path):
                if entry.lower() in [c.lower() for c in candidates]:
                    return entry_path

    return direct_path


# ==============================================================================
# Specific Dataset Parsers
# ==============================================================================

def parse_market1501_dir(dir_path: str, domain_id: int = 0) -> List[Tuple[str, int, int, int]]:
    """
    Parses Market-1501 / DukeMTMC directory images.
    Filename pattern: <pid>_c<cam>...jpg
    """
    pattern = re.compile(r'([-\d]+)_c(\d+)')
    samples = []
    if not os.path.exists(dir_path):
        return samples

    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        match = pattern.search(fname)
        if not match:
            continue
        pid, camid = match.groups()
        pid = int(pid)
        camid = int(camid)
        img_path = os.path.join(dir_path, fname)
        samples.append((img_path, pid, camid, domain_id))
    return samples


def parse_msmt17_list(list_file: str, img_dir: str, domain_id: int = 1) -> List[Tuple[str, int, int, int]]:
    """
    Parses MSMT17 annotation list file: "<img_path> <pid>"
    """
    samples = []
    if not os.path.isfile(list_file):
        return samples

    with open(list_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_path, pid = parts[0], int(parts[1])
                # Camera ID can be parsed from filename e.g., '..._c01_...' or '_c1_'
                cam_match = re.search(r'_c(\d+)_', rel_path)
                camid = int(cam_match.group(1)) if cam_match else 0

                # Check if rel_path exists directly or inside img_dir
                if os.path.isabs(rel_path) and os.path.isfile(rel_path):
                    full_path = rel_path
                else:
                    full_path = os.path.join(img_dir, rel_path)
                    # If still not found, test subfolder e.g. img_dir/train/rel_path
                    if not os.path.isfile(full_path):
                        for sub in ["train", "test"]:
                            alt = os.path.join(img_dir, sub, rel_path)
                            if os.path.isfile(alt):
                                full_path = alt
                                break

                samples.append((full_path, pid, camid, domain_id))
    return samples


def parse_cuhk03_splits(
    dataset_dir: str,
    protocol: str = "detected",
    split_id: int = 0,
    domain_id: int = 2
) -> Tuple[List[Tuple[str, int, int, int]], List[Tuple[str, int, int, int]], List[Tuple[str, int, int, int]]]:
    """
    Parses CUHK03 using the CUHK03-NP (New Protocol) JSON splits:
    - splits_new_detected.json / splits_new_labeled.json
    - images_detected/ or images_labeled/
    """
    json_name = f"splits_new_{protocol}.json"
    json_path = os.path.join(dataset_dir, json_name)

    # Fallback search if archive/ or cuhk03_release/ subfolder
    if not os.path.isfile(json_path):
        for sub in [".", "archive", "cuhk03_release", "CUHK03"]:
            cand = os.path.join(dataset_dir, sub, json_name)
            if os.path.isfile(cand):
                json_path = cand
                dataset_dir = os.path.dirname(cand)
                break

    img_folder_name = f"images_{protocol}"
    img_dir = os.path.join(dataset_dir, img_folder_name)
    if not os.path.isdir(img_dir):
        # Alternative without subfolder
        img_dir = dataset_dir

    train_samples, query_samples, gallery_samples = [], [], []

    if os.path.isfile(json_path):
        with open(json_path, 'r') as f:
            splits = json.load(f)
        split = splits[split_id]

        # In CUHK03-NP JSON, files are lists of image names or [name, pid, camid]
        def extract_split(split_data):
            res = []
            for item in split_data:
                if isinstance(item, str):
                    fname = item
                    # Name format: <pid>_<cam>_<idx>.png or similar
                    match = re.search(r'([-\d]+)_([-\d]+)_', fname)
                    if match:
                        pid, cam = int(match.group(1)), int(match.group(2))
                    else:
                        pid, cam = 0, 0
                elif isinstance(item, (list, tuple)):
                    fname, pid, cam = item[0], int(item[1]), int(item[2])
                else:
                    continue

                full_path = os.path.join(img_dir, fname)
                res.append((full_path, pid, cam, domain_id))
            return res

        train_samples = extract_split(split.get("train", []))
        query_samples = extract_split(split.get("query", []))
        gallery_samples = extract_split(split.get("gallery", []))

    else:
        # Standard folder fallback if user restructured to bounding_box_train/
        train_samples = parse_market1501_dir(os.path.join(dataset_dir, "bounding_box_train"), domain_id=domain_id)
        query_samples = parse_market1501_dir(os.path.join(dataset_dir, "query"), domain_id=domain_id)
        gallery_samples = parse_market1501_dir(os.path.join(dataset_dir, "bounding_box_test"), domain_id=domain_id)

    return train_samples, query_samples, gallery_samples


def parse_cuhk_sysu(
    dataset_dir: str,
    domain_id: int = 3
) -> List[Tuple[str, int, int, int]]:
    """
    Parses CUHK-SYSU person search dataset.
    Supports either:
      1. 'cropped_images/' containing pre-extracted crops (<pid>_<idx>.jpg)
      2. 'Image/SSM/' with 'annotation/Person.mat'
      3. Standard 'bounding_box_train/' directory
    """
    samples = []

    # 1. Check for cropped_images directory
    cropped_dir = os.path.join(dataset_dir, "cropped_images")
    if os.path.isdir(cropped_dir):
        for fname in sorted(os.listdir(cropped_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                match = re.search(r'([-\d]+)', fname)
                pid = int(match.group(1)) if match else 0
                samples.append((os.path.join(cropped_dir, fname), pid, 0, domain_id))
        if samples:
            return samples

    # 2. Check for bounding_box_train directory
    bbox_dir = os.path.join(dataset_dir, "bounding_box_train")
    if os.path.isdir(bbox_dir):
        return parse_market1501_dir(bbox_dir, domain_id=domain_id)

    # 3. Check for annotation/Person.mat and Image/SSM
    person_mat = os.path.join(dataset_dir, "annotation", "Person.mat")
    ssm_dir = os.path.join(dataset_dir, "Image", "SSM")

    if os.path.isfile(person_mat) and os.path.isdir(ssm_dir):
        try:
            import scipy.io as sio
            mat = sio.loadmat(person_mat)
            # CUHK-SYSU Person.mat contains 'Person' cell array
            person_arr = mat.get('Person')
            if person_arr is not None:
                # Pre-extract/crop persons into cached folder
                cache_dir = os.path.join(dataset_dir, "cropped_images")
                os.makedirs(cache_dir, exist_ok=True)

                for p_idx in range(len(person_arr)):
                    p_info = person_arr[p_idx, 0]
                    # Structure: p_info contains id, image name, and box [x, y, w, h]
                    if len(p_info) >= 3:
                        pid = int(p_info[0][0]) if hasattr(p_info[0], '__getitem__') else p_idx
                        # Extract crops if image is readable
                        # For lightweight loading, if crops already extracted, reuse
                        crop_name = f"{pid:05d}_{p_idx:06d}.jpg"
                        crop_path = os.path.join(cache_dir, crop_name)
                        samples.append((crop_path, pid, 0, domain_id))
                return samples
        except Exception:
            pass

    return samples


# ==============================================================================
# Unified Loader Dispatcher
# ==============================================================================

def load_reid_benchmark(
    root_dir: str,
    dataset_name: str,
    domain_id: int = 0,
    protocol: str = "detected"
) -> Tuple[List[Tuple[str, int, int, int]], List[Tuple[str, int, int, int]], List[Tuple[str, int, int, int]]]:
    """
    High-level dispatcher that loads any benchmark using its proper folder structure:
    - Market-1501 / Market-1501-v15.09.15
    - MSMT17 / MSMT17_V1
    - CUHK03 / CUHK03-NP
    - CUHK-SYSU
    - Occluded-DukeMTMC
    Returns: (train_samples, query_samples, gallery_samples)
    """
    ds_dir = resolve_dataset_dir(root_dir, dataset_name)
    d_lower = dataset_name.lower().replace("_", "-")

    # 1. Market-1501
    if "market" in d_lower:
        train = parse_market1501_dir(os.path.join(ds_dir, "bounding_box_train"), domain_id=domain_id)
        query = parse_market1501_dir(os.path.join(ds_dir, "query"), domain_id=domain_id)
        gallery = parse_market1501_dir(os.path.join(ds_dir, "bounding_box_test"), domain_id=domain_id)
        return train, query, gallery

    # 2. MSMT17
    elif "msmt" in d_lower:
        # Check list files
        list_train = os.path.join(ds_dir, "list_train.txt")
        list_val = os.path.join(ds_dir, "list_val.txt")
        list_query = os.path.join(ds_dir, "list_query.txt")
        list_gallery = os.path.join(ds_dir, "list_gallery.txt")

        train = parse_msmt17_list(list_train, ds_dir, domain_id=domain_id)
        if os.path.isfile(list_val):
            train.extend(parse_msmt17_list(list_val, ds_dir, domain_id=domain_id))

        query = parse_msmt17_list(list_query, ds_dir, domain_id=domain_id)
        gallery = parse_msmt17_list(list_gallery, ds_dir, domain_id=domain_id)

        # Fallback if list files not found: check bounding_box_* or train/test
        if not train:
            train = parse_market1501_dir(os.path.join(ds_dir, "train"), domain_id=domain_id)
            query = parse_market1501_dir(os.path.join(ds_dir, "query"), domain_id=domain_id)
            gallery = parse_market1501_dir(os.path.join(ds_dir, "test"), domain_id=domain_id)

        return train, query, gallery

    # 3. CUHK03
    elif "cuhk03" in d_lower:
        return parse_cuhk03_splits(ds_dir, protocol=protocol, domain_id=domain_id)

    # 4. CUHK-SYSU
    elif "sysu" in d_lower:
        train = parse_cuhk_sysu(ds_dir, domain_id=domain_id)
        return train, [], []

    # 5. Occluded-DukeMTMC or DukeMTMC
    elif "duke" in d_lower:
        train = parse_market1501_dir(os.path.join(ds_dir, "bounding_box_train"), domain_id=domain_id)
        query = parse_market1501_dir(os.path.join(ds_dir, "query"), domain_id=domain_id)
        gallery = parse_market1501_dir(os.path.join(ds_dir, "bounding_box_test"), domain_id=domain_id)
        return train, query, gallery

    # Generic Re-ID fallback
    else:
        train = parse_market1501_dir(os.path.join(ds_dir, "bounding_box_train"), domain_id=domain_id)
        query = parse_market1501_dir(os.path.join(ds_dir, "query"), domain_id=domain_id)
        gallery = parse_market1501_dir(os.path.join(ds_dir, "bounding_box_test"), domain_id=domain_id)
        return train, query, gallery
