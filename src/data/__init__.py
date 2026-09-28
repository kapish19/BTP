"""
DG-ReID Data Package
"""

from .transforms import build_transforms, Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor
from .occlusion import SyntheticOcclusionGenerator
from .dataset import (
    ReIDImageDataset,
    MultiDomainReIDDataset,
    resolve_dataset_dir,
    load_reid_benchmark,
    parse_market1501_dir,
    parse_msmt17_list,
    parse_cuhk03_splits,
    parse_cuhk_sysu
)
from .sampler import MultiDomainPKSampler
from .synthetic_data import generate_synthetic_reid_dataset

__all__ = [
    "build_transforms",
    "Compose",
    "Resize",
    "RandomHorizontalFlip",
    "Normalize",
    "ToTensor",
    "SyntheticOcclusionGenerator",
    "ReIDImageDataset",
    "MultiDomainReIDDataset",
    "resolve_dataset_dir",
    "load_reid_benchmark",
    "parse_market1501_dir",
    "parse_msmt17_list",
    "parse_cuhk03_splits",
    "parse_cuhk_sysu",
    "MultiDomainPKSampler",
    "generate_synthetic_reid_dataset"
]
