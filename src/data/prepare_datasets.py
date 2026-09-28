"""
Dataset preparation, organization, and validation utilities for:
- Market-1501 (Market-1501-v15.09.15 on Kaggle)
- MSMT17 (MSMT17_V1 on Kaggle)
- CUHK-SYSU (cuhk_sysu on Kaggle)
- CUHK03 (cuhk03 with CUHK03-NP protocol files)
- Occluded-DukeMTMC (from lightas/Occluded-DukeMTMC-Dataset)

Includes exact directory structure validation matching Kaggle and GitHub sources.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.data.dataset import resolve_dataset_dir
else:
    from .dataset import resolve_dataset_dir


DATASET_INFO = {
    "Market-1501": {
        "source": "Kaggle (https://www.kaggle.com/datasets/pengcw1/market-1501/data)",
        "folder_names": ["Market-1501-v15.09.15", "Market-1501"],
        "expected_items": ["bounding_box_train", "bounding_box_test", "query"],
        "num_train_pids": 751,
        "num_test_pids": 750,
        "num_cameras": 6,
        "tree": """
Market-1501-v15.09.15/
├── bounding_box_test/
├── bounding_box_train/
├── gt_bbox/
├── gt_query/
├── query/
└── readme.txt
        """
    },
    "MSMT17": {
        "source": "Kaggle (https://www.kaggle.com/datasets/ouassimaazzouzi/msmt17)",
        "folder_names": ["MSMT17_V1", "MSMT17"],
        "expected_items": ["train", "test", "list_train.txt", "list_val.txt", "list_query.txt", "list_gallery.txt"],
        "num_train_pids": 1041,
        "num_test_pids": 3060,
        "num_cameras": 15,
        "tree": """
MSMT17_V1/
├── test/
├── train/
├── list_gallery.txt
├── list_query.txt
├── list_train.txt
└── list_val.txt
        """
    },
    "CUHK-SYSU": {
        "source": "Kaggle (https://www.kaggle.com/datasets/manaschaiaonon/cuhk-sysu)",
        "folder_names": ["cuhk_sysu", "CUHK-SYSU"],
        "expected_items": ["Image", "annotation"],
        "num_train_pids": 5532,
        "num_test_pids": 2900,
        "num_cameras": 0,
        "tree": """
cuhk_sysu/
├── Image/
│   └── SSM/
├── annotation/
│   ├── test/
│   ├── Images.mat
│   ├── Person.mat
│   └── pool.mat
└── README.txt
        """
    },
    "CUHK03": {
        "source": "Kaggle (https://www.kaggle.com/datasets/priyanagda/cuhk03) + CUHK03-NP (https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)",
        "folder_names": ["cuhk03", "archive", "cuhk03_release"],
        "expected_items": ["splits_new_detected.json", "images_detected"],
        "num_train_pids": 767,
        "num_test_pids": 700,
        "num_cameras": 2,
        "tree": """
cuhk03/ (or archive/)
├── cuhk03_release/
│   ├── README.md
│   └── cuhk-03.mat
├── images_detected/
├── images_labeled/
├── splits_new_detected.json
├── splits_new_labeled.json
└── pairs.csv
        """
    },
    "Occluded-DukeMTMC": {
        "source": "GitHub (https://github.com/lightas/Occluded-DukeMTMC-Dataset) converted from DukeMTMC-reID",
        "folder_names": ["Occluded-DukeMTMC", "occluded_dukemtmc"],
        "expected_items": ["bounding_box_train", "bounding_box_test", "query"],
        "num_train_pids": 702,
        "num_test_pids": 1110,
        "num_cameras": 8,
        "tree": """
Occluded-DukeMTMC/
├── bounding_box_test/
├── bounding_box_train/
└── query/
        """
    }
}


def verify_dataset_structure(dataset_name: str, root_dir: str) -> bool:
    """
    Validates whether the expected dataset folder structure exists on disk.
    Automatically resolves alternate folder names (e.g. Market-1501-v15.09.15, MSMT17_V1, cuhk_sysu).
    """
    resolved_dir = resolve_dataset_dir(root_dir, dataset_name)
    key = None
    for k in DATASET_INFO.keys():
        if k.lower().replace("_", "-") == dataset_name.lower().replace("_", "-"):
            key = k
            break

    if key is None:
        print(f"[-] Unknown dataset key: {dataset_name}")
        return False

    info = DATASET_INFO[key]
    print(f"\n[+] Verifying {key} (Resolved Path: '{resolved_dir}')...")

    if not os.path.exists(resolved_dir):
        print(f"    [X] Directory not found: {resolved_dir}")
        print(f"    Expected candidate folder names: {info['folder_names']}")
        return False

    all_exist = True
    for item in info["expected_items"]:
        target_path = os.path.join(resolved_dir, item)
        if not os.path.exists(target_path):
            print(f"    [X] Missing expected item: {item}")
            all_exist = False
        else:
            print(f"    [OK] Found: {item}")

    if all_exist:
        print(f"[+] {key} structure verified successfully! Ready for training and evaluation.")
    else:
        print(f"[-] {key} incomplete. Refer to source: {info['source']}")
    return all_exist


def print_download_instructions():
    """
    Displays official access and download instructions for all benchmarks mentioned in the report.
    """
    print("=" * 80)
    print("BENCHMARK DATASET SOURCES & DIRECTORY SPECIFICATIONS")
    print("=" * 80)
    for name, info in DATASET_INFO.items():
        print(f"\nDataset: {name}")
        print(f"Source URL: {info['source']}")
        print(f"Expected Directory Structure:")
        print(info['tree'].strip())
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation & verification")
    parser.add_argument("--info", action="store_true", help="Print download instructions and folder trees")
    parser.add_argument("--verify", type=str, choices=list(DATASET_INFO.keys()), help="Dataset name to verify")
    parser.add_argument("--dir", type=str, default="./data", help="Directory where dataset is located")
    args = parser.parse_args()

    if args.info or len(sys.argv) == 1:
        print_download_instructions()
    if args.verify:
        verify_dataset_structure(args.verify, args.dir)
