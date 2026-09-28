"""
Reproducibility utilities for deterministic experiments and run provenance.
"""

import os
import random
import subprocess
from typing import Any, Dict
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Sets random seed for Python, NumPy, and PyTorch across CPU and GPU/MPS.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_git_commit_hash() -> str:
    """
    Retrieves the current git commit hash if available.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown_or_uncommitted"


def get_environment_info() -> Dict[str, Any]:
    """
    Gathers environment provenance metadata for reproducibility logs.
    """
    info = {
        "git_commit": get_git_commit_hash(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU/MPS"
    }
    return info
