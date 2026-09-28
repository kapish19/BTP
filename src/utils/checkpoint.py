"""
Checkpoint saving and resumption manager.
Saves model weights, optimizer/scheduler states, stage progress, and experiment metadata.
"""

import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from .reproducibility import get_git_commit_hash


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint_latest.pth",
    is_best: bool = False
):
    """
    Saves checkpoint state dictionary to disk.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Loads checkpoint state dictionary from disk.
    Returns metadata dictionary including epoch, stage, and best metrics.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    # Load optimizer state if resuming
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if resuming
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'stage': checkpoint.get('stage', 1),
        'best_rank1': checkpoint.get('best_rank1', 0.0),
        'best_map': checkpoint.get('best_map', 0.0),
        'config': checkpoint.get('config', {})
    }
    return metadata
