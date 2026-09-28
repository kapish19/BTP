"""
Checkpoint saving and resumption manager.
Saves model weights, optimizer/scheduler states, stage progress, and experiment metadata.
"""

import os
import shutil
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
    Uses shutil.copyfile to create checkpoint_best.pth efficiently without re-serializing.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Loads checkpoint state dictionary from disk.
    If strict is False, safely matches compatible tensor shapes and ignores
    mismatched classification/domain heads (essential for zero-shot evaluation).
    Returns metadata dictionary including epoch, stage, and best metrics.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        raw_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        raw_state_dict = checkpoint['state_dict']
    else:
        raw_state_dict = checkpoint

    if strict:
        model.load_state_dict(raw_state_dict, strict=True)
    else:
        model_state = model.state_dict()
        matched_dict = {}
        mismatched_keys = []

        for k, v in raw_state_dict.items():
            # Strip DDP / DataParallel 'module.' prefix if present
            clean_k = k[7:] if k.startswith("module.") else k
            if clean_k in model_state:
                if v.shape == model_state[clean_k].shape:
                    matched_dict[clean_k] = v
                else:
                    mismatched_keys.append((clean_k, tuple(v.shape), tuple(model_state[clean_k].shape)))

        model.load_state_dict(matched_dict, strict=False)
        if mismatched_keys:
            import logging
            logger = logging.getLogger("DGReID")
            msg = f"[Checkpoint] Loaded weights into model with {len(mismatched_keys)} shape-mismatched parameters skipped (e.g. classification/domain heads): " + \
                  ", ".join([f"{k} (ckpt: {s1} vs model: {s2})" for k, s1, s2 in mismatched_keys])
            logger.info(msg)

    # Load optimizer state if resuming
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            pass

    # Load scheduler state if resuming
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            pass

    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'stage': checkpoint.get('stage', 1),
        'best_rank1': checkpoint.get('best_rank1', 0.0),
        'best_map': checkpoint.get('best_map', 0.0),
        'config': checkpoint.get('config', {})
    }
    return metadata
