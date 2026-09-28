"""
Person Re-Identification evaluation metrics: CMC (Rank-1, 5, 10, 20) and mAP.
Follows standard Market-1501 protocol:
- Gallery images from same PID and same Camera are ignored.
- Gallery images with PID = -1 (junk) are ignored.
"""

from typing import Dict, List, Tuple
import numpy as np


def eval_reid(
    dist_mat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50
) -> Tuple[np.ndarray, float]:
    """
    Evaluates CMC and mAP using standard Market-1501 / DukeMTMC evaluation protocol.

    Args:
        dist_mat: Distance matrix of shape (num_query, num_gallery)
        q_pids: Query person IDs (num_query,)
        g_pids: Gallery person IDs (num_gallery,)
        q_camids: Query camera IDs (num_query,)
        g_camids: Gallery camera IDs (num_gallery,)
        max_rank: Maximum rank for CMC curve
    Returns:
        cmc: Cumulative Matching Characteristic array of shape (max_rank,)
        mAP: Mean Average Precision scalar (float)
    """
    num_q, num_g = dist_mat.shape
    if num_g < max_rank:
        max_rank = num_g

    # Sort gallery by increasing distance for each query
    indices = np.argsort(dist_mat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_camids[q_idx]

        # Order of gallery samples
        order = indices[q_idx]

        # Find gallery samples to ignore: same PID and same Cam, or PID == -1
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
        junk = (g_pids[order] == -1)
        keep = ~(remove | junk)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            # No valid positive matches in gallery for this query
            continue

        num_valid_q += 1

        # 1. Compute AP
        pos_idx = np.where(raw_cmc == 1)[0]
        num_pos = len(pos_idx)
        # Precision at each retrieved positive rank
        precision_at_k = np.arange(1, num_pos + 1) / (pos_idx + 1.0)
        ap = np.mean(precision_at_k)
        all_ap.append(ap)

        # 2. Compute CMC
        cmc = np.zeros(max_rank, dtype=np.float32)
        # First correct match index
        first_match = pos_idx[0]
        if first_match < max_rank:
            cmc[first_match:] = 1.0
        all_cmc.append(cmc)

    if num_valid_q == 0:
        return np.zeros(max_rank, dtype=np.float32), 0.0

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.mean(axis=0)
    mAP = float(np.mean(all_ap))

    return cmc, mAP


def compute_metrics(
    dist_mat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray
) -> Dict[str, float]:
    """
    Computes Rank-1, Rank-5, Rank-10, Rank-20 and mAP.
    Returns metrics formatted as percentages (0 to 100).
    """
    cmc, mAP = eval_reid(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=20)
    return {
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[4] * 100.0) if len(cmc) >= 5 else 0.0,
        "Rank-10": float(cmc[9] * 100.0) if len(cmc) >= 10 else 0.0,
        "Rank-20": float(cmc[19] * 100.0) if len(cmc) >= 20 else 0.0,
        "mAP": float(mAP * 100.0)
    }
