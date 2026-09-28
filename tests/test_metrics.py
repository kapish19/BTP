"""
Unit test for Re-ID evaluation metrics (CMC and mAP):
- Verifies perfect matching (100% Rank-1, 100% mAP).
- Verifies camera and junk exclusion rules.
"""

import unittest
import numpy as np
from src.evaluation.metrics import eval_reid, compute_metrics


class TestMetrics(unittest.TestCase):
    def test_perfect_retrieval(self):
        # 2 queries, 4 gallery items
        # Q0 (PID 1, Cam 1) -> G0 is PID 1, Cam 2 (perfect match)
        # Q1 (PID 2, Cam 1) -> G1 is PID 2, Cam 2 (perfect match)
        dist_mat = np.array([
            [0.1, 0.9, 0.8, 0.7],
            [0.8, 0.2, 0.9, 0.6]
        ])
        q_pids = np.array([1, 2])
        g_pids = np.array([1, 2, 3, 4])
        q_camids = np.array([1, 1])
        g_camids = np.array([2, 2, 3, 3])

        metrics = compute_metrics(dist_mat, q_pids, g_pids, q_camids, g_camids)
        self.assertAlmostEqual(metrics["Rank-1"], 100.0)
        self.assertAlmostEqual(metrics["mAP"], 100.0)

    def test_camera_exclusion(self):
        # Q0 matches G0 with distance 0.0, but G0 has same PID AND same Cam -> should be excluded!
        # G1 is correct PID with different Cam, distance 0.5.
        dist_mat = np.array([
            [0.0, 0.5, 0.9]
        ])
        q_pids = np.array([1])
        g_pids = np.array([1, 1, 2])
        q_camids = np.array([1])
        g_camids = np.array([1, 2, 3])  # G0 has cam 1 (same), G1 has cam 2 (different)

        metrics = compute_metrics(dist_mat, q_pids, g_pids, q_camids, g_camids)
        # G1 is top match after G0 exclusion
        self.assertAlmostEqual(metrics["Rank-1"], 100.0)
        self.assertAlmostEqual(metrics["mAP"], 100.0)


if __name__ == '__main__':
    unittest.main()
