"""
Re-ID Model Evaluator for clean and occlusion-robustness evaluation.
Extracts fused embeddings, computes distance matrices, and executes leave-one-domain-out evaluation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from ..data.occlusion import SyntheticOcclusionGenerator


class Evaluator:
    """
    Evaluator executing cross-domain retrieval and controlled synthetic target occlusion tests.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        distance_metric: str = "cosine"
    ):
        self.model = model
        self.device = device
        self.distance_metric = distance_metric

    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        occlusion_severity: str = "clean",
        occ_generator: Optional[SyntheticOcclusionGenerator] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract normalized features, PIDs, and camera IDs from dataloader.
        Optionally applies deterministic occlusion to inputs.
        """
        self.model.eval()
        features = []
        pids = []
        camids = []

        for batch in dataloader:
            imgs, batch_pids, batch_camids = batch[0], batch[1], batch[2]

            # Apply occlusion if requested
            if occlusion_severity != "clean" and occ_generator is not None:
                corrupted_batch = []
                for i in range(imgs.size(0)):
                    corr_img, _, _ = occ_generator.apply_controlled_occlusion(
                        imgs[i],
                        severity=occlusion_severity,
                        seed=42 + int(batch_pids[i])
                    )
                    corrupted_batch.append(corr_img)
                imgs = torch.stack(corrupted_batch, dim=0)

            imgs = imgs.to(self.device)
            # Model inference returns normalized feature f in R^{512}
            feats = self.model(imgs)
            features.append(feats.cpu().numpy())
            pids.extend(batch_pids.numpy() if isinstance(batch_pids, torch.Tensor) else batch_pids)
            camids.extend(batch_camids.numpy() if isinstance(batch_camids, torch.Tensor) else batch_camids)

        features = np.concatenate(features, axis=0)
        pids = np.array(pids)
        camids = np.array(camids)

        return features, pids, camids

    def compute_distance_matrix(self, q_feats: np.ndarray, g_feats: np.ndarray) -> np.ndarray:
        """
        Computes distance matrix between queries and gallery.
        """
        if self.distance_metric == "cosine":
            # Cosine distance: 1 - q * g^T (assuming L2 normalized)
            dist_mat = 1.0 - np.dot(q_feats, g_feats.T)
        else:
            # Euclidean distance
            m, n = q_feats.shape[0], g_feats.shape[0]
            dist_mat = np.power(q_feats, 2).sum(axis=1, keepdims=True).repeat(n, axis=1) + \
                       np.power(g_feats, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
            dist_mat -= 2 * np.dot(q_feats, g_feats.T)
            dist_mat = np.sqrt(np.maximum(dist_mat, 1e-12))

        return dist_mat

    def evaluate(
        self,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        occlusion_severities: List[str] = ["clean", "mild", "moderate", "severe"]
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs evaluation on clean and occluded query sets against clean gallery.
        Returns a dictionary mapping severity to metrics: {'clean': {'Rank-1': ..., 'mAP': ...}, ...}
        """
        occ_gen = SyntheticOcclusionGenerator(img_size=(256, 128))

        # Gallery is always clean
        g_feats, g_pids, g_camids = self.extract_features(gallery_loader, occlusion_severity="clean")

        results = {}
        for severity in occlusion_severities:
            q_feats, q_pids, q_camids = self.extract_features(
                query_loader,
                occlusion_severity=severity,
                occ_generator=occ_gen
            )
            dist_mat = self.compute_distance_matrix(q_feats, g_feats)
            metrics = compute_metrics(dist_mat, q_pids, g_pids, q_camids, g_camids)
            results[severity] = metrics

        return results
