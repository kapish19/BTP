"""
DG-ReID Losses Package
"""

from .cross_entropy import LabelSmoothingCrossEntropy
from .triplet import BatchHardTripletLoss, euclidean_dist
from .visibility_loss import VisibilityLoss
from .orthogonality_loss import OrthogonalityLoss
from .total_loss import TotalLoss

__all__ = [
    "LabelSmoothingCrossEntropy",
    "BatchHardTripletLoss",
    "euclidean_dist",
    "VisibilityLoss",
    "OrthogonalityLoss",
    "TotalLoss"
]
