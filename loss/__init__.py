"""
Loss functions for DG-ReID.
"""

from .make_loss import TotalLoss, make_loss
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss

__all__ = [
    "TotalLoss",
    "make_loss",
    "CrossEntropyLabelSmooth",
    "LabelSmoothingCrossEntropy",
    "TripletLoss"
]