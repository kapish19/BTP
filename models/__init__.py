"""
Models package for DG-ReID:
Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification.
"""

from .dgreid_model import DGReID, build_clip_visual
from .baseline import BaselineCLIP
from .grl import GradientReversalLayer, grad_reverse, compute_grl_alpha
from .domain_classifier import DomainClassifier
from .part_gat import PartVisibilityGAT
from .domain_factors import OrthogonalDomainFactors
from .vision_transformer import ViT
from .CLIP import Model as Clip

__all__ = [
    "DGReID",
    "BaselineCLIP",
    "GradientReversalLayer",
    "grad_reverse",
    "compute_grl_alpha",
    "DomainClassifier",
    "PartVisibilityGAT",
    "OrthogonalDomainFactors",
    "ViT",
    "Clip"
]