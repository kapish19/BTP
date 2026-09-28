"""
Solver package for DG-ReID.
"""

from .lr_scheduler import WarmupMultiStepLR
from .make_optimizer import make_optimizer, build_optimizer
from .make_optimizer_prompt import make_optimizer_prompt, make_optimizer_prompt_domain, make_optimizer_for_IE
from .scheduler_factory import create_scheduler
from .stage_manager import StageManager, get_current_stage

__all__ = [
    "make_optimizer",
    "build_optimizer",
    "make_optimizer_prompt",
    "make_optimizer_prompt_domain",
    "make_optimizer_for_IE",
    "create_scheduler",
    "WarmupMultiStepLR",
    "StageManager",
    "get_current_stage"
]
