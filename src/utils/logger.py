"""
Logging utilities for training, evaluation, and experiment tracking.
Supports console output and file logging.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "DG-ReID",
    log_dir: Optional[str] = None,
    log_filename: str = "train.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger with consistent formatting across stdout and file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_filename)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
