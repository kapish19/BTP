"""
Configuration loader and validator using YAML.
"""

from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads YAML configuration file into a Python dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_cfg: Dict[str, Any], override_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges override_cfg into base_cfg.
    """
    merged = base_cfg.copy()
    for key, value in override_cfg.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
