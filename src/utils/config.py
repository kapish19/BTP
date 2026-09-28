import os
from typing import Any, Dict, Optional
import yaml


def find_default_config(config_path: str) -> Optional[str]:
    """
    Locates default.yaml either from the working directory or relative to config_path.
    """
    if os.path.isfile("configs/default.yaml"):
        return os.path.abspath("configs/default.yaml")
    dir_path = os.path.dirname(os.path.abspath(config_path))
    candidate = os.path.join(dir_path, "default.yaml")
    if os.path.isfile(candidate):
        return candidate
    candidate_parent = os.path.join(dir_path, "..", "default.yaml")
    if os.path.isfile(candidate_parent):
        return os.path.abspath(candidate_parent)
    return None


def load_config(config_path: str, auto_merge_default: bool = True) -> Dict[str, Any]:
    """
    Loads YAML configuration file into a Python dictionary.
    If auto_merge_default is True and config_path is not default.yaml,
    automatically merges on top of base default.yaml.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    if auto_merge_default:
        default_path = find_default_config(config_path)
        if default_path and os.path.abspath(config_path) != os.path.abspath(default_path):
            with open(default_path, 'r') as f:
                base_cfg = yaml.safe_load(f) or {}
            config = merge_configs(base_cfg, config)

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
