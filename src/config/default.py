import importlib
import os
from typing import Any, Optional

from yacs.config import CfgNode as CN

_C = CN()


def get_cfg(config_file: Optional[str] = None, new_allowed: bool = True) -> CN:
    cfg = _C.clone()
    cfg.set_new_allowed(new_allowed)
    if config_file:
        cfg.merge_from_file(config_file)
    return cfg


def merge_cfg(cfg_1: CN, cfg_2: CN) -> CN:
    cfg_1 = cfg_1.clone()
    cfg_1.merge_from_other_cfg(cfg_2)
    return cfg_1


def find_config_path(filename: str) -> str:
    """Tries to find the config file in . or src/config.

    Args:
        filename: Config file name or path.

    Returns:
        Absolute path to config.
    """
    if os.path.isfile(filename):
        return filename
    # Try in src/config/
    try_path = os.path.join("src", "config", filename)
    if os.path.isfile(try_path):
        return try_path
    raise FileNotFoundError(f"Could not find config file: {filename}")


def import_class_from_path(path: str) -> Any:
    module = importlib.import_module(".".join(path.split(".")[:-1]))
    return getattr(module, path.split(".")[-1])
