"""Deterministic seeding helpers (project rule §6).

中文说明
--------
在训练、仿真、评测入口调用 ``set_global_seed``，保证可复现；对仅跑 dashboard
的进程延迟 import numpy/torch 以降低冷启动依赖。
"""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python, hash, NumPy, and PyTorch RNGs for reproducibility.

    NumPy and PyTorch are imported lazily so that lightweight services
    (dashboard, orchestrator) do not have to install them.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep at runtime
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
