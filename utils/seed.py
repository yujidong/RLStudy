"""统一随机种子管理：保证实验可复现。"""
from __future__ import annotations

import os
import random

import numpy as np

# 模块级共享 Generator：utils 内所有"未显式传 rng"的随机调用统一走它，
# set_seed() 会重置它——否则每次 np.random.default_rng() 都从 OS 熵源新开
# 一条随机流，set_seed(42) 之后就再也复现不出来。
_rng = np.random.default_rng()


def set_seed(seed: int) -> None:
    """统一设置 numpy / random / os.environ 的种子。

    若 PyTorch 已安装，也会一并设置（CPU + CUDA）。
    用法::

        from utils import set_seed
        set_seed(42)
    """
    global _rng
    random.seed(seed)
    np.random.seed(seed)
    _rng = np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Phase 1 不强依赖 torch
        pass


def get_rng() -> np.random.Generator:
    """返回共享的 numpy Generator（被 :func:`set_seed` 控制）。

    适用于不想每次调用都新建 ``default_rng()`` 的热路径（replay buffer 采样、
    ε-greedy 等），同时保持可复现。
    """
    return _rng
