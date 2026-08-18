"""PyTorch 相关的小工具：device 管理、梯度统计。
Phase 1 几乎用不到，留给 Phase 2+。"""
from __future__ import annotations

from typing import Dict


def get_device() -> str:
    """返回当前可用的最佳设备字符串（'cuda' 或 'cpu'）。"""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def grad_stats(model) -> Dict[str, float]:
    """返回模型梯度的统计信息：均值/标准差/全局范数。
    用于训练时监控梯度爆炸或消失。"""
    import torch

    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    if not grads:
        return {"mean": 0.0, "std": 0.0, "global_norm": 0.0, "n_params": 0}
    g = torch.cat(grads)
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "global_norm": float(g.norm(2)),
        "n_params": int(g.numel()),
    }


def count_parameters(model) -> int:
    """统计可训练参数数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
