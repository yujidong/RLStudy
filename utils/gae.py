"""Generalized Advantage Estimation (GAE) —— Ch08 的核心算法工具。

参考：Schulman et al. 2015, "High-Dimensional Continuous Control Using
Generalized Advantage Estimation"。

GAE 是 TD(λ) 思想（Ch04 §4.8）的 advantage 版本：
    Â_t^{GAE(γ,λ)} = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

其中 δ_{t+l} = R_{t+l+1} + γ V_φ(S_{t+l+1}) - V_φ(S_{t+l}) 是单步 TD error。
GAE 通过 λ 在 bias-variance 之间插值：
    λ = 0 → 一步 TD error（低方差、有偏）
    λ = 1 → MC advantage（无偏、高方差）

本模块提供：
- ``compute_td_errors``：从 (rewards, values, last_value, gamma, dones) 算单步 TD error
- ``compute_gae``：把 TD errors 衰减累加成 GAE
- ``compute_returns``：从 GAE 反推 critic 的 target return（G_t = Â_t + V_φ(s_t)）

所有函数同时接受 numpy 或 torch 输入（输出与输入类型一致）。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

Array = Union[np.ndarray, "torch.Tensor"]  # 后向兼容：torch 在签名中只作为字符串


def _is_tensor(x) -> bool:
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def compute_td_errors(
    rewards: Sequence[float],
    values: Array,
    last_value: float,
    gamma: float,
    dones: Optional[Sequence[bool]] = None,
) -> Array:
    """计算单步 TD error δ_t = R_{t+1} + γ V_φ(S_{t+1}) - V_φ(S_t)。

    对轨迹最后一步 t = T-1，用 last_value 作为 V_φ(S_T)（即 bootstrap value）。
    当 episode 自然终止时通常 last_value = 0；当截断（truncation）时应传入
    真实的 V_φ(S_T) 以避免 bootstrap bias。

    Parameters
    ----------
    rewards : shape [T]
    values : shape [T]       V_φ(S_0), V_φ(S_1), ..., V_φ(S_{T-1})
    last_value : float       V_φ(S_T) 的估计
    gamma : float            折扣因子
    dones : shape [T], optional
        每步是否终止。若 None，默认全 False（仅靠 last_value 处理边界）。

    Returns
    -------
    deltas : shape [T]       TD errors δ_0, δ_1, ..., δ_{T-1}
    """
    T = len(rewards)
    if dones is None:
        dones = [False] * T

    if _is_tensor(values):
        import torch
        deltas = torch.zeros(T, dtype=values.dtype, device=values.device)
        for t in range(T):
            v_next = last_value if t == T - 1 else values[t + 1]
            # 终止时 v_next = 0（absorbing state）
            if dones[t]:
                v_next = 0.0 if not isinstance(v_next, torch.Tensor) \
                    else torch.zeros_like(v_next)
            deltas[t] = rewards[t] + gamma * v_next - values[t]
        return deltas

    # numpy 分支
    values = np.asarray(values, dtype=np.float64)
    deltas = np.zeros(T, dtype=np.float64)
    for t in range(T):
        v_next = last_value if t == T - 1 else values[t + 1]
        if dones[t]:
            v_next = 0.0
        deltas[t] = rewards[t] + gamma * v_next - values[t]
    return deltas


def compute_gae(
    rewards: Sequence[float],
    values: Array,
    last_value: float,
    gamma: float,
    lam: float,
    dones: Optional[Sequence[bool]] = None,
) -> Array:
    """计算 GAE(γ, λ)：Â_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}。

    使用后向递归（O(T) 时间）：
        Â_{T-1} = δ_{T-1}
        Â_t = δ_t + γλ · Â_{t+1}    (t = T-2, ..., 0)

    这与 Ch04 §4.8.4 中"eligibility trace 后向视角"对单条轨迹的累积更新
    完全等价（见 §4.8.7 末尾"GAE = TD(λ) 思想的 advantage 版本"）。

    Parameters
    ----------
    rewards, values, last_value, gamma, dones
        见 ``compute_td_errors``。
    lam : float
        GAE 的 λ 参数 ∈ [0, 1]。

    Returns
    -------
    advantages : shape [T]
    """
    T = len(rewards)
    if dones is None:
        dones = [False] * T

    if _is_tensor(values):
        import torch
        advantages = torch.zeros(T, dtype=values.dtype, device=values.device)
        gae = torch.zeros((), dtype=values.dtype, device=values.device)
        for t in reversed(range(T)):
            v_next = last_value if t == T - 1 else values[t + 1]
            if dones[t]:
                # absorbing: 下一步 V 算 0；并重置 GAE accumulator
                v_next_zero = torch.zeros_like(gae)
                delta = rewards[t] + gamma * v_next_zero - values[t]
                gae = delta
            else:
                delta = rewards[t] + gamma * v_next - values[t]
                gae = delta + gamma * lam * gae
            advantages[t] = gae
        return advantages

    # numpy 分支
    values = np.asarray(values, dtype=np.float64)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        v_next = last_value if t == T - 1 else values[t + 1]
        if dones[t]:
            v_next = 0.0
            gae = 0.0  # 重置 trace
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


def compute_n_step_advantage(
    rewards: Sequence[float],
    values: Array,
    n: int,
    gamma: float,
    dones: Optional[Sequence[bool]] = None,
) -> Array:
    """n-step advantage Â_t^{(n)} = Σ_{k=0}^{n-1} γ^k δ_{t+k}（Ch04 §4.7.4 等价形式）。

    等价的另一种写法（Ch04 §4.7.1 的 n-step return 减 V(S_t)）：
        Â_t^{(n)} = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V_φ(S_{t+n}) - V_φ(S_t)

    对 t + n >= T 的部分用 0 填充（视作 absorbing）。

    Returns
    -------
    advantages : shape [T]
    """
    T = len(rewards)
    if dones is None:
        dones = [False] * T

    if _is_tensor(values):
        import torch
        adv = torch.zeros(T, dtype=values.dtype, device=values.device)
        for t in range(T):
            acc = torch.zeros((), dtype=values.dtype, device=values.device)
            discount = 1.0
            for k in range(n):
                if t + k >= T:
                    break
                v_next = values[t + k + 1] if t + k + 1 < T else 0.0
                if t + k + 1 >= T or dones[t + k]:
                    v_next = 0.0
                delta = rewards[t + k] + gamma * v_next - values[t + k]
                acc = acc + discount * delta
                discount *= gamma
                if dones[t + k]:
                    break
            adv[t] = acc
        return adv

    values = np.asarray(values, dtype=np.float64)
    adv = np.zeros(T, dtype=np.float64)
    for t in range(T):
        acc = 0.0
        discount = 1.0
        for k in range(n):
            if t + k >= T:
                break
            v_next = values[t + k + 1] if t + k + 1 < T else 0.0
            if t + k + 1 >= T or dones[t + k]:
                v_next = 0.0
            delta = rewards[t + k] + gamma * v_next - values[t + k]
            acc += discount * delta
            discount *= gamma
            if dones[t + k]:
                break
        adv[t] = acc
    return adv


def compute_returns_from_gae(
    advantages: Array, values: Array
) -> Array:
    """从 GAE 反推 critic target return：G_t = Â_t + V_φ(S_t)。

    这是 actor-critic 训练里最常用的 critic target（与 GAE 配对）。
    也可直接用 advantages.detach() + values.detach() 简化。
    """
    if _is_tensor(advantages):
        return advantages + values
    return np.asarray(advantages, dtype=np.float64) + np.asarray(values, dtype=np.float64)
