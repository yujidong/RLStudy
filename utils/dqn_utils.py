"""DQN 训练辅助工具：ε schedule、target network 更新、通用训练循环。

Ch06 笔记本里会反复用到这些工具——把它们封装到 utils 里，
让笔记本只展示算法主线，不被 boilerplate 淹没。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .seed import get_rng


# =============================================================================
# 1. ε schedule（Ch01 已用简单版，这里给更标准的衰减 schedule）
# =============================================================================

def linear_epsilon_schedule(
    step: int,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_steps: int = 1000,
) -> float:
    """线性衰减的 ε：从 eps_start 线性降到 eps_end，超过 decay_steps 后保持 eps_end。

    为什么用线性衰减：与指数衰减比，前期保持充分探索（eps_start 接近 1），
    后期平滑过渡到小 ε（让 greedy 主导但仍偶尔探索）。

    Parameters
    ----------
    step : int
        当前训练步数。
    eps_start : float
        初始 ε（如 1.0 = 完全随机）。
    eps_end : float
        终值 ε（如 0.05 = 95% greedy, 5% 随机）。
    decay_steps : int
        从 eps_start 衰减到 eps_end 所用的步数。

    Returns
    -------
    float
        当前 ε 值。
    """
    frac = min(1.0, step / decay_steps)
    return eps_start + (eps_end - eps_start) * frac


# =============================================================================
# 2. target network 更新
# =============================================================================

def hard_update(target_net: nn.Module, online_net: nn.Module) -> None:
    """硬拷贝：把 online_net 的参数直接复制到 target_net。

    Mnih et al. 2015 的 DQN 用这个：每隔 ``target_update_freq`` 步调一次。
    """
    target_net.load_state_dict(online_net.state_dict())


def polyak_update(target_net: nn.Module, online_net: nn.Module, tau: float = 0.005) -> None:
    """Polyak（软）更新：θ_target = τ * θ_online + (1-τ) * θ_target。

    每步调一次，让 target network "慢慢追" online net。
    常 τ=0.001~0.01。Polyak 比 hard update 更平滑，被 SAC、TD3 等现代算法采用。
    """
    with torch.no_grad():
        for tp, op in zip(target_net.parameters(), online_net.parameters()):
            tp.mul_(1 - tau).add_(op, alpha=tau)


# =============================================================================
# 3. ε-greedy 动作选择（PyTorch 版，Ch01 的 numpy 版基础上）
# =============================================================================

def epsilon_greedy_action(
    q_net: nn.Module,
    state: np.ndarray,
    epsilon: float,
    n_actions: int,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """ε-greedy 动作选择。PyTorch Q-network 输入 numpy state。

    Parameters
    ----------
    q_net : nn.Module
        用于评估当前 Q 值的网络（online net）。
    state : np.ndarray, shape [state_dim]
        当前状态。
    epsilon : float
        探索概率。
    n_actions : int
        动作数。
    rng : np.random.Generator, optional

    Returns
    -------
    int : 选中的动作。
    """
    if rng is None:
        rng = get_rng()  # 共享 Generator：set_seed 可控，且避免每个 env step 新建 RNG
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q = q_net(s).squeeze(0).cpu().numpy()
    return int(np.argmax(q))


# =============================================================================
# 4. DQN 训练步骤（单个 batch 更新）
# =============================================================================

def dqn_update_step(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    gamma: float,
    device: str = "cpu",
    use_double_dqn: bool = False,
) -> dict:
    """DQN 的单步梯度更新。在 Ch06 笔记里被反复调用。

    Parameters
    ----------
    online_net : Q 网络 Q(s, a; θ)
    target_net : 目标网络 Q(s, a; θ⁻)
    optimizer : PyTorch 优化器（如 Adam）
    batch : tuple of numpy arrays
        (states, actions, rewards, next_states, dones)，shape [batch, ...]
    gamma : float
        折扣因子。
    use_double_dqn : bool
        是否用 Double DQN（CH05 5.8.5）：用 online_net 选 a*，用 target_net 评估 Q(s', a*)。

    Returns
    -------
    dict : {"loss": float, "q_mean": float, "td_error_mean": float}
        监控指标。
    """
    states, actions, rewards, next_states, dones = batch
    s = torch.as_tensor(states, dtype=torch.float32, device=device)
    a = torch.as_tensor(actions, dtype=torch.long, device=device)
    r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    s_next = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    d = torch.as_tensor(dones, dtype=torch.float32, device=device)

    # 当前 Q 值 Q(s, a; θ)（gather 出选中动作的 Q）
    q_sa = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # target: R + γ * max_a' Q(s', a'; θ⁻) * (1 - done)
    with torch.no_grad():
        if use_double_dqn:
            # Double DQN: online 选 a*，target 评估
            a_star = online_net(s_next).argmax(dim=1, keepdim=True)
            q_snext_max = target_net(s_next).gather(1, a_star).squeeze(1)
        else:
            # 标准 DQN: target 自己选 max
            q_snext_max = target_net(s_next).max(dim=1)[0]
        target = r + gamma * q_snext_max * (1.0 - d)

    # MSE 损失（半梯度在 PyTorch 里自动——target 在 no_grad 内）
    loss = nn.functional.mse_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪（DQN 常用 trick，防止爆炸）
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    with torch.no_grad():
        td_error = (q_sa - target).abs().mean()

    return {
        "loss": float(loss.item()),
        "q_mean": float(q_sa.mean().item()),
        "td_error_mean": float(td_error.item()),
    }
