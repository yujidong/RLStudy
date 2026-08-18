"""PPO（Proximal Policy Optimization）核心更新逻辑 —— Ch09 的基础设施。

本模块把 PPO 的"一次更新"逻辑抽出来，作为 Ch09 笔记本和后续 Phase 3
（RLHF-PPO / GRPO 对比）的复用基础。和 ``utils/gae.py`` 一样遵循
"教学优先 + 实用"的设计原则。

参考：Schulman et al. 2017, "Proximal Policy Optimization Algorithms"。

PPO 相对于 A2C 的关键改动：
    1. **importance ratio** $r_t(\\theta) = \\pi_\\theta(a_t|s_t) / \\pi_{\\theta_{old}}(a_t|s_t)$
       —— 允许同一个 batch 用 K epochs 反复更新（多 epoch 数据重用）
    2. **clipped surrogate objective**
       $L^{CLIP} = \\mathbb{E}[\\min(r_t \\hat A_t, \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon) \\hat A_t)]$
       —— 把 TRPO 的硬 KL 约束换成软约束，防止策略一步走太远
    3. **KL early stopping**：每个 epoch 后估 KL(old || new)，超阈值就 break
    4. **advantage normalization**、**entropy bonus**、**value clipping**（可选）

本模块提供：
- ``compute_kl``：torch distributions 间的解析 KL（用于 early stopping 监控）
- ``approx_kl_from_ratio``：用 importance ratio 估计 KL（sample-based 无偏估计）
- ``compute_clip_objective``：把 PPO-Clip 目标单独暴露，方便在 notebook 里画图/分析
- ``ppo_update``：单次 PPO 多-epoch 更新（actor + critic + entropy + KL early stop）
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


# =============================================================================
# KL 散度估计
# =============================================================================
def compute_kl(old_dist, new_dist) -> torch.Tensor:
    """两个 torch.distributions 之间的 KL 散度（per-sample）。

    用 ``torch.distributions.kl_divergence``（已对 Categorical / Normal 实现
    解析形式）。返回 shape 与 batch 一致的 per-sample KL。
    """
    return torch.distributions.kl_divergence(old_dist, new_dist)


def approx_kl_from_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    """用 importance ratio 估计 KL：KL(old||new) ≈ mean((r - 1) - log r)。

    这是对 sample-based KL 的常用估计（Schulman 博客推荐形式），
    当 new 远离 old 时比解析 KL 更鲁棒（不需要在 new 下重新采样）。

    Parameters
    ----------
    log_ratio : Tensor, shape [batch]
        log π_new(a|s) - log π_old(a|s)
    """
    ratio = log_ratio.exp()
    return (ratio - 1.0) - log_ratio


# =============================================================================
# PPO-Clip surrogate objective（暴露出来方便画图/教学）
# =============================================================================
def compute_clip_objective(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    normalize_adv: bool = False,
) -> Dict[str, torch.Tensor]:
    """计算 PPO-Clip surrogate objective（per-sample + 标量 loss）。

    .. math::
        L^{CLIP}(\\theta) = \\mathbb{E}_t\\left[
            \\min\\big(r_t(\\theta) \\hat A_t,\\;
            \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat A_t\\big)
        \\right]

    Parameters
    ----------
    ratio : Tensor [batch]
        $r_t = \\pi_\\theta(a_t|s_t) / \\pi_{\\theta_{old}}(a_t|s_t)$
    advantages : Tensor [batch]
        $\\hat A_t$（GAE 估计）
    clip_eps : float
        clip 范围 $\\epsilon$（默认 0.2，PPO 原文推荐）。
    normalize_adv : bool
        是否对 advantage 做标准化（PPO 工程标配，但在 ``ppo_update``
        内部统一做；这里默认 False，便于教学画图）。

    Returns
    -------
    dict 含:
        - ``objective_per_sample`` [batch]：min(r A, clip(r) A)
        - ``loss`` []：-mean(objective_per_sample)（要 minimize）
        - ``clipped_mask`` [batch] float：哪些样本落到了 clip 区域
        - ``ratio_clipped`` [batch]：clip 后的 ratio
    """
    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    objective_per_sample = torch.min(ratio * advantages, ratio_clipped * advantages)
    clipped_mask = (ratio * advantages > ratio_clipped * advantages).float()

    return dict(
        objective_per_sample=objective_per_sample,
        loss=-objective_per_sample.mean(),
        clipped_mask=clipped_mask,
        ratio_clipped=ratio_clipped,
    )


# =============================================================================
# 一次完整的 PPO 更新（多 epoch × mini-batch）
# =============================================================================
def ppo_update(
    actor_critic,
    optimizer,
    traj: dict,
    gamma: float,
    lam: float,
    clip_eps: float = 0.2,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = 0.04,
    normalize_adv: bool = True,
    clip_value: bool = False,
    device: str = "cpu",
) -> Dict[str, float]:
    """PPO 的核心多-epoch 更新。

    流程（PPO 论文 Algorithm 1 的 PyTorch 实现）：
        1. 用旧策略采的 trajectory 算 GAE advantage（**一次性**，detach 后作为常量）
        2. 保存 old log π_old(a_t|s_t)（detach，作为 importance ratio 分母）
        3. for epoch in range(update_epochs):
              for each minibatch:
                  算 r_t = exp(log π_new - log π_old)
                  actor_loss = -PPO-Clip surrogate
                  critic_loss = (V_new - returns)^2 或 clipped value loss
                  entropy bonus
                  backward + grad clip + step
                  累计 KL
              若 epoch 内 mean KL > target_kl → early stop（保护策略）

    Parameters
    ----------
    actor_critic : ActorCritic
        共享 backbone 的 actor-critic 网络（见 ``utils.policy_networks``）。
        必须返回 (dist, value) 且支持 ``dist.log_prob(action)``。
    traj : dict
        收集到的 trajectory，需含 keys:
        ``states [N, state_dim]``、``actions [N]``、``rewards [N]``、
        ``dones [N]``、``values [N]``（旧 V 估计）、``log_probs [N]``（旧 log π）、
        ``last_value`` float（bootstrap）。
    gamma, lam : float
        GAE 参数。
    clip_eps : float
        PPO clip ε。
    update_epochs : int
        在同一批 on-policy 数据上反复多少 epochs（K，PPO 工程 magic）。
    minibatch_size : int
        每 epoch 切成 mini-batch 做 SGD（PPO 论文标准做法）。
    target_kl : float or None
        若不为 None，每个 epoch 后算 mean KL(old||new)，超阈值就 early stop
        （Schulman 推荐 target_kl ≈ 0.01 * d_action 或固定 0.04 for discrete）。
    clip_value : bool
        是否启用 value clipping（PPO 论文的可选项，通常收益不大）。

    Returns
    -------
    dict 含标量 metrics（每项都是所有 minibatch 的平均）：
        ``actor_loss / critic_loss / entropy / approx_kl / clip_fraction /
        grad_norm / n_epochs_done / early_stopped``。
    """
    from .gae import compute_gae

    states = torch.as_tensor(traj["states"], dtype=torch.float32, device=device)
    if torch.is_tensor(traj["actions"]):
        actions = traj["actions"].to(device).long()
    else:
        actions = torch.as_tensor(traj["actions"], dtype=torch.long, device=device)
    rewards = traj["rewards"]
    values_old_np = traj["values"]
    dones = traj["dones"]
    last_value = traj["last_value"]

    # 1. GAE advantage（一次性；多 epoch 都用同一个常量 advantage）
    advantages_np = compute_gae(
        rewards, values_old_np, last_value=last_value,
        gamma=gamma, lam=lam, dones=dones,
    )
    returns_np = advantages_np + np.asarray(values_old_np, dtype=np.float64)

    advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)
    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 2. old log π（importance ratio 分母，常量）
    if isinstance(traj["log_probs"], (list, tuple)):
        log_probs_old = torch.stack(traj["log_probs"]).to(device).float().detach()
    else:
        log_probs_old = torch.as_tensor(
            traj["log_probs"], dtype=torch.float32, device=device
        ).detach()
    values_old = torch.as_tensor(values_old_np, dtype=torch.float32, device=device)

    N = states.shape[0]
    mb_size = min(minibatch_size, N)

    # 累加每个 minibatch 的 metric，最后按总 minibatch 数平均
    acc = dict(
        actor_loss=0.0, critic_loss=0.0, entropy=0.0,
        approx_kl=0.0, clip_fraction=0.0, grad_norm=0.0,
    )
    n_minibatches_total = 0
    epochs_done = 0
    early_stopped = False

    for epoch in range(update_epochs):
        perm = torch.randperm(N, device=device)
        epoch_kl_sum = 0.0
        epoch_mb_count = 0
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            s_mb = states[idx]
            a_mb = actions[idx]
            adv_mb = advantages[idx]
            ret_mb = returns[idx]
            logp_old_mb = log_probs_old[idx]
            v_old_mb = values_old[idx]

            dist_new, values_new = actor_critic(s_mb)
            log_probs_new = dist_new.log_prob(a_mb)
            if log_probs_new.dim() > 1:  # 连续动作 sum over action dim
                log_probs_new = log_probs_new.sum(-1)
            entropy = dist_new.entropy()
            if entropy.dim() > 1:
                entropy = entropy.sum(-1)

            log_ratio = log_probs_new - logp_old_mb
            ratio = log_ratio.exp()

            clip_obj = compute_clip_objective(
                ratio, adv_mb, clip_eps=clip_eps, normalize_adv=False,
            )
            actor_loss = clip_obj["loss"]
            clip_frac = clip_obj["clipped_mask"].mean()

            if clip_value:
                v_clipped = v_old_mb + torch.clamp(
                    values_new - v_old_mb, -clip_eps, clip_eps
                )
                critic_loss = 0.5 * torch.max(
                    (values_new - ret_mb).pow(2),
                    (v_clipped - ret_mb).pow(2),
                ).mean()
            else:
                critic_loss = 0.5 * (values_new - ret_mb).pow(2).mean()

            entropy_loss = -entropy.mean()
            total_loss = (
                actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                actor_critic.parameters(), max_norm=max_grad_norm
            ).item()
            optimizer.step()

            with torch.no_grad():
                kl_mb = approx_kl_from_ratio(log_ratio).mean().item()

            acc["actor_loss"] += actor_loss.item()
            acc["critic_loss"] += critic_loss.item()
            acc["entropy"] += (-entropy_loss).item()
            acc["approx_kl"] += kl_mb
            acc["clip_fraction"] += clip_frac.item()
            acc["grad_norm"] += grad_norm
            epoch_kl_sum += kl_mb
            epoch_mb_count += 1
            n_minibatches_total += 1

        epochs_done = epoch + 1
        epoch_kl_mean = epoch_kl_sum / max(epoch_mb_count, 1)

        # KL early stopping
        if target_kl is not None and epoch_kl_mean > 1.5 * target_kl:
            early_stopped = True
            break

    norm = max(n_minibatches_total, 1)
    out = {k: v / norm for k, v in acc.items()}
    out["n_epochs_done"] = float(epochs_done)
    out["early_stopped"] = float(early_stopped)
    return out
