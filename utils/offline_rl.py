r"""Offline RL（CQL / IQL / Decision Transformer）—— Ch18 核心基础设施。

本章是 **Phase 4 第三章、整个 RLStudy 项目的终章**。Ch15 §15.6.3 列了 7 个开放
研究方向，本章展开其中第 4 条：

    **方向 4**：Offline RL / Decision Transformer——只用历史交互数据训策略，
    不再与环境交互。

--------------------------------------------------------------------
动机（§18.1）

在线 RL（Ch06 DQN / Ch09 PPO / Ch13 GRPO）都假设 agent 能反复 rollout：
   - Ch06：DQN 在线 ε-greedy 采，replay buffer 只是辅助
   - Ch09：PPO 在线 collect rollout 算 advantage
   - Ch13：GRPO 在线采 group 算 group baseline

但真实场景里 **很多任务不能反复试错**：
   - 医疗：不能反复"试"不同治疗方案
   - 自动驾驶：不能反复撞车
   - 工业控制：试错代价高
   - LLM 对齐：在线 rollout 也有成本（每条 trajectory 都要推理 + RM 评分）

**Offline RL** 的设定：只有一个预先收集的数据集 $\mathcal{D}$（由某些 behavior
policy $\pi_b$ 产生），训完策略直接部署，不再采数据。

--------------------------------------------------------------------
核心难点：distribution shift（§18.2）

数据集 $\mathcal{D}$ 只覆盖了 $\pi_b$ 访问过的 $(s, a)$。学到的 $\pi$ 如果去了
$\pi_b$ 没访问过的 (s, a)，Q 估计会**外推**（extrapolation error）。

具体来说：
- naive Q-learning 在 offline 数据上做 $\max_{a'} Q(s', a')$ 时，$\max$ 包括了
  $\pi_b$ 从没采过的 $a'$ —— Q 对这些 OOD action 的估计**不可靠**，且 DQN 的
  $\max$ 算子会系统性地**高估** OOD action 的 Q（因为 max 是上界）→ 灾难性失败。

三个解法方向（本章展开）：
- **CQL**（§18.3）：仍用 Q-learning + $\max$，但加正则把 OOD action 的 Q 压低
  —— "保守地"只信数据集内 (s, a) 的 Q
- **IQL**（§18.4）：完全不评估 OOD action 的 Q，只用 in-distribution 数据通过
  expectile regression 估 V，再用 $Q(s, a) \approx r + \gamma V(s')$
- **Decision Transformer**（§18.5）：彻底放弃 Q-learning，把 RL 重写成
  sequence modeling（return-conditioned 监督学习）

--------------------------------------------------------------------
模块组成

- :func:`collect_offline_dataset` —— 用给定 policy 采离线数据集
- :class:`OfflineDataset` —— 离线数据集容器（支持均匀采样、return-to-go 计算）
- :func:`offline_dqn_update_step` —— 标准 DQN 在 offline 数据上的单步更新
  （用于 §18.2 "naive offline DQN 会发散" 的对照实验）
- :class:`CQLTrainer` —— Conservative Q-Learning（Kumar et al. 2020）
- :class:`IQLTrainer` —— Implicit Q-Learning（Kostrikov et al. 2022）
- :class:`DecisionTransformer` —— Decision Transformer（Chen et al. 2021）

所有实现都是**教学版**（不追求 SOTA 速度），但数学严格、可观测、可对照实验。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import QNetwork, make_mlp
from .replay import ReplayBuffer
from .dqn_utils import (
    hard_update,
    polyak_update,
)
from .seed import get_rng


# =============================================================================
# 1. 数据收集 + OfflineDataset 容器
# =============================================================================
def collect_offline_dataset(
    env,
    policy: Callable[[np.ndarray], int],
    n_episodes: int,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> ReplayBuffer:
    r"""用给定 policy 在 env 上采 n_episodes 条轨迹，组装成 ReplayBuffer。

    这是 **offline RL 的数据准备 step**：之后所有训练（CQL / IQL / DT）都只读
    这个 buffer，不再调 ``env.step``。

    Parameters
    ----------
    env : CartPoleLite-like
        必须有 ``reset() -> s`` 和 ``step(a) -> (s', r, done, info)`` 接口。
    policy : callable s -> a
        Behavior policy $\pi_b$。可以是 ``random_policy``、固定启发式、
        或者用部分训过的 DQN。
    n_episodes : int
        采多少条 episode。
    seed : int, optional
        env 的随机种子。
    max_steps : int, optional
        覆盖 env 的默认 max_steps。

    Returns
    -------
    ReplayBuffer
        装满 transitions 的 buffer，容量 = sum(episode lengths)。
    """
    if seed is not None:
        env._rng = np.random.default_rng(seed)
    # 先跑一条探长度
    s = env.reset()
    done = False
    L = 0
    while not done:
        a = int(policy(s))
        s, r, done, _ = env.step(a)
        L += 1
    cap_guess = L * n_episodes
    buf = ReplayBuffer(capacity=cap_guess + 16, state_dim=env.observation_dim)
    # 正式采
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = int(policy(s))
            s_next, r, done, _ = env.step(a)
            buf.add(s, a, r, s_next, done)
            s = s_next
    return buf


class OfflineDataset:
    r"""离线数据集的轻量容器：均匀采样 + return-to-go 计算（DT 用）。

    与 :class:`utils.replay.ReplayBuffer` 的关系：
    - ``ReplayBuffer`` 只存 (s, a, r, s', done)，按 transition 采样
    - ``OfflineDataset`` 额外保留**轨迹边界**（episode 切分），方便算
      return-to-go $R_t = \sum_{t' \ge t} r_{t'}$

    内部存两个视图：
    - ``self.transitions``：flat (s, a, r, s', done)，CQL / IQL 用
    - ``self.episodes``：list of list，DT 用（按 episode 切，方便算 return-to-go）
    """

    def __init__(
        self,
        states: np.ndarray,           # [N, state_dim]
        actions: np.ndarray,          # [N]
        rewards: np.ndarray,          # [N]
        next_states: np.ndarray,      # [N, state_dim]
        dones: np.ndarray,            # [N]
        episode_boundaries: Sequence[Tuple[int, int]],  # [(start, end_exclusive), ...]
    ):
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.int64)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.next_states = np.asarray(next_states, dtype=np.float32)
        self.dones = np.asarray(dones, dtype=np.float32)
        self.episode_boundaries = list(episode_boundaries)
        self.n = self.states.shape[0]
        self.state_dim = self.states.shape[1]

    @classmethod
    def from_buffer(cls, buf: ReplayBuffer) -> "OfflineDataset":
        """从 :class:`ReplayBuffer` 构造（按 done 切 episode）。"""
        states = buf._states[: buf._size].copy()
        actions = buf._actions[: buf._size].copy()
        rewards = buf._rewards[: buf._size].copy()
        next_states = buf._next_states[: buf._size].copy()
        dones = buf._dones[: buf._size].copy()
        # 按 done=True 切 episode
        boundaries: List[Tuple[int, int]] = []
        start = 0
        for i in range(buf._size):
            if dones[i] >= 0.5:
                boundaries.append((start, i + 1))
                start = i + 1
        if start < buf._size:  # 最后一段没 done
            boundaries.append((start, buf._size))
        return cls(states, actions, rewards, next_states, dones, boundaries)

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None):
        """均匀随机采样 batch_size 条 transition。"""
        if rng is None:
            rng = get_rng()  # 共享 Generator：set_seed 可控，且避免每次采样新建 RNG
        idxs = rng.integers(0, self.n, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self) -> int:
        return self.n

    def return_to_go(self, gamma: float = 1.0) -> np.ndarray:
        r"""算每个 transition 的 return-to-go $R_t = \sum_{t' \ge t} \gamma^{t'-t} r_{t'}$。

        按 episode 内部倒序累加。DT 需要这个信号做 conditioning。
        """
        R = np.zeros(self.n, dtype=np.float32)
        for start, end in self.episode_boundaries:
            running = 0.0
            # 倒序累加
            for t in range(end - 1, start - 1, -1):
                running = self.rewards[t] + gamma * running
                R[t] = running
        return R

    def sample_episode(
        self, rng: Optional[np.random.Generator] = None
    ) -> Tuple[int, int]:
        """随机返回一条 episode 的 (start, end)。"""
        if rng is None:
            rng = get_rng()
        idx = rng.integers(0, len(self.episode_boundaries))
        return self.episode_boundaries[idx]


def random_policy_factory(n_actions: int, seed: int = 0):
    """构造一个均匀随机的 behavior policy。"""
    rng = np.random.default_rng(seed)

    def policy(state: np.ndarray) -> int:
        return int(rng.integers(n_actions))

    return policy


def heuristic_cartpole_policy(state: np.ndarray) -> int:
    """一个简单的启发式：根据杆子角度选 action（杆子往右倾就向右推）。

    比 pure random 强一点点，让数据集里有"撑住几步"的 trajectory。
    """
    # state = [x, x_dot, theta, theta_dot]
    theta, theta_dot = state[2], state[3]
    # 简单 PD 控制
    return int(0.5 * theta + 0.5 * theta_dot > 0)


# =============================================================================
# 2. Naive offline DQN（§18.2 对照实验：标准 DQN 直接喂 offline 数据）
# =============================================================================
def offline_dqn_update_step(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    gamma: float = 0.99,
) -> Dict[str, float]:
    """标准 DQN 单步更新，**忽略任何 offline 修正**（用于 §18.2 对照实验）。

    这就是 Ch06 的 ``dqn_update_step`` 的简化版，预期在纯 offline 数据上**发散**。
    """
    states, actions, rewards, next_states, dones = batch
    s = torch.as_tensor(states, dtype=torch.float32)
    a = torch.as_tensor(actions, dtype=torch.long)
    r = torch.as_tensor(rewards, dtype=torch.float32)
    s_next = torch.as_tensor(next_states, dtype=torch.float32)
    d = torch.as_tensor(dones, dtype=torch.float32)

    q_sa = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_snext_max = target_net(s_next).max(dim=1)[0]
        target = r + gamma * q_snext_max * (1.0 - d)
    loss = F.mse_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    with torch.no_grad():
        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "q_max": float(q_sa.max().item()),
            "td_error": float((q_sa - target).abs().mean().item()),
        }


# =============================================================================
# 3. CQL: Conservative Q-Learning（§18.3）
# =============================================================================
def cql_loss(
    q_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_net: nn.Module,
    gamma: float = 0.99,
    alpha: float = 5.0,
    n_actions: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""CQL 的单步 loss（Kumar et al. 2020, *Conservative Q-Learning for Offline
    Reinforcement Learning*）。

    总 loss:

    $$
    \mathcal{L}_{CQL}(\theta)
    = \underbrace{\mathcal{L}_{DQN}(\theta)}_{\text{Bellman}}
    + \alpha \underbrace{\left[
        \mathbb{E}_{s \sim \mathcal{D}}\!\left[\log \sum_a e^{Q(s, a)}\right]
        - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)]
      \right]}_{\text{CQL regularizer}}
    $$

    - $\mathcal{L}_{DQN}$：标准 Bellman 残差，$\max$ 包括 OOD action（**这就是
      offline Q-learning 的 extrapolation error 来源**）
    - CQL regularizer 第一项 $\log \sum_a e^{Q(s, a)}$ 是 **log-sum-exp**
      （soft-max），是 $\max_a Q(s, a)$ 的**光滑上界**——压低它就压低了所有 a 的
      Q 上界（特别是 OOD）
    - 第二项 $-\mathbb{E}[Q(s, a)]$ 让数据集内 (s, a) 的 Q **不被压低**
    - 组合起来：**OOD action 的 Q 被压低，in-distribution action 的 Q 维持高**
    - $\alpha$ 控制 conservative 程度（$\alpha \to 0$ 退化为 DQN）

    log-sum-exp 的性质（为什么用它代替 $\max$）：
    - $\max_a Q_a \le \log \sum_a e^{Q_a} \le \max_a Q_a + \log|A|$
    - 在 $|A|$ 小时（如 CartPole 的 2）两者差距 $\le \log 2 \approx 0.69$
    - log-sum-exp 对每个 $a$ 都有梯度（$\max$ 只对 argmax 有），训练更稳定

    Parameters
    ----------
    q_net : nn.Module
        在线 Q 网络 $Q(s, \cdot; \theta)$，输出 [batch, n_actions]
    states, actions, rewards, next_states, dones : Tensor
    target_net : nn.Module
        目标网络（Polyak 平均或 hard update 维护）
    gamma : float
    alpha : float
        CQL 强度系数。Kumar 2020 推荐 1.0-10.0（越大越保守）。
    n_actions : int, optional
        动作数。如不传，从 q_net 输出推断。

    Returns
    -------
    loss : scalar Tensor
    stats : dict
    """
    # ---- Bellman loss（同 DQN）----
    q_all = q_net(states)               # [B, n_actions]
    if n_actions is None:
        n_actions = q_all.size(-1)
    q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
    with torch.no_grad():
        q_snext_max = target_net(next_states).max(dim=1)[0]
        bellman_target = rewards + gamma * q_snext_max * (1.0 - dones)
    bellman_loss = F.mse_loss(q_sa, bellman_target)

    # ---- CQL regularizer ----
    # 第一项：log-sum-exp over actions（对 states 期望）
    #   log sum_a exp(Q(s, a))  shape [B]
    lse = torch.logsumexp(q_all, dim=1)            # [B]
    # 第二项：数据集内 (s, a) 的 Q
    q_data = q_sa                                   # [B]
    cql_reg = (lse - q_data).mean()                 # scalar

    loss = bellman_loss + alpha * cql_reg

    with torch.no_grad():
        stats = {
            "loss": float(loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "cql_reg": float(cql_reg.item()),
            "q_mean": float(q_sa.mean().item()),
            "q_max": float(q_sa.max().item()),
            "q_all_max": float(q_all.max().item()),  # 包括 OOD
            "lse": float(lse.mean().item()),
        }
    return loss, stats


class CQLTrainer:
    """CQL 训练循环：Q-learning + conservative regularizer。

    用法::

        q = QNetwork(state_dim, n_actions)
        tgt = QNetwork(state_dim, n_actions); hard_update(tgt, q)
        opt = torch.optim.Adam(q.parameters(), lr=1e-3)
        trainer = CQLTrainer(q, tgt, opt, dataset, n_actions=n_actions, alpha=5.0)
        for step in range(n_steps):
            stats = trainer.step()
        # 推理
        a = trainer.greedy_action(state)
    """

    def __init__(
        self,
        q_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: OfflineDataset,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 5.0,
        tau: float = 0.005,
        batch_size: int = 64,
        target_update_freq: int = 100,
        rng: Optional[np.random.Generator] = None,
    ):
        self.q_net = q_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.dataset = dataset
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = int(target_update_freq)
        self.rng = rng or np.random.default_rng()
        self.step_count = 0
        # 同步 target
        hard_update(self.target_net, self.q_net)

    def step(self) -> Dict[str, float]:
        batch = self.dataset.sample(self.batch_size, rng=self.rng)
        s, a, r, sn, d = [
            torch.as_tensor(x, dtype=torch.float32 if i != 1 else torch.long)
            for i, x in enumerate(batch)
        ]
        loss, stats = cql_loss(
            self.q_net, s, a, r, sn, d,
            target_net=self.target_net,
            gamma=self.gamma,
            alpha=self.alpha,
            n_actions=self.n_actions,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        # Polyak 更新 target
        polyak_update(self.target_net, self.q_net, tau=self.tau)
        self.step_count += 1
        return stats

    @torch.no_grad()
    def greedy_action(self, state: np.ndarray) -> int:
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q = self.q_net(s).squeeze(0)
        return int(q.argmax().item())


# =============================================================================
# 4. IQL: Implicit Q-Learning（§18.4）
# =============================================================================
def expectile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    expectile: float = 0.7,
) -> torch.Tensor:
    r"""Expectile regression loss（IQL 核心）。

    Expectile $\tau$ 是 minimize 下面这个 asymmetric squared loss 的解：

    $$
    \rho_\tau(u) = |\tau - \mathbb{1}(u < 0)| \cdot u^2
    $$

    - $\tau = 0.5$：退化为标准 MSE（解是 mean）
    - $\tau > 0.5$：对正残差（$u > 0$，即 target > pred）惩罚更重，解偏向 target 上尾
    - $\tau = 1$：解退化为 max

    IQL 用 $\tau = 0.7\sim 0.9$，让学到的 $V(s)$ 是 $\max_a Q(s, a)$ 的**乐观但
    不是 max** 的估计——既体现"在好 action 上 V 应该高"，又不像 $\max$ 那样
    对 OOD action 敏感。

    Loss to minimize:

    $$\mathcal{L}_V(V) = \mathbb{E}\left[ \rho_\tau(Q(s, a) - V(s)) \right]$$

    Parameters
    ----------
    pred : Tensor [B]   模型预测 $V(s)$
    target : Tensor [B] 监督信号 $Q(s, a)$（来自数据集的 a，不评估 OOD）
    expectile : float   $\tau \in (0, 1)$，常用 0.7-0.9

    Returns
    -------
    loss : scalar Tensor
    """
    diff = target - pred                      # u = target - pred
    weight = torch.where(diff > 0, expectile, (1.0 - expectile))
    return (weight * diff.pow(2)).mean()


class IQLTrainer:
    r"""IQL: Implicit Q-Learning（Kostrikov et al. 2022）。

    IQL 的核心思路：**完全不评估 OOD action 的 Q**。

    流程（§18.4 完整推导）：

    1. **V 网络**：用 expectile regression 从数据集的 (s, a) 学 $V(s)$，监督信号
       是 $Q_{\bar\theta}(s, a)$（**用数据集内的 a，不评估 OOD**）。$\tau > 0.5$
       让 $V(s)$ 偏向上尾（学到 $\max_a Q$ 的乐观近似）：

       $$\mathcal{L}_V(\psi) = \mathbb{E}_{(s, a) \sim \mathcal{D}}
         \left[\rho_\tau(Q_{\bar\theta}(s, a) - V_\psi(s))\right]$$

    2. **Q 网络**：用 Bellman target，但 target 里的 $\max_a$ 换成 $V(s')$：

       $$\mathcal{L}_Q(\theta) = \mathbb{E}\left[(r + \gamma V_\psi(s') - Q_\theta(s, a))^2\right]$$

       **关键**：target 里没有 $\max_a$ → 完全在分布内，不外推。

    3. **策略提取**（advantage-weighted regression）：

       $$\pi(a|s) \propto \exp(\beta (Q_\theta(s, a) - V_\psi(s)))$$

       即 advantage 大的 action 概率高。本章简化：直接用 $\arg\max_a Q$（演示用）。

    与 CQL 的对比：

    - **CQL**：仍然做 $\max_a Q(s', a')$（评估所有 OOD action），但加正则把 OOD
      Q 压低 → "保守地外推"
    - **IQL**：完全不评估 OOD action（用 V 替代 max）→ "不外推"

    Parameters
    ----------
    state_dim, n_actions : int
    hidden_dims : list of int
        V 和 Q 网络的隐藏层。
    gamma, expectile, beta : float
        gamma 折扣，expectile $\tau$（常用 0.7-0.9），beta 策略 softmax 温度倒数
    tau : float
        Polyak 系数（Q 和 V 的 target net 都用）
    lr : float
        学习率（V 和 Q 共用）
    batch_size : int
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        dataset: OfflineDataset,
        hidden_dims: Sequence[int] = (128, 128),
        gamma: float = 0.99,
        expectile: float = 0.7,
        beta: float = 3.0,
        tau: float = 0.005,
        lr: float = 1e-3,
        batch_size: int = 64,
        rng: Optional[np.random.Generator] = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.dataset = dataset
        self.gamma = gamma
        self.expectile = expectile
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.rng = rng or np.random.default_rng()

        # V 网络（标量输出）
        self.v_net = make_mlp(state_dim, 1, list(hidden_dims))
        # Q 网络（n_actions 输出）+ target Q 网络
        self.q_net = QNetwork(state_dim, n_actions, list(hidden_dims))
        self.q_target = QNetwork(state_dim, n_actions, list(hidden_dims))
        hard_update(self.q_target, self.q_net)

        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.step_count = 0

    def step(self) -> Dict[str, float]:
        batch = self.dataset.sample(self.batch_size, rng=self.rng)
        s, a, r, sn, d = batch
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.long)
        r = torch.as_tensor(r, dtype=torch.float32)
        sn = torch.as_tensor(sn, dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32)

        # 1) V update: minimize expectile_loss(V(s), Q_target(s, a_data))
        with torch.no_grad():
            q_target_sa = self.q_target(s).gather(1, a.unsqueeze(1)).squeeze(1)
        v_pred = self.v_net(s).squeeze(-1)
        v_loss = expectile_loss(v_pred, q_target_sa, self.expectile)
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # 2) Q update: minimize (r + gamma * V(s') - Q(s, a_data))^2  （V 来自 v_net，no max）
        with torch.no_grad():
            v_next = self.v_net(sn).squeeze(-1)
            q_target = r + self.gamma * (1.0 - d) * v_next
        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_sa, q_target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # Polyak 更新 q_target
        polyak_update(self.q_target, self.q_net, tau=self.tau)

        self.step_count += 1
        with torch.no_grad():
            adv = q_sa - v_pred
        return {
            "v_loss": float(v_loss.item()),
            "q_loss": float(q_loss.item()),
            "v_mean": float(v_pred.mean().item()),
            "q_mean": float(q_sa.mean().item()),
            "adv_mean": float(adv.mean().item()),
            "q_target_mean": float(q_target.mean().item()),
        }

    @torch.no_grad()
    def greedy_action(self, state: np.ndarray) -> int:
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q = self.q_net(s).squeeze(0)
        return int(q.argmax().item())

    @torch.no_grad()
    def softmax_action(
        self, state: np.ndarray, beta: Optional[float] = None
    ) -> int:
        r"""Advantage-weighted softmax 策略 $\pi(a|s) \propto \exp(\beta (Q - V))$。"""
        if beta is None:
            beta = self.beta
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q = self.q_net(s).squeeze(0)               # [n_actions]
        v = self.v_net(s).squeeze(-1)              # scalar
        logits = beta * (q - v)
        probs = F.softmax(logits, dim=0).cpu().numpy()
        return int(self.rng.choice(self.n_actions, p=probs))


# =============================================================================
# 5. Decision Transformer（§18.5）
# =============================================================================
class DecisionTransformer(nn.Module):
    r"""简化版 Decision Transformer（Chen et al. 2021）。

    把 RL 重写成 **return-conditioned sequence modeling**：

    - 输入序列：$(R_1, s_1, a_1, R_2, s_2, a_2, \dots)$，每个时间步是
      (return-to-go $R_t$, state $s_t$, action $a_t$) 三元组
    - 训练目标：给定 $(R_t, s_t)$ 预测 $a_t$（监督 cross-entropy / MSE）
    - 推理：给定目标 $R^*$，逐步生成 $a_t$；执行后观察 $r_{t+1}, s_{t+1}$，
      更新 $R_{t+1} = R_t - r_{t+1}$

    **关键洞察**：DT 学的是 "过去哪些 action 让 trajectory 达到了 return R"，
    不是 "如何最大化 return"。但**条件化在大 R 时**，生成的 action 倾向于
    "能达到大 R 的"，所以**经验上接近最优策略**——只要数据集里有高 return 的轨迹。

    与 Q-learning 的对比：

    - **Q-learning**（Ch06/CQL/IQL）：学 $Q(s, a)$，推理时 $\arg\max_a Q$
    - **DT**：学 $\pi(a | R, s)$，推理时给大 $R$，模型生成 $a$
    - DT **没有 Bellman equation**，没有 Q 值，没有外推问题
    - 代价：DT 需要**好数据集**（含高 return 轨迹），否则 "条件化在大 R" 时
      模型没见过这种 trajectory，会输出垃圾

    ----------------------------------------------------------------
    实现说明（教学简化）

    真实 DT 用 causal Transformer（GPT-style）。本章简化：
    - 输入：$(R_t, s_t)$ 拼接成一个向量（dim = 1 + state_dim）
    - 用一个小 MLP 预测 $a_t$（dim = n_actions logits）
    - 不用 attention，因为 CartPole 的"轨迹长度依赖"不强

    这样可以聚焦于 "return-conditioning" 的核心思想，不被 Transformer 细节淹没。
    §18.5.3 会讨论真实 DT 与本简化的区别。

    Parameters
    ----------
    state_dim : int
    n_actions : int
    hidden_dims : list of int
        MLP 隐藏层。
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        # 输入 = [R_t, s_t] 拼接
        in_dim = 1 + state_dim
        # 输出 = n_actions logits（分类）
        self.net = make_mlp(in_dim, n_actions, list(hidden_dims))

    def forward(self, return_to_go: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """前向：给定 (R_t, s_t)，预测 a_t 的 logits。

        Parameters
        ----------
        return_to_go : Tensor [B] 或 [B, T]    标量或序列
        state : Tensor [B, state_dim] 或 [B, T, state_dim]

        Returns
        -------
        logits : Tensor [B, n_actions] 或 [B, T, n_actions]
        """
        if return_to_go.dim() == 1:
            return_to_go = return_to_go.unsqueeze(-1)  # [B, 1]
        else:
            return_to_go = return_to_go.unsqueeze(-1)  # [B, T, 1]
        x = torch.cat([return_to_go, state], dim=-1)
        # 通用 flatten 再 forward（支持 2D 和 3D）
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            logits = self.net(x)
            logits = logits.reshape(B, T, -1)
        else:
            logits = self.net(x)
        return logits

    @torch.no_grad()
    def predict_action(
        self, return_to_go: float, state: np.ndarray, greedy: bool = True
    ) -> int:
        """推理时给一个 (R, s)，返回 action。"""
        rtg = torch.as_tensor([float(return_to_go)], dtype=torch.float32)
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(rtg, s).squeeze(0)
        if greedy:
            return int(logits.argmax().item())
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return int(np.random.choice(self.n_actions, p=probs))


def dt_loss(
    model: DecisionTransformer,
    rtg_batch: torch.Tensor,    # [B]
    state_batch: torch.Tensor,  # [B, state_dim]
    action_batch: torch.Tensor, # [B]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""DT 训练 loss：cross-entropy（or MSE for continuous）。

    $$\mathcal{L}_{DT} = -\mathbb{E}_{(R, s, a) \sim \mathcal{D}}[\log \pi_\theta(a | R, s)]$$
    """
    logits = model(rtg_batch, state_batch)  # [B, n_actions]
    loss = F.cross_entropy(logits, action_batch)
    with torch.no_grad():
        acc = float((logits.argmax(dim=-1) == action_batch).float().mean().item())
    return loss, {"loss": float(loss.item()), "action_acc": acc}


class DTTrainer:
    """Decision Transformer 训练循环。

    用法::

        dt = DecisionTransformer(state_dim, n_actions)
        opt = torch.optim.Adam(dt.parameters(), lr=1e-3)
        trainer = DTTrainer(dt, opt, dataset)
        for step in range(n_steps):
            stats = trainer.step()
    """

    def __init__(
        self,
        model: DecisionTransformer,
        optimizer: torch.optim.Optimizer,
        dataset: OfflineDataset,
        batch_size: int = 64,
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = rng or np.random.default_rng()
        self.step_count = 0

    def step(self) -> Dict[str, float]:
        # 采样 (s, a, ...)，算对应的 return-to-go
        s_np, a_np, _, _, _ = self.dataset.sample(self.batch_size, rng=self.rng)
        # 算全局 return-to-go（按 episode 切）
        # 为了效率，每次 step 不重算，缓存
        if not hasattr(self, "_rtg_cache"):
            self._rtg_cache = self.dataset.return_to_go(gamma=1.0)
        # 用同一组 idx 取 return-to-go（这里近似：直接随机采样 idx）
        idxs = self.rng.integers(0, self.dataset.n, size=self.batch_size)
        rtg_np = self._rtg_cache[idxs]

        rtg = torch.as_tensor(rtg_np, dtype=torch.float32)
        s = torch.as_tensor(s_np, dtype=torch.float32)
        a = torch.as_tensor(a_np, dtype=torch.long)
        loss, stats = dt_loss(self.model, rtg, s, a)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.step_count += 1
        return stats

    def reset_rtg_cache(self) -> None:
        if hasattr(self, "_rtg_cache"):
            del self._rtg_cache


@torch.no_grad()
def dt_rollout(
    model: DecisionTransformer,
    env,
    target_return: float,
    max_steps: int = 500,
    greedy: bool = True,
) -> Tuple[float, List[float]]:
    """用 DT 在 env 上跑一条 episode。

    流程：
    1. 给定 target_return R*
    2. 每步：模型预测 a_t = π(· | R, s_t)，执行，观察 r, s'
    3. 更新 R = R - r（return-to-go 递减）
    4. 重复直到 done 或 max_steps

    Returns
    -------
    total_reward : float
    rewards : list of per-step rewards
    """
    s = env.reset()
    R = float(target_return)
    total = 0.0
    rewards: List[float] = []
    for t in range(max_steps):
        a = model.predict_action(R, s, greedy=greedy)
        s_next, r, done, _ = env.step(a)
        rewards.append(float(r))
        total += float(r)
        R = R - float(r)  # return-to-go 递减
        s = s_next
        if done:
            break
    return total, rewards


# =============================================================================
# 6. 通用评估：用 greedy policy 在 env 上跑 N episodes
# =============================================================================
@torch.no_grad()
def evaluate_policy(
    policy_fn: Callable[[np.ndarray], int],
    env,
    n_episodes: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """用 policy_fn 在 env 上跑 n_episodes，返回 mean/std reward。"""
    if seed is not None:
        env._rng = np.random.default_rng(seed)
    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            a = int(policy_fn(s))
            s, r, done, _ = env.step(a)
            ep_reward += r
        rewards.append(ep_reward)
    arr = np.asarray(rewards, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "rewards": rewards,
    }


__all__ = [
    # dataset
    "collect_offline_dataset",
    "OfflineDataset",
    "random_policy_factory",
    "heuristic_cartpole_policy",
    # naive offline DQN
    "offline_dqn_update_step",
    # CQL
    "cql_loss",
    "CQLTrainer",
    # IQL
    "expectile_loss",
    "IQLTrainer",
    # DT
    "DecisionTransformer",
    "dt_loss",
    "DTTrainer",
    "dt_rollout",
    # 通用评估
    "evaluate_policy",
]
