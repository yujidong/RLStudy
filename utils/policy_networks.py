"""策略网络（Ch07+ 用）。

Ch07 起从"学 Q 值"切到"直接参数化策略 π_θ(a|s)"。本模块提供：

- ``CategoricalPolicy``：离散动作的 softmax 策略（输入状态 → logits → Categorical 分布）
- ``GaussianPolicy``：连续动作的高斯策略（输入状态 → μ, σ → Normal 分布），
  Ch07 不直接用，但为 Ch08/Ch09 PPO 在连续控制上做铺垫。

设计原则（沿用 ``utils/networks.py``）：
- 教学优先——清晰胜过灵活
- 默认 ReLU 激活、Xavier 初始化（与 QNetwork 一致）
- 直接返回 ``torch.distributions`` 对象，方便调用者 ``.sample()`` / ``.log_prob()`` / ``.entropy()``

策略梯度定理（Ch07）需要的关键量：
    ∇_θ log π_θ(a|s)
在 PyTorch 里通过 ``dist.log_prob(action)`` + ``backward()`` 自动获得——
这背后就是 score function trick（``∇ log π = ∇π / π``）的自动微分。
"""
from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def _init_xavier(net: nn.Module) -> None:
    """对 net 内所有 Linear 层用 Xavier-uniform 初始化（与 networks.py 一致）。"""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


# =============================================================================
# 离散动作策略
# =============================================================================
class CategoricalPolicy(nn.Module):
    """离散动作的 softmax 策略网络。

    前向：``forward(s) -> logits``，shape ``[batch, n_actions]``。
    输出 ``Categorical`` 分布对象，可调用：
        - ``.sample()``            按 π_θ 采样动作
        - ``.log_prob(action)``    计算 log π_θ(a|s) ——策略梯度定理核心量
        - ``.entropy()``           计算 H(π_θ(·|s)) ——PPO 等用做 entropy bonus

    用法：
        >>> policy = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[128, 128])
        >>> s = torch.randn(32, 4)
        >>> dist = policy(s)               # 返回 Categorical 分布
        >>> a = dist.sample()              # [32] 个动作
        >>> logp = dist.log_prob(a)        # [32] 个 log π(a|s)

    注：输出的是 **logits**（未归一化的 log 概率），由 ``Categorical`` 内部自动 softmax。
    数值上比显式 softmax 更稳（避免大 exp 溢出）。
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = (128, 128),
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if isinstance(hidden_dims, tuple):
            hidden_dims = list(hidden_dims)

        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        # 最后一层输出 logits（不加激活，Categorical 内部 softmax）
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

        _init_xavier(self.net)
        self.n_actions = n_actions

    def forward(self, s: torch.Tensor) -> Categorical:
        """s shape [batch, state_dim] → Categorical 分布对象。

        Returns
        -------
        torch.distributions.Categorical
            可 ``.sample()`` / ``.log_prob()`` / ``.entropy()``。
        """
        logits = self.net(s)
        return Categorical(logits=logits)

    # 便利方法：直接返回 action 和 log_prob，方便训练循环调用
    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = False) -> int:
        """单步采样（不计算梯度）。

        Parameters
        ----------
        s : torch.Tensor, shape [state_dim] or [1, state_dim]
            单个状态。
        deterministic : bool
            True 表示贪心（argmax logits），False 表示按 π_θ 随机。

        Returns
        -------
        int
            动作 id。
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        logits = self.net(s)
        if deterministic:
            return int(logits.argmax(dim=1).item())
        dist = Categorical(logits=logits)
        return int(dist.sample().item())


# =============================================================================
# 连续动作策略（Ch08/Ch09 PPO 用，Ch07 仅介绍）
# =============================================================================
class GaussianPolicy(nn.Module):
    """连续动作的高斯策略：π_θ(a|s) = N(a; μ_θ(s), σ_θ²)。

    网络输出两层头：
    - ``mu_head``：线性，输出 μ ∈ R^action_dim
    - ``log_std``：可学习参数（与状态无关，最简形式；状态相关版本见 Ch08）

    用法：
        >>> policy = GaussianPolicy(state_dim=4, action_dim=2, hidden_dims=[128, 128])
        >>> s = torch.randn(32, 4)
        >>> dist = policy(s)
        >>> a = dist.sample()                # [32, 2]
        >>> logp = dist.log_prob(a).sum(-1)  # [32]，连续动作 log_prob 是逐维求和

    注：``log_std`` 设为可学习参数（state-independent）——
    这是 PPO 原文 (Schulman et al. 2017) 的简化做法。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = (128, 128),
        activation: Type[nn.Module] = nn.ReLU,
        init_log_std: float = 0.0,
        action_low: Optional[float] = None,
        action_high: Optional[float] = None,
    ):
        super().__init__()
        if isinstance(hidden_dims, tuple):
            hidden_dims = list(hidden_dims)

        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, action_dim)

        # log_std 作为可学习参数（与状态无关）
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

        _init_xavier(self.backbone)
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, s: torch.Tensor) -> Normal:
        """s shape [batch, state_dim] → Normal 分布对象。"""
        h = self.backbone(s)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        dist = self.forward(s)
        if deterministic:
            a = dist.mean
        else:
            a = dist.sample()
        if self.action_low is not None and self.action_high is not None:
            a = torch.clamp(a, self.action_low, self.action_high)
        return a.squeeze(0)


# =============================================================================
# Value network（Ch08 Actor-Critic 用）
# =============================================================================
class ValueNetwork(nn.Module):
    """状态 → V(s) 标量的最简 critic 网络。

    前向：``forward(s) -> v``，shape ``[batch, 1]`` 或 ``[batch]``（squeeze 后）。
    Ch07 笔记本里我们临时手写过一个 VNetwork，这里把它正式作为基础设施：
    Actor-Critic 需要 V_φ 做 critic（学 V^π），用 TD error 或回归学。

    用法：
        >>> v_net = ValueNetwork(state_dim=4, hidden_dims=[128, 128])
        >>> s = torch.randn(32, 4)
        >>> v = v_net(s)              # [32] 标量价值
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = (128, 128),
        activation: Type[nn.Module] = nn.ReLU,
        squeeze: bool = True,
    ):
        super().__init__()
        if isinstance(hidden_dims, tuple):
            hidden_dims = list(hidden_dims)

        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        _init_xavier(self.net)
        # 输出层用更小初始化，让初始 V ≈ 0（与 reward scale 对齐）
        last_linear = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(last_linear.bias, -3e-3, 3e-3)
        self.squeeze = squeeze

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s shape [batch, state_dim] → v shape [batch] 或 [batch, 1]。"""
        out = self.net(s)
        if self.squeeze:
            return out.squeeze(-1)
        return out


# =============================================================================
# 共享 backbone 的 Actor-Critic 网络（Ch08 A2C 用）
# =============================================================================
class ActorCritic(nn.Module):
    """共享 backbone 的 Actor-Critic 网络（A2C / PPO 通用）。

    架构：
        s → [共享 backbone] → features
                               ├── actor head  → logits（离散）或 μ（连续）
                               └── critic head → V(s)

    共享 backbone 让 actor 和 critic 共享底层特征提取（通常更样本高效），
    这是 A3C (Mnih 2016) 和 PPO (Schulman 2017) 默认做法。

    离散动作：返回 (Categorical 分布, value)
    连续动作：返回 (Normal 分布, value)

    用法（离散）：
        >>> ac = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[128, 128])
        >>> s = torch.randn(32, 4)
        >>> dist, v = ac(s)            # Categorical, [32]
        >>> a = dist.sample()
        >>> logp = dist.log_prob(a)

    Parameters
    ----------
    state_dim : int
    n_actions : int         离散动作数（设 action_dim=None 表示离散）
    action_dim : int, optional
        若不为 None，则建连续动作的 Gaussian head（n_actions 被忽略）
    hidden_dims : list of int
        共享 backbone 的隐藏层尺寸
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_dims: List[int] = (128, 128),
        activation: Type[nn.Module] = nn.ReLU,
        init_log_std: float = 0.0,
    ):
        super().__init__()
        if isinstance(hidden_dims, tuple):
            hidden_dims = list(hidden_dims)
        if action_dim is None and n_actions is None:
            raise ValueError("必须指定 n_actions（离散）或 action_dim（连续）")

        # 共享 backbone（最后一层 hidden_dims 即为 feature 维度）
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        self.backbone = nn.Sequential(*layers)

        # 两个 head
        self.continuous = action_dim is not None
        if self.continuous:
            self.actor_head = nn.Linear(prev, action_dim)
            self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
            self.action_dim = action_dim
        else:
            self.actor_head = nn.Linear(prev, n_actions)
            self.n_actions = n_actions
        self.critic_head = nn.Linear(prev, 1)

        # 初始化（Xavier），critic head 用小初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.critic_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.critic_head.bias, -3e-3, 3e-3)

    def forward(self, s: torch.Tensor):
        """s shape [batch, state_dim] → (dist, value)。

        Returns
        -------
        dist : torch.distributions.Distribution
            Categorical（离散）或 Normal（连续）
        value : torch.Tensor, shape [batch]
            V_φ(s) 标量
        """
        features = self.backbone(s)
        value = self.critic_head(features).squeeze(-1)  # [batch]
        if self.continuous:
            mu = self.actor_head(features)
            std = self.log_std.exp().expand_as(mu)
            from torch.distributions import Normal
            return Normal(mu, std), value
        else:
            from torch.distributions import Categorical
            logits = self.actor_head(features)
            return Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = False):
        """单步采样。返回 (action_int_or_tensor, log_prob, value)。"""
        if s.dim() == 1:
            s = s.unsqueeze(0)
        dist, value = self.forward(s)
        if deterministic:
            if self.continuous:
                a = dist.mean
            else:
                a = dist.probs.argmax()
        else:
            a = dist.sample()
        logp = dist.log_prob(a)
        if self.continuous:
            logp = logp.sum(-1)
        return a, logp, value.squeeze(0)
