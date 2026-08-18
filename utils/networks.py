"""通用神经网络构造器（Ch06+ 用）。

Ch06 起从 numpy 切到 PyTorch。本模块提供项目所有章节通用的网络构造工具：
- ``make_mlp``：构造一个标准 MLP（多层感知机）
- ``QNetwork``：状态 → Q 值（每个动作一个输出）
- ``DuelingQNetwork``：状态 → V + A，组合出 Q（Ch06.8 用）

设计原则：
- 教学优先——清晰胜过灵活。隐藏不必要的选项。
- 默认用 ReLU 激活（Deep RL 主流选择）
- 默认用 Xavier 初始化（避免初始 Q 值过大/过小）
"""
from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[Type[nn.Module]] = None,
) -> nn.Sequential:
    """构造一个 MLP（多层感知机）。

    Parameters
    ----------
    input_dim : int
        输入维度（如 CartPoleLite 的 4）。
    output_dim : int
        输出维度（如 DQN 的动作数）。
    hidden_dims : list of int
        每个隐藏层的神经元数，例如 [128, 128]。
    activation : nn.Module class
        隐藏层激活函数，默认 ReLU。
    output_activation : nn.Module class, optional
        输出层激活函数。None 表示线性输出（Q 值不限制范围）。

    Returns
    -------
    nn.Sequential
        一个可训练的 MLP。

    Example
    -------
    >>> net = make_mlp(4, 2, [128, 128])
    >>> s = torch.randn(32, 4)  # batch=32
    >>> q = net(s)
    >>> q.shape
    torch.Size([32, 2])
    """
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation())

    net = nn.Sequential(*layers)
    _init_xavier(net)
    return net


def _init_xavier(net: nn.Module) -> None:
    """对 net 内所有 Linear 层用 Xavier-uniform 初始化。"""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class QNetwork(nn.Module):
    """状态 → Q 值（每个动作一个输出）的最简实现。

    前向：``forward(s) -> q_values``，shape ``[batch, n_actions]``。
    用法：
        >>> q = QNetwork(state_dim=4, n_actions=2, hidden_dims=[128, 128])
        >>> s = torch.randn(32, 4)
        >>> q(s).shape  # [32, 2]
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = (128, 128),
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        # tuple -> list 兼容
        if isinstance(hidden_dims, tuple):
            hidden_dims = list(hidden_dims)
        self.net = make_mlp(state_dim, n_actions, hidden_dims, activation=activation)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s shape [batch, state_dim] → q shape [batch, n_actions]。"""
        return self.net(s)


class DuelingQNetwork(nn.Module):
    """Dueling DQN 架构（Ch06.8）：Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)。

    把 Q 拆成"状态价值 V" + "动作优势 A"两个分支，让网络更容易学到
    "在某些状态下动作选择不重要"（如杆子很稳时左/右差别不大）。

    公式（van Hasselt et al. 2016）：
        Q(s, a) = V(s) + A(s, a) - (1/|A|) * sum_a' A(s, a')

    减去 A 的均值的目的是"可识别性"（identifiability）——
    否则 V 和 A 可以任意平移而不改变 Q（V+c, A-c 给出同样 Q）。
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
        # 共享 backbone
        backbone: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims[:-1]:
            backbone.append(nn.Linear(prev, h))
            backbone.append(activation())
            prev = h
        # 如果只有一层 hidden，则 backbone 为空，prev = state_dim
        # 最后一层 hidden 作为 V 和 A 分支的输入
        last_hidden = hidden_dims[-1] if hidden_dims else 64
        if hidden_dims:
            backbone.append(nn.Linear(prev, last_hidden))
            backbone.append(activation())
        self.backbone = nn.Sequential(*backbone)

        # V 分支：状态价值，1 维输出
        self.value_head = nn.Linear(last_hidden, 1)
        # A 分支：动作优势，n_actions 维输出
        self.advantage_head = nn.Linear(last_hidden, n_actions)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.n_actions = n_actions

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        features = self.backbone(s)
        v = self.value_head(features)            # [batch, 1]
        a = self.advantage_head(features)        # [batch, n_actions]
        # 减去 A 的均值保证可识别
        a_centered = a - a.mean(dim=1, keepdim=True)
        return v + a_centered                    # [batch, n_actions]
