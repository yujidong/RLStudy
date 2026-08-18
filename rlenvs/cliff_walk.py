"""CliffWalk —— Sutton-Barto 标准 4×12 悬崖网格，Ch05 必备。

复用 GridWorld 的实现，但提供一个更直白的 CliffWalk 子类，
额外跟踪每 episode 的「落崖次数」和「累计奖励」。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .grid_world import ACTIONS, GridWorld


class CliffWalk(GridWorld):
    """4×12 CliffWalk。

    - 起点在左下角 (3, 0)，终点在右下角 (3, 11)
    - 中间 row=3, col=1..10 是悬崖（-100）
    - 每步默认 -1 让智能体想尽快到达终点
    """

    def __init__(self, seed: Optional[int] = None):
        rows, cols = 4, 12
        # 悬崖的奖励字典
        cliff_cells = [(3, c) for c in range(1, 11)]
        rewards = {**{c: -100.0 for c in cliff_cells}, (3, 11): 0.0}
        super().__init__(
            shape=(rows, cols),
            terminals=[(3, 11)],
            slippery=(),
            rewards=rewards,
            default_reward=-1.0,
            start=(3, 0),
            seed=seed,
        )
        self.cliff_cells = cliff_cells
        # step() 每步都要判断是否落崖，预先展平成集合（避免热路径上反复重建）
        self._cliff_set = {self._flat(rc) for rc in cliff_cells}
        # 统计
        self.fell_in_cliff: int = 0
        self.episode_falls: list = []
        self.episode_return: float = 0.0
        self.episode_returns: list = []

    def reset(self) -> int:
        # step() 在 done=True 时已经推入了统计，reset 只清零累计
        self.episode_return = 0.0
        self.fell_in_cliff = 0
        return super().reset()

    def step(self, action: int):
        s_next, r, done, info = super().step(action)
        self.episode_return += r
        if s_next in self._cliff_set:
            self.fell_in_cliff += 1
        if done:
            self.episode_returns.append(self.episode_return)
            self.episode_falls.append(self.fell_in_cliff)
            self.episode_return = 0.0
            self.fell_in_cliff = 0
        return s_next, r, done, info

    def render(self, ax, state=None):
        if state is None:
            state = self.state
        ax.clear()
        for r in range(self.n_rows + 1):
            ax.axhline(r, color="black", linewidth=1.0)
        for c in range(self.n_cols + 1):
            ax.axvline(c, color="black", linewidth=1.0)
        # 悬崖
        from matplotlib.patches import Circle, Rectangle
        for (r, c) in self.cliff_cells:
            ax.add_patch(Rectangle((c, r), 1, 1, color="crimson", alpha=0.4))
        # 终点
        ax.add_patch(Rectangle((11, 3), 1, 1, color="gold", alpha=0.7))
        ax.text(11.5, 3.5, "G", ha="center", va="center", fontsize=12, fontweight="bold")
        # 起点
        ax.text(0.5, 3.5, "S", ha="center", va="center", fontsize=12, fontweight="bold", color="green")
        # 当前位置
        r, c = self._unflat(state)
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.3, color="navy"))
        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
