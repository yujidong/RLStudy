"""ClickWorld —— Ch00 用的极简交互环境。

用户在 2D 网格里点击设置「目标」，看随机游走的智能体如何抵达。
主要用于让学生第一节课就直观看到 agent-environment loop。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class ClickWorld:
    """N×N 网格，用户可以点击设置目标和惩罚格。

    智能体（默认随机策略）每步朝目标方向移动一格（带噪声），
    抵达目标 +1、踩到惩罚格 -1、其他 0。

    用法::

        env = ClickWorld(8)
        env.set_goal((3, 5))
        env.set_penalty((2, 2))
        for _ in range(50):
            env.step_random()
        env.render()  # 在 notebook 内画图

    提供 ``step_random()``（无策略）与 ``step()``（给定动作）两种接口。
    """

    def __init__(self, size: int = 8, seed: Optional[int] = None):
        self.size = int(size)
        self.goal: Optional[Tuple[int, int]] = None
        self.penalties: set = set()
        self._rng = np.random.default_rng(seed)
        # 当前状态 = (row, col)
        self.state: Tuple[int, int] = (
            int(self._rng.integers(self.size)),
            int(self._rng.integers(self.size)),
        )
        self.t = 0
        self.reward_history: List[float] = []
        self.trajectory: List[Tuple[int, int]] = [self.state]

    # ------------------------------------------------------------------
    # 设置
    # ------------------------------------------------------------------
    def set_goal(self, pos: Tuple[int, int]) -> None:
        self.goal = (int(pos[0]), int(pos[1]))

    def set_penalty(self, pos: Tuple[int, int]) -> None:
        self.penalties.add((int(pos[0]), int(pos[1])))

    def clear_penalty(self, pos: Tuple[int, int]) -> None:
        self.penalties.discard((int(pos[0]), int(pos[1])))

    def reset(self) -> Tuple[int, int]:
        self.state = (
            int(self._rng.integers(self.size)),
            int(self._rng.integers(self.size)),
        )
        self.t = 0
        self.reward_history = []
        self.trajectory = [self.state]
        return self.state

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        dr, dc = self.ACTIONS[action]
        r, c = self.state
        nr = max(0, min(self.size - 1, r + dr))
        nc = max(0, min(self.size - 1, c + dc))
        self.state = (nr, nc)
        reward = 0.0
        done = False
        if self.goal is not None and self.state == self.goal:
            reward = 1.0
            done = True
        elif self.state in self.penalties:
            reward = -1.0
        self.t += 1
        self.reward_history.append(reward)
        self.trajectory.append(self.state)
        return self.state, reward, done, {}

    def step_random(self) -> Tuple[Tuple[int, int], float, bool, dict]:
        """完全随机的策略，用于演示。"""
        a = int(self._rng.integers(len(self.ACTIONS)))
        return self.step(a)

    # ------------------------------------------------------------------
    # 可视化（matplotlib）
    # ------------------------------------------------------------------
    def render(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        if ax is None:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.clear()
        # 格线
        for i in range(self.size + 1):
            ax.axhline(i, color="lightgray", linewidth=0.8)
            ax.axvline(i, color="lightgray", linewidth=0.8)
        # 惩罚
        for (r, c) in self.penalties:
            ax.add_patch(Rectangle((c, r), 1, 1, color="crimson", alpha=0.5))
        # 目标
        if self.goal is not None:
            r, c = self.goal
            ax.add_patch(Rectangle((c, r), 1, 1, color="gold", alpha=0.7))
            ax.text(c + 0.5, r + 0.5, "G", ha="center", va="center", fontsize=14, fontweight="bold")
        # 轨迹
        if len(self.trajectory) > 1:
            ys = [r + 0.5 for r, c in self.trajectory]
            xs = [c + 0.5 for r, c in self.trajectory]
            ax.plot(xs, ys, "-", color="steelblue", alpha=0.5, linewidth=1.2)
        # 当前位置
        r, c = self.state
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.3, color="navy"))
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {self.t}  R = {sum(self.reward_history):.2f}")
        return ax
