"""RandomWalk —— Sutton-Barto 19/5 状态随机游走，Ch04 必备。

经典演示：一条直线上的 N 个状态，左右两端是终止态。
- 从中间状态出发
- 每步 50% 概率往左、50% 往右
- 落到右端 +1、左端 0（或 -1，按书本不同版本）

Sutton-Barto 19 状态版本中，所有状态的真实 V^π 都是 (s+1)/(N+1)，
我们在 ``true_values`` 中直接给出，方便对比 MC / TD 的估计。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class RandomWalk:
    """N 状态线性随机游走（默认 19 状态，Sutton-Barto 经典设置）。

    内部状态用 0..N+1 表示，其中 0 和 N+1 是终止态。
    对外暴露的 state 取 1..N。
    """

    def __init__(
        self,
        n_states: int = 19,
        left_reward: float = -1.0,
        right_reward: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_states = int(n_states)
        self.left_reward = float(left_reward)
        self.right_reward = float(right_reward)
        self._rng = np.random.default_rng(seed)
        # 内部状态：0..n_states+1（含两端终止态）
        self._state: int = self.n_states // 2 + 1

    @property
    def state(self) -> int:
        """对外返回 1..n_states。"""
        return self._state

    @property
    def nS(self) -> int:
        return self.n_states

    def reset(self) -> int:
        self._state = self.n_states // 2 + 1
        return self.state

    def step(self, action: Optional[int] = None) -> Tuple[int, float, bool, dict]:
        """action 在此环境里其实没意义（无控制的随机游走）。
        留下接口只为统一。"""
        if self._rng.random() < 0.5:
            self._state -= 1
        else:
            self._state += 1
        if self._state == 0:
            return self._state, self.left_reward, True, {}
        if self._state == self.n_states + 1:
            return self._state, self.right_reward, True, {}
        return self.state, 0.0, False, {}

    def true_values(self) -> np.ndarray:
        """计算 V^π：随机游走下两端分别为 left_reward、right_reward。
        解析解为线性插值。"""
        n = self.n_states
        # V[i] 表示状态 i+1 的真值
        v = np.linspace(self.left_reward, self.right_reward, n + 2)[1:-1]
        return v
