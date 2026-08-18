"""MultiArmedBandit —— 多臂老虎机，Ch01 的核心环境。

特性
----
- 支持 Gaussian / Bernoulli 奖励分布
- 平稳 / 非平稳（均值随机漂移）两种模式
- ``pull(arm)`` 返回标量奖励；``optimal_arm()`` 给出当前最优臂
- 完全暴露真实均值，方便计算 regret、做 oracle 比较
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class MultiArmedBandit:
    """N 臂老虎机。

    Parameters
    ----------
    n_arms : int
    reward_dist : 'gaussian' | 'bernoulli'
        gaussian：每个臂 ~ N(q_*, 1)
        bernoulli：每个臂以概率 q_* 输出 1
    q_star : 可选，长度 n_arms 的真实均值数组；None 则随机采样
    non_stationary : bool
        True 时每个 step 所有 q_* 都加上 N(0, drift_std) 噪声
    drift_std : 非平稳模式下每步的均值漂移标准差
    seed : int
    """

    def __init__(
        self,
        n_arms: int = 10,
        reward_dist: str = "gaussian",
        q_star: Optional[Sequence[float]] = None,
        non_stationary: bool = False,
        drift_std: float = 0.01,
        seed: Optional[int] = None,
    ):
        if reward_dist not in ("gaussian", "bernoulli"):
            raise ValueError(f"reward_dist 必须是 'gaussian' 或 'bernoulli'，得到 {reward_dist}")
        self.n_arms = int(n_arms)
        self.reward_dist = reward_dist
        self.non_stationary = non_stationary
        self.drift_std = drift_std
        self._rng = np.random.default_rng(seed)

        if q_star is None:
            if reward_dist == "gaussian":
                self.q_star = self._rng.normal(0, 1, size=self.n_arms).astype(float)
            else:  # bernoulli
                self.q_star = self._rng.uniform(0.1, 0.9, size=self.n_arms).astype(float)
        else:
            self.q_star = np.asarray(q_star, dtype=float)
            assert len(self.q_star) == self.n_arms

        self._q_star_initial = self.q_star.copy()
        self.t = 0
        self.last_action: Optional[int] = None

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.q_star = self._q_star_initial.copy()
        self.t = 0
        self.last_action = None

    def pull(self, arm: int) -> float:
        """拉杆一次。返回一个标量奖励。"""
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"arm 越界：{arm}")
        self.last_action = int(arm)
        if self.reward_dist == "gaussian":
            r = float(self._rng.normal(self.q_star[arm], 1.0))
        else:  # bernoulli
            r = 1.0 if self._rng.random() < self.q_star[arm] else 0.0
        if self.non_stationary:
            self.q_star += self._rng.normal(0, self.drift_std, size=self.n_arms)
        self.t += 1
        return r

    def step_random(self) -> float:
        """非平稳模式中模拟环境自然演化（不拉杆、只漂移）。
        留作扩展，一般不需要。"""
        if self.non_stationary:
            self.q_star += self._rng.normal(0, self.drift_std, size=self.n_arms)
        self.t += 1
        return self.q_star.mean()

    # ------------------------------------------------------------------
    # Oracle / 统计
    # ------------------------------------------------------------------
    def optimal_arm(self) -> int:
        return int(np.argmax(self.q_star))

    def expected_reward(self, arm: int) -> float:
        return float(self.q_star[arm])

    def regret(self, arm: int) -> float:
        """即时的「后悔」：最优臂期望 - 选定臂期望。"""
        return float(self.q_star[self.optimal_arm()] - self.q_star[arm])

    def cum_regret(self, arm_history: Sequence[int]) -> float:
        """给定动作序列，计算累计 regret（基于当前 q_star）。
        注意：非平稳模式下，每一步的实际最优臂会变，这里仅给一个近似。
        真实场景下应使用 ``tracker`` 一起记录。
        """
        cum = 0.0
        for a in arm_history:
            cum += self.regret(a)
            # 注：非平稳模式下无法精确还原历史漂移，这里只给平稳模式下的严格值
        return cum
