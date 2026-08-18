"""CartPoleLite —— 简化版 CartPole，Ch06 DQN 必备。

经典控制问题：小车在水平轨道上左右移动，杆子通过铰链连在小车上。
目标是通过左右推小车让杆子保持直立（不倒下）。

状态空间（连续，4 维）：
    [x, x_dot, theta, theta_dot]
    - x         : 小车水平位置（米）
    - x_dot     : 小车水平速度（米/秒）
    - theta     : 杆子与竖直方向的夹角（弧度），正值表示向右倾
    - theta_dot : 杆子的角速度（弧度/秒）

动作空间（离散，2 个）：
    0 = 向左推（force < 0）
    1 = 向右推（force > 0）

终止条件：
    - |x| > x_threshold       （小车出轨）
    - |theta| > theta_threshold（杆子倒下）
    - 步数 >= max_steps        （成功撑住，奖励 +1/步）

奖励：每步 +1.0（包括终止步），鼓励 agent 尽可能久地保持平衡。

物理参数使用 OpenAI Gym CartPole-v1 的标准设定（无摩擦、刚性杆），
但暴露所有内部状态方便教学（这是和 OpenAI Gym 的关键差异——
Gym 把物理藏在内部，CartPoleLite 把它们写在类属性里供读者探索）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class CartPoleLite:
    """简化版 CartPole 环境（连续状态、离散动作）。

    与项目其它环境（GridWorld、RandomWalk）的 API 一致：
        - ``reset() -> np.ndarray``：返回初始观测（shape [4]）
        - ``step(action) -> (next_state, reward, done, info)``
        - ``state`` 属性（np.ndarray，shape [4]）
        - ``render(ax, state=None)`` 在 matplotlib Axes 上画快照

    新增连续状态空间约定：
        - 状态是 ``np.ndarray``（不像 GridWorld 是 int）
        - 用 ``observation_dim`` 和 ``action_dim`` 替代 ``nS``（连续无 nS）
        - 但仍保留 ``nA`` 与离散动作环境一致

    Example:
        >>> env = CartPoleLite(seed=0)
        >>> s = env.reset()
        >>> s.shape
        (4,)
        >>> s_next, r, done, info = env.step(1)  # 向右推
    """

    # ---- 物理参数（标准 CartPole-v1 设定，暴露出来方便教学探索）----
    gravity = 9.8            # 重力加速度 (m/s²)
    mass_cart = 1.0          # 小车质量 (kg)
    mass_pole = 0.1          # 杆子质量 (kg)
    total_mass = mass_cart + mass_pole
    length = 0.5             # 杆子半长 (m) —— 实际杆长 2 * length
    polemass_length = mass_pole * length  # 杆子质量 × 半长
    force_mag = 10.0         # 推力大小 (N)
    tau = 0.02               # 单步时间间隔 (s)，相当于 50 Hz

    # ---- 终止阈值 ----
    x_threshold = 2.4        # 小车位置边界 (m)
    theta_threshold = 12 * np.pi / 180  # 杆子角度边界 = 12° ≈ 0.2094 rad

    def __init__(
        self,
        max_steps: int = 500,
        seed: Optional[int] = None,
        init_range: float = 0.05,
    ):
        """
        Parameters
        ----------
        max_steps : int
            单 episode 最多步数（达到则 done=True，视为成功）。
            CartPole-v1 默认 500。
        seed : int, optional
            随机种子。与其它环境一致，用 ``np.random.default_rng``。
        init_range : float
            初始状态扰动范围（±init_range）。设 0 表示从完美直立起步。
        """
        self.max_steps = int(max_steps)
        self.init_range = float(init_range)
        self._rng = np.random.default_rng(seed)

        # 内部状态：[x, x_dot, theta, theta_dot]
        self._state: np.ndarray = np.zeros(4, dtype=np.float64)
        self._step_count: int = 0

    # ---- 状态访问 ----
    @property
    def state(self) -> np.ndarray:
        """当前观测（shape [4]）。返回副本，改动不会影响环境内部状态。"""
        return self._state.copy()

    @property
    def observation_dim(self) -> int:
        return 4

    @property
    def nA(self) -> int:
        """离散动作数（与 GridWorld 等约定一致用 nA）。"""
        return 2

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def step_count(self) -> int:
        return self._step_count

    # ---- 核心接口 ----
    def reset(self) -> np.ndarray:
        """重置到接近直立的小扰动初始状态，返回观测（shape [4]）。

        与 Gym CartPole 一致：每个维度加 ±0.05 的均匀噪声，
        避免 agent 记住确定性初始状态。
        """
        self._state = self._rng.uniform(-self.init_range, self.init_range, size=4)
        self._step_count = 0
        return self._state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行 action，返回 (next_state, reward, done, info)。

        Parameters
        ----------
        action : int
            0 = 向左推（force = -force_mag）
            1 = 向右推（force = +force_mag）

        Returns
        -------
        next_state : np.ndarray, shape [4]
        reward : float  （每步 +1.0，包括终止步）
        done : bool     （小车出轨 / 杆子倒下 / 达到 max_steps）
        info : dict     （含 ``termination_reason`` 字段，方便调试）
        """
        if action not in (0, 1):
            raise ValueError(f"action 必须是 0 或 1，收到 {action}")

        x, x_dot, theta, theta_dot = self._state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # 标准 CartPole-v1 物理（OpenAI Gym 实现）
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler 积分（简单但够用；对教学目的足够准确）
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self._step_count += 1

        # 终止判定
        terminated = bool(
            abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
        )
        truncated = self._step_count >= self.max_steps
        done = terminated or truncated

        # 奖励：每步 +1.0（CartPole-v1 标准）。terminated 时仍然给 +1
        # （代表"撑过这一步"），但 truncated 时也算 +1（达到上限视为成功）。
        reward = 1.0

        info = {
            "termination_reason": (
                "out_of_bounds" if terminated and abs(x) > self.x_threshold
                else "pole_fell" if terminated
                else "max_steps" if truncated
                else None
            ),
            "step_count": self._step_count,
        }
        return self._state.copy(), reward, done, info

    def is_terminal(self, state: Optional[np.ndarray] = None) -> bool:
        """检查给定状态（或当前状态）是否终止。"""
        s = self._state if state is None else state
        return bool(abs(s[0]) > self.x_threshold or abs(s[2]) > self.theta_threshold)

    # ---- 可视化（被 utils.viz.animate_agent 复用）----
    def render(self, ax, state: Optional[np.ndarray] = None):
        """在 matplotlib Axes 上画当前状态（小车 + 杆子）。

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            要画到的 Axes（会被 clear 后重画）。
        state : np.ndarray, optional
            若提供，画这个状态而不是当前 ``self._state``（用于动画）。
        """
        s = self._state if state is None else state
        x, _, theta, _ = s

        ax.clear()
        # 地面
        ax.axhline(0, color="k", linewidth=1)
        # 轨道边界
        ax.plot([-self.x_threshold, -self.x_threshold], [-0.1, 0.1], "r-", linewidth=2)
        ax.plot([self.x_threshold, self.x_threshold], [-0.1, 0.1], "r-", linewidth=2)

        # 小车（矩形）
        cart_w, cart_h = 0.4, 0.2
        cart_x = x - cart_w / 2
        ax.add_patch(plt_rect(cart_x, -cart_h / 2, cart_w, cart_h, color="steelblue"))

        # 杆子（线段）
        pole_top_x = x + self.length * np.sin(theta)
        pole_top_y = self.length * np.cos(theta)
        ax.plot([x, pole_top_x], [0, pole_top_y], "k-", linewidth=3)
        # 杆子顶端的"重物"（视觉）
        ax.scatter([pole_top_x], [pole_top_y], s=80, c="crimson", zorder=5)

        # 标题显示当前状态
        ax.set_xlim(-self.x_threshold * 1.5, self.x_threshold * 1.5)
        ax.set_ylim(-0.5, self.length * 2 + 0.2)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(
            f"x={s[0]:+.2f}m  θ={s[2]:+.3f}rad ({np.degrees(s[2]):+.1f}°)  "
            f"step={self._step_count}"
        )


# 工具：避免在文件顶部 import matplotlib（让无 matplotlib 环境也能用环境）
def plt_rect(x, y, w, h, **kwargs):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, **kwargs)
