"""GridWorld —— 一个完全暴露内部状态的小型网格 MDP。

特性
----
- 任意 N×N 网格，支持墙壁、终止格、滑行（slippery）格
- **完整暴露** 转移张量 ``P[s, a, s']``、奖励函数 ``R[s, a]``、
  折扣后的回报矩阵等，方便 Ch02/03/04 中的动态规划与 TD 算法直接使用
- ``render`` 用 matplotlib 把当前状态画到 Axes 上
- 标准动作：0=↑, 1=→, 2=↓, 3=←

约定
----
- 状态用扁平索引表示：``s = row * ncols + col``
- 边界处试图走出去会原地不动
- 每步默认 -0.01 的「生存奖励」让智能体尽快到达终点；可在构造时覆盖
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# 动作常量
ACTIONS = [
    (-1, 0),  # 0: ↑  上
    (0, 1),   # 1: →  右
    (1, 0),   # 2: ↓  下
    (0, -1),  # 3: ←  左
]
ACTION_NAMES = ["↑", "→", "↓", "←"]
N_ACTIONS = 4


class GridWorld:
    """可配置的 N×N 网格 MDP。

    Parameters
    ----------
    shape : (rows, cols)
    terminals : 终止格的 (row, col) 列表
    walls : 墙体格的 (row, col) 列表
    slippery : 滑行格，每步以 prob_slip 概率随机选择其他动作
    rewards : dict，``{(row, col): reward_value}``，进入该格获得的奖励
              未指定的格子使用 ``default_reward``
    default_reward : 进入未指定格子的奖励（默认 -0.01）
    start : 起始格 (row, col)。None 则随机选非终止/非墙的格
    slip_prob : slippery 格上随机滑动的概率
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (5, 5),
        terminals: Sequence[Tuple[int, int]] = ((0, 4),),
        walls: Sequence[Tuple[int, int]] = (),
        slippery: Sequence[Tuple[int, int]] = (),
        rewards: Optional[dict] = None,
        default_reward: float = -0.01,
        start: Optional[Tuple[int, int]] = None,
        slip_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.shape = shape
        self.n_rows, self.n_cols = shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = N_ACTIONS
        self.action_names = list(ACTION_NAMES)

        self.terminals = set(terminals)
        self.walls = set(walls)
        self.slippery = set(slippery)
        self.rewards = dict(rewards or {})
        self.default_reward = default_reward
        self.start = start
        self.slip_prob = slip_prob
        self._rng = np.random.default_rng(seed)

        # 计算 P[s, a, s'] 与 R[s, a]
        self.P, self.R = self._build_model()

        # step() 热路径用的预计算集合（terminals/slippery 构造后不变）
        self._terminal_set = {self._flat(rc) for rc in self.terminals}
        self._slippery_set = {self._flat(rc) for rc in self.slippery}

        # 当前状态
        self._state: int = self._flat(self._sample_start())

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    @property
    def state(self) -> int:
        return self._state

    @property
    def nS(self) -> int:
        return self.n_states

    @property
    def nA(self) -> int:
        return self.n_actions

    def reset(self) -> int:
        """重置到起点。返回初始状态索引。"""
        self._state = self._flat(self._sample_start())
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """执行一步。

        Returns
        -------
        (next_state, reward, done, info)
        """
        s = self._state
        # 终止态原地停留
        if s in self._terminal_set:
            return s, 0.0, True, {}

        # slippery 格随机滑动
        if s in self._slippery_set and self._rng.random() < self.slip_prob:
            action = int(self._rng.integers(self.n_actions))

        # 转移：按 P[s, a, .] 采样。这里 P 基本是确定性的，
        # 但仍然从 P 里采样以保持通用。
        next_s = int(self._rng.choice(self.n_states, p=self.P[s, action]))
        r = float(self.R[s, action])
        self._state = next_s
        done = next_s in self._terminal_set
        return next_s, r, done, {}

    # ------------------------------------------------------------------
    # 模型访问（Ch02/03 必备）
    # ------------------------------------------------------------------
    def build_P(self) -> np.ndarray:
        return self.P.copy()

    def build_R(self) -> np.ndarray:
        return self.R.copy()

    def transition(self, s: int, a: int, s_next: int) -> float:
        return float(self.P[s, a, s_next])

    def expected_reward(self, s: int, a: int) -> float:
        return float(self.R[s, a])

    def is_terminal(self, s: int) -> bool:
        return s in self._terminal_set

    def state_to_xy(self, s: int) -> Tuple[int, int]:
        return self._unflat(s)

    def xy_to_state(self, r: int, c: int) -> int:
        return self._flat((r, c))

    # ------------------------------------------------------------------
    # 可视化（被 utils.viz.animate_agent 调用）
    # ------------------------------------------------------------------
    def render(self, ax, state: Optional[int] = None):
        """在 ax 上画出整个网格，高亮当前 state。"""
        from matplotlib.patches import Circle, Rectangle

        if state is None:
            state = self._state
        ax.clear()
        # 画格线
        for r in range(self.n_rows + 1):
            ax.axhline(r, color="black", linewidth=1.0)
        for c in range(self.n_cols + 1):
            ax.axvline(c, color="black", linewidth=1.0)
        # 墙
        for (r, c) in self.walls:
            ax.add_patch(Rectangle((c, r), 1, 1, color="black"))
        # 终止
        for (r, c) in self.terminals:
            ax.add_patch(Rectangle((c, r), 1, 1, color="gold", alpha=0.6))
            ax.text(c + 0.5, r + 0.5, "G", ha="center", va="center", fontsize=14, fontweight="bold")
        # slippery
        for (r, c) in self.slippery:
            ax.add_patch(Rectangle((c, r), 1, 1, color="#9ad", alpha=0.35))
        # 当前位置
        if state is not None:
            r, c = self._unflat(state)
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.3, color="crimson"))
        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # ------------------------------------------------------------------
    # 内部：构建 P 和 R
    # ------------------------------------------------------------------
    def _build_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """构建转移张量 P[s,a,s'] 和奖励矩阵 R[s,a]。"""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.full((self.n_states, self.n_actions), self.default_reward, dtype=float)

        for s in range(self.n_states):
            r, c = self._unflat(s)
            # 终止态：自循环，零奖励
            if (r, c) in self.terminals:
                P[s, :, s] = 1.0
                R[s, :] = 0.0
                continue
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                # 出界或撞墙：原地
                if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or (nr, nc) in self.walls:
                    P[s, a, s] = 1.0
                    # 墙与出界不给奖励（沿用默认）
                else:
                    s_next = self._flat((nr, nc))
                    P[s, a, s_next] = 1.0
                    # 进入终点格的特殊奖励
                    if (nr, nc) in self.rewards:
                        R[s, a] = self.rewards[(nr, nc)]
                    elif (nr, nc) in self.terminals:
                        R[s, a] = 1.0  # 默认终点奖励
        return P, R

    def _flat(self, rc: Tuple[int, int]) -> int:
        return rc[0] * self.n_cols + rc[1]

    def _unflat(self, s: int) -> Tuple[int, int]:
        return divmod(int(s), self.n_cols)

    def _terminal_flats(self) -> set:
        # 构造时已缓存到 self._terminal_set，这里直接返回
        return self._terminal_set

    def _sample_start(self) -> Tuple[int, int]:
        if self.start is not None:
            return self.start
        # 随机选一个非终止/非墙的格
        choices = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) not in self.terminals and (r, c) not in self.walls:
                    choices.append((r, c))
        return tuple(choices[int(self._rng.integers(len(choices)))])


# ---------------------------------------------------------------------------
# 经典预设环境（Sutton-Barto 风格）
# ---------------------------------------------------------------------------
def cliff_world_4x12(seed: Optional[int] = None) -> "GridWorld":
    """Sutton-Barto 标准 4×12 Cliff，留给 Ch05 直接复用。

    起点在左下角，终点在右下角，最下面一行中间是悬崖（-100）。
    """
    rows, cols = 4, 12
    cliff = [(3, c) for c in range(1, 11)]
    rewards = {**{c: -100.0 for c in cliff}, (3, 11): 0.0}
    return GridWorld(
        shape=(rows, cols),
        terminals=[(3, 11)],
        slippery=(),
        rewards=rewards,
        default_reward=-1.0,
        start=(3, 0),
        seed=seed,
    )


def small_grid_5x5(seed: Optional[int] = None) -> "GridWorld":
    """Ch02/03 用的小 5×5：起点左下，终点右上 +1。

    还在 (1,2) 放一个 -0.5 的「陷阱」用来演示 γ 对策略的影响。
    """
    return GridWorld(
        shape=(5, 5),
        terminals=[(0, 4)],
        walls=[(2, 2), (2, 3)],
        rewards={(0, 4): 1.0, (1, 2): -0.5},
        default_reward=-0.05,
        start=(4, 0),
        seed=seed,
    )


def bridge_grid(seed: Optional[int] = None) -> "GridWorld":
    """Ch02 练习用的「桥」网格：捷径需要过桥，但桥下有强惩罚。"""
    # 桥是 (1, 1..3)，桥下是 row 2，落入 -1
    return GridWorld(
        shape=(3, 5),
        terminals=[(1, 4)],
        rewards={(2, c): -1.0 for c in range(1, 4)},  # 桥下
        default_reward=-0.01,
        start=(1, 0),
        seed=seed,
    )
