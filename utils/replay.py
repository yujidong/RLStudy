"""Experience Replay Buffer（Ch06+ 用）。

存储 transitions ``(s, a, r, s_next, done)``，支持随机均匀采样 mini-batch。
这是 DQN 的核心技术之一（Mnih et al. 2015），打破时序相关性、提高样本效率。

本实现用 numpy 数组存储每个字段（不存 Python tuple 列表），
采样时直接索引切出 batch——比 list-based 快约 10x。
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .seed import get_rng


class Transition(NamedTuple):
    """单条 transition 的结构化表示（主要用于类型标注和文档）。"""
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]  # 终止态时为 None 或全 0（约定）
    done: bool


class ReplayBuffer:
    """固定容量的 experience replay buffer。

    用循环覆盖策略：满了之后从最老的开始覆盖。
    采样是均匀随机（不带 prioritization—— PER 留作 Ch06 练习）。

    Example
    -------
    >>> buf = ReplayBuffer(capacity=10000, state_dim=4)
    >>> for _ in range(100):
    ...     buf.add(s, a, r, s_next, done)
    >>> s_b, a_b, r_b, sn_b, d_b = buf.sample(32)
    >>> s_b.shape, a_b.shape, r_b.shape
    ((32, 4), (32,), (32,), (32, 4))
    """

    def __init__(self, capacity: int, state_dim: int, state_dtype=np.float32):
        """
        Parameters
        ----------
        capacity : int
            buffer 最大容量。满了之后新数据覆盖最老的。
        state_dim : int
            状态维度（如 CartPoleLite 的 4）。next_state 也用同样维度。
        state_dtype : numpy dtype
            状态的数值类型。默认 float32（PyTorch 默认）。
        """
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self._states = np.zeros((capacity, state_dim), dtype=state_dtype)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=state_dtype)
        self._dones = np.zeros(capacity, dtype=np.float32)  # 0/1 float（方便 torch 转换）
        self._ptr = 0   # 下一个写入位置
        self._size = 0  # 当前实际容量（min(ptr, capacity)）

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """添加一条 transition。如果 next_state 是 None（终止态），用零向量替代。"""
        idx = self._ptr
        self._states[idx] = np.asarray(state, dtype=self._states.dtype)
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        if next_state is None:
            self._next_states[idx] = 0
        else:
            self._next_states[idx] = np.asarray(next_state, dtype=self._next_states.dtype)
        self._dones[idx] = float(done)

        # 循环指针
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None):
        """均匀随机采样 batch_size 条 transition。

        Returns
        -------
        (states, actions, rewards, next_states, dones) : tuple of np.ndarray
            shapes: [batch, state_dim], [batch], [batch], [batch, state_dim], [batch]
        """
        if rng is None:
            rng = get_rng()  # 共享 Generator：set_seed 可控，且避免每次采样新建 RNG
        if self._size < batch_size:
            raise ValueError(
                f"buffer 里有 {self._size} 条，要求 batch_size={batch_size}"
            )
        idxs = rng.integers(0, self._size, size=batch_size)
        return (
            self._states[idxs],
            self._actions[idxs],
            self._rewards[idxs],
            self._next_states[idxs],
            self._dones[idxs],
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """方便训练循环判断是否可以开始采样。"""
        return self._size >= 1
