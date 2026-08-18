"""RandomWalk 冒烟测试。"""
import numpy as np
import pytest

from rlenvs import RandomWalk


def test_basic():
    env = RandomWalk(n_states=19, seed=0)
    assert env.nS == 19
    s = env.reset()
    assert 1 <= s <= 19


def test_true_values_linear():
    """19 状态随机游走的真值应为线性插值。"""
    env = RandomWalk(n_states=19, left_reward=-1.0, right_reward=1.0)
    v = env.true_values()
    assert len(v) == 19
    # linspace(-1, 1, 21)[1:-1]：v[0] = -1+2/20 = -0.9
    assert abs(v[0] - (-0.9)) < 1e-6
    assert abs(v[-1] - 0.9) < 1e-6
    # 中间值应为 0（左右对称）
    assert abs(v[9]) < 1e-6


def test_terminal_rewards():
    """从中间状态出发，应一半概率落到左端 -1、一半概率右端 +1。"""
    env = RandomWalk(n_states=19, left_reward=-1.0, right_reward=1.0, seed=0)
    rs = []
    for _ in range(2000):
        env.reset()
        done = False
        while not done:
            _, r, done, _ = env.step()
        rs.append(r)
    assert abs(np.mean(rs)) < 0.1, "左右对称设置下平均收益应接近 0"


def test_step_returns_tuple():
    env = RandomWalk(n_states=5, seed=0)
    env.reset()
    out = env.step()
    assert len(out) == 4
    s, r, done, info = out
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
