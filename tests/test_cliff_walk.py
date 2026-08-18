"""CliffWalk 冒烟测试。"""
import numpy as np
import pytest

from rlenvs import CliffWalk, cliff_world_4x12


def test_construction():
    env = CliffWalk(seed=0)
    assert env.shape == (4, 12)
    assert env.n_states == 48
    assert env.n_actions == 4


def test_start_state():
    env = CliffWalk(seed=0)
    s = env.reset()
    # 起点 (3, 0) = 3*12 + 0 = 36
    assert s == 36


def test_cliff_reward():
    """踩到悬崖应该 -100。"""
    env = CliffWalk(seed=0)
    env.reset()
    # 从 (3, 0) 往右一步到 (3, 1)，是悬崖
    s, r, done, _ = env.step(1)  # →
    assert r == -100.0
    # 悬崖不是终止态，但通常实现里落崖后继续 episode
    # 这里我们的实现不会终止，可以验证统计
    assert env.fell_in_cliff >= 1


def test_goal_is_terminal():
    """终点 (3, 11) 是终止态。"""
    env = CliffWalk(seed=0)
    env.reset()
    # 手动走过去：往右 11 步
    done = False
    s = env.state
    while not done:
        s, r, done, _ = env.step(1)
    # 终止时应在 (3, 11) = 47
    assert s == 47


def test_P_rows_sum_to_one():
    env = CliffWalk(seed=0)
    s = env.P.sum(axis=2)
    assert np.allclose(s, 1.0)


def test_factory_alias():
    """cliff_world_4x12 工厂应返回 GridWorld 实例（与 CliffWalk 兼容）。"""
    env = cliff_world_4x12(seed=0)
    assert env.shape == (4, 12)
    assert env.n_states == 48


def test_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env = CliffWalk(seed=0)
    env.reset()
    fig, ax = plt.subplots()
    env.render(ax, state=env.state)
    plt.close(fig)
