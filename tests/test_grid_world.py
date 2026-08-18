"""GridWorld 冒烟测试。"""
import numpy as np
import pytest

from rlenvs import GridWorld, bridge_grid, small_grid_5x5


def test_basic_construction():
    env = GridWorld(shape=(5, 5), terminals=[(0, 4)], start=(4, 0))
    assert env.n_states == 25
    assert env.n_actions == 4
    assert env.P.shape == (25, 4, 25)
    assert env.R.shape == (25, 4)


def test_P_rows_sum_to_one():
    env = small_grid_5x5(seed=0)
    s = env.P.sum(axis=2)
    assert np.allclose(s, 1.0), f"P 行和必须为 1，实际：{s}"


def test_terminal_self_loop():
    env = small_grid_5x5(seed=0)
    # 终点 (0, 4) = 0*5+4 = 4
    assert env.is_terminal(4)
    for a in range(4):
        assert env.P[4, a, 4] == 1.0
        assert env.R[4, a] == 0.0


def test_walls_block_movement():
    """撞墙/出界应该原地停留。"""
    env = GridWorld(shape=(3, 3), walls=[(1, 1)], start=(0, 0))
    # 从 (0,0) 往右（→）是 (0,1)，没问题
    s = env.xy_to_state(0, 0)
    a_right = 1
    next_s = np.argmax(env.P[s, a_right])
    assert env.state_to_xy(next_s) == (0, 1)
    # 从 (0,1) 往下（↓）撞 (1,1) 墙，应该原地
    s2 = env.xy_to_state(0, 1)
    a_down = 2
    next_s2 = np.argmax(env.P[s2, a_down])
    assert env.state_to_xy(next_s2) == (0, 1), "撞墙应原地停留"


def test_step_interface():
    env = small_grid_5x5(seed=42)
    s0 = env.reset()
    assert s0 == env.xy_to_state(4, 0)  # 起点
    s1, r, done, info = env.step(1)  # 右
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_presets():
    env_grid = small_grid_5x5(seed=0)
    env_bridge = bridge_grid(seed=0)
    assert env_grid.n_states == 25
    assert env_bridge.n_states == 15


def test_render_returns_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env = small_grid_5x5(seed=0)
    env.reset()
    fig, ax = plt.subplots()
    env.render(ax, state=env.state)
    plt.close(fig)
