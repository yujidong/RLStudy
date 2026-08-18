"""CartPoleLite 冒烟测试。"""
import numpy as np
import pytest

from rlenvs import CartPoleLite


def test_basic():
    env = CartPoleLite(seed=0)
    assert env.observation_dim == 4
    assert env.nA == 2
    assert env.action_dim == 2
    s = env.reset()
    assert s.shape == (4,)
    assert s.dtype == np.float64


def test_step_returns_tuple():
    env = CartPoleLite(seed=0)
    env.reset()
    out = env.step(1)
    assert len(out) == 4
    s, r, done, info = out
    assert s.shape == (4,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "termination_reason" in info


def test_invalid_action():
    env = CartPoleLite(seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(2)
    with pytest.raises(ValueError):
        env.step(-1)


def test_initial_state_near_upright():
    """reset 后 theta 应接近 0（小扰动）。"""
    env = CartPoleLite(seed=0, init_range=0.05)
    for _ in range(10):
        s = env.reset()
        assert abs(s[2]) < 0.1, f"初始 theta {s[2]} 过大"
        assert abs(s[0]) < 0.1, f"初始 x {s[0]} 过大"


def test_random_policy_episode_terminates():
    """随机策略下的 episode 必然终止（pole_fell 或 out_of_bounds）。"""
    env = CartPoleLite(seed=0, max_steps=500)
    rng = np.random.default_rng(0)
    steps_to_fall = []
    for _ in range(20):
        env.reset()
        done = False
        info = {}
        while not done:
            a = int(rng.integers(2))
            _, _, done, info = env.step(a)
        steps_to_fall.append(env.step_count)
        # 随机策略通常很快倒（< 100 步），但偶尔可能侥幸
    mean_steps = np.mean(steps_to_fall)
    assert mean_steps < 80, f"随机策略平均 {mean_steps} 步，不应这么久"
    # 所有 episode 都该终止（不能死循环）
    assert all(s <= 500 for s in steps_to_fall)


def test_heuristic_can_solve():
    """heuristic: 根据 theta + 0.5*theta_dot 选动作 → 应能撑到 max_steps。"""
    env = CartPoleLite(seed=42, max_steps=200)
    env.reset()
    done = False
    while not done:
        s = env.state
        score = s[2] + 0.5 * s[3]
        a = 1 if score > 0 else 0
        _, _, done, _ = env.step(a)
    assert env.step_count == 200, f"heuristic 应撑到 max_steps，实际 {env.step_count}"


def test_reproducibility():
    """同 seed 应得到相同初始状态序列。"""
    e1 = CartPoleLite(seed=42)
    e2 = CartPoleLite(seed=42)
    s1 = e1.reset()
    s2 = e2.reset()
    assert np.allclose(s1, s2)
    # 跑 10 步
    for _ in range(10):
        a = 1
        sn1, _, _, _ = e1.step(a)
        sn2, _, _, _ = e2.step(a)
        assert np.allclose(sn1, sn2)


def test_render_does_not_crash():
    """render 应能被调用而不抛错（不验证画出来的图）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env = CartPoleLite(seed=0)
    env.reset()
    fig, ax = plt.subplots()
    env.render(ax)
    # 再走几步后再 render（带 state 参数）
    env.step(1)
    env.step(1)
    env.render(ax, state=env.state)
    plt.close(fig)


def test_done_after_max_steps():
    env = CartPoleLite(seed=0, max_steps=10)
    env.reset()
    done = False
    info = {}
    while not done:
        # 用 heuristic 让 episode 不提前结束
        s = env.state
        score = s[2] + 0.5 * s[3]
        a = 1 if score > 0 else 0
        _, _, done, info = env.step(a)
    assert info["termination_reason"] == "max_steps"
    assert env.step_count == 10
