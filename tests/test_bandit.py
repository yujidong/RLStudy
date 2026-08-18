"""MultiArmedBandit 冒烟测试。"""
import numpy as np
import pytest

from rlenvs import MultiArmedBandit


def test_construction():
    env = MultiArmedBandit(n_arms=10, seed=0)
    assert env.n_arms == 10
    assert len(env.q_star) == 10


def test_pull_returns_scalar():
    env = MultiArmedBandit(n_arms=5, seed=0)
    r = env.pull(2)
    assert isinstance(r, float)


def test_pull_index_error():
    env = MultiArmedBandit(n_arms=3, seed=0)
    with pytest.raises(IndexError):
        env.pull(5)


def test_optimal_arm():
    q_star = [0.1, 0.2, 0.9, 0.3, 0.4]
    env = MultiArmedBandit(n_arms=5, q_star=q_star, seed=0)
    assert env.optimal_arm() == 2
    assert env.expected_reward(2) == 0.9
    assert env.regret(2) == 0.0  # 选最优时 regret=0


def test_gaussian_statistics():
    """高斯分布的样本均值应接近 q_star。"""
    q_star = np.array([0.5, -0.3, 1.0, 0.0, -1.5])
    env = MultiArmedBandit(n_arms=5, reward_dist="gaussian", q_star=q_star, seed=42)
    samples = []
    for _ in range(5000):
        samples.append(env.pull(2))
    # 因为 q_star[2] = 1.0、std=1，5000 样本均值应在 ±0.05 内
    assert abs(np.mean(samples) - 1.0) < 0.1


def test_bernoulli_range():
    env = MultiArmedBandit(n_arms=4, reward_dist="bernoulli", seed=0)
    for _ in range(100):
        r = env.pull(env._rng.integers(4))
        assert r in (0.0, 1.0)


def test_non_stationary_drift():
    """非平稳模式下 q_star 应随步数变化。"""
    env = MultiArmedBandit(n_arms=3, non_stationary=True, drift_std=0.1, seed=0)
    q0 = env.q_star.copy()
    for _ in range(100):
        env.pull(0)
    assert not np.allclose(q0, env.q_star), "非平稳模式 q_star 应该变化"


def test_reset():
    env = MultiArmedBandit(n_arms=3, seed=0)
    q_before = env.q_star.copy()
    for _ in range(50):
        env.pull(0)
    env.reset()
    assert np.allclose(q_before, env.q_star), "reset 后应恢复初始 q_star"
