"""utils/offline_rl.py 冒烟测试（Ch18 基础设施）。"""
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from rlenvs import CartPoleLite
from utils.networks import QNetwork
from utils.replay import ReplayBuffer
from utils.offline_rl import (
    CQLTrainer,
    DTTrainer,
    DecisionTransformer,
    IQLTrainer,
    OfflineDataset,
    cql_loss,
    collect_offline_dataset,
    dt_loss,
    dt_rollout,
    evaluate_policy,
    expectile_loss,
    heuristic_cartpole_policy,
    offline_dqn_update_step,
    random_policy_factory,
)


# -------------------- collect_offline_dataset / OfflineDataset --------------------
def test_collect_offline_dataset_basic():
    env = CartPoleLite(max_steps=20, seed=0)
    policy = random_policy_factory(env.nA, seed=1)
    buf = collect_offline_dataset(env, policy, n_episodes=5, seed=42)
    assert len(buf) > 0
    # 5 条 episode，每条最多 20 步 → 总 transition 数 <= 100
    assert len(buf) <= 100
    # 最后一帧应该 done=True（episode 自然结束）
    assert buf._dones[buf._size - 1] == 1.0


def test_collect_offline_dataset_with_heuristic_runs_longer():
    env1 = CartPoleLite(max_steps=200, seed=0)
    env2 = CartPoleLite(max_steps=200, seed=0)
    rp = random_policy_factory(env1.nA, seed=1)
    buf_random = collect_offline_dataset(env1, rp, n_episodes=3, seed=42)
    buf_heur = collect_offline_dataset(env2, heuristic_cartpole_policy, n_episodes=3, seed=42)
    # 启发式应该撑住更多步（更长的总 transitions）
    assert len(buf_heur) >= len(buf_random)


def test_offline_dataset_from_buffer_splits_episodes():
    env = CartPoleLite(max_steps=15, seed=0)
    rp = random_policy_factory(env.nA, seed=1)
    buf = collect_offline_dataset(env, rp, n_episodes=4, seed=42)
    ds = OfflineDataset.from_buffer(buf)
    assert len(ds) == len(buf)
    # 4 条 episode → 4 个边界
    assert len(ds.episode_boundaries) == 4
    # 边界应该覆盖所有 transitions
    starts = [b[0] for b in ds.episode_boundaries]
    ends = [b[1] for b in ds.episode_boundaries]
    assert starts[0] == 0
    assert ends[-1] == ds.n
    # 每 episode 最后帧 done=True
    for s, e in ds.episode_boundaries:
        assert ds.dones[e - 1] >= 0.5


def test_offline_dataset_sample_shapes():
    env = CartPoleLite(max_steps=10, seed=0)
    rp = random_policy_factory(env.nA, seed=1)
    buf = collect_offline_dataset(env, rp, n_episodes=2, seed=42)
    ds = OfflineDataset.from_buffer(buf)
    s, a, r, sn, d = ds.sample(8)
    assert s.shape == (8, 4)
    assert a.shape == (8,)
    assert r.shape == (8,)
    assert sn.shape == (8, 4)
    assert d.shape == (8,)


def test_return_to_go_simple_cases():
    # 构造一个已知数据集：episode = [(r=1), (r=2), (r=3)]，done=[0,0,1]
    states = np.zeros((3, 4), dtype=np.float32)
    next_states = np.zeros((3, 4), dtype=np.float32)
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    actions = np.array([0, 1, 0], dtype=np.int64)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ds = OfflineDataset(states, actions, rewards, next_states, dones, [(0, 3)])
    R = ds.return_to_go(gamma=1.0)
    # return-to-go: t=0 -> 1+2+3=6, t=1 -> 2+3=5, t=2 -> 3
    np.testing.assert_allclose(R, [6.0, 5.0, 3.0], rtol=1e-5)


def test_return_to_go_discount():
    states = np.zeros((2, 4), dtype=np.float32)
    next_states = np.zeros((2, 4), dtype=np.float32)
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int64)
    dones = np.array([0.0, 1.0], dtype=np.float32)
    ds = OfflineDataset(states, actions, rewards, next_states, dones, [(0, 2)])
    R = ds.return_to_go(gamma=0.5)
    # t=0 -> 1 + 0.5*1 = 1.5, t=1 -> 1
    np.testing.assert_allclose(R, [1.5, 1.0], rtol=1e-5)


# -------------------- expectile_loss --------------------
def test_expectile_loss_tau_half_proportional_to_mse():
    import torch.nn.functional as F
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 2.0, 2.0])
    # tau=0.5 时权重恒为 0.5，所以 loss = 0.5 * MSE
    expected_half_mse = 0.5 * F.mse_loss(pred, target)
    actual = expectile_loss(pred, target, expectile=0.5)
    torch.testing.assert_close(actual, expected_half_mse)


def F_mse(p, t):
    import torch.nn.functional as F
    return F.mse_loss(p, t)


def test_expectile_loss_tau_high_pushes_pred_up():
    # 当 target > pred 时（残差正），高 tau 应该给更大权重
    pred = torch.tensor([0.0])
    target = torch.tensor([1.0])
    loss_low = expectile_loss(pred, target, expectile=0.5)
    loss_high = expectile_loss(pred, target, expectile=0.9)
    # 高 tau 在正残差下权重更大 → loss 更大
    assert float(loss_high) > float(loss_low)


def test_expectile_loss_tau_low_pushes_pred_down():
    pred = torch.tensor([1.0])
    target = torch.tensor([0.0])  # 残差负
    loss_low = expectile_loss(pred, target, expectile=0.1)
    loss_high = expectile_loss(pred, target, expectile=0.9)
    # 残差负时，低 tau 权重大 → loss 大
    assert float(loss_low) > float(loss_high)


def test_expectile_loss_zero_at_pred_equals_target():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    for tau in [0.1, 0.5, 0.9]:
        loss = expectile_loss(pred, target, expectile=tau)
        assert float(loss) < 1e-12


# -------------------- cql_loss --------------------
def test_cql_loss_returns_scalar_and_stats():
    import torch
    q = QNetwork(4, 2, [16, 16])
    tgt = QNetwork(4, 2, [16, 16])
    s = torch.randn(8, 4)
    a = torch.randint(0, 2, (8,))
    r = torch.randn(8)
    sn = torch.randn(8, 4)
    d = torch.zeros(8)
    loss, stats = cql_loss(q, s, a, r, sn, d, target_net=tgt, alpha=1.0)
    assert loss.dim() == 0  # scalar
    assert "bellman_loss" in stats
    assert "cql_reg" in stats
    assert "lse" in stats


def test_cql_loss_alpha_zero_equals_dqn_loss():
    # alpha=0 时 CQL regularizer 不起作用，应该只剩 bellman loss
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    q = QNetwork(4, 2, [16, 16])
    tgt = QNetwork(4, 2, [16, 16])
    tgt.load_state_dict(q.state_dict())
    s = torch.randn(8, 4)
    a = torch.randint(0, 2, (8,))
    r = torch.randn(8)
    sn = torch.randn(8, 4)
    d = torch.zeros(8)
    loss, stats = cql_loss(q, s, a, r, sn, d, target_net=tgt, alpha=0.0)
    torch.testing.assert_close(loss, torch.tensor(stats["bellman_loss"]))


# -------------------- offline_dqn_update_step --------------------
def test_offline_dqn_update_step_runs():
    import torch
    torch.manual_seed(0)
    q = QNetwork(4, 2, [16, 16])
    tgt = QNetwork(4, 2, [16, 16])
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    batch = (
        np.random.randn(8, 4).astype(np.float32),
        np.random.randint(0, 2, (8,)),
        np.random.randn(8).astype(np.float32),
        np.random.randn(8, 4).astype(np.float32),
        np.zeros(8, dtype=np.float32),
    )
    stats = offline_dqn_update_step(q, tgt, opt, batch, gamma=0.99)
    assert "q_mean" in stats and "loss" in stats


# -------------------- CQLTrainer --------------------
def test_cql_trainer_step():
    import torch
    torch.manual_seed(0)
    env = CartPoleLite(max_steps=10, seed=0)
    rp = random_policy_factory(env.nA, seed=1)
    buf = collect_offline_dataset(env, rp, n_episodes=5, seed=42)
    ds = OfflineDataset.from_buffer(buf)
    q = QNetwork(4, 2, [32, 32])
    tgt = QNetwork(4, 2, [32, 32])
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    trainer = CQLTrainer(q, tgt, opt, ds, n_actions=2, alpha=1.0, batch_size=16)
    s0 = trainer.step()
    for _ in range(5):
        s = trainer.step()
    assert s["loss"] > 0
    assert trainer.step_count == 6


def test_cql_trainer_greedy_action():
    import torch
    torch.manual_seed(0)
    env = CartPoleLite(max_steps=10, seed=0)
    rp = random_policy_factory(env.nA, seed=1)
    buf = collect_offline_dataset(env, rp, n_episodes=3, seed=42)
    ds = OfflineDataset.from_buffer(buf)
    q = QNetwork(4, 2, [16, 16])
    tgt = QNetwork(4, 2, [16, 16])
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    trainer = CQLTrainer(q, tgt, opt, ds, n_actions=2, batch_size=8)
    a = trainer.greedy_action(np.zeros(4, dtype=np.float32))
    assert a in (0, 1)


# -------------------- IQLTrainer --------------------
def test_iql_trainer_step():
    import torch
    torch.manual_seed(0)
    env = CartPoleLite(max_steps=10, seed=0)
    rp = random_policy_factory(env.nA, seed=1)
    buf = collect_offline_dataset(env, rp, n_episodes=5, seed=42)
    ds = OfflineDataset.from_buffer(buf)
    trainer = IQLTrainer(
        state_dim=4, n_actions=2, dataset=ds,
        hidden_dims=[32, 32], expectile=0.7, batch_size=16,
    )
    for _ in range(5):
        s = trainer.step()
    assert s["q_loss"] >= 0
    assert s["v_loss"] >= 0
    assert trainer.step_count == 5


def test_iql_trainer_greedy_and_softmax_action():
    import torch
    torch.manual_seed(0)
    env = CartPoleLite(max_steps=10, seed=0)
    rp = random_policy_factory(env.nA, seed=1)
    buf = collect_offline_dataset(env, rp, n_episodes=3, seed=42)
    ds = OfflineDataset.from_buffer(buf)
    trainer = IQLTrainer(
        state_dim=4, n_actions=2, dataset=ds,
        hidden_dims=[16, 16], batch_size=8,
    )
    for _ in range(3):
        trainer.step()
    a = trainer.greedy_action(np.zeros(4, dtype=np.float32))
    assert a in (0, 1)
    a2 = trainer.softmax_action(np.zeros(4, dtype=np.float32), beta=1.0)
    assert a2 in (0, 1)


# -------------------- DecisionTransformer --------------------
def test_decision_transformer_forward_shapes():
    import torch
    dt = DecisionTransformer(state_dim=4, n_actions=2, hidden_dims=[16, 16])
    # 2D 输入：[B] return + [B, state_dim] state -> [B, n_actions]
    rtg = torch.tensor([10.0, 5.0])
    s = torch.randn(2, 4)
    logits = dt(rtg, s)
    assert logits.shape == (2, 2)


def test_decision_transformer_predict_action():
    dt = DecisionTransformer(state_dim=4, n_actions=2, hidden_dims=[16, 16])
    s = np.zeros(4, dtype=np.float32)
    a = dt.predict_action(10.0, s, greedy=True)
    assert a in (0, 1)
    a2 = dt.predict_action(10.0, s, greedy=False)
    assert a2 in (0, 1)


def test_dt_loss_runs():
    import torch
    dt = DecisionTransformer(state_dim=4, n_actions=2, hidden_dims=[16, 16])
    rtg = torch.tensor([10.0, 5.0, 1.0])
    s = torch.randn(3, 4)
    a = torch.tensor([0, 1, 0])
    loss, stats = dt_loss(dt, rtg, s, a)
    assert loss.dim() == 0
    assert "action_acc" in stats
    assert 0.0 <= stats["action_acc"] <= 1.0


def test_dt_trainer_step():
    import torch
    torch.manual_seed(0)
    env = CartPoleLite(max_steps=10, seed=0)
    rp = random_policy_factory(env.nA, seed=1)
    buf = collect_offline_dataset(env, rp, n_episodes=5, seed=42)
    ds = OfflineDataset.from_buffer(buf)
    dt = DecisionTransformer(state_dim=4, n_actions=2, hidden_dims=[32, 32])
    opt = torch.optim.Adam(dt.parameters(), lr=1e-3)
    trainer = DTTrainer(dt, opt, ds, batch_size=16)
    for _ in range(5):
        s = trainer.step()
    assert s["loss"] > 0
    assert trainer.step_count == 5


def test_dt_rollout_runs():
    import torch
    torch.manual_seed(0)
    env = CartPoleLite(max_steps=20, seed=0)
    dt = DecisionTransformer(state_dim=4, n_actions=2, hidden_dims=[16, 16])
    total, rewards = dt_rollout(dt, env, target_return=30.0, max_steps=20)
    assert total > 0
    assert len(rewards) > 0


# -------------------- evaluate_policy --------------------
def test_evaluate_policy_with_random():
    env = CartPoleLite(max_steps=10, seed=0)
    rp = random_policy_factory(env.nA, seed=1)
    result = evaluate_policy(rp, env, n_episodes=3, seed=42)
    assert "mean" in result and "std" in result
    assert len(result["rewards"]) == 3
    assert result["mean"] > 0  # 每步奖励 +1
