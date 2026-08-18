"""utils/grpo.py 冒烟测试（Ch13 基础设施）。"""
import pytest

torch = pytest.importorskip("torch")

from rlenvs import CharTokenizer, build_tiny_gpt
from utils.reward_model import RewardModel
from utils.grpo import (
    GRPOConfig,
    GRPOTrainer,
    compute_group_advantages,
)


# -------------------- compute_group_advantages --------------------
def test_group_advantages_zero_mean_unit_std():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = compute_group_advantages(rewards)
    # 标准化后均值 ≈ 0, 标准差 = 1
    assert abs(float(adv.mean())) < 1e-6
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-5


def test_group_advantages_batched():
    # [B=2, G=4]: 每行单独标准化
    rewards = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],   # std=0 → 除 (0 + eps) → 全 0
        [0.0, 0.0, 10.0, 10.0],
    ])
    adv = compute_group_advantages(rewards)
    # 第一行 std=0 → adv 全 0（eps 保护）
    assert torch.allclose(adv[0], torch.zeros(4), atol=1e-5)
    # 第二行均值 5、std=5；adv = (r-5)/5
    assert torch.allclose(adv[1], torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=1e-5)


def test_group_advantages_no_critic_dependency():
    # 函数本身不依赖任何神经网络：纯数学
    rewards = torch.randn(8)
    adv = compute_group_advantages(rewards)
    assert adv.shape == rewards.shape
    # 单调性：reward 大 → advantage 大
    sorted_r = rewards.argsort()
    sorted_a = adv[sorted_r]
    assert (sorted_a[1:] >= sorted_a[:-1] - 1e-6).all()


# -------------------- GRPOTrainer 端到端冒烟 --------------------
def _build_three_models(tok, d_model=16):
    """构造 actor / reference / reward_model（注意：没有 critic）。"""
    class ActorWrap(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, ids):
            return self.backbone(ids)

    def make_bb():
        return build_tiny_gpt(
            vocab_size=tok.vocab_size, d_model=d_model, n_heads=2,
            n_layers=1, d_ff=32, max_seq_len=64,
        )

    actor = ActorWrap(make_bb())
    reference = ActorWrap(make_bb())
    reference.load_state_dict(actor.state_dict())
    reward_model = RewardModel(make_bb())
    return actor, reference, reward_model


def test_grpo_trainer_no_critic_attribute():
    """核心承诺：GRPOTrainer 不应该有 critic / critic_opt / value_coef。"""
    torch.manual_seed(0)
    tok = CharTokenizer().train(
        "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    )
    actor, reference, reward_model = _build_three_models(tok)
    cfg = GRPOConfig(group_size=4, response_max_len=4, beta=0.05,
                     update_epochs=1, inner_minibatch_size=2, target_kl=None)
    trainer = GRPOTrainer(
        actor, reward_model, reference, pad_id=tok.pad_id, cfg=cfg,
    )

    # 关键：没有 critic / critic_opt / value head
    assert not hasattr(trainer, "critic")
    assert not hasattr(trainer, "critic_opt")
    # GRPOConfig 也不应该有 critic_lr / value_coef / gamma / lam
    forbidden_cfg_keys = {"critic_lr", "value_coef", "gamma", "lam"}
    assert forbidden_cfg_keys.isdisjoint(set(cfg.__dict__.keys())), (
        f"GRPOConfig 不应该有 critic 相关字段: "
        f"{forbidden_cfg_keys & set(cfg.__dict__.keys())}"
    )


def test_grpo_rollout_group_shape():
    torch.manual_seed(0)
    tok = CharTokenizer().train(
        "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    )
    actor, reference, reward_model = _build_three_models(tok)

    G = 4
    cfg = GRPOConfig(group_size=G, response_max_len=4, beta=0.05,
                     update_epochs=1, inner_minibatch_size=2, target_kl=None)
    trainer = GRPOTrainer(
        actor, reward_model, reference, pad_id=tok.pad_id, cfg=cfg,
    )

    prompts = [tok.encode("Q: How A:"), tok.encode("Q: What A:")]
    rollout = trainer.rollout_group(prompts)

    N = len(prompts) * G
    assert rollout["prompts"].size(0) == N
    assert rollout["responses"].size(0) == N
    assert rollout["log_probs_old"].shape == rollout["responses"].shape
    assert rollout["log_probs_ref"].shape == rollout["responses"].shape
    # **关键：没有 values_old**（没有 critic）
    assert "values_old" not in rollout
    # group_ids: 每个 prompt 一个 group
    assert rollout["group_ids"].size(0) == N
    # 两个 group, 每个 G 个样本
    unique_groups = torch.unique(rollout["group_ids"])
    assert unique_groups.numel() == 2
    for g in unique_groups:
        assert int((rollout["group_ids"] == g).sum()) == G


def test_grpo_update_runs_and_produces_metrics_no_critic_loss():
    torch.manual_seed(0)
    tok = CharTokenizer().train(
        "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    )
    actor, reference, reward_model = _build_three_models(tok)

    cfg = GRPOConfig(group_size=4, response_max_len=4, beta=0.05,
                     update_epochs=2, inner_minibatch_size=2, target_kl=None,
                     actor_lr=1e-4)
    trainer = GRPOTrainer(
        actor, reward_model, reference, pad_id=tok.pad_id, cfg=cfg,
    )

    prompts = [tok.encode("Q: How A:"), tok.encode("Q: What A:"),
               tok.encode("Q: Why A:"), tok.encode("Q: Yes A:")]
    rollout = trainer.rollout_group(prompts)
    stats = trainer.grpo_update(rollout)

    for k in ("actor_loss", "entropy", "approx_kl",
              "clip_fraction", "n_epochs_done", "mean_reward", "mean_kl_to_ref",
              "mean_abs_advantage"):
        assert k in stats, f"missing metric {k}"
        assert isinstance(stats[k], float)
    # **核心承诺：没有 critic_loss 字段**
    assert "critic_loss" not in stats, (
        "GRPO stats 不应该有 critic_loss —— 兑现 '去掉 value function' 承诺"
    )
    assert stats["n_epochs_done"] == float(cfg.update_epochs)


def test_grpo_train_loop_smoke():
    """End-to-end: train() should not raise and produce n_iters history entries."""
    torch.manual_seed(0)
    tok = CharTokenizer().train(
        "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    )
    actor, reference, reward_model = _build_three_models(tok)

    cfg = GRPOConfig(group_size=4, response_max_len=4, beta=0.05,
                     update_epochs=1, inner_minibatch_size=4, target_kl=None,
                     actor_lr=1e-4, print_every=99)
    trainer = GRPOTrainer(
        actor, reward_model, reference, pad_id=tok.pad_id, cfg=cfg,
    )
    prompts = [tok.encode("Q: How A:"), tok.encode("Q: What A:")]
    history = trainer.train(prompts, n_iters=3, n_prompts_per_iter=2, verbose=False)
    assert len(history) == 3
