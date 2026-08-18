"""utils/dpo.py 冒烟测试（Ch14 基础设施）。"""
import pytest

torch = pytest.importorskip("torch")

import copy

from rlenvs import CharTokenizer, build_tiny_gpt
from utils.reward_model import generate_preference_data
from utils.dpo import (
    DPOConfig,
    DPOTrainer,
    KTOTrainer,
    dpo_loss,
    kto_loss,
    prospect_value,
    sequence_log_probs,
)


# -------------------- helpers --------------------
def _build_two_models(tok, d_model=16):
    """构造 actor / reference（注意：没有 reward_model，没有 critic）。"""
    class ActorWrap(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.d_model = backbone.d_model

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
    return actor, reference


def _make_tok():
    return CharTokenizer().train(
        "Q: How is the weather? A: good bad ok fine yes no it day way "
        "hello world great very "
    )


# -------------------- sequence_log_probs --------------------
def test_sequence_log_probs_returns_shape_and_finite():
    torch.manual_seed(0)
    tok = _make_tok()
    actor, _ = _build_two_models(tok)
    p = tok.encode("Q: How A:").unsqueeze(0)
    r = tok.encode("good").unsqueeze(0)
    logp = sequence_log_probs(actor, p, r, pad_id=tok.pad_id)
    assert logp.shape == (1,)
    assert torch.isfinite(logp).all()
    # log prob of any sequence under softmax should be <= 0
    assert logp.item() <= 0.0


def test_sequence_log_probs_handles_padding():
    """batch 内不同长度样本，padding 不应影响 log prob。"""
    torch.manual_seed(0)
    tok = _make_tok()
    actor, _ = _build_two_models(tok)
    p1 = tok.encode("Q: How A:")
    p2 = tok.encode("Q: Yes A:")
    r1 = tok.encode("good")
    r2 = tok.encode("bad")
    from utils.reward_model import pad_to_length
    p_batch = pad_to_length([p1, p2], tok.pad_id)  # [2, max]
    r_batch = pad_to_length([r1, r2], tok.pad_id)
    logp = sequence_log_probs(actor, p_batch, r_batch, pad_id=tok.pad_id)
    # 单独算应该和 batch 算一致（因为逐样本 forward）
    logp1_solo = sequence_log_probs(actor, p1.unsqueeze(0), r1.unsqueeze(0), tok.pad_id)
    logp2_solo = sequence_log_probs(actor, p2.unsqueeze(0), r2.unsqueeze(0), tok.pad_id)
    assert torch.allclose(logp[0], logp1_solo[0], atol=1e-5)
    assert torch.allclose(logp[1], logp2_solo[0], atol=1e-5)


# -------------------- dpo_loss --------------------
def test_dpo_loss_at_init_is_log2():
    """actor == reference 时，Δ_w = Δ_l = 0，loss = softplus(0) = log(2)。"""
    torch.manual_seed(0)
    logp = torch.zeros(8)  # actor 和 ref 都给 logp=0 → delta=0
    loss, stats = dpo_loss(logp, logp, logp, logp, beta=0.1)
    import math
    assert abs(loss.item() - math.log(2)) < 1e-5
    assert abs(stats["reward_margin"]) < 1e-5
    # accuracy when margin == 0: (margin > 0) is all False → 0
    assert stats["reward_accuracy"] == 0.0


def test_dpo_loss_decreases_when_actor_prefers_winner():
    """如果 actor 给 winner 高 logp、loser 低 logp（相比 ref），loss 应该小。"""
    torch.manual_seed(0)
    # ref logp 都是 0；actor winner=+5, loser=-5
    actor_w = torch.tensor([5.0, 5.0, 5.0])
    actor_l = torch.tensor([-5.0, -5.0, -5.0])
    ref = torch.zeros(3)
    loss, stats = dpo_loss(actor_w, actor_l, ref, ref, beta=1.0)
    # margin = β(Δ_w - Δ_l) = 1 * (5 - (-5)) = 10 → softplus(-10) ≈ 0
    assert loss.item() < 1e-3
    assert stats["reward_accuracy"] == 1.0


def test_dpo_loss_gradient_flows_to_actor_only():
    """DPO loss 反传时，actor_logp 有梯度，ref_logp 不应有。"""
    actor_w = torch.zeros(4, requires_grad=True)
    actor_l = torch.zeros(4, requires_grad=True)
    ref_w = torch.zeros(4)  # no requires_grad
    ref_l = torch.zeros(4)
    loss, _ = dpo_loss(actor_w, actor_l, ref_w, ref_l, beta=0.1)
    loss.backward()
    assert actor_w.grad is not None
    assert actor_l.grad is not None
    # ref tensors 不应该有梯度
    assert ref_w.grad is None
    assert ref_l.grad is None


# -------------------- prospect_value + kto_loss --------------------
def test_prospect_value_loss_aversion():
    """|loss| > |gain| 当 lambda > 1（loss aversion）。"""
    x_gain = torch.tensor([1.0])
    x_loss = torch.tensor([-1.0])
    v = prospect_value(x_gain, lambda_aversion=2.25, gamma_gain=1.0, gamma_loss=1.0)
    v_l = prospect_value(x_loss, lambda_aversion=2.25, gamma_gain=1.0, gamma_loss=1.0)
    assert abs(v.item() - 1.0) < 1e-5  # gain = +1
    assert abs(v_l.item() + 2.25) < 1e-5  # loss = -lambda


def test_kto_loss_shape():
    logp_a = torch.tensor([0.5, -0.5, 0.3, -0.2], requires_grad=True)
    logp_r = torch.zeros(4)
    is_good = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss, stats = kto_loss(logp_a, logp_r, is_good, beta=0.5,
                           desirable_weight=1.0, undesirable_weight=1.5, tau=0.0)
    assert loss.dim() == 0
    assert "good_points" in stats and "bad_points" in stats
    loss.backward()
    assert logp_a.grad is not None


# -------------------- DPOTrainer 端到端冒烟 --------------------
def test_dpo_trainer_only_two_models():
    """核心承诺：DPOTrainer 只应该有 actor + reference，没有 critic / reward_model。"""
    torch.manual_seed(0)
    tok = _make_tok()
    actor, reference = _build_two_models(tok)
    cfg = DPOConfig(beta=0.1, batch_size=4, eval_every=10, print_every=100,
                    actor_lr=1e-3)
    trainer = DPOTrainer(actor, reference, pad_id=tok.pad_id, cfg=cfg)

    assert trainer.actor is actor
    assert trainer.reference is reference
    assert not hasattr(trainer, "critic")
    assert not hasattr(trainer, "critic_opt")
    assert not hasattr(trainer, "reward_model")
    # 没有这些 PPO 字段
    forbidden = {"clip_eps", "update_epochs", "target_kl", "gamma", "lam",
                 "critic_lr", "value_coef", "entropy_coef"}
    assert forbidden.isdisjoint(set(cfg.__dict__.keys())), (
        f"DPOConfig 不应该有 RLHF/PPO 字段: "
        f"{forbidden & set(cfg.__dict__.keys())}"
    )


def test_dpo_trainer_reference_is_frozen():
    torch.manual_seed(0)
    tok = _make_tok()
    actor, reference = _build_two_models(tok)
    cfg = DPOConfig(beta=0.1, batch_size=4, actor_lr=1e-3)
    trainer = DPOTrainer(actor, reference, pad_id=tok.pad_id, cfg=cfg)
    for p in trainer.reference.parameters():
        assert not p.requires_grad


def test_dpo_train_loop_runs_and_margin_grows():
    """端到端：DPO 训练循环跑得动，reward_margin 应该 > 0（actor 学到偏好）。"""
    torch.manual_seed(0)
    tok = _make_tok()
    actor, reference = _build_two_models(tok)
    prefs = generate_preference_data(tok, n_samples=30, seed=0)
    cfg = DPOConfig(beta=0.1, batch_size=8, actor_lr=5e-4,
                    eval_every=1000, print_every=1000)
    trainer = DPOTrainer(actor, reference, pad_id=tok.pad_id, cfg=cfg)
    trainer.train(prefs, n_iters=30, verbose=False)
    assert len(trainer.history) == 30
    # 第一步 margin 应该接近 0（actor == ref），最后一步应该 > 0
    assert trainer.history[0]["reward_margin"] < trainer.history[-1]["reward_margin"]
    assert trainer.history[-1]["reward_margin"] > 0


def test_dpo_evaluate_with_external_rm():
    """evaluate() 可以接受外部 RewardModel，返回 RM-based 指标。"""
    torch.manual_seed(0)
    tok = _make_tok()
    actor, reference = _build_two_models(tok)
    from utils.reward_model import RewardModel
    rm_backbone = build_tiny_gpt(vocab_size=tok.vocab_size, d_model=16,
                                  n_heads=2, n_layers=1, d_ff=32, max_seq_len=64)
    rm = RewardModel(rm_backbone)
    prefs = generate_preference_data(tok, n_samples=20, seed=0)
    cfg = DPOConfig(beta=0.1, batch_size=4, actor_lr=1e-3)
    trainer = DPOTrainer(actor, reference, pad_id=tok.pad_id, cfg=cfg)
    stats = trainer.evaluate(prefs, val_reward_model=rm)
    for key in ("val_dpo_acc", "val_dpo_margin", "val_rm_mean_reward_w",
                "val_rm_mean_reward_l", "val_rm_margin", "val_rm_acc"):
        assert key in stats


# -------------------- KTOTrainer 冒烟 --------------------
def test_kto_trainer_runs():
    torch.manual_seed(0)
    tok = _make_tok()
    actor, reference = _build_two_models(tok)
    prefs = generate_preference_data(tok, n_samples=20, seed=0)
    cfg = DPOConfig(beta=0.1, batch_size=8, actor_lr=5e-4,
                    print_every=1000, kto_undesirable_weight=1.5)
    trainer = KTOTrainer(actor, reference, pad_id=tok.pad_id, cfg=cfg)
    trainer.train(prefs, n_iters=10, verbose=False)
    assert len(trainer.history) == 10
    last = trainer.history[-1]
    for k in ("kto_loss", "good_points", "bad_points", "kto_accuracy"):
        assert k in last
