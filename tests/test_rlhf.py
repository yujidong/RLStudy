"""utils/rlhf.py 冒烟测试（Ch12 基础设施）。"""
import pytest

torch = pytest.importorskip("torch")

from rlenvs import CharTokenizer, build_tiny_gpt
from utils.reward_model import RewardModel
from utils.rlhf import RLHFConfig, RLHFTrainer, ValueHead


# -------------------- ValueHead --------------------
def test_value_head_per_token_shape():
    tok = CharTokenizer().train("Q: A: good bad ok" + "abcdefghijklmnopqrstuvwxyz ")
    backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32
    )
    vh = ValueHead(backbone, d_model=16)
    x = tok.encode("Q: A: good").unsqueeze(0)  # [1, T]
    out = vh(x)
    assert out.shape == (1, x.size(1))
    assert out.dtype == torch.float32


# -------------------- RLHFTrainer end-to-end smoke --------------------
def test_rlhf_trainer_rollout_shape():
    torch.manual_seed(0)
    tok = CharTokenizer().train(
        "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    )

    class ActorWrap(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, ids):
            return self.backbone(ids)

    ref_backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
    )
    actor_backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
    )
    critic_backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
    )
    rm_backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
    )

    actor = ActorWrap(actor_backbone)
    reference = ActorWrap(ref_backbone)
    reference.load_state_dict(actor.state_dict())
    critic = ValueHead(critic_backbone, d_model=16)
    reward_model = RewardModel(rm_backbone)

    cfg = RLHFConfig(response_max_len=4, beta=0.05, update_epochs=1,
                     inner_minibatch_size=2, target_kl=None)
    trainer = RLHFTrainer(
        actor, critic, reward_model, reference, pad_id=tok.pad_id, cfg=cfg
    )

    prompts = [tok.encode("Q: How A:"), tok.encode("Q: What A:"),
               tok.encode("Q: Why A:"), tok.encode("Q: Yes A:")]
    rollout = trainer.rollout_responses(prompts)

    B = len(prompts)
    assert rollout["prompts"].size(0) == B
    assert rollout["responses"].size(0) == B
    assert rollout["log_probs_old"].shape == rollout["responses"].shape
    assert rollout["log_probs_ref"].shape == rollout["responses"].shape
    assert rollout["values_old"].shape == rollout["responses"].shape
    # KL to ref should be ~0 initially (actor == reference)
    mask = rollout["response_mask"]
    kl = ((rollout["log_probs_old"] - rollout["log_probs_ref"]) * mask).sum() / mask.sum().clamp(min=1)
    assert abs(float(kl)) < 1e-3


def test_rlhf_trainer_update_runs_and_produces_metrics():
    torch.manual_seed(0)
    tok = CharTokenizer().train(
        "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    )

    class ActorWrap(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, ids):
            return self.backbone(ids)

    def make_bb():
        return build_tiny_gpt(
            vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
        )

    actor = ActorWrap(make_bb())
    reference = ActorWrap(make_bb())
    reference.load_state_dict(actor.state_dict())
    critic = ValueHead(make_bb(), d_model=16)
    reward_model = RewardModel(make_bb())

    cfg = RLHFConfig(response_max_len=4, beta=0.05, update_epochs=2,
                     inner_minibatch_size=2, target_kl=None,
                     actor_lr=1e-4, critic_lr=1e-3)
    trainer = RLHFTrainer(
        actor, critic, reward_model, reference, pad_id=tok.pad_id, cfg=cfg
    )

    prompts = [tok.encode("Q: How A:"), tok.encode("Q: What A:"),
               tok.encode("Q: Why A:"), tok.encode("Q: Yes A:")]
    rollout = trainer.rollout_responses(prompts)
    stats = trainer.rlhf_update(rollout)

    for k in ("actor_loss", "critic_loss", "entropy", "approx_kl",
              "clip_fraction", "n_epochs_done", "mean_reward", "mean_kl_to_ref"):
        assert k in stats, f"missing metric {k}"
        assert isinstance(stats[k], float)
    assert stats["n_epochs_done"] == float(cfg.update_epochs)


def test_rlhf_train_loop_smoke():
    """End-to-end: train() should not raise and produce n_iters history entries."""
    torch.manual_seed(0)
    tok = CharTokenizer().train(
        "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    )

    class ActorWrap(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, ids):
            return self.backbone(ids)

    def make_bb():
        return build_tiny_gpt(
            vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
        )

    actor = ActorWrap(make_bb())
    reference = ActorWrap(make_bb())
    reference.load_state_dict(actor.state_dict())
    critic = ValueHead(make_bb(), d_model=16)
    reward_model = RewardModel(make_bb())

    cfg = RLHFConfig(response_max_len=4, beta=0.05, update_epochs=1,
                     inner_minibatch_size=4, target_kl=None,
                     actor_lr=1e-4, critic_lr=1e-3, print_every=99)
    trainer = RLHFTrainer(
        actor, critic, reward_model, reference, pad_id=tok.pad_id, cfg=cfg
    )
    prompts = [tok.encode("Q: How A:"), tok.encode("Q: What A:")]
    history = trainer.train(prompts, n_iters=3, group_size=4, verbose=False)
    assert len(history) == 3
