"""utils/reward_model.py 冒烟测试（Ch11 基础设施）。"""
import pytest

torch = pytest.importorskip("torch")

from rlenvs import CharTokenizer, build_tiny_gpt
from utils.reward_model import (
    RewardModel,
    bradley_terry_loss,
    generate_preference_data,
    make_preference_batch,
    pad_to_length,
    predict_rewards,
    reward_accuracy,
    true_reward,
)


# -------------------- true_reward / generate_preference_data --------------------
def test_true_reward_keyword_and_length():
    # 'good' 给加分；'bad' 给扣分
    r_good = true_reward("", "very good", keyword_weight=2.0, len_weight=0.5)
    r_bad = true_reward("", "it is bad", keyword_weight=2.0, len_weight=0.5)
    assert r_good > 0
    assert r_bad < 0
    assert r_good > r_bad


def test_generate_preference_data_basic():
    tok = CharTokenizer().train(
        "Q: How are you? A: good bad ok fine yes no it day way hello world great very"
    )
    data = generate_preference_data(tok, n_samples=20, seed=0)
    assert len(data) > 0
    for s in data:
        # winner 的 ground truth reward 必须严格大于 loser
        assert s["r_w"] > s["r_l"]
        assert s["r_diff"] > 0
        # 所有字段都存在
        for k in ("prompt", "winner", "loser", "prompt_ids", "winner_ids", "loser_ids"):
            assert k in s
        assert s["prompt_ids"].dtype == torch.long


def test_generate_preference_data_reproducible():
    tok = CharTokenizer().train("Q: A: good bad ok fine yes no " + "abcdefghijklmnopqrstuvwxyz ")
    d1 = generate_preference_data(tok, n_samples=10, seed=7)
    d2 = generate_preference_data(tok, n_samples=10, seed=7)
    assert len(d1) == len(d2)
    for a, b in zip(d1, d2):
        assert a["prompt"] == b["prompt"]
        assert a["winner"] == b["winner"]
        assert a["loser"] == b["loser"]


# -------------------- pad_to_length / make_preference_batch --------------------
def test_pad_to_length_shapes():
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])]
    out = pad_to_length(seqs, pad_id=0)
    assert out.shape == (3, 4)
    assert out.dtype == torch.long
    # 短的尾部应填 pad
    assert out[1, 2].item() == 0 and out[1, 3].item() == 0


def test_pad_to_length_max_len_truncates():
    seqs = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2])]
    out = pad_to_length(seqs, pad_id=0, max_len=3)
    assert out.shape == (2, 3)
    assert out[0].tolist() == [1, 2, 3]  # 截断


def test_make_preference_batch():
    tok = CharTokenizer().train("Q: A: good bad ok")
    data = generate_preference_data(tok, n_samples=5, seed=0)
    b = make_preference_batch(data, pad_id=tok.pad_id)
    n = len(data)
    assert b["prompt_ids"].shape[0] == n
    assert b["winner_ids"].shape[0] == n
    assert b["loser_ids"].shape[0] == n
    assert b["r_diff"].shape == (n,)
    assert (b["r_diff"] > 0).all()


# -------------------- RewardModel --------------------
def test_reward_model_forward_shape():
    tok = CharTokenizer().train("Q: A: good bad ok" + "abcdefghijklmnopqrstuvwxyz ")
    backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32
    )
    rm = RewardModel(backbone)
    p = tok.encode("Q: A:").unsqueeze(0)
    r = tok.encode("good").unsqueeze(0)
    out = rm(p, r)
    assert out.shape == (1,)
    assert out.dtype == torch.float32


def test_reward_model_batch_forward():
    tok = CharTokenizer().train("Q: A: good bad ok" + "abcdefghijklmnopqrstuvwxyz ")
    backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32
    )
    rm = RewardModel(backbone)
    data = generate_preference_data(tok, n_samples=8, seed=1)
    b = make_preference_batch(data, pad_id=tok.pad_id)
    r_w = rm(b["prompt_ids"], b["winner_ids"])
    r_l = rm(b["prompt_ids"], b["loser_ids"])
    assert r_w.shape == (8,) and r_l.shape == (8,)


# -------------------- bradley_terry_loss --------------------
def test_bradley_terry_loss_scalar_and_grad():
    tok = CharTokenizer().train("Q: A: good bad ok" + "abcdefghijklmnopqrstuvwxyz ")
    backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32
    )
    rm = RewardModel(backbone)
    data = generate_preference_data(tok, n_samples=4, seed=0)
    b = make_preference_batch(data, pad_id=tok.pad_id)
    loss = bradley_terry_loss(rm, b["prompt_ids"], b["winner_ids"], b["loser_ids"])
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # 初始 loss 在 log(2) ≈ 0.69 附近（reward model 还没训；具体值随 init 有方差）
    assert 0.3 < float(loss.detach()) < 1.5
    loss.backward()
    assert rm.reward_head.weight.grad is not None


def test_bradley_terry_loss_is_softplus():
    """验证 -log sigma(r_w - r_l) == softplus(r_l - r_w) 数值上等价。"""
    import torch.nn.functional as F

    torch.manual_seed(0)
    rw = torch.tensor([1.5, -2.0, 0.3])
    rl = torch.tensor([-0.5, 1.0, 0.1])
    direct = -torch.log(torch.sigmoid(rw - rl) + 1e-12).mean()
    softplus = F.softplus(rl - rw).mean()
    assert torch.allclose(direct, softplus, atol=1e-4)


# -------------------- reward_accuracy / predict_rewards --------------------
def test_reward_accuracy_perfect_rm():
    """如果一个 RM 完美排序 winner > loser，accuracy = 1。"""
    import torch.nn as nn

    tok = CharTokenizer().train("Q: A: good bad ok" + "abcdefghijklmnopqrstuvwxyz ")

    class PerfectRM(nn.Module):
        """mock：直接返回 ground-truth reward（winner 严格高于 loser）。"""

        def forward(self, prompt_ids, response_ids):
            rewards = []
            for i in range(prompt_ids.size(0)):
                # 去掉 padding 字符再 decode（保证 mock 真"完美"）
                ids = response_ids[i]
                ids = ids[ids != tok.pad_id]
                resp = tok.decode(ids)
                rewards.append(true_reward("", resp))
            return torch.tensor(rewards, dtype=torch.float32)

    data = generate_preference_data(tok, n_samples=10, seed=0)
    rm = PerfectRM()
    acc = reward_accuracy(rm, data, pad_id=tok.pad_id)
    # 现在 mock 是真"完美"：winner 严格高于 loser
    assert acc == 1.0


def test_predict_rewards_shape():
    tok = CharTokenizer().train("Q: A: good bad ok" + "abcdefghijklmnopqrstuvwxyz ")
    backbone = build_tiny_gpt(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32
    )
    rm = RewardModel(backbone)
    data = generate_preference_data(tok, n_samples=10, seed=0)
    out = predict_rewards(rm, data, pad_id=tok.pad_id, which="winner")
    assert out.shape == (10,)
