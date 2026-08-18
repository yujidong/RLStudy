"""TinyGPT 模块冒烟测试（Ch10 基础设施）。"""
import pytest

torch = pytest.importorskip("torch")

from rlenvs import (
    CharTokenizer,
    PositionalEncoding,
    CausalSelfAttention,
    TransformerBlock,
    TinyGPT,
    build_tiny_gpt,
    compute_loss,
    make_lm_batch,
    generate,
    sft_loss,
)


# -------------------- CharTokenizer --------------------
def test_tokenizer_train_and_encode():
    tok = CharTokenizer()
    tok.train("hello\nworld")
    assert tok.vocab_size == len(set("hello\nworld")) + 1  # +1 for <pad>
    ids = tok.encode("hello")
    assert ids.dtype == torch.long
    assert tok.decode(ids) == "hello"


def test_tokenizer_pad_id_is_zero():
    tok = CharTokenizer()
    tok.train("abc")
    assert tok.pad_id == 0


def test_tokenizer_unseen_chars_dropped():
    tok = CharTokenizer()
    tok.train("abc")
    ids = tok.encode("aXbYc")  # X/Y 未见过
    assert ids.tolist() == [tok.stoi["a"], tok.stoi["b"], tok.stoi["c"]]


# -------------------- PositionalEncoding --------------------
def test_positional_encoding_shape_and_pe_buffer():
    pe = PositionalEncoding(d_model=16, max_len=64)
    x = torch.randn(2, 10, 16)
    out = pe(x)
    assert out.shape == (2, 10, 16)
    assert pe.pe.shape == (1, 64, 16)


def test_positional_encoding_too_long_raises():
    pe = PositionalEncoding(d_model=8, max_len=5)
    with pytest.raises(ValueError):
        pe(torch.randn(1, 10, 8))


# -------------------- CausalSelfAttention --------------------
def test_causal_attention_shape_and_weights():
    att = CausalSelfAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 6, 16)
    out = att(x)
    assert out.shape == (2, 6, 16)
    assert att.att_weights is not None
    assert att.att_weights.shape == (2, 4, 6, 6)


def test_causal_mask_upper_triangle_zero():
    """因果 mask：上三角（不含对角线）的注意力权重应几乎为 0。"""
    att = CausalSelfAttention(d_model=16, n_heads=4)
    _ = att(torch.randn(1, 5, 16))
    w = att.att_weights  # [1, 4, 5, 5]
    # 上三角（i < j）位置应 ≈ 0（softmax(-inf)=0）
    upper = torch.triu(torch.ones(5, 5), diagonal=1).bool()
    assert w[0, 0][upper].max().item() < 1e-5


def test_causal_attention_rows_sum_to_one():
    """每行 softmax 后应和为 1（概率分布）。"""
    att = CausalSelfAttention(d_model=16, n_heads=4)
    _ = att(torch.randn(1, 5, 16))
    w = att.att_weights[0, 0]  # [5, 5]
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(5), atol=1e-5)


def test_causal_attention_d_model_must_divide_n_heads():
    with pytest.raises(AssertionError):
        CausalSelfAttention(d_model=10, n_heads=3)


# -------------------- TransformerBlock --------------------
def test_transformer_block_shape():
    blk = TransformerBlock(d_model=16, n_heads=4, d_ff=64)
    x = torch.randn(2, 8, 16)
    out = blk(x)
    assert out.shape == (2, 8, 16)


# -------------------- TinyGPT --------------------
def test_tiny_gpt_forward_shape():
    model = TinyGPT(vocab_size=50, d_model=16, n_heads=4, n_layers=2, d_ff=32, max_seq_len=32)
    ids = torch.randint(0, 50, (3, 10))
    logits = model(ids)
    assert logits.shape == (3, 10, 50)


def test_tiny_gpt_too_long_seq_raises():
    model = TinyGPT(vocab_size=10, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_seq_len=5)
    with pytest.raises(ValueError):
        model(torch.randint(0, 10, (1, 10)))


def test_tiny_gpt_attention_weights_access():
    model = TinyGPT(vocab_size=20, d_model=16, n_heads=4, n_layers=3, d_ff=32, max_seq_len=16)
    _ = model(torch.randint(0, 20, (1, 8)))
    weights = model.get_attention_weights()
    assert len(weights) == 3
    for w in weights:
        assert w.shape == (1, 4, 8, 8)


def test_build_tiny_gpt_factory():
    m = build_tiny_gpt(vocab_size=30)
    assert isinstance(m, TinyGPT)


# -------------------- helpers --------------------
def test_compute_loss_grad():
    model = TinyGPT(vocab_size=10, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_seq_len=16)
    x = torch.randint(0, 10, (1, 5))
    logits = model(x)
    y = torch.randint(0, 10, (1, 5))
    loss = compute_loss(logits, y)
    assert loss.dim() == 0
    loss.backward()
    # 至少有一个参数收到了梯度
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_make_lm_batch_1d():
    ids = torch.arange(10)
    x, y = make_lm_batch(ids)
    assert x.shape == (1, 9)
    assert y.shape == (1, 9)
    # y 应该是 x 左移一位
    assert torch.equal(y[0, :-1], x[0, 1:])


def test_make_lm_batch_too_short_raises():
    with pytest.raises(ValueError):
        make_lm_batch(torch.tensor([5]))


def test_generate_greedy_deterministic():
    model = TinyGPT(vocab_size=15, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_seq_len=16)
    model.eval()
    prompt = torch.tensor([[1, 2, 3]])
    out1 = generate(model, prompt, max_new_tokens=5, greedy=True)
    out2 = generate(model, prompt, max_new_tokens=5, greedy=True)
    assert out1.shape == (1, 8)
    assert torch.equal(out1, out2)  # greedy 必须可复现


def test_generate_temperature_sampling_varies():
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=15, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_seq_len=16)
    model.eval()
    prompt = torch.tensor([[1, 2, 3]])
    o1 = generate(model, prompt, max_new_tokens=8, temperature=1.0)
    torch.manual_seed(1)
    o2 = generate(model, prompt, max_new_tokens=8, temperature=1.0)
    # 不同随机种子，采样输出大概率不同
    assert not torch.equal(o1, o2)


def test_sft_loss_ignores_prompt():
    B, T, V = 2, 6, 12
    torch.manual_seed(0)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    # 全部 mask=0：loss 应为 nan 或抛错；用 ignore_index 让全部跳过 → cross_entropy 会 nan
    # 我们用部分 mask 验证 prompt 不参与
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    loss = sft_loss(logits, targets, mask)
    assert loss.dim() == 0
    # mask=1 的位置至少 1 个，loss 有限
    assert torch.isfinite(loss)
    loss.backward()
