"""TinyGPT —— 一个 ~1M 参数的 mini-GPT，纯 PyTorch 手写实现（Ch10）。

本章是 Phase 3 的基础设施：给 Ch11（Reward Modeling）、Ch12（RLHF-PPO）、
Ch13（GRPO）提供一个**能在 CPU 上训得动**的小型 decoder-only Transformer。

设计原则（与项目其它模块一致）：
- **教学优先**：每个组件（attention、block、LM head）都拆成独立可读的类，
  内部状态（注意力权重）可访问——这是 Ch00 承诺的"注意力热力图"的基础。
- **从零实现**：不使用 ``torch.nn.Transformer`` / HuggingFace。所有 attention、
  positional encoding、causal mask 手写。
- **小而能学**：vocab ~ 100、d_model=64、4 heads、4-6 layers → ~100k-1M 参数，
  CPU 上 5000 步内能学到 char-level pattern。

模块组成：
- :class:`CharTokenizer` —— char-level tokenizer（encode/decode + vocab 持久化）
- :class:`PositionalEncoding` —— sinusoidal（与 Attention is All You Need 一致）
- :class:`CausalSelfAttention` —— 多头带 causal mask，暴露 ``att_weights``
- :class:`TransformerBlock` —— Pre-LN：attention + FFN + 双残差 + 双 LayerNorm
- :class:`TinyGPT` —— embedding + N×block + LayerNorm + LM head
- :func:`build_tiny_gpt` —— 项目默认配置工厂函数
- :func:`compute_loss` —— teacher-forcing 下的 cross-entropy loss
- :func:`generate` —— 支持 greedy / temperature / top-k 的自回归采样
- :func:`sft_loss` —— SFT（条件 LM）的 loss：只在 response token 上算 cross-entropy

forward 的输出约定：
    ``TinyGPT.forward(input_ids) -> logits``，shape ``[batch, seq_len, vocab_size]``。
    注意力权重在每个 block 的 ``block.att.att_weights`` 里（每次 forward 后更新），
    形状 ``[batch, n_heads, seq_len, seq_len]``。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Tokenizer（char-level）
# =============================================================================
class CharTokenizer:
    """字符级 tokenizer —— 简单可靠，适合教学。

    把每个字符当作一个 token。vocab 由训练语料自动构建：
    ``train(corpus_text)`` 扫一遍文本收集所有出现过的字符。

    API：
        - ``train(text)``：从文本构建词表
        - ``encode(text) -> torch.LongTensor [seq_len]``
        - ``decode(ids) -> str``
        - 属性 ``vocab_size``、``stoi``、``itos``

    特殊 token：``<pad>``（id=0，padding 用，但本章训练不依赖它）。
    """

    PAD_TOKEN = "<pad>"

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []

    def train(self, text: str) -> "CharTokenizer":
        """从文本构建词表。保证可复现：按字符首次出现顺序排序。"""
        # 特殊 token 先入表
        chars: list[str] = [self.PAD_TOKEN]
        seen = {self.PAD_TOKEN}
        for ch in text:
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD_TOKEN]

    def encode(self, text: str) -> torch.Tensor:
        """文本 → LongTensor（id 序列）。未见字符跳过（char-level 几乎不会遇到）。"""
        ids = [self.stoi[c] for c in text if c in self.stoi]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Sequence[int] | torch.Tensor) -> str:
        """id 序列 → 字符串。"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


# =============================================================================
# 2. Positional Encoding（sinusoidal）
# =============================================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding（Vaswani et al. 2017）。

    对位置 ``pos`` 和维度 ``i``：
        PE(pos, 2k)   = sin(pos / 10000^{2k / d_model})
        PE(pos, 2k+1) = cos(pos / 10000^{2k / d_model})

    作为 buffer 注册（不参与训练，但 ``.to(device)`` 会跟着走）。
    长度上限 ``max_len``，超出会报错。

    forward 输入 shape ``[batch, seq_len, d_model]``，输出 = 输入 + PE[:, :seq_len]。
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 频率分母：10000^{2k / d_model}
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape [1, max_len, d_model]，便于广播
        self.register_buffer("pe", pe.unsqueeze(0))
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model] → x + PE[:, :seq_len]。"""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} > max_len={self.max_len}，请增大 max_len"
            )
        return x + self.pe[:, :seq_len, :]


# =============================================================================
# 3. Multi-Head Causal Self-Attention
# =============================================================================
class CausalSelfAttention(nn.Module):
    """多头因果自注意力（decoder-only）。

    数学定义（单头、序列长度 T、key 维度 d_k）：

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

        其中 Q = X W_Q, K = X W_K, V = X W_V，X ∈ R^{T × d_model}

    多头：把 d_model 切成 ``n_heads`` 份，每份 d_k = d_model // n_heads，
    各自做一次 attention，最后 concat 起来经线性映射回 d_model。

    Causal mask：用下三角矩阵（含对角线）把"未来 token"的注意力分数设为 -inf，
    保证位置 i 只能看 positions ≤ i（自回归性质）。

    暴露的内部状态：
        - ``att_weights``：最后一次 forward 的注意力概率矩阵，
          shape ``[batch, n_heads, T, T]``——画热力图用。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} 必须能被 n_heads={n_heads} 整除"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Q/K/V 合在一个大 Linear 里（工程上比 3 个小 Linear 快）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 保存最后一次的注意力权重（hook 用）
        self.att_weights: Optional[torch.Tensor] = None
        # causal mask 只依赖 (T, device)，缓存避免每次 forward 重建
        # （generate 时每个新 token 都要 forward 一次，热路径）
        self._mask_cache: dict = {}
        # 初始化
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Xavier-uniform（与 "Attention is All You Need" 推荐一致）
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, T, d_model] → [batch, T, d_model]。"""
        B, T, C = x.shape  # batch, seq_len, d_model
        H, D = self.n_heads, self.d_k

        # 一次算出 Q/K/V，再 reshape 成 [B, H, T, D]
        qkv = self.qkv_proj(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # 注意力分数：[B, H, T, D] × [B, H, D, T] = [B, H, T, T]
        # 缩放因子 1/sqrt(d_k)（防止 softmax 饱和——见 §10.3 推导）
        scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)

        # Causal mask：上三角（不含对角线）置 -inf
        # mask[i, j] = 0 if j <= i else -inf
        key = (T, x.device)
        if key not in self._mask_cache:
            self._mask_cache[key] = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )
        scores = scores.masked_fill(self._mask_cache[key], float("-inf"))

        # softmax 沿最后一维（key 维）
        att = F.softmax(scores, dim=-1)  # [B, H, T, T]
        att = self.dropout(att)
        # 暴露出来给可视化用（detached copy 避免 graph 泄漏）
        self.att_weights = att.detach()

        # 加权求和：[B, H, T, T] × [B, H, T, D] = [B, H, T, D]
        out = att @ v
        # 合并 head 维度：[B, H, T, D] → [B, T, H*D] = [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# =============================================================================
# 4. Transformer Block（Pre-LN：attention + FFN + residual + LayerNorm）
# =============================================================================
class TransformerBlock(nn.Module):
    """一个 Pre-LN Transformer block。

    Pre-LN（Xiong et al. 2020）：

        h = x + Attention(LayerNorm(x))
        out = h + FFN(LayerNorm(h))

    相比 Post-LN（原 Transformer 论文）：

        h = LayerNorm(x + Attention(x))
        out = LayerNorm(h + FFN(h))

    Pre-LN 训练更稳定（梯度能直通底层），是 GPT-2 之后的事实标准。
    本章默认 Pre-LN。

    FFN（Position-wise）：

        FFN(x) = Linear(GELU(Linear(x)))
        中间维度通常是 d_model 的 4 倍。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + 残差
        x = x + self.drop(self.att(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# =============================================================================
# 5. TinyGPT（embedding + N×block + LN + LM head）
# =============================================================================
class TinyGPT(nn.Module):
    """从零搭的 decoder-only mini-GPT。

    结构（forward 流）：

        1. token embedding: input_ids [B, T] → [B, T, d_model]
        2. × sqrt(d_model) scaling（Vaswani 经验，让 embedding 量级匹配 PE）
        3. + sinusoidal positional encoding
        4. dropout
        5. N × TransformerBlock（causal self-attention + FFN，Pre-LN 残差）
        6. final LayerNorm
        7. LM head：Linear(d_model → vocab_size)（与 embedding 权重解耦，教学清晰）

    参数量（默认配置 d_model=64, n_heads=4, n_layers=4, d_ff=256, vocab=100）：
        - token embedding: 100 * 64 = 6.4k
        - 每个 block: qkv+out (2*64*64) + LayerNorm 2*128 + FFN (64*256*2) ≈ 70k
        - 4 个 block ≈ 280k
        - LM head: 64 * 100 = 6.4k
        合计 ≈ 300k 参数（vocab 越大占比越大）。

    访问注意力权重：``model.blocks[i].att.att_weights``（最后一次 forward 后）。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        # LM head：不与 embedding 共享权重（教学清晰，代价是多了 vocab*d_model 参数）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.lm_head.weight)
        # FFN/attention 已在各自类里初始化

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, T] (long) → logits: [B, T, vocab_size]。"""
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"seq_len={T} > max_seq_len={self.max_seq_len}"
            )
        # 1. token embedding + scale + positional encoding
        x = self.tok_emb(input_ids) * (self.d_model ** 0.5)
        x = self.pos_enc(x)
        x = self.drop(x)
        # 2. N × transformer block
        for block in self.blocks:
            x = block(x)
        # 3. final LayerNorm + LM head
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    def get_attention_weights(self) -> List[torch.Tensor]:
        """返回每个 block 最后一次 forward 的注意力权重（list of tensors）。

        每个 tensor shape: [batch, n_heads, T, T]。给热力图可视化用。
        """
        return [b.att.att_weights for b in self.blocks]


# =============================================================================
# 6. 工厂函数 + 训练/采样辅助
# =============================================================================
def build_tiny_gpt(
    vocab_size: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 256,
    max_seq_len: int = 128,
    dropout: float = 0.0,
) -> TinyGPT:
    """项目默认配置工厂函数。"""
    return TinyGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )


def compute_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Teacher-forcing 下的 next-token cross-entropy loss。

    输入：
        logits: [B, T, V]（模型 forward 输出）
        targets: [B, T]（每个位置的下一个 token id）

    标准做法：logits[:, :-1] 预测 targets[:, 1:]，但调用方通常已经对齐，
    这里直接对每个位置算 cross-entropy 后平均（``ignore_index`` 处不计）。
    """
    # view 成 [B*T, V] 和 [B*T]
    V = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1).long(),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


def make_lm_batch(
    input_ids: torch.Tensor, block_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把一段 token 序列变成 (x, y) 用于 LM 训练（teacher forcing）。

    给定 ``input_ids`` shape ``[T]`` 或 ``[B, T]``：
        x = input_ids[:, :-1]   （前 T-1 个 token 作为输入）
        y = input_ids[:, 1:]    （后 T-1 个 token 作为 target）

    这是 next-token prediction 的标准构造。``block_size`` 用来 sanity check。
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.size(-1) < 2:
        raise ValueError("序列长度至少要 2 才能做 next-token 预测")
    if block_size is not None and input_ids.size(-1) > block_size:
        input_ids = input_ids[..., :block_size]
    x = input_ids[..., :-1]
    y = input_ids[..., 1:]
    return x, y


@torch.no_grad()
def generate(
    model: TinyGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    greedy: bool = False,
) -> torch.Tensor:
    """自回归采样。

    参数：
        input_ids: prompt，shape ``[T]`` 或 ``[B, T]``
        max_new_tokens: 最多生成多少个新 token
        temperature: > 1 更随机，< 1 更确定，= 0 退化为 greedy（推荐直接设 greedy=True）
        top_k: 只在概率最高的 K 个 token 里采（None = 不截断）
        greedy: True 表示贪心（取 argmax），忽略 temperature/top_k

    返回：shape ``[B, T + max_new_tokens]``。
    """
    was_training = model.training
    model.eval()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    B = input_ids.size(0)
    out = input_ids
    block_size = model.max_seq_len

    for _ in range(max_new_tokens):
        # 如果序列超过 block_size，截最后 block_size 个（sliding window）
        cond = out if out.size(1) <= block_size else out[:, -block_size:]
        logits = model(cond)  # [B, T, V]
        logits = logits[:, -1, :]  # 只取最后一个位置 → [B, V]

        if greedy or temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
        else:
            logits = logits / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                # 把 top_k 之外的 logits 置 -inf
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, [-1]]
                logits = torch.where(
                    logits < thresh, torch.full_like(logits, float("-inf")), logits
                )
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]

        out = torch.cat([out, next_id], dim=1)

    model.train(was_training)  # 生成完恢复原模式，避免影响后续训练
    return out


def sft_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    response_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """SFT（Supervised Fine-Tuning）的 loss：只在 response token 上算。

    SFT 数据格式：``prompt + response``，我们只想让模型学 response 部分
    （prompt 是条件，不该算损失——否则模型既学"怎么提问"又学"怎么回答"）。

    输入：
        logits: [B, T, V]
        targets: [B, T]，prompt 部分填 ``ignore_index``（默认 -100），
                 response 部分填真实 token id
        response_mask: [B, T]，1 表示该位置是 response（要算 loss），0 是 prompt
        ignore_index: cross-entropy 里跳过的 label

    工程实现：直接用 ``F.cross_entropy`` 的 ``ignore_index`` 机制——
    把 prompt 位置的 target 设成 ``ignore_index`` 即可，不用自己 mask。
    但保留 response_mask 是为了统计有效 token 数（算 per-token loss）。
    """
    V = logits.size(-1)
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1).long()
    # 把 mask=0 的位置（prompt）填成 ignore_index
    flat_mask = response_mask.reshape(-1).bool()
    flat_targets = torch.where(flat_mask, flat_targets, torch.full_like(flat_targets, ignore_index))
    loss = F.cross_entropy(
        flat_logits, flat_targets, ignore_index=ignore_index, reduction="mean"
    )
    return loss
