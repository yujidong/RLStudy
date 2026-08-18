"""Build notebooks/ch10_tiny_gpt.ipynb from a Python cell list.

Usage:
    python build_ch10.py

This keeps the notebook source under version control in a readable form,
matching the project's `build_notebooks.py` style.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from nb_helpers import Cell, md, code, build_notebook, save


# =============================================================================
# Notebook content
# =============================================================================
def ch10() -> List[Cell]:
    return [
        # ---------------------------------------------------------------------
        # Title + 学习目标 + 承诺
        # ---------------------------------------------------------------------
        md("""# 第 10 章：从零搭 TinyGPT —— Phase 3 的基础设施（LLM 预训练）

> **Phase 1 + Phase 2** 给了我们一个完整的"表格 + 函数逼近 + 策略梯度 + PPO"工具箱。
> Phase 3 要把这些工具**搬到大语言模型（LLM）上**——但 LLM 自己怎么来的？
> 本章回答这个问题：
>
> > **从零实现一个 mini-GPT，让它在 CPU 上能训、能生成、能画注意力热力图。**
>
> 本章**不涉及任何 RL**——纯深度学习。它是 Phase 3 的"基础设施章"，给后续
> Ch11（Reward Modeling）、Ch12（RLHF-PPO）、Ch13（GRPO）提供一个能跑的 base model。
>
> **本章核心等式**（Vaswani et al. 2017，Transformer 的灵魂）：
>
> $$\\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$
>
> 这一行的精髓是：**把"查表"软化成"加权平均"**，权重由 query 和 key 的相似度决定。
> 后面所有的 GPT/Claude/Gemini 都建立在这一行之上。除以 $\\sqrt{d_k}$ 不是花拳绣腿——
> 我们会在 §10.3 严格证明它**防止 softmax 饱和、梯度消失**。

## 学习目标

1. 理解 **为什么 Phase 3 需要一个 base model**（RLHF 三阶段 = SFT + RM + RL）
2. 实现 **char-level tokenizer** + **sinusoidal positional encoding**
3. **完整推出 self-attention**：从"软性查表"直觉到 $\\text{softmax}(QK^T/\\sqrt{d_k})V$
4. **严格证明为什么要除 $\\sqrt{d_k}$**（点积方差爆炸 → softmax 饱和）
5. 实现 **multi-head attention + causal mask**（下三角保证自回归）
6. 组装 **Transformer block**（attention + FFN + LayerNorm + residual，Pre-LN vs Post-LN）
7. 训练 **next-token prediction**（cross-entropy + teacher forcing）→ loss 显著下降
8. 在 prompt → response 数据上做 **SFT**（Supervised Fine-Tuning）—— 为 Ch11/12 铺垫
9. 实现并对比 **sampling 策略**（greedy / temperature / top-k）
10. 画出 **注意力热力图**（Ch00 章节图承诺的可视化），观察不同 head 学到的模式

## 承接的 Phase 1 / Phase 2 承诺（3 处）

| 出处 | 承诺原文 | 本章兑现节 |
|---|---|---|
| **Ch00** | **"Ch10 从零搭 TinyGPT，注意力热力图"** | **全章 + §10.4 热力图** |
| **Ch02 §2.1** | 马尔可夫性 + "LLM 上下文窗口就是为近似马尔可夫性服务的" | §10.6（LM as 马尔可夫过程） |
| **Ch00** | "Ch06+ PyTorch"（设计原则） | 全章（继承 Ch06-09 的 PyTorch 基础设施） |

## 术语速查

| 术语 | 含义 | 首次出现 |
|---|---|---|
| **token** | 文本的最小单元（这里 = 一个字符） | §10.2 |
| **embedding** | 把 token id 映射成一个稠密向量 | §10.2 |
| **positional encoding (PE)** | 给每个位置一个独特的"位置向量"（因为 attention 本身无序） | §10.2 |
| **self-attention** | 序列里每个位置都"看"其它位置，加权聚合信息 | §10.3 |
| **Q, K, V** | query / key / value——注意力三件套 | §10.3 |
| **multi-head** | 把 $d_{model}$ 切成 $h$ 份，各自做 attention，最后 concat | §10.4 |
| **causal mask** | 下三角 mask，保证位置 $i$ 只看 $\\le i$ 的位置（自回归） | §10.4 |
| **FFN** | 两层 MLP + GELU，attention 之后的"逐位置非线性变换" | §10.5 |
| **LayerNorm** | 把每个位置的向量归一化（均值 0、方差 1） | §10.5 |
| **Pre-LN / Post-LN** | LayerNorm 放在残差里/外——Pre-LN 训练更稳 | §10.5 |
| **teacher forcing** | 训练时用真实前缀作输入，不是模型自己的生成 | §10.6 |
| **SFT** | Supervised Fine-Tuning，在 prompt→response 上做条件 LM | §10.7 |
| **greedy / temperature / top-k** | 三种采样策略 | §10.8 |

## 本章路线图

| 节 | 主题 | 关键产出 |
|---|---|---|
| 10.1 | 为什么需要 base model | Phase 3 总览（RLHF 三阶段） |
| 10.2 | Tokenizer + embedding | `CharTokenizer` + `PositionalEncoding` |
| 10.3 | **Self-attention 完整推导** | $\\text{softmax}(QK^T/\\sqrt{d_k})V$ + **$\\sqrt{d_k}$ 证明** |
| 10.4 | Multi-head + causal mask | `CausalSelfAttention` + 注意力热力图 |
| 10.5 | Transformer block | Pre-LN vs Post-LN + `TransformerBlock` |
| 10.6 | LM head + cross-entropy + teacher forcing | next-token prediction 训练目标 |
| 10.7 | **SFT**（Supervised Fine-Tuning） | 为 Ch11/12 提供条件 LM 模型 |
| 10.8 | Sampling + 小结 | greedy / temperature / top-k + Ch11 预告 |
"""),
        # ---------------------------------------------------------------------
        # Setup cell
        # ---------------------------------------------------------------------
        code("""# 常规设置：找项目根、载入库
import sys, pathlib, time, math
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 本章新基础设施：纯 PyTorch 实现的 mini-GPT
from rlenvs import (
    CharTokenizer, PositionalEncoding,
    CausalSelfAttention, TransformerBlock,
    TinyGPT, build_tiny_gpt,
    compute_loss, make_lm_batch, generate, sft_loss,
)
from utils import set_seed
from utils.torch_utils import get_device, count_parameters

set_seed(42)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = get_device()
print(f"PyTorch: {torch.__version__}, device = {DEVICE}")
print(f"本章新基础设施：rlenvs/tiny_gpt.py")
print(f"  - CharTokenizer / PositionalEncoding")
print(f"  - CausalSelfAttention / TransformerBlock / TinyGPT")
print(f"  - compute_loss / make_lm_batch / generate / sft_loss")
print(f"训练语料：data/tiny_corpus.txt（合成小语料，CPU 可训）")
"""),
        # ---------------------------------------------------------------------
        # 10.1 为什么需要 base model
        # ---------------------------------------------------------------------
        md("""## 10.1 为什么需要 base model（Phase 3 总览）

### 10.1.1 RLHF 三阶段：SFT → RM → RL

Phase 3 的目标是用 RL 训练 LLM。整个故事——也就是 OpenAI 的 **InstructGPT 配方**
（Ouyang et al. 2022）和后续所有 ChatGPT/Claude 的核心——分三阶段：

```
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │ 1. SFT       │ ───→ │ 2. Reward    │ ───→ │ 3. RL        │
   │   (指令微调) │      │    Modeling  │      │   (PPO/GRPO) │
   │   Ch10       │      │   Ch11       │      │   Ch12/13    │
   └──────────────┘      └──────────────┘      └──────────────┘
   base GPT  →  会跟指令  →  知道什么是"好"  →  往"好"的方向优化
```

1. **SFT（Supervised Fine-Tuning）**：拿一个预训练好的 base GPT，在
   "prompt → 理想 response"数据上做条件 LM 训练。出来的模型"会回答问题"。
2. **RM（Reward Modeling）**：让人类（或更强的模型）给两个 response 排名，
   训练一个 reward model $r(x, y)$ 预测哪个更好（Bradley-Terry 模型）。
3. **RL**：把 SFT 模型当 actor，把 RM 当 reward function，用 PPO/GRPO 优化。
   加 KL penalty 防止跑偏（不要离 reference model 太远）。

### 10.1.2 本章的角色：把"阶段 1"和"base 模型"都搞定

本章会同时做两件事：

- **从零搭一个 base GPT**（embedding + attention + block + LM head）——这是"预训练"的简化版。
- **在 prompt→response 数据上做 SFT**——这是阶段 1 的简化版。

这样 Ch11 拿到的就是一个"会回答问题"的模型，可以直接做 RM 和 RL。

### 10.1.3 一个关键观察：LLM 状态 = 已生成 token 序列（承接 Ch02 §2.1）

Ch02 §2.1 我们讨论过"马尔可夫性"。LLM **严格不是**马尔可夫的——
它生成下一个 token 时看的是**整个历史** $s_t = (y_1, y_2, \\dots, y_t)$。

但在工程上，LLM 有一个**上下文窗口** $T$（如 GPT-4 是 128k），超出窗口的内容
就"看不见"了。在窗口内，状态可以写成 $s_t = (y_{t-T+1}, \\dots, y_t)$，
**形式上是一个 $T$ 阶马尔可夫过程**。

> 这就是为什么我们后面能用 PPO/GRPO（马尔可夫假设的算法）训 LLM：
> 上下文窗口内近似马尔可夫。如果窗口外的内容真的重要，RLHF 会失效——
> 这也是为什么长上下文模型是研究热点。

本章的 TinyGPT 设上下文窗口 = 128（`max_seq_len`），是一个简化但**结构正确**的 LLM。

### 10.1.4 为什么从零实现（不用 HuggingFace）

工程上你当然用 HF Transformers。但本章目标是**理解**：
- attention 矩阵怎么算
- causal mask 长什么样
- Pre-LN 和 Post-LN 区别在哪
- 为什么除 $\\sqrt{d_k}$

这些细节决定了你后面能不能调 RLHF。所以我们手写，**每个组件 < 50 行**。
"""),
        code("""# Phase 3 总览：用一个示意图把 RLHF 三阶段画出来
fig, ax = plt.subplots(figsize=(11, 3.2))

stages = [
    ("1. Base GPT\\n(预训练)", "Ch10", "#4C72B0"),
    ("2. SFT\\n(指令微调)", "Ch10 §10.7", "#55A868"),
    ("3. Reward Model\\n(偏好学习)", "Ch11", "#C44E52"),
    ("4. RLHF-PPO / GRPO\\n(RL 优化)", "Ch12 / Ch13", "#8172B2"),
]
for i, (name, chap, color) in enumerate(stages):
    ax.add_patch(plt.Rectangle((i*2.5, 0.3), 2.2, 1.0, facecolor=color, alpha=0.35, edgecolor=color, linewidth=2))
    ax.text(i*2.5+1.1, 0.95, name, ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(i*2.5+1.1, 0.45, chap, ha='center', va='center', fontsize=9, color=color)
    if i < len(stages)-1:
        ax.annotate('', xy=(i*2.5+2.45, 0.8), xytext=(i*2.5+2.2, 0.8),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax.text(5, -0.05, "本章 Ch10 同时搞定第 1 步（base GPT）和第 2 步（SFT）",
        ha='center', fontsize=10, style='italic', color='#444')
ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.2, 1.6)
ax.axis('off')
ax.set_title('Phase 3 蓝图：从 base GPT 到 RLHF', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
"""),
        # ---------------------------------------------------------------------
        # 10.2 Tokenizer + embedding
        # ---------------------------------------------------------------------
        md("""## 10.2 Tokenizer + embedding

### 10.2.1 Tokenizer：文本 → 整数序列

神经网络只能吃数字。**tokenizer** 把文本切成一个个 **token**，每个 token 映射成一个整数 id。

主流三种：

| 方案 | 例子 | vocab size | 优点 | 缺点 |
|---|---|---|---|---|
| **char-level** | `'a'`, `'b'`, `' '` | ~100 | 简单、OOV 免疫 | 序列长 |
| **word-level** | `'hello'`, `'world'` | ~10⁵ | 序列短 | OOV、大小写、标点 |
| **BPE / subword** | `'hel'`, `'lo'` | ~10⁴ | 平衡，主流 LLM 用的 | 实现复杂 |

本章用 **char-level**——简单可靠，适合教学。我们的合成语料 vocab 大约 40 个字符。

### 10.2.2 Token embedding：整数 → 稠密向量

token id 是个无序的整数（id=5 和 id=6 没有"近"的关系）。**embedding** 把它映射成
一个 $d_{model}$ 维向量。这个映射是**可学习的**——模型自己决定怎么安排。

$$\\text{emb}(\\text{token}=i) = E[i, :] \\in \\mathbb{R}^{d_{model}}, \\quad E \\in \\mathbb{R}^{V \\times d_{model}}$$

其中 $E$ 是 embedding 矩阵，$V$ 是 vocab size。

### 10.2.3 Positional encoding（PE）：attention 本身无序，得自己加位置

attention 是"集合操作"——它不知道 $y_3$ 在 $y_5$ 前面。所以我们要**显式**给每个位置
一个独特的位置向量。两种主流选择：

| 方案 | 形式 | 是否可学习 |
|---|---|---|
| **sinusoidal**（Vaswani 2017） | $\\sin/\\cos$ 不同频率 | 否（固定公式） |
| **learned**（GPT-2） | 一个 `[max_len, d_model]` embedding 矩阵 | 是 |

本章用 **sinusoidal**——经典、不增加参数、不同位置有不同周期。

$$PE(pos, 2k) = \\sin\\!\\left(\\frac{pos}{10000^{2k/d_{model}}}\\right), \\quad PE(pos, 2k+1) = \\cos\\!\\left(\\frac{pos}{10000^{2k/d_{model}}}\\right)$$

> **直觉**：低维用高频（短周期，相邻位置区分明显），高维用低频（长周期，捕捉远距离结构）。
> 像傅里叶变换——用一组正弦波"编码"任意位置。
"""),
        code("""# 10.2.4 看一下语料 + tokenizer 的实际行为

corpus_path = ROOT / 'data' / 'tiny_corpus.txt'
text = corpus_path.read_text(encoding='utf-8')
print(f"语料长度: {len(text)} 字符")
print(f"前 250 字符预览:\\n{repr(text[:250])}")
print()

tok = CharTokenizer().train(text)
print(f"vocab size: {tok.vocab_size}")
print(f"vocab 字符列表: {tok.itos}")
print()

# ids：完整语料的 token 序列，后续训练都用它（注意：这是个会被复用的全局变量）
ids = tok.encode(text)
print(f"完整语料 token 数: {ids.numel()}（{len(text)} 字符 → {ids.numel()} tokens，char-level 1:1）")
print()

sample = "Q: What is the color of the sky?\\n"
sample_ids = tok.encode(sample)
print(f"示例 encode('{sample.strip()}'):")
print(f"  sample_ids: {sample_ids.tolist()}")
print(f"  decode 回来: {repr(tok.decode(sample_ids))}")
print(f"  正确还原: {tok.decode(sample_ids) == sample}")
"""),
        code("""# 10.2.5 可视化 sinusoidal positional encoding（不同维度的频率）
d_model_demo = 64
max_len_demo = 64
pe = PositionalEncoding(d_model_demo, max_len=max_len_demo)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左：选 4 个维度画 PE[:, d] 随 pos 的变化
pe_table = pe.pe[0].numpy()  # [max_len, d_model]
dims_to_show = [0, 4, 16, 60]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for d, c in zip(dims_to_show, colors):
    axes[0].plot(np.arange(max_len_demo), pe_table[:, d], label=f'dim {d}', color=c, linewidth=1.8)
axes[0].set_xlabel('position'); axes[0].set_ylabel('PE value')
axes[0].set_title('Sinusoidal PE: 不同维度的频率')
axes[0].legend(); axes[0].grid(alpha=0.3)

# 右：完整 PE 热力图 [max_len, d_model]
im = axes[1].imshow(pe_table, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_xlabel('dimension'); axes[1].set_ylabel('position')
axes[1].set_title('Positional Encoding 全貌 (sin/cos pattern)')
plt.colorbar(im, ax=axes[1], label='PE value')

plt.tight_layout(); plt.show()

# 验证：低维度高频、高维度低频
print(f"dim 0 周期 ~ 几个位置（高频）；dim 60 周期很长（低频）")
print(f"dim 0 在 0-10 内已经振荡多次；dim 60 在 0-60 内才振荡 1 次")
"""),
        # ---------------------------------------------------------------------
        # 10.3 Self-attention 完整推导
        # ---------------------------------------------------------------------
        md("""## 10.3 Self-attention 完整推导（本章核心 1/2）

### 10.3.1 直觉：从"硬查表"到"软性查表"

想象你有一个**字典**（key-value pairs）：

```
keys:    ["苹果", "香蕉", "猫"]
values:  [fruit,  fruit,  animal]
```

给一个 **query** "橘子"，你怎么查？硬查表找不到完全匹配的。
但你**知道**"橘子"和"苹果""香蕉"都是水果，所以应该把它们的 value 平均一下。

**Attention 的核心思想**：query 不用硬匹配某个 key，而是**和所有 key 算相似度**，
然后用相似度当权重，对所有 value 加权平均。

数学上，给定一个 query $q$ 和一组 $(k_i, v_i)$：

$$\\text{output} = \\sum_i \\alpha_i \\, v_i, \\qquad \\alpha_i = \\frac{\\exp(q \\cdot k_i)}{\\sum_j \\exp(q \\cdot k_j)}$$

注意 $\\alpha_i$ 就是 $\\text{softmax}(q \\cdot k_i)$——所有权重和为 1，权重正比于 $q$ 和 $k_i$ 的点积（相似度）。

### 10.3.2 推广到 batch：Q, K, V 都是矩阵

实际中我们一次处理整个序列。设序列长度 $T$，每个 token 先经三个不同的线性变换得到
$Q, K, V \\in \\mathbb{R}^{T \\times d_k}$：

$$Q = X W_Q, \\quad K = X W_K, \\quad V = X W_V, \\quad X \\in \\mathbb{R}^{T \\times d_{model}}$$

其中 $W_Q, W_K \\in \\mathbb{R}^{d_{model} \\times d_k}$, $W_V \\in \\mathbb{R}^{d_{model} \\times d_v}$。

那么**所有 $T$ 个 query 同时算**：

$$\\boxed{\\;\\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V\\;}$$

逐项解释：

| 表达式 | shape | 含义 |
|---|---|---|
| $Q K^T$ | $[T, T]$ | 第 $i$ 行第 $j$ 列 = 第 $i$ 个 query 与第 $j$ 个 key 的点积（相似度） |
| $\\div \\sqrt{d_k}$ | $[T, T]$ | 缩放（见 §10.3.3） |
| $\\text{softmax}(\\cdot)$（沿最后一维） | $[T, T]$ | 每个 query 对所有 key 的权重分布（行和为 1） |
| $\\cdot V$ | $[T, d_v]$ | 用权重对 value 加权求和 → 每个 query 的输出 |

> **关键直觉**：$QK^T$ 这一步是"**每个位置都问一遍所有位置**：'你跟我有多相关？'"，
> 然后 softmax 归一化、用这个相关性加权聚合 value。整个 self-attention 就是这么个操作。

### 10.3.3 为什么除 $\\sqrt{d_k}$（严格证明，本章必须兑现）

这是 Transformer 论文里**最容易跳过但最关键**的细节。我们严格证明。

**命题**：若 $q, k \\in \\mathbb{R}^{d_k}$，各分量 $q_i, k_i$ 独立同分布、均值 0、方差 1，
则点积 $q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i$ 的**均值为 0、方差为 $d_k$**。

**证明**：

$$\\mathbb{E}[q \\cdot k] = \\mathbb{E}\\!\\left[\\sum_{i=1}^{d_k} q_i k_i\\right] = \\sum_{i=1}^{d_k} \\mathbb{E}[q_i]\\,\\mathbb{E}[k_i] = 0 \\cdot 0 = 0$$

（用了独立性 $\\mathbb{E}[q_i k_i] = \\mathbb{E}[q_i]\\,\\mathbb{E}[k_i]$）

$$\\text{Var}(q \\cdot k) = \\sum_{i=1}^{d_k} \\text{Var}(q_i k_i) = \\sum_{i=1}^{d_k} \\mathbb{E}[q_i^2 k_i^2] - 0 = \\sum_{i=1}^{d_k} \\mathbb{E}[q_i^2]\\,\\mathbb{E}[k_i^2] = \\sum_{i=1}^{d_k} 1 \\cdot 1 = d_k$$

**推论**：标准差 $= \\sqrt{d_k}$。当 $d_k$ 大（如 64），点积值很容易到 ±8 甚至 ±16。

**为什么这是灾难**？看 softmax：

$$\\text{softmax}(z)_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$

当 $z$ 里有一个值特别大（比如 $z_1 = 16$，其它 ~0），$e^{16} \\approx 10^7 \\gg 1$，
softmax 输出几乎就是 one-hot（$\\alpha_1 \\approx 1$，其它 $\\approx 0$）。
此时**梯度消失**：$\\partial \\alpha_i / \\partial z_j$ 几乎为 0，模型学不动。

**解决方案**：除以 $\\sqrt{d_k}$，把方差从 $d_k$ 压回 1，标准差从 $\\sqrt{d_k}$ 压回 1：

$$\\text{Var}\\!\\left(\\frac{q \\cdot k}{\\sqrt{d_k}}\\right) = \\frac{d_k}{(\\sqrt{d_k})^2} = 1$$

这样点积值控制在 $O(1)$ 量级，softmax 梯度健康。

<details>
<summary><b>数值验证（点开看）</b></summary>

我们用 numpy 直接验证：随机生成 $q, k \\in \\mathbb{R}^{d_k}$，$d_k$ 从 1 到 256，
看点积的方差是不是 $\\approx d_k$，除以 $\\sqrt{d_k}$ 后是不是 $\\approx 1$。
""")
        ,
        code("""# 10.3.4 数值验证：点积方差随 d_k 线性增长，除以 sqrt(d_k) 后稳定在 1
torch.manual_seed(0)
np.random.seed(0)

dks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
n_trials = 5000
raw_var = []
scaled_var = []
max_abs_raw = []
max_abs_scaled = []
max_softmax_raw = []  # softmax 的最大值（接近 1 说明饱和）

for dk in dks:
    # q, k 各分量 ~ N(0,1)
    q = np.random.randn(n_trials, dk)
    k = np.random.randn(n_trials, dk)
    dots = np.sum(q * k, axis=1)               # shape [n_trials]
    scaled = dots / np.sqrt(dk)
    raw_var.append(dots.var())
    scaled_var.append(scaled.var())
    max_abs_raw.append(np.percentile(np.abs(dots), 99))
    max_abs_scaled.append(np.percentile(np.abs(scaled), 99))
    # 模拟 softmax 饱和：3 个 key 的点积
    q3 = np.random.randn(n_trials, 3, dk)
    k3 = np.random.randn(n_trials, 3, dk)
    d3 = np.einsum('ntd,ntd->nt', q3, k3)  # [n_trials, 3]
    sm_raw = np.exp(d3) / np.exp(d3).sum(axis=1, keepdims=True)
    sm_scaled = np.exp(d3 / np.sqrt(dk)) / np.exp(d3 / np.sqrt(dk)).sum(axis=1, keepdims=True)
    max_softmax_raw.append(sm_raw.max(axis=1).mean())     # 平均的最大 softmax 概率

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

axes[0].plot(dks, raw_var, 'o-', label='Var(q·k) 实测', color='#d62728', linewidth=2)
axes[0].plot(dks, dks, '--', label='Var = d_k (理论)', color='gray', alpha=0.7)
axes[0].plot(dks, scaled_var, 's-', label='Var(q·k/√d_k) 实测', color='#2ca02c', linewidth=2)
axes[0].axhline(1, color='#2ca02c', linestyle='--', alpha=0.5)
axes[0].set_xscale('log'); axes[0].set_xlabel('d_k'); axes[0].set_ylabel('Variance')
axes[0].set_title('点积方差：除 √d_k 把方差从 d_k 压回 1')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(dks, max_softmax_raw, 'o-', label='未缩放：平均最大 softmax 概率', color='#d62728', linewidth=2)
# scaled 后的平均最大 softmax 概率
q3 = np.random.randn(n_trials, 3, dk); k3 = np.random.randn(n_trials, 3, dk)
ms_scaled_list = []
for dk in dks:
    q3 = np.random.randn(n_trials, 3, dk); k3 = np.random.randn(n_trials, 3, dk)
    d3 = np.einsum('ntd,ntd->nt', q3, k3)
    sm = np.exp(d3/np.sqrt(dk))/np.exp(d3/np.sqrt(dk)).sum(axis=1, keepdims=True)
    ms_scaled_list.append(sm.max(axis=1).mean())
axes[1].plot(dks, ms_scaled_list, 's-', label='除 √d_k：平均最大 softmax 概率', color='#2ca02c', linewidth=2)
axes[1].axhline(1/3, color='gray', linestyle='--', alpha=0.6, label='均匀分布 = 1/3')
axes[1].set_xscale('log'); axes[1].set_xlabel('d_k')
axes[1].set_ylabel('mean max softmax prob (3 keys)')
axes[1].set_title('softmax 饱和：未缩放时最大概率 → 1')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()

print("结论：")
print(f"  - 未缩放：d_k=256 时，3-key softmax 的最大概率 ≈ {max_softmax_raw[-1]:.3f}（几乎 one-hot）")
print(f"  - 缩放后：稳定在 ≈ {ms_scaled_list[-1]:.3f}（接近均匀 1/3）")
print(f"  - 这就是为什么要除 √d_k：防止 softmax 饱和 + 梯度消失")
"""),
        md("""
</details>

### 10.3.5 用我们的手写 attention 验证公式

我们用一个最小例子检查 `CausalSelfAttention`（暂不开 causal mask，下一节才讲），
确认它做的就是 $\\text{softmax}(QK^T/\\sqrt{d_k})V$。
"""),
        code("""# 10.3.6 手写 attention vs 公式手算（单 head，无 mask）
torch.manual_seed(0)

d_model, T = 8, 4
# 一个小 attention（n_heads=1 = d_model，暂不开 mask 我们手动算）
att = CausalSelfAttention(d_model=d_model, n_heads=1)
att.eval()
x = torch.randn(1, T, d_model)

# 模型 forward
out_model = att(x)
print(f"模型 forward 输出 shape: {out_model.shape}")

# 手算：Q, K, V
Wq, Wk, Wv = att.qkv_proj.weight.split(d_model, dim=0)  # 每个 [d_model, 3d] split 后 [d_model, d_model]
# 注意 qkv_proj 输出是 [B,T,3d]，按最后一维 split 3 段
qkv = att.qkv_proj(x)  # [1, T, 3d]
q, k, v = qkv.split(d_model, dim=-1)
print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

# scaled dot-product
dk = d_model
scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)  # [1, T, T]
print(f"\\nQK^T/√d_k (scores):\\n{scores[0].detach().numpy().round(3)}")
att_weights = F.softmax(scores, dim=-1)
print(f"\\nsoftmax(scores) (attention weights):\\n{att_weights[0].detach().numpy().round(3)}")
print(f"  每行和（应该都是 1）: {att_weights[0].sum(dim=-1).detach().numpy().round(4)}")

out_manual = att_weights @ v  # [1, T, d_model]
# out_manual 还没经过 out_proj，模型最终输出会过 out_proj，所以我们检查 out_proj 前
# 我们的 att.out_proj 是 identity 吗？不是。比较 att.out_proj 输入：
# 模型里 out = att_weights @ v 再 view 再 out_proj(out)
out_before_proj = out_model
# 直接验证：把模型的 out_proj 也手算
expected = (att_weights @ v).transpose(1, 2).contiguous().view(1, T, d_model)
expected = att.out_proj(expected)
diff = (out_model - expected).abs().max().item()
print(f"\\n手算 vs 模型 forward 最大差异: {diff:.2e}（应该 ~0，证明我们实现的就是公式）")

# 模型存的 att_weights
print(f"\\n模型记录的 att_weights shape: {att.att_weights.shape}")
print(f"和手算一致: {torch.allclose(att.att_weights, att_weights.detach(), atol=1e-6)}")
"""),
        # ---------------------------------------------------------------------
        # 10.4 Multi-head attention + causal mask
        # ---------------------------------------------------------------------
        md("""## 10.4 Multi-head attention + causal mask（本章核心 2/2）

### 10.4.1 Multi-head：把 $d_{model}$ 切成 $h$ 份并行做

单个 attention 学一种"匹配模式"。但语言里同时有多种关系：语法（主谓一致）、
语义（同义词）、长程依赖（指代消解）。**Multi-head** 让模型同时学多种模式：

> 把 $d_{model}$ 维向量切成 $h$ 个 head，每个 head 维度 $d_k = d_{model}/h$，
> 各自做一次 attention，最后 concat 起来经一个线性映射回 $d_{model}$。

$$\\text{MultiHead}(X) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O$$

$$\\text{where}\\; \\text{head}_i = \\text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

每个 head 用**独立的投影矩阵** $W_i^Q, W_i^K, W_i^V \\in \\mathbb{R}^{d_{model} \\times d_k}$，
所以每个 head 关注不同的子空间。

**直觉**：不同 head 像不同"专家"——有的 head 关注"前一个名词"，有的关注"句末标点"，
有的关注"前缀匹配"。我们会在 §10.4.4 的热力图里**亲眼看到**这种分化。

### 10.4.2 Causal mask：自回归的核心约束

GPT 是 decoder-only——**生成第 $t$ 个 token 时只能看 $\\le t$ 的位置**。
否则训练时就"偷看"了未来，推理（生成）时没未来可看，训练和推理不一致（exposure bias）。

实现上，在 $QK^T$ 得到的 $[T, T]$ 相似度矩阵上，**把上三角（不含对角线）置 $-\\infty$**：

```
mask (T=5):                 scores after mask:
[0, -inf, -inf, -inf, -inf]   [q0·k0,  -inf,   -inf,   -inf,   -inf  ]
[0,   0,  -inf, -inf, -inf]   [q1·k0,  q1·k1,  -inf,   -inf,   -inf  ]
[0,   0,    0,  -inf, -inf]   [q2·k0,  q2·k1,  q2·k2,  -inf,   -inf  ]
[0,   0,    0,    0,  -inf]   [q3·k0,  q3·k1,  q3·k2,  q3·k3,  -inf  ]
[0,   0,    0,    0,    0 ]   [q4·k0,  q4·k1,  q4·k2,  q4·k3,  q4·k4]
```

softmax 后 $-\\infty$ 的位置变成 0，自然就不参与加权。下三角（含对角线）保留——
**这就是 causal mask**。

> **承接 Ch02 §2.1 的马尔可夫性**：causal mask 让 LLM 的状态严格等于"已生成 token 序列"，
> 形式上是 $T$ 阶马尔可夫过程（$T$ = 上下文窗口）。这正是后面能用 PPO/GRPO 的理论基础。
"""),
        code("""# 10.4.3 用我们的 CausalSelfAttention 演示 causal mask 的效果
torch.manual_seed(0)
T, d_model, n_heads = 6, 16, 4
att = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
att.eval()
x = torch.randn(1, T, d_model)
_ = att(x)

# 取第一个 head 的 attention weights
W = att.att_weights[0, 0]  # [T, T]
print(f"attention weights (head 0, T={T}):")
print(W.numpy().round(3))
print()

# 验证：上三角（i < j）应该全 0，每行和应该 = 1
upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
print(f"上三角（i<j）的最大值: {W[upper].max().item():.2e}（应该 ≈ 0，证明 causal mask 生效）")
print(f"每行和: {W.sum(dim=-1).numpy().round(4)}（应该都是 1，softmax 性质）")

# 画 causal mask 的样子
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
im0 = axes[0].imshow(torch.tril(torch.ones(T, T)).numpy(), cmap='Blues', vmin=0, vmax=1)
axes[0].set_title(f'Causal mask (下三角 = 允许关注)\\nT={T}')
axes[0].set_xlabel('key position j'); axes[0].set_ylabel('query position i')
plt.colorbar(im0, ax=axes[0])
for i in range(T):
    for j in range(T):
        val = '✓' if j <= i else '✗'
        axes[0].text(j, i, val, ha='center', va='center', fontsize=10,
                     color='white' if j<=i else 'red')

# 真实 attention weights 热力图（head 0）
im1 = axes[1].imshow(W.numpy(), cmap='viridis', vmin=0)
axes[1].set_title('实际 attention weights (head 0)\\n上三角为 0 = causal mask')
axes[1].set_xlabel('key position j'); axes[1].set_ylabel('query position i')
plt.colorbar(im1, ax=axes[1])
plt.tight_layout(); plt.show()
"""),
        md("""### 10.4.4 注意力热力图（Ch00 章节图承诺的可视化）

下面我们训练一个小模型（200 步快速版），然后**画每一层每一个 head 的注意力矩阵**——
这是 Ch00 章节图明确承诺的可视化。你会看到不同 head 学到不同的关注模式。

（训练循环细节在 §10.6 详讲，这里先用一个快速版生成热力图。）
"""),
        code("""# 10.4.5 快速训练一个迷你模型，准备画热力图
# （正式训练在 §10.6，这里只训 400 步拿到能看 attention 的模型）
torch.manual_seed(42)
np.random.seed(42)

demo_model = build_tiny_gpt(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=3, d_ff=256, max_seq_len=64)
print(f"演示模型参数量: {count_parameters(demo_model):,}")

block_size = 48
opt = torch.optim.AdamW(demo_model.parameters(), lr=2e-3)
B = 32
demo_losses = []
for step in range(400):
    ix = torch.randint(0, ids.numel() - block_size - 1, (B,))
    x = torch.stack([ids[i:i+block_size] for i in ix])
    y = torch.stack([ids[i+1:i+1+block_size] for i in ix])
    logits = demo_model(x)
    loss = compute_loss(logits, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(demo_model.parameters(), 1.0)
    opt.step()
    demo_losses.append(loss.item())
print(f"演示训练完成: loss {demo_losses[0]:.3f} → {demo_losses[-1]:.3f}")
"""),
        code("""# 10.4.6 画所有 layer × head 的注意力热力图（Ch00 章节图承诺）
# 用一段典型文本做 forward，然后画每层每头的 attention
demo_text = "Q: What is the color of the sky?\\nA: The color"
demo_ids = tok.encode(demo_text).unsqueeze(0)  # [1, T]
T_demo = demo_ids.size(1)
demo_model.eval()
with torch.no_grad():
    _ = demo_model(demo_ids)

all_weights = demo_model.get_attention_weights()  # list of [1, H, T, T]
n_layers = len(all_weights)
n_heads = all_weights[0].size(1)

fig, axes = plt.subplots(n_layers, n_heads, figsize=(3*n_heads, 2.8*n_layers), squeeze=False)
for li in range(n_layers):
    for hi in range(n_heads):
        ax = axes[li][hi]
        W = all_weights[li][0, hi].numpy()  # [T, T]
        ax.imshow(W, cmap='hot', vmin=0, vmax=W.max())
        ax.set_title(f'L{li} H{hi}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        # 标几个 token 在 y 轴
        if hi == 0:
            tick_positions = list(range(0, T_demo, 5))
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([demo_text[min(i, len(demo_text)-1)] for i in tick_positions], fontsize=7)

fig.suptitle(f'注意力热力图：{n_layers} 层 × {n_heads} 头\\n输入: "{demo_text}"\\n'
             f'每个子图 [T,T]，纵轴=query pos，横轴=key pos，亮=高权重',
             fontsize=12, y=1.0)
plt.tight_layout(); plt.show()

print(f"\\n观察：")
print(f"  - 不同 head 关注模式不同：有的集中在近期（对角线带），有的关注特定位置")
print(f"  - 浅层（L0）通常关注局部 / 标点")
print(f"  - 深层可能关注语义相关位置（如 'color' 关注 'sky'）")
print(f"  - 上三角恒为 0 = causal mask 生效")
"""),
        # ---------------------------------------------------------------------
        # 10.5 Transformer block
        # ---------------------------------------------------------------------
        md("""## 10.5 Transformer block（attention + FFN + LayerNorm + residual）

单个 attention 只能"加权聚合信息"，不能"逐位置非线性变换"。
一个完整的 **Transformer block** 把 attention、FFN、LayerNorm、残差都组装起来。

### 10.5.1 FFN（Position-wise Feed-Forward Network）

attention 之后，每个位置**独立**过一个两层 MLP（不同位置不交换信息）：

$$\\text{FFN}(x) = W_2 \\cdot \\text{GELU}(W_1 x + b_1) + b_2$$

中间维度通常是 $d_{model}$ 的 4 倍（如 $d_{model}=64 \\to d_{ff}=256$）。
为什么 4 倍？经验值，让 FFN 有"放大-压缩"的容量。

**GELU**（Gaussian Error Linear Unit）是 ReLU 的平滑版：
$\\text{GELU}(x) = x \\cdot \\Phi(x)$，其中 $\\Phi$ 是标准正态的 CDF。
GPT 系列都用 GELU。

### 10.5.2 LayerNorm + 残差：让深网络可训

**LayerNorm**：把每个位置的 $d_{model}$ 维向量归一化（均值 0、方差 1），
再加可学习的 $\\gamma, \\beta$。

$$\\text{LN}(x) = \\gamma \\cdot \\frac{x - \\mu}{\\sigma} + \\beta, \\quad \\mu = \\frac{1}{d}\\sum_i x_i, \\;\\sigma^2 = \\frac{1}{d}\\sum_i (x_i - \\mu)^2$$

**残差连接**（ResNet 的核心）：$x \\to x + f(x)$，让梯度能直通底层，
深网络（几十层）才训得动。

### 10.5.3 Pre-LN vs Post-LN（关键工程选择）

两种 LayerNorm 放法：

**Post-LN**（原 Transformer 论文，Vaswani 2017）：

$$h = \\text{LN}(x + \\text{Attn}(x))$$
$$\\text{out} = \\text{LN}(h + \\text{FFN}(h))$$

**Pre-LN**（Xiong et al. 2020，GPT-2 之后的事实标准）：

$$h = x + \\text{Attn}(\\text{LN}(x))$$
$$\\text{out} = h + \\text{FFN}(\\text{LN}(h))$$

**为什么 Pre-LN 更稳**？Post-LN 的残差路径上插了 LN，**梯度无法直通到底层**——
深网络底层梯度消失。Pre-LN 把 LN 移到分支里，主干是干净的残差路径，梯度健康。

> **本章用 Pre-LN**（`TransformerBlock` 默认），与 GPT-2/3 一致。
> 你可以在练习里换成 Post-LN 对比训练稳定性。
"""),
        code("""# 10.5.4 检查 TransformerBlock 的前向 + 参数
torch.manual_seed(0)
blk = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
print(f"一个 TransformerBlock 的子模块:")
print(f"  ln1:        LayerNorm(64)         -> {sum(p.numel() for p in blk.ln1.parameters()):,} params")
print(f"  att:        CausalSelfAttention    -> {sum(p.numel() for p in blk.att.parameters()):,} params")
print(f"  ln2:        LayerNorm(64)         -> {sum(p.numel() for p in blk.ln2.parameters()):,} params")
print(f"  ffn:        Linear-GELU-Linear    -> {sum(p.numel() for p in blk.ffn.parameters()):,} params")
print(f"  block total: {sum(p.numel() for p in blk.parameters()):,} params")

# 验证 forward
x = torch.randn(2, 10, 64)
out = blk(x)
print(f"\\nforward: x {tuple(x.shape)} → out {tuple(out.shape)}")
print(f"残差连接：输出 shape 必须等于输入 shape（{out.shape == x.shape}）")
"""),
        # ---------------------------------------------------------------------
        # 10.6 LM head + cross-entropy + teacher forcing
        # ---------------------------------------------------------------------
        md("""## 10.6 LM head + cross-entropy + teacher forcing

### 10.6.1 整体结构：把组件拼起来

`TinyGPT` = token embedding + PE + N × TransformerBlock + final LayerNorm + LM head：

```
input_ids [B, T]
    ↓ token embedding + √d_model scaling
    ↓ + sinusoidal PE
    ↓ dropout
[B, T, d_model]
    ↓ TransformerBlock × N
    ↓ final LayerNorm
    ↓ LM head: Linear(d_model → vocab_size)
logits [B, T, V]
```

LM head 就是把 $d_{model}$ 维隐状态映射到 $V$ 维 logits，每个位置预测下一个 token 的分布。

### 10.6.2 训练目标：next-token cross-entropy

给定一段 token 序列 $(y_1, y_2, \\dots, y_T)$，模型在每个位置 $t$ 预测 $y_{t+1}$：

$$\\mathcal{L}(\\theta) = -\\frac{1}{T} \\sum_{t=1}^{T} \\log p_\\theta(y_{t+1} | y_{\\le t})$$

这就是**交叉熵**（cross-entropy）损失。每个位置独立算，平均一下。

### 10.6.3 Teacher forcing：训练用真实前缀，不用模型自己的生成

训练时，位置 $t$ 的**输入**是真实的 $y_t$（不是模型在 $t-1$ 生成的 $\\hat y_t$）。
这叫 **teacher forcing**。

**好处**：训练高效稳定（每个位置都是"已知正确答案的前缀"，不用等模型生成）。
**坏处**：训练和推理分布不一致——推理时模型只能用自己的生成（exposure bias）。
RLHF（Ch12）会部分解决这个问题——但本章只用 teacher forcing，先训出 base model。

工程实现：直接把 `input_ids[:, :-1]` 当输入、`input_ids[:, 1:]` 当目标即可。
（我们封装在 `make_lm_batch` 里。）

### 10.6.4 完整训练循环

下面是本章的核心训练 cell：把上面所有组件串起来，在合成语料上训 TinyGPT。
"""),
        code("""# 10.6.5 正式训练：TinyGPT 在合成语料上的 next-token prediction
# 这个 cell 是本章的"主实验"，~30-40 秒
torch.manual_seed(42)
np.random.seed(42)

# 模型配置（约 800k 参数，CPU 可训）
CONFIG = dict(vocab_size=tok.vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64)
model = build_tiny_gpt(**CONFIG)
n_params = count_parameters(model)
print(f"模型配置: {CONFIG}")
print(f"参数量: {n_params:,}（{n_params/1e3:.0f}k）")

# 训练超参
block_size = 48
batch_size = 32
n_steps = 1200
lr = 1e-3
weight_decay = 0.01
grad_clip = 1.0

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# warmup + cosine decay（用简单实现）
def lr_lambda(step):
    warmup = 50
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, n_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# 训练循环
train_losses = []
val_losses = []
val_every = 100
# 简单验证集：用最后 200 个 token 作 held-out
val_ids = ids[-200:]
train_ids = ids[:-200]

t0 = time.time()
for step in range(n_steps):
    # 训练 batch：从 train_ids 随机切 block_size 长度的片段
    ix = torch.randint(0, train_ids.numel() - block_size - 1, (batch_size,))
    x = torch.stack([train_ids[i:i+block_size] for i in ix])
    y = torch.stack([train_ids[i+1:i+1+block_size] for i in ix])
    logits = model(x)
    loss = compute_loss(logits, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step(); sched.step()
    train_losses.append(loss.item())

    # 定期在 val 上评估
    if step % val_every == 0 or step == n_steps - 1:
        model.eval()
        with torch.no_grad():
            vx = val_ids[:block_size].unsqueeze(0)
            vy = val_ids[1:block_size+1].unsqueeze(0)
            vloss = compute_loss(model(vx), vy)
            val_losses.append((step, vloss.item()))
        model.train()

    if step % 200 == 0 or step == n_steps - 1:
        print(f"step {step:4d}  train_loss={loss.item():.3f}  lr={sched.get_last_lr()[0]:.2e}  elapsed={time.time()-t0:.1f}s")

print(f"\\n训练完成：{n_steps} 步，总耗时 {time.time()-t0:.1f}s")
print(f"train loss: {train_losses[0]:.3f} → {train_losses[-1]:.3f}")
print(f"val loss:   {val_losses[0][1]:.3f} → {val_losses[-1][1]:.3f}")
print(f"perplexity: {math.exp(train_losses[-1]):.2f}（越低越好；vocab={tok.vocab_size}）")
"""),
        code("""# 10.6.6 训练 + 验证 loss 曲线
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, color='#1f77b4', alpha=0.25, linewidth=0.7, label='train (raw)')
# 平滑
window = 30
if len(train_losses) > window:
    smooth_train = np.convolve(train_losses, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(window-1, len(train_losses)), smooth_train, color='#1f77b4', linewidth=2.5, label='train (smoothed)')
val_steps, val_ls = zip(*val_losses)
ax.plot(val_steps, val_ls, 'o-', color='#d62728', linewidth=2, markersize=6, label='validation')
ax.set_xlabel('training step'); ax.set_ylabel('cross-entropy loss')
ax.set_title(f'TinyGPT 训练曲线（{n_params:,} params, {n_steps} steps）')
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout(); plt.show()

print(f"\\n逐项解读:")
print(f"  - 初始 loss ≈ {train_losses[0]:.3f} ≈ ln({tok.vocab_size}) = {math.log(tok.vocab_size):.3f}（随机猜测的正确值）")
print(f"  - 最终 train loss ≈ {train_losses[-1]:.3f}")
print(f"  - 最终 val loss   ≈ {val_losses[-1][1]:.3f}")
gap = val_losses[-1][1] - train_losses[-1]
if gap > 0.5:
    print(f"  - train/val gap = {gap:.2f} > 0.5：有一些过拟合（小语料 + char-level 容易记忆）")
    print(f"    → 这正是为什么真实 LLM 训练用 100B+ tokens、weight decay、dropout")
else:
    print(f"  - train/val gap = {gap:.2f}：没明显过拟合")
"""),
        code("""# 10.6.7 生成样本：用训练好的模型生成文本
model.eval()
prompts = [
    "Q: What is the color of the sky",
    "The sun",
    "Q: How many legs does a cat",
    "One and one",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, p in zip(axes.flat, prompts):
    torch.manual_seed(0)
    prompt_ids = tok.encode(p)
    out = generate(model, prompt_ids, max_new_tokens=40, temperature=0.7, top_k=5)
    generated = tok.decode(out[0])
    ax.text(0.02, 0.5, f"prompt:\\n{p}\\n\\ngenerated:\\n{generated}",
            transform=ax.transAxes, fontsize=10, family='monospace', verticalalignment='center',
            bbox=dict(facecolor='#f0f0f0', edgecolor='gray', boxstyle='round,pad=0.5'))
    ax.axis('off')
    ax.set_title(f'prompt: "{p[:30]}..."', fontsize=10)
plt.suptitle('TinyGPT 生成样本（temperature=0.7, top_k=5）', fontsize=12)
plt.tight_layout(); plt.show()

print("\\n解读：")
print("  - 模型学到了 Q/A 格式：给 'Q: ...' 能续上 'A: ...'")
print("  - 学到了颜色/数量等局部模式")
print("  - 长距离连贯性差（模型太小、语料太小）—— 这是 base model 的局限，RLHF 会改")
"""),
        code("""# 10.6.8 模型报告：参数量、显存、训练速度
print("=" * 60)
print("TinyGPT 模型报告")
print("=" * 60)
print(f"配置: d_model={CONFIG['d_model']}, n_heads={CONFIG['n_heads']}, "
      f"n_layers={CONFIG['n_layers']}, d_ff={CONFIG['d_ff']}, vocab={tok.vocab_size}")
print(f"参数量: {n_params:,}")
print(f"参数量（MB, fp32）: {n_params * 4 / 1024 / 1024:.2f} MB")

# 各部分参数占比
emb_params = sum(p.numel() for p in model.tok_emb.parameters())
att_params = sum(p.numel() for p in model.blocks.parameters())
head_params = sum(p.numel() for p in model.lm_head.parameters())
ln_params = (sum(p.numel() for p in model.ln_final.parameters()))
print(f"\\n参数分布:")
print(f"  token embedding:  {emb_params:>8,} ({emb_params/n_params*100:.1f}%)")
print(f"  blocks (att+ffn): {att_params:>8,} ({att_params/n_params*100:.1f}%)")
print(f"  final LayerNorm:  {ln_params:>8,} ({ln_params/n_params*100:.2f}%)")
print(f"  LM head:          {head_params:>8,} ({head_params/n_params*100:.1f}%)")
print(f"\\n训练速度: {n_steps / (time.time() - t0):.1f} steps/s（CPU）")
print(f"训练时长: {time.time() - t0:.1f}s for {n_steps} steps")
"""),
        # ---------------------------------------------------------------------
        # 10.7 SFT
        # ---------------------------------------------------------------------
        md("""## 10.7 SFT（Supervised Fine-Tuning）

### 10.7.1 为什么需要 SFT

预训练（base GPT）只学了"接龙"——给任意前缀续一段统计上合理的文本。
但它**不会回答问题**——你问 "Q: ...?"，它可能续出另一个 "Q: ...?"（因为训练分布里 Q/A 配对存在）。

SFT 的目标：在 **prompt → response** 数据上训练，让模型学会"看到 prompt 就给 response"。

$$\\mathcal{L}_{SFT}(\\theta) = -\\frac{1}{|R|}\\sum_{t \\in R} \\log p_\\theta(y_t | x, y_{<t})$$

其中 $x$ 是 prompt，$y$ 是 response，$R$ 是 **response 部分的位置集合**。
注意：**只在 response 上算 loss**——prompt 是条件，不该算（否则模型既学"怎么提问"又学"怎么回答"）。

### 10.7.2 数据格式

每条样本 = prompt + response，例如：

```
prompt:   "Q: What is 2+3?\\n"
response: "A: 2+3=5."
```

我们把它拼成 `prompt + response`，并构造一个 mask 标记哪些位置是 response：
```
tokens:  Q : ... 2 + 3 ? \\n A : 2 + 3 = 5 .
mask:    0 0  0  0 0 0 0  0 1 1 1 1 1 1 1 1
                          └ response 从这里开始
```

训练时只对 mask=1 的位置算 cross-entropy。

### 10.7.3 SFT 演示

下面我们构造几条 SFT 数据，用 `sft_loss` 训练，看模型能否学会"答得对、答得简短"。

> **与 Ch11/12 的衔接**：SFT 后的模型就是 RLHF 阶段 1 的输出，Ch11 会以此为基础训 reward model，
> Ch12 会用 PPO 在它上面做 RL。
"""),
        code("""# 10.7.4 构造 SFT 数据集（prompt → response）
# 用合成算术数据：prompt 是 "X plus Y is"，response 是 "=Z."
# 注意：预训练语料里也有同格式的句子，SFT 是"专门强化"这种指令行为
import random
random.seed(42)

def make_sft_example(a, b):
    \"\"\"构造 a+b 的问答样本。返回 (full_ids, prompt_len)。\"\"\"
    prompt = f"{a} plus {b} is "
    response = f"{a+b}."
    full = prompt + response
    full_ids = tok.encode(full)
    prompt_len = len(tok.encode(prompt))
    return full_ids, prompt_len

# 构造训练集（a, b ∈ [0, 6]，留出大数测试泛化）
sft_data = []
for a in range(0, 7):
    for b in range(0, 7):
        full_ids, prompt_len = make_sft_example(a, b)
        if len(full_ids) <= 50:
            sft_data.append((full_ids, prompt_len))
print(f"SFT 数据集大小: {len(sft_data)}")
print(f"示例: {tok.decode(sft_data[10][0])!r}")
print(f"  prompt_len = {sft_data[10][1]}")

# 展示几条
for i in [0, 9, 18, 27, 35]:
    print(f"  [{i}] {tok.decode(sft_data[i][0])!r}")
"""),
        code("""# 10.7.5 SFT 训练：只在 response 上算 loss
# 用一个新初始化的小模型（不沿用预训练的，演示从零开始 SFT 也能学到 pattern）
torch.manual_seed(42)
np.random.seed(42)
sft_model = build_tiny_gpt(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=3, d_ff=256, max_seq_len=64)
print(f"SFT 模型参数量: {count_parameters(sft_model):,}")

block_size = 48
opt_sft = torch.optim.AdamW(sft_model.parameters(), lr=2e-3, weight_decay=0.01)
n_sft_steps = 500
sft_losses = []

t_sft = time.time()
for step in range(n_sft_steps):
    # 随机采样一个 batch
    batch_idx = random.sample(range(len(sft_data)), k=16)
    batch = [sft_data[i] for i in batch_idx]

    # 构造 batch tensors（pad 到同一长度）
    max_len = min(block_size, max(len(f) for f, _ in batch))
    x = torch.full((16, max_len), tok.pad_id, dtype=torch.long)
    y = torch.full((16, max_len), -100, dtype=torch.long)
    mask = torch.zeros((16, max_len), dtype=torch.long)
    for i, (full_ids, plen) in enumerate(batch):
        L = min(len(full_ids), max_len)
        x[i, :L] = full_ids[:L]
        # response mask: prompt 之后的 position 算 loss
        # 注意：位置 t 预测 t+1，所以 mask[t]=1 表示 t 位置的输出（=t+1 token）属于 response
        for t in range(L):
            # token at position t+1 is response if t+1 >= plen
            if t + 1 < L and t + 1 >= plen:
                mask[i, t] = 1
                y[i, t] = full_ids[t + 1]
            else:
                y[i, t] = -100  # 被 ignore_index 跳过

    logits = sft_model(x)
    loss = sft_loss(logits, y, mask)
    opt_sft.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
    opt_sft.step()
    sft_losses.append(loss.item())

    if step % 100 == 0 or step == n_sft_steps - 1:
        print(f"step {step:4d}  sft_loss={loss.item():.3f}  elapsed={time.time()-t_sft:.1f}s")

print(f"\\nSFT 训练完成: {n_sft_steps} 步，耗时 {time.time()-t_sft:.1f}s")
print(f"loss: {sft_losses[0]:.3f} → {sft_losses[-1]:.3f}")
"""),
        code("""# 10.7.6 SFT 后测试：给 prompt 看 response 对不对
sft_model.eval()
# 测试集：训练时 a,b ∈ [0,6]，这里测试 [0,6] 内的（held-in）和 [7,9] 的（held-out，看泛化）
test_cases = [(2, 3), (5, 1), (4, 4), (0, 6), (6, 6), (7, 2), (8, 3), (9, 5)]
print("SFT 模型在 prompt 上的表现（greedy decode）：\\n")

correct_in = 0; correct_out = 0; n_in = 0; n_out = 0
for a, b in test_cases:
    prompt = f"{a} plus {b} is "
    prompt_ids = tok.encode(prompt)
    out = generate(sft_model, prompt_ids, max_new_tokens=4, temperature=0.0, greedy=True)
    full = tok.decode(out[0])
    response = full[len(prompt):].strip()
    expected_str = str(a+b)
    # 看生成的第一个数字是不是正确答案的起点（小模型常过度生成'.'，所以只看首个数字）
    first_digit = next((ch for ch in response if ch.isdigit()), None)
    is_correct = (first_digit == expected_str[0])
    in_dist = a <= 6 and b <= 6
    if in_dist:
        n_in += 1; correct_in += int(is_correct)
    else:
        n_out += 1; correct_out += int(is_correct)
    mark = "OK" if is_correct else "X "
    tag = "held-in " if in_dist else "held-out"
    print(f"  [{mark}] ({tag}) prompt={prompt.strip()!r}")
    print(f"          response={response!r}  expected starts with {expected_str!r}")

print(f"\\nheld-in 首数字准确率:  {correct_in}/{n_in}")
print(f"held-out 首数字准确率: {correct_out}/{n_out}")
print("\\n解读:")
print("  - held-in：SFT 见过这种格式，应该答对（首数字正确）")
print("  - held-out (a,b > 6)：char-level 小模型学不会真正加法（要算法推理，不是模式匹配）")
print("  - 这正是 RLHF 的动机：SFT 只学'模仿格式'，要让它真'算对'得用 RM + RL（Ch11/12）")
print("  - 注意：模型把多 token 答案（如 '12'）当字符续写，容易在 '.' 上重复——")
print("    这是 char-level 小模型的局限，真实 LLM 用 BPE + 大模型能避免")

# SFT loss 曲线
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(sft_losses, color='#9467bd', alpha=0.3, linewidth=0.6)
window = 20
if len(sft_losses) > window:
    sm = np.convolve(sft_losses, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(window-1, len(sft_losses)), sm, color='#9467bd', linewidth=2.5, label='smoothed')
ax.set_xlabel('SFT step'); ax.set_ylabel('SFT loss (response only)')
ax.set_title(f'SFT 训练曲线（{n_sft_steps} steps）'); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()
"""),
        # ---------------------------------------------------------------------
        # 10.8 Sampling + 小结
        # ---------------------------------------------------------------------
        md("""## 10.8 Sampling 策略 + 小结

### 10.8.1 三种 sampling 策略

生成时，模型在每个位置输出一个 vocab 上的分布。**怎么从这个分布选下一个 token**？
三种主流策略：

| 策略 | 公式 | 特点 |
|---|---|---|
| **Greedy** | $\\hat y_t = \\arg\\max_y p(y \\| \\cdot)$ | 确定性、易重复、保守 |
| **Temperature** | $p_T(y) \\propto p(y)^{1/T}$ | $T<1$ 更确定，$T>1$ 更随机 |
| **Top-k** | 在概率最高的 $k$ 个里重新归一化采样 | 截断长尾、避免低概率烂 token |

> **Temperature 数学**：$p_T(y) = \\text{softmax}(\\log p(y) / T)$。
> $T \\to 0$ 退化为 greedy；$T \\to \\infty$ 退化为均匀；
> $T = 1$ 不变。$T$ 控制"自信度"——温度低 = 模型更自信（更敢押注高概率 token）。

> **Top-k 直觉**：vocab 里大量低概率 token（噪声、错别字）几乎不该采。
> Top-k 只保留前 $k$ 个候选（如 $k=5$），把剩下的概率质量归零，重新归一化后采样。
> 既保留多样性，又避免极端输出。

### 10.8.2 实验：同一模型同一 prompt，不同策略的输出对比
"""),
        code("""# 10.8.3 同一 prompt 不同 sampling 策略对比
model.eval()
prompt = "Q: What is the color of the sky"
prompt_ids = tok.encode(prompt)
print(f"prompt: {prompt!r}\\n")

strategies = [
    ("greedy",           dict(greedy=True)),
    ("temperature=0.3",  dict(temperature=0.3, top_k=None)),
    ("temperature=1.0",  dict(temperature=1.0, top_k=None)),
    ("temperature=1.5",  dict(temperature=1.5, top_k=None)),
    ("top-k=3 (T=0.7)",  dict(temperature=0.7, top_k=3)),
    ("top-k=10 (T=0.7)", dict(temperature=0.7, top_k=10)),
]

for name, kw in strategies:
    print(f"--- {name} ---")
    for trial in range(2):
        torch.manual_seed(trial * 7 + 1)
        out = generate(model, prompt_ids, max_new_tokens=35, **kw)
        gen = tok.decode(out[0]).replace(prompt, "").strip()
        print(f"  trial {trial}: {gen!r}")
    print()

print("\\n观察:")
print("  greedy 永远输出一样；高 temperature 输出多样但可能跑偏")
print("  top-k 在多样性和质量之间平衡——是 LLM 生产环境的主流选择")
"""),
        code("""# 10.8.4 可视化：temperature 如何改变概率分布
# 取一个真实位置的 logits 看分布
demo_input = tok.encode("Q: What is the color of the sky?\\nA:")
with torch.no_grad():
    logits = model(demo_input.unsqueeze(0))[0, -1]  # 最后位置的 logits

top_n = 15
top_vals, top_idx = torch.topk(logits, top_n)
top_chars = [tok.itos[i] for i in top_idx.tolist()]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, T in zip(axes, [0.5, 1.0, 2.0]):
    scaled = top_vals / T
    probs = F.softmax(scaled, dim=0).numpy()
    colors = ['#d62728' if c == 'T' else '#1f77b4' for c in top_chars]  # 高亮 'T'（"color of the sky is...")
    ax.bar(range(top_n), probs, color=colors, alpha=0.7)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([repr(c) for c in top_chars], rotation=45, fontsize=8)
    ax.set_title(f'T = {T}（{"更确定" if T<1 else "更随机" if T>1 else "原分布"}）')
    ax.set_ylabel('probability'); ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, max(0.5, probs.max()*1.2))
plt.suptitle('Temperature 对下一个 token 分布的影响\\n（红 = 模型最自信的 "T"）', fontsize=12)
plt.tight_layout(); plt.show()
"""),
        md("""### 10.8.5 关键公式再回顾

| 公式 | 含义 | 出现节 |
|---|---|---|
| $\\text{Attention}(Q,K,V) = \\text{softmax}(QK^T/\\sqrt{d_k})V$ | scaled dot-product attention | §10.3（核心） |
| $\\text{Var}(q \\cdot k) = d_k$，$\\text{Var}(q \\cdot k/\\sqrt{d_k}) = 1$ | 为什么除 $\\sqrt{d_k}$（防 softmax 饱和） | §10.3.3（证明） |
| $\\text{MultiHead} = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O$ | 多头并行 | §10.4 |
| $\\text{FFN}(x) = W_2\\,\\text{GELU}(W_1 x + b_1) + b_2$ | 逐位置非线性 | §10.5 |
| $\\mathcal{L}_{LM} = -\\frac{1}{T}\\sum_t \\log p_\\theta(y_{t+1} \\| y_{\\le t})$ | next-token cross-entropy | §10.6 |
| $\\mathcal{L}_{SFT} = -\\frac{1}{\\|R\\|}\\sum_{t \\in R} \\log p_\\theta(y_t \\| x, y_{<t})$ | SFT loss（只在 response 上） | §10.7 |
| $p_T(y) = \\text{softmax}(\\log p(y)/T)$ | temperature sampling | §10.8 |

### 10.8.6 本章收获

1. **char-level tokenizer + sinusoidal PE** = 把文本变成模型能吃的向量
2. **self-attention = 软性查表**：$\\text{softmax}(QK^T/\\sqrt{d_k})V$
3. **为什么要除 $\\sqrt{d_k}$**：点积方差 $= d_k$，不缩放 softmax 会饱和、梯度消失
4. **multi-head**：让模型同时学多种关系（语法、语义、长程）
5. **causal mask**：下三角保证自回归，LLM 状态 = 已生成 token 序列（近似马尔可夫）
6. **Transformer block** = Pre-LN + attention + FFN + 双残差
7. **next-token cross-entropy + teacher forcing** = 高效训练
8. **SFT**：在 prompt→response 上做条件 LM，**只在 response 上算 loss**（关键工程点）
9. **sampling**：greedy / temperature / top-k 三选一，生产环境常用 top-k + 中等 temperature

### 10.8.7 兑现的承诺核对

| 出处 | 承诺 | 兑现 |
|---|---|---|
| **Ch00** | "Ch10 从零搭 TinyGPT，**注意力热力图**" | §10.4.6 全 layer × head 热力图 ✓ |
| **Ch02 §2.1** | 马尔可夫性 + "LLM 上下文窗口就是为近似马尔可夫性服务的" | §10.1.3 + §10.4.2 causal mask 论证 ✓ |
| **Ch00** | "Ch06+ PyTorch" | 全章复用 Ch06-09 的 PyTorch 基础设施 ✓ |

---

## 下一章预告：Ch11 Reward Modeling

我们现在的 TinyGPT 已经会"答问题"（SFT 后）。但 SFT 学的是"模仿训练数据"——
数据里有什么就学什么，**不知道什么是"好"什么是"坏"**。

下一章 Ch11 解决这个问题：

> **训练一个 reward model $r(x, y)$**，让它预测人类觉得哪个 response 更好。

核心数学是 **Bradley-Terry 模型**：

$$P(y_w \\succ y_l | x) = \\sigma\\big(r(x, y_w) - r(x, y_l)\\big)$$

训练 loss（成对偏好数据）：

$$\\mathcal{L}_{RM} = -\\log \\sigma\\big(r(x, y_w) - r(x, y_l)\\big)$$

架构上 reward model = TinyGPT + 一个 scalar head（把 $d_{model}$ 维隐状态压成 1 个数）。

我们还会讨论 **reward 过优化**（Goodhart's Law）——这是 Ch12 RLHF-PPO 里 KL penalty 的动机。

**到 Ch13**，把本章的 SFT 模型当 actor、Ch11 的 RM 当 reward、用 PPO/GRPO 优化——
**那就是项目的终极目标**。
"""),
        # ---------------------------------------------------------------------
        # 最终: Phase 3 起点示意
        # ---------------------------------------------------------------------
        code("""# Phase 3 起点：本章完成的 TinyGPT 是后续章节的基础
print("=" * 60)
print("Ch10 完成 —— Phase 3 基础设施就位")
print("=" * 60)
print("本章交付:")
print(f"  - rlenvs/tiny_gpt.py: TinyGPT 模型 + tokenizer + sampling")
print(f"      参数量: {n_params:,}（CPU 可训）")
print(f"      训练 loss: {train_losses[0]:.3f} → {train_losses[-1]:.3f}")
print(f"  - data/tiny_corpus.txt: 合成训练语料")
print(f"  - notebooks/ch10_tiny_gpt.ipynb: 本章")
print(f"  - tests/test_tiny_gpt.py: 20 个冒烟测试")
print()
print("Phase 3 路线图:")
print("  Ch10 TinyGPT (本章)    ✓  base + SFT 模型")
print("  Ch11 Reward Modeling   下  Bradley-Terry + 偏好数据 + 过优化")
print("  Ch12 RLHF-PPO             4 模型 + KL penalty + PPO on tokens")
print("  → Ch13 GRPO ★终极目标    group sampling + 无 critic")
print("  Ch14 DPO/KTO              避免 RL 的替代方案")
print("=" * 60)
"""),

        md("""## 10.9 📝 练习

### 练习 1（必做）：Post-LN vs Pre-LN

本章的 `TransformerBlock` 是 Pre-LN（Xiong et al. 2020 推荐）。**任务**：

1. 写一个 `PostLNBlock`：把残差加法放在 LayerNorm 之前（h = LayerNorm(x + Attention(x))）
2. 用同样的层数/宽度在 tiny_corpus 上训练两者（同 seed），对比 loss 曲线
3. 观察前 100 步：Post-LN 的 loss 是否更难降、更震荡？

<details><summary>提示</summary>

- 只改 `forward` 里三行的顺序，参数量不变，对比才公平
- Pre-LN 的残差路径上没有归一化，梯度能直通底层——深网络训练更稳；toy 规模（2-4 层）差异不大，但趋势可见
</details>

**预期结果**：2 层时两者接近；把 n_layers 调到 4-6，Post-LN 训练明显更不稳定。

### 练习 2（选做）：采样策略系统对比

把本章 §10.8 的采样对比做成小实验矩阵：temperature ∈ {0.5, 1.0, 1.5} × top_k ∈ {None, 5}，固定 prompt 各采 10 条，人工归类输出质量（连贯/重复/胡言乱语），画成热力表。

*（开放练习，无参考答案——结论写进你自己的笔记更有价值。）
> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch10 的自测题再进入下一章。"""),
    ]


if __name__ == "__main__":
    cells = ch10()
    nb = build_notebook(cells)
    out_path = Path(__file__).parent / "notebooks" / "ch10_tiny_gpt.ipynb"
    save(nb, out_path)
    print(f"Wrote {out_path}")
