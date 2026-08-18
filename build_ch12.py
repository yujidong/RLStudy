"""Build notebooks/ch12_rlhf_ppo.ipynb via nbformat.

Run:  python build_ch12.py
This produces the .ipynb file. Then execute it with nbconvert.
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch12")
cells = _nb.cells            # build_notebooks.py 适配层读取的模块级列表
md, code = _nb.md, _nb.code


# =============================================================================
# Title / overview
# =============================================================================
md(r"""# 第 12 章：RLHF-PPO —— InstructGPT 配方（4 模型协调）

> **Ch10** 给我们一个会接龙的 TinyGPT；**Ch11** 给我们一个会评分的 Reward Model。
> 但 RM 自己不会"生成更好的回答"——它只能判断。
>
> 本章的核心问题：
>
> > **如何用 RL（强化学习）把 SFT 模型 $\pi_{ref}$ 调成"在 RM 眼里更高分"的 $\pi_\theta$？**
>
> 答案是 **InstructGPT 配方**（Ouyang et al. 2022）：把语言生成当成 token-level MDP，
> 用 **PPO**（Ch09）做策略梯度更新，加 **KL penalty** 防止跑偏。

本章兑现 **Ch00 的两个核心承诺**：

| 出处 | 承诺原文 | 本章兑现节 |
|---|---|---|
| **Ch00** | **"Ch12 RLHF-PPO，4 模型仪表盘 + InstructGPT 配方"** | **§12.2 4 模型 + §12.5 完整训练循环** |
| **Ch00** | **"RLHF 三阶段：SFT → RM → PPO"**（隐含） | **§12.1 总览** |
| **Ch02 §2.2** | γ 的取值讨论（"γ∈[0.9,0.99]：越大越难学，但策略越优"） | **§12.4 token-level MDP** |
| **Ch11 §11.6** | KL penalty 预告 | **§12.3 KL penalty 完整推导** |

## 学习目标

1. 理解 **RLHF 三阶段**（SFT → RM → PPO）的整体逻辑
2. 拆解 **4 模型架构**：actor $\pi_\theta$ / critic $V_\phi$ / reward $r$ / reference $\pi_{ref}$
3. **完整推出 KL penalty**：从约束 MDP → Lagrangian → 调整后的 reward
4. 把 LLM 生成建模成 **token-level MDP**（state = prefix, action = next token）
5. 实现 **PPO 在 token 序列上**（复用 Ch09 的 clip / GAE 数学）
6. 跑通 **完整训练循环**：rollout → reward → KL → GAE → PPO update
7. 复现 **Reward hacking 与 KL penalty 的防御作用**
8. 与 **InstructGPT 论文对比**（简化版 vs 论文版的差异）

## 承接的 Ch11 工作

| 模块 | 出处 | 本章用法 |
|---|---|---|
| **TinyGPT** | Ch10 | actor / reference / critic / reward 的 backbone |
| **RewardModel** | Ch11 §11.4 | reward model $r(x, y)$，本章直接复用、冻结 |
| **compute_gae** | Ch08 §8.4 / utils/gae.py | advantage 估计 |
| **compute_clip_objective** | Ch09 §9.3 / utils/ppo.py | PPO-Clip surrogate |
| **PPO + KL early stopping** | Ch09 §9.4-9.5 | 多-epoch 数据重用 |

## 术语速查

| 术语 | 含义 | 首次出现 |
|---|---|---|
| **actor** $\pi_\theta$ | 要训练的策略（生成 response 的 LLM） | §12.2 |
| **critic** $V_\phi$ | 价值函数，预测每个 prefix 的 expected return | §12.2 |
| **reward model** $r$ | 把 (prompt, response) → 标量（Ch11） | §12.2 |
| **reference** $\pi_{ref}$ | 冻结的 SFT 模型，KL penalty 的"锚点" | §12.2 |
| **KL penalty** | $r_{total} = r - \beta \cdot \text{KL}(\pi \| \pi_{ref})$ | §12.3 |
| **token-level MDP** | 把语言生成建模成 step-by-step RL | §12.4 |
| **rollout** | 用当前 actor 采 G 个 response | §12.5 |
| **reward hacking** | RL 找到 RM 的漏洞，proxy reward↑ 但真实质量↓ | §12.6 |

## 本章路线图（8 节）

| 节 | 主题 | 关键产出 |
|---|---|---|
| 12.1 | RLHF 总览 | SFT → RM → PPO 三阶段全貌 |
| 12.2 | 4 模型架构 | actor / critic / reward / reference 职责图 |
| 12.3 | **KL penalty 完整推导** | 约束 MDP → Lagrangian → 调整后 reward |
| 12.4 | Token-level MDP | state/action/transition + γ 选择 |
| 12.5 | **完整训练循环** | rollout + reward + KL + GAE + PPO update |
| 12.6 | Reward hacking 与 KL 防御 | 实验演示 |
| 12.7 | 与 InstructGPT 论文对比 | 简化版 vs 论文版差异表 |
| 12.8 | 小结 + GRPO 预告 | PPO 的 critic 在 LLM 上太贵 → Ch13 |""")

code(r"""# 常规设置：找项目根、载入库
import sys, pathlib, time, math, random, copy
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

# Ch10/Ch11 基础设施
from rlenvs import (
    CharTokenizer, TinyGPT, build_tiny_gpt,
    generate, make_lm_batch, sft_loss,
)
# Ch11 reward model
from utils.reward_model import (
    RewardModel, bradley_terry_loss,
    generate_preference_data, make_preference_batch, pad_to_length,
    reward_accuracy, predict_rewards, true_reward,
)
# 本章新基础设施
from utils import set_seed
from utils.torch_utils import get_device, count_parameters
from utils.rlhf import RLHFConfig, RLHFTrainer, ValueHead

set_seed(42)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# 本章模型很小（< 50k 参数），CPU 反而比 GPU 快（避免数据搬运）。
# 强制用 CPU，避免 trainer 把数据移到 GPU 但模型还在 CPU 的不匹配。
DEVICE = "cpu"
print(f"PyTorch: {torch.__version__}, device = {DEVICE} (CPU 对小模型更快)")
print(f"本章新基础设施: utils/rlhf.py")
print(f"  - ValueHead       (TinyGPT backbone + per-token value head)")
print(f"  - RLHFConfig      (β, γ, λ, clip ε, K epochs, ...)")
print(f"  - RLHFTrainer     (4 模型协调器: actor / critic / reward / reference)")""")

# =============================================================================
# 12.1 RLHF 总览
# =============================================================================
md(r"""## 12.1 RLHF 总览：SFT → RM → PPO 三阶段

### 12.1.1 为什么需要 RLHF？

**Ch10 的 SFT** 教会 TinyGPT "接龙"（next-token prediction），让它能生成通顺的字符序列。
**Ch11 的 RM** 给了我们一个评分器：能判断"哪个 response 更好"。

但 SFT 模型有两个根本问题：

1. **训练目标是 token-level 交叉熵**，但人类偏好是**序列级**判断（"整体上好不好"）。
   SFT 优化的是"每个位置概率"，不直接优化"整段好不好"。
2. **SFT 学的是模仿**（imitation），不是**对齐**（alignment）。
   SFT 数据有限，模型只会复述见过的 pattern，不会主动避免"看似通顺但实际不好"的回答。

RLHF 的洞察：**直接用 RM 的标量 reward 当 RL 的优化目标**，
让策略梯度把 $\pi_\theta$ 推向"RM 评分更高"的方向——这就是"用 RL 对齐 SFT 模型"。

### 12.1.2 三阶段总览

InstructGPT (Ouyang et al. 2022) 把整个 pipeline 拆成三阶段：

```
            Stage 1: SFT                Stage 2: RM              Stage 3: RLHF-PPO
        ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
prompt  │ 监督数据          │       │ 偏好数据          │       │ 4 模型协调        │
  +     │ (prompt, ideal   │  →    │ (prompt, y_w,    │  →    │ actor + critic   │
answer  │  response)       │       │  y_l)            │       │ + reward + ref   │
        └──────────────────┘       └──────────────────┘       └──────────────────┘
             ↓                          ↓                          ↓
        π_SFT (base)               r_θ(x, y)               π_RLHF (= actor π_θ)
```

| 阶段 | 数据 | 损失 | 产出 | 出处 |
|---|---|---|---|---|
| **1. SFT** | (prompt, ideal response) | next-token CE | $\pi_{ref}$ (= $\pi_{SFT}$) | Ch10 §10.7 |
| **2. RM** | (prompt, winner, loser) | Bradley-Terry | $r_\theta(x, y)$ | Ch11 §11.5 |
| **3. PPO** | rollout from $\pi_\theta$ | PPO-Clip + KL penalty | $\pi_\theta$（**本章**） | 本章 |

> **关键洞察**：三阶段是"**两轮优化**"——
> 第一轮（SFT）用模仿学习教模型"怎么说话"；
> 第二轮（RM + PPO）用 RL 教模型"说什么更好"。

### 12.1.3 本章聚焦：Stage 3（RLHF-PPO）

本章假设 Stage 1（SFT）和 Stage 2（RM）已经完成（Ch10 / Ch11），重点是 **Stage 3**：
用 RM 的标量 reward 当 RL 优化目标，把 $\pi_{ref}$ 调成 $\pi_\theta$。

但这里有个**核心矛盾**：
- 我们想最大化 $r(x, y)$（RM 评分）
- 但 RM 是个**proxy**（Ch11 §11.6 的 Goodhart's Law），直接最大化会把 $\pi_\theta$ 推到 RM 的漏洞里
- 所以要加 **KL penalty**：让 $\pi_\theta$ 不能跑离 $\pi_{ref}$ 太远

这就是本章 §12.3 要推导的核心公式：

$$\boxed{\;\text{RLHF 目标} = \mathbb{E}_{x, y \sim \pi_\theta}\big[ r(x, y) \big] - \beta \cdot \mathbb{E}_x\big[ \text{KL}(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)) \big]\;}$$""")

code(r"""# 12.1.4 复用 Ch10/Ch11 的成果：先训好一个 tokenizer + SFT + RM
# （RM 用 Ch11 的合成数据；SFT 这里用未训练的 TinyGPT 作 ref——
#  教学简化：我们不做完整 SFT，而是把"随机初始化 + 一点 warmup"的 TinyGPT
#  当作 reference。这样能展示"RLHF 让 response 在 RM 眼里变好"的趋势。）

# 1) Tokenizer (与 Ch11 一致)
corpus = (
    "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    "Q: What do you think? A: Q: Is it good? A: Q: Tell me a word. A: Q: How are you? A: "
)
tok = CharTokenizer().train(corpus)
print(f"tokenizer vocab size: {tok.vocab_size}")
print(f"vocab: {''.join(tok.itos)}")
print()

# 2) 合成偏好数据（与 Ch11 一致）
KEYWORD_W = 3.0
LEN_W = 0.3
TARGET_LEN = 6
train_prefs = generate_preference_data(
    tok, n_samples=300, seed=0,
    keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN,
)
val_prefs = generate_preference_data(
    tok, n_samples=80, seed=999,
    keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN,
)
print(f"训练偏好对: {len(train_prefs)}, 验证偏好对: {len(val_prefs)}")
print()

# 3) 训练 reward model (Ch11 的 pipeline)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
RM_D_MODEL = 32
RM_STEPS = 500
rm_backbone = build_tiny_gpt(
    vocab_size=tok.vocab_size,
    d_model=RM_D_MODEL, n_heads=4, n_layers=2, d_ff=64, max_seq_len=64,
)
reward_model = RewardModel(rm_backbone)
print(f"Reward model 参数量: {count_parameters(reward_model):,}")

rm_opt = torch.optim.AdamW(reward_model.parameters(), lr=1.5e-3, weight_decay=0.01)
t0 = time.time()
rm_losses = []
for step in range(RM_STEPS):
    reward_model.train()
    idx = random.sample(range(len(train_prefs)), 32)
    bs = [train_prefs[i] for i in idx]
    b = make_preference_batch(bs, pad_id=tok.pad_id)
    loss = bradley_terry_loss(reward_model, b['prompt_ids'], b['winner_ids'], b['loser_ids'])
    rm_opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
    rm_opt.step()
    rm_losses.append(loss.item())

final_acc = reward_accuracy(reward_model, val_prefs, pad_id=tok.pad_id)
print(f"\nRM 训练: {RM_STEPS} 步, 耗时 {time.time()-t0:.1f}s")
print(f"val accuracy: {final_acc:.1%}  (门槛 70%)")
print(f"通过 70% 门槛: {'是' if final_acc > 0.70 else '否'}")

# 训练完冻结 RM（RLHF 阶段不再动）
for p in reward_model.parameters():
    p.requires_grad_(False)
reward_model.eval()""")

code(r"""# 12.1.5 看看 RM 学到了什么：常用 response 的 reward 分布
# （和 Ch11 §11.5 类似，但本章要用它当 RL 的优化目标）
import textwrap

# 列举 Ch11 response pool 里所有出现过的 response
all_responses = sorted({s['winner'] for s in train_prefs} | {s['loser'] for s in train_prefs})

probe_prompt = tok.encode("Q: How is the weather? A:").unsqueeze(0)
print(f"RM 评分（probe prompt = 'Q: How is the weather? A:'）:")
print(f"  {'response':<15}  {'RM reward':>10}  {'true reward':>12}")
print(f"  {'-'*15}  {'-'*10}  {'-'*12}")
with torch.no_grad():
    for r in sorted(all_responses, key=lambda x: -reward_model(probe_prompt, tok.encode(x).unsqueeze(0)).item()):
        rm_r = reward_model(probe_prompt, tok.encode(r).unsqueeze(0)).item()
        gt_r = true_reward("", r, keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN)
        print(f"  {r:<15}  {rm_r:>+10.3f}  {gt_r:>+12.3f}")

print("\n关键观察: RM 学到了 'good' / 'very good' 等高分 response（RM reward > 0）；")
print("random gibberish 评分较低。这是后续 RLHF 要 'uphill' 的方向。")""")

# =============================================================================
# 12.2 4 模型架构
# =============================================================================
md(r"""## 12.2 4 模型架构：actor / critic / reward / reference

RLHF-PPO 同时协调 **4 个神经网络**。这是和 CartPole-PPO（Ch09，只有 actor + critic）
最大的区别，也是本章的工程难点。

### 12.2.1 4 模型职责图

```
                          ┌─────────────────────┐
                          │  reference π_ref    │   ← 冻结（SFT 后的 TinyGPT）
                  ref log │  (frozen TinyGPT)   │      用于算 KL penalty
              ┌───────────┤                     │
              │           └─────────────────────┘
              │                       ↑
              │                       │ (KL anchor)
              ▼                       │
┌─────────────────────┐     ┌─────────────────────┐
│  actor π_θ          │     │  reward r(x, y)     │   ← 冻结（Ch11 训好的 RM）
│  (训练中 TinyGPT)   │────▶│  (frozen RewardModel)│      标量 reward
│  forward → logits   │     │  forward → scalar    │
└─────────────────────┘     └─────────────────────┘
        ▲                              │
        │ PPO update                   │ r(x, y)
        │                              ▼
┌─────────────────────┐     ┌─────────────────────┐
│  critic V_φ         │     │  KL penalty          │
│  (训练中 ValueHead) │◀────│  -β · log(π/π_ref)   │
│  forward → V(s_t)   │     │  + r(x,y) (last)     │
└─────────────────────┘     └─────────────────────┘
                                  ↓
                          per-token reward r_t
```

### 12.2.2 4 模型一览表

| 模型 | 符号 | 实现 | 训练 | 输入 → 输出 |
|---|---|---|---|---|
| **actor** | $\pi_\theta$ | TinyGPT | **训练** | `prefix [B, T]` → `logits [B, T, V]` |
| **critic** | $V_\phi$ | ValueHead | **训练** | `prefix [B, T]` → `values [B, T]` |
| **reward** | $r$ | RewardModel (Ch11) | 冻结 | `(prompt, resp)` → `scalar [B]` |
| **reference** | $\pi_{ref}$ | TinyGPT | 冻结 | `prefix [B, T]` → `logits [B, T, V]` |

### 12.2.3 为什么需要 4 个？

| 模型 | 解决的问题 |
|---|---|
| **actor** | 要优化的对象（策略）。每个 token 给一个 logits 分布 |
| **critic** | PPO 用 GAE 算 advantage，需要 $V_\phi(s_t)$（baseline）。**没有 critic → 高方差**（Ch07 §7.6 / Ch08） |
| **reward** | RL 的优化目标来源。Ch11 训好的 RM 当**冻结的 reward 函数** |
| **reference** | KL penalty 的"锚点"。防止 actor 跑离 SFT 模型太远（reward hacking） |

> **关键工程点**：reference 和 reward 在 RLHF 阶段**都冻结**。
> 这跟"Actor-Critic"两个网络训练、其他两个只读，是 InstructGPT 配方的核心约定。

### 12.2.4 与 CartPole-PPO 的对比（Ch09）

| 维度 | CartPole-PPO (Ch09) | RLHF-PPO (本章) |
|---|---|---|
| 模型数 | 2（actor + critic，共享 backbone） | **4**（actor + critic + reward + reference） |
| "state" | 4 维向量 `[cart_pos, cart_vel, pole_ang, pole_vel]` | **token 序列**（变长 prefix） |
| "action" | 离散 `{left, right}` | 词表里的某个 token（vocab_size 维） |
| reward 来源 | 环境 `env.step()` | **reward model** $r(x, y)$ |
| 训练阶段 reward | 每步都有（+1 / -10） | **稀疏**：只有最后一个 token 有 r(x, y)；其它靠 KL penalty |
| 数据 shape | `[N, state_dim]` 定长 | `[B, T_p + T_r]` 变长（需要 padding） |""")

code(r"""# 12.2.5 实例化 4 模型
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# 共享的小模型配置（CPU 上每步 < 0.1s）
GPT_D_MODEL = 32
GPT_N_HEADS = 4
GPT_N_LAYERS = 2
GPT_D_FF = 64
GPT_MAX_SEQ = 64

def make_gpt():
    # 造一个新的小 TinyGPT（用于 actor / reference / critic backbone）
    return build_tiny_gpt(
        vocab_size=tok.vocab_size,
        d_model=GPT_D_MODEL, n_heads=GPT_N_HEADS,
        n_layers=GPT_N_LAYERS, d_ff=GPT_D_FF,
        max_seq_len=GPT_MAX_SEQ,
    )

# Actor：要训练的策略
class Actor(nn.Module):
    # 薄包装：让 backbone 暴露 forward(ids) -> logits 接口（和 generate 兼容）
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, ids):
        return self.backbone(ids)

# 1) actor π_θ
actor = Actor(make_gpt())
# 2) reference π_ref (与 actor 初始权重相同 —— SFT 后的快照)
reference = Actor(make_gpt())
reference.load_state_dict(actor.state_dict())  # 初始 π_ref == π_θ
# 3) critic V_φ (复用 TinyGPT 作 backbone + value head)
critic = ValueHead(make_gpt(), d_model=GPT_D_MODEL)
# 4) reward model r (Ch11 训好的，已冻结)
# (已在 12.1.4 训练好)

print("4 模型架构实例化完成:")
print(f"  actor (π_θ):       {count_parameters(actor):,} params (trainable)")
print(f"  reference (π_ref): {count_parameters(reference):,} params (frozen)")
print(f"  critic (V_φ):      {count_parameters(critic):,} params (trainable, value head only)")
print(f"  reward (r):        {count_parameters(reward_model):,} params (frozen)")
print(f"\n总计 (训练中): actor + critic = {count_parameters(actor) + count_parameters(critic):,} params")
print(f"     (冻结):    reference + reward = {count_parameters(reference) + count_parameters(reward_model):,} params")

# 验证 reference == actor 初始权重（KL should be 0 initially）
test_prompt = tok.encode("Q: How A:").unsqueeze(0)
with torch.no_grad():
    la = actor(test_prompt)
    lr = reference(test_prompt)
    print(f"\n初始 KL(actor || reference) 应该 = 0: max |logits diff| = {(la - lr).abs().max().item():.2e}")""")

# =============================================================================
# 12.3 KL penalty 推导
# =============================================================================
md(r"""## 12.3 KL penalty 完整推导：从约束到正则

这是本章数学最核心的一节。目标：**严格推出**为什么 RLHF 的"调整后 reward"长这样：

$$r_{\text{total}}(x, y) = r(x, y) - \beta \cdot \log\frac{\pi(y|x)}{\pi_{ref}(y|x)}$$

### 12.3.1 起点：约束优化问题

我们想最大化 RM 给的 reward，但**不能让 $\pi_\theta$ 跑离 $\pi_{ref}$ 太远**（否则就 reward hacking）：

$$\max_\pi \;\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)}\big[ r(x, y) \big] \quad \text{s.t.} \quad \mathbb{E}_{x \sim \mathcal{D}}\big[\text{KL}\big(\pi(\cdot|x) \,\|\, \pi_{ref}(\cdot|x)\big)\big] \le \epsilon$$

- 左边：**期望 reward**（在 prompt 分布 $\mathcal{D}$ 和策略 $\pi$ 下）
- 约束：每个 prompt 上的 KL 散度平均不超过 $\epsilon$
- $\epsilon$ 是预算：允许策略偏离 reference 多少

> **为什么是 KL 而不是别的距离？**
> KL 散度 $\text{KL}(\pi \| \pi_{ref}) = \sum_y \pi(y|x) \log\frac{\pi(y|x)}{\pi_{ref}(y|x)}$
> 是分布空间里"自然"的距离度量（信息几何）。它惩罚 $\pi$ 在 $\pi_{ref}$ 几乎不分配概率的 $y$ 上突然给高概率——正是 reward hacking 的典型表现。

### 12.3.2 Lagrangian 转化

约束优化 → 无约束：引入 Lagrange 乘子 $\beta \ge 0$：

$$L(\pi, \beta) = \mathbb{E}_{x, y}\big[ r(x, y) \big] - \beta \cdot \Big( \mathbb{E}_x\big[\text{KL}(\pi \| \pi_{ref})\big] - \epsilon \Big)$$

对偶问题（KKT 条件）：

$$\max_\pi \min_{\beta \ge 0} L(\pi, \beta) \quad \Longleftrightarrow \quad \min_{\beta \ge 0} \max_\pi L(\pi, \beta)$$

$\epsilon$ 是常数，对 $\pi$ 的优化没影响，所以**对 $\pi$ 来说**等价于：

$$\max_\pi \;\mathbb{E}_{x, y}\big[ r(x, y) \big] - \beta \cdot \mathbb{E}_x\big[\text{KL}\big(\pi(\cdot|x) \,\|\, \pi_{ref}(\cdot|x)\big)\big]$$

### 12.3.3 求最优策略 $\pi^*$（变分法 / Gibbs 不等式）

把内层期望展开：

$$\mathbb{E}_{x \sim \mathcal{D}} \bigg[ \sum_y \pi(y|x) \, r(x, y) - \beta \sum_y \pi(y|x) \log\frac{\pi(y|x)}{\pi_{ref}(y|x)} \bigg]$$

对每个 $x$，内层是关于 $\pi(\cdot|x)$ 的泛函：

$$F_x[\pi] = \sum_y \pi(y) \, r(x, y) - \beta \sum_y \pi(y) \log\frac{\pi(y)}{\pi_{ref}(y|x)}$$

约束 $\sum_y \pi(y) = 1$。用 Lagrange 乘子 $\lambda(x)$ 求 $\pi^*$：

$$\frac{\partial}{\partial \pi(y)} \Big[ F_x - \lambda(x) \big(\sum_y \pi(y) - 1\big) \Big] = 0$$

$$r(x, y) - \beta \bigg[ \log\frac{\pi(y)}{\pi_{ref}(y|x)} + 1 \bigg] - \lambda(x) = 0$$

解出 $\pi(y)$：

$$\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} = \frac{r(x, y) - \lambda(x) - \beta}{\beta}$$

$$\boxed{\;\pi^*(y|x) = \pi_{ref}(y|x) \cdot \exp\!\bigg( \frac{r(x, y)}{\beta} \bigg) \bigg/ Z(x) \;}$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(r(x, y)/\beta)$ 是归一化常数（被 $\lambda$ 吸收）。

### 12.3.4 等价的"reward shaping"形式

**关键观察**：$\pi^*$ 是 $\pi_{ref}$ 经过 $\exp(r/\beta)$ reweighting 的分布。
这等价于在 RL 里优化一个**调整后的 reward**：

$$\tilde{r}(x, y) = r(x, y) - \beta \log\frac{\pi(y|x)}{\pi_{ref}(y|x)}$$

**证明**：考虑一个新的 MDP，其 reward 是 $\tilde{r}$。最优策略最大化

$$\mathbb{E}_{y \sim \pi}\big[ \tilde{r}(x, y) \big] = \mathbb{E}_y [r(x, y)] - \beta \, \text{KL}(\pi \| \pi_{ref})$$

对 $\pi$ 求变分最优，得到的 $\pi^*$ 与 §12.3.3 完全一致——**所以两种形式等价**。

> **这是 RLHF 工程上最常用的形式**：不用真的去解 §12.3.3 的归一化积分（在 token 空间里不可行），
> 而是把 KL penalty 当成 reward shaping，直接喂给标准 PPO。

### 12.3.5 Per-token 形式（实际实现）

对一条 response $y = (y_0, y_1, \dots, y_{T-1})$，假设 token 级独立（其实是自回归条件独立）：

$$\log \frac{\pi(y|x)}{\pi_{ref}(y|x)} = \sum_{t=0}^{T-1} \log \frac{\pi(y_t | x, y_{<t})}{\pi_{ref}(y_t | x, y_{<t})}$$

所以 KL penalty 可以**分摊到每个 token**：

$$r_t = \begin{cases} -\beta \cdot \log\dfrac{\pi(a_t | s_t)}{\pi_{ref}(a_t | s_t)} & t = 0, 1, \dots, T-2 \\[6pt] r(x, y) - \beta \cdot \log\dfrac{\pi(a_{T-1} | s_{T-1})}{\pi_{ref}(a_{T-1} | s_{T-1})} & t = T-1 \end{cases}$$

- 前 $T-1$ 步：每步只算 KL 项（reward model 还没看到完整 response，没法打分）
- 最后一步：加上 reward model 的标量 $r(x, y)$

### 12.3.6 β 的物理意义

| $\beta$ | 行为 | 物理意义 |
|---|---|---|
| $\beta \to \infty$ | $\pi^* \to \pi_{ref}$ | KL 约束极严，几乎不动 |
| $\beta$ 适中 | $\pi^*$ 平衡 reward + KL | InstructGPT 用 0.01-0.1 |
| $\beta \to 0$ | $\pi^*$ 趋向 $\arg\max_y r(x, y)$ | 纯 reward 最大化 → Goodhart's Law |

**教学结论**：$\beta$ 是**对齐 vs 性能**的旋钮。
- 小 $\beta$ → 高 reward 但远离 SFT（可能 reward hacking）
- 大 $\beta$ → 安全但 reward 提升有限

§12.6 会用实验展示这个 trade-off。""")

code(r"""# 12.3.7 数值演示：β 的 trade-off 曲线
# 我们模拟一个 toy 场景：ref 是均匀分布 over 4 actions，true reward 是 [1, 0, 0, 0]，
# 看不同 β 下 π* 的形状、reward、KL。

# toy: V=4 actions, π_ref = uniform, r = [1, 0, 0, 0]
π_ref = torch.tensor([0.25, 0.25, 0.25, 0.25])
r = torch.tensor([1.0, 0.0, 0.0, 0.0])

betas = np.linspace(0.05, 2.0, 50)
rewards = []
kls = []
for beta in betas:
    # π*(y) ∝ π_ref(y) exp(r(y)/β)
    logp = torch.log(π_ref) + r / beta
    π_star = torch.softmax(logp, dim=0)
    # E[r] = sum_y π*(y) * r(y)
    er = float((π_star * r).sum())
    # KL(π* || π_ref) = sum_y π*(y) log(π*(y) / π_ref(y))
    kl = float((π_star * (torch.log(π_star + 1e-12) - torch.log(π_ref))).sum())
    rewards.append(er)
    kls.append(kl)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(betas, rewards, 'o-', color='#1f77b4', label=r'$\mathbb{E}_{\pi^*}[r]$')
ax.plot(betas, kls, 's-', color='#d62728', label=r'$\mathrm{KL}(\pi^* \| \pi_{ref})$')
ax.set_xlabel(r'$\beta$ (KL penalty 强度)')
ax.set_ylabel('value')
ax.set_title(r'β trade-off: 大 β → 安全低 reward；小 β → 高 reward 但 KL 大')
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.5)

# 右图：π* 在不同 β 下的形状
ax = axes[1]
for beta in [0.1, 0.3, 0.5, 1.0]:
    logp = torch.log(π_ref) + r / beta
    π_star = torch.softmax(logp, dim=0).numpy()
    ax.bar(np.arange(4) + beta/3, π_star, width=0.1, label=f'β={beta}')
ax.set_xticks(np.arange(4))
ax.set_xticklabels([f'a{i}\n(r={r[i]:.0f})' for i in range(4)])
ax.set_ylabel(r'$\pi^*(a)$')
ax.set_title(r'不同 β 下 $\pi^*$ 分布：小 β → 集中在 argmax；大 β → 接近 ref')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(str(ROOT / 'assets' / 'ch12_beta_tradeoff.png'), dpi=110, bbox_inches='tight')
plt.show()

print("观察: β=0.1 时 π* 几乎全压在 a0 (argmax reward)，但 KL 大；")
print("      β=1.0 时 π* 接近 uniform ref，reward 提升小但 KL 小。")""")

# =============================================================================
# 12.4 Token-level MDP
# =============================================================================
md(r"""## 12.4 Token-level MDP：把 LLM 生成建模成 RL

### 12.4.1 把语言生成看成 MDP

RLHF 把"生成一段 response"看成 token-by-token 的决策过程，每一步是一个 MDP step：

| MDP 元素 | LLM RLHF 中的含义 |
|---|---|
| **state** $s_t$ | 已生成的 prefix $(x, y_{<t})$ |
| **action** $a_t$ | 下一个 token $y_t$（在词表 $V$ 里选一个） |
| **transition** | 确定性 $s_{t+1} = (s_t, a_t) = (x, y_{<t+1})$ |
| **reward** $r_t$ | KL penalty 每步都有；最后一步加 RM 的 $r(x, y)$（§12.3.5） |
| **terminal** | 生成到 max_len 或 EOS |
| **policy** $\pi_\theta(a_t \| s_t)$ | actor 给下一个 token 的 softmax 分布 |

### 12.4.2 为什么用 token-level 而不是 sequence-level？

**理论上**可以两种建模：

1. **Sequence-level**：1 step = 1 个完整 response，reward 是 $r(x, y)$
2. **Token-level**：$T$ steps = $T$ 个 token，reward 是 §12.3.5 的 per-token

InstructGPT 选 token-level，原因：
- **信用分配**：sequence-level 看不到"哪个 token 贡献了 reward"——一个标量怎么分给 30 个 token？
- **GAE 优势**：token-level 可以用 GAE 把 reward 反传到早期 token（"开头的好 token 也会得到 reward"）
- **更细的控制**：KL penalty 也是 per-token 的，自然对齐

### 12.4.3 γ 选择（应用 Ch02 §2.2 的原则）

**Ch02 §2.2 的原则**：γ 越大越难学、但策略越优——RLHF 里 response 只有一轮对话的长度，不需要趋近 1。

| 任务类型 | 典型 response 长度 | 推荐 γ |
|---|---|---|
| 单轮 QA | 5-50 token | 0.9-0.95 |
| 多轮对话 | 100-500 token | 0.95-0.99 |
| 长篇生成（文章） | 1000+ token | 0.99-0.999 |

**为什么 0.95 不是 1.0？**
- $\gamma = 1$：所有 token 同等权重——但早期 token 其实影响后面所有 token 的"路径"
- $\gamma < 1$：后期 token 折扣更多——鼓励"早出关键 token"，符合 prompt-response 的因果性
- 0.95 是经验最佳点：$0.95^{20} \approx 0.36$，20 token 后的 reward 衰减到 1/3——仍能传信号

**为什么不是更小（如 0.5）？**
- $\gamma$ 太小 → 信号传不远 → "long-context reasoning" 学不到
- $\gamma = 0.5$ 时 $0.5^{10} \approx 0.001$，10 token 之外的 reward 几乎看不见

本章用 **γ = 0.95**（§12.5 配置）。

### 12.4.4 GAE 在 token 序列上

GAE（Ch08）在 token-level MDP 上的形式完全不变：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

只不过现在：
- $r_t$ 是 §12.3.5 的 per-token reward
- $V_\phi(s_t) = $ critic 对 prefix $(x, y_{<t})$ 的 value 估计
- $T$ 是这条 response 的长度

**bootstrap**：最后一个 token 后 episode 自然终止（terminal），$V_\phi(s_T) = 0$。

### 12.4.5 Critic 的输出形式

reward model 只输出**整段一个标量**；critic 要输出**每个 token 一个 value**：

```
        prompt tokens   |   response tokens
        Q :   space A   |   g  o  o  d     !
s_t     ↑               |   ↑  ↑  ↑  ↑     ↑   ← "state"  = prefix up to here
V(s_t)                  |   ?  ?  ?  ?     ?   ← critic predicts each
```

实现上，让 critic 复用 TinyGPT 作 backbone，加一个 **value head**（Linear(d_model → 1)），
取每个 token 位置的 hidden state 过 value head：

```python
class ValueHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.value_ln = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, 1)
        # 通过 forward hook 抓 ln_final 输入
        backbone.ln_final.register_forward_hook(...)

    def forward(self, input_ids):
        _ = self.backbone(input_ids)
        hidden = self._hidden  # [B, T, d_model]
        # per-token value
        values = self.value_head(self.value_ln(hidden)).squeeze(-1)  # [B, T]
        return values
```

由于 causal mask，hidden[b, t] 只"看到" `input_ids[b, :t+1]`，
所以 `values[b, t] = V_φ((x, y_{<t+1}))` —— 严格对应 state 的定义。""")

code(r"""# 12.4.6 验证 critic 输出 shape + 因果性
# 构造一个简短的 (prompt, response) 看 critic 给出的 value 序列

test_prompt = tok.encode("Q: How A:")
test_resp = tok.encode("good")
full = torch.cat([test_prompt, test_resp]).unsqueeze(0)
print(f"prompt:   '{tok.decode(test_prompt)}'  (len={len(test_prompt)})")
print(f"response: '{tok.decode(test_resp)}'  (len={len(test_resp)})")
print(f"full:     '{tok.decode(full[0])}'  (len={full.size(1)})")
print()

with torch.no_grad():
    values = critic(full)
print(f"critic output shape: {values.shape} (should be [1, T_p + T_r])")
print(f"per-token values:")
T_p = len(test_prompt)
for t in range(full.size(1)):
    prefix = tok.decode(full[0, :t+1])
    v = values[0, t].item()
    marker = ' (response token)' if t >= T_p else ''
    print(f"  pos {t}: prefix={prefix!r:30}  V_φ = {v:+.3f}{marker}")

print()
print("观察: 每个 token 位置一个 V_φ 值；")
print("      pos t 的 V_φ 只依赖 input_ids[:t+1]（causal mask 保证）。")
print("      这正是 §12.4 token-level MDP 里 V_φ(s_t) 的定义。")""")

# =============================================================================
# 12.5 完整训练循环
# =============================================================================
md(r"""## 12.5 完整训练循环：rollout → reward → KL → GAE → PPO update

这是本章的工程核心。我们用 `RLHFTrainer`（`utils/rlhf.py`）把 4 模型协调起来。

### 12.5.1 一个 outer iteration 的伪代码

```python
for outer_iter in range(n_iters):
    # 1. ROLLOUT: 用当前 actor 对 G 个 prompt 各采一个 response
    rollout = rollout_responses(prompts)
    # rollout = {
    #   prompts: [G, T_p], responses: [G, T_r],
    #   log_probs_old: [G, T_r],  # log π_θ(a_t|s_t) at rollout time (detached)
    #   log_probs_ref: [G, T_r],  # log π_ref(a_t|s_t)  (frozen)
    #   values_old: [G, T_r],     # V_φ(s_t) at rollout time (detached)
    # }

    # 2. REWARD + KL PENALTY (per token)
    rm_rewards = reward_model(prompts, responses)         # [G]
    for b in range(G):
        Lr = response_lens[b]
        for t in range(Lr):
            token_rewards[b, t] = -beta * (log_probs_old[b,t] - log_probs_ref[b,t])
        # 最后一个 token 加 RM reward
        token_rewards[b, Lr-1] += rm_rewards[b]

    # 3. GAE ADVANTAGES
    for b in range(G):
        adv = compute_gae(token_rewards[b, :Lr], values_old[b, :Lr],
                          last_value=0, gamma=0.95, lam=0.95)
        advantages[b, :Lr] = adv
        returns[b, :Lr] = adv + values_old[b, :Lr]   # critic target

    # 4. PPO MULTI-EPOCH UPDATE (actor + critic, NOT reward/ref)
    for epoch in range(K):
        for minibatch in batch:
            # 重新 forward actor / critic
            new_log_pi = actor(...).log_softmax(-1)
            new_values = critic(...)
            # importance ratio
            ratio = exp(new_log_pi - log_probs_old)
            # PPO-Clip surrogate
            actor_loss = -E[min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)]
            critic_loss = E[(new_values - returns)^2]
            entropy = -E[sum_a π log π]
            # backward + step (both actor + critic optimizers)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            loss.backward(); opt.step()
        # KL early stopping
        if mean_kl(old || new) > 1.5 * target_kl: break
```

### 12.5.2 关键工程决策

| 决策 | 选择 | 理由 |
|---|---|---|
| **reward / ref 冻结** | `requires_grad_(False)` | RM 训完后不动（InstructGPT 配方） |
| **actor + critic 分别 optim** | 两个 `AdamW` | lr 不同（actor lr < critic lr） |
| **mini-batch 切分** | 按"response"为单位 | 保留 trajectory 结构 |
| **advantage normalization** | 整个 batch 一起 normalize | PPO 工程标配（Ch09 §9.6） |
| **pad 位置不算 loss** | `loss * response_mask` | 否则会被 pad 稀释 |
| **KL early stopping** | 每个 inner epoch 后算 KL(actor_old‖actor_new) | 防止 actor 一步走太远 |

### 12.5.3 与 Ch09 `ppo_update` 的关系

| 维度 | Ch09 `ppo_update` (CartPole) | 本章 `RLHFTrainer.rlhf_update` |
|---|---|---|
| 状态空间 | 定长向量 `[N, state_dim]` | token 序列（变长，需 pad） |
| Advantage 算法 | GAE（同） | GAE（**完全复用** `utils.gae.compute_gae`） |
| Clip objective | `compute_clip_objective`（同） | **完全复用** |
| KL early stopping | approx_kl_from_ratio（同） | **完全复用** |
| Critic target | GAE adv + V_old | GAE adv + V_old（**数学完全一样**） |
| forward | `actor_critic(s) → (dist, V)` | actor / critic 分别 forward（token 序列） |

> **核心信息**：PPO 在 token 序列上和 CartPole 上**数学完全一样**，
> 只是数据 shape 和 forward 调用方式不同。""")

code(r"""# 12.5.4 配置 RLHF 训练
# （应用 Ch02 §2.2 的 γ 原则：单轮对话用 0.95）

cfg = RLHFConfig(
    # KL penalty
    beta=0.02,             # §12.3: 小 β，让 RM 信号占主导（toy RM 量级小）
    # Discount / GAE (Ch02 §2.2: 单轮对话用 0.9-0.95)
    gamma=0.95,
    lam=0.95,
    # PPO
    clip_eps=0.2,          # PPO 论文默认
    update_epochs=4,       # K=4 (Ch09 §9.5 推荐)
    inner_minibatch_size=8,
    entropy_coef=0.002,    # 小一点，避免 LM entropy 太大发散
    value_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.05,        # token-level KL early stop
    # Rollout
    response_max_len=8,
    temperature=1.0,
    top_k=None,
    # Optimizers
    actor_lr=5e-4,         # 小模型可以稍大 lr
    critic_lr=1e-3,
    # Reporting
    print_every=5,
)
print("RLHF 配置:")
for k, v in cfg.__dict__.items():
    print(f"  {k:25} = {v}")""")

code(r"""# 12.5.5 构造 RLHFTrainer + 训练
trainer = RLHFTrainer(
    actor=actor,
    critic=critic,
    reward_model=reward_model,
    reference=reference,
    pad_id=tok.pad_id,
    cfg=cfg,
    device=DEVICE,
)

# 训练用的 prompts pool（每个 outer iter 从中随机抽 G 个）
prompts_pool = [
    tok.encode("Q: How is the weather? A:"),
    tok.encode("Q: Is it good? A:"),
    tok.encode("Q: Tell me a word. A:"),
    tok.encode("Q: How are you? A:"),
    tok.encode("Q: What do you think? A:"),
]
print(f"prompts pool: {len(prompts_pool)} 个 prompt")

# 训练前先记一个 baseline reward（initial actor）
print("\n[训练前 baseline]")
with torch.no_grad():
    baseline_rollout = trainer.rollout_responses(prompts_pool)
    bl_token_rewards, bl_rm_rewards, bl_kl = trainer.compute_token_rewards(
        baseline_rollout['prompts'], baseline_rollout['responses'],
        baseline_rollout['response_lens'],
        baseline_rollout['log_probs_old'], baseline_rollout['log_probs_ref'],
    )
    mask = baseline_rollout['response_mask']
    baseline_rm_mean = float(
        ((bl_token_rewards * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    )
    print(f"  mean per-response reward (RM + KL, sum over tokens): {baseline_rm_mean:+.3f}")
    print(f"  mean RM scalar reward: {bl_rm_rewards.mean().item():+.3f}")
    print(f"  mean KL(actor || ref) per token: {((bl_kl * mask).sum() / mask.sum()).item():+.4f}")
    print(f"  mean response length: {baseline_rollout['response_lens'].float().mean().item():.1f}")

# **保存训练前的 actor 快照**（用于训练后公平对比）
pre_actor_snapshot = copy.deepcopy(actor.state_dict())

# 训练
print("\n[开始 RLHF-PPO 训练]")
N_ITERS = 50
GROUP_SIZE = 12
t0 = time.time()
history = trainer.train(prompts_pool, n_iters=N_ITERS, group_size=GROUP_SIZE, verbose=True)
train_time = time.time() - t0
print(f"\n训练完成: {N_ITERS} iters, 耗时 {train_time:.1f}s ({train_time/N_ITERS:.2f}s/iter)")""")

code(r"""# 12.5.6 训练曲线：4 模型仪表盘（兑现 Ch00 承诺）
# 4 个关键 metric: reward, KL, entropy, response length —— 一眼看出训练健康度

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

iters = np.arange(len(history))

# 1) REWARD (RM 给的 scalar)
ax = axes[0, 0]
rewards = [h['mean_reward'] for h in history]
ax.plot(iters, rewards, color='#1f77b4', alpha=0.4, linewidth=0.7)
if len(rewards) >= 5:
    window = 5
    smooth_r = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(iters[window-1:], smooth_r, color='#1f77b4', linewidth=2.0, label=f'smoothed (w={window})')
ax.axhline(rewards[0], color='gray', linestyle='--', alpha=0.5, label=f'baseline = {rewards[0]:.3f}')
ax.set_xlabel('outer iter')
ax.set_ylabel('mean RM reward')
ax.set_title('Reward (RM scalar, higher = better)')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# 2) KL to reference
ax = axes[0, 1]
kl_ref = [h['mean_kl_to_ref'] for h in history]
ax.plot(iters, kl_ref, color='#d62728', alpha=0.4, linewidth=0.7)
if len(kl_ref) >= 5:
    smooth_kl = np.convolve(kl_ref, np.ones(5)/5, mode='valid')
    ax.plot(iters[4:], smooth_kl, color='#d62728', linewidth=2.0)
ax.set_xlabel('outer iter')
ax.set_ylabel('KL(actor || reference)')
ax.set_title('KL to reference (β 控制: 不应爆炸)')
ax.grid(alpha=0.3)

# 3) Entropy
ax = axes[1, 0]
ent = [h['entropy'] for h in history]
ax.plot(iters, ent, color='#2ca02c', alpha=0.4, linewidth=0.7)
if len(ent) >= 5:
    smooth_e = np.convolve(ent, np.ones(5)/5, mode='valid')
    ax.plot(iters[4:], smooth_e, color='#2ca02c', linewidth=2.0)
ax.set_xlabel('outer iter')
ax.set_ylabel('entropy (nats)')
ax.set_title('Policy entropy (declines = more confident)')
ax.grid(alpha=0.3)

# 4) Response length
ax = axes[1, 1]
rl = [h['mean_response_len'] for h in history]
ax.plot(iters, rl, color='#9467bd', alpha=0.4, linewidth=0.7)
if len(rl) >= 5:
    smooth_rl = np.convolve(rl, np.ones(5)/5, mode='valid')
    ax.plot(iters[4:], smooth_rl, color='#9467bd', linewidth=2.0)
ax.axhline(cfg.response_max_len, color='gray', linestyle='--', alpha=0.5,
           label=f'max_len = {cfg.response_max_len}')
ax.set_xlabel('outer iter')
ax.set_ylabel('tokens')
ax.set_title('Response length (reward hacking 征兆: 长度爆炸)')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle('Ch12 §12.5 RLHF-PPO 4 模型仪表盘 (Ch00 承诺)', fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(str(ROOT / 'assets' / 'ch12_dashboard.png'), dpi=110, bbox_inches='tight')
plt.show()

# 统计训练前后
print(f"\n[训练前后对比]")
print(f"  RM reward:   {rewards[0]:+.3f} → {rewards[-1]:+.3f} (max = {max(rewards):+.3f})")
print(f"  KL to ref:   {kl_ref[0]:+.4f} → {kl_ref[-1]:+.4f} (max = {max(kl_ref):+.4f})")
print(f"  Entropy:     {ent[0]:.3f} → {ent[-1]:.3f}")
print(f"  Resp length: {rl[0]:.1f} → {rl[-1]:.1f}")""")

code(r"""# 12.5.7 训练前后样本对比：看 RLHF 后 response 在 RM 眼里是否更好
# （这是验收的核心：reward 应该上升）

def sample_responses(actor_model, prompts, n_per_prompt=2, max_new=10, temp=1.0):
    # 用当前 actor 对每个 prompt 采 n_per_prompt 个 response（禁止 pad token）
    actor_model.eval()
    results = []
    forbidden = {tok.pad_id}
    with torch.no_grad():
        for p in prompts:
            p_t = p.to(DEVICE).long()
            for _ in range(n_per_prompt):
                # 直接调 trainer 的 _sample_response（保证和训练时同样的采样逻辑）
                backbone = actor_model.backbone if hasattr(actor_model, 'backbone') else actor_model
                p_b = p_t.unsqueeze(0)
                full = RLHFTrainer._sample_response(
                    backbone, p_b, max_new, temperature=temp, forbidden_ids=forbidden
                )
                # full shape [1, T_p + T_r]，取 [0] 后再切片
                resp_ids = full[0, p_t.size(0):]
                results.append({
                    'prompt_ids': p_t,
                    'response_ids': resp_ids,
                    'prompt_str': tok.decode(p_t),
                    'response_str': tok.decode(resp_ids),
                })
    return results

# 用**训练后的 actor** 采一批 response
print("=" * 70)
print("训练前后样本对比（同 prompt, 同初始 actor, 10 samples/prompt）")
print("=" * 70)

# 多采一些样本（10/prompt）来减少 noise
N_SAMPLES_PER_PROMPT = 10

# 训练后的 actor
torch.manual_seed(0)  # 固定采样种子，保证可比
post_samples = sample_responses(actor, prompts_pool, n_per_prompt=N_SAMPLES_PER_PROMPT, max_new=cfg.response_max_len)
# 训练前的 actor: 用保存的 pre_actor_snapshot 装在一个新实例里
pre_actor = Actor(make_gpt())
pre_actor.load_state_dict(pre_actor_snapshot)
torch.manual_seed(0)  # 同样的采样种子
pre_samples = sample_responses(pre_actor, prompts_pool, n_per_prompt=N_SAMPLES_PER_PROMPT, max_new=cfg.response_max_len)

# 用 RM 给每个 response 打分
def rm_score(samples):
    with torch.no_grad():
        # pad
        max_p = max(s['prompt_ids'].size(0) for s in samples)
        max_r = max(s['response_ids'].size(0) for s in samples)
        P = torch.full((len(samples), max_p), tok.pad_id, dtype=torch.long)
        R = torch.full((len(samples), max_r), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(samples):
            P[i, :s['prompt_ids'].size(0)] = s['prompt_ids']
            R[i, :s['response_ids'].size(0)] = s['response_ids']
        return reward_model(P, R).numpy()

pre_scores = rm_score(pre_samples)
post_scores = rm_score(post_samples)

print(f"\n{'prompt':<28}  {'response':<18}  {'RM reward':>10}  {'来源':<8}")
print("-" * 70)
# 配对打印
for prompt_str in sorted({s['prompt_str'] for s in pre_samples}):
    pre_for_p = [(s, r) for s, r in zip(pre_samples, pre_scores) if s['prompt_str'] == prompt_str]
    post_for_p = [(s, r) for s, r in zip(post_samples, post_scores) if s['prompt_str'] == prompt_str]
    for (s, r) in pre_for_p[:1]:
        print(f"{s['prompt_str']:<28}  {s['response_str']!r:<18}  {r:>+10.3f}  {'训练前':<8}")
    for (s, r) in post_for_p[:1]:
        print(f"{s['prompt_str']:<28}  {s['response_str']!r:<18}  {r:>+10.3f}  {'训练后':<8}")
    print()

print(f"\n训练前平均 RM reward: {pre_scores.mean():+.3f}")
print(f"训练后平均 RM reward: {post_scores.mean():+.3f}")
print(f"提升: {post_scores.mean() - pre_scores.mean():+.3f}")
print(f"\n验收: reward 提升 > 0? {'是 ✓' if post_scores.mean() > pre_scores.mean() else '否 ✗'}")""")

# =============================================================================
# 12.6 Reward hacking 与防御
# =============================================================================
md(r"""## 12.6 Reward hacking 与 KL penalty 防御

### 12.6.1 Goodhart's Law 复习（承接 Ch11 §11.6）

> **"当一个度量成为目标时，它就不再是个好度量。"** —— Goodhart 1975

在 RLHF 里：
- **真实偏好** $r^*(x, y)$ 不可观测（在人类大脑里）
- **RM** $r_\theta(x, y)$ 是 $r^*$ 的 proxy（从有限偏好数据学出来的）
- 当我们用 PPO 最大化 $r_\theta$ 时，agent 会找到 RM 的**漏洞**
  → $r_\theta \uparrow$ 但 $r^* \downarrow$

**典型 reward hacking 模式**：
- 重复"good good good good"（RM 看到 'good' 就加分）
- 加奇怪的标点 / 大写（学到 RM 的 artifact）
- response 变得异常长（如果 RM 偏好长 response）

### 12.6.2 KL penalty 作为防御

KL penalty 从两个层面缓解 reward hacking：

1. **直接惩罚**：$\pi_\theta$ 离 $\pi_{ref}$ 越远，KL 项越大，total reward 越低
   → 抑制"跑到 RM 漏洞"的极端 response
2. **隐式正则**：$\pi_{ref}$ 是 SFT 学到的"自然语言" prior，
   偏离它意味着说"非自然语言"——这本身就是 reward hacking 的征兆

### 12.6.3 实验：扫不同 β，看 reward vs KL trade-off

下面我们做几个**短**训练，每个用不同的 β，看 RM reward 和 KL 如何变化。
（实验设计：每个 β 训练相同步数，比较最终 reward 和 KL。）""")

code(r"""# 12.6.4 β 扫描实验：不同 β 下的 reward / KL trade-off
# 用相同 seed, 相同步数, 只改 β

N_BETA_ITERS = 20
BETA_GROUP = 8
BETAS_TO_TRY = [0.0, 0.02, 0.1, 0.5]   # 0 = 无 KL penalty (reward hacking 风险)

beta_results = {}

torch.manual_seed(42); np.random.seed(42); random.seed(42)
for beta in BETAS_TO_TRY:
    print(f"\n=== β = {beta} ===")
    # 重新构造 4 模型（保证每个 β 起点相同）
    bb_actor = make_gpt()
    bb_ref = make_gpt(); bb_ref.load_state_dict(bb_actor.state_dict())
    bb_critic = make_gpt()
    this_actor = Actor(bb_actor)
    this_reference = Actor(bb_ref)
    this_critic = ValueHead(bb_critic, d_model=GPT_D_MODEL)

    this_cfg = RLHFConfig(
        beta=beta, gamma=0.95, lam=0.95,
        clip_eps=0.2, update_epochs=4, inner_minibatch_size=4,
        entropy_coef=0.001, value_coef=0.5, max_grad_norm=0.5,
        target_kl=0.05 if beta > 0 else 1.0,  # β=0 时关 KL early stop
        response_max_len=10, temperature=1.0, top_k=None,
        actor_lr=3e-4, critic_lr=1e-3, print_every=99,
    )
    this_trainer = RLHFTrainer(
        this_actor, this_critic, reward_model, this_reference,
        pad_id=tok.pad_id, cfg=this_cfg, device=DEVICE,
    )
    h = this_trainer.train(prompts_pool, n_iters=N_BETA_ITERS, group_size=BETA_GROUP, verbose=False)
    beta_results[beta] = h

print("\nβ 扫描完成。")""")

code(r"""# 12.6.5 β trade-off 曲线（reward vs KL）

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

colors = {0.0: '#d62728', 0.02: '#ff7f0e', 0.1: '#1f77b4', 0.5: '#2ca02c'}

# 左图：每个 β 的 reward 曲线
ax = axes[0]
for beta in BETAS_TO_TRY:
    h = beta_results[beta]
    rewards = [x['mean_reward'] for x in h]
    iters = np.arange(len(rewards))
    ax.plot(iters, rewards, 'o-', color=colors[beta],
            label=f'β={beta}', markersize=4, alpha=0.8)
ax.set_xlabel('outer iter')
ax.set_ylabel('mean RM reward')
ax.set_title('不同 β 下 reward 演化\n(β=0 无 KL penalty: reward 涨快但风险大)')
ax.legend()
ax.grid(alpha=0.3)

# 右图：每个 β 的 KL 演化
ax = axes[1]
for beta in BETAS_TO_TRY:
    h = beta_results[beta]
    kls = [x['mean_kl_to_ref'] for x in h]
    iters = np.arange(len(kls))
    ax.plot(iters, kls, 'o-', color=colors[beta],
            label=f'β={beta}', markersize=4, alpha=0.8)
ax.set_xlabel('outer iter')
ax.set_ylabel('KL(actor || reference)')
ax.set_title('不同 β 下 KL 演化\n(β 小 → KL 大；β=0 时 KL 可能爆炸)')
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle('Ch12 §12.6 β trade-off 实验', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(str(ROOT / 'assets' / 'ch12_beta_scan.png'), dpi=110, bbox_inches='tight')
plt.show()

# 汇总表
print(f"\n{'β':>5} | {'final reward':>13} | {'max reward':>11} | {'final KL':>10} | {'max KL':>9}")
print("-" * 60)
for beta in BETAS_TO_TRY:
    h = beta_results[beta]
    fr = h[-1]['mean_reward']
    mr = max(x['mean_reward'] for x in h)
    fkl = h[-1]['mean_kl_to_ref']
    mkl = max(abs(x['mean_kl_to_ref']) for x in h)
    print(f"{beta:>5} | {fr:>+13.3f} | {mr:>+11.3f} | {fkl:>+10.4f} | {mkl:>9.4f}")

print()
print("观察:")
print(" - β=0 (无 KL penalty): KL 快速增长, 没有锚 → reward hacking 风险高")
print(" - β=0.5 (强 KL): KL 一直接近 0, reward 提升受限")
print(" - β=0.02-0.1 (中等): reward 上升, KL 可控 → InstructGPT 推荐区间")""")

code(r"""# 12.6.6 Token-level reward 演化（KL penalty 在每个 token 上的贡献）
# 选一条 response，看每个 token 的 reward 拆解（KL penalty vs RM scalar）

# 用当前训好的 actor 采一条 response
test_prompt = tok.encode("Q: Is it good? A:").to(DEVICE)
with torch.no_grad():
    rollout = trainer.rollout_responses([test_prompt])

# 算 per-token reward
token_rewards, rm_rewards, kl_per_token = trainer.compute_token_rewards(
    rollout['prompts'], rollout['responses'],
    rollout['response_lens'],
    rollout['log_probs_old'], rollout['log_probs_ref'],
)
b = 0
resp = rollout['responses'][b]
mask = rollout['response_mask'][b]
Lr = int(rollout['response_lens'][b].item())

print(f"Prompt: '{tok.decode(rollout['prompts'][b])}'")
print(f"Response (len={Lr}): '{tok.decode(resp[:Lr])}'")
print(f"RM scalar reward: {rm_rewards[b].item():+.3f}")
print()
print(f"{'pos':>3}  {'token':>6}  {'KL contribution':>16}  {'RM reward':>10}  {'total r_t':>10}")
print("-" * 60)
for t in range(Lr):
    char = tok.decode([int(resp[t].item())])
    kl_term = -cfg.beta * kl_per_token[b, t].item()
    rm_term = rm_rewards[b].item() if t == Lr - 1 else 0.0
    total = token_rewards[b, t].item()
    print(f"{t:>3}  {char!r:>6}  {kl_term:>+16.4f}  {rm_term:>+10.4f}  {total:>+10.4f}")

# 可视化 per-token reward 拆解
fig, ax = plt.subplots(figsize=(11, 4))
positions = np.arange(Lr)
chars = [tok.decode([int(resp[t].item())]) for t in range(Lr)]
kl_contribs = [-cfg.beta * kl_per_token[b, t].item() for t in range(Lr)]
rm_contribs = [rm_rewards[b].item() if t == Lr - 1 else 0.0 for t in range(Lr)]
ax.bar(positions - 0.2, kl_contribs, width=0.4, color='#d62728', label='KL penalty (-β log π/π_ref)')
ax.bar(positions + 0.2, rm_contribs, width=0.4, color='#1f77b4', label='RM scalar reward (last token only)')
ax.set_xticks(positions)
ax.set_xticklabels([f'pos {t}\n{c!r}' for t, c in enumerate(chars)])
ax.set_ylabel('reward contribution')
ax.set_title(f"Per-token reward 拆解（β={cfg.beta}）\n"
             f"Prompt: '{tok.decode(rollout['prompts'][b])}'  Response: '{tok.decode(resp[:Lr])}'")
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(str(ROOT / 'assets' / 'ch12_token_reward_decomp.png'), dpi=110, bbox_inches='tight')
plt.show()""")

# =============================================================================
# 12.7 与 InstructGPT 论文对比
# =============================================================================
md(r"""## 12.7 与 InstructGPT 论文对比

我们的简化实现和 [Ouyang et al. 2022, *Training language models to follow instructions
with human feedback*](https://arxiv.org/abs/2203.02155) 有哪些差异？

### 12.7.1 简化版 vs 论文版

| 维度 | 本章简化版 | InstructGPT 论文 |
|---|---|---|
| **base model** | TinyGPT (~30k 参数, char-level) | GPT-3 (175B 参数, BPE) |
| **SFT 数据** | 无 SFT（用未训练 TinyGPT 作 ref） | ~13k prompt-ideal_response（人工标注） |
| **偏好数据** | 300 条规则合成（rule-based winner/loser） | ~33k 偏好对（人工标注，每个 ~7s） |
| **RM 架构** | TinyGPT + scalar head (~30k 参数) | GPT-3 + scalar head（与 SFT 同尺寸） |
| **RM loss** | Bradley-Terry（同） | Bradley-Terry（同）+ 多 RM ensemble |
| **RLHF algorithm** | PPO + KL penalty | PPO + KL penalty（同） |
| **PPO horizon** | 10 tokens | 1024 tokens（甚至更长） |
| **PPO batch** | 12 responses × 10 tokens = 120 token-steps | 1024 prompts × 1024 tokens ≈ 1M token-steps |
| **KL penalty β** | 0.05（手动调） | 0.01-0.1（按 task 调） |
| **PPO epochs K** | 4 | 通常 1-2 |
| **freeze list** | reward + reference | reward + reference（同） |
| **额外 trick** | 无 | PTX (pretraining mix), KL early stop, value clipping |

### 12.7.2 我们保留的核心要素（与论文一致）

| 要素 | 本章实现 | 论文对应 |
|---|---|---|
| **4 模型协调** | actor + critic + reward + reference | ✓ 论文 Figure 2 |
| **KL penalty on every token** | $r_t = -\beta \log \frac{\pi(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$ | ✓ 论文 Eq. (2) |
| **RM reward 加在最后 token** | $r_{T-1} += r(x, y)$ | ✓ 论文（隐含） |
| **PPO-Clip + KL early stop** | `clip_eps=0.2, target_kl=0.05` | ✓ 论文（同参数） |
| **advantage normalization** | `normalize_advantage=True` | ✓ 论文 |
| **RM 冻结** | `requires_grad_(False)` | ✓ 论文 |

### 12.7.3 我们省略的（教学简化）

1. **PTX (Pretraining Mix)**：论文在 RL loss 里混入预训练梯度（保留 LM 能力）。
   我们没做，因为 char-level TinyGPT 已经很小，不需要。
2. **多 RM ensemble**：论文用多个 RM 集成减少 RM noise；我们用单个 RM。
3. **大规模 batch size**：论文用 1024×1024；我们用 12×10（CPU 限制）。
4. **EOS 处理**：论文让模型可以生成 EOS 终止；我们用固定长度（简化）。
5. **Long horizon credit assignment**：论文 response 可以 1024+ tokens；
   我们 10 tokens（信号传播容易，省时间）。

### 12.7.4 关键洞察：为什么简化后还能工作？

RLHF 的核心思想——**用 KL penalty 平衡 reward 和 reference**——与模型大小无关。
PPO 的数学（GAE + Clip + KL early stop）也与模型无关。

所以即使 30k 参数的 TinyGPT 也能演示**完整 pipeline**：
- RM 训到 80%+ accuracy（Ch11）
- PPO 让 actor 在 RM 眼里 reward 上升（§12.5）
- KL penalty 控制不爆炸（§12.6）

这是教学 vs 工程的差异——**理解原理用小模型就够**，**生产部署才需要大模型**。""")

# =============================================================================
# 12.8 小结 + GRPO 预告
# =============================================================================
md(r"""## 12.8 小结 + GRPO 预告

### 12.8.1 本章核心收获

| 概念 | 一句话总结 | 出处 |
|---|---|---|
| **RLHF 三阶段** | SFT → RM → PPO；用 RL 对齐 SFT 模型 | §12.1 |
| **4 模型架构** | actor / critic / reward / reference 同时协调 | §12.2 |
| **KL penalty 推导** | 从约束 $\text{KL} \le \epsilon$ → Lagrangian → reward shaping | §12.3 |
| **token-level MDP** | state = prefix, action = next token, reward per-token | §12.4 |
| **PPO on tokens** | 复用 Ch09 的 clip + GAE，只是数据 shape 变了 | §12.5 |
| **reward hacking** | Goodhart's Law；KL penalty 是主要防御 | §12.6 |
| **InstructGPT 配方** | 简化版与论文版数学完全一致，只是规模差异 | §12.7 |

### 12.8.2 关键公式速查

| 公式 | 含义 | 出处 |
|---|---|---|
| $\max_\pi \mathbb{E}[r] \text{ s.t. } \mathbb{E}[\text{KL}] \le \epsilon$ | RLHF 约束目标 | §12.3.1 |
| $\pi^*(y\|x) \propto \pi_{ref}(y\|x) \exp(r(x,y)/\beta)$ | KL-约束下的最优策略 | §12.3.3 |
| $r_{total}(x, y) = r(x, y) - \beta \log \frac{\pi}{\pi_{ref}}$ | 调整后 reward（reward shaping） | §12.3.4 |
| $r_t = -\beta \log \frac{\pi(a_t\|s_t)}{\pi_{ref}(a_t\|s_t)}$ + last: $+ r(x,y)$ | per-token reward | §12.3.5 |
| $\hat{A}_t^{\text{GAE}} = \sum_l (\gamma\lambda)^l \delta_{t+l}$ | GAE（token 序列上不变） | §12.4.4 |
| $L^{CLIP} = \mathbb{E}[\min(r_t \hat A_t, \text{clip}(r_t) \hat A_t)]$ | PPO-Clip（同 Ch09） | §12.5 |

### 12.8.3 RLHF-PPO 的"代价"

本章展示了完整的 InstructGPT 配方，但有一个**重大工程问题**被我们略过了：

> **critic $V_\phi$ 在 LLM 上太贵了。**

具体来说：
1. **参数量翻倍**：actor + critic 都要训一个 transformer（critic 不能省，PPO 需要 baseline）
2. **训练不稳定**：critic 要在 token 序列上学准 $V_\phi(s_t)$，而 $s_t$ 是高维 prefix，value 学习本身就难
3. **推理开销**：每次 PPO update 都要跑一遍 critic forward（额外 ~50% 计算量）

对 Ch12 的 30k 参数 TinyGPT，这不是问题。但对 70B 参数的 LLaMA/Qwen，**critic 是 PPO 的最大瓶颈**。

### 12.8.4 Ch13 预告：GRPO（Group Relative Policy Optimization）

DeepSeek 2024 年提出的 [GRPO](https://arxiv.org/abs/2402.03300) 解决了这个问题：

> **关键洞察**：用 **group sampling**（同一 prompt 采 G 个 response）的**组内相对 reward**
> 当 advantage，**完全不需要 critic**。

| 维度 | PPO（本章） | GRPO（Ch13） |
|---|---|---|
| 模型数 | **4**（actor + critic + reward + reference） | **3**（actor + reward + reference） |
| Advantage 来源 | $V_\phi$ baseline + GAE | $\frac{r_i - \text{mean}(r)}{\text{std}(r)}$（group 内标准化） |
| 参数开销 | actor + critic 都训 | 只训 actor |
| 训练稳定性 | critic 学不准 → 不稳 | 无 critic，更稳 |
| 计算 / token | 高（要 forward critic） | 低（只 forward actor） |

**Ch13 的核心问题**：
> **去掉 critic 后，PPO 还能 work 吗？GRPO 的数学保证是什么？**

答案是：**group baseline 是 REINFORCE 的一个特例**，理论上保证无偏，
而且对 LLM 这种"reward 稀疏、value 难学"的场景反而更合适。

> → **下一章：Ch13 GRPO —— 砍掉 critic，PPO 还能 work**""")

code(r"""# Ch12 完成总结
print("=" * 70)
print("Ch12 RLHF-PPO 完成 —— Phase 3 第三块（InstructGPT 配方）拼图就位")
print("=" * 70)
print("本章交付:")
print(f"  - utils/rlhf.py")
print(f"      ValueHead                  (TinyGPT backbone + per-token value head)")
print(f"      RLHFConfig                 (β={cfg.beta}, γ={cfg.gamma}, K={cfg.update_epochs}, ...)")
print(f"      RLHFTrainer                (4 模型协调器)")
print(f"        .rollout_responses       (采样 G 个 response)")
print(f"        .compute_token_rewards   (RM + KL penalty → per-token)")
print(f"        .rlhf_update             (PPO 多-epoch 更新)")
print(f"  - notebooks/ch12_rlhf_ppo.ipynb: 本章")
print()
print(f"4 模型参数量:")
def all_params(m):
    return sum(p.numel() for p in m.parameters())
print(f"  actor (π_θ):       {all_params(actor):,} params (trainable: {count_parameters(actor):,})")
print(f"  reference (π_ref): {all_params(reference):,} params (frozen)")
print(f"  critic (V_φ):      {all_params(critic):,} params (trainable: {count_parameters(critic):,})")
print(f"  reward (r):        {all_params(reward_model):,} params (frozen)")
print()
print(f"训练效果:")
final_r = history[-1]['mean_reward']
max_r = max(h['mean_reward'] for h in history)
print(f"  初始 RM reward:    {history[0]['mean_reward']:+.3f}")
print(f"  最终 RM reward:    {final_r:+.3f} (max = {max_r:+.3f})")
print(f"  最终 KL to ref:    {history[-1]['mean_kl_to_ref']:+.4f}")
print(f"  训练耗时:          {train_time:.1f}s ({train_time/len(history):.2f}s/iter)")
print()
print("Phase 3 路线图:")
print("  Ch10 TinyGPT           ✓  base + SFT 模型")
print("  Ch11 Reward Modeling   ✓  Bradley-Terry + 偏好数据 + 过优化")
print("  Ch12 RLHF-PPO          ✓  4 模型 + KL penalty + InstructGPT 配方（本章）")
print("  → Ch13 GRPO ★终极目标    group sampling + 无 critic")
print("  Ch14 DPO/KTO              避免 RL 的替代方案")
print("=" * 70)""")

# =============================================================================
# Build notebook
# =============================================================================
md(r"""## 12.9 📝 练习

### 练习 1（必做）：β=0 消融——亲手制造 reward hacking

**任务**：把 `RLHFConfig(beta=...)` 设为 0（完全关掉 KL penalty），其它不变，训练 30 个 iteration，对比 β=0.02 的正常运行：

1. 画 mean_reward 曲线（RM 打分）
2. 观察 mean_response_len 和 entropy 的变化
3. 采样几条 response 打印出来看看

**预期结果**：β=0 时 RM 分数涨得**更快**，但 response 逐渐退化——变长、复读、或钻 RM 的空子（keyword 堆砌）。这就是 Goodhart：优化代理奖励 ≠ 优化真奖励。KL penalty 正是把 policy 拴在 reference 附近的缰绳。

### 练习 2（选做）：response_max_len 的影响

response_max_len 从 16 改成 8 / 24，观察最优 β 是否移动（提示：更长的 response 有更多"作弊空间"，可能需要更大的 β）。

*（开放练习，无参考答案——把观察写进笔记，Ch14 DPO 会再次用到这组直觉。）
> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch12 的自测题再进入下一章。""")

if __name__ == "__main__":
    # 直接运行时写出 notebook（被 build_notebooks.py import 时不写盘）
    _nb.write("ch12_rlhf_ppo.ipynb")
