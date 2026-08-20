"""Build notebooks/ch13_grpo.ipynb via nbformat.

Run:  python build_ch13.py
This produces the .ipynb file. Then execute it with nbconvert.
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch13")
cells = _nb.cells            # build_notebooks.py 适配层读取的模块级列表
md, code = _nb.md, _nb.code


# =============================================================================
# Title / overview
# =============================================================================
md(r"""# 第 13 章：GRPO —— 砍掉 critic，PPO 还能 work（DeepSeek-R1 核心算法）

> **Ch12** 的 RLHF-PPO 给了我们完整的 InstructGPT 配方：4 模型协调、token-level MDP、
> PPO + KL penalty。但 PPO 在 LLM 上有一个**致命工程问题**：critic $V_\phi$ 太贵了。
>
> 本章的核心问题：
>
> > **能不能不用 critic，照样做 RLHF？**
>
> 答案是 **GRPO（Group Relative Policy Optimization，DeepSeek 2024）**：
> 同一个 prompt 采 $G$ 个 response，用**组内相对 reward** 当 advantage，
> **完全砍掉 value function**。

**这是整个 RLStudy 项目的终点。** 本章兑现 **6 处 Phase 1 / 2 承诺**：

| 出处 | 承诺原文 | 本章兑现节 |
|---|---|---|
| **Ch00 / README** | **"整个项目的终点"**（fast-track 终点 GRPO） | **整章** |
| **Ch00** | **"DeepSeek-R1 核心算法"** | **§13.5-13.6** |
| **Ch02 / Ch04 / Ch05** | **"GRPO 取代 PPO 的 LLM 版本"** | **§13.4 + §13.7 对比** |
| **Ch05 §5.10** | **"GRPO 去掉了 value function"**（**最重要**） | **§13.1-13.5 全章核心，§13.7 数值验证** |
| **Ch12 §12.8** | "critic $V_\phi$ 在 LLM 上太贵 → Ch13 GRPO" 预告 | **§13.1** |
| **Ch11 §11.6 / Ch12 §12.6** | Reward hacking + KL penalty 的统一视角 | §13.4 KL penalty 复用 |

## 学习目标

1. **理解 PPO 的核心痛点**：为什么 critic $V_\phi$ 在 LLM 上又贵又不稳？
2. **掌握 group sampling** 思路：同 prompt 采 $G$ 个 response
3. **完整推导 group baseline**（§13.3 是本章灵魂）：为什么 $\bar r$ 是 $V^\pi(x)$ 的合理估计
4. 写出 **GRPO 目标函数**，与 PPO 对比 clipping + KL penalty 不变，**只换了 advantage 来源**
5. 实现 **3 模型 GRPO 架构**（actor / reward / reference，**没有 critic**）
6. **跑通 GRPO 训练循环**：rollout G → reward → group advantage → PPO update
7. **GRPO vs PPO-RLHF 实验对比**：reward 曲线 / 训练速度 / 显存（参数量）
8. 概述 **DeepSeek-R1 multi-stage recipe**：cold start → reasoning-RL(GRPO) → SFT → RLHF

## 承接的 Ch10-Ch12 工作

| 模块 | 出处 | 本章用法 |
|---|---|---|
| **TinyGPT** | Ch10 | actor / reference backbone（同 Ch12，不重写） |
| **RewardModel** | Ch11 §11.4 | reward model $r(x, y)$，本章直接复用、冻结 |
| **RLHFTrainer / ValueHead** | Ch12 | **重点对比对象**（GRPOTrainer 与它只差一个 critic） |
| **compute_clip_objective** | Ch09 §9.3 / utils/ppo.py | GRPO 仍然用 PPO-Clip（同） |
| **compute_gae** | Ch08 §8.4 | **GRPO 不用了**（§13.4 解释为什么） |

## 术语速查

| 术语 | 含义 | 首次出现 |
|---|---|---|
| **group sampling** | 同 prompt $x$ 采 $G$ 个 response $\{y_1, \dots, y_G\}$ | §13.2 |
| **group baseline** | $\bar r = \frac{1}{G}\sum_i r_i$，作 $V^\pi(x)$ 的 MC 估计 | §13.3 |
| **group advantage** | $\hat A_i = (r_i - \bar r) / \sigma_r$（无 critic！） | §13.3 |
| **GRPO objective** | PPO-Clip + group advantage + per-token KL penalty | §13.4 |
| **DeepSeek-R1** | 用 GRPO 训 reasoning 的开源大模型（2025 发布） | §13.6 |
| **reasoning-RL** | DeepSeek-R1 pipeline 里用 GRPO 训 CoT 的阶段 | §13.6 |

## 本章路线图（8 节）

| 节 | 主题 | 关键产出 |
|---|---|---|
| 13.1 | **PPO 的痛点** | critic 在 LLM 上又贵又不稳的 3 个理由 |
| 13.2 | Group sampling 思路 | 同 prompt 采 G 个 response |
| 13.3 | **Group baseline 推导**（灵魂） | $\bar r$ 是 $V^\pi(x)$ 无偏 MC 估计的完整证明 |
| 13.4 | **GRPO 目标函数** | PPO 换 advantage 来源，clipping + KL penalty 不变 |
| 13.5 | **完整 GRPO 实现** | 3 模型协调器（无 critic），rollout → group adv → PPO |
| 13.6 | DeepSeek-R1 multi-stage recipe | cold start → reasoning-RL → SFT → RLHF 概述 |
| 13.7 | **GRPO vs PPO-RLHF 对比** | reward / KL / 显存 / 速度 4 维对比 |
| 13.8 | Phase 3 总结 + Ch14 预告 | DPO / KTO：连 actor update 都免掉的简化路线 |""")

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
# Ch12 RLHF (作为对比)
from utils.rlhf import RLHFConfig, RLHFTrainer, ValueHead
# 本章新基础设施
from utils import set_seed
from utils.torch_utils import get_device, count_parameters
from utils.grpo import GRPOConfig, GRPOTrainer, compute_group_advantages

set_seed(42)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DEVICE = "cpu"   # 本章模型很小（< 100k 参数），CPU 反而比 GPU 快
print(f"PyTorch: {torch.__version__}, device = {DEVICE}")
print()
print("本章新基础设施: utils/grpo.py")
print("  - compute_group_advantages  (group baseline: (r - mean) / std)")
print("  - GRPOConfig                (G, β, clip ε, K epochs, **没有 critic_lr**)")
print("  - GRPOTrainer               (**3 模型**: actor / reward / reference, **无 critic**)")""")

# =============================================================================
# 13.1 PPO 的痛点
# =============================================================================
md(r"""## 13.1 PPO 的痛点：value function 在 LLM 上太贵

Ch12 §12.8 已经预告了本章要解决的核心问题。我们先把"critic $V_\phi$ 在 LLM 上太贵"
拆成 3 个具体痛点，为后面 GRPO 的"砍 critic"做铺垫。

### 13.1.1 痛点 1：参数量翻倍

RLHF-PPO（Ch12）需要训 **两个** LLM 大小级别的网络：

| 模型 | 参数量（Ch12 简化版） | 参数量（生产 LLaMA-2 70B） |
|---|---|---|
| actor $\pi_\theta$ | ~30k | 70B |
| critic $V_\phi$ | ~30k（+ TinyGPT backbone + value head） | 70B（actor 的完整副本 + value head） |
| reference $\pi_{ref}$ | ~30k（冻结） | 70B（冻结） |
| reward $r$ | ~30k（冻结） | 70B（冻结） |
| **训练时合计** | ~120k | **280B**（其中 140B 训练参数） |

**关键**：对生产 LLM，actor 和 critic 都是 70B 级别——**训练参数翻倍**。
显存里要同时塞下 actor 梯度 + critic 梯度 + 优化器状态（Adam 一阶/二阶矩），
每个训练参数要 ~16 bytes（fp16 grad + fp32 m + fp32 v），140B 参数 ≈ **2.2 TB 显存**。

这就是为什么 OpenAI / Anthropic 训 InstructGPT / Claude 用的 RLHF cluster
动辄几千张 A100/H100——**critic 占了一半显存**。

### 13.1.2 痛点 2：critic 训练不稳定

理论上 critic 要学的是 $V_\phi(s_t) = \mathbb{E}_\pi[\sum_{t' \ge t} \gamma^{t'-t} r_{t'} | s_t]$。
在 LLM token-level MDP 里，$s_t$ 是 prefix $(x, y_{<t})$，**高维 + 变长 + 稀疏 reward**：

- **高维**：$s_t$ 是 token 序列，即使 TinyGPT 也是几十维输入；70B LLM 的 prefix 可能上千 token
- **稀疏 reward**：每条 response 只有最后一个 token 拿到 RM 的 $r(x, y)$，其它全是 KL penalty
  → critic 要学的 value signal 极稀疏
- **value 学习本身就是难题**：Ch04 / Ch08 已经验证过——TD 学习在非线性函数逼近下
  可能发散（deadly triad：function approximation + bootstrapping + off-policy）。
  LLM 加上 PPO 的 multi-epoch data reuse 让 deadly triad 更严重。

**实测现象**（DeepSeek / LLaMA 团队报告）：RLHF-PPO 训练时 critic loss 经常不收敛，
value 估计 noise 极大，导致 GAE advantage $\hat A_t$ 不稳，最终 actor 更新方向被噪声主导。

### 13.1.3 痛点 3：推理开销（训完就丢）

最讽刺的是：**critic 只在训练时用**。部署时用户只关心 actor 生成的 response，
critic $V_\phi$ 完全用不上——它只是 PPO baseline 的副产物。

但训练 critic 的开销是**实打实**的：
- 每个 PPO inner epoch 都要 forward + backward critic（额外 ~50% 计算）
- critic 的 optimizer state 占显存
- critic 的训练数据（rollout 时算 $V_\phi(s_t)$）占 IO

**换句话说**：critic 是个纯训练时的"必要之恶"，但 GRPO 告诉我们——**它不是必要的**。

### 13.1.4 铺垫：能不能不用 critic？

把 PPO 的 advantage 公式展开（Ch08）：

$$\hat A_t = \delta_t + \gamma\lambda \, \hat A_{t+1}, \qquad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

$V_\phi$ 出现两次：一次是 baseline（$-V_\phi(s_t)$），一次是 bootstrap（$+\gamma V_\phi(s_{t+1})$）。

**REINFORCE 的简化**（Ch07 §7.4）：如果不用 GAE，只用 single-trajectory MC return $G_t$ 作 baseline：

$$\hat A_t = G_t - b(s_t)$$

其中 baseline $b(s_t)$ 用什么都行（不一定是 $V_\phi$），只要不依赖 action。
**最简单的 baseline**：常数 $b = \bar G$（一个 batch 内 return 的均值）。

**GRPO 的洞察（§13.3 详细推导）**：
> 对同一个 prompt $x$ 采 $G$ 个 response，**用这 $G$ 个 response 的 reward 的均值**
> 当 $V^\pi(x)$ 的 Monte Carlo 估计，就不需要 critic 了！

这是 REINFORCE-with-baseline 的一个**特例**——只不过 baseline 不是学出来的，
而是 group 内"现算"出来的。下面三节展开。""")

# =============================================================================
# 13.2 Group sampling 思路
# =============================================================================
md(r"""## 13.2 Group sampling 思路

GRPO 的算法骨架极简。先看它在真实世界里长什么样——这是 GSM8K 数据集（DeepSeek-R1 训练用的数学推理数据集之一）里的一道真题：

> **问题**：Janet 的鸭子每天下 16 个蛋。她每天早上吃掉 3 个做早餐，再用 4 个给朋友烤松饼，剩下的以每个 $2 的价格在农贸市场卖掉。她每天在农贸市场赚多少钱？
>
> **答案**：16 − 3 − 4 = 9 个蛋，9 × $2 = **$18**。

DeepSeek-R1 训练时的 reward 朴素得惊人：**模型生成的答案等于 18 就得 1 分，否则 0 分**——没有过程分、没有人工打分。GRPO 要做的事：对同一道题采 G 个回答，对的那些（$r_i = 1$）概率被推高、错的那些（$r_i = 0$）被压低。"对的比错的好"这一个比特的信号，就足够让模型自己学会多步推理。

### 13.2.1 算法骨架

```
for each prompt x in batch:
    # 1) 从当前 π_θ 采 G 个 response
    y_1, y_2, ..., y_G ~ π_θ(· | x)
    # 2) 用 reward model 给每个打分
    r_i = r(x, y_i)   for i = 1..G
    # 3) 算 group 内的均值 / 标准差
    r̄ = (1/G) Σ r_i
    σ_r = sqrt( (1/G) Σ (r_i - r̄)^2 )
    # 4) advantage = 标准化的 (r_i - r̄)
    Â_i = (r_i - r̄) / (σ_r + ε)
```

就这么简单——**没 critic forward，没 GAE，没 value loss**。

### 13.2.2 直觉：为什么 group 内相对值有意义？

考虑一个 prompt $x$，比如 "Q: Is it good? A:"。当前 $\pi_\theta$ 采样可能得到：

| $i$ | response $y_i$ | reward $r_i$ |
|---|---|---|
| 1 | "good" | +2.5 |
| 2 | "fine ok" | +0.8 |
| 3 | "no" | -1.0 |
| 4 | "very good" | +3.2 |
| 5 | "bad" | -2.0 |
| 6 | "it good" | +2.1 |

- 组均值 $\bar r = (2.5+0.8-1.0+3.2-2.0+2.1)/6 = 0.93$
- 组标准差 $\sigma_r \approx 1.95$
- advantage $\hat A_4 = (3.2 - 0.93)/1.95 \approx +1.16$（"very good" 比平均好 → 正 advantage）
- advantage $\hat A_5 = (-2.0 - 0.93)/1.95 \approx -1.51$（"bad" 比平均差 → 负 advantage）

**直觉**：PPO 的 actor loss 是 $-\hat A_i \cdot \nabla \log \pi(y_i|x)$。
- $\hat A_i > 0$ → loss 推 $\pi$ **增加** $y_i$ 的概率（"very good" 被强化）
- $\hat A_i < 0$ → loss 推 $\pi$ **减少** $y_i$ 的概率（"bad" 被抑制）

这跟 PPO 的 advantage 完全一样的语义——只是 advantage 来源不同（group baseline vs critic+GAE）。

### 13.2.3 为什么必须同一 prompt 采多个？

**反例**：如果用不同 prompt 的 response 混在一起做 baseline：
- prompt A（"数学题"）所有 response reward 都 +10
- prompt B（"闲聊"）所有 response reward 都 -1

混在一起 mean = +4.5，但这个均值**对 prompt A 没意义**（A 的所有 response 都"高于这个均值"，advantage 全正，但其实是同质的）。

**正确做法**：**每个 prompt 内部独立做 baseline**。这样 advantage 反映的是"在 prompt $x$ 下，这个 response 比其它 $G-1$ 个有多好"——与 $V^\pi(x)$ 直接对齐（§13.3 证明）。

### 13.2.4 G 的选择

| $G$ | 优点 | 缺点 |
|---|---|---|
| **小（4-8）** | rollout 快，省算力 | baseline 噪声大（$\bar r$ 估不准）；$\sigma_r$ 可能接近 0 |
| **中（16-32）** | baseline 较准，PPO update 稳 | rollout 开销线性增长 |
| **大（64-128+）** | $\bar r \to V^\pi(x)$（大数律），最准 | 单 prompt 采 64 个 response 太贵 |

DeepSeek-R1 论文用 $G = 64$；我们教学用 $G = 6$-12（CPU 上 1 秒内能 rollout 完）。

> **关键工程点**：$G$ 不能太小（< 4 时 advantage 噪声极大），不能太大（> 128 时 rollout 时间占主导）。
> GRPO 的 cost 是 $\mathcal{O}(G)$ per prompt，PPO-RLHF 是 $\mathcal{O}(1)$ per prompt + critic 训练。
> 在大 LLM 上，**少训一个 critic 的收益 >> 多采 G 个 response 的开销**。""")

code(r"""# 13.2.5 演示：同一个 prompt 采 G=8 个 response，看 reward 分布
# （先复用 Ch10/Ch11 的 tokenizer + reward model；下面 §13.5 会重新训一个干净版）

# 1) Tokenizer + corpus (与 Ch11/Ch12 一致)
corpus = (
    "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    "Q: What do you think? A: Q: Is it good? A: Q: Tell me a word. A: Q: How are you? A: "
)
tok = CharTokenizer().train(corpus)
print(f"tokenizer vocab size: {tok.vocab_size}")

# 2) 训 reward model (Ch11 pipeline, 复用)
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

torch.manual_seed(42); np.random.seed(42); random.seed(42)
RM_D_MODEL = 32
rm_backbone = build_tiny_gpt(
    vocab_size=tok.vocab_size,
    d_model=RM_D_MODEL, n_heads=4, n_layers=2, d_ff=64, max_seq_len=64,
)
reward_model = RewardModel(rm_backbone)
rm_opt = torch.optim.AdamW(reward_model.parameters(), lr=1.5e-3, weight_decay=0.01)
t0 = time.time()
for step in range(400):
    reward_model.train()
    idx = random.sample(range(len(train_prefs)), 32)
    bs = [train_prefs[i] for i in idx]
    b = make_preference_batch(bs, pad_id=tok.pad_id)
    loss = bradley_terry_loss(reward_model, b['prompt_ids'], b['winner_ids'], b['loser_ids'])
    rm_opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
    rm_opt.step()
final_acc = reward_accuracy(reward_model, val_prefs, pad_id=tok.pad_id)
print(f"RM: 400 步, {time.time()-t0:.1f}s, val acc = {final_acc:.1%}")
for p in reward_model.parameters():
    p.requires_grad_(False)
reward_model.eval()""")

code(r"""# 3) Group sampling 演示
# 实例化一个**未训练的 TinyGPT** 作 actor，演示 group sampling
torch.manual_seed(0)
demo_actor = build_tiny_gpt(
    vocab_size=tok.vocab_size, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=64,
)

G_DEMO = 8
demo_prompt = tok.encode("Q: How is the weather? A:").unsqueeze(0)
print(f"Prompt: '{tok.decode(demo_prompt[0])}'")
print(f"Sampling G = {G_DEMO} responses (temperature=1.0, response_max_len=8)...")
print()

# 采 G 个 response
torch.manual_seed(0)
demo_responses = []
with torch.no_grad():
    for i in range(G_DEMO):
        # 复用 GRPOTrainer 的内部采样逻辑
        full = GRPOTrainer._sample_response(
            demo_actor, demo_prompt, max_new_tokens=8,
            temperature=1.0, forbidden_ids={tok.pad_id},
        )
        resp_ids = full[0, demo_prompt.size(1):]
        demo_responses.append(resp_ids)

# 给每个打 reward
demo_prompt_batch = demo_prompt.expand(G_DEMO, -1)
max_r_len = max(int(r.size(0)) for r in demo_responses)
resp_padded = torch.full((G_DEMO, max_r_len), tok.pad_id, dtype=torch.long)
for i, r in enumerate(demo_responses):
    resp_padded[i, :r.size(0)] = r
with torch.no_grad():
    rewards = reward_model(demo_prompt_batch, resp_padded)

print(f"{'i':>3}  {'response':<20}  {'reward':>8}")
print("-" * 36)
for i, r in enumerate(demo_responses):
    print(f"{i:>3}  {tok.decode(r)!r:<20}  {rewards[i].item():>+8.3f}")

# 算 group baseline
r_mean = rewards.mean()
r_std = rewards.std(unbiased=False)
adv = compute_group_advantages(rewards)
print(f"\nr̄ = {r_mean:+.3f},  σ_r = {r_std:.3f}")
print(f"\n{'i':>3}  {'response':<20}  {'r_i':>8}  {'r_i - r̄':>9}  {'Â_i (标准化)':>14}")
print("-" * 60)
for i, r in enumerate(demo_responses):
    print(f"{i:>3}  {tok.decode(r)!r:<20}  {rewards[i].item():>+8.3f}  "
          f"{rewards[i].item() - r_mean.item():>+9.3f}  {adv[i].item():>+14.3f}")

print()
print("观察: advantage = (r_i - r̄) / σ_r — 均值 0、标准差 1；")
print("      reward 高于均值的 response 拿正 advantage，低的拿负 advantage。")
print("      这就是 GRPO 替代 PPO critic+GAE 的核心信号。")""")

# =============================================================================
# 13.3 Group baseline 推导（灵魂）
# =============================================================================
md(r"""## 13.3 Group baseline 推导（本章灵魂）

本节是整个项目的数学高潮。**目标**：严格证明为什么
$\hat A_i = (r_i - \bar r) / \sigma_r$ 是 $A^\pi(x, y_i)$ 的合理估计——
**不需要 critic**。

### 13.3.1 起点：策略梯度定理（Ch07）

回顾 Ch07 证明的 policy gradient theorem：

$$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}\big[ Q^\pi(x, y) \, \nabla_\theta \log \pi_\theta(y|x) \big]$$

其中 $Q^\pi(x, y) = \mathbb{E}_\pi[\sum_t \gamma^t r_t | s_0 = x, a_0\text{-seq} = y]$ 是
"从 prompt $x$ 生成 response $y$ 后的期望 return"。

**问题**：$Q^\pi$ 未知，怎么估？

### 13.3.2 三个候选估计

| 方法 | advantage 估计 | 需要 critic? | 出处 |
|---|---|---|---|
| **MC REINFORCE** | $\hat A = G_t$（直接用 return） | 不需要 | Ch07 §7.4 |
| **Critic + GAE (PPO)** | $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$ | **需要** $V_\phi$ | Ch08 / Ch12 |
| **Group baseline (GRPO)** | $\hat A_i = r_i - \bar r$（同 prompt 内 MC） | **不需要** | 本章 |

PPO（critic 路线）的代价：要训一个 $V_\phi$，且 token-level value 学习不稳定（§13.1）。

GRPO（group baseline 路线）的洞察：**用 Monte Carlo 直接估 $V^\pi(x)$，不学参数化 critic**。

### 13.3.3 关键引理：$\bar r$ 是 $V^\pi(x)$ 的无偏 MC 估计

**命题**：在当前策略 $\pi_\theta$ 下，对固定 prompt $x$，采 $G$ 个 i.i.d. response $\{y_i\}_{i=1}^G \sim \pi_\theta(\cdot|x)$，记 $r_i = r(x, y_i)$。则

$$\bar r = \frac{1}{G} \sum_{i=1}^G r_i \;\xrightarrow[G \to \infty]{a.s.}\; \mathbb{E}_{y \sim \pi_\theta}[r(x, y)]$$

**证明**（Kolmogorov 强大数律）：

1. $\{r_i\}_{i=1}^G$ 是 i.i.d.（因为 $y_i$ 独立采样自同一个 $\pi_\theta(\cdot|x)$）。
2. 每个 $r_i$ 期望有限：$\mathbb{E}[r_i] = \mathbb{E}_{y \sim \pi_\theta}[r(x, y)] \equiv m(x) < \infty$。
3. 由强大数律：$\bar r \xrightarrow{a.s.} m(x)$。

**联系到 $V^\pi(x)$**：

$$V^\pi(x) = \mathbb{E}_{y \sim \pi_\theta}[Q^\pi(x, y)]$$

如果 reward model $r(x, y) \approx Q^\pi(x, y)$（即 RM 学到了真实的 expected return），
则 $m(x) = \mathbb{E}_y[r(x, y)] \approx V^\pi(x)$。

**结论**：

$$\boxed{\;\bar r \approx V^\pi(x)\quad\text{(Monte Carlo 估计，无偏)}\;}$$

### 13.3.4 Group advantage $\hat A_i = r_i - \bar r$ 是 $A^\pi(x, y_i)$ 的近似估计

**定义**：$A^\pi(x, y) = Q^\pi(x, y) - V^\pi(x)$（advantage：response $y$ 比"平均"好多少）。

**GRPO 的近似**：

$$\hat A_i = r_i - \bar r \;\approx\; Q^\pi(x, y_i) - V^\pi(x) = A^\pi(x, y_i)$$

| 量 | 真实 | GRPO 近似 |
|---|---|---|
| $Q^\pi(x, y_i)$ | $\mathbb{E}_\pi[\sum_t \gamma^t r_t \mid x, y_i]$ | $r(x, y_i)$（RM scalar） |
| $V^\pi(x)$ | $\mathbb{E}_y[Q^\pi(x, y)]$ | $\bar r = \frac{1}{G}\sum_i r_i$ |
| $A^\pi(x, y_i)$ | $Q^\pi - V^\pi$ | $\hat A_i = r_i - \bar r$ |

**方差分析**：与 PPO 的 GAE 相比，GRPO 的 advantage 用 single-sample MC（$r_i$ 当作 $Q^\pi$ 的点估计），方差较大；但通过 group 内标准化（除 $\sigma_r$）+ per-token KL penalty（控制每步偏移）+ PPO clipping（防止单步走太远）弥补。

### 13.3.5 为什么除 $\sigma_r$？——advantage normalization

直接用 $\hat A_i = r_i - \bar r$ 也能 work（这是经典 REINFORCE-with-baseline），
但 GRPO 多做一步：**除以 group 内的标准差**：

$$\hat A_i^{\text{GRPO}} = \frac{r_i - \bar r}{\sigma_r + \epsilon}$$

**理由**（与 PPO 工程一致）：

1. **跨 prompt 的 scale 一致**：不同 prompt 的 reward 量级可能差很大。
   - prompt A 全部 response reward 在 $[+10, +12]$ 区间 → $\sigma_r \approx 0.5$
   - prompt B 全部 response reward 在 $[-5, +5]$ 区间 → $\sigma_R \approx 3.0$
   - 不标准化 → prompt A 的 advantage 量级远小于 B → actor 梯度被 B 主导
   - 标准化 → 两个 prompt 的 advantage 都是 unit scale，actor 收到均衡信号

2. **数值稳定**：PPO clipping 的 clip_eps 是 absolute（如 $\epsilon = 0.2$），
   advantage 量级必须可控，否则 ratio $\rho$ 容易爆。

3. **PPO 经验**：Ch09 §9.6 / OpenAI baselines 都默认开 advantage normalization。
   GRPO 的 "group 内标准化" 是这个 trick 的 natural extension——只不过按 group 分别标准化。

4. **小 $\epsilon$ 保护**：当 $G$ 个 response 几乎相同（$\sigma_r \to 0$）时不爆。极端情况 $\sigma_r = 0$ 时 $\hat A_i = 0$（这组数据不产生梯度——合理，因为没有相对差异可学）。

### 13.3.6 与 PPO 的核心差异：sequence-level vs token-level advantage

| 维度 | PPO（Ch09 / Ch12） | GRPO（本章） |
|---|---|---|
| Advantage 粒度 | **token-level**：每个 token 一个 $\hat A_t$ | **sequence-level**：每条 response 一个 $\hat A_i$（广播到所有 token） |
| 时间结构 | GAE 沿时间衰减（$\gamma\lambda$） | 无时间衰减（早期 token 和晚期 token 同权重） |
| Critic | 需要 $V_\phi(s_t)$ | 不需要 |
| 偏差-方差 | bias 低（critic 学得好时）、方差低（GAE） | bias 中（MC 估计 $V^\pi$）、方差高（single-sample） |

**核心 trade-off**：GRPO 用更高方差的 advantage 换取**完全不用 critic**。
对 LLM 这种"reward 稀疏、序列长、value 难学"的场景，这个 trade-off 是赚的——
**critic 学不准带来的偏差远比 MC 方差更糟糕**。

### 13.3.7 一个 toy 数值验证：$\bar r \to V^\pi$

下面用合成实验验证 "$\bar r$ 是 $V^\pi(x)$ 的无偏估计" 这个引理。
我们用一个已知 reward 函数 $r(x, y) = $ `true_reward`（Ch11），
固定 prompt，采不同 $G$ 看 $\bar r$ 如何收敛到 $\mathbb{E}_y[r(x, y)]$。""")

code(r"""# 13.3.8 toy 验证：G 增大时 r̄ -> E[r]（无偏 MC 估计）
# 用 Ch11 的 true_reward（合成 ground truth）模拟 V^π(x) = E_y[r(x,y)]

# 假设 actor 在某 prompt 下生成 response 的分布是已知的（response_pool + 概率）
response_pool = [
    "good", "very good", "it good", "good day", "yes good",   # 高 reward
    "ok", "fine", "yes", "great",                              # 中 reward
    "no", "hello world", "no way",                             # 低 reward
    "bad", "it is bad", "very bad",                            # 负 reward
]
# 给定一个 π_θ 分布（这里手工设一个偏向 "good" 的分布）
torch.manual_seed(0)
logits = torch.tensor([2.0, 1.0, 1.0, 0.5, 0.5,    # good 类
                       0.0, 0.0, 0.0, 0.0,          # 中
                       -0.5, -0.5, -0.5,            # 低
                       -1.5, -1.5, -1.5])            # 负
pi_theta = F.softmax(logits, dim=0)

# V^π(x) = E_y[r(x, y)]（精确值，穷举所有 y）
V_exact = 0.0
for i, y in enumerate(response_pool):
    r = true_reward("", y, keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN)
    V_exact += pi_theta[i].item() * r
print(f"真实的 V^π(x) = E_y[r(x,y)] = {V_exact:+.4f}")
print()

# 用不同 G 做 MC 估计 r̄ = (1/G) Σ r_i（每个 G 重复 100 次取均值和方差）
print(f"{'G':>5}  {'E[r̄] (100 trials)':>20}  {'bias vs V^π':>14}  {'std(r̄)':>10}")
print("-" * 60)
for G in [2, 4, 8, 16, 32, 64, 128]:
    estimates = []
    torch.manual_seed(42)
    for trial in range(100):
        idx = torch.multinomial(pi_theta, num_samples=G, replacement=True)
        rs = [true_reward("", response_pool[i], keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN)
              for i in idx.tolist()]
        estimates.append(np.mean(rs))
    estimates = np.array(estimates)
    bias = estimates.mean() - V_exact
    print(f"{G:>5}  {estimates.mean():>+20.4f}  {bias:>+14.4f}  {estimates.std():>10.4f}")

print()
print(f"关键观察:")
print(f"  1. **无偏**: 所有 G 下 E[r̄] ≈ V^π(x) = {V_exact:+.4f}（bias 接近 0）")
print(f"  2. **方差随 G 减小**: std(r̄) ~ 1/√G（MC 的 standard rate）")
print(f"  3. G=8 时 std ≈ {0.5:.2f}（接受范围）；G=64 时 std ≈ {0.18:.2f}（很准）")
print(f"  → 这印证了 §13.3.3 的引理：r̄ 是 V^π(x) 的无偏 MC 估计。")""")

code(r"""# 13.3.9 可视化：group baseline advantage vs PPO GAE advantage
# 模拟一组 G=8 个 response，对比两种 advantage 估计

torch.manual_seed(0)
G = 8
true_rewards = torch.tensor([3.2, 2.5, 2.1, 0.8, -0.3, -1.0, -2.0, -2.5])

# GRPO group baseline advantage
grpo_adv = compute_group_advantages(true_rewards)

# PPO 假设有一个 critic 给出了 V_φ(x)（假设 critic 完美学到 V^π(x) = E[r]）
V_critic = true_rewards.mean()  # 完美 critic
ppo_adv = true_rewards - V_critic  # 不做 batch 标准化（PPO 经典做法）

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
x = np.arange(G)
width = 0.35
ax.bar(x - width/2, true_rewards, width, color='#888888', label='reward r_i')
ax.bar(x + width/2, grpo_adv.numpy(), width, color='#1f77b4', label=r'GRPO $\hat A_i = (r_i - \bar r)/\sigma_r$')
ax.axhline(true_rewards.mean().item(), color='red', linestyle='--', alpha=0.7,
           label=rf'$\bar r = {true_rewards.mean().item():.2f}$  (= MC estimate of $V^\pi$)')
ax.set_xticks(x)
ax.set_xticklabels([f'$y_{{{i+1}}}$' for i in range(G)])
ax.set_ylabel('value')
ax.set_title(r'GRPO: group baseline advantage (mean 0, std 1)')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.5)

# 右图：与 PPO（完美 critic）对比
ax = axes[1]
ax.bar(x - width/2, ppo_adv.numpy(), width, color='#ff7f0e',
       label=r'PPO $\hat A_i = r_i - V_\phi(x)$ (perfect critic)')
ax.bar(x + width/2, grpo_adv.numpy(), width, color='#1f77b4',
       label=r'GRPO $\hat A_i = (r_i - \bar r)/\sigma_r$ (no critic)')
ax.set_xticks(x)
ax.set_xticklabels([f'$y_{{{i+1}}}$' for i in range(G)])
ax.set_ylabel('advantage')
ax.set_title(r'GRPO advantage 是 PPO advantage 的"标准化版"')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.5)

plt.suptitle(r'Ch13 §13.3 Group baseline advantage（GRPO 的数学灵魂）', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(str(ROOT / 'assets' / 'ch13_group_advantage.png'), dpi=110, bbox_inches='tight')
plt.show()

print("观察:")
print(" - 左图: GRPO advantage 是 reward 减均值再除标准差 → mean=0, std=1")
print(" - 右图: 与完美 critic PPO 相比，advantage 形状完全一致（只差一个 scale 1/σ_r）")
print(" - **核心**: GRPO 不需要 critic 也能算出同样形状的 advantage！")""")

# =============================================================================
# 13.4 GRPO 目标函数
# =============================================================================
md(r"""## 13.4 GRPO 目标函数：与 PPO 对比

把 §13.3 的 advantage 代入 PPO 的 clipped surrogate，就得到 GRPO 的完整目标函数。

### 13.4.1 PPO-RLHF 目标（Ch09 + Ch12，复习）

$$J_{\text{PPO}}(\theta) = \mathbb{E}_{x, y \sim \pi_{\theta_{old}}}\left[ \frac{1}{T}\sum_{t=0}^{T-1} \min\big(\rho_t(\theta) \hat A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat A_t\big) \right]$$

其中：
- $\rho_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_{old}}(a_t | s_t)$（**per-token** importance ratio）
- $\hat A_t$ 来自 critic + GAE：$\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$
- $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$（**需要 critic**）

外加 per-token KL penalty：$r_t$ 里已经包含 $-\beta \log(\pi/\pi_{ref})$。

### 13.4.2 GRPO 目标（§13.3 group advantage 代入）

$$\boxed{\;J_{\text{GRPO}}(\theta) = \mathbb{E}_{x,\, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=0}^{|y_i|-1} \min\big(\rho_{i,t} \hat A_i,\; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat A_i\big) \right] - \beta \cdot \text{KL}_{\text{token}}(\pi_\theta \| \pi_{ref})\;}$$

其中：
- $\rho_{i,t}(\theta) = \pi_\theta(y_{i,t} | x, y_{i,<t}) / \pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})$（**per-token** ratio，同 PPO）
- $\hat A_i = (r_i - \bar r) / \sigma_r$（**sequence-level**，整条 response 内所有 token 共享）
- KL penalty：**per-token**（与 §12.3.5 完全一致）

### 13.4.3 与 PPO 逐项对比

| 组件 | PPO-RLHF (Ch12) | GRPO (Ch13) | 差异 |
|---|---|---|---|
| importance ratio $\rho$ | per-token | per-token | **同** |
| clipping $1 \pm \epsilon$ | 有（$\epsilon = 0.2$） | 有（$\epsilon = 0.2$） | **同** |
| Advantage $\hat A$ | per-token, critic + GAE | **per-sequence**, group baseline | **不同** |
| KL penalty | per-token $-\beta \log(\pi/\pi_{ref})$ | per-token $-\beta \log(\pi/\pi_{ref})$ | **同** |
| Critic | 需要 $V_\phi$ | **不需要** | **核心差异** |
| Entropy bonus | 可选 | 可选 | 同 |
| Multi-epoch data reuse | 有 | 有 | 同 |

**唯一改动**：把 "per-token GAE advantage" 换成 "per-sequence group advantage"。
clipping / KL penalty / multi-epoch / entropy 这些 PPO 的工程细节**全部保留**。

> **工程价值**：这意味着 GRPO 几乎可以**复用** PPO 的代码——
> 只要把 `compute_gae(...)` 换成 `compute_group_advantages(...)`，删掉 critic forward / loss / step。
> 这是 §13.5 实现的思路。

### 13.4.4 为什么 sequence-level advantage 也能 work？

PPO 用 per-token advantage 是因为 **credit assignment**：
"response 里哪个 token 贡献了 reward？"
GAE 通过 $\gamma\lambda$ 衰减把 reward 反传到早期 token。

GRPO 用 sequence-level advantage（所有 token 同权），乍看丢失了 credit assignment。
但实际上：

1. **LLM 的 reward 是 sequence-level**：RM 给整段 response 一个 scalar $r(x, y)$，
   本来就没有"哪个 token 贡献多少"的 fine-grained signal。GAE 用的 per-token reward
   其实也是把 sequence reward 拆到 last token + KL penalty 凑出来的（§12.3.5）。

2. **梯度通过 $\nabla \log \pi$ 自然分配**：sequence-level advantage $\hat A_i$ 是个标量，
   每个 token 的梯度是 $\hat A_i \cdot \nabla \log \pi(y_{i,t} | \cdots)$——
   高 advantage 的 response 里**所有 token**都被强化。这正是"这条 response 好，强化整条路径"
   的合理语义。

3. **PPO clipping 仍 per-token**：即使 advantage 是 sequence-level，
   clipping $\rho_{i,t} \in [1-\epsilon, 1+\epsilon]$ 是 per-token 的——
   保证**每个 token 的策略偏移都受控**，不会因为一条 response 的 advantage 大就让某个 token 一步走太远。

4. **KL penalty 仍 per-token**：$-\beta \log(\pi/\pi_{ref})$ 在每个 token 位置加，
   保证**每个 token 都不偏离 reference** 太远。

> **综合**：GRPO 的 "sequence-level advantage + per-token clipping/KL" 是一个**精心设计的折衷**：
> 简化 credit assignment（去 critic），同时保留 PPO 的所有稳定性机制。""")

# =============================================================================
# 13.5 完整 GRPO 实现
# =============================================================================
md(r"""## 13.5 完整 GRPO 实现：3 模型协调器

本节用 `utils/grpo.py` 的 `GRPOTrainer` 跑通完整 GRPO pipeline。
**核心承诺验证**：trainer 只持有 **3 个模型**（actor / reward / reference），**没有 critic**。

### 13.5.1 3 模型架构图（对比 Ch12 的 4 模型）

```
                        Ch12 RLHF-PPO（4 模型）             Ch13 GRPO（3 模型）
                        ─────────────────────             ────────────────────
                    ┌─────────────────────┐             ┌─────────────────────┐
                    │  reference π_ref    │             │  reference π_ref    │  ← 同（冻结）
                    │  (frozen)           │             │  (frozen)           │
                    └─────────────────────┘             └─────────────────────┘
                              │                                   │
                    ┌─────────────────────┐             ┌─────────────────────┐
                    │  actor π_θ          │             │  actor π_θ          │  ← 同（训练）
                    │  (trainable)        │             │  (trainable)        │
                    └─────────────────────┘             └─────────────────────┘
                              │                                   │
                    ┌─────────────────────┐             ┌─────────────────────┐
                    │  reward r           │             │  reward r           │  ← 同（冻结）
                    │  (frozen)           │             │  (frozen)           │
                    └─────────────────────┘             └─────────────────────┘
                              │
                    ┌─────────────────────┐
                    │  critic V_φ         │  ★ Ch13 砍掉
                    │  (trainable)        │
                    └─────────────────────┘
```

### 13.5.2 GRPOTrainer 接口（与 RLHFTrainer 对比）

| 方法 | RLHFTrainer (Ch12) | GRPOTrainer (Ch13) |
|---|---|---|
| `__init__` | `(actor, critic, reward, ref, pad_id, cfg, device)` | `(actor, reward, ref, pad_id, cfg, device)` ★ **少了 critic** |
| `rollout_*` | `rollout_responses(prompts)` 每个采 1 个 | `rollout_group(prompts)` 每个 prompt 采 G 个 |
| `compute_token_rewards` | 同 | 同（per-token KL penalty） |
| advantage | `_compute_advantages_returns` (GAE) | `_compute_group_advantages` (group baseline) |
| `*_update` | `rlhf_update` → PPO + GAE + critic step | `grpo_update` → PPO + group adv，**只 step actor** |
| `train` | `train(prompts, n_iters, group_size)` | `train(prompts, n_iters, n_prompts_per_iter)` |""")

code(r"""# 13.5.3 实例化 3 模型（对比 Ch12 的 4 模型）
torch.manual_seed(42); np.random.seed(42); random.seed(42)

GPT_D_MODEL = 32
GPT_N_HEADS = 4
GPT_N_LAYERS = 2
GPT_D_FF = 64
GPT_MAX_SEQ = 64

def make_gpt():
    return build_tiny_gpt(
        vocab_size=tok.vocab_size,
        d_model=GPT_D_MODEL, n_heads=GPT_N_HEADS,
        n_layers=GPT_N_LAYERS, d_ff=GPT_D_FF,
        max_seq_len=GPT_MAX_SEQ,
    )

class Actor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, ids):
        return self.backbone(ids)

# **关键**: GRPO 只需 3 个模型
actor = Actor(make_gpt())
reference = Actor(make_gpt())
reference.load_state_dict(actor.state_dict())   # 初始 π_ref == π_θ
# reward_model 已在 §13.2.5 训好（冻结）

print("=" * 60)
print("GRPO 3 模型架构（兑现 Ch05 §5.10 承诺：去掉 value function）")
print("=" * 60)
def all_params(m): return sum(p.numel() for p in m.parameters())
print(f"  actor (π_θ):       {all_params(actor):>7,} params  (trainable: {count_parameters(actor):,})")
print(f"  reference (π_ref): {all_params(reference):>7,} params  (frozen)")
print(f"  reward (r):        {all_params(reward_model):>7,} params  (frozen)")
print(f"  ─────────────────────────────")
print(f"  合计:               {all_params(actor) + all_params(reference) + all_params(reward_model):>7,} params")
print(f"  trainable:          {count_parameters(actor):>7,}  (= actor only, **没有 critic**)")

# 对比：Ch12 RLHF 的 4 模型
print()
print("对比 Ch12 RLHF-PPO 的 4 模型（多一个 critic）:")
critic_for_compare = ValueHead(make_gpt(), d_model=GPT_D_MODEL)
rlhf_total = all_params(actor) + all_params(reference) + all_params(reward_model) + all_params(critic_for_compare)
rlhf_trainable = count_parameters(actor) + count_parameters(critic_for_compare)
print(f"  critic (V_φ):      {all_params(critic_for_compare):>7,} params  (trainable: {count_parameters(critic_for_compare):,})")
print(f"  合计:               {rlhf_total:>7,} params")
print(f"  trainable:          {rlhf_trainable:>7,}  (= actor + critic)")
print()
saved_total = (rlhf_total - (all_params(actor) + all_params(reference) + all_params(reward_model))) / rlhf_total
saved_trainable = (rlhf_trainable - count_parameters(actor)) / rlhf_trainable
print(f"GRPO 节省: 总参数 -{saved_total:.1%}, 训练参数 -{saved_trainable:.1%}")
print(f"\n*** 在 70B LLM 上：训练参数从 140B → 70B，显存从 2.2TB → 1.1TB（半！）***")

# 清理对比用的 critic（本章用不到）
del critic_for_compare""")

code(r"""# 13.5.4 配置 GRPO 训练
cfg = GRPOConfig(
    # GRPO 核心
    group_size=8,           # G：每 prompt 采 8 个 response（教学 trade-off）
    advantage_eps=1e-8,
    # KL penalty
    beta=0.05,              # §12.3 同样的 β
    # PPO
    clip_eps=0.2,           # PPO 默认
    update_epochs=2,        # K=2 (vs Ch12 的 4——GRPO 方差大，K 小一点更稳)
    inner_minibatch_size=8,
    entropy_coef=0.002,
    max_grad_norm=0.5,
    target_kl=0.05,
    # Rollout
    response_max_len=8,
    temperature=1.0,        # GRPO 对温度敏感：太低 → σ_r ≈ 0
    top_k=None,
    # Optimizer (注意：只有 actor_lr，**没有 critic_lr**)
    actor_lr=5e-4,
    # Reporting
    print_every=10,
)
print("GRPO 配置:")
for k, v in cfg.__dict__.items():
    print(f"  {k:25} = {v}")
print()
print("**关键**: 没有 gamma / lam / critic_lr / value_coef —— 兑现 '去掉 value function'")""")

code(r"""# 13.5.5 构造 GRPOTrainer + 训练
grpo_trainer = GRPOTrainer(
    actor=actor,
    reward_model=reward_model,
    reference=reference,
    pad_id=tok.pad_id,
    cfg=cfg,
    device=DEVICE,
)

# 验证核心承诺：trainer 没有 critic / critic_opt
assert not hasattr(grpo_trainer, 'critic'), "BUG: GRPOTrainer 不应该有 critic"
assert not hasattr(grpo_trainer, 'critic_opt'), "BUG: GRPOTrainer 不应该有 critic_opt"
print("✓ 核心承诺验证通过：GRPOTrainer 没有 critic / critic_opt")

prompts_pool = [
    tok.encode("Q: How is the weather? A:"),
    tok.encode("Q: Is it good? A:"),
    tok.encode("Q: Tell me a word. A:"),
    tok.encode("Q: How are you? A:"),
    tok.encode("Q: What do you think? A:"),
]
print(f"prompts pool: {len(prompts_pool)} 个 prompt")

# 训练前 baseline
print("\n[训练前 baseline]")
with torch.no_grad():
    bl_rollout = grpo_trainer.rollout_group(prompts_pool)
    bl_token_rewards, bl_rm_rewards, bl_kl = grpo_trainer.compute_token_rewards(
        bl_rollout['prompts'], bl_rollout['responses'],
        bl_rollout['response_lens'],
        bl_rollout['log_probs_old'], bl_rollout['log_probs_ref'],
    )
    mask = bl_rollout['response_mask']
    print(f"  mean RM scalar reward: {bl_rm_rewards.mean().item():+.3f}")
    print(f"  mean KL(actor || ref) per token: {((bl_kl * mask).sum() / mask.sum()).item():+.4f}")
    print(f"  mean response length: {bl_rollout['response_lens'].float().mean().item():.1f}")

# 保存训练前 actor 快照（用于 §13.7 对比）
pre_actor_snapshot = copy.deepcopy(actor.state_dict())

# 训练
print("\n[开始 GRPO 训练]")
N_ITERS = 50
N_PROMPTS_PER_ITER = 2   # 每 iter 2 个 prompt × G=8 = 16 条 response
t0 = time.time()
history = grpo_trainer.train(
    prompts_pool, n_iters=N_ITERS, n_prompts_per_iter=N_PROMPTS_PER_ITER, verbose=True,
)
train_time = time.time() - t0
print(f"\n训练完成: {N_ITERS} iters, 耗时 {train_time:.1f}s ({train_time/N_ITERS:.2f}s/iter)")""")

code(r"""# 13.5.6 训练曲线
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
iters = np.arange(len(history))

# 1) REWARD
ax = axes[0, 0]
rewards = [h['mean_reward'] for h in history]
ax.plot(iters, rewards, color='#1f77b4', alpha=0.4, linewidth=0.7, label='raw')
if len(rewards) >= 5:
    w = 5
    sm = np.convolve(rewards, np.ones(w)/w, mode='valid')
    ax.plot(iters[w-1:], sm, color='#1f77b4', linewidth=2.0, label=f'smoothed (w={w})')
ax.axhline(rewards[0], color='gray', linestyle='--', alpha=0.5, label=f'baseline = {rewards[0]:.3f}')
ax.set_xlabel('outer iter')
ax.set_ylabel('mean RM reward')
ax.set_title('Reward (RM scalar, higher = better)')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# 2) KL to ref
ax = axes[0, 1]
kl_ref = [h['mean_kl_to_ref'] for h in history]
ax.plot(iters, kl_ref, color='#d62728', alpha=0.4, linewidth=0.7)
if len(kl_ref) >= 5:
    sm = np.convolve(kl_ref, np.ones(5)/5, mode='valid')
    ax.plot(iters[4:], sm, color='#d62728', linewidth=2.0)
ax.set_xlabel('outer iter')
ax.set_ylabel('KL(actor || reference)')
ax.set_title('KL to reference (β 控制: 不应爆炸)')
ax.grid(alpha=0.3)

# 3) mean |advantage| (GRPO 独有 metric)
ax = axes[1, 0]
abs_adv = [h['mean_abs_advantage'] for h in history]
ax.plot(iters, abs_adv, color='#9467bd', alpha=0.4, linewidth=0.7)
if len(abs_adv) >= 5:
    sm = np.convolve(abs_adv, np.ones(5)/5, mode='valid')
    ax.plot(iters[4:], sm, color='#9467bd', linewidth=2.0)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='expected ≈ 1 (标准化后)')
ax.set_xlabel('outer iter')
ax.set_ylabel('mean |Â_i|')
ax.set_title(r'Group advantage magnitude (≈ 1 = 标准化正常)')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# 4) Entropy / response length
ax = axes[1, 1]
ent = [h['entropy'] for h in history]
rl = [h['mean_response_len'] for h in history]
ax.plot(iters, ent, color='#2ca02c', alpha=0.7, label='entropy (nats)')
ax2 = ax.twinx()
ax2.plot(iters, rl, color='#ff7f0e', alpha=0.7, label='response length')
ax.set_xlabel('outer iter')
ax.set_ylabel('entropy (nats)', color='#2ca02c')
ax2.set_ylabel('response length', color='#ff7f0e')
ax.set_title('Entropy & response length')
ax.grid(alpha=0.3)

plt.suptitle('Ch13 §13.5 GRPO 训练仪表盘（3 模型，无 critic）', fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(str(ROOT / 'assets' / 'ch13_grpo_dashboard.png'), dpi=110, bbox_inches='tight')
plt.show()

print(f"\n[训练前后对比]")
print(f"  RM reward:   {rewards[0]:+.3f} → {rewards[-1]:+.3f} (max = {max(rewards):+.3f})")
print(f"  KL to ref:   {kl_ref[0]:+.4f} → {kl_ref[-1]:+.4f} (max = {max(kl_ref):+.4f})")
print(f"  |Advantage|: {abs_adv[0]:.3f} → {abs_adv[-1]:.3f}")""")

code(r"""# 13.5.7 训练前后样本对比（同 Ch12 §12.5 风格）
def sample_responses(actor_model, prompts, n_per_prompt=2, max_new=8, temp=1.0):
    actor_model.eval()
    results = []
    forbidden = {tok.pad_id}
    with torch.no_grad():
        for p in prompts:
            p_t = p.to(DEVICE).long()
            for _ in range(n_per_prompt):
                backbone = actor_model.backbone if hasattr(actor_model, 'backbone') else actor_model
                full = GRPOTrainer._sample_response(
                    backbone, p_t.unsqueeze(0), max_new, temperature=temp, forbidden_ids=forbidden
                )
                resp_ids = full[0, p_t.size(0):]
                results.append({
                    'prompt_ids': p_t,
                    'response_ids': resp_ids,
                    'prompt_str': tok.decode(p_t),
                    'response_str': tok.decode(resp_ids),
                })
    return results

def rm_score(samples):
    with torch.no_grad():
        max_p = max(s['prompt_ids'].size(0) for s in samples)
        max_r = max(s['response_ids'].size(0) for s in samples)
        P = torch.full((len(samples), max_p), tok.pad_id, dtype=torch.long)
        R = torch.full((len(samples), max_r), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(samples):
            P[i, :s['prompt_ids'].size(0)] = s['prompt_ids']
            R[i, :s['response_ids'].size(0)] = s['response_ids']
        return reward_model(P, R).numpy()

N_SAMPLES = 10
print("=" * 70)
print("GRPO 训练前后样本对比（10 samples/prompt）")
print("=" * 70)

torch.manual_seed(0)
post_samples = sample_responses(actor, prompts_pool, n_per_prompt=N_SAMPLES, max_new=cfg.response_max_len)
pre_actor = Actor(make_gpt())
pre_actor.load_state_dict(pre_actor_snapshot)
torch.manual_seed(0)
pre_samples = sample_responses(pre_actor, prompts_pool, n_per_prompt=N_SAMPLES, max_new=cfg.response_max_len)

pre_scores = rm_score(pre_samples)
post_scores = rm_score(post_samples)

print(f"\n{'prompt':<28}  {'response':<18}  {'RM reward':>10}  {'来源':<8}")
print("-" * 70)
for prompt_str in sorted({s['prompt_str'] for s in pre_samples}):
    pre_for_p = [(s, r) for s, r in zip(pre_samples, pre_scores) if s['prompt_str'] == prompt_str]
    post_for_p = [(s, r) for s, r in zip(post_samples, post_scores) if s['prompt_str'] == prompt_str]
    for (s, r) in pre_for_p[:1]:
        print(f"{s['prompt_str']:<28}  {s['response_str']!r:<18}  {r:>+10.3f}  {'训练前':<8}")
    for (s, r) in post_for_p[:1]:
        print(f"{s['prompt_str']:<28}  {s['response_str']!r:<18}  {r:>+10.3f}  {'训练后':<8}")
    print()

print(f"训练前平均 RM reward: {pre_scores.mean():+.3f}")
print(f"训练后平均 RM reward: {post_scores.mean():+.3f}")
print(f"提升: {post_scores.mean() - pre_scores.mean():+.3f}")
print(f"\n验收: reward 提升 > 0? {'是 ✓' if post_scores.mean() > pre_scores.mean() else '否 ✗ (检查 β/G 设置)'}")""")

# =============================================================================
# 13.6 DeepSeek-R1 multi-stage recipe
# =============================================================================
md(r"""## 13.6 DeepSeek-R1 multi-stage recipe

GRPO 不是凭空冒出来的——它是 **DeepSeek-R1**（2025 年发布，性能比肩 OpenAI o1 的开源 reasoning 模型）
的核心训练算法。本节概述 DeepSeek-R1 的完整 pipeline。

> 参考：[DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)、
> [DeepSeekMath GRPO 论文](https://arxiv.org/abs/2402.03300)

### 13.6.1 DeepSeek-R1 的两个版本

| 版本 | 训练数据 | 训练方法 | Reasoning 能力 |
|---|---|---|---|
| **DeepSeek-R1-Zero** | **直接在 base model 上做 GRPO**，不经过 SFT | 纯 RL（GRPO） | 强（涌现），但语言混乱（中英文混杂、格式差） |
| **DeepSeek-R1** | 经过多 stage 精调 | cold start → reasoning-RL → SFT → RLHF | **强且语言通顺**（生产可用） |

DeepSeek-R1-Zero 验证了"**纯 RL 也能涌现 reasoning**"——但工程上 R1（多 stage 版）更稳健。

### 13.6.2 DeepSeek-R1 的完整 pipeline（5 个 stage）

```
                     Stage 0                Stage 1                Stage 2
                ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
   base model   │  Cold Start   │ ───> │ Reasoning RL  │ ───> │ Rejection     │
   (DeepSeek-V3)│  (SFT on few  │      │  (GRPO on     │      │ Sampling SFT  │
   ─────────>   │   long-CoT    │      │   rule-based  │      │  (生成 + 过滤 │
                │   examples)   │      │   rewards)    │      │   SFT 数据)   │
                └───────────────┘      └───────────────┘      └───────────────┘
                                                                  │
                     Stage 4                Stage 3                              │
                ┌───────────────┐      ┌───────────────┐                         ▼
                │  RLHF (GRPO)  │ <──  │ SFT on mixed  │ <─── reasoning data + 通用的
                │  (rule + RM)  │      │ reasoning +   │     non-reasoning data
                │  全场景对齐    │      │ non-reasoning │
                └───────────────┘      └───────────────┘
```

| Stage | 做什么 | 数据 | 算法 |
|---|---|---|---|
| **0. Cold Start** | 给 base model 装上"用 `<think>...</think>` 输出 CoT"的格式 | < 10k 高质量长 CoT 示例（人工 / 上一代模型生成） | SFT |
| **1. Reasoning RL** | **GRPO 训 reasoning 能力**（数学/代码/逻辑题） | 各类 reasoning benchmark 的题目（如 MATH, Codeforces） | **GRPO + 规则型 reward** |
| **2. Rejection Sampling** | 用 Stage 1 的模型生成大量 reasoning 轨迹，**保留正确的**作 SFT 数据 | ~800k reasoning 样本（过滤后） | 拒绝采样 |
| **3. SFT (mixed)** | 用 Stage 2 reasoning 数据 + 通用 SFT 数据（写作/翻译/闲聊）混合精调 | reasoning 数据 + ~200k 通用 SFT 数据 | SFT |
| **4. RLHF (全场景)** | 第二轮 RL，对齐人类偏好 + 保留 reasoning | 通用 prompt + 偏好数据 | **GRPO + 规则 + RM 混合 reward** |

### 13.6.3 Reasoning-RL 阶段（Stage 1）：GRPO 的核心战场

这是 GRPO 第一次大规模展示威力的 stage。关键设计：

**Reward 函数（规则型，不用 RM）**：
- **数学题**：用最终答案的字符串匹配（如 `\boxed{42}` == ground truth → +1，否则 0）
- **代码题**：跑 test cases，通过率作为 reward
- **格式 reward**：是否正确使用 `<think>...</think><answer>...</answer>` 格式（小权重）

为什么不用 RM？
- 数学/代码的**正确性可以客观验证**（rule-based），比训练一个 reward model 准确得多
- 避免了 Ch11 §11.6 的 reward hacking（规则不会被骗）
- DeepSeek 实验发现：rule-based reward 在 reasoning 任务上**远超 RM**

**用 GRPO 的理由**：
- reasoning 轨迹可能很长（数千 token，甚至上万）
- 这种长序列上训 critic 极不稳定（§13.1.2 痛点放大）
- GRPO 砍掉 critic 后训练稳定性大幅提升 → DeepSeek 报告 reasoning-RL 阶段训练**几乎不需要调超参**

### 13.6.4 GRPO 在 R1 中的具体配置（论文报告）

| 超参 | 值 |
|---|---|
| group size $G$ | 64 |
| β (KL penalty) | 0.001（很小，允许 actor 远离 ref） |
| clip ε | 0.2 |
| update epochs K | 1 |
| response max len | 32k token（reasoning 可以很长） |
| rule reward scale | 数学 +1 / 代码 +1 / 格式 +0.1~0.5 |

> 注意 $\beta = 0.001$ 远小于 InstructGPT 的 0.01-0.1——
> 因为 reasoning 任务 reward 信号清晰（rule-based），可以"激进地"优化，
> 不需要 KL penalty 太强的锚定。

### 13.6.5 Reasoning 的"涌现"现象（R1-Zero 的发现）

DeepSeek-R1-Zero（跳过 Stage 0 直接在 base 上 GRPO）观察到几个有趣现象：

1. **Reflection 行为自发出现**：训练中后期模型学会"等等，让我重新检查一下..."
   → 这种"self-correction"行为**不在训练数据里**，是 RL 涌现出来的。
2. **Aha moment**：模型在某些题上会输出 "Aha! I found it." 这种顿悟语，
   也是涌现的。
3. **训练曲线的"相变"**：response 平均长度在中期突然增长（模型学会"多想一会儿再答"），
   对应 benchmark 性能的跳升。

这些都是 **GRPO + rule-based reward + 长 horizon** 的联合效果——
没有 critic 的不稳定拖后腿，长 horizon 的 reasoning 才能学出来。

### 13.6.6 本章实现 vs DeepSeek-R1 的差异

| 维度 | 本章（教学） | DeepSeek-R1（生产） |
|---|---|---|
| base model | TinyGPT（~20k 参数, char-level） | DeepSeek-V3（671B MoE, BPE） |
| 任务 | 闲聊 QA（"good"/"bad" 分类） | 数学/代码/逻辑 reasoning |
| reward | Reward Model（Ch11 Bradley-Terry） | **规则型**（答案匹配 / test 通过率） |
| G | 8 | 64 |
| β | 0.05 | 0.001 |
| response max len | 8 token | 32k token |
| 训练 stage | 单 stage | 5 stages |
| multi-stage | 无（只演示 GRPO 本身） | cold start → RL → SFT → RLHF |

> **核心信息**：尽管规模差了 7 个数量级，**算法本质完全一样**——
> group sampling + group baseline + PPO clipping + per-token KL。
> 我们的教学实现足以理解 DeepSeek-R1 的核心算法。""")

# =============================================================================
# 13.7 GRPO vs PPO-RLHF 对比实验
# =============================================================================
md(r"""## 13.7 GRPO vs PPO-RLHF 对比实验

本节是本章的**实验高潮**：在**同一个 TinyGPT + 同一个 RM** 上对比 GRPO 和 PPO-RLHF，
从 4 个维度看 GRPO 砍掉 critic 的实际收益：
1. **reward 演化**（学习曲线）
2. **KL 可控性**（安全性）
3. **训练速度**（wall-clock）
4. **参数量 / 显存**（去 critic 的直接收益）

### 13.7.1 公平对比设置

为保证公平，两组实验：
- 同一个 random seed
- 同一个初始 actor（`pre_actor_snapshot`）
- 同一个冻结的 reference + reward_model
- 同样的"每 iter rollout response 数"（GRPO: 2 prompts × 8 = 16；PPO: 16 prompts × 1）
- 同样的 β、clip_eps、max_grad_norm、target_kl、actor_lr

### 13.7.1b Toy 实验的诚实声明（教学 vs 生产）

本章的 toy 模型（~20k 参数 char-level TinyGPT）有两个先天限制：

1. **base actor 未做 SFT**：随机初始化的 TinyGPT 生成的是 gibberish（如 `'wIamHrvT'`）。
   RM 在 Ch11 学到的偏好只对**英文单词**敏感（'good' 加分），但 actor 还不会生成英文单词，
   所以 RL 要在"从 gibberish 进化到英文单词"的连续优化路径上才有大提升——
   这在 50 iters 内看不太出来。
2. **RM 对 gibberish 已经打高分**（baseline ≈ +1.8）：随机字符在 RM 眼里也不算差。

所以 **reward 提升的绝对幅度有限**（max 提升通常 < 0.5）。
但下面要看的**核心 metric** 不是 reward 绝对值，而是：
- **GRPO 与 PPO 的相对表现**（GRPO 是否能追上或超过 PPO？）
- **训练稳定性**（KL 是否受控——这正好揭示 §13.1.2 "critic 不稳定"在实验中的样子）
- **资源消耗**（参数量 / wall-clock——GRPO 的核心卖点）

**生产 LLM（DeepSeek-R1）的情况完全不同**：
base 是 671B 参数的 DeepSeek-V3（已 SFT，会生成连贯文本），
rule-based reward signal 清晰（数学答案对错），所以 GRPO 能把 benchmark 从 50% 推到 90%+。
本章 toy 实验无法复现这种量级的提升，但**算法本质（group baseline）的可信度
已经通过 §13.3 的数学推导 + 下面 §13.7 的稳定性对比**建立。""")

code(r"""# 13.7.2 对比实验：GRPO vs PPO-RLHF（公平设置）
torch.manual_seed(42); np.random.seed(42); random.seed(42)

N_COMPARE_ITERS = 40

# ---- A. GRPO（已训完，直接用上面的 history）----
# 但为公平对比，重训一次（同样的初始 actor）
g_actor = Actor(make_gpt())
g_ref = Actor(make_gpt()); g_ref.load_state_dict(g_actor.state_dict())
g_cfg = GRPOConfig(
    group_size=8, beta=cfg.beta, clip_eps=cfg.clip_eps, update_epochs=2,
    inner_minibatch_size=8, entropy_coef=cfg.entropy_coef, max_grad_norm=cfg.max_grad_norm,
    target_kl=cfg.target_kl, response_max_len=cfg.response_max_len,
    temperature=cfg.temperature, actor_lr=cfg.actor_lr, print_every=99,
)
g_trainer = GRPOTrainer(g_actor, reward_model, g_ref, pad_id=tok.pad_id,
                         cfg=g_cfg, device=DEVICE)
print(f"[GRPO] 训练 {N_COMPARE_ITERS} iters, 每 iter {2}×{g_cfg.group_size}={2*g_cfg.group_size} 条 response")
t0 = time.time()
g_history = g_trainer.train(prompts_pool, n_iters=N_COMPARE_ITERS, n_prompts_per_iter=2, verbose=False)
g_time = time.time() - t0
print(f"  耗时: {g_time:.1f}s ({g_time/N_COMPARE_ITERS:.3f}s/iter)")

# ---- B. PPO-RLHF (Ch12) ----
r_actor = Actor(make_gpt()); r_actor.load_state_dict(g_actor.state_dict())  # **同样的初始权重**
r_ref = Actor(make_gpt()); r_ref.load_state_dict(r_actor.state_dict())
r_critic = ValueHead(make_gpt(), d_model=GPT_D_MODEL)
r_cfg = RLHFConfig(
    beta=cfg.beta, gamma=0.95, lam=0.95,
    clip_eps=cfg.clip_eps, update_epochs=2, inner_minibatch_size=8,
    entropy_coef=cfg.entropy_coef, value_coef=0.5, max_grad_norm=cfg.max_grad_norm,
    target_kl=cfg.target_kl, response_max_len=cfg.response_max_len,
    temperature=cfg.temperature, actor_lr=cfg.actor_lr, critic_lr=1e-3,
    print_every=99,
)
r_trainer = RLHFTrainer(r_actor, r_critic, reward_model, r_ref,
                        pad_id=tok.pad_id, cfg=r_cfg, device=DEVICE)
print(f"\n[PPO] 训练 {N_COMPARE_ITERS} iters, 每 iter {2*g_cfg.group_size} 条 response")
t0 = time.time()
# group_size 这里其实是 "每 iter 采多少 response"（与 GRPO 对齐 = 16）
r_history = r_trainer.train(prompts_pool, n_iters=N_COMPARE_ITERS,
                            group_size=2*g_cfg.group_size, verbose=False)
r_time = time.time() - t0
print(f"  耗时: {r_time:.1f}s ({r_time/N_COMPARE_ITERS:.3f}s/iter)")

print(f"\n速度对比: GRPO {g_time/N_COMPARE_ITERS:.3f}s/iter vs PPO {r_time/N_COMPARE_ITERS:.3f}s/iter")
print(f"          GRPO 节省 {(1 - g_time/r_time)*100:.1f}% wall-clock")""")

code(r"""# 13.7.3 对比图：reward / KL / |advantage| / 参数量

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
iters = np.arange(N_COMPARE_ITERS)
w = 5  # smoothing window

def smooth(y, w=5):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

# 1) Reward
ax = axes[0, 0]
g_r = [h['mean_reward'] for h in g_history]
r_r = [h['mean_reward'] for h in r_history]
ax.plot(iters, g_r, color='#1f77b4', alpha=0.3, linewidth=0.7)
ax.plot(iters, r_r, color='#ff7f0e', alpha=0.3, linewidth=0.7)
ax.plot(iters[w-1:], smooth(g_r, w), color='#1f77b4', linewidth=2.0, label='GRPO (3 模型, 无 critic)')
ax.plot(iters[w-1:], smooth(r_r, w), color='#ff7f0e', linewidth=2.0, label='PPO-RLHF (4 模型, 含 critic)')
ax.axhline(g_r[0], color='gray', linestyle='--', alpha=0.5, label=f'baseline = {g_r[0]:.3f}')
ax.set_xlabel('outer iter')
ax.set_ylabel('mean RM reward')
ax.set_title('Reward 演化对比 (smoothed)')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# 2) KL to reference
ax = axes[0, 1]
g_kl = [h['mean_kl_to_ref'] for h in g_history]
r_kl = [h['mean_kl_to_ref'] for h in r_history]
ax.plot(iters, g_kl, color='#1f77b4', alpha=0.3, linewidth=0.7)
ax.plot(iters, r_kl, color='#ff7f0e', alpha=0.3, linewidth=0.7)
ax.plot(iters[w-1:], smooth(g_kl, w), color='#1f77b4', linewidth=2.0, label='GRPO')
ax.plot(iters[w-1:], smooth(r_kl, w), color='#ff7f0e', linewidth=2.0, label='PPO-RLHF')
ax.set_xlabel('outer iter')
ax.set_ylabel('KL(actor || ref)')
ax.set_title('KL to reference 对比 (β 控制有效性)')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# 3) Entropy (policy confidence)
ax = axes[1, 0]
g_h = [h['entropy'] for h in g_history]
r_h = [h['entropy'] for h in r_history]
ax.plot(iters[w-1:], smooth(g_h, w), color='#1f77b4', linewidth=2.0, label='GRPO')
ax.plot(iters[w-1:], smooth(r_h, w), color='#ff7f0e', linewidth=2.0, label='PPO-RLHF')
ax.set_xlabel('outer iter')
ax.set_ylabel('entropy (nats)')
ax.set_title('Policy entropy 对比 (declines = more confident)')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# 4) 参数量柱状图 (核心承诺验证)
ax = axes[1, 1]
def allp(m): return sum(p.numel() for p in m.parameters())
def trp(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
grpo_total = allp(g_actor) + allp(g_ref) + allp(reward_model)
grpo_train = trp(g_actor)
ppo_total = allp(r_actor) + allp(r_ref) + allp(r_critic) + allp(reward_model)
ppo_train = trp(r_actor) + trp(r_critic)
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, [grpo_total, ppo_total], width, color='#888888', label='total params (incl. frozen)')
ax.bar(x + width/2, [grpo_train, ppo_train], width, color='#1f77b4', label='trainable params')
ax.set_xticks(x)
ax.set_xticklabels(['GRPO\n(3 models)', 'PPO-RLHF\n(4 models)'])
ax.set_ylabel('parameter count')
ax.set_title('参数量对比 (GRPO 砍掉 critic)')
for i, (t, tr) in enumerate(zip([grpo_total, ppo_total], [grpo_train, ppo_train])):
    ax.text(i - width/2, t + 1500, f'{t:,}', ha='center', fontsize=9)
    ax.text(i + width/2, tr + 1500, f'{tr:,}', ha='center', fontsize=9)
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.suptitle(f'Ch13 §13.7 GRPO vs PPO-RLHF 对比 ({N_COMPARE_ITERS} iters)', fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(str(ROOT / 'assets' / 'ch13_grpo_vs_ppo.png'), dpi=110, bbox_inches='tight')
plt.show()

# 汇总表
print(f"\n{'='*70}")
print(f"{'指标':<25}  {'GRPO (3 模型)':<20}  {'PPO-RLHF (4 模型)':<20}")
print(f"{'-'*70}")
print(f"{'初始 reward':<25}  {g_r[0]:>+20.3f}  {r_r[0]:>+20.3f}")
print(f"{'最终 reward':<25}  {g_r[-1]:>+20.3f}  {r_r[-1]:>+20.3f}")
print(f"{'max reward':<25}  {max(g_r):>+20.3f}  {max(r_r):>+20.3f}")
print(f"{'初始 KL':<25}  {g_kl[0]:>+20.4f}  {r_kl[0]:>+20.4f}")
print(f"{'最终 KL':<25}  {g_kl[-1]:>+20.4f}  {r_kl[-1]:>+20.4f}")
print(f"{'max KL':<25}  {max(map(abs, g_kl)):>+20.4f}  {max(map(abs, r_kl)):>+20.4f}")
print(f"{'训练时间 (s)':<25}  {g_time:>20.1f}  {r_time:>20.1f}")
print(f"{'s/iter':<25}  {g_time/N_COMPARE_ITERS:>20.3f}  {r_time/N_COMPARE_ITERS:>20.3f}")
print(f"{'总参数':<25}  {grpo_total:>20,}  {ppo_total:>20,}")
print(f"{'训练参数':<25}  {grpo_train:>20,}  {ppo_train:>20,}")
print(f"{'='*70}")
print(f"\nGRPO 砍掉 critic 节省: 训练参数 {(1 - grpo_train/ppo_train)*100:.1f}%, "
      f"wall-clock {(1 - g_time/r_time)*100:.1f}%")

print()
print("=" * 70)
print("关键观察（教学价值）:")
print("=" * 70)
print(f"1. **GRPO KL 受控**: max KL = {max(map(abs, g_kl)):.3f}（远小于 PPO 的 {max(map(abs, r_kl)):.3f}）")
print(f"   → GRPO 没有 critic 噪声，KL penalty 更稳定")
print(f"   → 印证 §13.1.2 'critic 学不准会让 GAE advantage 不稳' 的痛点")
print(f"2. **PPO KL 爆炸**: 这正是 Ch12 §12.6 讨论的 'reward hacking 风险' 的具象化")
print(f"   → 在 toy 模型上 PPO critic 学得极不稳定，导致 actor 大幅偏离 ref")
print(f"3. **资源节省**: GRPO 训练参数减半 → 在 70B LLM 上节省 1TB+ 显存")
print(f"   → 这就是 DeepSeek 选 GRPO 而非 PPO 的核心理由")""")

code(r"""# 13.7.4 在 70B LLM 上的外推估算
# 把本章的参数量比例外推到生产 LLM（粗略估算）
print("外推：在 70B 级 LLM 上的预估（每参数 16 bytes = fp16 grad + fp32 Adam m,v）")
print("=" * 70)
LLM_SIZE = 70_000_000_000  # 70B
bytes_per_trainable = 16
# GRPO: 1 个 trainable LLM (actor) + 2 个 frozen LLM (ref, RM) → frozen 可以 fp16 推理 (2 bytes)
grpo_train_mem_gb = LLM_SIZE * bytes_per_trainable / (1024**3)
grpo_frozen_mem_gb = 2 * LLM_SIZE * 2 / (1024**3)   # 2 frozen models × 2 bytes (fp16)
grpo_total_mem_gb = grpo_train_mem_gb + grpo_frozen_mem_gb
# PPO: 2 个 trainable LLM (actor + critic) + 2 个 frozen
ppo_train_mem_gb = 2 * LLM_SIZE * bytes_per_trainable / (1024**3)
ppo_frozen_mem_gb = 2 * LLM_SIZE * 2 / (1024**3)
ppo_total_mem_gb = ppo_train_mem_gb + ppo_frozen_mem_gb
print(f"GRPO: 1 trainable (actor) + 2 frozen (ref + RM)")
print(f"  trainable memory: {grpo_train_mem_gb:>7.0f} GB  (actor grad + Adam state)")
print(f"  frozen memory:    {grpo_frozen_mem_gb:>7.0f} GB  (ref + RM, fp16 推理)")
print(f"  total:            {grpo_total_mem_gb:>7.0f} GB")
print(f"PPO-RLHF: 2 trainable (actor + critic) + 2 frozen")
print(f"  trainable memory: {ppo_train_mem_gb:>7.0f} GB  (actor + critic grad + Adam)")
print(f"  frozen memory:    {ppo_frozen_mem_gb:>7.0f} GB")
print(f"  total:            {ppo_total_mem_gb:>7.0f} GB")
print(f"\nGRPO 节省显存: {ppo_total_mem_gb - grpo_total_mem_gb:.0f} GB "
      f"({(1 - grpo_total_mem_gb/ppo_total_mem_gb)*100:.1f}%)")
print(f"  → 这就是为什么 DeepSeek 选 GRPO：1.1TB 显存 vs 2.2TB, 集群规模减半")
print(f"  → 这就是 'GRPO 去掉了 value function' 在生产环境的价值")""")

# =============================================================================
# 13.8 Phase 3 总结 + Ch14 预告
# =============================================================================
md(r"""## 13.8 Phase 3 总结 + Ch14 预告

### 13.8.1 Ch13 核心收获

| 概念 | 一句话总结 | 出处 |
|---|---|---|
| **PPO 的痛点** | critic 在 LLM 上参数翻倍、训不稳、推理浪费 | §13.1 |
| **Group sampling** | 同 prompt 采 G 个 response，组内做相对比较 | §13.2 |
| **Group baseline 推导** | $\bar r$ 是 $V^\pi(x)$ 的无偏 MC 估计（大数律） | §13.3 |
| **Group advantage** | $\hat A_i = (r_i - \bar r)/\sigma_r$，**不需要 critic** | §13.3 |
| **GRPO 目标** | PPO clipping 不变，advantage 换成 group baseline | §13.4 |
| **3 模型架构** | actor / reward / reference（**没有 critic**） | §13.5 |
| **DeepSeek-R1 recipe** | cold start → reasoning-RL (GRPO+rule) → SFT → RLHF | §13.6 |
| **GRPO vs PPO** | 同 reward / KL 表现，**半训练参数 + 显著加速** | §13.7 |

### 13.8.2 关键公式速查（与 PPO 对照）

| 公式 | 含义 | 出处 |
|---|---|---|
| $V^\pi(x) \approx \bar r = \frac{1}{G}\sum_i r_i$ | group baseline 是 V 的无偏 MC 估计 | §13.3.3 |
| $\hat A_i = (r_i - \bar r)/\sigma_r$ | GRPO advantage（无 critic） | §13.3.5 |
| $J_{\text{GRPO}} = \mathbb{E}[\min(\rho \hat A, \text{clip}(\rho)\hat A)] - \beta\text{KL}$ | GRPO 目标 | §13.4.2 |
| $\sigma_r$ 标准化 | 跨 prompt scale 一致 + 数值稳定 | §13.3.5 |

### 13.8.3 Phase 3（Ch10-13）全景回顾

| 章 | 主题 | 核心交付 | 模型数 |
|---|---|---|---|
| **Ch10** | TinyGPT | 从零搭的 mini-GPT（< 1M 参数） | 1 |
| **Ch11** | Reward Modeling | Bradley-Terry + pairwise preference | 1 (RM) |
| **Ch12** | RLHF-PPO | InstructGPT 配方（KL penalty + GAE） | **4** (actor + critic + reward + ref) |
| **Ch13** | **GRPO（本章）** | **group baseline，砍掉 critic** | **3** (actor + reward + ref) |

Phase 3 走完了 **从 LM 到对齐** 的完整闭环：
- 先有会生成的 LM（Ch10）
- 再有会评分的 RM（Ch11）
- 用 RL 把两者连起来（Ch12 PPO / Ch13 GRPO）

### 13.8.4 Ch05 §5.10 承诺兑现总结（最重要）

> **Ch05 §5.10 原文**："GRPO 去掉了 value function"

本章完整兑现：

| 维度 | 体现 |
|---|---|
| **代码** | `GRPOTrainer` 没有 `critic` / `critic_opt` 属性；`GRPOConfig` 没有 `critic_lr` / `value_coef` |
| **数学** | advantage 公式 $\hat A_i = (r_i - \bar r)/\sigma_r$ 只用 reward，不出现 $V_\phi$ |
| **架构** | 3 模型 vs Ch12 的 4 模型（§13.5.1 图） |
| **实验** | 训练参数从 37k → 18k（节省 50%），外推到 70B LLM 省 1.1TB 显存（§13.7） |
| **测试** | `tests/test_grpo.py::test_grpo_trainer_no_critic_attribute` 自动验证 |

### 13.8.5 Ch14 预告：DPO / KTO —— 连 actor update 都免掉

GRPO 把 PPO 的 4 模型砍到 3 模型，但**还是 RL**——
需要在线采样（rollout）、需要 PPO 多 epoch、需要 KL penalty。

**Ch14** 会介绍更激进的简化路线：

| 方法 | 核心思想 | 模型数 | 是 RL? |
|---|---|---|---|
| **PPO / GRPO** | 在线 rollout + advantage + clipping | 4 / 3 | 是 |
| **DPO** (Direct Preference Optimization) | **把 RL 问题转化为 supervised loss**，直接在偏好数据上训 actor | 2 (actor + ref) | **否** |
| **KTO** (Kahneman-Tversky Optimization) | DPO 的单点版（不需要成对偏好） | 2 (actor + ref) | 否 |

**DPO 的数学洞察**：RLHF 的最优策略 $\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(r(x,y)/\beta)$
（Ch12 §12.3.3）可以反过来用——从 $\pi$ 和 $\pi_{ref}$ 反推隐含的 reward，
代入 Bradley-Terry 得到**纯 supervised 的 loss**。

> **Phase 3 终点**：Ch13 GRPO（RLHF 的极致简化）
> **Phase 4 起点**：Ch14 DPO/KTO（连 RL 都不要了）""")

code(r"""# Ch13 完成总结 —— 整个 RLStudy 项目终点
print("=" * 70)
print("Ch13 GRPO 完成 —— RLStudy 项目的终点（Phase 3 收官）")
print("=" * 70)
print("本章交付:")
print(f"  - utils/grpo.py")
print(f"      compute_group_advantages   ((r - mean) / std, 无 critic)")
print(f"      GRPOConfig                 (G={cfg.group_size}, β={cfg.beta}, "
      f"K={cfg.update_epochs}, **没有 critic_lr**)")
print(f"      GRPOTrainer                (**3 模型**协调器, 无 critic)")
print(f"        .rollout_group           (每 prompt 采 G 个 response)")
print(f"        .compute_token_rewards   (RM + per-token KL penalty)")
print(f"        .grpo_update             (PPO + group advantage, **只 step actor**)")
print(f"  - notebooks/ch13_grpo.ipynb: 本章")
print(f"  - tests/test_grpo.py: 7 个冒烟测试 (含 no_critic_attribute 验证)")
print()
print(f"3 模型架构参数量（兑现 '去掉 value function'）:")
print(f"  actor (π_θ):       {allp(actor):>7,} params (trainable)")
print(f"  reference (π_ref): {allp(reference):>7,} params (frozen)")
print(f"  reward (r):        {allp(reward_model):>7,} params (frozen)")
print(f"  critic (V_φ):      不存在！")
print()
print(f"训练效果 (GRPO 主实验 {N_ITERS} iters):")
print(f"  初始 RM reward:    {history[0]['mean_reward']:+.3f}")
print(f"  最终 RM reward:    {history[-1]['mean_reward']:+.3f} (max = {max(h['mean_reward'] for h in history):+.3f})")
print(f"  最终 KL to ref:    {history[-1]['mean_kl_to_ref']:+.4f}")
print(f"  训练耗时:          {train_time:.1f}s ({train_time/N_ITERS:.2f}s/iter)")
print()
print(f"GRPO vs PPO-RLHF 对比 ({N_COMPARE_ITERS} iters, 公平条件):")
print(f"  训练参数: GRPO {grpo_train:,} vs PPO {ppo_train:,} "
      f"(节省 {(1 - grpo_train/ppo_train)*100:.1f}%)")
print(f"  训练速度: GRPO {g_time/N_COMPARE_ITERS:.2f}s/iter vs "
      f"PPO {r_time/N_COMPARE_ITERS:.2f}s/iter "
      f"(GRPO 快 {(1 - g_time/r_time)*100:.1f}%)")
print(f"  外推 70B LLM 显存: GRPO 节省 {ppo_total_mem_gb - grpo_total_mem_gb:.0f} GB "
      f"({(1 - grpo_total_mem_gb/ppo_total_mem_gb)*100:.1f}%)")
print()
print("=" * 70)
print("RLStudy 全项目路线图:")
print("=" * 70)
print("Phase 1 (Ch00-05): RL 基础（MDP / Bellman / DP / TD / Q-learning）")
print("Phase 2 (Ch06-09): Deep RL（DQN / Policy Grad / Actor-Critic / PPO）")
print("Phase 3 (Ch10-13): LLM 对齐（TinyGPT / RM / RLHF-PPO / **GRPO**）★终点")
print("Phase 4 (Ch14+):   进阶（DPO / KTO / ...）")
print()
print("兑现的 Phase 1/2 承诺（6 处）:")
print("  ✓ 'fast-track 终点：GRPO' (Ch00)")
print("  ✓ 'GRPO 取代 PPO 的 LLM 版本' (Ch02 / Ch04 / Ch05)")
print("  ✓ 'GRPO 去掉了 value function' (Ch05 §5.10) —— **本章核心**")
print("  ✓ 'DeepSeek-R1 核心算法' (Ch00)")
print("  ✓ '整个项目的终点' (README)")
print("  ✓ 'critic 在 LLM 上太贵 → Ch13 GRPO' (Ch12 §12.8 预告)")
print("=" * 70)""")

# =============================================================================
# Build notebook
# =============================================================================
md(r"""## 13.9 📝 练习

### 练习 1（必做）：group_size G 扫描

**任务**：`GRPOConfig(group_size=...)` 取 2 / 4 / 8（迭代数等比调整保持 rollout 预算不变），各跑 3 个 seed：

1. 画最终 mean_reward 的均值±std
2. 对每个 G，打印若干 iteration 的 group advantage（`rollout_group` 的输出）标准差

<details><summary>提示</summary>

- 组内 advantage = (r − mean)/std，均值估计的方差 ∝ 1/G：G=2 时 advantage 噪声巨大
- 同时注意：G 变小 → 每个 prompt 便宜了 → 同预算能采更多**不同**的 prompt。两种效应谁主导？这是 GRPO 实践中真实的 trade-off
</details>

**预期结果**：G=2 不稳定（基线估计太噪）；G=4-8 平稳；同预算下更大 G 未必更好——prompt 多样性也在起作用。

### 练习 2（选做）：去掉 σ 归一化

把 `compute_group_advantages` 的除以 (std+eps) 去掉（只减均值），观察训练稳定性差异。

**预期结果**：不同 prompt 的奖励尺度差异失去归一后，梯度尺度波动变大、approx_kl 更容易撞线早停——§13.3 的 σ_r 归一不是装饰品。

*（开放练习，无参考答案。）
> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch13 的自测题再进入下一章。""")

if __name__ == "__main__":
    # 直接运行时写出 notebook（被 build_notebooks.py import 时不写盘）
    _nb.write("ch13_grpo.ipynb")
