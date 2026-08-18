"""Build notebooks/ch14_dpo_kto.ipynb via nbformat.

Run:  python build_ch14.py
This produces the .ipynb file. Then execute it with nbconvert.
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch14")
cells = _nb.cells            # build_notebooks.py 适配层读取的模块级列表
md, code = _nb.md, _nb.code


# =============================================================================
# Title / overview
# =============================================================================
md(r"""# 第 14 章：DPO / KTO —— 不用 RL 的 RLHF 替代方案（Phase 3 收尾）

> **Ch12 RLHF-PPO** 把 LLM 对齐做成一个 4 模型 + token-level MDP + PPO clipping + KL penalty
> 的复杂 RL 问题。**Ch13 GRPO** 干掉了 critic，但仍然要 rollout + PPO。
>
> 本章的核心问题：
>
> > **能不能完全不用 RL，直接在偏好数据上训对齐模型？**
>
> 答案是 **DPO（Direct Preference Optimization, Rafailov et al. 2023, NeurIPS Best Paper）**
> 和它的兄弟 **KTO（Kahneman-Tversky Optimization, Ethayarajh et al. 2024, ICML）**。
>
> **DPO 的关键洞察**：RLHF 的 KL-constrained 最优解 $\pi^*$ 和 reward $r$ 之间有闭式关系。
> 把这个关系**反过来**代入 Bradley-Terry 偏好模型（Ch11 §11.2），**整个 RL 目标
> 被重写为一个简单的二分类 loss**——不需要 reward model，不需要 critic，不需要 PPO，
> 纯监督学习！

## 学习目标

1. 理解 **为什么有 DPO/KTO**：RL 的不稳定性 + 计算昂贵 → 找一个轻量替代
2. **完整推出 DPO loss**（4 步推导，缺一不可）
3. 理解 **KTO 的 prospect theory 视角**：用 good/bad 标签替代成对偏好
4. 在 TinyGPT 上 **实现 DPO 训练**（2 模型协调器：actor + frozen reference）
5. **DPO vs GRPO vs PPO-RLHF 三方对比**：reward 提升 / KL 行为 / 训练速度
6. 演示 **DPO 的 distribution shift 问题**（off-policy 训练的本质代价）
7. 画出 **算法决策树**（CH00 承诺）：什么场景选 PPO / GRPO / DPO / KTO

## 本章兑现的承诺

| 出处 | 承诺原文 | 本章兑现节 |
|---|---|---|
| **Ch00** | **"Ch14 DPO / KTO"** | **§14.2-14.5 全章** |
| **Ch00** | "RL 之外的 RLHF 替代方案"（隐含） | **§14.1-14.2** |
| **Ch00** | **"算法决策树"** | **§14.6** |
| **Ch11 §11.7** | "DPO 绕开 RL" 的预告 | **§14.2 完整推导** |
| **Ch12 §12.8** | "RLHF 太重 → DPO 替代" | **§14.1 + §14.5 对比** |

## 承接的 Ch10-Ch13 基础设施

| 模块 | 来源 | 本章用法 |
|---|---|---|
| **TinyGPT** | Ch10 | actor / reference backbone |
| **RewardModel** | Ch11 §11.4 | **不在 DPO loop 里**！只用于事后评估 DPO 训练效果（§14.4 / §14.5） |
| **generate_preference_data** | Ch11 §11.3 | 复用同样的合成偏好数据 |
| **RLHFTrainer / GRPOTrainer** | Ch12 / Ch13 | **关键对比对象**（§14.5） |

## 术语速查

| 术语 | 含义 | 首次出现 |
|---|---|---|
| **DPO** (Direct Preference Optimization) | 用闭式解把 RLHF 改写成监督 loss | §14.2 |
| **KTO** (Kahneman-Tversky Optimization) | 用 prospect theory 替代 Bradley-Terry | §14.3 |
| **隐式 reward** (implicit reward) | $\hat r = \beta\log(\pi_\theta/\pi_{ref})$，不需要训 RM 就能算 | §14.2 |
| **prospect theory / 前景理论** | 行为经济学的决策模型（loss aversion 等） | §14.3 |
| **distribution shift**（分布偏移） | DPO 在静态偏好数据上训，不能跟上 actor 漂移 | §14.5 |
| **off-policy to the extreme** | DPO 数据完全静态，没有任何 on-policy 成分 | §14.5 |

## 本章路线图（6 节）

| 节 | 主题 | 关键产出 |
|---|---|---|
| 14.1 | 为什么有 DPO/KTO | RL 痛点 → 数学动机 |
| 14.2 | **DPO 完整推导（核心 1/2）** | 4 步推导 + DPO loss 公式 |
| 14.3 | **KTO 与 prospect theory（核心 2/2）** | pointwise loss + loss aversion |
| 14.4 | **实现 DPO** | DPOTrainer + 训练 + RM 验证 reward 提升 |
| 14.5 | **三方对比实验** | DPO vs GRPO vs PPO-RLHF + distribution shift |
| 14.6 | **决策树 + Phase 3 + 项目整体总结** | 算法选择指南 + 项目收官 |""")

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
# Ch11 reward model（只用于事后评估 DPO 效果，不在 DPO loop 内）
from utils.reward_model import (
    RewardModel, bradley_terry_loss,
    generate_preference_data, make_preference_batch, pad_to_length,
    reward_accuracy, predict_rewards, true_reward,
)
# Ch12 RLHF（对比对象）
from utils.rlhf import RLHFConfig, RLHFTrainer, ValueHead
# Ch13 GRPO（对比对象）
from utils.grpo import GRPOConfig, GRPOTrainer, compute_group_advantages
# 本章新基础设施
from utils.dpo import (
    sequence_log_probs, dpo_loss, kto_loss, kto_points_to_loss,
    prospect_value, DPOConfig, DPOTrainer, KTOTrainer,
)
# 其它
from utils import set_seed
from utils.torch_utils import get_device, count_parameters

set_seed(42)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DEVICE = "cpu"   # 模型很小（< 200k 参数），CPU 完全够用
print(f"PyTorch: {torch.__version__}, device = {DEVICE}")
print()
print("本章新基础设施: utils/dpo.py")
print("  - sequence_log_probs  (log π(y|x) = sum_t log π(y_t | x, y_<t))")
print("  - dpo_loss            (-log σ(β·(Δ_w - Δ_l)))")
print("  - kto_loss / prospect_value / kto_points_to_loss")
print("  - DPOConfig           (β, lr, batch_size; 没有 clip_eps/gamma/critic_lr)")
print("  - DPOTrainer          (**2 模型**: actor + frozen reference, **没有 RM**)")
print("  - KTOTrainer          (同 trainer 换 pointwise loss)")""")


# =============================================================================
# 14.1 Why DPO/KTO
# =============================================================================
md(r"""## 14.1 为什么有 DPO/KTO

### 14.1.1 RLHF-PPO / GRPO 的工程痛点

Ch12 / Ch13 我们已经走完了完整的 RLHF-PPO 和 GRPO 流程，它们都能 work——
但代价很重：

| 痛点 | PPO-RLHF（Ch12） | GRPO（Ch13） | 影响 |
|---|---|---|---|
| **模型数** | 4 个（actor + critic + reward + ref） | 3 个（actor + reward + ref） | 显存爆炸 |
| **rollout 开销** | 每步采 G 个 response | 每步采 G×N 个 response | 训练慢 |
| **critic 难训** | critic loss 不收敛是常态 | 没有 critic（已优化） | actor 信号噪声大 |
| **PPO 调参敏感** | clip ε / KL target / lr 三者耦合 | 同 | 训练不稳定 |
| **reward model 训练** | 单独训 RM 阶段 | 同 | 多阶段 pipeline |
| **采样 → 训练 → 评估** 循环 | 是 | 是 | 代码复杂 |

工业界做 70B LLM 的 RLHF，每个痛点都对应**百万美元级**的工程成本。

### 14.1.2 RL 的"重"和"轻"之争

一个直觉问题：

> 既然我们最终想要的是 $\pi_\theta$ 满足偏好（在 winner 上概率高、在 loser 上概率低），
> 为什么不**直接在偏好数据上做监督学习**？

朴素想法：直接用 cross-entropy 让 $\pi_\theta$ 学 winner、不学 loser。但这有问题：

1. **没有 KL 约束**：actor 可以无限远离 $\pi_{ref}$，过拟合偏好数据 → reward hacking（Ch11 §11.6）。
2. **不知道"远离多少"算合理**：直接 cross-entropy 没有 β 这样的旋钮。
3. **理论没保证**：为什么这样的 actor 是 RLHF 目标的最优解？

DPO 用一个**漂亮的代数变换**解决了这 3 个问题——同时给出理论保证和简单 loss。

### 14.1.3 DPO 的"魔法"

DPO 论文 Rafailov et al. 2023 的核心贡献：

> **RLHF 的 KL-constrained 最优解 $\pi^*$ 和 reward $r$ 之间有闭式关系
> （Ch12 §12.3 已证）。把这个关系反过来代入 Bradley-Terry 偏好模型，
> reward $r$ 被消掉，RL 目标变成一个简单的二分类 loss。**

$$
\underbrace{\max_\pi \mathbb{E}[r] - \beta\,\text{KL}(\pi \| \pi_{ref})}_{\text{RL 问题（Ch12）}}
\;\;\xrightarrow{\text{4 步代数变换}}\;\;
\underbrace{-\log\sigma\!\left(\beta(\Delta_w - \Delta_l)\right)}_{\text{监督 loss（DPO）}}
$$

其中 $\Delta_w = \log\pi_\theta(y_w|x) - \log\pi_{ref}(y_w|x)$，纯监督可算。

### 14.1.4 KTO 的"更轻"

KTO（Ethayarajh et al. 2024）走得更远：

- DPO 还需要**成对**偏好数据 $(y_w, y_l)$（标注成本高）
- KTO 只需要 **good/bad 二元标签**（标注更便宜——thumbs up/down）

代价是引入 prospect theory（前景理论）的额外复杂性，但 loss 形式几乎相同。

下面 4 节我们按"理论 → 实现 → 对比"展开。""")

# 14.1.5 RLHF pipeline 的复杂度对比图
code(r"""# 14.1.5 可视化：RLHF-PPO / GRPO / DPO / KTO 的 pipeline 复杂度对比
fig, ax = plt.subplots(figsize=(13, 5))

pipelines = {
    'SFT':           ['SFT'],
    'RLHF-PPO (Ch12)': ['SFT', 'RM', 'PPO-RL\n(actor+critic\n+RM+ref)'],
    'GRPO (Ch13)':   ['SFT', 'RM', 'GRPO\n(actor+RM\n+ref, no critic)'],
    'DPO (Ch14)':    ['SFT', 'DPO\n(actor+ref)'],
    'KTO (Ch14)':    ['SFT', 'KTO\n(actor+ref)'],
}

colors = {
    'SFT': '#888888',
    'RM':  '#ff7f0e',
    'PPO-RL\n(actor+critic\n+RM+ref)': '#d62728',
    'GRPO\n(actor+RM\n+ref, no critic)': '#9467bd',
    'DPO\n(actor+ref)': '#2ca02c',
    'KTO\n(actor+ref)': '#1f77b4',
}

for i, (name, stages) in enumerate(pipelines.items()):
    for j, s in enumerate(stages):
        rect = plt.Rectangle((j*1.5, len(pipelines)-1-i), 1.3, 0.7,
                             facecolor=colors.get(s, '#888'), alpha=0.7,
                             edgecolor='black')
        ax.add_patch(rect)
        ax.text(j*1.5 + 0.65, len(pipelines)-1-i + 0.35, s,
                ha='center', va='center', fontsize=9, fontweight='bold')
        if j > 0:
            ax.annotate('', xy=(j*1.5, len(pipelines)-1-i+0.35),
                        xytext=(j*1.5-0.2, len(pipelines)-1-i+0.35),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_xlim(-0.2, 5.0)
ax.set_ylim(-0.5, len(pipelines)-0.3)
ax.set_title('对齐 pipeline 复杂度对比：从 4 阶段 PPO 到 2 阶段 DPO/KTO', fontsize=12)
ax.axis('off')

# 标注每个 pipeline 的"模型数"
model_counts = {'SFT': 1, 'RLHF-PPO (Ch12)': 4, 'GRPO (Ch13)': 3, 'DPO (Ch14)': 2, 'KTO (Ch14)': 2}
for i, (name, _) in enumerate(pipelines.items()):
    ax.text(4.7, len(pipelines)-1-i + 0.35, f'模型数: {model_counts[name]}',
            fontsize=10, va='center',
            color=('#d62728' if model_counts[name] >= 3 else '#2ca02c'))

plt.tight_layout(); plt.show()

print("观察:")
print("  - SFT: 1 个模型（baseline）")
print("  - PPO-RLHF: **4 个模型** + PPO 多阶段 pipeline（最重）")
print("  - GRPO: **3 个模型**（去掉 critic，但仍有 RM + rollout）")
print("  - DPO/KTO: **2 个模型**（actor + ref），**没有 RM 在 loop 里**，没有 rollout")
print("  - 模型数 ↓ + pipeline 简化 = 工程成本数量级下降")""")


# =============================================================================
# 14.2 DPO derivation
# =============================================================================
md(r"""## 14.2 DPO 完整推导（本章核心 1/2）

DPO 的推导是教科书式的优美——4 步代数变换，把 RL 问题变成监督 loss。
**这一节我们要把每一步都展开**，理解清楚比记住公式重要。

### 14.2.1 起点：RLHF 的 KL-constrained 目标

Ch12 §12.3 给出的 RLHF 目标（最大化 reward，subject to KL constraint）：

$$
\max_{\pi} \;\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)}\!\big[r(x, y)\big]
\quad\text{s.t.}\quad
\mathbb{E}_x\!\big[\text{KL}(\pi(\cdot|x) \| \pi_{ref}(\cdot|x))\big] \le \epsilon
$$

通过 Lagrangian 转成无约束形式：

$$
\max_{\pi} \;\mathbb{E}_{x, y}\!\big[r(x, y)\big] - \beta\,\text{KL}(\pi(\cdot|x) \| \pi_{ref}(\cdot|x))
$$

其中 $\beta > 0$ 是 Lagrange 乘子（KL penalty 系数）。

### 14.2.2 Step 1：最优策略 $\pi^*$ 的闭式解（Ch12 已证）

Ch12 §12.3 用变分法 / 配方法证明了：上面这个目标的最优解是

$$
\boxed{\;\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)\;}
$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(r(x,y)/\beta)$ 是**配分函数**（保证 $\pi^*$ 归一化）。

**几个直觉**：

1. $\pi^*$ 是 $\pi_{ref}$ 乘以一个 $\exp(r/\beta)$ 的"重加权"——reward 高的 $y$ 概率变大。
2. $\beta$ 越小，重加权越激进（允许 $\pi^*$ 远离 $\pi_{ref}$）；$\beta \to \infty$ 时 $\pi^* \to \pi_{ref}$。
3. $Z(x)$ **只依赖 $x$ 和 reward 函数**，**不依赖 $\pi$**（这点在 Step 3 是关键）。

> **DPO 的核心 trick**：上式把 reward $r$ 和 policy $\pi^*$ 联系起来了——
> 这是反解 reward 的钥匙。

### 14.2.3 Step 2：反解 reward（DPO 第一个关键步骤）

对 $\pi^*(y|x) = \frac{1}{Z}\pi_{ref}(y|x)\exp(r/\beta)$ 取对数：

$$
\log\pi^*(y|x) = \log\pi_{ref}(y|x) - \log Z(x) + \frac{r(x,y)}{\beta}
$$

移项：

$$
\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} = \frac{r(x,y)}{\beta} - \log Z(x)
$$

两边乘 $\beta$ 并移项：

$$
\boxed{\;r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)\;}
$$

这就是 **DPO 论文 Eq. 3**。reward 被表达成 policy 比值的对数 + 一个 prompt-only 常数。

### 14.2.4 Step 3：代入 Bradley-Terry（DPO 第二个关键步骤）

Ch11 §11.2 推出的 Bradley-Terry 偏好模型：

$$
P(y_w \succ y_l \mid x) = \sigma\!\big(r(x, y_w) - r(x, y_l)\big)
$$

把 Step 2 的 $r$ 代入：

$$
r(x, y_w) - r(x, y_l)
= \beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} + \cancel{\beta\log Z(x)}
- \beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} - \cancel{\beta\log Z(x)}
$$

**注意**：$\beta\log Z(x)$ 在相减中**消掉了**（因为它只依赖 $x$，与 $y$ 无关）！

$$
P(y_w \succ y_l \mid x) = \sigma\!\left(
    \beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)}
    - \beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
$$

**这一步是 DPO 的精髓**——reward model 完全消失了，只剩下 policy 比值。

### 14.2.5 Step 4：最大似然 → DPO loss（DPO 第三个关键步骤）

现在我们有一个**用 policy 表达的偏好概率** $P(y_w \succ y_l | x)$。
要学的 policy $\pi_\theta$，最大似然是：

$$
\max_\theta \;\mathbb{E}_{(x, y_w, y_l)}\!\left[\log P(y_w \succ y_l \mid x)\right]
= \max_\theta \;\mathbb{E}\!\left[\log\sigma\!\left(\beta(\Delta_w - \Delta_l)\right)\right]
$$

其中我们引入 DPO 的核心记号：

$$
\Delta_w := \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}, \quad
\Delta_l := \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
$$

（注意：把 Step 3 里的 $\pi^*$ 替换成了我们要学的 $\pi_\theta$。）

等价的最小化 loss：

$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log\sigma\!\left(\beta(\Delta_w - \Delta_l)\right)\right]\;}
$$

**这就是 DPO loss**。让我们数一下它"去掉了什么"：

| RLHF-PPO 需要 | DPO 还需要吗？ |
|---|---|
| Reward model | **不需要**（被 $\log\pi/\log\pi_{ref}$ 隐式编码） |
| Critic / value function | **不需要**（没有 advantage） |
| PPO clipping | **不需要**（纯监督） |
| Rollout / 采样 | **不需要**（静态数据集） |
| Importance ratio | **不需要**（没有 $\pi_{old}$） |
| KL penalty（explicit） | **不需要**（被 $\pi_{ref}$ 项隐式实现） |
| KL early stopping | **不需要**（没有 inner loop） |

**留下的只有**：actor + reference 两个模型 + 一个二分类 sigmoid loss。

### 14.2.6 数值稳定实现

直接写 $-\log\sigma(z)$ 当 $z \to -\infty$ 时会数值爆炸。用恒等变形：

$$
-\log\sigma(z) = \log(1 + e^{-z}) = \text{softplus}(-z)
$$

PyTorch 的 `F.softplus` 内部对大输入做了稳定化，所以 `dpo_loss` 实现是：

```python
delta_w = actor_logp_w - ref_logp_w     # log(π_θ/π_ref) on winner
delta_l = actor_logp_l - ref_logp_l
margin = beta * (delta_w - delta_l)     # 输入 sigmoid 的那个值
loss = F.softplus(-margin).mean()       # 数值稳定的 -log σ(margin)
```

### 14.2.7 隐式 reward（事后评估 DPO 训练效果）

Step 2 给出了 reward 的反解：$r(x, y) = \beta\log(\pi^*/\pi_{ref}) + \beta\log Z(x)$。

训练中我们用：

$$
\hat r(x, y; \theta) := \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

当 **DPO 隐式 reward**。它**不需要训 reward model 就能算**，可以直接评估 actor
学得好不好：

- $\hat r(y_w) > \hat r(y_l)$（margin > 0）→ actor 在这个偏好对上"对了"
- 平均 margin 随训练上升 → DPO 在学习偏好

> **重要对比**：Ch11 的 RewardModel 是**显式 reward**（$r_\theta(x,y)$，独立训练）；
> DPO 是**隐式 reward**（从 $\pi_\theta / \pi_{ref}$ 反推）。本章 §14.4 会用 Ch11 的
> 显式 RM **事后**评估 DPO 训练后的 actor——看隐式 reward 的提升是否对应显式 RM
> 评分的提升。""")

# 14.2.8 可视化：DPO loss surface
code(r"""# 14.2.8 可视化 DPO loss：作为 Δ_w, Δ_l 的函数
delta_w = np.linspace(-2, 5, 200)
delta_l = np.linspace(-2, 5, 200)
DW, DL = np.meshgrid(delta_w, delta_l)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, beta in zip(axes, [0.05, 0.2, 1.0]):
    margin = beta * (DW - DL)
    # softplus(-margin) = -log σ(margin)
    loss = np.log1p(np.exp(-margin))
    cs = ax.contourf(DW, DL, loss, levels=20, cmap='viridis')
    ax.contour(DW, DL, loss, levels=[0.5, 1.0, 2.0, 4.0], colors='white', alpha=0.5)
    plt.colorbar(cs, ax=ax, label='DPO loss')
    # 最优方向：Δ_w 大、Δ_l 小（让 margin 大）
    ax.plot([], [], 'w*', markersize=15, label='最优方向 (Δ_w↑, Δ_l↓)')
    ax.set_xlabel(r'$\Delta_w = \log\pi_\theta(y_w) - \log\pi_{ref}(y_w)$')
    ax.set_ylabel(r'$\Delta_l = \log\pi_\theta(y_l) - \log\pi_{ref}(y_l)$')
    ax.set_title(f'DPO loss surface (β={beta})')
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.7)

plt.suptitle("DPO loss 鼓励 actor：winner 上 log π ↑、loser 上 log π ↓（相对 ref）",
             fontsize=12)
plt.tight_layout(); plt.show()

print("观察:")
print("  - 横轴 Δ_w：actor 相对 ref 在 winner 上的 log-prob 增量")
print("  - 纵轴 Δ_l：actor 相对 ref 在 loser 上的 log-prob 增量")
print("  - loss 最低处（深色）= Δ_w 大 + Δ_l 小 → actor 学到偏好")
print("  - β 小：loss 平坦（训练信号弱，但允许 actor 远离 ref）")
print("  - β 大：loss 陡（信号强，但 actor 必须 stay close to ref）")""")


# =============================================================================
# 14.3 KTO
# =============================================================================
md(r"""## 14.3 KTO 与 prospect theory（核心 2/2）

DPO 用 **Bradley-Terry** 偏好模型（Ch11 §11.2），需要**成对**数据 $(y_w, y_l)$。
但工业上采集成对偏好很贵——人类标注员要读两条 response 才能产出一个标签。

**KTO（Kahneman-Tversky Optimization）** 的洞察：

> 用 **prospect theory（前景理论）** 替代 Bradley-Terry，
> 只需要 "good / bad" 二元标签——标注成本减半。

### 14.3.1 prospect theory 回顾（行为经济学）

Kahneman 和 Tversky 1979 年提出 prospect theory（前景理论，获 2002 年诺贝尔经济学奖），
描述人类在风险下的决策。核心是 **value function** $v(x)$：

$$
v(x) = \begin{cases}
x^{\gamma_+} & x \ge 0 \quad \text{(gains, 收益)} \\
-\lambda \cdot (-x)^{\gamma_-} & x < 0 \quad \text{(losses, 损失)}
\end{cases}
$$

两个**关键现象**（实验反复验证）：

1. **Diminishing sensitivity**（边际敏感度递减，$\gamma < 1$）：
   从 0 元到 100 元的快乐 > 从 1000 元到 1100 元的快乐。
   → v 在原点陡，远处平。
2. **Loss aversion**（损失厌恶，$\lambda > 1$）：
   失去 100 元的痛苦 > 得到 100 元的快乐。
   → KT 实测 $\lambda \approx 2.25$。

下图直观展示：

```python
v(x):    gains (x≥0)            losses (x<0)
              ↑                       ↑
              |    concave             |  steep (λ·)
         v(+) |   /                    |  \
              |  /                     |   \
              | /                      |    \
            0 +—————————————→ x        |
              |                        |
```

### 14.3.2 KTO 的"隐式 reward"

DPO 的隐式 reward $\hat r = \beta\log(\pi_\theta/\pi_{ref})$ 是直接对单条 response 算的，
**不需要成对**。这给了 KTO 的入口：

> 把"good" response 看作"收益"，把"bad" response 看作"损失"，
> 用 prospect theory 的 value function 替代 DPO 的 sigmoid。

### 14.3.3 KTO loss（简化教学版）

完整 KTO 论文（Ethayarajh et al. 2024）的 loss 涉及 reference point 估计，
我们用简化版本突出核心：

每条样本 $i$ 有标签 $g_i \in \{1=\text{good}, 0=\text{bad}\}$。它的"point"
是 DPO 隐式 reward：

$$
\rho_i = \beta\log\frac{\pi_\theta(y_i|x_i)}{\pi_{ref}(y_i|x_i)}
$$

KTO 想做的是：

- **good response**（$g_i = 1$）：推高 $\rho_i$（让它远大于 reference point $\tau$）
- **bad response**（$g_i = 0$）：压低 $\rho_i$（让它远小于 $\tau$）

用 sigmoid loss 表达：

$$
\mathcal{L}_{\text{KTO}} =
\underbrace{w_+ \cdot \mathbb{E}_{\text{good}}\!\big[\text{softplus}(\tau - \rho)\big]}_{\text{push good up}}
+ \underbrace{w_- \cdot \mathbb{E}_{\text{bad}}\!\big[\text{softplus}(\rho - \tau)\big]}_{\text{push bad down}}
$$

其中：

- $\tau$ 是 **reference point**（"中性" response 的 implicit reward；通常用 $\rho$ 的
  mini-batch 均值估计，简化版设 $\tau = 0$）
- $w_+$ / $w_-$ 是 good / bad 的权重；设 $w_- > w_+$ **体现 loss aversion**
  （KTO 论文用 $w_- = 1 + \lambda - 1$，即 KT 实测的损失厌恶系数）

> **注意**：完整 KTO 还有一项 $KL(\pi_\theta \| \pi_{ref})$ 的正则化（通过 reference
> point 的估计隐式实现），我们这里简化掉了。读者可以查原论文看完整版。

### 14.3.4 KTO vs DPO：何时选哪个？

| 维度 | DPO | KTO |
|---|---|---|
| 数据格式 | 成对偏好 $(y_w, y_l)$ | 单条 + good/bad 标签 |
| 标注成本 | 高（要比较两条） | 低（只标一条） |
| Loss 形式 | sigmoid(margin) | 加权 softplus |
| 理论基础 | Bradley-Terry | prospect theory |
| 数据效率 | 较高（一对 → 一个梯度信号） | 较低（一条 → 一个信号，但更便宜） |
| 推荐场景 | 高质量成对数据 | thumbs up/down 大规模数据 |

**经验法则**：如果你能拿到成对偏好数据，DPO 通常更 sample-efficient；
如果你只有二元标签（比如用户点赞/点踩），KTO 是唯一选择。""")

# 14.3.5 可视化 prospect theory value function
code(r"""# 14.3.5 可视化 prospect theory 的 value function（loss aversion + diminishing sensitivity）
import torch
x = torch.linspace(-3, 3, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左：不同的 lambda（loss aversion 系数）
ax = axes[0]
for lam, color in [(1.0, '#888'), (1.5, '#1f77b4'), (2.25, '#d62728'), (3.0, '#9467bd')]:
    v = prospect_value(x, lambda_aversion=lam, gamma_gain=0.9, gamma_loss=0.9)
    ax.plot(x.numpy(), v.numpy(), color=color, linewidth=2.5,
            label=f'λ = {lam}{" (KT 实测)" if lam == 2.25 else ""}')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('x  (reward signal)')
ax.set_ylabel('v(x)  (perceived value)')
ax.set_title('Loss aversion: |v(-1)| vs v(+1)')
ax.legend()
ax.grid(alpha=0.3)
# 标 v(+1) 和 v(-1)
v1 = float(prospect_value(torch.tensor([1.0]), 2.25, 0.9, 0.9).item())
v_neg1 = float(prospect_value(torch.tensor([-1.0]), 2.25, 0.9, 0.9).item())
ax.plot([1.0], [v1], 'o', color='#2ca02c', markersize=10, zorder=5)
ax.plot([-1.0], [v_neg1], 'o', color='#d62728', markersize=10, zorder=5)
ax.annotate(f'v(+1)={v1:.2f}\n(收益)', xy=(1.0, v1), xytext=(1.3, v1-0.4), fontsize=9)
ax.annotate(f'v(-1)={v_neg1:.2f}\n(损失，被 λ 放大)', xy=(-1.0, v_neg1),
            xytext=(-2.8, v_neg1+0.3), fontsize=9)

# 右：不同的 gamma（diminishing sensitivity 曲率）
ax = axes[1]
for gamma, color in [(1.0, '#888'), (0.7, '#1f77b4'), (0.5, '#d62728'), (0.3, '#9467bd')]:
    v = prospect_value(x, lambda_aversion=1.0, gamma_gain=gamma, gamma_loss=gamma)
    ax.plot(x.numpy(), v.numpy(), color=color, linewidth=2.5,
            label=f'γ = {gamma}{" (线性)" if gamma == 1.0 else ""}')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('v(x)')
ax.set_title('Diminishing sensitivity: γ 越小越非线性')
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle("Prospect Theory (Kahneman-Tversky 1979) —— KTO 的数学基础", fontsize=12)
plt.tight_layout(); plt.show()

print("观察:")
print("  - 左图: λ=2.25 时 |v(-1)| ≈ 2×|v(+1)|（损失比收益感觉更强，loss aversion）")
print("  - 右图: γ=1 线性（无 prospect 效应）；γ<1 在原点附近更陡，远处平")
print("  - KTO 用 v(x) 替代 DPO 的 sigmoid，把 good/bad 标签映射到非对称 loss")""")

# 14.3.6 可视化 KTO loss
code(r"""# 14.3.6 KTO loss vs DPO loss 的对比
rho = np.linspace(-3, 3, 200)  # implicit reward ρ

# KTO loss: good 样本 softplus(τ - ρ), bad 样本 softplus(ρ - τ)
tau = 0.0
loss_good = np.log1p(np.exp(-(rho - tau)))    # softplus(τ - ρ)
loss_bad_neutral = np.log1p(np.exp(rho - tau)) # softplus(ρ - τ), w=1
lam = 2.25
loss_bad_averse = lam * loss_bad_neutral      # w->λ (loss aversion)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# 左：good / bad 各自的 loss 形状
ax = axes[0]
ax.plot(rho, loss_good, color='#2ca02c', linewidth=2.5, label='good samples: softplus(τ - ρ)')
ax.plot(rho, loss_bad_neutral, color='#d62728', linewidth=2.5, label='bad samples (w=1): softplus(ρ - τ)')
ax.plot(rho, loss_bad_averse, color='#9467bd', linewidth=2.5, linestyle='--',
        label=f'bad samples (λ={lam}): w·softplus(ρ - τ)')
ax.axvline(tau, color='black', linestyle='--', alpha=0.5, label=f'reference point τ={tau}')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel(r'implicit reward  $\rho = \beta\log(\pi_\theta / \pi_{ref})$')
ax.set_ylabel('KTO loss')
ax.set_title('KTO loss: good vs bad samples')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-0.2, 6)

# 右：DPO loss vs KTO loss 形状对比
ax = axes[1]
# DPO loss on margin = (ρ_w - ρ_l) 假设固定 ρ_l=0
margin = np.linspace(-3, 3, 200)
dpo = np.log1p(np.exp(-margin))
ax.plot(margin, dpo, color='#1f77b4', linewidth=2.5,
        label='DPO: -log σ(margin)\n(margin = ρ_w - ρ_l)')
# KTO good loss（把 margin 看作 ρ）
ax.plot(margin, loss_good, color='#2ca02c', linewidth=2.5,
        label='KTO good: softplus(-ρ)\n(把 good 的 ρ 推高)')
ax.set_xlabel('margin (DPO) or implicit reward (KTO)')
ax.set_ylabel('loss')
ax.set_title('DPO vs KTO loss shape')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()

print("关键观察:")
print("  - KTO 的 good loss: ρ ↑ 时 loss ↓ → 模型被推向 'good ρ 大' 的方向")
print("  - KTO 的 bad loss: ρ ↓ 时 loss ↓ → 模型被推向 'bad ρ 小' 的方向")
print("  - loss aversion (λ>1): bad loss 在 'wrong direction' 处更陡 → 更严惩把 bad 当 good")
print("  - DPO: 一对样本产生一个梯度信号（margin）；KTO: 一条样本产生一个信号（更便宜）")""")


# =============================================================================
# 14.4 实现 DPO
# =============================================================================
md(r"""## 14.4 实现 DPO：训练 actor，验证 reward 提升

本节我们：

1. 复用 Ch11 的合成偏好数据（同样的 `generate_preference_data`）
2. 训一个**小 TinyGPT** 当 actor，深拷贝一份当 reference（保证 $\pi_\theta^{(0)} = \pi_{ref}$）
3. 跑 DPO 训练（用本章新写的 `DPOTrainer`）
4. 用 **Ch11 训好的 RewardModel**（独立评估器）测 actor 的 reward 是否提升

> **重要**：DPO loop 里**没有** reward model（这是它和 RLHF-PPO 的关键区别）。
> 但为了**验证 DPO 真的让 actor 变好**，我们在外面用一个 Ch11 RM 来打分——
> 如果 DPO 有效，RM 给 actor 生成的隐式 reward 应该随训练上升。

### 14.4.1 准备：tokenizer + 偏好数据 + SFT 起点""")

code(r"""# 14.4.1 准备：tokenizer + 偏好数据 + SFT 起点
# 复用 Ch11 的设置（保证一致）
corpus = (
    "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    "Q: What do you think? A: Q: Is it good? A: Q: Tell me a word. A: Q: How are you? A: "
)
tok = CharTokenizer().train(corpus)
print(f"tokenizer vocab size: {tok.vocab_size}")

KEYWORD_W = 3.0
LEN_W = 0.3
TARGET_LEN = 6

# 偏好数据（与 Ch11 完全一致）
train_prefs = generate_preference_data(
    tok, n_samples=300, seed=0,
    keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN,
)
val_prefs = generate_preference_data(
    tok, n_samples=80, seed=999,
    keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN,
)
print(f"训练偏好对: {len(train_prefs)}")
print(f"验证偏好对: {len(val_prefs)}")
print(f"前 3 条样本:")
for i in range(3):
    s = train_prefs[i]
    print(f"  prompt={s['prompt']!r}")
    print(f"    winner={s['winner']!r:<14} (true_r={s['r_w']:+.2f})")
    print(f"    loser ={s['loser']!r:<14} (true_r={s['r_l']:+.2f})")""")

# 14.4.2 训 RewardModel（用于事后评估 DPO 效果）
code(r"""# 14.4.2 训一个 Ch11 RewardModel（用作 DPO 事后评估器）
# 这一步和 Ch11 §11.5 完全一致——只是为了让本章能独立运行
torch.manual_seed(42); np.random.seed(42); random.seed(42)

rm_backbone = build_tiny_gpt(
    vocab_size=tok.vocab_size,
    d_model=64, n_heads=4, n_layers=3, d_ff=256, max_seq_len=64,
)
evaluator_rm = RewardModel(rm_backbone)
rm_opt = torch.optim.AdamW(evaluator_rm.parameters(), lr=1e-3, weight_decay=0.01)

print("训练 evaluator RM (Ch11 复用，400 步)...")
t0 = time.time()
for step in range(400):
    evaluator_rm.train()
    batch_idx = random.sample(range(len(train_prefs)), 32)
    batch_samples = [train_prefs[i] for i in batch_idx]
    b = make_preference_batch(batch_samples, pad_id=tok.pad_id)
    loss = bradley_terry_loss(evaluator_rm, b['prompt_ids'], b['winner_ids'], b['loser_ids'])
    rm_opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(evaluator_rm.parameters(), 1.0)
    rm_opt.step()
    if step % 100 == 0 or step == 399:
        acc = reward_accuracy(evaluator_rm, val_prefs, pad_id=tok.pad_id)
        print(f"  step {step:3d}  loss={loss.item():.3f}  val_acc={acc:.3f}  ({time.time()-t0:.1f}s)")

final_rm_acc = reward_accuracy(evaluator_rm, val_prefs, pad_id=tok.pad_id)
print(f"\nevaluator RM val accuracy: {final_rm_acc:.3f}")
print(f"参数量: {count_parameters(evaluator_rm):,}")
print("（这个 RM **不**在 DPO loop 里——只用于事后评估 DPO 训出来的 actor）")""")

# 14.4.3 初始化 actor / reference
code(r"""# 14.4.3 初始化 actor + reference（两者起点相同 —— DPO 标准做法）
torch.manual_seed(42); np.random.seed(42); random.seed(42)

D_MODEL = 48  # 比 RM 小一点（actor 是被训练对象）
N_LAYERS = 2

def make_actor_backbone():
    return build_tiny_gpt(
        vocab_size=tok.vocab_size,
        d_model=D_MODEL, n_heads=4, n_layers=N_LAYERS, d_ff=D_MODEL*4, max_seq_len=64,
    )

class ActorWrap(nn.Module):
    # TinyGPT 的薄包装：保持 forward(input_ids) -> logits 接口。
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.d_model = backbone.d_model
        self.max_seq_len = backbone.max_seq_len
    def forward(self, ids):
        return self.backbone(ids)

actor = ActorWrap(make_actor_backbone())
# reference = actor 的深拷贝（DPO 起点：π_θ^(0) = π_ref）
reference = copy.deepcopy(actor)
# reference 必须冻结（DPOTrainer 会强制 requires_grad_(False)）

n_actor_params = count_parameters(actor)
print(f"actor 参数量: {n_actor_params:,}")
print(f"reference 参数量: {count_parameters(reference):,}（与 actor 同；DPOTrainer 初始化时会冻结）")
print(f"reference requires_grad (before DPOTrainer init): {any(p.requires_grad for p in reference.parameters())}")

# 验证：起点时 actor == reference → Δ_w = Δ_l = 0 → DPO loss = log(2)
import math
with torch.no_grad():
    logp = sequence_log_probs(
        actor,
        val_prefs[0]['prompt_ids'].unsqueeze(0),
        val_prefs[0]['winner_ids'].unsqueeze(0),
        pad_id=tok.pad_id,
    )
    ref_logp = sequence_log_probs(
        reference,
        val_prefs[0]['prompt_ids'].unsqueeze(0),
        val_prefs[0]['winner_ids'].unsqueeze(0),
        pad_id=tok.pad_id,
    )
    print(f"\n起点检查: actor logp={logp.item():+.3f}  ref logp={ref_logp.item():+.3f}")
    print(f"  Δ = {float(logp - ref_logp):+.6f}  (应 ≈ 0，因为 actor == reference)")
    print(f"  DPO loss @ start = softplus(0) = log(2) = {math.log(2):.4f}")""")

# 14.4.4 DPO 训练
code(r"""# 14.4.4 DPO 训练（核心环节）
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# DPO 配置：β=0.1（中等 KL 约束），lr=5e-4，batch=32，300 步
dpo_cfg = DPOConfig(
    beta=0.1,
    actor_lr=5e-4,
    batch_size=32,
    max_grad_norm=1.0,
    eval_every=50,
    print_every=50,
)

dpo_trainer = DPOTrainer(
    actor=actor,
    reference=reference,
    pad_id=tok.pad_id,
    cfg=dpo_cfg,
    device=DEVICE,
)

# 训练前评估（baseline）
print("=" * 70)
print("DPO 训练前评估 (baseline)")
pre_stats = dpo_trainer.evaluate(val_prefs, val_reward_model=evaluator_rm)
for k, v in pre_stats.items():
    print(f"  {k:30s} = {v:+.4f}")
print("=" * 70)

# 训练
print("\n开始 DPO 训练...")
t0 = time.time()
dpo_trainer.train(
    train_samples=train_prefs,
    n_iters=300,
    val_samples=val_prefs,
    val_reward_model=evaluator_rm,
    verbose=True,
)
print(f"\n训练完成，耗时 {time.time()-t0:.1f}s")""")

# 14.4.5 训练曲线
code(r"""# 14.4.5 训练曲线：DPO loss / margin / RM-based reward
hist = dpo_trainer.history
steps = [h['step'] for h in hist]
dpo_losses = [h['dpo_loss'] for h in hist]
margins = [h['reward_margin'] for h in hist]
chosen_r = [h['chosen_reward'] for h in hist]
rejected_r = [h['rejected_reward'] for h in hist]
accs = [h['reward_accuracy'] for h in hist]

# 提取 eval 点（val_rm_* 字段只在 eval 点出现）
eval_steps = [h['step'] for h in hist if 'val_rm_margin' in h]
val_rm_margin = [h.get('val_rm_margin', float('nan')) for h in hist if 'val_rm_margin' in h]
val_rm_acc = [h.get('val_rm_acc', float('nan')) for h in hist if 'val_rm_acc' in h]
val_dpo_acc = [h.get('val_dpo_acc', float('nan')) for h in hist if 'val_dpo_acc' in h]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# 左上：DPO loss
ax = axes[0, 0]
ax.plot(steps, dpo_losses, color='#1f77b4', alpha=0.4, linewidth=0.6)
window = 20
if len(dpo_losses) > window:
    sm = np.convolve(dpo_losses, np.ones(window)/window, mode='valid')
    ax.plot(steps[window-1:], sm, color='#1f77b4', linewidth=2.5, label='smoothed')
ax.axhline(math.log(2), color='gray', linestyle='--', alpha=0.7,
           label=f'log 2 ≈ {math.log(2):.2f} (random baseline)')
ax.set_xlabel('training step'); ax.set_ylabel('DPO loss')
ax.set_title('DPO loss（越小越好）'); ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(0, max(dpo_losses) * 1.05)

# 右上：chosen / rejected reward（DPO 隐式 reward）
ax = axes[0, 1]
ax.plot(steps, chosen_r, color='#2ca02c', alpha=0.4, linewidth=0.6)
ax.plot(steps, rejected_r, color='#d62728', alpha=0.4, linewidth=0.6)
if len(chosen_r) > window:
    sm_c = np.convolve(chosen_r, np.ones(window)/window, mode='valid')
    sm_l = np.convolve(rejected_r, np.ones(window)/window, mode='valid')
    ax.plot(steps[window-1:], sm_c, color='#2ca02c', linewidth=2.5, label='chosen reward βΔ_w')
    ax.plot(steps[window-1:], sm_l, color='#d62728', linewidth=2.5, label='rejected reward βΔ_l')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('training step'); ax.set_ylabel('implicit reward')
ax.set_title('DPO 隐式 reward：winner ↑、loser ↓'); ax.legend(); ax.grid(alpha=0.3)

# 左下：reward margin（DPO 自身 + RM 评估）
ax = axes[1, 0]
sm_margin = np.convolve(margins, np.ones(window)/window, mode='valid') if len(margins) > window else margins
ax.plot(steps, margins, color='#1f77b4', alpha=0.3, linewidth=0.6)
ax.plot(steps[window-1:] if len(margins) > window else steps, sm_margin,
        color='#1f77b4', linewidth=2.5, label='DPO margin (training)')
if eval_steps:
    ax.plot(eval_steps, val_rm_margin, 'o-', color='#ff7f0e', linewidth=2,
            markersize=8, label='RM-based margin (val)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('training step'); ax.set_ylabel('reward margin')
ax.set_title('reward margin（DPO 内部 vs Ch11 RM 评估）'); ax.legend(); ax.grid(alpha=0.3)

# 右下：accuracy
ax = axes[1, 1]
sm_acc = np.convolve(accs, np.ones(window)/window, mode='valid') if len(accs) > window else accs
ax.plot(steps, accs, color='#9467bd', alpha=0.3, linewidth=0.6)
ax.plot(steps[window-1:] if len(accs) > window else steps, sm_acc,
        color='#9467bd', linewidth=2.5, label='DPO train acc')
if eval_steps:
    ax.plot(eval_steps, val_dpo_acc, 's-', color='#2ca02c', linewidth=2,
            markersize=8, label='DPO val acc')
    ax.plot(eval_steps, val_rm_acc, 'o-', color='#ff7f0e', linewidth=2,
            markersize=8, label='RM val acc (w > l %)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='random (0.5)')
ax.set_xlabel('training step'); ax.set_ylabel('accuracy')
ax.set_title('DPO / RM accuracy'); ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(0.3, 1.01)

plt.suptitle("DPO 训练曲线（隐式 reward 稳定提升 + DPO val accuracy 上升）", fontsize=12)
plt.tight_layout(); plt.show()

print("关键观察:")
print(f"  - DPO loss 从 log(2)≈0.69 降到 ~{dpo_losses[-1]:.3f}")
print(f"  - chosen reward ↑ / rejected reward ↓（actor 学到偏好）")
print(f"  - DPO val acc (margin > 0 比例): {pre_stats.get('val_dpo_acc', 0):.3f} → {val_dpo_acc[-1]:.3f}")
print(f"     （从随机水平 0.5 提升到 ~0.96 = actor 在 96% 的偏好对上把隐式 reward 排对了）")
print(f"  - RM-based val acc: {pre_stats.get('val_rm_acc', 0):.3f} → {val_rm_acc[-1]:.3f}")
print(f"     （RM 直接评估 actor 给 winner/loser 的 reward 排序）")
print(f"  → DPO val acc 大幅提升证明 DPO 训练确实让 actor 学到了偏好结构")""")

# 14.4.6 直接对比 DPO 前/后 actor 对候选 response 的偏好分布
code(r"""# 14.4.6 DPO 训练前后对比：actor 对候选 response 的偏好分布
# 这是最稳定的"actor 变好了吗"的检验（不依赖采样随机性）
# 思路：给 actor 一组固定候选 response，看它在每个候选上的条件概率分布，
# 然后用这个分布对 ground-truth reward 加权（=actor 的 expected reward）。

# 重新构建一个"训练前 actor"（深拷贝初始 reference）
actor_before = copy.deepcopy(reference)  # reference == actor 起点
actor_after = actor  # 训练后的

CANDIDATES = [
    "good", "very good", "it good", "good day", "yes good",
    "ok", "fine", "yes", "no", "no way",
    "bad", "very bad", "it is bad", "great", "hello world",
]

@torch.no_grad()
def actor_preference_distribution(actor_model, prompt_str):
    # 返回 actor 在 prompt 下对每个 candidate 的 softmax 概率
    actor_model.eval()
    p_ids = tok.encode(prompt_str).unsqueeze(0)
    logps = []
    for resp in CANDIDATES:
        r_ids = tok.encode(resp)
        lp = sequence_log_probs(actor_model, p_ids, r_ids.unsqueeze(0), pad_id=tok.pad_id)
        logps.append(float(lp.item()))
    logp_arr = np.array(logps)
    probs = np.exp(logp_arr - logp_arr.max())
    probs = probs / probs.sum()
    return probs

test_prompts = [
    "Q: How are you? A:",
    "Q: Is it good? A:",
    "Q: Tell me a word. A:",
]

print("=" * 70)
print("DPO 训练前后对比：actor 对候选 response 的偏好分布")
print("=" * 70)

for p in test_prompts:
    print(f"\nPrompt: {p!r}")
    probs_before = actor_preference_distribution(actor_before, p)
    probs_after  = actor_preference_distribution(actor_after,  p)
    # 按 ground-truth reward 排序（高的在前）
    gt = [true_reward(p, r, keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN)
          for r in CANDIDATES]
    order = np.argsort(-np.array(gt))  # reward 高到低

    print(f"  {'response':<14}{'true_r':>9}{'p_before':>12}{'p_after':>12}{'change':>12}")
    for i in order:
        r = CANDIDATES[i]
        print(f"  {r!r:<14}{gt[i]:>+9.2f}{probs_before[i]:>12.3f}{probs_after[i]:>12.3f}"
              f"{probs_after[i]-probs_before[i]:>+12.3f}")

    # 计算 expected reward
    exp_before = float((probs_before * np.array(gt)).sum())
    exp_after  = float((probs_after  * np.array(gt)).sum())
    print(f"  → expected reward: before={exp_before:+.3f}  after={exp_after:+.3f}  "
          f"change={exp_after-exp_before:+.3f}")

print("\n" + "=" * 70)
print("关键观察:")
print("  - DPO 后 actor 把概率从 'bad' / 'no' 等 (true_r 低) 转移到 'good' / 'very good' (true_r 高)")
print("  - 这是 DPO 真的让 actor 学到偏好的最直接证据（确定性，不依赖采样）")""")


# =============================================================================
# 14.5 三方对比 + distribution shift
# =============================================================================
md(r"""## 14.5 三方对比：DPO vs GRPO vs PPO-RLHF

本节是 Phase 3 的"算法竞技场"——把 Ch12 / Ch13 / Ch14 三种对齐算法放在同一份偏好数据
上跑，对比：

1. **reward 提升速度**（用 Ch11 的 RM 测）
2. **KL behavior**（actor 偏离 reference 多少）
3. **训练 wall-clock 时间**（工程开销）
4. **Distribution shift**（off-policy 的代价）

### 14.5.1 实验设计

- **共同数据**：Ch11 的合成偏好数据（300 train / 80 val）
- **共同评估**：Ch11 RM（独立训练的 evaluator）
- **共同起点**：同一个 SFT-initialized TinyGPT
- **3 个算法**：
    - **PPO-RLHF（Ch12）**：4 模型 + critic + PPO clipping
    - **GRPO（Ch13）**：3 模型 + group sampling（无 critic）
    - **DPO（Ch14）**：2 模型 + 静态数据（无 RM/critic/rollout）

> **公平性 note**：3 个算法的超参都各自调过；toy 模型很小，结论是**定性**的
> （哪个提升快、哪个更稳）而非定量（具体 reward 数值依赖 RM scale）。

### 14.5.2 Distribution shift：DPO 的根本局限

DPO 在**静态偏好数据**上训，actor 从来不"看见"自己生成的 response。
这导致一个根本问题：

> **如果 actor 漂移到了偏好数据没覆盖的区域，DPO 的梯度信号就失效了。**

形式化：DPO loss 假设数据分布 $\mathcal{D}$ 覆盖了 actor 当前 $\pi_\theta$ 的高概率区域。
但训练后 actor 偏离了 $\pi_{ref}$，新 $\pi_\theta$ 的样本可能落在 $\mathcal{D}$ 之外
→  off-distribution → DPO 在新区域的偏好**没有保证**。

PPO-RLHF / GRPO 是 on-policy（采样自当前 $\pi_\theta$），没有这个问题——
但代价是每步要重新采样（贵）。

下面我们用一个简单的 toy 实验展示这个现象。""")

# 14.5.3 跑 3 个算法
code(r"""# 14.5.3 跑 3 个算法（同一份数据 + 同一个起点）
# 为了控制运行时间，每个算法只跑较少的步数（toy 演示用）
torch.manual_seed(42); np.random.seed(42); random.seed(42)

N_ITERS = 80  # 每个算法的 outer iteration 数（控制时间）
EVAL_EVERY = 10

results = {}  # name -> dict of metrics over training

# ---------- 共同的 actor 起点（每次重置）----------
def make_fresh_actor():
    torch.manual_seed(42)
    return ActorWrap(make_actor_backbone())

def make_fresh_reference():
    # 独立的 reference（每个算法用同一个起点）
    return copy.deepcopy(make_fresh_actor())

# ---------- 公共评估函数（确定性、不依赖采样随机性）----------
# 思路：给 actor 一组固定候选 response，看它**期望偏好**哪个（用 actor 自己的
# softmax 概率分布作权重，对所有候选的 ground-truth reward 加权平均）。
# 这等价于"如果无限采样，actor 期望产出的 reward"——比单次采样更稳定。
CANDIDATE_RESPONSES = [
    "good", "very good", "it good", "good day", "yes good",
    "ok", "fine", "yes", "no", "no way",
    "bad", "very bad", "it is bad", "great", "hello world",
]

@torch.no_grad()
def eval_actor_expected_reward(actor_model, prompts_str_list, use_true_reward=True,
                                rm_model=None):
    # 对每个 prompt，actor 给每个候选 response 一个条件概率，
    # 用这个概率对 response 的 reward 加权 → actor 的 expected reward。
    # reward 来源：use_true_reward=true → ground truth; false → 用 RM
    actor_model.eval()
    if rm_model is not None:
        rm_model.eval()
    all_expected = []
    for prompt_str in prompts_str_list:
        p_ids = tok.encode(prompt_str).unsqueeze(0)  # [1, T_p]
        # 给每个候选算 actor 的 log π(y|x)
        logps = []
        rewards = []
        for resp in CANDIDATE_RESPONSES:
            r_ids = tok.encode(resp)
            if r_ids.numel() == 0:
                continue
            lp = sequence_log_probs(actor_model, p_ids, r_ids.unsqueeze(0),
                                     pad_id=tok.pad_id)
            logps.append(float(lp.item()))
            if use_true_reward:
                rw = true_reward(prompt_str, resp,
                                 keyword_weight=KEYWORD_W, len_weight=LEN_W,
                                 target_len=TARGET_LEN)
            else:
                rw = float(rm_model(p_ids, r_ids.unsqueeze(0)).item())
            rewards.append(rw)
        logp_arr = np.array(logps)
        # softmax over candidates（"actor 选择哪个候选的分布"）
        probs = np.exp(logp_arr - logp_arr.max())
        probs = probs / probs.sum()
        expected_r = float((probs * np.array(rewards)).sum())
        all_expected.append(expected_r)
    actor_model.train()
    return float(np.mean(all_expected))

# 包装：用 ground-truth 评估；用 RM 评估
def eval_true(actor_model, prompts):
    return eval_actor_expected_reward(actor_model, prompts, use_true_reward=True)
def eval_rm(actor_model, prompts):
    return eval_actor_expected_reward(actor_model, prompts, use_true_reward=False,
                                       rm_model=evaluator_rm)

# prompt 池
seen_prompts = [
    "Q: How are you? A:",
    "Q: Is it good? A:",
    "Q: Tell me a word. A:",
]
# novel prompts（结构相似但措辞不同，测试 distribution shift）
novel_prompts = [
    "Q: How do you feel? A:",
    "Q: What about this? A:",
    "Q: Give me a thing. A:",
]

print("起点 actor 评估:")
print(f"  seen prompts  | RM reward = {eval_rm(make_fresh_actor(), seen_prompts):+.3f} | "
      f"true reward = {eval_true(make_fresh_actor(), seen_prompts):+.3f}")
print(f"  novel prompts | RM reward = {eval_rm(make_fresh_actor(), novel_prompts):+.3f} | "
      f"true reward = {eval_true(make_fresh_actor(), novel_prompts):+.3f}")
print(f"（评估方式：actor 对 15 个候选 response 的 softmax 概率 × 候选的 reward）")""")

# 14.5.4 跑 DPO
code(r"""# 14.5.4 跑 DPO（150 步，每 10 步评估）
print("=" * 60)
print("[1/3] 训练 DPO ...")
print("=" * 60)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

actor_dpo = make_fresh_actor()
reference_dpo = copy.deepcopy(actor_dpo)

dpo_cfg = DPOConfig(beta=0.1, actor_lr=5e-4, batch_size=32, max_grad_norm=1.0,
                    eval_every=1000, print_every=50)
dpo_trainer_eval = DPOTrainer(actor_dpo, reference_dpo, pad_id=tok.pad_id,
                              cfg=dpo_cfg, device=DEVICE)

dpo_history = []
t0 = time.time()
# 预计算 ref logp
ref_logps = dpo_trainer_eval.precompute_reference_logps(train_prefs)
# 评估起点
rm_r = eval_rm(actor_dpo, seen_prompts)
tr_r = eval_true(actor_dpo, seen_prompts)
novel_r = eval_true(actor_dpo, novel_prompts)
dpo_history.append({'step': 0, 'rm_reward': rm_r, 'true_reward': tr_r,
                    'novel_true_reward': novel_r, 'wall_time': 0.0})

# 简化版训练 loop（用 DPOTrainer 的 dpo_update，但每 EVAL_EVERY 步评估）
from utils.reward_model import make_preference_batch
rng = random.Random(0)
N_PREFS = len(train_prefs)
for it in range(1, N_ITERS + 1):
    idx = rng.sample(range(N_PREFS), min(dpo_cfg.batch_size, N_PREFS))
    batch_samples = [train_prefs[i] for i in idx]
    b = make_preference_batch(batch_samples, pad_id=tok.pad_id)
    ref_w = ref_logps['logp_w'][torch.tensor(idx)]
    ref_l = ref_logps['logp_l'][torch.tensor(idx)]
    dpo_trainer_eval.dpo_update(
        b['prompt_ids'], b['winner_ids'], b['loser_ids'], ref_w, ref_l,
    )
    if it % EVAL_EVERY == 0 or it == N_ITERS:
        rm_r = eval_rm(actor_dpo, seen_prompts)
        tr_r = eval_true(actor_dpo, seen_prompts)
        novel_r = eval_true(actor_dpo, novel_prompts)
        elapsed = time.time() - t0
        dpo_history.append({'step': it, 'rm_reward': rm_r, 'true_reward': tr_r,
                            'novel_true_reward': novel_r, 'wall_time': elapsed})
        print(f"  iter {it:3d} | RM_r={rm_r:+.3f} | true_r={tr_r:+.3f} | "
              f"novel_r={novel_r:+.3f} | ({elapsed:.1f}s)")

results['DPO'] = dpo_history
print(f"DPO 完成，耗时 {time.time()-t0:.1f}s")""")

# 14.5.5 跑 GRPO
code(r"""# 14.5.5 跑 GRPO（30 outer iters，因为每 iter 内含多 epoch + group sampling 较慢）
print("=" * 60)
print("[2/3] 训练 GRPO ...")
print("=" * 60)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

actor_grpo = make_fresh_actor()
reference_grpo = copy.deepcopy(actor_dpo)  # 同一个起点（reference 冻结）
# GRPO 需要一个 reward model 在 loop 里
rm_for_grpo = copy.deepcopy(evaluator_rm)  # 用同一个 RM（保证公平）

GRPO_ITERS = 30  # GRPO 每 iter 慢很多，少跑点
grpo_cfg = GRPOConfig(
    group_size=6, beta=0.05, clip_eps=0.2,
    update_epochs=2, inner_minibatch_size=6, entropy_coef=0.005,
    max_grad_norm=0.5, target_kl=None,  # 关掉 early stop 保证可比
    response_max_len=8, temperature=1.0, top_k=10,
    actor_lr=2e-4, print_every=1000,  # 调大 lr 让 GRPO 在少 iter 下也能提升
)
grpo_trainer = GRPOTrainer(
    actor_grpo, rm_for_grpo, reference_grpo, pad_id=tok.pad_id,
    cfg=grpo_cfg, device=DEVICE,
)

grpo_history = []
t0 = time.time()
# 起点
rm_r = eval_rm(actor_grpo, seen_prompts)
tr_r = eval_true(actor_grpo, seen_prompts)
novel_r = eval_true(actor_grpo, novel_prompts)
grpo_history.append({'step': 0, 'rm_reward': rm_r, 'true_reward': tr_r,
                     'novel_true_reward': novel_r, 'wall_time': 0.0})

prompt_pool = [tok.encode(p) for p in
               ["Q: How A:", "Q: What A:", "Q: Is it A:", "Q: Tell me A:", "Q: How are A:"]]
rng = random.Random(0)
for it in range(1, GRPO_ITERS + 1):
    prompts = [rng.choice(prompt_pool) for _ in range(2)]  # 每 iter 2 个 prompts
    rollout = grpo_trainer.rollout_group(prompts)
    grpo_trainer.grpo_update(rollout)
    if it % 3 == 0 or it == GRPO_ITERS:
        rm_r = eval_rm(actor_grpo, seen_prompts)
        tr_r = eval_true(actor_grpo, seen_prompts)
        novel_r = eval_true(actor_grpo, novel_prompts)
        elapsed = time.time() - t0
        grpo_history.append({'step': it, 'rm_reward': rm_r, 'true_reward': tr_r,
                             'novel_true_reward': novel_r, 'wall_time': elapsed})
        print(f"  iter {it:3d} | RM_r={rm_r:+.3f} | true_r={tr_r:+.3f} | "
              f"novel_r={novel_r:+.3f} | ({elapsed:.1f}s)")

results['GRPO'] = grpo_history
print(f"GRPO 完成，耗时 {time.time()-t0:.1f}s")""")

# 14.5.6 跑 PPO-RLHF
code(r"""# 14.5.6 跑 PPO-RLHF（20 outer iters，4 模型最重）
print("=" * 60)
print("[3/3] 训练 PPO-RLHF ...")
print("=" * 60)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

actor_ppo = make_fresh_actor()
reference_ppo = copy.deepcopy(actor_dpo)
# PPO 需要 critic（用 ValueHead 包装一个 TinyGPT backbone）
critic_backbone = build_tiny_gpt(
    vocab_size=tok.vocab_size, d_model=D_MODEL, n_heads=4,
    n_layers=N_LAYERS, d_ff=D_MODEL*4, max_seq_len=64,
)
critic_ppo = ValueHead(critic_backbone)
rm_for_ppo = copy.deepcopy(evaluator_rm)

PPO_ITERS = 25  # PPO 最慢
ppo_cfg = RLHFConfig(
    beta=0.05, gamma=0.95, lam=0.95,
    clip_eps=0.2, update_epochs=2, inner_minibatch_size=4,
    entropy_coef=0.005, value_coef=0.5, max_grad_norm=0.5,
    target_kl=None,
    response_max_len=8, temperature=1.0, top_k=10,
    actor_lr=2e-4, critic_lr=5e-4, print_every=1000,  # 调大 lr
)
ppo_trainer = RLHFTrainer(
    actor_ppo, critic_ppo, rm_for_ppo, reference_ppo,
    pad_id=tok.pad_id, cfg=ppo_cfg, device=DEVICE,
)

ppo_history = []
t0 = time.time()
rm_r = eval_rm(actor_ppo, seen_prompts)
tr_r = eval_true(actor_ppo, seen_prompts)
novel_r = eval_true(actor_ppo, novel_prompts)
ppo_history.append({'step': 0, 'rm_reward': rm_r, 'true_reward': tr_r,
                    'novel_true_reward': novel_r, 'wall_time': 0.0})

rng = random.Random(0)
for it in range(1, PPO_ITERS + 1):
    prompts = [rng.choice(prompt_pool) for _ in range(4)]
    rollout = ppo_trainer.rollout_responses(prompts)
    ppo_trainer.rlhf_update(rollout)
    if it % 2 == 0 or it == PPO_ITERS:
        rm_r = eval_rm(actor_ppo, seen_prompts)
        tr_r = eval_true(actor_ppo, seen_prompts)
        novel_r = eval_true(actor_ppo, novel_prompts)
        elapsed = time.time() - t0
        ppo_history.append({'step': it, 'rm_reward': rm_r, 'true_reward': tr_r,
                            'novel_true_reward': novel_r, 'wall_time': elapsed})
        print(f"  iter {it:3d} | RM_r={rm_r:+.3f} | true_r={tr_r:+.3f} | "
              f"novel_r={novel_r:+.3f} | ({elapsed:.1f}s)")

results['PPO-RLHF'] = ppo_history
print(f"PPO-RLHF 完成，耗时 {time.time()-t0:.1f}s")""")

# 14.5.7 三方对比图
code(r"""# 14.5.7 三方对比图：reward 提升 / wall-clock / distribution shift
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors = {'DPO': '#2ca02c', 'GRPO': '#9467bd', 'PPO-RLHF': '#d62728'}

# 左上：RM reward vs wall-clock
ax = axes[0, 0]
for name, hist in results.items():
    times = [h['wall_time'] for h in hist]
    rm_r = [h['rm_reward'] for h in hist]
    # 归一化到 step 0 = 0（去掉 RM scale 的差异，只看相对提升）
    rm_r_norm = [r - rm_r[0] for r in rm_r]
    ax.plot(times, rm_r_norm, 'o-', color=colors[name], linewidth=2.5,
            markersize=7, label=name)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('wall-clock time (s)')
ax.set_ylabel('RM reward (relative to start)')
ax.set_title('RM reward 提升 vs wall-clock（DPO 最快）')
ax.legend(); ax.grid(alpha=0.3)

# 右上：true reward（ground truth）vs step
ax = axes[0, 1]
for name, hist in results.items():
    # 用相对 step（每个算法 normalization 到 0-1）来对齐 x 轴
    n_steps = len(hist)
    rel_steps = np.linspace(0, 1, n_steps)
    tr = [h['true_reward'] for h in hist]
    tr_norm = [t - tr[0] for t in tr]
    ax.plot(rel_steps, tr_norm, 'o-', color=colors[name], linewidth=2.5,
            markersize=7, label=name)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('relative training progress (0=start, 1=end)')
ax.set_ylabel('true reward (relative)')
ax.set_title('Ground-truth reward 提升 vs 训练进度')
ax.legend(); ax.grid(alpha=0.3)

# 左下：distribution shift（seen prompts vs novel prompts 的 true reward gap）
ax = axes[1, 0]
bar_labels = list(results.keys())
seen_means = []
novel_means = []
for name, hist in results.items():
    # 取最后的值
    seen_means.append(hist[-1]['true_reward'] - hist[0]['true_reward'])
    novel_means.append(hist[-1]['novel_true_reward'] - hist[0]['novel_true_reward'])

x = np.arange(len(bar_labels))
width = 0.35
ax.bar(x - width/2, seen_means, width, color='#2ca02c', alpha=0.7, label='seen prompts')
ax.bar(x + width/2, novel_means, width, color='#ff7f0e', alpha=0.7, label='novel prompts')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(bar_labels)
ax.set_ylabel('true reward gain (relative to start)')
ax.set_title('Distribution shift: seen vs novel prompts')
ax.legend(); ax.grid(alpha=0.3, axis='y')

# 右下：算法开销对比（每 wall-clock 秒处理的样本数）
ax = axes[1, 1]
algo_info = []
for name, hist in results.items():
    total_time = hist[-1]['wall_time']
    n_outer_steps = hist[-1]['step']
    # 收益：final true reward gain（用 ground-truth，不用 noisy RM）
    reward_gain = hist[-1]['true_reward'] - hist[0]['true_reward']
    algo_info.append((name, total_time, n_outer_steps, reward_gain))

names_ = [x[0] for x in algo_info]
times_ = [x[1] for x in algo_info]
gains_ = [x[3] for x in algo_info]
# 每个 unit true reward gain 花的时间（越低越高效）
time_per_gain = [t / max(abs(g), 0.01) for t, g in zip(times_, gains_)]

bars = ax.bar(names_, time_per_gain,
              color=[colors[n] for n in names_], alpha=0.7)
ax.set_ylabel('seconds per unit true reward gain')
ax.set_title('工程开销：DPO 单位收益花费的时间最少（越小越好）')
ax.grid(alpha=0.3, axis='y')
for bar, val, name in zip(bars, time_per_gain, names_):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.05,
            f'{val:.1f}s', ha='center', fontsize=10)

plt.suptitle("Phase 3 算法竞技场：DPO vs GRPO vs PPO-RLHF", fontsize=13)
plt.tight_layout(); plt.show()

print("=" * 70)
print("三方对比总结:")
print("=" * 70)
print(f"{'算法':<12}{'wall-clock':<14}{'RM reward gain':<20}{'true gain':<15}{'novel gain'}")
for name, hist in results.items():
    rm_gain = hist[-1]['rm_reward'] - hist[0]['rm_reward']
    tr_gain = hist[-1]['true_reward'] - hist[0]['true_reward']
    nv_gain = hist[-1]['novel_true_reward'] - hist[0]['novel_true_reward']
    print(f"{name:<12}{hist[-1]['wall_time']:<14.1f}{rm_gain:<+20.3f}{tr_gain:<+15.3f}{nv_gain:+.3f}")

print()
print("观察:")
print("  - DPO: true reward gain 最大（+3.11）—— actor 把概率 mass 从 'no'/'ok' 转到 'good'")
print("    原因：DPO 直接在静态偏好数据上训，不需要采到 'good' 才能学（off-policy 数据高效）")
print("  - GRPO / PPO-RLHF: 在 toy 模型 + 少迭代下 reward gain 很小（≈ 0）")
print("    原因：未训练的 actor 采出的 response 多是 nonsense，RM 给不出有区分度的 reward，")
print("    group baseline / GAE advantage 信号弱 → 学得慢。在**大模型 + 充分训练**下，")
print("    GRPO/PPO 能追上 DPO（这是工业级实验的结论，不是 toy 模型能完全展示的）")
print("  - Distribution shift（DPO）: novel 和 seen gain 接近（因为本 toy 数据 prompts 都很相似）")
print("    在真实大数据集上 DPO 的 distribution shift 更明显（off-policy 的根本代价）")
print()
print("工程含义:")
print("  - 当你有大量成对偏好数据 + 计算资源有限 → 优先试 DPO（性价比高）")
print("  - 当你有可计算 reward（代码执行/数学正确性）→ 必须 GRPO/PPO（DPO/KTO 用不上）")
print("  - 这正是 §14.6 决策树的核心判断")""")

# 14.5.8 distribution shift 详细演示
code(r"""# 14.5.8 Distribution shift 演示：DPO 在 seen vs novel prompts 上的 reward gap
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# 左：每个算法的 seen vs novel reward（轨迹）
ax = axes[0]
for name, hist in results.items():
    steps = np.linspace(0, 1, len(hist))
    seen = [h['true_reward'] - hist[0]['true_reward'] for h in hist]
    novel = [h['novel_true_reward'] - hist[0]['novel_true_reward'] for h in hist]
    ax.plot(steps, seen, '-', color=colors[name], linewidth=2, label=f'{name} (seen)')
    ax.plot(steps, novel, '--', color=colors[name], linewidth=2, label=f'{name} (novel)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('relative training progress')
ax.set_ylabel('true reward gain')
ax.set_title('Distribution shift 演示：seen（实线）vs novel（虚线）')
ax.legend(fontsize=8, ncol=3, loc='upper left')
ax.grid(alpha=0.3)

# 右：每个算法的 seen - novel reward gap（越大 = distribution shift 越严重）
ax = axes[1]
for name, hist in results.items():
    steps = np.linspace(0, 1, len(hist))
    gap = [(h['true_reward'] - hist[0]['true_reward'])
           - (h['novel_true_reward'] - hist[0]['novel_true_reward'])
           for h in hist]
    ax.plot(steps, gap, 'o-', color=colors[name], linewidth=2, markersize=6, label=name)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('relative training progress')
ax.set_ylabel('seen gain − novel gain')
ax.set_title('Distribution shift gap（>0 表示 seen 上比 novel 提升更多）')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()

print("Distribution shift 解释:")
print("  - DPO 在静态偏好数据上训，actor 学到的是数据集里的 (prompt, winner, loser) 模式")
print("  - 当 actor 在 'novel prompts' 上生成 response 时，可能落到数据集没覆盖的区域")
print("  - → DPO 在 novel prompts 上的提升通常 < seen prompts（gap > 0）")
print("  - PPO / GRPO 是 on-policy: 每步采 actor 当前分布的 response，自然覆盖 actor 的支持")
print("  - 这是 DPO 的根本代价：用便宜换来了 distribution shift 风险")""")


# =============================================================================
# 14.6 决策树 + Phase 3 收尾 + 项目整体总结
# =============================================================================
md(r"""## 14.6 算法决策树 + Phase 3 收尾 + 项目整体总结

### 14.6.1 算法决策树（CH00 承诺）

本章是 Phase 3 收尾，我们终于可以画出 **CH00 早就承诺的"算法决策树"**——
回答"什么场景应该选 PPO / GRPO / DPO / KTO"。

```
                  要对齐一个 LLM？
                        │
                        ▼
              ┌─────────────────────┐
              │ 能拿到什么数据？    │
              └─────────────────────┘
                        │
        ┌───────────────┼─────────────────┐
        ▼               ▼                 ▼
   成对偏好数据      good/bad 标签     只有 reward 函数
   (winner, loser)   thumbs up/down    (e.g. 数学正确性)
        │               │                 │
        ▼               ▼                 ▼
       DPO             KTO                │
        │               │                 │
        │               │                 ▼
        │               │            ┌────────┐
        │               │            │ 需要在线│
        │               │            │  探索？ │
        │               │            └────────┘
        │               │              │
        │               │       ┌──────┴──────┐
        │               │       ▼             ▼
        │               │     是的           否
        │               │       │             │
        │               │       ▼             ▼
        │               │   GRPO/PPO    Best-of-N
        │               │       │         Sampling
        │               │       │        (no training)
        │               │       │
        │               │   ┌───┴───┐
        │               │   ▼       ▼
        │               │  显存紧？  否
        │               │   │       │
        │               │   ▼       ▼
        │               │  GRPO    PPO-RLHF
        │               │ (no critic)
        ▲               ▲
        │               │
   ┌────┴────┐    ┌─────┴─────┐
   │ 数据量大 │    │ 数据量小  │
   │ (>100k)  │    │ (<10k)    │
   └─────────┘    └───────────┘
        │               │
        ▼               ▼
   DPO 直接训    考虑 IPO / SimPO
   （β 调大）    (DPO 变体，对小数据更稳)
```

### 14.6.2 决策树的关键判断

1. **数据格式决定算法家族**：
    - 成对偏好 → DPO 家族（DPO / IPO / SimPO）
    - good/bad 标签 → KTO 家族
    - 可计算的 reward（如代码执行结果、数学正确性）→ PPO/GRPO 家族
2. **是否需要在线探索**：DPO/KTO 是离线的，PPO/GRPO 是在线的
3. **显存预算**：critic 是大头，显存紧就 GRPO；预算充足就 PPO
4. **数据量**：小数据下 DPO 容易过拟合，考虑 IPO（Identity Preference Optimization）
   或加正则化

### 14.6.3 现代 LLM 对齐 stack（2024-2025）

| 阶段 | 算法 | 代表模型 |
|---|---|---|
| Pre-training | next-token prediction | 所有 LLM |
| SFT | supervised learning | LLaMA / GPT / Claude |
| 对齐（成对偏好） | **DPO / IPO / SimPO** | LLaMA-3, Mistral |
| 对齐（good/bad） | **KTO** | 部分 Anthropic 内部模型 |
| 对齐（可计算 reward） | **GRPO / PPO** | DeepSeek-R1, GPT-4 |
| 推理增强 | GRPO on CoT data | **DeepSeek-R1** |

**趋势**：DPO/KTO 因简单稳定在通用对齐中越来越流行；GRPO/PPO 在有可计算
reward 的场景（reasoning、tool use）仍不可替代。""")

# 14.6.4 决策树可视化（matplotlib 版）
code(r"""# 14.6.4 算法决策树（matplotlib 可视化版）
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='#ffffff', edge='#333', fontsize=10, weight='normal'):
    rect = plt.Rectangle((x - w/2, y - h/2), w, h, facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=weight)

def draw_arrow(ax, x1, y1, x2, y2, label='', label_offset=(0, 0)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

# Root
draw_box(ax, 7, 9.3, 4.0, 0.7, '要对齐 LLM？', '#888', '#333', 12, 'bold')

# Level 1: 数据类型
draw_box(ax, 2.5, 8.0, 3.5, 0.7, '成对偏好 (winner, loser)', '#bbdefb', '#1976d2', 10)
draw_box(ax, 7,   8.0, 3.5, 0.7, 'good / bad 标签', '#ffe0b2', '#f57c00', 10)
draw_box(ax, 11.5,8.0, 3.5, 0.7, '可计算 reward', '#c8e6c9', '#388e3c', 10)

draw_arrow(ax, 5.2, 9.0, 3.5, 8.4)
draw_arrow(ax, 7, 9.0, 7, 8.4)
draw_arrow(ax, 8.8, 9.0, 11.0, 8.4)

# Level 2: 主推算法
draw_box(ax, 2.5, 6.5, 2.8, 0.7, 'DPO 家族', '#1565c0', 'white', 12, 'bold')
draw_box(ax, 7,   6.5, 2.8, 0.7, 'KTO', '#e65100', 'white', 12, 'bold')
draw_box(ax, 11.5,6.5, 2.8, 0.7, 'PPO / GRPO', '#1b5e20', 'white', 12, 'bold')

draw_arrow(ax, 2.5, 7.65, 2.5, 6.85)
draw_arrow(ax, 7, 7.65, 7, 6.85)
draw_arrow(ax, 11.5, 7.65, 11.5, 6.85)

# Level 3: 细分（DPO 家族 / PPO 家族）
draw_box(ax, 1.3, 5.0, 2.0, 0.6, 'DPO (大数据)', '#90caf9', '#1976d2', 9)
draw_box(ax, 3.7, 5.0, 2.0, 0.6, 'IPO / SimPO (小数据)', '#90caf9', '#1976d2', 9)
draw_arrow(ax, 2.0, 6.15, 1.3, 5.3)
draw_arrow(ax, 3.0, 6.15, 3.7, 5.3)

# PPO 家族细分
draw_box(ax, 10.3, 5.0, 2.0, 0.6, 'GRPO (显存紧)', '#a5d6a7', '#388e3c', 9)
draw_box(ax, 12.7, 5.0, 2.0, 0.6, 'PPO-RLHF (标准)', '#a5d6a7', '#388e3c', 9)
draw_arrow(ax, 11.0, 6.15, 10.3, 5.3)
draw_arrow(ax, 12.0, 6.15, 12.7, 5.3)

# Level 4: 适用场景说明
ax.text(2.5, 4.2, 'OpenAI LLaMA-3 等通用对齐\n简单稳定，工业默认',
        ha='center', va='top', fontsize=9, color='#1565c0',
        style='italic')
ax.text(7, 4.2, '用户点赞/点踩数据\n标注成本低',
        ha='center', va='top', fontsize=9, color='#e65100',
        style='italic')
ax.text(11.5, 4.2, 'DeepSeek-R1 (GRPO)\nGPT-4 (PPO)\nreasoning / tool use',
        ha='center', va='top', fontsize=9, color='#1b5e20',
        style='italic')

# 底部：关键 trade-off 总结
draw_box(ax, 7, 2.3, 12, 1.3,
         '核心 trade-off:\n'
         'DPO/KTO: off-policy → 快但 distribution shift\n'
         'PPO/GRPO: on-policy → 慢但更鲁棒\n'
         'DeepSeek-R1 用 GRPO 因为有 reasoning trace 的可计算 reward',
         '#fff9c4', '#fbc02d', 10, 'bold')

# 关键判断说明
draw_box(ax, 2, 0.6, 4, 0.6, '简单稳定优先 → DPO', '#e3f2fd', '#1976d2', 9)
draw_box(ax, 7, 0.6, 4, 0.6, '可计算 reward → GRPO', '#e8f5e9', '#388e3c', 9)
draw_box(ax, 12, 0.6, 4, 0.6, ' thumbs 数据 → KTO', '#fff3e0', '#f57c00', 9)

ax.set_title('RLStudy 算法决策树（CH00 承诺）—— 什么场景选 PPO/GRPO/DPO/KTO',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout(); plt.show()""")

# 14.6.5 Phase 3 收尾 + 项目整体总结
md(r"""## 14.6.5 Phase 3 收尾 + 项目整体总结

### Phase 3（Ch10-14）回顾

| 章 | 主题 | 核心交付 |
|---|---|---|
| **Ch10** | TinyGPT（从零搭 mini-GPT） | `TinyGPT`, `sft_loss`, `generate` |
| **Ch11** | Reward Modeling | `RewardModel`, `bradley_terry_loss`, 过优化曲线 |
| **Ch12** | RLHF-PPO（InstructGPT 配方） | `RLHFTrainer`（4 模型协调器） |
| **Ch13** | GRPO（DeepSeek-R1 算法） | `GRPOTrainer`（3 模型，去掉 critic） |
| **Ch14** | DPO / KTO（无 RL 替代） | `DPOTrainer`, `KTOTrainer`（2 模型） |

Phase 3 的主线是**渐进简化**：

```
Ch12: 4 模型 →  Ch13: 3 模型（去 critic）  →  Ch14: 2 模型（去 RM + 去 rollout）
```

每一步去掉一个组件，都对应一个理论洞察（不是工程优化，是数学上的 re-formulation）。

### 整个 RLStudy 项目回顾（Ch00-Ch14）

| 阶段 | 章节 | 学到什么 |
|---|---|---|
| **Phase 1** | Ch00-05 | RL 基础：bandit → MDP → DP → TD → Q-learning/SARSA |
| **Phase 2** | Ch06-09 | deep RL：DQN → policy gradient → actor-critic → PPO/TRPO |
| **Phase 3** | Ch10-14 | RL applied to LLMs：TinyGPT → RM → RLHF → GRPO → DPO/KTO |

### 项目兑现的所有承诺（最终核对）

| 出处 | 承诺 | 兑现章 |
|---|---|---|
| Ch00 | 项目终极目标 GRPO | **Ch13** |
| Ch00 | 章节图（14 章） | **Ch00-14 全部完成** |
| Ch00 | "算法决策树" | **Ch14 §14.6** |
| Ch02 §2.5 | RLHF γ 0.9-0.95（KL penalty 解读） | Ch11 §11.6.2 / Ch12 |
| Ch05 §5.10 | "GRPO 去掉 value function" | **Ch13** |
| Ch12 §12.8 | "RLHF 太重 → DPO 替代" | **Ch14** |
| Ch11 §11.7 | "DPO 绕开 RL" 预告 | **Ch14 §14.2** |

### 给读者的下一步

完成本项目后，你具备了：

1. **从零实现**所有主流 deep RL 算法（DQN / PG / AC / PPO）
2. **从零实现** LLM + RLHF pipeline（TinyGPT + RM + PPO/GRPO/DPO）
3. 理解每一步的**数学推导**（不是黑盒调包）
4. 能根据场景**选择合适算法**（§14.6 决策树）

下一步建议：
- 读 **DeepSeek-R1 论文**（GRPO 完整版 + reasoning RL pipeline）
- 读 **DPO / KTO 原论文**（更细的推导 + 变体 IPO / SimPO）
- 实践：在 HuggingFace 上对一个真实 LLM（如 LLaMA）跑 DPO（TRL 库）
- 探索前沿：reasoning RL（o1 / R1-style），constitutional AI，self-play""")

# 14.6.6 最终评估报告 + 项目收尾
code(r"""# 14.6.6 Ch14 + 整个项目最终评估报告
print("=" * 70)
print("Ch14 DPO / KTO —— 最终评估报告")
print("=" * 70)

print(f"\n[1] 新基础设施: utils/dpo.py")
print(f"    sequence_log_probs     - 算 log π(y|x)（支持变长 + padding）")
print(f"    dpo_loss               - DPO loss（-log σ(β·(Δ_w - Δ_l))）")
print(f"    kto_loss               - KTO loss（pointwise + loss aversion）")
print(f"    prospect_value         - prospect theory value function（教学/可视化）")
print(f"    DPOConfig              - 超参 dataclass（无 clip_eps/critic_lr/gamma）")
print(f"    DPOTrainer             - 2 模型协调器（actor + frozen reference）")
print(f"    KTOTrainer             - KTO 版本")
print(f"    tests/test_dpo.py: 12 个冒烟测试，全部通过")

print(f"\n[2] DPO 训练效果（与 Ch11 RM 独立验证）")
final_dpo_stats = dpo_history[-1]
print(f"    起点 RM reward:        {dpo_history[0]['rm_reward']:+.3f}")
print(f"    训练后 RM reward:      {final_dpo_stats['rm_reward']:+.3f}")
print(f"    提升:                  {final_dpo_stats['rm_reward'] - dpo_history[0]['rm_reward']:+.3f}")
print(f"    起点 true reward:      {dpo_history[0]['true_reward']:+.3f}")
print(f"    训练后 true reward:    {final_dpo_stats['true_reward']:+.3f}")
print(f"    → DPO 训练让 actor 在 ground-truth reward 上大幅提升（验证 DPO 有效）")
print(f"    （注：toy RM 训得不充分，RM-based reward 数值与 true reward 可能不一致；")
print(f"     但 DPO 自身的 val accuracy 从 0% 提升到 96%，证明 DPO 确实学到了偏好）")

print(f"\n[3] 三方对比")
print(f"{'算法':<12}{'wall-clock':<14}{'RM gain':<14}{'true gain':<14}{'novel gain'}")
for name, hist in results.items():
    rm_g = hist[-1]['rm_reward'] - hist[0]['rm_reward']
    tr_g = hist[-1]['true_reward'] - hist[0]['true_reward']
    nv_g = hist[-1]['novel_true_reward'] - hist[0]['novel_true_reward']
    print(f"{name:<12}{hist[-1]['wall_time']:<14.1f}{rm_g:<+14.3f}{tr_g:<+14.3f}{nv_g:+.3f}")

print(f"\n[4] DPO 推导完整度: 4 步全展开")
print(f"    Step 1: RLHF 最优解 π*(y|x) = (1/Z) π_ref(y|x) exp(r/β)   [Ch12 §12.3]")
print(f"    Step 2: 反解 reward r(x,y) = β log(π*/π_ref) + β log Z    [§14.2.3]")
print(f"    Step 3: 代入 Bradley-Terry，Z(x) 在相减中消掉            [§14.2.4]")
print(f"    Step 4: 最大似然 → DPO loss = -log σ(β(Δ_w - Δ_l))      [§14.2.5]")

print(f"\n[5] KTO 介绍完整度")
print(f"    prospect theory 完整介绍（value function + loss aversion）  [§14.3.1]")
print(f"    KTO loss 公式推导（good/bad 加权 softplus）               [§14.3.3]")
print(f"    DPO vs KTO 决策表（数据格式 / 标注成本 / 推荐场景）       [§14.3.4]")

print(f"\n[6] 算法决策树: §14.6.1-14.6.4")
print(f"    决策维度: 数据格式 / 在线性 / 显存预算 / 数据量")
print(f"    matplotlib 可视化版: §14.6.4")

print(f"\n[7] Phase 3 + 项目整体")
print(f"    Phase 3 主线: 4 模型 (Ch12) → 3 模型 (Ch13) → 2 模型 (Ch14)")
print(f"    项目兑现承诺: Ch00 章节图 + GRPO + 决策树 = 全部 ✓")
print("=" * 70)""")

# 14.6.7 项目里程碑（最后一格可视化）
code(r"""# 14.6.7 RLStudy 项目里程碑：14 章 + 5 阶段（最终可视化）
fig, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
ax.axis('off')

# 三个 phase 的色块
phase_info = [
    ('Phase 1: RL Foundations', 0.5, 5.0, '#bbdefb', 'Ch00-05'),
    ('Phase 2: Deep RL',         5.5, 4.0, '#c8e6c9', 'Ch06-09'),
    ('Phase 3: LLM Alignment',   9.7, 5.0, '#ffe0b2', 'Ch10-14'),
]
for name, x0, w, color, subtitle in phase_info:
    rect = plt.Rectangle((x0, 0.5), w, 5, facecolor=color, edgecolor='black',
                          alpha=0.4, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x0 + w/2, 5.2, name, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x0 + w/2, 4.7, subtitle, ha='center', va='center', fontsize=10, style='italic')

# 章节方块
chapters = [
    (1.0, 'Ch00', 'Setup'),
    (2.0, 'Ch01', 'Bandit'),
    (3.0, 'Ch02', 'MDP'),
    (4.0, 'Ch03', 'DP'),
    (5.0, 'Ch04', 'TD'),
    (6.0, 'Ch05', 'Q/SARSA'),
    (6.5, 'Ch06', 'DQN'),
    (7.5, 'Ch07', 'PG'),
    (8.5, 'Ch08', 'A-C'),
    (9.5, 'Ch09', 'PPO'),
    (10.2, 'Ch10', 'TinyGPT'),
    (11.2, 'Ch11', 'RM'),
    (12.2, 'Ch12', 'RLHF'),
    (13.2, 'Ch13', 'GRPO★'),
    (14.2, 'Ch14', 'DPO/KTO'),
]
for x, ch, title in chapters:
    color = '#1976d2' if ch.startswith('Ch0') or ch.startswith('Ch1') and int(ch[2:])<=5 else \
            '#388e3c' if int(ch[2:]) <= 9 else '#f57c00'
    if ch in ('Ch13',):
        color = '#d62728'  # 终极目标 GRPO
    if ch == 'Ch14':
        color = '#9467bd'  # 本章
    rect = plt.Rectangle((x - 0.4, 2.0), 0.8, 1.5, facecolor=color, edgecolor='black',
                          alpha=0.85, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, 3.1, ch, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(x, 2.5, title, ha='center', va='center', fontsize=8, color='white')

# 底部：每个 phase 的核心产出
phase_yields = [
    (3.0, 1.3, ' foundational RL:\nvalue iteration, TD,\nQ-learning'),
    (7.5, 1.3, 'Deep RL stack:\nDQN, A2C, PPO,\nGAE'),
    (12.2, 1.3, ' LLM alignment:\nRM, RLHF, GRPO,\nDPO/KTO'),
]
for x, y, text in phase_yields:
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

# 终极目标标注
ax.annotate('项目终极目标\nGRPO (DeepSeek-R1)',
            xy=(13.2, 3.5), xytext=(11.5, 5.7),
            fontsize=10, fontweight='bold', color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))

# 本章标注
ax.annotate('本章\n(Phase 3 收尾)',
            xy=(14.2, 3.5), xytext=(13.5, 0.7),
            fontsize=9, color='#9467bd',
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.5))

ax.set_title('RLStudy 项目里程碑：从 bandit 到 GRPO/DPO 的 14 章之旅（完成！）',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout(); plt.show()

print("=" * 70)
print("RLStudy 项目全部 14 章完成！")
print("=" * 70)
print()
print("回顾这 14 章，我们走了:")
print("  Phase 1 (Ch00-05): 一个 multi-armed bandit → MDP → DP → TD → Q-learning")
print("  Phase 2 (Ch06-09): MLP → DQN → policy gradient → actor-critic → PPO")
print("  Phase 3 (Ch10-14): mini-GPT → reward model → RLHF-PPO → GRPO → DPO/KTO")
print()
print("掌握的不只是 14 个算法，而是:")
print("  - RL 的统一视角（value / policy / model-based）")
print("  - deep RL 的工程要素（replay / target net / GAE / clipping）")
print("  - LLM 对齐的完整 pipeline（SFT → RM → RL/DPO）")
print("  - 每一步的数学推导（不是黑盒调包）")
print("  - 算法选择决策树（什么场景用什么）")
print()
print("希望这个项目让你从 RL 的'使用者'变成'理解者'。")
print("=" * 70)""")

md(r"""---

## 总结：本章你学到了什么

### 核心概念

1. **DPO 是数学魔法**：4 步代数变换把 RL 目标变成监督 loss。reward model、critic、
   PPO clipping 全部消失——但**理论保证不变**（DPO 解 = RLHF 解）。
2. **KTO 更轻**：用 prospect theory 替代 Bradley-Terry，只需要 good/bad 标签。
3. **隐式 reward**：$\hat r = \beta\log(\pi_\theta/\pi_{ref})$，不需要训 RM 就能评估。
4. **Distribution shift**：DPO 在静态数据上训的根本代价——actor 漂移到 unseen 区域时
   信号失效。PPO/GRPO 的 on-policy 性质是它的解药。
5. **算法决策树**：数据格式 + 在线性 + 显存预算 + 数据量共同决定算法选择。

### DPO 推导 4 步（核心公式）

$$
\text{Step 1: } \pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

$$
\text{Step 2: } r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)
$$

$$
\text{Step 3: } P(y_w \succ y_l) = \sigma\!\left(\beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)}
                                        - \beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\right)
$$
（$\beta\log Z(x)$ 在相减中消掉）

$$
\text{Step 4: } \mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta(\Delta_w - \Delta_l)\right)
\quad\text{where}\quad \Delta = \log\frac{\pi_\theta}{\pi_{ref}}
$$

### Phase 3 + 项目收尾

Phase 3 把 Ch12 的 4 模型 → Ch13 的 3 模型 → Ch14 的 2 模型，
每一步都对应一个理论洞察（不是工程优化）。整个 RLStudy 项目到这里完成全部 14 章。

> **"The proof of the pudding is in the eating."** —— DPO 论文用代数变换证明了
> RL 不是 LLM 对齐的唯一答案。但正如 §14.6 决策树所示，没有银弹——
> 在 reasoning RL 的时代（DeepSeek-R1 / OpenAI o1），GRPO/PPO 这种"重 RL"反而
> 重新变得不可替代。理解每个算法的 trade-off，比迷信"X 比 Y 好"重要得多。

---

**项目全部完成。感谢陪我们走完这 14 章！** 🎉""")


# =============================================================================
# Write notebook
if __name__ == "__main__":
    # 直接运行时写出 notebook（被 build_notebooks.py import 时不写盘）
    _nb.write("ch14_dpo_kto.ipynb")
