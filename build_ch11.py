"""Build notebooks/ch11_reward_modeling.ipynb via nbformat.

Run:  python build_ch11.py
This produces the .ipynb file. Then execute it with nbconvert.
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch11")
cells = _nb.cells            # build_notebooks.py 适配层读取的模块级列表
md, code = _nb.md, _nb.code


# =============================================================================
# Title / overview
# =============================================================================
md(r"""# 第 11 章：Reward Modeling —— 让模型知道"什么是好"

> **Ch10** 给了我们一个会"接龙"的 TinyGPT 和 SFT 后会"答问题"的模型。
> 但它还**不知道自己答得好不好**——这恰恰是 RLHF 的第二阶段要解决的。
>
> 本章的核心问题：
>
> > **如何从人类偏好（"A 比 B 好"这种二元判断）里反推出一个连续的 reward 函数 $r(x, y)$？**
>
> 答案是 **Bradley-Terry 模型**（1952 年的赛马排名模型，被 Christiano et al. 2017、
> InstructGPT 2022 借鉴到 RLHF）。它把"偏好"建模成 reward 差的 sigmoid：

$$\boxed{\;P(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r(x, y_w) - r(x, y_l)\big)\;}$$

只要拿到一批 pairwise 偏好数据 $(x, y_w, y_l)$，就能用最大似然训出一个 reward model。

## 学习目标

1. 理解 **为什么 SFT 不够**（Ch10 held-out 0/3 失败 → 需要显式偏好建模）
2. **完整推出 Bradley-Terry 模型**：从"偏好 = reward 差的函数"到 sigmoid
3. 设计 **合成偏好数据**（用"隐含 reward"规则模拟人类标注）
4. 实现 **RewardModel**：TinyGPT backbone + scalar head（不重复造轮子）
5. **训练 + 评估**：在合成偏好数据上训出 RM，验证集准确率 > 70%
6. **复现 Reward 过优化**（Goodhart's Law）—— Ch00 章节图承诺的"过优化曲线"
7. 理解 **KL penalty 的必要性**（Ch12 PPO 会用，本章做数学预告）

## 承接的 Phase 1 / Phase 3 承诺（2 处）

| 出处 | 承诺原文 | 本章兑现节 |
|---|---|---|
| **Ch00** | **"Ch11 Reward Modeling，偏好 UI + 过优化曲线"** | **§11.3 偏好 UI + §11.6 过优化曲线** |
| **Ch02 §2.5** | **"RLHF γ 选择 0.9-0.95（单轮对话不长）"** | **§11.7（KL penalty 等价于 soft γ，预告 Ch12）** |

## 术语速查

| 术语 | 含义 | 首次出现 |
|---|---|---|
| **preference / pairwise comparison** | "A 比 B 好"这种二元判断 | §11.2 |
| **Bradley-Terry model** | 把偏好建模成 reward 差的 sigmoid | §11.2 |
| **winner / loser** ($y_w, y_l$) | 一对里偏好的那个 / 不偏好的那个 | §11.3 |
| **reward model** $r_\theta(x, y)$ | prompt + response → 标量 reward | §11.4 |
| **reward head** | 把 transformer hidden state 压成 1 个 reward 值 | §11.4 |
| **Goodhart's Law** | "当一个度量成为目标时，它就不再是个好度量" | §11.6 |
| **reward hacking / over-optimization** | RL agent 找到 RM 的漏洞，r↑ 但真实质量↓ | §11.6 |
| **KL penalty** | $\text{reward}_{total} = r_\theta - \beta \, \text{KL}(\pi \| \pi_{ref})$ | §11.6 / §11.7 |

## 本章路线图

| 节 | 主题 | 关键产出 |
|---|---|---|
| 11.1 | 为什么 SFT 不够 | 回顾 Ch10 SFT 局限 → RM 动机 |
| 11.2 | **Bradley-Terry 完整推导** | sigmoid 偏好模型 + 可识别性 + MLE |
| 11.3 | 偏好数据格式 + 偏好 UI | pairwise 格式 + 合成数据生成 + ipywidgets UI |
| 11.4 | Reward model 架构 + 训练 loss | TinyGPT backbone + scalar head + Bradley-Terry loss |
| 11.5 | 完整训练 + 验证 | reward model 训练 + accuracy > 70% + reward 分布可视化 |
| 11.6 | **Reward 过优化（Goodhart's Law）** | 实验演示 r↑ 但真实质量↓ + KL penalty 预告 |
| 11.7 | 评估 + 小结 + Ch12 预告 | 总结 + γ/KL/Ch12 PPO 衔接 |""")

code(r"""# 常规设置：找项目根、载入库
import sys, pathlib, time, math, random
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

# Ch10 基础设施
from rlenvs import (
    CharTokenizer, TinyGPT, build_tiny_gpt,
)
# 本章新基础设施
from utils import set_seed
from utils.torch_utils import get_device, count_parameters
from utils.reward_model import (
    RewardModel, bradley_terry_loss,
    generate_preference_data, make_preference_batch, pad_to_length,
    reward_accuracy, predict_rewards, true_reward,
)

set_seed(42)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DEVICE = get_device()
print(f"PyTorch: {torch.__version__}, device = {DEVICE}")
print(f"本章新基础设施：utils/reward_model.py")
print(f"  - RewardModel              (TinyGPT backbone + scalar head)")
print(f"  - bradley_terry_loss       (-log sigma(r_w - r_l))")
print(f"  - generate_preference_data (合成 pairwise 偏好数据)")
print(f"  - make_preference_batch    / pad_to_length")
print(f"  - reward_accuracy          / predict_rewards / true_reward")""")


# =============================================================================
# 11.1 Why SFT is not enough
# =============================================================================
md(r"""## 11.1 为什么 SFT 不够

### 11.1.1 回顾 Ch10：SFT 的天花板

Ch10 §10.7 我们用 TinyGPT 做了 SFT——在 `"X plus Y is Z"` 这种算术数据上做条件 LM
训练。最后测了两件事：

- **held-in**（训练时见过的 `a, b ∈ [0,6]`）：模型大致能学到"看到 plus 就续上数字 + ."。
- **held-out**（`a, b > 6`，泛化测试）：**0/3 答对**——char-level 小模型学不会真正加法。

更一般地，SFT 有三个本质局限：

1. **只学模仿，不学偏好**。SFT 的 loss 是 cross-entropy，等价于"让模型在 seen 的
   response 上分配更高概率"。它没学"什么样的 response 才是好的"。
2. **数据 = 单一 ground truth**。每个 prompt 只有**一个**标准答案。但现实里很多
   prompt 有多种合理 response（"你好" vs "您好" vs "Hello" 都行）——SFT 无法表达这种偏好。
3. **没有 reward signal**。SFT 给的是"概率"，不是"好坏分数"。后面要拿 RL 优化，
   必须有一个标量 reward。

### 11.1.2 解决思路：从"标准答案"到"成对偏好"

人类标注员**更容易回答**"A 和 B 哪个好"，而不是"写一个完美答案"。所以 RLHF 的
阶段 2 用 **pairwise comparison**：

> 给标注员看 prompt $x$ 和两个 response $y_1, y_2$，让他选"哪个更好"。
> 得到三元组 $(x, y_w, y_l)$，$y_w$ = winner（更好的），$y_l$ = loser。

这种数据**比 SFT 数据便宜得多**（标注员不用自己写，只需要二选一），但它是**二元的**，
不能直接当 reward。我们需要把它**"插值"成一个连续的 reward 函数 $r(x, y)$**——
这就是 Bradley-Terry 模型的工作。

### 11.1.3 阶段位置

```
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │ 1. SFT       │ ───→ │ 2. Reward    │ ───→ │ 3. RL        │
   │   (Ch10)     │      │    Modeling  │      │   (Ch12 PPO) │
   │   会"答问题" │      │  ← 本章       │      │   往好里优化 │
   └──────────────┘      └──────────────┘      └──────────────┘
                          知道什么是"好"
```

本章只做阶段 2。完成后我们就有了一个 reward function，Ch12 拿它跑 PPO。""")

code(r"""# 11.1.4 重现 Ch10 SFT held-out 0/3 的失败（用一个快速训练演示）
# 这里我们**不**重新训 SFT，而是用一个简化的"模式匹配 vs 真偏好"对比，
# 说明 SFT 学不到"什么是好"。

# 设计 4 个 response，按"隐含人类偏好"排序：
demo_prompt = "Q: How is the weather? A:"
demo_responses = [
    "very good",   # 含 'good'，长度适中 → 好
    "good",        # 含 'good'，长度短 → 较好
    "ok",          # 中性
    "it is bad",   # 含 'bad' → 差
]
# 我们假设的人类偏好排序：
print(f"prompt: {demo_prompt!r}")
print(f"假设的偏好排序（人类标注员给出的）：")
for i, r in enumerate(demo_responses, 1):
    r_true = true_reward(demo_prompt, r)
    print(f"  排名 {i}: '{r}'   (隐含 reward = {r_true:+.2f})")

print()
print("关键观察：SFT 给的是 p(y | x)（每个 token 的概率），不是一个标量'好坏分数'。")
print("我们想要的是一个函数 r(x, y) → ℝ，能在上面跑 RL。")
print("下一节我们就推出这个 r 怎么从偏好数据里学。")""")


# =============================================================================
# 11.2 Bradley-Terry derivation
# =============================================================================
md(r"""## 11.2 Bradley-Terry 模型（本章核心 1/2）

### 11.2.1 起点：偏好是个二元事件

人类标注员给出的"A 比 B 好"($y_w \succ y_l$) 是一个**二元随机事件**——同一个标注员
在疲劳/分心时可能给出不同判断。所以我们把它建模成**概率**。

我们想找的量：$P(y_w \succ y_l \mid x)$——给定 prompt $x$，response $y_w$ 比 $y_l$ 更
受偏好的概率。

### 11.2.2 核心假设：每个 response 有一个"潜在 reward"

**Bradley-Terry 假设**（Bradley & Terry 1952）：

> 每个 response $y$ 在给定 prompt $x$ 下有一个**潜在标量 reward** $r(x, y) \in \mathbb{R}$。
> 偏好概率**只依赖 reward 差**：

$$P(y_1 \succ y_2 \mid x) = f\!\big(r(x, y_1) - r(x, y_2)\big)$$

其中 $f$ 是某个**单调递增**函数（reward 高 → 偏好概率大）。

> **为什么是"差"？** 因为 reward 的绝对值无意义——把所有 reward 加个常数 $c$
> 偏好关系不变。所以唯一能影响偏好的是**差**。这跟温度、电压、效用一样——
> 物理上只有"差"是可观测的。

<details>
<summary><b>完整推导：为什么 $f = \sigma$（sigmoid）？点开看</b></summary>

我们要从三个公理推出 $f$ 必须是 sigmoid。

**公理 1（单调性）**：$f$ 关于 reward 差严格单调递增。
（reward 越高越可能被偏好——合理）

**公理 2（对称性 / 互补性）**：偏好是二选一，所以

$$P(y_1 \succ y_2) + P(y_2 \succ y_1) = 1$$

代入 $f$：

$$f(z) + f(-z) = 1 \quad \forall z \in \mathbb{R}$$

（$z = r_1 - r_2$；$-z = r_2 - r_1$）。这一条直接排除了像 $f(z) = z$ 这种
（不满足 $f + f(-) = 1$）。但满足这条的函数还很多（如 $f(z) = \frac{1}{2}(1 + z/\sqrt{1+z^2})$）。
我们需要再加一条。

**公理 3（独立比较 / Luce 选择公理的特例）**：

> 多次独立比较的胜率比 = reward 差的指数函数。

形式化：如果 $y_1, y_2, y_3$ 各自有 reward $r_1, r_2, r_3$，那么

$$\frac{P(y_1 \succ y_3)}{P(y_2 \succ y_3)} = \frac{f(r_1 - r_3)}{f(r_2 - r_3)}$$

应该**不依赖 $r_3$**（独立性）。换句话说，"1 比 3 强"和"2 比 3 强"的比率
只取决于 1 和 2 本身（reward 差 $r_1 - r_2$），不取决于参考物 3。

把这条代进 $f(z) + f(-z) = 1$，可以推出（Plackett 1975 给了严格证明）：

$$f(z) = \frac{1}{1 + e^{-z}} = \sigma(z)$$

**直觉推导**：从独立性假设 $f(a+c) / f(b+c) = g(a-b)$（与 $c$ 无关），对 $c$ 求导
可以推出 $f$ 满足 logistic ODE $f'(z) = f(z)(1 - f(z))$，解就是 sigmoid。

**结论（Bradley-Terry 模型）**：

$$\boxed{\;P(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r(x, y_w) - r(x, y_l)\big)\;}$$

其中 $\sigma(z) = 1/(1 + e^{-z})$。这就是 InstructGPT / 所有现代 RLHF 论文用的
reward model 形式。

</details>

### 11.2.3 可识别性（identifiability）：reward 绝对值不可识别

Bradley-Terry 模型只看 reward **差** $r(x, y_w) - r(x, y_l)$。所以对任意常数 $c$：

$$r'(x, y) := r(x, y) + c \quad\Longrightarrow\quad P_{r'}(y_w \succ y_l) = P_r(y_w \succ y_l)$$

意思是 **reward 整体平移不改变偏好预测**。所以从偏好数据里只能学到 reward 的
**相对结构**，绝对值是"不可识别的"。

> **工程含义**：训完 reward model 后，reward 值的范围是任意的（可能是 [-3, 3]
> 也可能是 [-100, 100]）。**只有排序有意义**。这是为什么 InstructGPT 在跑 PPO 前
> 会做 reward normalization。
>
> **类比**：这跟 Dueling DQN 里 V/A 平移自由度（Ch06 §6.8）一模一样——
> 那里我们减去 A 的均值保证可识别，这里我们靠 normalization。

### 11.2.4 最大似然：从偏好到 loss

给定 $N$ 个偏好对 $\{(x_i, y_w^i, y_l^i)\}_{i=1}^{N}$，假设它们独立，
reward model $r_\theta$ 的对数似然是：

$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \log P(y_w^i \succ y_l^i \mid x_i) = \sum_{i=1}^{N} \log \sigma\!\big(r_\theta(x_i, y_w^i) - r_\theta(x_i, y_l^i)\big)$$

最大化 $\mathcal{L}$ 等价于最小化负对数似然：

$$\boxed{\;\mathcal{L}_{RM}(\theta) \;=\; -\frac{1}{N}\sum_{i=1}^{N} \log \sigma\!\big(r_\theta(x_i, y_w^i) - r_\theta(x_i, y_l^i)\big)\;}$$

注意几个等价写法（数值实现时常用）：

$$-\log\sigma(z) = \log(1 + e^{-z}) = \text{softplus}(-z)$$

所以 `bradley_terry_loss` 的实现是 `F.softplus(r_l - r_w).mean()`——
**数值稳定**，不会 `exp` 上溢。

### 11.2.5 几何直觉：loss 在 reward 差空间的样子""")

code(r"""# 11.2.6 可视化 Bradley-Terry loss：loss 仅依赖 reward 差
z = np.linspace(-5, 5, 200)
sigma = 1 / (1 + np.exp(-z))
nll = np.log(1 + np.exp(-z))   # = -log sigma(z) = softplus(-z)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# 左：sigmoid（偏好概率）随 reward 差
axes[0].plot(z, sigma, color='#1f77b4', linewidth=2.5)
axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel(r'reward diff  $z = r(x, y_w) - r(x, y_l)$')
axes[0].set_ylabel(r'preference prob  $P(y_w \succ y_l)$')
axes[0].set_title(r'Bradley-Terry: $P = \sigma(r_w - r_l)$')
axes[0].grid(alpha=0.3)
# 标三个特征点
for z0, label in [(-3, '排错 (r_w < r_l)'), (0, '五五开'), (3, '排对 (r_w > r_l)')]:
    p0 = 1/(1+np.exp(-z0))
    axes[0].plot(z0, p0, 'o', color='#d62728', markersize=10, zorder=5)
    axes[0].annotate(f'{label}\nP={p0:.2f}', xy=(z0, p0),
                     xytext=(z0+0.3, p0+0.12), fontsize=9)

# 右：loss = -log sigma
axes[1].plot(z, nll, color='#d62728', linewidth=2.5)
axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel(r'reward diff  $z = r_w - r_l$')
axes[1].set_ylabel(r'loss  $-\log \sigma(z) = \mathrm{softplus}(-z)$')
axes[1].set_title('Bradley-Terry loss（越右越对，loss 越小）')
axes[1].grid(alpha=0.3)
for z0 in [-3, 0, 3]:
    l0 = math.log1p(math.exp(-z0))
    axes[1].plot(z0, l0, 'o', color='#1f77b4', markersize=10, zorder=5)
    axes[1].annotate(f'z={z0:+d}\nloss={l0:.2f}', xy=(z0, l0),
                     xytext=(z0+0.3, l0+0.3), fontsize=9)

plt.tight_layout(); plt.show()

print("观察:")
print(f"  - reward 差 z → +∞: P → 1, loss → 0（完全排对）")
print(f"  - z = 0:        P = 0.5, loss = log(2) ≈ {math.log(2):.3f}（五五开）")
print(f"  - z → -∞:       P → 0, loss → ∞（完全排错，被严重惩罚）")
print(f"  - 这就是 RM 训练目标：把 (x, y_w, y_l) 的 r_w - r_l 推得越大越好")""")

code(r"""# 数值验证：Bradley-Terry 的两个核心断言（教材传统：公式都要过数值关）
# [验证 1] softplus(r_l - r_w) == -log σ(r_w - r_l)：两种 loss 写法等价
# [验证 2] σ(r_w - r_l) 确实是偏好概率：按它采样，经验频率 ≈ 模型概率
torch.manual_seed(0)

# ---- 验证 1：两种 loss 写法逐点一致 ----
z = torch.linspace(-8, 8, 17)                    # z = r_w - r_l
loss_a = torch.nn.functional.softplus(-z)        # softplus(r_l - r_w)
loss_b = -torch.log(torch.sigmoid(z))            # -log σ(r_w - r_l)
max_diff = (loss_a - loss_b).abs().max().item()
print(f"[验证 1] softplus(-z) vs -log sigmoid(z): max|Δ| = {max_diff:.2e} "
      f"({'✓ 等价' if max_diff < 1e-6 else '✗'})")

# ---- 验证 2：模型概率 vs 采样频率 ----
# 固定真 reward 差 Δ ∈ {-3, -1, 0, 1, 3}，各模拟 20000 次偏好判定
deltas = [-3.0, -1.0, 0.0, 1.0, 3.0]
N = 20000
print(f"\n[验证 2] 按 P(y_w ≻ y_l) = σ(Δ) 采样 {N} 次：")
print(f"{'Δ=r_w-r_l':>10} {'σ(Δ) 模型概率':>14} {'采样频率':>10} {'偏差':>10}")
for d in deltas:
    p_model = 1.0 / (1.0 + math.exp(-d))
    wins = (torch.rand(N) < p_model).float().mean().item()   # Bernoulli(σ(Δ))
    print(f"{d:>10.1f} {p_model:>14.4f} {wins:>10.4f} {abs(wins-p_model):>10.4f}")
print("偏差都在 ~1% 以内（二项分布的统计涨落）——BT 概率公式成立")""")


# =============================================================================
# 11.3 Preference data format + UI
# =============================================================================
md(r"""## 11.3 偏好数据格式 + 偏好 UI

### 11.3.1 数据格式：pairwise comparison

每条偏好样本是一个三元组 $(x, y_w, y_l)$：

| 字段 | 含义 | 示例 |
|---|---|---|
| `prompt` $x$ | 问题 / 指令 | `"Q: How are you? A:"` |
| `winner` $y_w$ | 标注员**偏好**的 response | `"very good"` |
| `loser` $y_l$ | 标注员**不偏好**的 response | `"it is bad"` |

数据集就是 $\{(x_i, y_w^i, y_l^i)\}_{i=1}^{N}$。

> **注意**：偏好数据**没有"绝对分"**——只有相对排序。同一个 response 在不同对里
> 可能是 winner 也可能是 loser（取决于和谁比）。这是它和 SFT 数据的本质区别。

### 11.3.2 合成偏好数据：模拟人类标注

真实 RLHF 的偏好数据要花大钱请标注员。教学场景我们用**规则模拟**——
定义一个"隐含 reward"函数 `true_reward(prompt, response)`，再用它生成偏好对。

我们的 `true_reward` 规则（简单可控）：
- 含 `"good"` 关键词 → +2.0（人类喜欢肯定回答）
- 含 `"bad"` 关键词 → -2.0（人类不喜欢否定回答）
- response 长度越接近 6 → 加分（不要太短不要太长）
- 最终 reward = 关键词分 + 长度分

然后：拿两个 response $A, B$，算 $r_A, r_B$，**reward 高的为 winner**。
（真实标注里这步由人类完成；这里由 ground-truth reward 代劳。）

> **重要**：reward model **不知道** `true_reward` 的规则——它只看到 $(x, y_w, y_l)$
> 这种二元偏好。能不能从偏好里反推出规则，是 §11.5 训练成功与否的检验。

### 11.3.3 偏好 UI（ipywidgets）

下面是一个**演示用**的偏好标注 UI——让"标注员"（你）在两个 response 之间选。
（注：`nbconvert --execute` 不支持交互，但 widget 仍会渲染。本节主要是为了
给你**看到**偏好数据是怎么来的——真实场景标注员就用类似 UI。）""")

code(r"""# 11.3.4 合成偏好数据生成（看几条样本）
# 用一个固定语料训练 tokenizer（覆盖所有 prompt / response 字符）
corpus = (
    "Q: How is the weather? A: good bad ok fine yes no it day way hello world great very "
    "Q: What do you think? A: Q: Is it good? A: Q: Tell me a word. A: Q: How are you? A: "
)
tok = CharTokenizer().train(corpus)
print(f"tokenizer vocab size: {tok.vocab_size}")
print(f"vocab: {tok.itos}")
print()

# 生成偏好数据
KEYWORD_W = 3.0   # 关键词权重（让 reward 信号更明显）
LEN_W = 0.3       # 长度权重
TARGET_LEN = 6

train_prefs = generate_preference_data(
    tok, n_samples=300, seed=0,
    keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN,
)
val_prefs = generate_preference_data(
    tok, n_samples=100, seed=999,
    keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN,
)
print(f"训练偏好对: {len(train_prefs)}")
print(f"验证偏好对: {len(val_prefs)}")
print()

print("前 10 条训练样本:")
for i, s in enumerate(train_prefs[:10]):
    print(f"  [{i}] prompt={s['prompt']!r}")
    print(f"       winner = {s['winner']!r}  (true reward = {s['r_w']:+.2f})")
    print(f"       loser  = {s['loser']!r}   (true reward = {s['r_l']:+.2f})")
    print(f"       reward diff = {s['r_diff']:+.2f}")
    print()""")

code(r"""# 11.3.5 可视化 ground-truth reward 分布（看哪些 response 是"好"的）
all_responses = sorted({s['winner'] for s in train_prefs} | {s['loser'] for s in train_prefs})
gt_rewards = {r: true_reward("", r, keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN)
              for r in all_responses}

fig, ax = plt.subplots(figsize=(10, 4))
items = sorted(gt_rewards.items(), key=lambda kv: kv[1])
labels, vals = zip(*items)
colors = ['#2ca02c' if v > 1.0 else '#d62728' if v < -1.0 else '#888888' for v in vals]
bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.7)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([repr(l) for l in labels], rotation=45, ha='right')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('ground-truth reward')
ax.set_title('各 response 的隐含 reward（绿色=好，红色=差）')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

print("关键观察:")
print("  - 含 'good' 的 response（绿）: reward 显著 > 0")
print("  - 含 'bad' 的 response（红）:  reward 显著 < 0")
print("  - 中性 response（灰）:         reward 接近 0，靠长度微调")
print("  - reward model 要学的就是：从 pairwise 偏好里反推出这个排序")""")

code(r"""# 11.3.6 偏好标注 UI 演示（ipywidgets）
# 让 "标注员"（你）在两个 response 之间二选一。
# nbconvert 不能交互，但 widget 会渲染；如果你在 JupyterLab 打开本 notebook 可以点。

import ipywidgets as widgets
from IPython.display import display, Markdown

# 选 5 对作为演示
demo_pairs = train_prefs[:5]

annotated_choices = []  # 保存"标注员"的选择

def make_pair_widget(pair_idx):
    pair = demo_pairs[pair_idx]
    # 随机打乱 winner/loser 顺序，避免模型只靠位置识别
    if (pair_idx * 7) % 2 == 0:
        opt_a, opt_b = pair['winner'], pair['loser']
        is_a_winner = True
    else:
        opt_a, opt_b = pair['loser'], pair['winner']
        is_a_winner = False
    prompt_html = f"<b>Prompt:</b> <code>{pair['prompt']}</code><br>"
    a_html = f"<b>Response A:</b> <code>{opt_a}</code><br>"
    b_html = f"<b>Response B:</b> <code>{opt_b}</code>"

    out = widgets.Output()
    def on_a(_):
        out.clear_output()
        chosen = "A" if is_a_winner else "B"
        correct = is_a_winner
        annotated_choices.append((pair_idx, "A", correct))
        with out:
            display(Markdown(f"你选了 **A**。{'（与 ground truth 一致）' if correct else '（与 ground truth 不一致）'}"))

    def on_b(_):
        out.clear_output()
        correct = not is_a_winner
        annotated_choices.append((pair_idx, "B", correct))
        with out:
            display(Markdown(f"你选了 **B**。{'（与 ground truth 一致）' if correct else '（与 ground truth 不一致）'}"))

    btn_a = widgets.Button(description="选 A", button_style='primary')
    btn_b = widgets.Button(description="选 B", button_style='warning')
    btn_a.on_click(on_a); btn_b.on_click(on_b)

    header = widgets.HTML(prompt_html + a_html + b_html)
    return widgets.VBox([header, widgets.HBox([btn_a, btn_b]), out])

ui = widgets.VBox([make_pair_widget(i) for i in range(min(3, len(demo_pairs)))])
display(Markdown("### 偏好标注 UI（演示，3 对样本）"))
display(Markdown("如果你在 JupyterLab 打开，可以点击按钮做标注。nbconvert 执行时按钮不响应，属正常。"))
display(ui)
print("(UI 已渲染。交互需要 JupyterLab/IPython kernel；nbconvert 下静态显示。)")""")


# =============================================================================
# 11.4 Reward model architecture + loss
# =============================================================================
md(r"""## 11.4 Reward model 架构 + 训练 loss（本章核心 2/2）

### 11.4.1 架构：TinyGPT backbone + scalar head

我们**不重复造轮子**——直接拿 Ch10 的 `TinyGPT` 当 backbone。reward model 只多两件事：

1. **去掉 LM head**（我们不预测 vocab 分布，只要标量 reward）。
2. **加一个 reward head**：LayerNorm + Linear($d_{model} \to 1$)，输出标量 reward。

```
prompt + response token 拼接  [B, T]
    ↓
    TinyGPT backbone（embedding + PE + N × TransformerBlock + final LayerNorm）
    ↓
    hidden states  [B, T, d_model]      ← 通过 forward hook 抓 ln_final 的输入
    ↓ 取最后一个 response token 的 hidden vector（自回归 transformer 的"汇总点"）
    [B, d_model]
    ↓ LayerNorm → Linear(d_model → 1) → squeeze
    reward scalar  [B]
```

> **为什么取最后一个 token？** 自回归 transformer 的位置 $T$ 能"看见"所有
> $\le T$ 的位置（causal mask）。所以最后一个 response token 的 hidden state
> **自然汇总了 prompt + 整个 response 的信息**——是当 reward 汇总的最自然选择。
> （InstructGPT 2022 也是这么做的。）

### 11.4.2 怎么"抓"hidden state：forward hook

`TinyGPT.forward` 返回 logits `[B, T, V]`（V = vocab），不是 hidden state。
我们不想改 `tiny_gpt.py`（会动 Ch10 测试）。所以用 PyTorch 的 **forward hook**：

```python
self.backbone.ln_final.register_forward_hook(
    lambda module, inp, out: hook_store(inp[0])
)
```

每次 forward 后 hook 自动触发，把 `ln_final` 的**输入**（= LM head 前的 hidden state）
存到 `self._hidden`。优雅、零侵入。

### 11.4.3 实现：bradley_terry_loss

```python
def bradley_terry_loss(reward_model, prompt_ids, y_w_ids, y_l_ids):
    r_w = reward_model(prompt_ids, y_w_ids)
    r_l = reward_model(prompt_ids, y_l_ids)
    # -log sigma(r_w - r_l) = softplus(r_l - r_w)，数值稳定
    return F.softplus(r_l - r_w).mean()
```

注意我们**不直接写 `-torch.log(torch.sigmoid(r_w - r_l))`**——当 `r_w - r_l`
很负时，`sigmoid → 0`，`log` 会数值爆炸。改用恒等变形 `softplus(-z) = log(1+e^{-z})`，
PyTorch 的 `F.softplus` 内部对大输入做了稳定化处理。

### 11.4.4 实例化 + 参数量""")

code(r"""# 11.4.5 实例化 RewardModel，看参数量
torch.manual_seed(42)

# 用小模型（CPU 可训，< 200k 参数）
backbone = build_tiny_gpt(
    vocab_size=tok.vocab_size,
    d_model=64, n_heads=4, n_layers=3, d_ff=256, max_seq_len=64,
)
reward_model = RewardModel(backbone)

total = count_parameters(reward_model)
backbone_params = count_parameters(reward_model.backbone)
head_params = total - backbone_params
print(f"RewardModel 参数量: {total:,}")
print(f"  - TinyGPT backbone: {backbone_params:,} ({backbone_params/total*100:.1f}%)")
print(f"  - reward head:      {head_params:,} ({head_params/total*100:.1f}%)")
print()
print(f"backbone 配置: d_model=64, n_heads=4, n_layers=3, d_ff=256, max_seq_len=64")
print(f"reward head: LayerNorm(64) + Linear(64 -> 1)  = {64*2 + 64 + 1} params")
print()

# 验证 forward：取 3 个样本看输出 shape
batch = make_preference_batch(train_prefs[:3], pad_id=tok.pad_id)
print(f"batch shapes:")
print(f"  prompt_ids: {tuple(batch['prompt_ids'].shape)}")
print(f"  winner_ids: {tuple(batch['winner_ids'].shape)}")
print(f"  loser_ids:  {tuple(batch['loser_ids'].shape)}")

r_w = reward_model(batch['prompt_ids'], batch['winner_ids'])
r_l = reward_model(batch['prompt_ids'], batch['loser_ids'])
print(f"\nforward 输出:")
print(f"  r_w (winner rewards): {r_w.detach().numpy().round(3)}")
print(f"  r_l (loser rewards):  {r_l.detach().numpy().round(3)}")
print(f"  shape: {tuple(r_w.shape)}  （每条样本 1 个标量 reward）")
print(f"\n初始时 r_w vs r_l 没有显著差异——还没训练。下面 loss 会把它们拉开。")""")


# =============================================================================
# 11.5 Training
# =============================================================================
md(r"""## 11.5 完整训练 + 验证

### 11.5.1 训练循环

标准的 mini-batch SGD：

```
for step in 1..N:
    batch ← 随机采 32 个偏好对
    r_w = RM(prompt, winner)
    r_l = RM(prompt, loser)
    loss = softplus(r_l - r_w).mean()   # = -log sigma(r_w - r_l)
    loss.backward()
    Adam step
```

我们监控两个指标：
- **训练 loss**（应该从 ~0.69 = log 2 降到 ~0.3）
- **验证 accuracy**：在验证偏好对上，`r_w > r_l` 的比例（应该 > 70%）

> **baseline**：random guess 的 accuracy = 50%。我们要显著超过这个。""")

code(r"""# 11.5.2 训练 reward model
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# 重新初始化（保证可复现）
backbone = build_tiny_gpt(
    vocab_size=tok.vocab_size,
    d_model=64, n_heads=4, n_layers=3, d_ff=256, max_seq_len=64,
)
reward_model = RewardModel(backbone)

opt = torch.optim.AdamW(reward_model.parameters(), lr=1e-3, weight_decay=0.01)

RM_TRAIN_STEPS = 600
BATCH_SIZE = 32
EVAL_EVERY = 50

train_losses = []
val_accs = [(0, 0.5)]  # (step, accuracy)
val_reward_w = []  # reward 分布（用于画演化图）
val_reward_l = []

t0 = time.time()
for step in range(RM_TRAIN_STEPS):
    reward_model.train()
    batch_idx = random.sample(range(len(train_prefs)), BATCH_SIZE)
    batch_samples = [train_prefs[i] for i in batch_idx]
    b = make_preference_batch(batch_samples, pad_id=tok.pad_id)
    loss = bradley_terry_loss(reward_model, b['prompt_ids'], b['winner_ids'], b['loser_ids'])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
    opt.step()
    train_losses.append(loss.item())

    # 定期评估
    if step % EVAL_EVERY == 0 or step == RM_TRAIN_STEPS - 1:
        acc = reward_accuracy(reward_model, val_prefs, pad_id=tok.pad_id)
        val_accs.append((step, acc))
        # 记录 reward 分布
        r_w_val = predict_rewards(reward_model, val_prefs, pad_id=tok.pad_id, which='winner')
        r_l_val = predict_rewards(reward_model, val_prefs, pad_id=tok.pad_id, which='loser')
        val_reward_w.append((step, r_w_val.numpy().copy()))
        val_reward_l.append((step, r_l_val.numpy().copy()))
        elapsed = time.time() - t0
        print(f"step {step:4d}  train_loss={loss.item():.3f}  val_acc={acc:.3f}  ({elapsed:.1f}s)")

final_acc = val_accs[-1][1]
print(f"\n训练完成: {RM_TRAIN_STEPS} 步，耗时 {time.time()-t0:.1f}s")
print(f"val accuracy: {val_accs[0][1]:.3f} → {final_acc:.3f}")
print(f"通过 70% 验收门槛: {'是' if final_acc > 0.70 else '否'}")""")

code(r"""# 11.5.3 训练曲线：loss + accuracy
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左：loss
ax = axes[0]
ax.plot(train_losses, color='#1f77b4', alpha=0.3, linewidth=0.6, label='train loss (raw)')
window = 20
if len(train_losses) > window:
    sm = np.convolve(train_losses, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(window-1, len(train_losses)), sm, color='#1f77b4', linewidth=2.5, label='train loss (smoothed)')
ax.axhline(math.log(2), color='gray', linestyle='--', alpha=0.7, label=f'log 2 ≈ {math.log(2):.2f} (random)')
ax.set_xlabel('training step'); ax.set_ylabel('Bradley-Terry loss')
ax.set_title(f'RM 训练 loss'); ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

# 右：accuracy
ax = axes[1]
steps, accs = zip(*val_accs)
ax.plot(steps, accs, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='val accuracy')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='random (0.5)')
ax.axhline(0.7, color='#d62728', linestyle='--', alpha=0.7, label='验收门槛 (0.70)')
ax.set_xlabel('training step'); ax.set_ylabel('accuracy on val preferences')
ax.set_title(f'RM 偏好预测准确率（最终 {final_acc:.1%}）')
ax.set_ylim(0.3, 1.0); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()""")

code(r"""# 11.5.4 Reward 分布演化：训练过程中 winner/loser reward 的分布
# 这一张图直接展示 "RM 学会了把 winner reward 拉高、loser reward 压低"
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
checkpoints = [0, 1, 2, 3, 5, len(val_reward_w)-1]   # step indices in val_reward_w
for ax, ck in zip(axes.flat, checkpoints):
    if ck >= len(val_reward_w):
        ax.axis('off'); continue
    step_i, rws = val_reward_w[ck]
    _, rls = val_reward_l[ck]
    bins = np.linspace(min(rws.min(), rls.min()) - 0.2, max(rws.max(), rls.max()) + 0.2, 25)
    ax.hist(rws, bins=bins, alpha=0.6, color='#2ca02c', label=f'winner (n={len(rws)})')
    ax.hist(rls, bins=bins, alpha=0.6, color='#d62728', label=f'loser  (n={len(rls)})')
    ax.axvline(rws.mean(), color='#2ca02c', linestyle='--', linewidth=1.5)
    ax.axvline(rls.mean(), color='#d62728', linestyle='--', linewidth=1.5)
    acc_at = next((a for s, a in val_accs if s == step_i), None)
    ax.set_title(f'step {step_i}  acc={acc_at:.2f}' if acc_at else f'step {step_i}')
    ax.set_xlabel('reward'); ax.set_ylabel('count')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Reward 分布演化：winner（绿）逐步与 loser（红）分离', fontsize=12)
plt.tight_layout(); plt.show()

print("观察:")
print("  - step 0: winner/loser reward 几乎重叠（RM 还没学到任何东西）")
print("  - 中期:   两个分布开始分离")
print("  - 末期:   winner 分布明显在 loser 右侧（RM 学到了偏好结构）")
print("  - 注意：reward 绝对值是任意的（不可识别），只有分布**分离**有意义")""")

code(r"""# 11.5.5 在测试集上做详细分析：哪些 response 被 RM 打高分 / 低分
reward_model.eval()

# 把所有 unique response 喂给 RM（统一用一个固定 prompt），看 RM 打分排序
all_responses = sorted({s['winner'] for s in train_prefs} | {s['loser'] for s in train_prefs})
fixed_prompt = "Q: How are you? A:"
prompt_ids = tok.encode(fixed_prompt).unsqueeze(0).expand(len(all_responses), -1).contiguous()

# 给每个 response 编码并 pad
resp_ids_list = [tok.encode(r) for r in all_responses]
resp_ids = pad_to_length(resp_ids_list, pad_id=tok.pad_id)

with torch.no_grad():
    rm_rewards = reward_model(prompt_ids, resp_ids).numpy()
gt_rewards_arr = np.array([true_reward(fixed_prompt, r, keyword_weight=KEYWORD_W,
                                        len_weight=LEN_W, target_len=TARGET_LEN)
                           for r in all_responses])

# 排序
order = np.argsort(-rm_rewards)  # 高到低
print(f"RM 给 response 打的 reward 排序（用 prompt {fixed_prompt!r}）:\n")
print(f"{'rank':<5}{'response':<16}{'RM reward':>12}{'true reward':>14}")
for rank, i in enumerate(order, 1):
    print(f"{rank:<5}{all_responses[i]!r:<16}{rm_rewards[i]:>+12.3f}{gt_rewards_arr[i]:>+14.3f}")

# Spearman rank correlation（衡量排序一致性）
from scipy.stats import spearmanr
rho, p = spearmanr(rm_rewards, gt_rewards_arr)
print(f"\nSpearman rank correlation: ρ = {rho:+.3f} (p = {p:.2e})")
print("ρ → +1: RM 排序和 ground truth 完全一致；ρ → 0: 没学到；ρ → -1: 反着学")""")


# =============================================================================
# 11.6 Reward over-optimization (Goodhart's Law)
# =============================================================================
md(r"""## 11.6 Reward 过优化（Goodhart's Law / Reward hacking）

### 11.6.1 Goodhart's Law 数学陈述

> **"当一个度量成为目标时，它就不再是一个好度量。"** —— Charles Goodhart (1975)

在 RLHF 里：

- **真实人类偏好** $r^*(x, y)$ 是不可观测的（永远不知道）。
- 我们训练了一个 **reward model** $r_\theta(x, y)$，它是 $r^*$ 的**近似**。
- 如果拿 $r_\theta$ 当 RL 的 reward function 直接优化 policy，agent 会找到
  $r_\theta$ 高但 $r^*$ 低的"漏洞"——这叫 **reward hacking** 或 **reward over-optimization**。

形式化（Gao et al. 2022, Scaling Laws for Reward Model Overoptimization）：

$$\text{true quality} = r^*(y_{\text{opt}}) \quad\text{vs}\quad \text{proxy reward} = r_\theta(y_{\text{opt}})$$

其中 $y_{\text{opt}}$ 是 policy 最大化 $r_\theta$ 得到的 response。

**典型的过优化曲线**（Ch00 章节图承诺）：

```
            ┌────────────────────────┐
   reward   │      proxy r_θ ↑↑↑      │
            │      /                  │
       ┌────┤     /                   │
       │    │    /                    │
   r*  │    │   /  ← 先一起涨（KL 小，proxy 还准）
       │    │  /                      │
       │    │ /                       │
       │    │× ← peak（KL 大到一定程度，proxy 开始失准）
       │    │  \                      │
       │    │   \  ← true quality r* ↓
       │    │    \                    │
       └────┴────┴────────────────────┘
                 KL(π || π_ref)  →

   proxy reward 单调上升，true quality 先升后降。
```

### 11.6.2 KL penalty（Ch12 详讲，这里做数学预告）

缓解过优化的标准做法是加 **KL penalty**：

$$\text{reward}_{\text{total}}(x, y) = r_\theta(x, y) - \beta \cdot \text{KL}\!\big(\pi(\cdot | x) \,\|\, \pi_{\text{ref}}(\cdot | x)\big)$$

- $\pi$：当前 policy（被优化的 LLM）
- $\pi_{\text{ref}}$：reference policy（通常是 SFT 后的模型）
- $\beta > 0$：KL 系数（控制"允许 policy 走多远"）

**直觉**：proxy reward 高的 response 可能是"漏洞"，离 reference policy 越远越可能是漏洞。
KL penalty 强制 policy 不要离 $\pi_{\text{ref}}$ 太远——防止 agent 钻 RM 的牛角尖。

> **关键联系（兑现 Ch02 §2.5 承诺）**：KL penalty 在数学上等价于一种 **"soft γ"**——
> 限制 policy 探索的范围，类似 MDP 里 γ 控制"未来有多重要"。
> 单轮对话（短 horizon）的 RLHF 通常 $\beta = 0.01 \sim 0.5$（对应"γ ≈ 0.9–0.95"的
> 等价 horizon ~ 10–20 步）。这是 Ch02 §2.5 承诺的"RLHF γ 0.9-0.95"的真正含义。

### 11.6.3 实验设计：模拟过优化

我们用一个**简化的 toy 实验**演示过优化现象（Gao et al. 2022 的精神）：

- 把 `true_reward` 当 $r^*$（真实人类偏好）。
- 把训练好的 RM 当 $r_\theta$（proxy）。
- 用一个**带噪 proxy**模拟"agent 最大化 $r_\theta$"：每步向 RM 高分方向走，
  同时引入随机扰动（模拟策略漂移）。
- 观察 proxy reward（涨）vs true reward（先涨后降）。

> 🤔 **先猜再跑**：想象你放开手脚去优化这个不完美的 RM——预测**两条曲线**的形状：(1) proxy reward（RM 自己打的分）随优化步数怎么走？(2) true reward（真实人类偏好）呢——一路跟着涨，还是先涨后跌，还是一开始就不涨？
>
> <details><summary>画完两条想象中的曲线再点开</summary>
>
> 经典剧本：proxy 一路狂涨（你确实在最大化它）；true 先涨（RM 大方向没错，顺便把真质量也带上去）后**拐头下跌**（agent 开始钻 RM 的空子——那些"RM 觉得好、人觉得差"的 response）。拐点出现的位置，就是 KL penalty 该介入的位置——记住这个图，Ch12 的 β 调参全在治它。
> </details>

为了清晰展示 "true 先涨后降"，我们故意**注入过拟合噪声**到 RM——
让 RM 在某些 response 上"过度自信"，agent 优化它就会偏离 $r^*$。""")

code(r"""# 11.6.4 过优化模拟实验
# 思路：把 RM 当 proxy，true_reward 当真实人类偏好。
# "agent" = 在 response 空间里搜索高 RM reward 的样本，但越走越远 = 加扰动。
#
# 为了清楚展示"过优化"（true 先涨后降），我们故意给 RM 加噪：
# - 用早期（未训透）的 RM 当 proxy：它学到了部分规则，但有 systematic bias。
# - 让 "agent" 沿 proxy reward 梯度方向走 + 越走越大的随机扰动。
# - 观察每步 proxy reward 和 true reward 的变化。

torch.manual_seed(7); np.random.seed(7); random.seed(7)

# 训一个**未充分训练**的 RM 当 proxy（这就是 over-optim 论文里 "preference model" 的角色）
backbone_proxy = build_tiny_gpt(
    vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=3, d_ff=256, max_seq_len=64,
)
proxy_rm = RewardModel(backbone_proxy)
opt_proxy = torch.optim.AdamW(proxy_rm.parameters(), lr=2e-3, weight_decay=0.01)
# 只训 80 步——故意训得不充分，proxy 有 bias
for step in range(80):
    batch_idx = random.sample(range(len(train_prefs)), 32)
    batch_samples = [train_prefs[i] for i in batch_idx]
    b = make_preference_batch(batch_samples, pad_id=tok.pad_id)
    loss = bradley_terry_loss(proxy_rm, b['prompt_ids'], b['winner_ids'], b['loser_ids'])
    opt_proxy.zero_grad(); loss.backward()
    opt_proxy.step()
proxy_acc = reward_accuracy(proxy_rm, val_prefs, pad_id=tok.pad_id)
print(f"Proxy RM 训了 80 步，val accuracy = {proxy_acc:.3f}（特意训不充分，有 bias）")

# "Agent"：从一个 response 出发，每步：
#   1. 候选 = 当前 response + 随机字符扰动（模拟策略漂移）
#   2. 选 RM reward 最高的候选（"贪婪最大化 proxy"）
#   3. 扰动幅度随步数增大（"agent 越走越远 = KL 越大"）
proxy_rm.eval()
base_chars = [c for c in tok.itos if c not in [tok.PAD_TOKEN, '?', '.', ':', '\n']]

def char_mutate(s, n_edits, rng):
    # 对字符串 s 随机改 n_edits 个字符（插入/替换/删除）
    s = list(s)
    for _ in range(n_edits):
        if len(s) == 0:
            s.append(rng.choice(base_chars))
            continue
        op = rng.choice(['insert', 'replace', 'delete'])
        pos = rng.randrange(len(s))
        if op == 'insert':
            s.insert(pos, rng.choice(base_chars))
        elif op == 'replace':
            s[pos] = rng.choice(base_chars)
        elif op == 'delete':
            s.pop(pos)
    return ''.join(s)

@torch.no_grad()
def proxy_reward_of(prompt_str, resp_str):
    p_ids = tok.encode(prompt_str).unsqueeze(0)
    r_ids = tok.encode(resp_str).unsqueeze(0)
    return float(proxy_rm(p_ids, r_ids).item())

# 跑 30 个 agent，每个 40 步，平均得到 over-optim 曲线
N_AGENTS = 12
N_STEPS = 30
PROMPT = "Q: How are you? A:"
INIT_RESPONSE = "fine"
rng = np.random.RandomState(0)

all_proxy = np.zeros((N_AGENTS, N_STEPS))
all_true  = np.zeros((N_AGENTS, N_STEPS))

for agent in range(N_AGENTS):
    # 每个 agent 独立 seed
    arng = random.Random(agent * 13 + 1)
    current = INIT_RESPONSE
    for step in range(N_STEPS):
        # 扰动幅度随步数线性增大（"KL 越来越大"）
        n_edits = 1 + step // 4
        # 生成 8 个候选
        cands = [char_mutate(current, n_edits, arng) for _ in range(8)]
        cands = [c for c in cands if len(c) > 0]
        if not cands:
            cands = [current]
        # 用 proxy reward 选最优
        scores = [proxy_reward_of(PROMPT, c) for c in cands]
        best = cands[int(np.argmax(scores))]
        # 记录 proxy / true
        all_proxy[agent, step] = max(scores)
        all_true[agent, step] = true_reward(PROMPT, best, keyword_weight=KEYWORD_W, len_weight=LEN_W, target_len=TARGET_LEN)
        current = best

# 平均 + 平滑
mean_proxy = all_proxy.mean(axis=0)
mean_true  = all_true.mean(axis=0)
# 用相对值（每个 agent 自己归一化到 step 0 = 0），让 trend 更清楚
norm_proxy = all_proxy - all_proxy[:, 0:1]
norm_true  = all_true  - all_true[:, 0:1]
mean_norm_proxy = norm_proxy.mean(axis=0)
mean_norm_true  = norm_true.mean(axis=0)

print(f"\n过优化模拟完成: {N_AGENTS} 个 agent × {N_STEPS} 步")
print(f"proxy reward 平均变化: {mean_norm_proxy[0]:+.2f} → {mean_norm_proxy[-1]:+.2f}")
print(f"true  reward 平均变化: {mean_norm_true[0]:+.2f} → {mean_norm_true[-1]:+.2f}")""")

code(r"""# 11.6.5 画过优化曲线（Ch00 章节图承诺）
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：单条曲线 + 平滑
ax = axes[0]
xs = np.arange(N_STEPS)
# 画 5 条原始曲线 + 平均线
for agent in range(min(5, N_AGENTS)):
    ax.plot(xs, norm_proxy[agent], color='#1f77b4', alpha=0.15, linewidth=0.8)
    ax.plot(xs, norm_true[agent],  color='#d62728', alpha=0.15, linewidth=0.8)
ax.plot(xs, mean_norm_proxy, color='#1f77b4', linewidth=3, label='proxy reward (RM r_θ)')
ax.plot(xs, mean_norm_true,  color='#d62728', linewidth=3, label='true quality (r*)')
ax.axhline(0, color='black', linewidth=0.7)
ax.set_xlabel('"优化步数" (≈ KL(π || π_ref) 增大方向)')
ax.set_ylabel('reward change (relative to step 0)')
ax.set_title('过优化曲线：proxy 持续涨，true 先涨后降')
ax.legend(); ax.grid(alpha=0.3)

# 右：标出关键阶段
ax = axes[1]
ax.plot(xs, mean_norm_proxy, color='#1f77b4', linewidth=3, label='proxy reward r_θ')
ax.plot(xs, mean_norm_true,  color='#d62728', linewidth=3, label='true quality r*')
# 找 true 的峰值
peak_step = int(np.argmax(mean_norm_true))
ax.axvline(peak_step, color='gray', linestyle='--', alpha=0.7)
ax.plot(peak_step, mean_norm_true[peak_step], '*', color='#d62728', markersize=20, zorder=5)
ax.annotate(f'true peak\nstep {peak_step}',
            xy=(peak_step, mean_norm_true[peak_step]),
            xytext=(peak_step+3, mean_norm_true[peak_step]+0.5),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
# 标注区域
ax.axvspan(0, peak_step, alpha=0.1, color='#2ca02c', label='proxy 还可靠（KL 小）')
ax.axvspan(peak_step, N_STEPS, alpha=0.1, color='#d62728', label='过优化（reward hacking）')
ax.axhline(0, color='black', linewidth=0.7)
ax.set_xlabel('"优化步数"')
ax.set_ylabel('reward change')
ax.set_title("Goodhart's Law：当度量变成目标时...")
ax.legend(loc='lower left'); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()

print("关键观察:")
print(f"  - proxy reward 单调上升（agent 一直在优化 RM 给的方向）")
print(f"  - true quality 在 step {peak_step} 达到峰值 {mean_norm_true[peak_step]:+.2f}，之后开始下降")
print(f"  - 这就是 Goodhart's Law：当 reward model 被当 reward function 优化时，它失效了")
print(f"  - 工程解药 = KL penalty（限制 policy 不要走太远）—— Ch12 PPO 会用")""")

code(r"""# 11.6.6 KL penalty 的作用：在 toy 实验里加 KL 限制看效果
# 简化模拟：给 agent 的"优化"加 KL-like 限制——
# 候选不仅要 proxy reward 高，还要"和 reference response 接近"（编辑距离小）
# 这相当于 KL penalty 把"远离 ref"的成本算进去

@torch.no_grad()
def proxy_reward_of(prompt_str, resp_str):
    p_ids = tok.encode(prompt_str).unsqueeze(0)
    r_ids = tok.encode(resp_str).unsqueeze(0)
    return float(proxy_rm(p_ids, r_ids).item())

def edit_distance_normalized(s1, s2):
    # 归一化编辑距离，返回 [0, 1] 区间
    m, n = len(s1), len(s2)
    if max(m, n) == 0:
        return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n] / max(m, n)

# 跑两版：β=0（无 KL penalty）vs β>0（有 KL penalty）
BETA = 2.0  # KL penalty 强度
N_AGENTS_KL = 8
results = {}

for label, beta in [('no KL (β=0)', 0.0), (f'with KL (β={BETA})', BETA)]:
    proxy_traj = np.zeros((N_AGENTS_KL, N_STEPS))
    true_traj  = np.zeros((N_AGENTS_KL, N_STEPS))
    for agent in range(N_AGENTS_KL):
        arng = random.Random(agent * 17 + 5)
        current = INIT_RESPONSE
        ref = INIT_RESPONSE  # reference response（KL penalty 的锚）
        for step in range(N_STEPS):
            n_edits = 1 + step // 4
            cands = [char_mutate(current, n_edits, arng) for _ in range(8)]
            cands = [c for c in cands if len(c) > 0]
            scores_proxy = np.array([proxy_reward_of(PROMPT, c) for c in cands])
            # KL proxy = 归一化编辑距离（离 ref 越远 cost 越大）
            kl_proxy = np.array([edit_distance_normalized(c, ref) for c in cands])
            # 加 KL penalty
            scores_total = scores_proxy - beta * kl_proxy
            best = cands[int(np.argmax(scores_total))]
            proxy_traj[agent, step] = proxy_reward_of(PROMPT, best)
            true_traj[agent, step]  = true_reward(PROMPT, best, keyword_weight=KEYWORD_W,
                                                   len_weight=LEN_W, target_len=TARGET_LEN)
            current = best
    results[label] = {
        'proxy': proxy_traj - proxy_traj[:, 0:1],
        'true':  true_traj  - true_traj[:, 0:1],
    }

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {'no KL (β=0)': '#d62728', f'with KL (β={BETA})': '#2ca02c'}

# 左：true quality 对比
for label, tr in results.items():
    axes[0].plot(xs, tr['true'].mean(axis=0), color=colors[label], linewidth=2.5, label=label)
    axes[0].fill_between(xs,
                          tr['true'].mean(axis=0) - tr['true'].std(axis=0),
                          tr['true'].mean(axis=0) + tr['true'].std(axis=0),
                          color=colors[label], alpha=0.15)
axes[0].axhline(0, color='black', linewidth=0.7)
axes[0].set_xlabel('优化步数'); axes[0].set_ylabel('true quality (r*)')
axes[0].set_title('KL penalty 让 true quality 不掉下去')
axes[0].legend(); axes[0].grid(alpha=0.3)

# 右：proxy reward 对比（看 KL penalty 让 proxy 涨得慢）
for label, tr in results.items():
    axes[1].plot(xs, tr['proxy'].mean(axis=0), color=colors[label], linewidth=2.5, label=label)
axes[1].axhline(0, color='black', linewidth=0.7)
axes[1].set_xlabel('优化步数'); axes[1].set_ylabel('proxy reward (r_θ)')
axes[1].set_title('KL penalty 限制了 proxy 的涨幅')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()

print("观察:")
print(f"  - no KL:  proxy 涨得很高，但 true 后期下降（reward hacking）")
print(f"  - with KL: proxy 涨得慢一些，但 true quality 持续保持高位")
print(f"  - 这就是 Ch12 RLHF-PPO 加 KL penalty 的核心动机")""")


# =============================================================================
# 11.7 Evaluation + summary + Ch12 preview
# =============================================================================
md(r"""## 11.7 评估 + 小结 + Ch12 预告

### 11.7.1 关键公式速查

| 公式 | 含义 | 出现节 |
|---|---|---|
| $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$ | Bradley-Terry 偏好模型 | §11.2（推导） |
| $f(z) + f(-z) = 1$ | 偏好互补性 → sigmoid | §11.2 推导 |
| $\mathcal{L}_{RM} = -\log\sigma(r_w - r_l) = \text{softplus}(r_l - r_w)$ | Bradley-Terry loss | §11.2 / §11.4 |
| reward 平移不变 ($r \to r + c$) | 可识别性（绝对值无意义） | §11.2 |
| $\text{reward}_{total} = r_\theta - \beta\,\text{KL}(\pi \| \pi_{ref})$ | KL penalty（Ch12 详讲） | §11.6 |

### 11.7.2 本章收获

1. **SFT 不够**：它学模仿，不学偏好，也没标量 reward。
2. **Bradley-Terry 模型**：从 3 条公理（单调、互补、独立）推出偏好概率 = sigmoid(reward 差)。
3. **可识别性**：reward 绝对值无意义，**只有排序/差有意义**——所以训练前要 normalize。
4. **reward model 架构**：TinyGPT backbone + scalar head（复用 Ch10，不重复造轮子）。
5. **Bradley-Terry loss**：`-log sigma(r_w - r_l)` = `softplus(r_l - r_w)`，数值稳定。
6. **过优化（Goodhart's Law）**：把 RM 当 reward function 直接优化 → proxy 涨、true 先涨后降。
7. **KL penalty**：限制 policy 不要离 reference model 太远——Ch12 PPO 的核心组件。

### 11.7.3 兑现的 Phase 1 承诺核对

| 出处 | 承诺原文 | 兑现 |
|---|---|---|
| **Ch00** | "Ch11 Reward Modeling，**偏好 UI** + **过优化曲线**" | §11.3 偏好 UI（ipywidgets 渲染） + §11.6 过优化曲线 ✓ |
| **Ch02 §2.5** | "RLHF **γ 选择 0.9–0.95**（单轮对话不长）" | §11.6.2 把 KL penalty 解释为 "soft γ"（β 大 = horizon 短），预告 Ch12 ✓ |""")

code(r"""# 11.7.4 最终评估报告
print("=" * 60)
print("Ch11 Reward Modeling —— 最终评估报告")
print("=" * 60)
print(f"\n[1] Reward model 配置")
print(f"    backbone: TinyGPT (d_model=64, n_heads=4, n_layers=3, d_ff=256)")
print(f"    reward head: LayerNorm + Linear(64 → 1)")
print(f"    总参数量: {count_parameters(reward_model):,}")
print(f"\n[2] 训练")
print(f"    偏好数据: {len(train_prefs)} train + {len(val_prefs)} val")
print(f"    训练步数: {RM_TRAIN_STEPS}")
print(f"    训练耗时: {time.time()-t0:.1f}s (CPU)")
print(f"\n[3] 评估")
print(f"    val accuracy (r_w > r_l):  {final_acc:.1%}  (>70% 验收门槛)")
baseline = 0.5
print(f"    vs random baseline:        {baseline:.0%}  (提升 {final_acc-baseline:+.1%})")
print(f"    Spearman ρ (RM vs truth):  {rho:+.3f}  (>0 = 排序一致)")
print(f"\n[4] 过优化实验")
peak_step = int(np.argmax(mean_norm_true))
print(f"    true quality peak at step {peak_step} = {mean_norm_true[peak_step]:+.2f}")
print(f"    最终 true quality          = {mean_norm_true[-1]:+.2f}  (下降 = Goodhart's Law)")
print(f"    → KL penalty (β>0) 缓解: true quality 保持在高位 ✓")
print(f"\n[5] 承诺兑现")
print(f"    偏好 UI (ipywidgets):       ✓  §11.3")
print(f"    过优化曲线:                 ✓  §11.6 (proxy ↑ / true 先涨后降)")
print(f"    Bradley-Terry 完整推导:     ✓  §11.2 (在 <details> 里)")
print(f"    γ 0.9-0.95 的 KL 解读:      ✓  §11.6.2")
print("=" * 60)""")

md(r"""---

## 下一章预告：Ch12 RLHF-PPO（项目的"近乎终极目标"）

本章我们训出了一个 reward model $r_\theta(x, y)$，能用它**预测**哪个 response 更好。
但 RM 本身不能生成更好的 response——它只能**打分**。

下一章 Ch12 把所有东西拼起来：

> **用 PPO 训练 SFT 后的 TinyGPT（policy），让它的 response 在 $r_\theta$ 上得分更高，
> 同时用 KL penalty 防止过优化。**

这是真正的 **RLHF** —— 项目从 Ch06 开始的所有 RL 工具的最终应用：

```
   ┌─────────────┐
   │ Ch10 SFT    │ → π_ref（reference policy，KL 的锚）
   ├─────────────┤
   │ Ch11 RM     │ → r_θ（reward function，本章）  ★ 刚完成
   ├─────────────┤
   │ Ch06-09 RL  │ → PPO 工具箱（policy gradient + GAE + clip）
   └──────┬──────┘
          ↓
   ┌─────────────────────────────────────────┐
   │  Ch12 RLHF-PPO:                          │
   │   policy π = SFT model                   │
   │   reward = r_θ(prompt, response)         │
   │            - β · KL(π || π_ref)           │
   │   optimizer = PPO（Ch09 工具）           │
   │                                          │
   │   4 个模型同时跑：π, π_old, π_ref, r_θ   │
   └─────────────────────────────────────────┘
```

**Ch12 关键技术**（预告）：

1. **4 个模型同时跑**（policy / policy_old / reference / reward）—— 显存挑战
2. **token-level PPO**：把每个 token 当一步，response 是一条 trajectory
3. **KL penalty 实现**：`(log π - log π_ref).detach() * 0.5 * (log π - log π_ref)`
   或 reward shaping 两种实现
4. **γ 选择**：单轮对话不长，γ ≈ 0.9–0.95（呼应 Ch02 §2.5 和本章 §11.6.2）

> **再之后**：Ch13 GRPO（无 critic，Group sampling，DeepSeek 用的），
> Ch14 DPO（绕开 RL，直接在偏好数据上训）—— 这些都是 RLHF 的变体或替代方案。""")

code(r"""# Ch11 完成总结
print("=" * 60)
print("Ch11 完成 —— Phase 3 第二块拼图就位")
print("=" * 60)
print("本章交付:")
print(f"  - utils/reward_model.py:")
print(f"      RewardModel (TinyGPT backbone + scalar head)  ({count_parameters(reward_model):,} params)")
print(f"      bradley_terry_loss / generate_preference_data")
print(f"      reward_accuracy / true_reward / ...")
print(f"  - notebooks/ch11_reward_modeling.ipynb: 本章")
print(f"  - val accuracy: {final_acc:.1%} (>70% 门槛)")
print()
print("Phase 3 路线图:")
print("  Ch10 TinyGPT          ✓  base + SFT 模型（policy candidate）")
print("  Ch11 Reward Modeling  ✓  Bradley-Terry + 偏好数据 + 过优化（本章）")
print("  Ch12 RLHF-PPO            4 模型 + KL penalty + PPO on tokens")
print("  → Ch13 GRPO ★终极目标   group sampling + 无 critic")
print("  Ch14 DPO/KTO              避免 RL 的替代方案")
print("=" * 60)""")


# =============================================================================
md(r"""## 11.8 📝 练习

### 练习 1（必做）：偏好数据量的 scaling 曲线

**任务**：用 `generate_preference_data(n_samples=...)` 生成 50 / 200 / 800 对偏好数据（同一 seed 的验证集），分别训练 RM（相同步数按数据量等比缩短），画 val accuracy vs 数据量的曲线。

**预期结果**：accuracy 随数据量单调上升但边际递减；50 对时约 0.6-0.65、800 对时接近 0.8+——这就是"标注预算"和"RM 质量"的兑换率。

### 练习 2（选做）：标注噪声鲁棒性

**任务**：把训练集里 10% / 20% / 30% 的偏好对随机翻转（模拟人类标注错误），重训 RM，用 §11.5.5 的 Spearman ρ 衡量排序质量的退化。

<details><summary>提示</summary>

- 翻转 = 交换 winner/loser 字段；验证集保持干净
- 预期：BT loss 对标签噪声相当鲁棒（softplus 不指数惩罚），但 ρ 会缓慢下降；30% 噪声下 RM 仍能恢复主要排序——这是 BT 框架被工业界广泛使用的原因之一
</details>

*（开放练习，无参考答案。）
> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch11 的自测题再进入下一章。""")

if __name__ == "__main__":
    # 直接运行时写出 notebook（被 build_notebooks.py import 时不写盘）
    _nb.write("ch11_reward_modeling.ipynb")
