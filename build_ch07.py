r"""Build ch07_policy_gradient.ipynb.

一次性脚本：构造 Ch07 笔记本（8 节内容）。运行后产物在 notebooks/。
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch07")
cells = _nb.cells            # build_notebooks.py 适配层读取的模块级列表
md, code = _nb.md, _nb.code


# =============================================================================
# Chapter intro + setup
# =============================================================================

md(r"""# 第 7 章：策略梯度定理 —— 直接优化策略

> **Ch06 DQN 学的是 Q 值**，再 `argmax_a` 间接得到策略。
> **本章反过来**：直接参数化策略 $\pi_\theta(a|s)$，对**策略的目标函数** $J(\theta)$ 求梯度。
>
> **本章核心等式**（RL 中最优雅的结果之一）：
>
> $$\nabla_\theta J(\theta) \;=\; \mathbb{E}_\pi\Big[\,\nabla_\theta \log \pi_\theta(a|s)\cdot Q^\pi(s,a)\,\Big]$$
>
> 它告诉我们：**只要会算 $\nabla_\theta \log \pi$（network 自动微分）和 $Q^\pi$（MC 估计），就能直接优化策略**——完全不需要 Q 网络。

## 学习目标

1. 理解**为什么直接参数化 $\pi$**（DQN 只能处理离散动作 + 确定性策略的两大限制）
2. 掌握 **策略梯度定理** 的完整证明（测度变换 + score function trick）
3. 实现 **REINFORCE**（MC 策略梯度）算法
4. 理解 **高方差问题** 与 **baseline** 为什么不偏（严格证明）
5. 推出 **advantage 形式** $Q - V$（铺垫 Ch08 Actor-Critic）
6. 理解 **on-policy 限制**（复用 Ch05）

## 承接的 Phase 1 承诺（3 处）

| 出处 | 承诺 | 本章兑现节 |
|---|---|---|
| Ch00 学习路径 | "Fast-track: Ch00 → Ch01 → Ch05 → **Ch07** → Ch09 → Ch13" | §7.1 跳读提示 |
| Ch00 章节图 | "策略梯度定理" | §7.3 完整证明 |
| Ch05 章末 | "Phase 1 结束 → Ch07 策略梯度定理" | 全章 |

## 跳读路径（给没读 Ch06 的读者）

本章是 **fast-track 路径** 的一站，理论上只依赖 **Ch01 + Ch05**：

- **Ch01** 给了 RL 的基本概念（探索 vs 利用、样本均值、随机逼近）
- **Ch05** 给了 on/off-policy 的区分、$J(\pi)$ 目标函数、$Q^\pi$ 的定义
- **Ch02** 的 MDP 形式化（$J(\pi) = \mathbb{E}_\pi[\sum_t \gamma^t r_t]$）也会用到
- **Ch06 DQN 不是必需的**——本章从零搭策略网络。但 Ch06 的 PyTorch 基础（`nn.Linear`、训练循环）会让代码看起来更熟悉

**没读 Ch06 的读者**只需要知道：
- PyTorch 的 `nn.Module` 是一个神经网络类，`forward` 定义前向计算
- 梯度由 `loss.backward()` 自动计算（autodiff）
- `torch.optim.Adam` 是梯度下降优化器

下面用到的 `make_mlp` / `QNetwork` 等基础设施在 `utils/networks.py`，本章新加的 `CategoricalPolicy` 在 `utils/policy_networks.py`。

## 术语速查（第一次出现时加脚注，这里给完整列表）

| 术语 | 一句话解释 |
|---|---|
| **策略 (policy)** $\pi(a\|s)$ | 给定状态 $s$，选动作 $a$ 的概率分布 |
| **参数化策略** $\pi_\theta(a\|s)$ | 用参数 $\theta$（神经网络权重）表示的策略 |
| **score function** $\nabla_\theta \log \pi_\theta$ | $\log$ 概率的梯度，也叫"得分函数"，是策略梯度的核心量 |
| **score function trick** | $\nabla_\theta \pi = \pi \cdot \nabla_\theta \log \pi$，把 $\sum \pi f$ 变成 $\mathbb{E}[\nabla\log\pi \cdot f]$ |
| **测度变换 / likelihood ratio trick** | score function trick 的另一名字（因为 $\pi/\pi=1$ 像两个测度的比） |
| **on-policy** | 用当前策略 $\pi_\theta$ 采的数据训练 $\pi_\theta$ 自身（Ch05 §5.10） |
| **trajectory / rollout** | 一个 episode 的 $(s_0, a_0, r_1, s_1, a_1, r_2, \dots)$ 序列 |
| **return** $G_t$ | 从 $t$ 起的累计折扣奖励 $G_t = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}$（Ch02 §2.5） |
| **蒙特卡洛 (MC) 估计** | 用一次 episode 的样本 return 直接估计 $Q^\pi$（Ch04 §4.3） |
| **baseline** $b(s)$ | 减去的一个**只依赖状态**的量，降低方差但不引入偏差 |
| **advantage** $A^\pi(s,a)$ | $Q^\pi(s,a) - V^\pi(s)$，"这个动作比平均水平好多少" |
| **Categorical 分布** | 有限离散动作的分布（PyTorch 里 `torch.distributions.Categorical`） |""")

code("""# 常规设置：找项目根、载入库
import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rlenvs import CartPoleLite
from utils import set_seed
from utils.networks import make_mlp
from utils.policy_networks import CategoricalPolicy
from utils.torch_utils import get_device, count_parameters

set_seed(42)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = get_device()
print(f"PyTorch: {torch.__version__}, device = {DEVICE}")""")

# =============================================================================
# 7.1 从 value 到 policy
# =============================================================================

md(r"""## 7.1 从 value 到 policy：为什么直接参数化 $\pi$

### 7.1.1 DQN 的两大限制

**回顾 Ch06**：DQN 学 $Q_\theta(s, a)$，策略通过 $\arg\max_a Q_\theta(s, a)$ 隐式给出。这种方法叫 **value-based**。它有两个本质限制：

**限制 1：只能处理离散动作**

DQN 架构 A（Ch06 §6.2）网络输出 $|\mathcal{A}|$ 维 Q 值，最后 `argmax` 选动作。但连续动作空间（如机器人关节力矩 $\in \mathbb{R}^6$、车辆方向盘角度 $\in [-\pi, \pi]$）**根本无法枚举**——`argmax` 都做不了。

| 任务 | 动作空间 | DQN 能用？ |
|---|---|---|
| CartPole（左右推） | 离散 {0, 1} | ✓ |
| Atari（按键） | 离散 18 个 | ✓ |
| 围棋（落子位置） | 离散 361 | ✓（AlphaGo） |
| 机器人行走 | 连续 $\mathbb{R}^6$ | ✗ |
| 自动驾驶方向 | 连续 $[-\pi, \pi]$ | ✗ |
| LLM 生成 token | 离散 $\sim 10^5$ 词 | ✓（但通常用 PG） |

**限制 2：学的是确定性策略**（`argmax`）

`argmax` 永远选一个动作——这是**确定性策略**。但有些任务**本质上需要随机策略**：

- **石头剪刀布**：确定性策略会被对手破解（你永远出石头 → 对手永远出布）。**纳什均衡是均匀随机**——$\pi(\text{石头}) = \pi(\text{剪刀}) = \pi(\text{布}) = 1/3$。
- **押宝游戏 / 潜行游戏**：面对会学习的对手，确定性策略会被利用
- **探索**：确定性策略没法探索（除非加 $\epsilon$-greedy 这种 hack）

### 7.1.2 解法：直接参数化 $\pi_\theta(a|s)$

**policy-based** 方法的思路：**直接学一个概率分布** $\pi_\theta(a|s)$，让网络输出"选每个动作的概率"。

**离散动作**：softmax 策略
$$
\pi_\theta(a|s) = \frac{\exp(z_a(s; \theta))}{\sum_{a'} \exp(z_{a'}(s; \theta))}
$$
其中 $z(s; \theta) \in \mathbb{R}^{|\mathcal{A}|}$ 是网络的 **logits** 输出。

**连续动作**：高斯策略（Ch08/Ch09 用）
$$
\pi_\theta(a|s) = \mathcal{N}\big(a;\, \mu_\theta(s),\, \sigma_\theta^2\big)
$$
其中 $\mu_\theta(s)$ 是网络输出的均值（$\sigma$ 可学或固定）。

### 7.1.3 policy-based 的优缺点

| | value-based (DQN) | policy-based |
|---|---|---|
| 离散动作 | ✓ | ✓ |
| 连续动作 | ✗ | ✓ |
| 随机策略 | ✗（需 $\epsilon$-greedy） | ✓（天然的） |
| 样本效率 | 高（off-policy） | 低（on-policy，§7.7） |
| 方差 | 低 | **高**（§7.5） |
| 收敛性 | 不稳（致命三件套） | 通常更稳（无 bootstrap） |

**核心权衡**：policy-based 用**样本效率**换**表达力**和**稳定性**。LLM RLHF 几乎全部用 policy-based（PPO/GRPO），原因之一就是连续的 token 概率分布天然适合直接参数化。

### 7.1.4 本章用到的策略网络

我们在 `utils/policy_networks.py` 提供了 `CategoricalPolicy`（softmax 策略网络）。来看一眼：""")

code("""# 演示 CategoricalPolicy 的基本用法
torch.manual_seed(0)

policy = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[64, 64])
print(f"参数量: {count_parameters(policy)}")

# 单个状态
s = torch.tensor([0.1, 0.0, 0.05, 0.2], dtype=torch.float32)
print(f"\\n状态 s = {s.tolist()}")

# forward 返回 Categorical 分布
dist = policy(s.unsqueeze(0))
probs = dist.probs[0]
print(f"π(left|s)  = {probs[0].item():.4f}")
print(f"π(right|s) = {probs[1].item():.4f}")
print(f"sum = {probs.sum().item():.4f}  （softmax 保证归一）")

# 采样动作
actions = [policy.act(s) for _ in range(1000)]
print(f"\\n采样 1000 次: P(left)={actions.count(0)/1000:.3f}, P(right)={actions.count(1)/1000:.3f}")
print(f"（与上面的解析概率接近，因为大数律）")

# 关键：log_prob —— 策略梯度定理的核心量
a = torch.tensor([1])
print(f"\\nlog π(a=1|s) = {dist.log_prob(a).item():.4f}")
print(f"exp(logp) = {dist.log_prob(a).exp().item():.4f}  （应等于 probs[1]）")

# 梯度：score function 自动可微
dist = policy(s.unsqueeze(0))
logp = dist.log_prob(a)
logp.backward()
grad_norm = sum((p.grad ** 2).sum() for p in policy.parameters()).sqrt()
print(f"\\n∇_θ log π(a=1|s) 的 L2 范数 = {grad_norm.item():.4f}")
msg = "  → 这就是 score function，策略梯度的核心量"
print(msg)""")

# =============================================================================
# 7.2 目标函数 J(theta)
# =============================================================================

md(r"""## 7.2 目标函数 $J(\theta)$

### 7.2.1 复用 Ch02 的 $J(\pi)$

**回顾 Ch02 §2.6**：对策略 $\pi$，目标函数（discounted episodic return）为

$$
J(\pi) := \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right] = \mathbb{E}_\pi\!\left[G_0\right]
$$

现在把 $\pi$ 换成参数化的 $\pi_\theta$：

$$
\boxed{\; J(\theta) := \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right] \;}
$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $\theta$ | 策略网络参数（PyTorch `nn.Linear` 的权重 + bias） |
| $\mathbb{E}_{\pi_\theta}$ | 期望在 $\pi_\theta$ 引发的轨迹分布上取（详见下方展开） |
| $\gamma \in [0, 1)$ | 折扣因子（复用 Ch02） |
| $R_{t+1}$ | 第 $t$ 步行动后的奖励 |

### 7.2.2 轨迹分布的精确展开

"期望在 $\pi_\theta$ 上"是什么意思？把所有随机性显式写出来——MDP 的轨迹分布是

$$
\Pr_\theta(\tau) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_0, \dots, a_{t-1}, s_t) \cdot p(s_{t+1}|s_t, a_t)
$$

**逐项解读**：

| 项 | 含义 |
|---|---|
| $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots)$ | 一条 episode 轨迹 |
| $\rho_0(s_0)$ | 初始状态分布（CartPole 是 $\pm 0.05$ 均匀） |
| $\pi_\theta(a_t\|s_t)$ | 我们的策略网络（假设马尔可夫，只看当前 $s_t$） |
| $p(s_{t+1}\|s_t, a_t)$ | 环境的转移概率（CartPole 的物理方程） |

于是

$$
J(\theta) = \sum_\tau \Pr_\theta(\tau) \cdot R(\tau), \qquad R(\tau) := \sum_{t=0}^{T-1}\gamma^t r_{t+1}
$$

**关键观察**：$\Pr_\theta(\tau)$ **显式依赖 $\theta$**（通过 $\pi_\theta$ 的连乘）。这就让 $\nabla_\theta J$ 看起来很难——因为 $\theta$ 在概率里。

### 7.2.3 平均奖励版本（旁注，本章不用）

对于**没有自然 episode 边界**的任务（如连续控制），常用**平均奖励**目标：

$$
J_{\text{avg}}(\theta) = \lim_{T \to \infty} \frac{1}{T}\mathbb{E}_\pi\!\left[\sum_{t=0}^{T-1} R_{t+1}\right]
$$

策略梯度定理在两种设定下都成立。本章只用 episodic 版本（CartPole 每 episode 最多 500 步）。

### 7.2.4 想优化 $J(\theta)$，需要 $\nabla_\theta J$

梯度下降要求我们能算 $\nabla_\theta J(\theta)$。但 $J(\theta)$ 是个期望，里面是 $\Pr_\theta(\tau) \cdot R(\tau)$，$\theta$ 在概率里——直接求导看起来很恶心。

**下一节的策略梯度定理**给出一个让人惊讶的简洁答案：**只需要 $\nabla_\theta \log \pi_\theta$ 和 $R(\tau)$**——$\theta$ 在概率里的难题被一个技巧（score function trick）一举化解。""")

# =============================================================================
# 7.3 策略梯度定理
# =============================================================================

md(r"""## 7.3 策略梯度定理（**核心推导**）

### 7.3.1 定理陈述

**策略梯度定理**（Silver 2014 lecture；Sutton et al. 1999；Williams 1992 REINFORCE）：

$$
\boxed{\;\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\pi_\theta}\!\Big[\,\nabla_\theta \log \pi_\theta(a|s)\cdot Q^{\pi_\theta}(s,a)\,\Big]\;}
$$

**逐项解读**：

| 项 | 含义 |
|---|---|
| $\nabla_\theta J(\theta)$ | 目标函数对参数的梯度 |
| $\mathbb{E}_{\pi_\theta}$ | 期望在**状态分布** $d^{\pi_\theta}(s)$ 和**动作分布** $\pi_\theta(a\|s)$ 上取 |
| $\nabla_\theta \log \pi_\theta(a\|s)$ | **score function**（$\log \pi$ 的梯度） |
| $Q^{\pi_\theta}(s, a)$ | 当前策略下的 action-value（Ch02 §2.5 定义） |

**优雅之处**：梯度形如 $\mathbb{E}[\text{自动可微的量} \times \text{可估计的量}]$——神经网络自动算 $\nabla \log \pi$，MC 样本估 $Q^\pi$。

下面分两步严格证明：**（A）简版**（trajectory 形式，直觉），**（B）通用版**（含状态分布，正式）。

### 7.3.2 简版证明：trajectory 形式

我们先证明一个等价的 trajectory 形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[\nabla_\theta \log \Pr_\theta(\tau) \cdot R(\tau)\Big]
$$

这个形式更直观（一条轨迹一个权重），但本质上是通用定理的特例。

<details>
<summary><b>完整证明：trajectory 形式（点开看）</b></summary>

**Step 1**：写出 $J(\theta)$ 的 trajectory 形式

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int \Pr_\theta(\tau) R(\tau)\, d\tau
$$

**Step 2**：对 $\theta$ 求梯度（梯度穿过积分号）

$$
\nabla_\theta J(\theta) = \int \nabla_\theta \Pr_\theta(\tau) \cdot R(\tau)\, d\tau
$$

**Step 3**：**关键技巧**——用 $\nabla_\theta \Pr_\theta = \Pr_\theta \cdot \nabla_\theta \log \Pr_\theta$（**这是 score function trick 的核心恒等式**，由链式法则立刻得到：$\nabla \log p = \nabla p / p$ 所以 $\nabla p = p \nabla \log p$）。

$$
\nabla_\theta J(\theta) = \int \Pr_\theta(\tau) \cdot \nabla_\theta \log \Pr_\theta(\tau) \cdot R(\tau)\, d\tau = \mathbb{E}_{\tau \sim \pi_\theta}\Big[\nabla_\theta \log \Pr_\theta(\tau) \cdot R(\tau)\Big]
$$

**这一步叫"测度变换"**（measure change）——把 $\nabla p$ 这种"概率密度的梯度"（很难直接处理）换成 $\nabla \log p \cdot p$，再写成期望（可以采样估计）。

**Step 4**：展开 $\log \Pr_\theta(\tau)$。回顾 §7.2.2

$$
\Pr_\theta(\tau) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)
$$

取 log：

$$
\log \Pr_\theta(\tau) = \log \rho_0(s_0) + \sum_{t=0}^{T-1} \big[\log \pi_\theta(a_t|s_t) + \log p(s_{t+1}|s_t, a_t)\big]
$$

**Step 5**：对 $\theta$ 求梯度。

**关键观察**：$\log \rho_0$、$\log p(s_{t+1}|s_t, a_t)$ 都**不依赖 $\theta$**（初始分布和环境动态是固定的）。只有 $\log \pi_\theta$ 依赖。所以

$$
\nabla_\theta \log \Pr_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

**奇迹出现了**：环境的复杂动态 $p(s'\|s, a)$ 完全消失！我们**不需要知道环境模型**——这就是为什么策略梯度是 **model-free** 的。

**Step 6**：代入 Step 3 的结果

$$
\nabla_\theta J(\theta) = \mathbb{E}_\tau\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]
$$

这就完成了 trajectory 形式的证明。$\blacksquare$

</details>

**直觉解读**：
- $R(\tau)$ 大（好轨迹）→ $\nabla \log \pi$ 朝着"增大这条轨迹上所有动作概率"的方向
- $R(\tau)$ 小（差轨迹）→ 朝着"减小"的方向
- 期望下来，$\theta$ 沿"提高好动作概率、降低差动作概率"的方向移动

### 7.3.3 通用版证明：$Q^\pi$ 形式

trajectory 形式对每条轨迹整体加权——但**同一条轨迹上的所有动作都用同一个 $R(\tau)$ 加权**。这"对早期的动作不公平"：早期动作的好坏不该由整条轨迹的总回报来评判（后期动作的运气也混进来了）。

**通用形式**用 $Q^\pi(s_t, a_t)$（从 $(s_t, a_t)$ 起的期望 return）替换 $R(\tau)$，让每个动作只对自己**之后**的回报负责：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
$$

其中 $G_t = \sum_{k=t}^{T-1}\gamma^{k-t} R_{k+1}$ 是从 $t$ 起的折扣 return（Ch02 §2.5）。注意 $G_t$ 是 $Q^\pi(s_t, a_t)$ 的 MC 估计。

<details>
<summary><b>完整证明：通用 $Q^\pi$ 形式（点开看）</b></summary>

**简化处理**：我们用 discounted episodic 设定（即 $J(\theta) = \mathbb{E}[\sum_t \gamma^t R_{t+1}]$，$\gamma < 1$）。

**Step 1**：定义 discounted state visitation distribution

$$
d^{\pi_\theta}(s) := \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s | s_0 \sim \rho_0, \pi_\theta)
$$

这表示"从 $\rho_0$ 出发，按 $\pi_\theta$ 走，所有时间步上访问 $s$ 的折扣累计概率"。

**Step 2**：$J(\theta)$ 改写。由 Ch02 §2.6 的标准推导

$$
J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(a|s) Q^{\pi_\theta}(s, a)
$$

**Step 3**：对 $\theta$ 求梯度。注意 $Q^{\pi_\theta}$ 和 $d^{\pi_\theta}$ 都依赖 $\theta$——理论上要全求导。但 **策略梯度定理的精妙在于：只需要对显式出现的 $\pi_\theta(a|s)$ 求导，$Q^{\pi_\theta}$ 和 $d^{\pi_\theta}$ 的隐式依赖可以忽略**。

<details>
<summary>为什么可以忽略 $Q$ 和 $d$ 对 $\theta$ 的依赖？（进阶，点开看）</summary>

这是一个**简化证明**，完整版见 Sutton et al. (1999) "Policy Gradient Methods for RL with Parametrized Actions"。

**关键事实**：在 discounted episodic 设定下，可以证明 $\sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s,a)$ 已经包含了**所有对 $\theta$ 的一阶依赖**——$Q$ 和 $d$ 的隐式依赖**正好相消**。

**直觉**：$Q^{\pi_\theta}(s, a) = \mathbb{E}[\sum_k \gamma^k R_{t+k+1} | s_t=s, a_t=a, \pi_\theta]$ 依赖 $\theta$（因为未来动作按 $\pi_\theta$ 选）。但这个未来依赖**已经被 $d^{\pi_\theta}(s')$ 在 $\sum_{s'}$ 中"吸收"了**——所以再算一次会重复计算。

**严格论证**：用 Bellman 期望方程 $Q^\pi(s,a) = \sum_{s'} p(s'|s,a)\sum_{a'} \pi(a'|s') [r + \gamma Q^\pi(s', a')]$ 展开，可以看到 $\theta$ 在 $\pi$ 里出现的次数**正好是 discounted state visitation $d^{\pi_\theta}$ 的定义里 $\pi$ 出现的次数**——所以二者的 $\theta$ 依赖是同一个东西的两种写法。

实践中这等价于"我们只关心 $\pi$ 直接出现的位置"，结果和完整证明一致。

</details>

**Step 4**：对显式 $\pi_\theta(a|s)$ 求梯度

$$
\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a)
$$

**Step 5**：**score function trick**——$\nabla \pi = \pi \cdot \nabla \log \pi$

$$
\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a)
$$

**Step 6**：把 $\sum_a \pi_\theta(a|s) [\dots]$ 识别为 $\mathbb{E}_{a \sim \pi_\theta}[\dots]$

$$
\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\!\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a)\Big]
$$

**Step 7**：把 $\sum_s d^{\pi_\theta}(s) \mathbb{E}_{a \sim \pi_\theta}[\dots]$ 识别为 $\mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}[\dots]$

$$
\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta(\cdot|s)}\!\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a)\Big]\;}
$$

这就是定理。$\blacksquare$

</details>

### 7.3.4 数值验证：score function trick 在小例子里对吗？

在我们用 PyTorch 的 autodiff 实现 $\nabla \log \pi$ 之前，先用一个**1 维 toy 例子**手算一遍 score function trick，确认数学对。""")

code("""# Toy 验证：score function trick ∇π = π · ∇log π 在数值上对吗？
# 用 2 动作 softmax，θ 是单个 logit 偏移

# 设 logits = [θ, 0]，π(0) = softmax(θ, 0)[0] = e^θ / (e^θ + 1)
# 解析：log π(0|θ) = θ - log(e^θ + 1)
# ∇_θ log π(0|θ) = 1 - e^θ / (e^θ + 1) = 1 - π(0|θ)
# ∇_θ log π(1|θ) = 0 - e^θ / (e^θ + 1) = -π(0|θ)

import numpy as np

def softmax_pi0(theta):
    return np.exp(theta) / (np.exp(theta) + 1.0)

def analytic_score(theta, a):
    p0 = softmax_pi0(theta)
    if a == 0:
        return 1 - p0
    else:
        return -p0

# 用 PyTorch autodiff 验证
thetas = [-2.0, -0.5, 0.0, 0.5, 2.0]
print(f"{'theta':>8} {'a':>3} {'analytic':>12} {'autograd':>12} {'diff':>12}")
for theta_val in thetas:
    theta = torch.tensor(theta_val, requires_grad=True)
    for a in [0, 1]:
        # 每个 (theta, a) 重新建计算图，避免 backward 释放后被复用
        logits = torch.stack([theta, torch.tensor(0.0)])
        dist = Categorical(logits=logits)
        logp = dist.log_prob(torch.tensor(a))
        logp.backward()
        autograd_score = theta.grad.item()
        analytic = analytic_score(theta_val, a)
        print(f"{theta_val:>8.2f} {a:>3} {analytic:>12.6f} {autograd_score:>12.6f} {abs(autograd_score-analytic):>12.2e}")
        theta.grad.zero_()

msg = "\\n→ autodiff 与解析公式完全一致，验证 score function trick 正确。"
print(msg)""")

code("""# 完整验证：策略梯度定理的 MC 估计与真实有限差分梯度一致吗？
# 用一个 1 维状态、2 动作的简化问题：s 固定，奖励 = (a == 0)
# 即最优策略是 π(0) = 1

torch.manual_seed(0)
np.random.seed(0)

theta = torch.tensor([0.5, -0.3], requires_grad=True)  # 2 维 logits

def get_pi(theta):
    # 返回 (π(0), π(1)) 概率
    e = np.exp(theta.detach().numpy())
    return e / e.sum()

def expected_return(theta):
    # s 固定，按 π_θ 采样 a，奖励 = (a == 0). J(θ) = π(0)
    return get_pi(theta)[0]

# 策略梯度定理预测：∇_θ J = E[∇_θ log π(a|s) · Q(s,a)]
# 这里 Q(s,a) = 1 if a=0 else 0
# ∇_θ J = π(0) · ∇log π(0) · 1 + π(1) · ∇log π(1) · 0 = π(0) · ∇log π(0)
J_val = expected_return(theta)
score_a0 = torch.tensor(get_pi(theta)[0])  # 加权

dist = Categorical(logits=theta)
logp_a0 = dist.log_prob(torch.tensor(0))
logp_a0.backward()
pg_pred = theta.grad.clone() * score_a0  # π(0) · ∇log π(0)

# 解析梯度：J = π(0) = e^{θ_0} / (e^{θ_0} + e^{θ_1})
# ∂J/∂θ_0 = π(0)·π(1), ∂J/∂θ_1 = -π(0)·π(1)
p0, p1 = get_pi(theta)
analytic = torch.tensor([p0 * p1, -p0 * p1])

# 数值有限差分
eps = 1e-5
fd_grad = torch.zeros(2)
for i in range(2):
    tp = theta.detach().clone(); tp[i] += eps
    tm = theta.detach().clone(); tm[i] -= eps
    fd_grad[i] = float((expected_return(tp) - expected_return(tm)) / (2 * eps))

print(f"J(θ) = π(0) = {p0:.6f}")
print(f"\\n∇_θ J 各方法对比：")
print(f"  策略梯度定理 (π·∇log π):  [{pg_pred[0]:+.6f}, {pg_pred[1]:+.6f}]")
print(f"  解析 (π0·π1, -π0·π1):     [{analytic[0]:+.6f}, {analytic[1]:+.6f}]")
print(f"  有限差分:                  [{fd_grad[0]:+.6f}, {fd_grad[1]:+.6f}]")
msg = "\\n→ 三者一致，策略梯度定理正确。"
print(msg)""")

# =============================================================================
# 7.4 REINFORCE
# =============================================================================

md(r"""## 7.4 REINFORCE 算法（MC 策略梯度）

### 7.4.1 从定理到算法

策略梯度定理给出 $\nabla_\theta J = \mathbb{E}[\nabla \log \pi \cdot Q^\pi]$，但 $Q^\pi$ 是期望——我们没法直接算。**REINFORCE**（Williams 1992）用 **MC 样本 return** 估计 $Q^\pi$：

$$
Q^{\pi_\theta}(s_t, a_t) \approx G_t := \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}
$$

即用 episode 实际收到的折扣累计奖励作为 $Q^\pi$ 的无偏估计（Ch04 §4.3 MC 方法）。

### 7.4.2 REINFORCE 完整算法

```
初始化策略网络 π_θ（随机权重）
for episode = 1, 2, ...:
    1. 用 π_θ 跑一整条 episode，收集 (s_0, a_0, r_1, s_1, a_1, ..., s_{T-1}, a_{T-1}, r_T)
    2. 对每个 t = 0, ..., T-1，计算 G_t = Σ_{k=t}^{T-1} γ^{k-t} r_{k+1}
    3. loss = -Σ_t log π_θ(a_t|s_t) · G_t   ← 一个 episode 一起更新
    4. θ ← θ + α · ∇_θ (loss 的相反数)   ← 梯度上升 J
```

**逐项解读**：

| 项 | 含义 |
|---|---|
| $G_t$ | 从 $t$ 起的实际 return（MC 估计 $Q^\pi$） |
| $-\log \pi_\theta(a_t\|s_t) \cdot G_t$ | 单步 loss（**取负号**因为 PyTorch 默认做**梯度下降** loss，而我们要**梯度上升** $J$） |
| $\sum_t$ | 整条 episode 所有 step 加起来一起 backward |

**为什么用负号**：定理给出 $\nabla J$，我们要做 $+ \alpha \nabla J$（梯度上升）。但 `loss.backward()` 给出 $\nabla \text{loss}$，优化器做 $\theta - \alpha \nabla \text{loss}$。设 `loss = -J` 就让 $\theta - \alpha \nabla (-J) = \theta + \alpha \nabla J$ ✓。

### 7.4.3 完整实现

我们来在 CartPoleLite 上实现 REINFORCE。""")

code("""def compute_returns(rewards, gamma):
    \"\"\"从奖励列表计算 G_t = Σ_{k=t}^{T-1} γ^{k-t} r_{k+1}（从后往前累积）.

    返回 list[float]，长度与 rewards 相同。
    \"\"\"
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# 单元测试
r = [1, 1, 1, 1, 1]
g = compute_returns(r, gamma=0.9)
print(f"rewards = {r}")
print(f"G_t (γ=0.9) = {[f'{x:.3f}' for x in g]}")
# 手算：G_4 = 1, G_3 = 1 + 0.9·1 = 1.9, G_2 = 1 + 0.9·1.9 = 2.71, ...
expected = [1 + 0.9 + 0.81 + 0.729 + 0.6561,
            1 + 0.9 + 0.81 + 0.729,
            1 + 0.9 + 0.81,
            1 + 0.9, 1]
print(f"expected   = {[f'{x:.3f}' for x in expected]}")
assert np.allclose(g, expected), "compute_returns 错了"
print("✓ compute_returns 正确")""")


code("""def collect_episode(env, policy, gamma, device='cpu'):
    \"\"\"用 policy 跑一整条 episode，返回 (states, actions, returns, log_probs, sum_r).

    使用 PyTorch autodiff 的计算图（保留 log_prob），方便之后 backward。
    \"\"\"
    states = []
    actions = []
    rewards = []
    log_probs = []  # 每步的 log π(a_t|s_t)，保留梯度

    s = env.reset()
    done = False
    while not done:
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        dist = policy(s_t)
        a = dist.sample()
        logp = dist.log_prob(a)

        a_int = int(a.item())
        s_next, r, done, _ = env.step(a_int)

        states.append(s)
        actions.append(a_int)
        rewards.append(r)
        log_probs.append(logp)

        s = s_next

    returns = compute_returns(rewards, gamma)
    sum_r = sum(rewards)

    # 转 tensor
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    log_probs_t = torch.stack(log_probs).squeeze(-1)  # [T]
    return states, actions, returns_t, log_probs_t, sum_r


# 在没训练的策略上跑一条 episode，看输出
torch.manual_seed(0)
env = CartPoleLite(seed=0, max_steps=500)
policy = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[128, 128])

states, actions, returns, log_probs, total_r = collect_episode(env, policy, gamma=0.99)
print(f"episode 长度 = {len(actions)}")
print(f"total reward = {total_r}")
print(f"G_0 = {returns[0].item():.2f}  （从 t=0 起的折扣 return）")
print(f"log_probs 形状 = {log_probs.shape}, 是否在计算图里 = {log_probs.requires_grad}")""")


code("""def train_reinforce(
    env, policy,
    n_episodes=500,
    gamma=0.99,
    lr=1e-3,
    use_baseline=False,           # 7.5 节会开启
    baseline=None,                # callable: states -> b(s) tensor
    seed=42,
    verbose=True,
    print_every=50,
    device='cpu',
):
    \"\"\"REINFORCE（含可选 baseline）.

    Returns
    -------
    dict with 'episode_rewards', 'episode_losses', 'grad_norms'
    \"\"\"
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []
    episode_losses = []
    grad_norms = []

    for ep in range(n_episodes):
        states, actions, returns, log_probs, total_r = collect_episode(
            env, policy, gamma, device=device
        )

        # 计算"信号"——baseline 之前是 G_t，之后是 G_t - b(s)
        if use_baseline and baseline is not None:
            s_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
            b = baseline(s_tensor).squeeze(-1).detach()  # baseline 不参与 actor 梯度
            signal = returns - b
        else:
            signal = returns

        # 策略梯度 loss = -mean( log π(a_t|s_t) · signal_t )
        # 用 mean 而不是 sum 让 lr 在不同 episode 长度下保持稳定
        loss = -(log_probs * signal).mean()

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（防爆）
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0).item()
        optimizer.step()

        episode_rewards.append(total_r)
        episode_losses.append(loss.item())
        grad_norms.append(gnorm)

        if verbose and ep % print_every == 0:
            avg_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"ep {ep:>3} | reward={total_r:>5.0f} | avg(10)={avg_r:>5.1f} | "
                  f"loss={loss.item():>+8.3f} | gnorm={gnorm:.2f}")

    return dict(
        episode_rewards=np.array(episode_rewards),
        episode_losses=np.array(episode_losses),
        grad_norms=np.array(grad_norms),
    )


# 训练 REINFORCE
torch.manual_seed(42)
np.random.seed(42)
env = CartPoleLite(seed=0, max_steps=500)
reinforce_policy = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[128, 128])
print(f"参数量: {count_parameters(reinforce_policy)}")
print("开始训练 REINFORCE ...")
reinforce_metrics = train_reinforce(
    env, reinforce_policy, n_episodes=400, gamma=0.99, lr=1e-3, seed=42, print_every=50
)
last_10 = reinforce_metrics['episode_rewards'][-10:].mean()
print(f"\\n训练完成。最后 10 episodes 平均 reward: {last_10:.1f}")""")


code("""# 画 REINFORCE 训练曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# 1. episode reward
ax = axes[0]
ax.plot(reinforce_metrics['episode_rewards'], color='#9ec5e8', alpha=0.5, label='raw')
if len(reinforce_metrics['episode_rewards']) > 20:
    sm = np.convolve(reinforce_metrics['episode_rewards'], np.ones(20)/20, mode='valid')
    ax.plot(sm, color='crimson', linewidth=2, label='smoothed (w=20)')
ax.axhline(500, color='g', linestyle='--', alpha=0.5, label='max (500)')
ax.axhline(100, color='orange', linestyle='--', alpha=0.5, label='验收线 100')
ax.set_xlabel('episode'); ax.set_ylabel('reward')
ax.set_title('REINFORCE 训练曲线'); ax.legend(); ax.grid(alpha=0.3)

# 2. loss
ax = axes[1]
ax.plot(reinforce_metrics['episode_losses'], alpha=0.5)
ax.set_xlabel('episode'); ax.set_ylabel('policy loss')
ax.set_title('REINFORCE loss'); ax.grid(alpha=0.3)

# 3. 梯度范数
ax = axes[2]
ax.plot(reinforce_metrics['grad_norms'], alpha=0.5)
ax.set_xlabel('episode'); ax.set_ylabel('gradient norm')
ax.set_title('梯度范数（看方差有多大）'); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()

print(f"梯度范数：mean={reinforce_metrics['grad_norms'].mean():.2f}, "
      f"std={reinforce_metrics['grad_norms'].std():.2f}")
print(f"  → 高 std 反映策略梯度的高方差问题（下一节解决）")""")


md(r"""### 7.4.4 评估训练后的策略

训练完用 `deterministic=True`（取 argmax）评估，看 agent 能撑多久。""")

code("""# 评估
def evaluate_policy(env, policy, n_episodes=5, deterministic=False, seed=0):
    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        total_r = 0
        done = False
        while not done:
            s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                with torch.no_grad():
                    a = int(policy(s_t).logits.argmax(dim=1).item())
            else:
                a = policy.act(s_t.squeeze(0))
            s, r, done, _ = env.step(a)
            total_r += r
        rewards.append(total_r)
    return rewards

# 注意：评估用同一个 seed 看 deterministic 策略的表现
np.random.seed(0)
env_eval = CartPoleLite(seed=0, max_steps=500)
rewards_det = evaluate_policy(env_eval, reinforce_policy, n_episodes=5, deterministic=True)
rewards_sto = evaluate_policy(env_eval, reinforce_policy, n_episodes=5, deterministic=False)
print(f"评估（deterministic, argmax）: rewards = {rewards_det}, mean = {np.mean(rewards_det):.1f}")
print(f"评估（stochastic, 按 π_θ）:    rewards = {rewards_sto}, mean = {np.mean(rewards_sto):.1f}")
print(f"\\n注：CartPole 的确定性最优策略存在（左右推到位即可），")
print(f"所以 deterministic 通常比 stochastic 表现更稳。")""")

# =============================================================================
# 7.5 Baseline
# =============================================================================

md(r"""## 7.5 高方差问题与 baseline

### 7.5.1 高方差的根源

REINFORCE 用 $G_t$（一整条 episode 的 return）估计 $Q^\pi(s_t, a_t)$。但 $G_t$ 的**方差非常大**——它包含了 $t$ 以后所有步的随机性（环境动态、未来动作的探索）。

**例子**：CartPole 上一个 episode 有 200 步。$G_0$ 包含 200 个奖励，每个都有噪声。即使策略很好，某次 episode 也可能因为早期一个 bad luck 导致 $G_0$ 很低；反之亦然。

**数学上**：$\text{Var}(G_t) = \sum_{k=t}^{T-1} \gamma^{2(k-t)} \text{Var}(R_{k+1}) + \text{cross terms}$——对独立噪声简化为 $\approx \sigma_r^2 / (1 - \gamma^2)$，对 $\gamma = 0.99$ 这个值是 $\approx 50 \cdot \sigma_r^2$。

**后果**：梯度 $\nabla \log \pi \cdot G_t$ 的方差爆炸，SGD 收敛极慢。这就是为什么 REINFORCE 实践中需要**几百到几千 episodes** 才能稳定。

### 7.5.2 解法：减一个 baseline

**核心想法**：把"绝对好"换成"相对好"——减去一个**只依赖状态 $s$** 的参考值 $b(s)$（baseline）

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot \big(Q^\pi(s,a) - b(s)\big)\Big]
$$

**直觉**：$b(s) \approx V^\pi(s)$（该状态的"平均"价值），那 $Q^\pi(s, a) - b(s) \approx A^\pi(s, a)$（advantage，§7.6）。

- 如果 $Q^\pi(s, a) > V^\pi(s)$ → $A > 0$ → 增大 $\pi(a|s)$
- 如果 $Q^\pi(s, a) < V^\pi(s)$ → $A < 0$ → 减小 $\pi(a|s)$

**信号从"绝对大小"变成"相对好坏"**，方差大幅下降（因为 $V^\pi$ 已经吸收了大部分状态相关的方差）。

### 7.5.3 baseline 为什么不偏（关键证明）

减去 baseline 会不会让梯度**不再指向** $\nabla J$？答案是不会——baseline 项的期望**正好为 0**：

$$
\mathbb{E}_{a \sim \pi_\theta}\!\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\Big] = 0
$$

<details>
<summary><b>完整证明：baseline 不引入偏差（点开看）</b></summary>

**定理**：对任何**只依赖 $s$**（不依赖 $a$）的函数 $b(s)$，

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\!\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\Big] = 0
$$

**证明**：

$$
\begin{aligned}
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\!\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\Big]
&= \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \\
&= b(s) \cdot \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)
\end{aligned}
$$

**关键步骤**：用 **score function 恒等式** $\nabla \log \pi = \nabla \pi / \pi$

$$
\pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) = \pi_\theta(a|s) \cdot \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \nabla_\theta \pi_\theta(a|s)
$$

代入：

$$
\sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) = \sum_a \nabla_\theta \pi_\theta(a|s) = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0
$$

最后一步用 $\sum_a \pi_\theta(a|s) = 1$（概率归一化），它的梯度当然是 0。

**结论**：$b(s) \cdot 0 = 0$。$\blacksquare$

</details>

**核心洞见**：baseline 不偏的关键是"**$b(s)$ 与 $a$ 无关**"——它不能"偏向"任何动作。这就是为什么 baseline **只依赖 $s$** 是充分必要条件（如果 $b$ 依赖 $a$，证明立刻失效）。

### 7.5.4 最优 baseline（理论）

理论上**最小化方差**的 baseline 是（信号加权的 mean）

$$
b^*(s) = \frac{\mathbb{E}_{a \sim \pi}\big[(\nabla_\theta \log \pi)^2 \cdot Q^\pi(s, a)\big]}{\mathbb{E}_{a \sim \pi}\big[(\nabla_\theta \log \pi)^2\big]}
$$

实践中通常简化为 $b(s) \approx V^\pi(s)$（state value，与梯度无关）。这就引出下一节的 advantage 形式。

### 7.5.5 实践：用 $V$ 网络做 baseline

下面我们用一个**简单的 $V_\phi$ 网络**作为 baseline（**注意：这里只是用它做减法，不做 critic 的梯度学习——那是 Ch08 的事**）。为简化，我们用一个"running mean of returns"作为 baseline——一种 state-independent 的简化版，用来直接展示 baseline 的降方差效果。""")

code("""# 实验：对比 REINFORCE 有无 baseline 的方差
# baseline 选最简形式：b = running mean of G_t（state-independent，但 episode-dependent）

torch.manual_seed(42)
np.random.seed(42)

# 跑 5 个 seeds 看方差
seeds = [0, 1, 2, 3, 4]
n_episodes = 300

reinforce_runs = []
reinforce_baseline_runs = []

for seed in seeds:
    env = CartPoleLite(seed=seed, max_steps=500)

    # 1. 无 baseline
    policy = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[128, 128])
    metrics = train_reinforce(env, policy, n_episodes=n_episodes, lr=1e-3,
                              seed=seed, verbose=False)
    reinforce_runs.append(metrics['episode_rewards'])

    # 2. 有 baseline（用当前 episode 的 G_t 均值——state-independent）
    # baseline 是一个简单函数：输入 states，输出它们的"平均值" tensor
    def simple_baseline(states_t, return_mean_cache=[0.0]):
        # 用当前 episode 所有 G_t 的均值作为常数 baseline（每 episode 重设）
        # 这里返回的就是平均值的向量——具体值在 train_reinforce_baseline 里设置
        n = states_t.shape[0]
        return torch.full((n,), return_mean_cache[0])

    # 简化版：直接 inline 一个轻量训练循环，每 episode 用 G_t 均值作 baseline
    env = CartPoleLite(seed=seed, max_steps=500)
    policy_b = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[128, 128])
    opt = torch.optim.Adam(policy_b.parameters(), lr=1e-3)
    ep_rewards_b = []
    for ep in range(n_episodes):
        states, actions, returns, log_probs, total_r = collect_episode(env, policy_b, gamma=0.99)
        b = returns.mean().detach()  # state-independent baseline = E[G_t] of this episode
        signal = returns - b
        loss = -(log_probs * signal).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_b.parameters(), 10.0)
        opt.step()
        ep_rewards_b.append(total_r)
    reinforce_baseline_runs.append(np.array(ep_rewards_b))

reinforce_runs = np.array(reinforce_runs)
reinforce_baseline_runs = np.array(reinforce_baseline_runs)

print(f"\\n5 seeds × {n_episodes} episodes 的结果：")
print(f"  REINFORCE         最后 10 ep mean: {reinforce_runs[:, -10:].mean():.1f}, "
      f"std across seeds: {reinforce_runs[:, -10:].mean(axis=1).std():.1f}")
print(f"  REINFORCE+base    最后 10 ep mean: {reinforce_baseline_runs[:, -10:].mean():.1f}, "
      f"std across seeds: {reinforce_baseline_runs[:, -10:].mean(axis=1).std():.1f}")""")


code("""# 画对比
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# 1. 训练曲线
ax = axes[0]
mean_r = reinforce_runs.mean(axis=0)
mean_rb = reinforce_baseline_runs.mean(axis=0)
if len(mean_r) > 20:
    sm_r = np.convolve(mean_r, np.ones(20)/20, mode='valid')
    sm_rb = np.convolve(mean_rb, np.ones(20)/20, mode='valid')
    ax.plot(sm_r, label='REINFORCE', linewidth=2, color='steelblue')
    ax.plot(sm_rb, label='REINFORCE + baseline', linewidth=2, color='crimson')
    # 加 ±std 阴影
    std_r = reinforce_runs.std(axis=0)
    std_rb = reinforce_baseline_runs.std(axis=0)
    x = np.arange(len(sm_r))
    ax.fill_between(x, sm_r - std_r[10:10+len(x)], sm_r + std_r[10:10+len(x)],
                    color='steelblue', alpha=0.2)
    ax.fill_between(x, sm_rb - std_rb[10:10+len(x)], sm_rb + std_rb[10:10+len(x)],
                    color='crimson', alpha=0.2)
ax.axhline(100, color='orange', linestyle='--', alpha=0.6, label='验收线 100')
ax.set_xlabel('episode'); ax.set_ylabel('reward (smoothed)')
ax.set_title('REINFORCE ± baseline 训练曲线（5 seeds）'); ax.legend(); ax.grid(alpha=0.3)

# 2. 方差对比（最后 50 episodes 的方差）
ax = axes[1]
last_50_var_r = reinforce_runs[:, -50:].var(axis=1)  # per-seed 方差
last_50_var_rb = reinforce_baseline_runs[:, -50:].var(axis=1)
ax.bar(['REINFORCE', 'REINFORCE\\n+ baseline'],
       [last_50_var_r.mean(), last_50_var_rb.mean()],
       yerr=[last_50_var_r.std(), last_50_var_rb.std()],
       color=['steelblue', 'crimson'], alpha=0.8)
ax.set_ylabel('per-seed Var(reward) over last 50 episodes')
ax.set_title('方差对比（baseline 显著降低）'); ax.grid(alpha=0.3, axis='y')

plt.tight_layout(); plt.show()

print(f"\\n观察：")
print(f"  - baseline 版的 per-seed 方差显著更低（{last_50_var_rb.mean():.1f} vs {last_50_var_r.mean():.1f}）")
print(f"  - baseline 版的最终 reward 通常也更高、更稳（信号 / 噪声比更高）")""")

# =============================================================================
# 7.6 Advantage 形式
# =============================================================================

md(r"""## 7.6 Advantage 形式 $Q - V$（铺垫 Ch08）

### 7.6.1 从 baseline 到 advantage

上一节的 baseline $b(s) \approx V^\pi(s)$ 是"该状态的平均价值"。代入策略梯度定理：

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi\!\Big[\nabla_\theta \log \pi(a|s) \cdot \big(Q^\pi(s,a) - V^\pi(s)\big)\Big] = \mathbb{E}_\pi\!\Big[\nabla_\theta \log \pi(a|s) \cdot A^\pi(s,a)\Big]
$$

其中 **advantage** 定义为：

$$
\boxed{\; A^\pi(s, a) := Q^\pi(s, a) - V^\pi(s) \;}
$$

**逐项解读**：

| 项 | 含义 |
|---|---|
| $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$ | 状态 $s$ 的"平均"价值 |
| $A^\pi(s, a)$ | "在状态 $s$ 选动作 $a$ 比平均水平好多少" |
| $A > 0$ | 这个动作比平均好 → 增大概率 |
| $A < 0$ | 这个动作比平均差 → 减小概率 |

### 7.6.2 advantage 的几何直觉

把 $Q^\pi(s, \cdot)$ 看成关于 $a$ 的函数（对每个 $s$）：

- $V^\pi(s) = \mathbb{E}_a[Q^\pi(s, a)]$ 是这条曲线的**平均高度**
- $A^\pi(s, a)$ 是曲线在 $a$ 处的**偏离平均的量**

策略梯度做的是："**对偏离平均的动作按其偏离程度调整概率**"——比平均好的多选，差的少选。

### 7.6.3 MC 估计 advantage

REINFORCE 用 $G_t$ 估计 $Q^\pi(s_t, a_t)$。如果同时有个 $V_\phi$ 网络估计 $V^\pi$，advantage 的 MC 估计就是

$$
\hat A_t = G_t - V_\phi(s_t)
$$

**这就是 REINFORCE + learned baseline**。注意 $V_\phi$ **不参与策略梯度**（在 autodiff 里 detach），它的作用纯粹是降方差。

### 7.6.4 但 $V_\phi$ 怎么学？

如果 $V_\phi$ 不参与 actor 的梯度，那它怎么更新？答案：**单独用一个回归 loss 学**

$$
L_{\text{critic}}(\phi) = \mathbb{E}\!\Big[(G_t - V_\phi(s_t))^2\Big]
$$

这是**监督回归**：target 是 MC return $G_t$，prediction 是 $V_\phi(s_t)$。

**这就是 Ch08 Actor-Critic 的核心**：actor（$\pi_\theta$）用策略梯度学，critic（$V_\phi$）用回归学。本章只用 baseline（critic 仅做减法），不做 critic 的完整训练——那是 Ch08 的事。

### 7.6.5 TD error 作为 advantage 的近似（预告）

计算 $G_t$ 必须等 episode 结束（MC）。**能不能用 bootstrap 一步估计 advantage**？可以——就是 Ch04 的 TD error

$$
\delta_t = R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

$\delta_t$ 是 $A^\pi$ 的一步近似（bias-variance 折中）。**Ch08 会严格推导 GAE** $\hat A_t = \sum_l (\gamma \lambda)^l \delta_{t+l}$——一个 bias-variance 可调的 advantage 估计。""")

code(r"""# 演示：用一个小 V_φ 网络做 baseline，看 advantage 信号
torch.manual_seed(0)

# 简易 V 网络（输入 state，输出 scalar）
class VNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dims=[64, 64]):
        super().__init__()
        self.net = make_mlp(state_dim, 1, hidden_dims)
    def forward(self, s):
        return self.net(s).squeeze(-1)  # [batch]

v_net = VNetwork(state_dim=4, hidden_dims=[64, 64])

# 用 REINFORCE 训好的策略跑一个 episode
env = CartPoleLite(seed=0, max_steps=500)
states, actions, returns, log_probs, total_r = collect_episode(env, reinforce_policy, gamma=0.99)
states_t = torch.as_tensor(np.array(states), dtype=torch.float32)

# V 网络未训练——输出随机
v_values = v_net(states_t).detach()
advantages = returns - v_values

print(f"episode 长度 = {len(actions)}, total reward = {total_r}")
print(f"\\nG_t（return）的统计：mean={returns.mean():.2f}, std={returns.std():.2f}")
print(f"V_φ(s_t)（未训练）的统计：mean={v_values.mean():.2f}, std={v_values.std():.2f}")
print(f"Â_t = G_t - V_φ(s_t) 的统计：mean={advantages.mean():.2f}, std={advantages.std():.2f}")
print(f"\\n→ V_φ 未训练时，advantage ≈ G_t（没有降方差效果）。")
print(f"→ Ch08 会用回归训练 V_φ，让它接近 V^π，此时 advantage 才真正降方差。")

# 可视化 G_t vs V_φ(s_t) vs Â_t
fig, ax = plt.subplots(figsize=(11, 4))
t = np.arange(len(actions))
ax.plot(t, returns.numpy(), 'o-', label=r'$G_t$ (return)', linewidth=1.5, markersize=4)
ax.plot(t, v_values.numpy(), 's-', label=r'$V_\phi(s_t)$ (untrained)', linewidth=1.5, markersize=4)
ax.bar(t, advantages.numpy(), alpha=0.3, color='gray', label=r'$\hat A_t = G_t - V_\phi$')
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlabel('t (step in episode)'); ax.set_ylabel('value')
ax.set_title('G_t, V_φ(s_t), advantage Â_t（V_φ 未训练）'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()""")

# =============================================================================
# 7.7 on-policy 限制
# =============================================================================

md(r"""## 7.7 on-policy 限制（复用 Ch05 §5.10）

### 7.7.1 为什么策略梯度是 on-policy

**回顾 Ch05 §5.10**：on-policy 指用**当前**策略 $\pi_\theta$ 采的数据训练 $\pi_\theta$ 自身；off-policy 指用**其它**（behavior）策略采的数据训练 $\pi_\theta$。

策略梯度定理 $\nabla J = \mathbb{E}_{\pi_\theta}[\nabla \log \pi \cdot Q^\pi]$ 里的期望**显式在 $\pi_\theta$ 上取**：

$$
\mathbb{E}_{s \sim d^{\pi_\theta},\; a \sim \pi_\theta(\cdot|s)}[\dots]
$$

这意味着采数据必须用 $\pi_\theta$ 自身。**一旦 $\theta$ 更新，旧数据失效**——因为旧数据来自不同的 $\pi_{\theta_{\text{old}}}$，对应的状态分布 $d^{\pi_{\theta_{\text{old}}}}$ 和动作分布都不一样。

### 7.7.2 对比 DQN（off-policy）

**Ch06 DQN 是 off-policy**：用 $\epsilon$-greedy（behavior）采数据，但学的是 $Q^*$（target 是 $\max$ 不是 $\pi$）。这让 DQN 可以用 replay buffer 重复利用旧数据——**样本效率高**。

| | DQN (off-policy) | REINFORCE (on-policy) |
|---|---|---|
| Replay buffer | ✓（核心组件） | ✗（数据用一次就扔） |
| 样本效率 | 高 | 低 |
| 数据分布 | 任意 behavior | 必须 = target $\pi_\theta$ |
| 数学严格性 | 用 IS 才严格（否则 off-bias） | 直接成立 |

### 7.7.3 off-policy 策略梯度（旁注，Chapter 9 PPO 用）

理论上可以用 **importance sampling** 把 off-policy 数据"修正"为 on-policy：

$$
\nabla J = \mathbb{E}_{\text{old data}}\!\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \nabla \log \pi_\theta(a|s) \cdot Q^\pi(s,a)\right]
$$

比率 $\pi_\theta / \pi_{\theta_{\text{old}}}$ 叫 **importance ratio** $r_t$。但 $r_t$ 在 $\theta$ 偏离 $\theta_{\text{old}}$ 时方差爆炸——**Ch09 PPO-Clip 就是用 clipping 来限制 $r_t$**，让 off-policy 数据可以有限重用。本章不展开。

### 7.7.4 代码验证：REINFORCE 不能用 replay buffer

下面做个反面实验：把旧 episode 的数据存下来重复用——看是否会失败。""")

code("""# 反面实验：用旧 episode 的数据训练当前 policy（off-policy）
# 预期：训练不稳定 / 学不到东西，因为分布不匹配

torch.manual_seed(0)
np.random.seed(0)

env = CartPoleLite(seed=0, max_steps=500)
policy = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[64, 64])
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

# 用初始 policy 采 50 条 episode，存下来
print("采 50 条 episode 作为 'stale data'（旧策略采的）...")
stale_episodes = []
for _ in range(50):
    states, actions, returns, log_probs, total_r = collect_episode(env, policy, gamma=0.99)
    stale_episodes.append((states, actions, returns.clone().detach()))

print(f"  平均 reward = {np.mean([r for _, _, r, _, _ in [(*e, None, None, None)] for e in stale_episodes] if False else [e[2][0].item() for e in stale_episodes]):.1f}")

# 现在用这批 stale data 反复训练 policy 100 次（模拟 "replay buffer"）
losses = []
for it in range(100):
    # 随机抽一个 stale episode
    idx = np.random.randint(len(stale_episodes))
    states_s, actions_s, returns_s = stale_episodes[idx]

    # 关键问题：log_probs 必须重新用当前 policy 算（不能复用旧的）
    s_t = torch.as_tensor(np.array(states_s), dtype=torch.float32)
    a_t = torch.as_tensor(actions_s, dtype=torch.long)
    dist = policy(s_t)
    log_probs_new = dist.log_prob(a_t)

    loss = -(log_probs_new * returns_s).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    opt.step()
    losses.append(loss.item())

# 评估"训练后"的策略
np.random.seed(1)
rewards_after = evaluate_policy(CartPoleLite(seed=0, max_steps=500), policy, n_episodes=5)
print(f"\\n用 stale data 训练 100 次后，policy 的 reward: {rewards_after}")
print(f"对比：stale data 是用 random policy 采的（reward ~10），policy 只见到了低 reward 的轨迹，")
print(f"且数据分布与当前 policy 严重不匹配（state distribution drift），所以训练几乎学不到东西。")
print(f"\\n→ 这就是 on-policy 的本质限制。要 off-policy 必须用 importance sampling 或 PPO clipping。")""")

# =============================================================================
# 7.8 小结 + Ch08 预告
# =============================================================================

md(r"""## 7.8 小结 + Actor-Critic 预告

### 7.8.1 本章核心收获

1. **policy-based vs value-based**：DQN 只能离散动作 + 确定性策略；策略梯度天然支持连续动作 + 随机策略
2. **策略梯度定理**：$\nabla_\theta J = \mathbb{E}_\pi[\nabla \log \pi \cdot Q^\pi]$——RL 中最优雅的等式
3. **两个关键技巧**：
   - **score function trick**：$\nabla \pi = \pi \nabla \log \pi$，把"概率的梯度"换成"梯度的期望"
   - **测度变换**：把 $\sum_a \pi \cdot f$ 写成 $\mathbb{E}_{a \sim \pi}[f]$
4. **REINFORCE 算法**：MC 策略梯度，用 $G_t$ 估 $Q^\pi$
5. **baseline 不偏**：因为 $\sum_a \pi \nabla \log \pi = \nabla \sum_a \pi = \nabla 1 = 0$
6. **advantage 形式**：$Q^\pi - V^\pi$，从"绝对好"变"相对好"，降方差
7. **on-policy 限制**：策略梯度必须用当前 $\pi_\theta$ 采的数据

### 7.8.2 REINFORCE 的两大痛点

| 痛点 | 原因 | Ch08/Ch09 的解法 |
|---|---|---|
| **高方差** | MC return $G_t$ 方差大 | 用 $V_\phi$ 估 baseline + TD-based advantage |
| **样本效率低** | on-policy，数据用一次就扔 | Ch09 PPO：importance sampling + clipping，允许有限重用 |
| **训练慢** | 必须等 episode 结束才能更新 | Ch08 用 TD error，每步就能更新 |

### 7.8.3 与 Phase 1 兑现的承诺（回顾）

| 出处 | 承诺 | 本章是否兑现 |
|---|---|---|
| Ch00 fast-track | Ch07 在没读 Ch06 时也能懂 | ✓ §7.1 给了跳读路径 + PyTorch 速查 |
| Ch00 章节图 | "策略梯度定理" | ✓ §7.3 完整证明（trajectory + 通用两版） |
| Ch05 章末 | "Phase 1 结束 → Ch07 策略梯度定理" | ✓ 全章 |

### 7.8.4 关键公式速查表

| 公式 | 含义 | 出现节 |
|---|---|---|
| $J(\theta) = \mathbb{E}_\pi[\sum_t \gamma^t R_{t+1}]$ | 目标函数 | §7.2 |
| $\nabla J = \mathbb{E}[\nabla \log \pi \cdot Q^\pi]$ | 策略梯度定理 | §7.3 |
| $\nabla \pi = \pi \cdot \nabla \log \pi$ | score function trick | §7.3 |
| $\text{loss} = -\log \pi(a_t\|s_t) \cdot G_t$ | REINFORCE loss | §7.4 |
| $\nabla J = \mathbb{E}[\nabla \log \pi \cdot (Q - b(s))]$ | + baseline | §7.5 |
| $A^\pi = Q^\pi - V^\pi$ | advantage | §7.6 |

---

下一章：**第 8 章 — Actor-Critic + GAE**。

本章我们说"$V_\phi$ 仅做 baseline、不参与 actor 梯度"——Ch08 把这个限制打破：

- **critic $V_\phi$** 也参与训练（用 TD 误差学）
- **actor** 用 **TD-based advantage**（不需要等 episode 结束，每步更新）
- **GAE** $\hat A_t = \sum_l (\gamma \lambda)^l \delta_{t+l}$ 给出 bias-variance 可调的 advantage 估计——直接兑现 Ch04 §4.8 的承诺"GAE = TD(λ) 的 advantage 版本"

这是从 REINFORCE 到 PPO 的关键一步。""")

code("""# 可视化训练后的策略分布 π(a|s) 在某个状态下的演化
# 选一个代表性状态（杆子稍微向右倾——应该选 action=1=向右推）

test_states = np.array([
    [0.0, 0.0,  0.00, 0.0],   # 完美直立
    [0.0, 0.0,  0.05, 0.0],   # 向右倾一点
    [0.0, 0.0, -0.05, 0.0],   # 向左倾一点
    [0.0, 0.0,  0.15, 0.0],   # 向右倾很多
    [0.0, 0.0, -0.15, 0.0],   # 向左倾很多
    [0.5, 0.0,  0.00, 0.0],   # 小车偏右
    [-0.5, 0.0, 0.00, 0.0],   # 小车偏左
])
labels = ['直立', '右倾 0.05', '左倾 0.05', '右倾 0.15', '左倾 0.15', '车偏右', '车偏左']

with torch.no_grad():
    s_t = torch.as_tensor(test_states, dtype=torch.float32)
    dist = reinforce_policy(s_t)
    probs = dist.probs.numpy()  # [7, 2]

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, probs[:, 0], width, label='π(left|s)', color='steelblue')
ax.bar(x + width/2, probs[:, 1], width, label='π(right|s)', color='crimson')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
ax.set_ylabel('probability'); ax.set_ylim(0, 1)
ax.set_title('学到的策略分布 π(a|s)（REINFORCE 训练后）')
ax.legend(); ax.grid(alpha=0.3, axis='y')
for i in range(len(labels)):
    ax.text(x[i] - width/2, probs[i, 0] + 0.02, f'{probs[i,0]:.2f}', ha='center', fontsize=8)
    ax.text(x[i] + width/2, probs[i, 1] + 0.02, f'{probs[i,1]:.2f}', ha='center', fontsize=8)
plt.tight_layout(); plt.show()

print("观察：")
print("  - 直立时：π ≈ 0.5/0.5（两种动作差不多——合理，直立时随便推）")
print("  - 右倾时：π(right) 显著高（要把杆子推回去）")
print("  - 左倾时：π(left) 显著高")
print("  - 这就是策略网络学到的'条件反射'")""")

md(r"""## 7.9 📝 练习

### 练习 1（必做）：REINFORCE + entropy bonus

**任务**：
1. 复用本章的 `train_reinforce` 脚手架，在 loss 里加 `- ent_coef · H(π(·|s))`（entropy 用 `-(p·log p).sum(-1)` 对 batch 求均值）
2. ent_coef 取 0 / 0.01 / 0.05 三档，在 CartPoleLite 上对比训练曲线
3. 观察熵曲线：entropy bonus 应该让策略分布"死得慢"

<details><summary>提示</summary>

- `CategoricalPolicy` 返回分布对象，`dist.entropy()` 直接可用
- loss 取负号梯度下降：total = pg_loss − ent_coef·entropy（要最大化 entropy 就减它）
- 系数过大会让策略保持接近均匀、reward 上不去——这就是探索/利用权衡在系数层面的样子
</details>

**预期结果**：适度的 entropy bonus（~0.01）让训练更稳、后期曲线方差更小；0.05 时探索过强、收敛变慢。

### 练习 2（挑战）：策略梯度估计的方差可视化

本章用有限差分验证过单条轨迹的策略梯度估计。**任务**：对同一个小问题，采样 n ∈ {1, 2, 5, 10, 50, 100} 条轨迹各重复 200 次，画出"估计值的标准差 vs n"（对数坐标）。

**预期结果**：std ∝ 1/√n——蒙特卡洛估计的标准图像；同时均值几乎不变（无偏性）。这就是后面 Ch08 要用 critic/baseline 降方差的动机：不改变均值、只压低这条线。

*（开放练习，无参考答案。）
> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch07 的自测题再进入下一章。""")


if __name__ == "__main__":
    # 直接运行时写出 notebook（被 build_notebooks.py import 时不写盘）
    _nb.write("ch07_policy_gradient.ipynb")
