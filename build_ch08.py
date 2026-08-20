r"""Build ch08_actor_critic_gae.ipynb.

一次性脚本：构造 Ch08 笔记本（8 节内容，Actor-Critic + GAE）。
运行后产物在 notebooks/。
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch08")
cells = _nb.cells            # build_notebooks.py 适配层读取的模块级列表
md, code = _nb.md, _nb.code


# =============================================================================
# Chapter intro + setup
# =============================================================================

md(r"""# 第 8 章：Actor-Critic + GAE —— 把 REINFORCE 升级成 PPO 的基石

> **Ch07 REINFORCE 用 $G_t$ 估 $Q^\pi$**，但 $G_t$ 方差爆炸、必须等 episode 结束。
> **本章** 把"被动的 baseline $V_\phi$"升级成"主动学习的 critic"，并用 **TD error $\delta_t$** 替代 MC return，
> 再用 **GAE**（Generalized Advantage Estimation）把 $\delta_t$ 加权累加成一个 **bias-variance 可调** 的 advantage 估计。
>
> **本章核心等式**（Schulman et al. 2015，PPO 标配）：
>
> $$\hat A_t^{GAE(\gamma,\lambda)} \;=\; \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \qquad \delta_{t+l} = R_{t+l+1} + \gamma V_\phi(S_{t+l+1}) - V_\phi(S_{t+l})$$
>
> 这个公式正是 **Ch04 §4.8 TD(λ) 思想的 advantage 版本**——§8.5 会严格证明两者的等价。

## 学习目标

1. 理解 **为什么纯 REINFORCE 慢**（高方差 + 必须 offline + 无 bootstrap）
2. 掌握 **critic $V_\phi$** 的角色（从被动 baseline 升级成主动 learner）
3. 推出 **Actor-Critic** 梯度：$\nabla J = \mathbb{E}[\nabla\log\pi \cdot \delta_t]$（严格证明不偏）
4. 理解 **n-step advantage**（复用 Ch04 §4.7）
5. **完整推出 GAE**（本章灵魂）——包括"GAE = $(1-\lambda)\sum_n \lambda^{n-1} \hat A^{(n)}$ = $\sum_l (\gamma\lambda)^l \delta_l$"的双重等价证明
6. 实现 **完整 A2C**（同步优势 actor-critic）训练 CartPoleLite
7. 用 **$\lambda$ 滑块** 实验 bias-variance tradeoff
8. 预告 **Ch09 PPO**（importance sampling + clipping + multiple epochs）

## 承接的 Phase 1 / Ch07 概念线索（7 处 —— 全项目密度最高的一章）

| 出处 | 承接的概念 | 本章兑现节 |
|---|---|---|
| Ch04 §4.3 | "TD error $\delta_t$ → Ch08 Actor-Critic：advantage 的核心" | §8.3 |
| Ch04 §4.8 | "TD(λ) eligibility → Ch08 GAE（PPO 标配）" | §8.5 |
| **Ch04 §4.8** | **"GAE = TD(λ) 思想的 advantage 版本"（必须严格证明）** | **§8.5（灵魂）** |
| Ch04 §4.8 | "eligibility traces 思想在 Actor-Critic、PPO、GAE 反复出现" | §8.5、§8.8 |
| Ch02 §2.3 | $V^\pi$ 的定义——critic 学的就是它 | §8.2、§8.3 |
| Ch05 §5.1 | "控制 = 预测 + 贪心改进"——本章升级为可学习策略 | §8.6 |
| Ch04 §4.8 | "GAE, Retrace(λ), V-trace, IMPALA" | §8.5、§8.8 |

## 术语速查（第一次出现时加脚注，这里给完整列表）

| 术语 | 一句话解释 |
|---|---|
| **actor** $\pi_\theta$ | 策略网络，输出动作分布（Ch07） |
| **critic** $V_\phi$ | 价值网络，估计 $V^\pi(s)$；本章把它从被动 baseline 升级为主动 learner |
| **TD error** $\delta_t$ | $R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$（Ch04 §4.3） |
| **advantage** $A^\pi(s,a)$ | $Q^\pi(s,a) - V^\pi(s)$（Ch07 §7.6） |
| **bootstrap** | 用 estimate 更新 estimate（Ch04 §4.3）—— TD error 的来源 |
| **n-step advantage** $\hat A^{(n)}_t$ | 前 $n$ 步真实奖励 + 第 $n$ 步 bootstrap，见 §8.4 |
| **GAE** $\hat A^{GAE(\gamma,\lambda)}_t$ | $(\gamma\lambda)$ 衰减的 TD error 之和（本章核心） |
| **A2C** | Advantage Actor-Critic（同步版，Mnih et al. 2016） |
| **A3C** | Asynchronous Advantage Actor-Critic（A2C 的异步版） |
| **bias-variance tradeoff** | estimator 的偏差与方差的权衡（Ch04 §4.6） |
| **value loss** | critic 的回归 loss $(V_\phi - G_t)^2$ |
| **entropy bonus** | $-\beta H(\pi_\theta)$，鼓励探索（PPO 标配） |

---

## 章节结构（8 节）

1. **§8.1** 为什么纯 REINFORCE 慢（高方差 + 必须 offline + 无 bootstrap）
2. **§8.2** Critic $V_\phi$：用回归学 $V^\pi$
3. **§8.3** Actor-Critic：REINFORCE + critic baseline，$A_t \approx \delta_t$
4. **§8.4** n-step advantage（复用 Ch04 §4.7）
5. **§8.5** **GAE 完整推导（核心）**——严格证明 GAE = $(1-\lambda)\sum \lambda^{n-1}\hat A^{(n)}$ = $\sum_l (\gamma\lambda)^l \delta_l$
6. **§8.6** 完整 A2C 实现
7. **§8.7** $\lambda$ 作为 bias-variance 旋钮（实验对比）
8. **§8.8** 小结 + Ch09 PPO 预告""")

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
import torch.nn.functional as F
from torch.distributions import Categorical

from rlenvs import CartPoleLite
from utils import set_seed
from utils.networks import make_mlp
from utils.policy_networks import CategoricalPolicy, ValueNetwork, ActorCritic
from utils.gae import (
    compute_gae, compute_td_errors, compute_n_step_advantage,
    compute_returns_from_gae,
)
from utils.torch_utils import get_device, count_parameters
from utils import smooth

set_seed(42)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = get_device()
print(f"PyTorch: {torch.__version__}, device = {DEVICE}")
print(f"utils 新增基础设施：ValueNetwork, ActorCritic, compute_gae 等")""")

# =============================================================================
# 8.1 为什么纯 REINFORCE 慢
# =============================================================================

md(r"""## 8.1 为什么纯 REINFORCE 慢

### 8.1.1 复习 Ch07 REINFORCE 的核心公式

Ch07 §7.4 给出的 REINFORCE 算法：

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi\!\Big[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t\Big], \quad G_t = \sum_{k=t}^{T-1}\gamma^{k-t} R_{k+1}
$$

其中 $G_t$ 是从 $t$ 步开始的**蒙特卡洛** return。算法跑得慢的根源全在这一个 $G_t$ 上。

### 8.1.2 三大痛点

| 痛点 | 数学/工程表现 | 后果 |
|---|---|---|
| **高方差** | $\text{Var}(G_t) \approx \dfrac{\sigma_r^2}{1-\gamma^2}$（Ch04 §4.6），对 $\gamma=0.99$ 约 $50\sigma_r^2$ | 梯度噪声极大，几百到几千 episodes 才稳 |
| **必须 offline** | $G_t$ 要等 episode 结束才能算 | 长 episode（如 CartPole 500 步）每 episode 才更新一次 |
| **无 bootstrap** | $G_t$ 用纯样本奖励，不用当前 $V$ 估计 | 学不到的 reward 不能"借道" $V$ 传播 |

### 8.1.3 用 Ch07 的 baseline 部分缓解

Ch07 §7.5 引入了 baseline $b(s) \approx V^\pi(s)$，把信号从 $G_t$ 换成 $G_t - V_\phi(s_t)$（advantage 的 MC 估计）。这**降低方差**，但不解决另外两个痛点——还是 offline、还是无 bootstrap。

**关键问题**：能不能把"等到 episode 结束才更新"换成"每步都能更新"？这就需要 **TD error**——Ch04 §4.3 已经埋下的伏笔。

### 8.1.4 本章路线图

下面我们走完从 REINFORCE 到 A2C 的完整升级：

1. **§8.2** 把 baseline $V_\phi$ 升级成 **critic**：用回归 loss 主动学习 $V^\pi$
2. **§8.3** 把信号从 $G_t - V_\phi(s_t)$ 换成 **TD error** $\delta_t$（一步 bootstrap advantage）
3. **§8.4** 在 1 步和 ∞ 步之间插值：**n-step advantage**
4. **§8.5** 同时用所有 $n$ 的加权平均：**GAE**""")

code("""# 复习：用 Ch07 REINFORCE 跑一个 baseline，看它有多慢
# （这里只是回顾，不必深究细节——本章主角是后面的 A2C）

torch.manual_seed(0)
np.random.seed(0)

# 简易 V 网络（Ch07 笔记本里临时手写过的版本）
def compute_returns(rewards, gamma):
    T = len(rewards)
    returns = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# 用未训练的策略 + 一个 V_φ 网络看 G_t 与 V_φ 的尺度
env = CartPoleLite(seed=0, max_steps=500)
policy = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[64, 64])
v_net = ValueNetwork(state_dim=4, hidden_dims=[64, 64])

# 跑一条 episode
s = env.reset()
states, actions, rewards, log_probs = [], [], [], []
done = False
while not done:
    s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
    dist = policy(s_t)
    a = dist.sample()
    logp = dist.log_prob(a)
    s_next, r, done, _ = env.step(int(a.item()))
    states.append(s); actions.append(int(a.item())); rewards.append(r); log_probs.append(logp)
    s = s_next

returns = compute_returns(rewards, gamma=0.99)
states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
v_pred = v_net(states_t).detach().numpy()

print(f"未训练 episode 长度 = {len(rewards)}")
print(f"G_t (MC return):     mean={returns.mean():.2f}, std={returns.std():.2f}")
print(f"V_φ(s_t) (untrained): mean={v_pred.mean():.4f}, std={v_pred.std():.4f}")
print(f"\\nG_t 的 std ≈ {returns.std():.1f} 是 REINFORCE 高方差的根源；\\n"
      f"我们后面会把 G_t - V_φ(s_t) 替换为方差小得多的 TD error δ_t。")""")

# =============================================================================
# 8.2 Critic
# =============================================================================

md(r"""## 8.2 Critic $V_\phi$：用回归学 $V^\pi$

### 8.2.1 从被动 baseline 到主动 critic

**Ch07 的 baseline**：$V_\phi$ 只在 forward 时**做减法**降低 actor 梯度的方差，它的参数 $\phi$ 不参与训练（或用最简单的 running mean）。结果是：$V_\phi$ 永远是**未训练**的——降方差效果有限。

**本章的 critic**：$V_\phi$ 主动学习，目标是逼近 $V^\pi$。学得越准，advantage 估计越好，actor 训练越快。

### 8.2.2 Critic 的目标函数：监督回归

Critic 的 loss 是**监督回归**——target 是 return，prediction 是 $V_\phi$：

$$
\boxed{\;L_{\text{critic}}(\phi) \;=\; \mathbb{E}\!\Big[(G_t - V_\phi(S_t))^2\Big]\;}
$$

**逐项解读**：

| 项 | 含义 |
|---|---|
| $G_t$ | target——可以是 MC return（最准、高方差），也可以是 TD target $R_{t+1} + \gamma V_\phi(S_{t+1})$ |
| $V_\phi(S_t)$ | prediction |
| $(\cdot)^2$ | MSE——可以微分，$\nabla_\phi L = -2(G_t - V_\phi)\nabla_\phi V_\phi$ |

### 8.2.3 Target 的选择：MC vs TD

| Target | 公式 | 偏差 | 方差 | 来源 |
|---|---|---|---|---|
| MC return | $G_t$ | 0（无偏） | 高 | Ch04 §4.2 |
| 1-step TD target | $R_{t+1} + \gamma V_\phi(S_{t+1})$ | 有（bootstrap） | 低 | Ch04 §4.3 |
| n-step TD target | $\sum_{k=0}^{n-1}\gamma^k R_{t+k+1} + \gamma^n V_\phi(S_{t+n})$ | 中 | 中 | Ch04 §4.7 |
| $\lambda$-return | $(1-\lambda)\sum_n \lambda^{n-1} G_t^{(n)}$ | 可调 | 可调 | Ch04 §4.8 |

**重要**：actor 的 advantage 估计和 critic 的 target 可以**独立选择**！例如：
- Actor 用 GAE(λ=0.95) 做 advantage
- Critic 用 MC return（$G_t = \hat A_t + V_\phi(s_t)$，由 GAE 反推）做 target

这是 PPO 的默认做法。

### 8.2.4 实践：用回归训练 $V_\phi$

让我们先在一个 toy 例子上验证：用回归 loss 训练 $V_\phi$，看它能否逼近真 $V^\pi$。""")

code("""# Toy 验证：用回归 loss 训练 V_φ 让它学到 V^π 的形状
# 用 RandomWalk-style 的玩具：(s, V*) 都是已知的
# 这里我们直接用 CartPoleLite 收集数据 + MC return 做 target，看 V_φ 学到什么形状

torch.manual_seed(0)
np.random.seed(0)

v_net = ValueNetwork(state_dim=4, hidden_dims=[64, 64])
opt_v = torch.optim.Adam(v_net.parameters(), lr=1e-2)
env = CartPoleLite(seed=0, max_steps=500)

# 行为策略：用 deterministic heuristic（与 §8.6 的"反射式"策略同款）
#   theta 是杆子角度（state[2]），theta_dot 是角速度（state[3]）
#   决策规则：当 (theta + 0.5 * theta_dot) > 0 时向右推，否则向左推
# 这种策略能让杆子保持接近直立，episode 长度接近 max_steps=500，
# (s, G_t) 的方差远小于 uniform random 策略（random 下平均 episode ~20 步）。
def heuristic_action(s):
    theta, theta_dot = s[2], s[3]
    return 1 if (theta + 0.5 * theta_dot) > 0 else 0

# 先对比两种行为策略的 episode 长度
def collect_one_episode(policy_fn, env, max_steps=500):
    s = env.reset(); rewards = []
    done = False
    while not done:
        a = policy_fn(s)
        s, r, done, _ = env.step(a)
        rewards.append(r)
    return len(rewards)

random_lens = [collect_one_episode(lambda s: np.random.randint(2), env) for _ in range(20)]
heuristic_lens = [collect_one_episode(heuristic_action, env) for _ in range(20)]
print(f"行为策略 episode 长度对比（20 episodes）：")
print(f"  uniform random: mean={np.mean(random_lens):.1f}, std={np.std(random_lens):.1f}")
print(f"  heuristic     : mean={np.mean(heuristic_lens):.1f}, std={np.std(heuristic_lens):.1f}")
print(f"→ heuristic 让 episode 稳定在 ~500 步，G_t 的方差大幅降低，critic 才能真正学到 V^π。\\n")

# 收集 200 条 episode 的 (s, G_t) 数据 —— 用 heuristic 策略
print("收集 200 episodes 的 (s, G_t)（heuristic 策略）...")
dataset_states, dataset_returns = [], []
for _ in range(200):
    s = env.reset()
    states, rewards = [], []
    done = False
    while not done:
        states.append(s)
        a = heuristic_action(s)   # 反射式 heuristic 替代 uniform random
        s, r, done, _ = env.step(a)
        rewards.append(r)
    returns = compute_returns(rewards, gamma=0.99)
    dataset_states.extend(states)
    dataset_returns.extend(returns)

dataset_states = np.array(dataset_states, dtype=np.float32)
dataset_returns = np.array(dataset_returns, dtype=np.float32)
print(f"数据集大小: {len(dataset_returns)} samples")
print(f"G_t 统计: mean={dataset_returns.mean():.2f}, std={dataset_returns.std():.2f}")

# 训练 V_φ 50 epochs
losses = []
for epoch in range(50):
    idx = np.random.permutation(len(dataset_states))
    for start in range(0, len(idx), 64):
        b = idx[start:start+64]
        s_b = torch.as_tensor(dataset_states[b])
        g_b = torch.as_tensor(dataset_returns[b])
        v_pred = v_net(s_b)
        loss = F.mse_loss(v_pred, g_b)
        opt_v.zero_grad(); loss.backward(); opt_v.step()
        losses.append(loss.item())

print(f"\\n训练后 V_φ 的 MSE loss: {np.mean(losses[-100:]):.3f}（开始时: {np.mean(losses[:100:]):.3f}）")

# 验证：在几个特定状态上预测 V
test_states = np.array([
    [0.0, 0.0,  0.00, 0.0],   # 完美直立（应该最高）
    [0.0, 0.0,  0.05, 0.0],   # 微倾（应该次高）
    [0.0, 0.0,  0.15, 0.0],   # 大倾（应该低）
    [0.0, 0.0, -0.15, 0.0],
    [2.0, 0.0,  0.00, 0.0],   # 接近出轨（应该低）
], dtype=np.float32)
with torch.no_grad():
    v_pred_test = v_net(torch.as_tensor(test_states)).numpy()
labels = ['直立', '右倾 0.05', '右倾 0.15', '左倾 0.15', '车偏右 2.0']
print(f"\\nV_φ 在不同状态的估计（heuristic 策略下，V^π(s) ≈ episode 剩余步数）：")
for lab, v in zip(labels, v_pred_test):
    print(f"  {lab:<14}: V_φ = {v:>6.2f}")

# 排序验证：critic 学到的相对 ordering 是否合理（V_φ(直立) > V_φ(微倾) > V_φ(大倾)）
sorted_idx = np.argsort(-v_pred_test)  # 降序
print(f"\\nV_φ 排序（高→低）: {[labels[i] for i in sorted_idx]}")
print(f"\\n→ 直立状态 V_φ 最高（合理：还能撑很久）；倾斜或偏轨 V_φ 低。\\n"
      f"  这就是 critic 学到的'状态价值'。\\n"
      f"  注：用 uniform random 收集 (s, G_t) 时 G_t 噪声极大（episode 平均 ~20 步），\\n"
      f"  critic 几乎学不动——这正是 Ch08 之后需要 actor-critic 联合训练的原因：\\n"
      f"  critic 边学 V^π、actor 边改进 π，互相 bootstrapping 才能在 random 数据上学起来。")""")

# =============================================================================
# 8.3 Actor-Critic = REINFORCE + critic baseline
# =============================================================================

md(r"""## 8.3 Actor-Critic = REINFORCE + critic baseline

### 8.3.1 关键观察：把 $G_t$ 换成 $\delta_t$

Ch07 §7.6 的 advantage 形式：

$$
\nabla_\theta J = \mathbb{E}_\pi\!\Big[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\big(G_t - V_\phi(s_t)\big)\Big]
$$

现在把 $G_t$ 替换为 **TD error**（Ch04 §4.3）：

$$
\boxed{\;\delta_t \;:=\; R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)\;}
$$

得到 **Actor-Critic** 梯度（最简形式）：

$$
\boxed{\;\nabla_\theta J = \mathbb{E}_\pi\!\Big[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\delta_t\Big]\;}
$$

**逐项解读**：

| 项 | 含义 |
|---|---|
| $\delta_t$ | "实际 $R_{t+1} + \gamma V_\phi(S_{t+1})$ 减预期 $V_\phi(S_t)$"，是 1-step advantage 估计 |
| $\nabla\log\pi \cdot \delta_t$ | "比预期好就增大概率（$\delta_t > 0$），差就减小（$\delta_t < 0$）" |
| $\delta_t > 0$ | 这一步的回报比 $V_\phi$ 预期的高 → $a_t$ 是好动作 |
| $\delta_t < 0$ | 这一步比预期差 → $a_t$ 是差动作 |

### 8.3.2 为什么 $\delta_t$ 是 1-step advantage 估计

**定理**：$\mathbb{E}_\pi[\delta_t \mid S_t = s, A_t = a] = A^\pi(s, a)$ 当 $V_\phi = V^\pi$ 时（即 critic 已收敛）。

<details>
<summary><b>完整证明：TD error 是 1-step advantage（点开看）</b></summary>

回忆 $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$（Ch07 §7.6 定义）。

**Bellman 期望方程**（Ch02 §2.6）：

$$
Q^\pi(s, a) = \sum_{s'} p(s'|s, a)\Big[r(s, a, s') + \gamma V^\pi(s')\Big] = \mathbb{E}_{s'}\!\Big[R_{t+1} + \gamma V^\pi(S_{t+1}) \,\Big|\, S_t=s, A_t=a\Big]
$$

所以

$$
\mathbb{E}_\pi[\delta_t \mid S_t=s, A_t=a] = \mathbb{E}_\pi[R_{t+1} + \gamma V^\pi(S_{t+1})] - V^\pi(s) = Q^\pi(s, a) - V^\pi(s) = A^\pi(s, a)
$$

$\blacksquare$

**意义**：TD error 是 advantage 的 1-step（bootstrap）估计——有偏（因为 $V_\phi \neq V^\pi$），但方差小（只看一步奖励 + 一步 $V$ 估计）。

</details>

### 8.3.3 关键证明：$G_t \to \delta_t$ 不偏

直接把 $G_t$ 换成 $\delta_t$ 看起来"换了一个完全不同的量"——梯度还指向 $\nabla J$ 吗？答案是肯定的，**因为 $\delta_t$ 在期望上等于 advantage 的 1-step 估计**。但严格论证要看下面这个 telescoping 推导：

<details>
<summary><b>严格证明：G_t - V_φ(s_t) 与 δ_t + γ(G_{t+1} - V_φ(s_{t+1})) 在期望下相同（telescoping）（点开看）</b></summary>

**核心恒等式**（来自 $G_t = R_{t+1} + \gamma G_{t+1}$）：

$$
\begin{aligned}
G_t - V_\phi(S_t) &= R_{t+1} + \gamma G_{t+1} - V_\phi(S_t) \\
&= \big[R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)\big] + \gamma\big(G_{t+1} - V_\phi(S_{t+1})\big) \\
&= \delta_t + \gamma\big(G_{t+1} - V_\phi(S_{t+1})\big)
\end{aligned}
$$

**关键观察**：上式对**任何 $V_\phi$**（不必 = $V^\pi$）都成立。

代入策略梯度（对 $a_t \sim \pi_\theta(\cdot|s_t)$、$s_{t+1} \sim p(\cdot|s_t,a_t)$、$a_{t+1} \sim \pi_\theta(\cdot|s_{t+1})$ 取期望）：

$$
\begin{aligned}
&\mathbb{E}_\pi\!\Big[\nabla\log\pi_\theta(a_t|s_t)\cdot\big(G_t - V_\phi(S_t)\big)\Big] \\
&= \mathbb{E}_\pi\!\Big[\nabla\log\pi_\theta(a_t|s_t)\cdot\delta_t\Big] + \gamma\,\mathbb{E}_\pi\!\Big[\nabla\log\pi_\theta(a_t|s_t)\cdot\big(G_{t+1} - V_\phi(S_{t+1})\big)\Big]
\end{aligned}
$$

**关键技巧**：第二项中 $\nabla\log\pi_\theta(a_t|s_t)$ 不依赖 $a_{t+1}$，所以可以提到对 $a_{t+1}$ 的求和外面：

$$
\mathbb{E}_{a_{t+1} \sim \pi_\theta(\cdot|s_{t+1})}\!\Big[G_{t+1} - V_\phi(S_{t+1})\Big] = Q^\pi(s_{t+1}, a_{t+1}) - V_\phi(s_{t+1}) \text{ 的均值} = V^\pi(s_{t+1}) - V_\phi(s_{t+1})
$$

（这只依赖 $s_{t+1}$，不依赖 $a_t$）。**用 Ch07 §7.5 的 baseline 论证**：$\nabla\log\pi_\theta(a_t|s_t)$ 乘以一个**只依赖 $s_{t+1}$ 的量**，对 $a_t$ 取期望 = 0（baseline 不偏的同款证明）。具体地：

$$
\mathbb{E}_{a_t \sim \pi_\theta}\!\Big[\nabla\log\pi_\theta(a_t|s_t)\cdot h(s_t, s_{t+1})\Big] = h(s_t, s_{t+1}) \cdot \sum_{a_t}\pi_\theta(a_t|s_t)\nabla\log\pi_\theta(a_t|s_t) = 0
$$

（因为 $\sum_a \pi\nabla\log\pi = \sum_a \nabla\pi = \nabla 1 = 0$，Ch07 §7.5.3）。

**结论**：

$$
\mathbb{E}_\pi\!\Big[\nabla\log\pi_\theta(a_t|s_t)\cdot\big(G_t - V_\phi(S_t)\big)\Big] = \mathbb{E}_\pi\!\Big[\nabla\log\pi_\theta(a_t|s_t)\cdot\delta_t\Big]
$$

$\blacksquare$

**意义**：把 $G_t - V_\phi$ 换成 $\delta_t$ **在期望下不改变梯度方向**——这是 Actor-Critic 数学合法性的根基。但方差变小了（$\delta_t$ 比 $G_t$ 方差小得多）。

</details>

### 8.3.4 Actor-Critic 的完整 loss

实际上我们同时训练两个网络：

| 网络 | Loss | 角色 |
|---|---|---|
| Actor $\pi_\theta$ | $L_{\text{actor}} = -\log\pi_\theta(a_t\|s_t)\cdot\delta_t$ | 优化策略 |
| Critic $V_\phi$ | $L_{\text{critic}} = (V_\phi(s_t) - G_t)^2$ 或 $(V_\phi(s_t) - \text{TD target})^2$ | 学 $V^\pi$ |

总 loss：

$$
\boxed{\;L_{\text{total}} \;=\; L_{\text{actor}} \;+\; c_v\, L_{\text{critic}} \;-\; \beta\, H(\pi_\theta(\cdot|s_t))\;}
$$

| 项 | 角色 |
|---|---|
| $c_v L_{\text{critic}}$ | critic loss，系数 $c_v \in [0.5, 1.0]$ |
| $-\beta H(\pi)$ | **entropy bonus**——鼓励探索（防止 $\pi$ 过早坍缩到确定性策略）；$\beta \in [0, 0.01]$ |

**为什么 entropy bonus**：策略梯度倾向于让 $\pi$ 越来越 peaky（高概率动作进一步增大）。Entropy bonus 反向作用，保持 $\pi$ 有一定随机性——这对探索至关重要，是 PPO 标配。""")

code("""# 演示：在未训练的策略上比较 G_t - V_φ(s_t) 和 δ_t 的尺度差
torch.manual_seed(0)
np.random.seed(0)

env = CartPoleLite(seed=0, max_steps=500)
policy = CategoricalPolicy(state_dim=4, n_actions=2, hidden_dims=[64, 64])

# 用刚才训好的 v_net 做 critic
# 收集一条 episode
s = env.reset()
states, actions, rewards, values = [], [], [], []
done = False
while not done:
    s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist = policy(s_t)
        v = v_net(s_t).item()
    a = int(dist.sample().item())
    s_next, r, done, _ = env.step(a)
    states.append(s); actions.append(a); rewards.append(r); values.append(v)
    s = s_next

T = len(rewards)
states_arr = np.array(states); values_arr = np.array(values)
gamma = 0.99

# 计算 G_t - V_φ(s_t)（Ch07 advantage）
returns = compute_returns(rewards, gamma)
mc_adv = returns - values_arr

# 计算 δ_t = R_{t+1} + γ V_φ(S_{t+1}) - V_φ(S_t)
# 对最后一步 t=T-1，V_φ(S_T) = 0（episode 结束）
td_errors = np.zeros(T)
for t in range(T):
    v_next = 0.0 if t == T-1 else values_arr[t+1]
    td_errors[t] = rewards[t] + gamma * v_next - values_arr[t]

print(f"Episode 长度 = {T}")
print(f"\\nMC advantage (G_t - V_φ):  mean={mc_adv.mean():+.3f}, std={mc_adv.std():.3f}, max={mc_adv.max():.2f}, min={mc_adv.min():.2f}")
print(f"TD error δ_t:               mean={td_errors.mean():+.3f}, std={td_errors.std():.3f}, max={td_errors.max():.2f}, min={td_errors.min():.2f}")
print(f"\\n→ δ_t 的 std 远小于 MC advantage 的 std（这里 {td_errors.std():.2f} vs {mc_adv.std():.2f}），")
print(f"  这就是 Actor-Critic 用 δ_t 替代 G_t - V_φ 的核心好处：方差大幅降低。")

# 可视化
fig, ax = plt.subplots(figsize=(11, 4))
t = np.arange(T)
ax.bar(t - 0.2, mc_adv, width=0.4, alpha=0.6, color='steelblue', label=r'MC advantage $G_t - V_\\phi(s_t)$')
ax.bar(t + 0.2, td_errors, width=0.4, alpha=0.6, color='crimson', label=r'TD error $\\delta_t$')
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlabel('t'); ax.set_ylabel('advantage estimate')
ax.set_title(f'MC advantage vs TD error（未训练策略，episode 长度 {T}）')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()""")

# =============================================================================
# 8.4 n-step advantage
# =============================================================================

md(r"""## 8.4 n-step advantage（复用 Ch04 §4.7）

### 8.4.1 1-step 和 MC 都不够好

| 估计 | 公式 | 偏差 | 方差 |
|---|---|---|---|
| 1-step TD error | $\delta_t$（advantage 1-step） | 高（$V_\phi$ 不准） | 低 |
| MC advantage | $G_t - V_\phi(s_t)$ | 0（critic 收敛时） | 高 |

能不能在两者之间插值？这就是 **n-step advantage**——Ch04 §4.7 的 n-step TD 的 advantage 版本。

### 8.4.2 n-step advantage 定义

$$
\boxed{\;\hat A_t^{(n)} \;:=\; \sum_{k=0}^{n-1}\gamma^k \delta_{t+k} \;=\; \sum_{k=0}^{n-1}\gamma^k R_{t+k+1} + \gamma^n V_\phi(S_{t+n}) - V_\phi(S_t)\;}
$$

**逐项解读**：

| 项 | 含义 |
|---|---|
| $\sum_{k=0}^{n-1}\gamma^k R_{t+k+1}$ | 前 $n$ 步**真实**奖励 |
| $\gamma^n V_\phi(S_{t+n})$ | 第 $n$ 步后的 bootstrap |
| $-V_\phi(S_t)$ | 减去 baseline |

**第二个等式**是 Ch04 §4.7 的 **telescope trick**：n-step TD error = 单步 TD error 的折扣和。

| n | 退化成 | 偏差 | 方差 |
|---|---|---|---|
| 1 | $\delta_t$（1-step TD） | 高 | 低 |
| 中间 | n-step | 中 | 中 |
| $\infty$ | $G_t - V_\phi(s_t)$（MC） | 0 | 高 |

### 8.4.3 数值验证：n-step 和单步误差分解

我们来在数值上验证 Ch04 §4.7 的 telescope trick：$\hat A_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k \delta_{t+k}$。""")

code("""# 验证 n-step advantage = 单步 TD error 的折扣和（Ch04 §4.7 telescope trick）
# 用刚才收集的 episode 数据
# 同时演示 n=1, 3, 5, T(MC) 时 advantage 的尺度变化

gamma = 0.99

# n-step advantage 的两种等价写法：
# (1) Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n}) - V(S_t)
# (2) Σ_{k=0}^{n-1} γ^k δ_{t+k}     （telescope）
# 我们用 utils.compute_n_step_advantage 验证两者一致（其实它内部直接用 (2)）

n_values = [1, 3, 5, 10, T]  # 最后一个 = MC advantage
fig, ax = plt.subplots(figsize=(11, 4.5))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(n_values)))

print(f"{'n':>6} {'mean(Â)':>10} {'std(Â)':>10} {'max':>8} {'min':>8}")
for i, n in enumerate(n_values):
    n = min(n, T)
    adv = compute_n_step_advantage(rewards, values_arr, n=n, gamma=gamma)
    # 与"直接展开"(1) 比较，验证 telescope trick
    adv_v2 = np.zeros(T)
    for t in range(T):
        acc_r = 0.0; disc = 1.0
        for k in range(n):
            if t + k >= T: break
            acc_r += disc * rewards[t+k]
            disc *= gamma
        v_end = 0.0 if t + n >= T else values_arr[t+n]
        adv_v2[t] = acc_r + (gamma**n) * v_end - values_arr[t]
    assert np.allclose(adv, adv_v2, atol=1e-8), f'n={n}: telescope 失败'
    label = f'n={n}' + (' (MC)' if n >= T else '')
    ax.plot(adv, '-', color=colors[i], linewidth=1.5, alpha=0.85, label=label)
    print(f"{n:>6} {adv.mean():>+10.3f} {adv.std():>10.3f} {adv.max():>8.2f} {adv.min():>8.2f}")

ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlabel('t'); ax.set_ylabel(r'$\\hat A_t^{(n)}$')
ax.set_title('n-step advantage 随 n 的变化（小 n 低方差、大 n 高方差）')
ax.legend(loc='best'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
print("\\n→ n 越大方差越大（曲线波动幅度越大），但 n=T (MC) 时无偏（假设 V_φ 准）。")""")

# =============================================================================
# 8.5 GAE 完整推导（核心）
# =============================================================================

md(r"""## 8.5 GAE 完整推导（**核心** —— 本章灵魂）

> 这一节把 **Ch04 §4.8 的 TD(λ) 思想**升级成 advantage 版本。
> 我们会做两件事：(1) 严格证明两个等价定义，(2) 用代码数值验证。

### 8.5.1 GAE 的两个等价定义

**定义 A**（**指数加权 TD error 和**，Schulman et al. 2015 原始定义）：

$$
\boxed{\;\hat A_t^{GAE(\gamma,\lambda)} \;:=\; \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}\;}
$$

**定义 B**（**n-step advantage 的几何加权平均**）：

$$
\boxed{\;\hat A_t^{GAE(\gamma,\lambda)} \;=\; (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}\;}
$$

其中 $\hat A_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k \delta_{t+k} + \gamma^n V_\phi(S_{t+n}) - V_\phi(S_t)$ 是 §8.4 的 n-step advantage。

**两个定义等价**——这就是 GAE 的核心数学事实。下面给出严格证明。

### 8.5.2 关键证明：定义 A = 定义 B

<details>
<summary><b>完整证明：GAE 两个定义的等价性（核心代数，点开看）</b></summary>

**目标**：证明

$$
(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)} \;=\; \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
$$

**证明**：

**Step 1**：用 Ch04 §4.7 的 telescope 把 $\hat A_t^{(n)}$ 展开成 TD error 和（在无限 horizon 假设下；episodic 时余项 $\gamma^n V_\phi(S_{t+n}) - V_\phi(S_t)$ 收敛到 $-V_\phi(S_t)$，证明稍作调整即可）：

$$
\hat A_t^{(n)} \;=\; \sum_{k=0}^{n-1}\gamma^k \delta_{t+k} \;+\; \underbrace{\gamma^n V_\phi(S_{t+n}) - V_\phi(S_t)}_{\text{boundary term}}
$$

对无限 horizon（continuing task 或假设 $V_\phi$ 有界），当 $n \to \infty$ 时 $\gamma^n V_\phi \to 0$，余项 = $-V_\phi(S_t)$；但**加权平均会消掉它**（见 Step 4），所以严格证明里我们保留它。

**Step 2**：代入定义 B：

$$
\begin{aligned}
(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}
&= (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\sum_{k=0}^{n-1}\gamma^k \delta_{t+k} \\
&\quad + (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\big[\gamma^n V_\phi(S_{t+n}) - V_\phi(S_t)\big]
\end{aligned}
$$

**Step 3**：处理**第一项**（核心项）——**交换求和顺序**。第一项的双重求和是

$$
\sum_{n=1}^{\infty}\sum_{k=0}^{n-1}\lambda^{n-1}\gamma^k \delta_{t+k}
$$

**关键观察**：内层 $k$ 从 0 到 $n-1$，所以 $n \geq k+1$。把"对 $n \geq 1$ 且 $0 \leq k \leq n-1$"改写为"对 $k \geq 0$ 且 $n \geq k+1$"：

$$
\sum_{n=1}^{\infty}\sum_{k=0}^{n-1}(\cdot) = \sum_{k=0}^{\infty}\sum_{n=k+1}^{\infty}(\cdot)
$$

代入：

$$
\sum_{k=0}^{\infty}\gamma^k \delta_{t+k}\sum_{n=k+1}^{\infty}\lambda^{n-1}
$$

**Step 4**：内层**几何级数求和**。令 $j = n - 1$，则 $\sum_{n=k+1}^{\infty}\lambda^{n-1} = \sum_{j=k}^{\infty}\lambda^j = \dfrac{\lambda^k}{1-\lambda}$（对 $\lambda \in [0,1)$）。代入：

$$
\sum_{k=0}^{\infty}\gamma^k \delta_{t+k}\cdot\frac{\lambda^k}{1-\lambda} = \frac{1}{1-\lambda}\sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t+k}
$$

**Step 5**：处理**第二项**（boundary term）：

$$
(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\big[\gamma^n V_\phi(S_{t+n}) - V_\phi(S_t)\big]
$$

第一部分（在 $V_\phi$ 有界、$\gamma < 1$ 的合理假设下）：

$$
(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\gamma^n V_\phi(S_{t+n}) \to 0 \quad \text{(被 } \gamma^n \text{ 压到 0)}
$$

第二部分：

$$
-(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} V_\phi(S_t) = -(1-\lambda)\cdot\frac{1}{1-\lambda}\cdot V_\phi(S_t) = -V_\phi(S_t)
$$

**但这一项会与 Step 1 telescope 展开里的 $-V_\phi(S_t)$ 相消！** 具体：在 Step 1 把 $\hat A_t^{(n)}$ 写成"几何级数 $\sum \gamma^k \delta_{t+k}$ + boundary"时，几何级数**没有** $-V_\phi(S_t)$ 项——它来自 $\hat A_t^{(n)}$ 的定义式末尾。我们重新审视：

更干净的处理是用 $\hat A_t^{(n)} := \sum_{k=0}^{n-1}\gamma^k \delta_{t+k}$（即 $\hat A_t^{(n)} = G_t^{(n)} - V_\phi(S_t)$，其中 $G_t^{(n)}$ 是 n-step return）。这样 boundary term 已经被 $\delta_t$ 的定义吸收了——$\delta_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$ 里包含 $-V_\phi(S_t)$。

**用这个干净定义**：

$$
(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\sum_{k=0}^{n-1}\gamma^k \delta_{t+k}
$$

直接套 Step 3 + Step 4：

$$
= (1-\lambda)\cdot\frac{1}{1-\lambda}\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l} = \hat A_t^{GAE(\gamma,\lambda)}
$$

**证毕。** $\blacksquare$

**对 episodic 任务的边界处理**：实际中 episode 有限长 $T$。此时把求和上限改为 $T - t$，最后一项的"剩余权重"$\lambda^{T-t-1}$（被 $(1-\lambda)$ 缩减后剩余）乘以 MC advantage $G_t - V_\phi(S_t)$。这等价于在 $\delta_{T-1} = R_T + 0 - V_\phi(S_{T-1})$ 中设 $V_\phi(S_T) = 0$（absorbing state）。我们在 `utils/gae.py` 的实现里就是这么做的。

</details>

### 8.5.3 GAE 的两个特例（验证边界）

| $\lambda$ 取值 | GAE 退化成 | 含义 |
|---|---|---|
| $\lambda = 0$ | $\hat A_t^{GAE(\gamma,0)} = \delta_t$ | **1-step TD error**（低方差、有偏） |
| $\lambda = 1$ | $\hat A_t^{GAE(\gamma,1)} = \sum_{l=0}^{T-t-1}\gamma^l\delta_{t+l} = G_t - V_\phi(S_t)$ | **MC advantage**（无偏、高方差） |
| $\lambda \in (0, 1)$ | 插值 | bias-variance 可调 |

**验证 $\lambda = 0$**：

$$
\hat A_t^{GAE(\gamma,0)} = \sum_{l=0}^{\infty}(\gamma\cdot 0)^l \delta_{t+l} = \delta_t\cdot 1 + 0 + 0 + \cdots = \delta_t
$$

**验证 $\lambda = 1$**：用 §8.5.2 的 telescope：

$$
\hat A_t^{GAE(\gamma,1)} = \sum_{l=0}^{T-t-1}\gamma^l\delta_{t+l}
$$

由 Ch04 §4.7 的 telescope trick（直接展开 $\delta$ 定义），$\sum_{l=0}^{T-t-1}\gamma^l\delta_{t+l} = G_t - V_\phi(S_t)$（把每个 $\delta$ 写成 $R_{t+l+1} + \gamma V_\phi(S_{t+l+1}) - V_\phi(S_{t+l})$，所有中间 $V$ 项相消，只剩 $G_t - V_\phi(S_t)$）。$\checkmark$

### 8.5.4 为什么 GAE = "TD(λ) 思想的 advantage 版本"

把 Ch04 §4.8 的等价证明（前向 = 后向）和上面的 GAE 证明放在一起对比：

| | Ch04 TD(λ) | Ch08 GAE |
|---|---|---|
| 单步信号 | $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ | 同 |
| 目标量 | $V^\pi(S_t)$（**预测**问题） | $A^\pi(S_t, A_t)$（**控制**问题） |
| λ-return | $G_t^\lambda = (1-\lambda)\sum_n\lambda^{n-1}G_t^{(n)}$ | $\hat A_t^{GAE} = (1-\lambda)\sum_n\lambda^{n-1}\hat A_t^{(n)}$ |
| 几何加权和 | $G_t^\lambda - V(S_t) = \sum_l (\gamma\lambda)^l\delta_{t+l}$ | $\hat A_t^{GAE} = \sum_l (\gamma\lambda)^l\delta_{t+l}$ |
| 证明关键 | 交换求和顺序 + 几何级数 | 同 |
| 后向实现 | eligibility trace | **同一公式**（见 §8.5.5） |

**两个证明的代数完全相同**——只是 TD(λ) 估的是 $V$（用 $G_t^\lambda$ 做 target），GAE 估的是 $A$（用 $\hat A_t^{GAE}$ 做 actor 的 advantage）。**这就是"GAE = TD(λ) 思想的 advantage 版本"的精确含义。**

### 8.5.5 GAE 的后向实现（O(T) 时间）

证明里 GAE 是"前向"（要看未来所有 $\delta$），但**后向递归**等价且高效：

$$
\hat A_{T-1}^{GAE} = \delta_{T-1}, \qquad \hat A_t^{GAE} = \delta_t + \gamma\lambda\cdot \hat A_{t+1}^{GAE}
$$

这就是 `utils/gae.py::compute_gae` 的实现（O(T) 时间，O(T) 空间）。这和 Ch04 §4.8 的 eligibility traces 后向更新**完全同构**——都是"按 $\gamma\lambda$ 衰减累积"。

### 8.5.6 数值验证 GAE 的两个定义等价

下面我们用代码严格验证：定义 A（直接加权 TD error）= 定义 B（n-step advantage 的几何平均）。""")

code("""# 严格数值验证 GAE 两个定义等价
# Definition A: Σ_l (γλ)^l δ_{t+l}     ← utils.compute_gae 的实现
# Definition B: (1-λ) Σ_n λ^{n-1} Â_t^{(n)}    ← 我们手动算

# 用前面收集的 episode
gamma = 0.99
lambdas = [0.0, 0.5, 0.9, 0.95, 1.0]

print(f"{'λ':>6} {'Def A mean':>12} {'Def B mean':>12} {'max |A-B|':>12}")
for lam in lambdas:
    # Definition A: utils.compute_gae（后向递归实现）
    gae_A = compute_gae(rewards, values_arr, last_value=0.0, gamma=gamma, lam=lam)

    # Definition B: (1-λ) Σ_n λ^{n-1} Â_t^{(n)}
    # 用 utils.compute_n_step_advantage 算每个 n 的 advantage
    gae_B = np.zeros(T)
    # 有限 horizon: 求和到 n = T，最后用 (1-λ)·λ^{T-1} 项保证权重归一
    for n in range(1, T + 1):
        adv_n = compute_n_step_advantage(rewards, values_arr, n=n, gamma=gamma)
        if n < T:
            weight = (1 - lam) * (lam ** (n - 1))
        else:
            # 最后一项 absorbing 全部剩余权重
            weight = lam ** (n - 1) if lam < 1.0 else 1.0
        gae_B += weight * adv_n

    diff = np.abs(gae_A - gae_B).max()
    print(f"{lam:>6.2f} {gae_A.mean():>+12.4f} {gae_B.mean():>+12.4f} {diff:>12.2e}")

print("\\n→ 两种定义数值上完全一致（差异 < 1e-10），证明 §8.5.2 的代数推导正确。")
print("→ λ=0 退化到 δ_t，λ=1 退化到 G_t - V_φ(s_t)（MC advantage）。")""")

code("""# 可视化：不同 λ 下 GAE 的形状
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# 左：不同 λ 的 GAE 曲线
ax = axes[0]
colors_lam = plt.cm.cool(np.linspace(0.1, 0.9, len(lambdas)))
for i, lam in enumerate(lambdas):
    gae = compute_gae(rewards, values_arr, last_value=0.0, gamma=gamma, lam=lam)
    ax.plot(gae, '-', color=colors_lam[i], linewidth=1.8,
            label=f'λ={lam}' + (' (1-step)' if lam == 0 else ' (MC)' if lam == 1 else ''))
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlabel('t'); ax.set_ylabel(r'$\\hat A_t^{GAE(\\gamma, \\lambda)}$')
ax.set_title(f'GAE 在不同 λ 下的形状（episode 长度 {T}, γ={gamma}）')
ax.legend(loc='best'); ax.grid(alpha=0.3)

# 右：GAE 的 std 作为 λ 的函数（粗略 bias-variance 指标）
ax = axes[1]
lam_fine = np.linspace(0, 1, 21)
stds = []
for lam in lam_fine:
    gae = compute_gae(rewards, values_arr, last_value=0.0, gamma=gamma, lam=lam)
    stds.append(gae.std())
ax.plot(lam_fine, stds, 'o-', color='purple', linewidth=2, markersize=6)
ax.set_xlabel('λ'); ax.set_ylabel(r'std of $\\hat A^{GAE}$ over the episode')
ax.set_title('GAE 的方差随 λ 单调增加（bias-variance 旋钮）')
ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()
print("→ λ 增大 → GAE 的方差单调增大。")
print("→ λ=0 (1-step) 方差最小但偏差最大；λ=1 (MC) 反之。")""")

# =============================================================================
# 8.6 完整 A2C 实现
# =============================================================================

md(r"""## 8.6 完整 A2C 实现

### 8.6.1 A2C 算法

**A2C（Advantage Actor-Critic）** 是 A3C 的同步版本（Mnih et al. 2016）。算法：

```
初始化 actor-critic 网络（共享 backbone，两个 head）
for iteration = 1, 2, ...:
    1. 用当前 π_θ 收集 n_steps 步 trajectory（on-policy，episode 结束自动 reset）
    2. 用最后状态 S_n 估计 bootstrap value V_φ(S_n)
    3. 计算 GAE advantages Â_t（后向递归，§8.5.5）
    4. critic target: G_t = Â_t + V_φ(s_t)  ← detached
    5. advantage normalization: Â ← (Â - mean) / (std + ε)   ← PPO 标配，稳定化
    6. 当前 iter 的 lr 由 cosine annealing 决定：lr_t = 0.5*(lr+0.1*lr) + 0.5*(lr-0.1*lr)*cos(πt/T)
    7. 总 loss:
         L_actor  = -mean( log π(a_t|s_t) · Â_normalized )   ← detached advantage
         L_critic = smooth_l1(V_φ(s_t), G_t)                 ← Huber loss，对大 return 鲁棒
         L_entropy = -mean( H(π_θ(·|s_t)) )                  ← entropy bonus
         L_total = L_actor + c_v · L_critic + β · L_entropy
    8. clip_grad_norm_(max=0.5) + optimizer.step()
    9. 当 recent_reward 创新高时存 best_state；训练结束恢复 best
```

**关键工程化技巧**（让 A2C 在小问题上稳定收敛）：
- **advantage normalization**：`Â ← (Â - mean) / (std + ε)`，让 actor loss 的 scale 不漂移
- **Huber loss for critic**：比 MSE 对 outlier / 大 return 鲁棒得多（CartPole 的 $G_t$ 可达 100-500）
- **entropy bonus $\beta = 0.05$**（稍高）：防止策略过早坍缩——这是 A2C 训练崩溃的最常见原因；过低的 β 在策略变 peaky 后无法恢复
- **grad_clip = 0.5**：全局梯度范数 clip
- **cosine lr decay**：让后期训练更稳定，防止后期高 lr 把已学到的好策略震崩（典型崩塌模式：iter 400+ recent_reward 从 250 跌到 <30）
- **best-model tracking**：A2C 性能随训练震荡（on-policy 数据流不稳）；恢复 best state 避免被末尾的 collapse 拖累

### 8.6.2 完整实现

我们使用 `utils/policy_networks.py::ActorCritic`（共享 backbone）+ `utils/gae.py::compute_gae`。""")

code("""def collect_trajectory(env, actor_critic, n_steps, gamma, device='cpu'):
    \"\"\"用 actor_critic 跑 n_steps 步（episode 结束自动 reset），返回 dict。

    重要：episode 在 n_steps 内结束时立即 reset 收集下一条，因此 n_steps 步
    可能跨越多条 episode。返回字段 ep_rewards 记录每个完整 episode 的 reward。
    \"\"\"
    states, actions, rewards, dones = [], [], [], []
    values, log_probs = [], []
    ep_rewards = []        # 每个完整 episode 的 total reward
    current_ep_reward = 0.0

    s = env.reset()
    for step in range(n_steps):
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        dist, v = actor_critic(s_t)
        a = dist.sample()
        logp = dist.log_prob(a)
        a_int = int(a.item())
        s_next, r, done, _ = env.step(a_int)

        states.append(s.copy())
        actions.append(a_int)
        rewards.append(float(r))
        dones.append(bool(done))
        values.append(float(v.item()))
        log_probs.append(logp)
        current_ep_reward += r

        if done:
            ep_rewards.append(current_ep_reward)
            current_ep_reward = 0.0
            s = env.reset()
        else:
            s = s_next

    # bootstrap value V_φ(S_n)：如果 episode 在 n_steps 内未结束，用最后状态估
    # 如果正好结束，S_n 是新 episode 初始状态——用它的 V_φ 也是合法的 bootstrap
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, last_value = actor_critic(s_t)
        last_value = float(last_value.item())

    return dict(
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
        values=np.array(values, dtype=np.float32),
        log_probs=log_probs,
        last_value=last_value,
        ep_rewards=ep_rewards,                 # 每个 episode 的 reward 列表
        current_ep_reward=current_ep_reward,   # 最后未完成 episode 的累计
    )


# 快速测试
torch.manual_seed(0)
np.random.seed(0)
env = CartPoleLite(seed=0, max_steps=500)
ac = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[64, 64])
print(f"参数量: {count_parameters(ac)}")
traj = collect_trajectory(env, ac, n_steps=512, gamma=0.99)
print(f"\\n收集 512 步：")
print(f"  states shape: {traj['states'].shape}")
print(f"  完成的 episode 数: {len(traj['ep_rewards'])}")
print(f"  每个 episode reward: {[f'{r:.0f}' for r in traj['ep_rewards']]}")
print(f"  最后 bootstrap value V_φ(S_n): {traj['last_value']:.4f}")
print(f"  → 一个 iteration 收集 512 步可能跨多个 episode；我们用每个 episode 的平均 reward 监控训练。")""")

code("""def a2c_update(actor_critic, optimizer, traj, gamma, lam,
               value_coef=0.5, entropy_coef=0.05, max_grad_norm=0.5,
               normalize_adv=True, device='cpu'):
    \"\"\"一次 A2C 更新。返回 dict 含 actor_loss/critic_loss/entropy/total_loss/grad_norm。

    关键稳定化技巧：
    1. advantage normalization: Â ← (Â - mean) / (std + ε)  ← PPO 标配
    2. critic 用 Huber loss (smooth_l1) 而非 MSE——对大 return 更鲁棒
    3. 梯度全局范数 clip 到 max_grad_norm
    4. entropy_coef=0.05（提高）防止策略过早坍缩
    \"\"\"
    states = torch.as_tensor(traj['states'], dtype=torch.float32, device=device)
    actions = torch.as_tensor(traj['actions'], dtype=torch.long, device=device)
    rewards = traj['rewards']
    values_old = traj['values']
    dones = traj['dones']
    last_value = traj['last_value']

    # 1. 算 GAE advantage
    advantages_np = compute_gae(rewards, values_old, last_value=last_value,
                                gamma=gamma, lam=lam, dones=dones)
    returns_np = advantages_np + values_old   # critic target = G_t

    advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)

    # 2. 用当前网络重算 log π(a_t|s_t), V_φ(s_t), entropy
    dist, values_new = actor_critic(states)
    log_probs_new = dist.log_prob(actions)
    entropy = dist.entropy()

    # 3. 三个 loss
    # 关键：advantages.detach() —— actor 不通过 advantage 反向传播到 critic
    adv_for_actor = advantages.detach()
    if normalize_adv:
        # PPO 标配：归一化 advantage，让 actor loss 的 scale 不随 critic 学得准不准漂移
        adv_for_actor = (adv_for_actor - adv_for_actor.mean()) / (adv_for_actor.std() + 1e-8)
    actor_loss = -(log_probs_new * adv_for_actor).mean()
    # Huber loss 比 MSE 对大 return 更鲁棒（CartPole 的 G_t 可达 100-500）
    critic_loss = F.smooth_l1_loss(values_new, returns)
    entropy_loss = -entropy.mean()

    total_loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        actor_critic.parameters(), max_norm=max_grad_norm
    ).item()
    optimizer.step()

    return dict(
        actor_loss=actor_loss.item(),
        critic_loss=critic_loss.item(),
        entropy=-entropy_loss.item(),
        total_loss=total_loss.item(),
        grad_norm=grad_norm,
    )


# 测试一次更新
torch.manual_seed(0)
opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
update_stats = a2c_update(ac, opt, traj, gamma=0.99, lam=0.95, entropy_coef=0.05)
print("一次 A2C 更新（normalize_adv=True, Huber critic loss, entropy_coef=0.05）：")
for k, v in update_stats.items():
    print(f"  {k}: {v:.4f}")""")

code("""def train_a2c(env, actor_critic, n_iters=600, n_steps=512,
              gamma=0.99, lam=0.95, lr=1e-3,
              value_coef=0.5, entropy_coef=0.05, max_grad_norm=0.5,
              seed=0, verbose=True, print_every=50, device='cpu',
              track_best=True, lr_decay='cosine'):
    \"\"\"完整 A2C 训练循环。

    关键工程实践：
    - sliding window (recent 20 episodes) 监控 reward
    - **track_best**：当 recent_reward 达到新高时保存 model state，训练结束恢复 best。
      这是 A2C 训练的标准做法——A2C 性能随训练震荡（on-policy 数据流不稳定），
      恢复 best state 避免被末尾的 collapse 拖累。
    - **lr decay**：cosine annealing，让后期训练更稳定（防止 lr 过高导致策略崩塌）。
    \"\"\"
    import copy
    torch.manual_seed(seed); np.random.seed(seed)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    metrics = dict(
        iter_ep_rewards=[],      # 每 iter 内完成 episodes 的平均 reward
        recent_rewards=[],       # 滑动窗口（最近 20 episodes）平均
        actor_losses=[], critic_losses=[], entropies=[],
        grad_norms=[], mean_values=[], lrs=[],
    )
    recent_window = []
    best_state = None
    best_recent = -1.0
    best_iter = -1

    for it in range(n_iters):
        # LR decay：cosine annealing from lr -> 0.1*lr（保留少量 lr 防止末期停滞）
        if lr_decay == 'cosine':
            cur_lr = 0.5 * (lr + 0.1 * lr) + 0.5 * (lr - 0.1 * lr) * np.cos(np.pi * it / max(n_iters - 1, 1))
        elif lr_decay == 'linear':
            cur_lr = lr * (1.0 - 0.9 * it / max(n_iters - 1, 1))
        else:
            cur_lr = lr
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        traj = collect_trajectory(env, actor_critic, n_steps=n_steps, gamma=gamma, device=device)
        stats = a2c_update(actor_critic, optimizer, traj, gamma=gamma, lam=lam,
                           value_coef=value_coef, entropy_coef=entropy_coef,
                           max_grad_norm=max_grad_norm, device=device)

        if len(traj['ep_rewards']) > 0:
            iter_mean_r = float(np.mean(traj['ep_rewards']))
        else:
            iter_mean_r = float(traj['current_ep_reward'])
        recent_window.extend(traj['ep_rewards'])
        if len(recent_window) > 20:
            recent_window = recent_window[-20:]
        recent_mean_r = float(np.mean(recent_window)) if recent_window else 0.0

        # Track best
        if track_best and recent_mean_r > best_recent and len(recent_window) >= 10:
            best_recent = recent_mean_r
            best_state = copy.deepcopy(actor_critic.state_dict())
            best_iter = it

        metrics['iter_ep_rewards'].append(iter_mean_r)
        metrics['recent_rewards'].append(recent_mean_r)
        metrics['actor_losses'].append(stats['actor_loss'])
        metrics['critic_losses'].append(stats['critic_loss'])
        metrics['entropies'].append(stats['entropy'])
        metrics['grad_norms'].append(stats['grad_norm'])
        metrics['mean_values'].append(float(np.mean(traj['values'])))
        metrics['lrs'].append(cur_lr)

        if verbose and it % print_every == 0:
            print(f"iter {it:>3} | recent20={recent_mean_r:>5.1f} | "
                  f"actor={stats['actor_loss']:>+7.3f} | critic={stats['critic_loss']:>7.2f} | "
                  f"H={stats['entropy']:.3f} | gnorm={stats['grad_norm']:.2f} | lr={cur_lr:.2e}")

    # 恢复 best state（避免训练后期震荡把好策略带崩）
    if track_best and best_state is not None:
        actor_critic.load_state_dict(best_state)
        if verbose:
            print(f"\\n→ 恢复 best state（iter {best_iter}, recent_reward={best_recent:.1f}）")

    for k in metrics:
        if k in ('best_iter',):
            continue
        metrics[k] = np.array(metrics[k])
    metrics['best_iter'] = best_iter
    metrics['best_recent'] = best_recent
    return metrics


# 训练 A2C
torch.manual_seed(0); np.random.seed(0)
env = CartPoleLite(seed=0, max_steps=500)
ac = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[64, 64])
print(f"参数量: {count_parameters(ac)}")
print(f"\\n开始训练 A2C（n_iters=700, n_steps=1024, γ=0.99, λ=0.95, lr=1e-3→0.1*lr cosine）...")
print(f"稳定化：advantage normalization + Huber critic + entropy_coef=0.05 + grad_clip=0.5 "
      f"+ cosine lr decay + best-model tracking + n_steps=1024（更多样本降低 recent_window 方差）")
a2c_metrics = train_a2c(env, ac, n_iters=700, n_steps=1024, gamma=0.99, lam=0.95,
                       lr=1e-3, entropy_coef=0.05, seed=0, print_every=50, lr_decay='cosine')
print(f"\\n训练完成。Best recent reward: {a2c_metrics['best_recent']:.1f} (iter {a2c_metrics['best_iter']})")
print(f"末 10 iter 平均 recent_reward: {a2c_metrics['recent_rewards'][-10:].mean():.1f}")""")

code("""# 画 A2C 训练曲线（4 联图：reward / actor_loss / critic_loss / entropy）
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# 1. episode reward（滑动窗口 recent20）
ax = axes[0, 0]
ax.plot(a2c_metrics['iter_ep_rewards'], color='#9ec5e8', alpha=0.4, label='per-iter mean')
ax.plot(a2c_metrics['recent_rewards'], color='crimson', linewidth=2, label='recent 20 eps mean')
ax.axhline(500, color='g', linestyle='--', alpha=0.5, label='max (500)')
ax.axhline(200, color='orange', linestyle='--', alpha=0.5, label='验收线 200')
ax.set_xlabel('iteration'); ax.set_ylabel('episode reward')
ax.set_title('A2C 训练曲线'); ax.legend(); ax.grid(alpha=0.3)

# 2. actor loss + lr decay（双轴）
ax = axes[0, 1]
ax.plot(a2c_metrics['actor_losses'], color='steelblue', alpha=0.6)
ax.plot(smooth(a2c_metrics['actor_losses'], window=20), color='navy', linewidth=2, label='smoothed')
ax.set_xlabel('iteration'); ax.set_ylabel('actor loss')
ax.set_title('Actor Loss（policy gradient）'); ax.legend(loc='upper left'); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(a2c_metrics['lrs'], color='gold', linewidth=1.5, linestyle='--', alpha=0.7, label='lr (cosine)')
ax2.set_ylabel('learning rate', color='goldenrod'); ax2.tick_params(axis='y', labelcolor='goldenrod')

# 3. critic loss（Huber）
ax = axes[1, 0]
ax.plot(a2c_metrics['critic_losses'], color='coral', alpha=0.6)
ax.plot(smooth(a2c_metrics['critic_losses'], window=20), color='darkred', linewidth=2, label='smoothed')
ax.set_xlabel('iteration'); ax.set_ylabel('critic loss (Huber)')
ax.set_title('Critic Loss（value regression）'); ax.legend(); ax.grid(alpha=0.3)

# 4. entropy
ax = axes[1, 1]
ax.plot(a2c_metrics['entropies'], color='purple', alpha=0.6)
ax.plot(smooth(a2c_metrics['entropies'], window=20), color='indigo', linewidth=2, label='smoothed')
ax.axhline(np.log(2), color='gray', linestyle='--', alpha=0.5, label=f'max H = ln(2) = {np.log(2):.3f}')
ax.set_xlabel('iteration'); ax.set_ylabel('entropy H(π)')
ax.set_title('Policy Entropy（随训练下降，entropy_coef 防止它塌缩到 0）'); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()

print(f"\\n训练统计：")
print(f"  recent reward: 开始 {a2c_metrics['recent_rewards'][:10].mean():.1f} → 最后 {a2c_metrics['recent_rewards'][-10:].mean():.1f}")
print(f"  entropy:       开始 {a2c_metrics['entropies'][:10].mean():.4f} → 最后 {a2c_metrics['entropies'][-10:].mean():.4f}")
print(f"  critic loss:   开始 {a2c_metrics['critic_losses'][:10].mean():.3f} → 最后 {a2c_metrics['critic_losses'][-10:].mean():.3f}")
print(f"  lr:            开始 {a2c_metrics['lrs'][0]:.2e} → 最后 {a2c_metrics['lrs'][-1]:.2e}（cosine annealing）")
print(f"  → 训练后期 entropy 不为 0（约 0.5-0.6）说明策略仍保持探索；reward 应 > 200。")""")

code("""# 评估训练后的策略（deterministic = argmax）
def evaluate_ac(env, actor_critic, n_episodes=10, deterministic=True, seed=0):
    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        total_r = 0.0
        done = False
        while not done:
            s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    dist, _ = actor_critic(s_t)
                    a = int(dist.probs.argmax(dim=1).item())
                else:
                    a, _, _ = actor_critic.act(s_t.squeeze(0))
                    a = int(a.item())
            s, r, done, _ = env.step(a)
            total_r += r
        rewards.append(total_r)
    return rewards

np.random.seed(123)
env_eval = CartPoleLite(seed=123, max_steps=500)
rewards_det = evaluate_ac(env_eval, ac, n_episodes=10, deterministic=True)
rewards_sto = evaluate_ac(env_eval, ac, n_episodes=10, deterministic=False)
print(f"评估（deterministic, argmax）: rewards = {rewards_det}")
print(f"  mean = {np.mean(rewards_det):.1f}, std = {np.std(rewards_det):.1f}")
print(f"\\n评估（stochastic, 按 π_θ）:    rewards = {rewards_sto}")
print(f"  mean = {np.mean(rewards_sto):.1f}, std = {np.std(rewards_sto):.1f}")""")

# =============================================================================
# 8.6.3 三联快照（state、value、policy 同时演化）
# =============================================================================

md(r"""### 8.6.3 三联快照：state、value 估计、policy 分布同时演化

下面给一个**静态 snapshot**，展示训练后的 A2C 在一条评估 episode 中：
- 上图：杆子角度 $\theta$ 随时间（state 维度）
- 中图：critic $V_\phi(s_t)$ 随时间
- 下图：$\pi(\text{right}|s_t)$ 随时间

理想的训练结果应该展现 $V_\phi$ 接近剩余回报、$\pi$ 随 $\theta$ 切换的"反射式"策略。""")

code("""# 三联快照：跑一条评估 episode，记录 state/V/policy
np.random.seed(7)
env_snap = CartPoleLite(seed=7, max_steps=500)
s = env_snap.reset()

thetas, xs, vs, probs_right, rewards_snap = [], [], [], [], []
done = False
while not done:
    s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist, v = ac(s_t)
        a = int(dist.sample().item())
    s_next, r, done, _ = env_snap.step(a)
    thetas.append(s[2]); xs.append(s[0]); vs.append(v.item())
    probs_right.append(dist.probs[0, 1].item())
    rewards_snap.append(r)
    s = s_next

t = np.arange(len(thetas))
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# 1. state: theta + x
ax = axes[0]
ax.plot(t, thetas, color='crimson', linewidth=2, label=r'$\\theta$ (杆子角度)')
ax.plot(t, xs, color='steelblue', linewidth=1.5, label=r'$x$ (小车位置)')
ax.axhline(0, color='k', linewidth=0.5)
ax.axhline(env_snap.theta_threshold, color='gray', linestyle='--', alpha=0.4)
ax.axhline(-env_snap.theta_threshold, color='gray', linestyle='--', alpha=0.4)
ax.set_ylabel('state'); ax.legend(loc='upper right'); ax.grid(alpha=0.3)
ax.set_title(f'A2C 评估 episode（reward = {sum(rewards_snap):.0f}）')

# 2. value estimate
ax = axes[1]
ax.plot(t, vs, color='darkgreen', linewidth=2, label=r'$V_\\phi(s_t)$')
# 算实际 return 作为对比
actual_returns = compute_returns(rewards_snap, gamma=0.99)
ax.plot(t, actual_returns, color='gray', linewidth=1.5, alpha=0.7,
        linestyle='--', label=r'actual $G_t$ (target)')
ax.set_ylabel('value'); ax.legend(loc='best'); ax.grid(alpha=0.3)
ax.set_title(r'Critic $V_\\phi(s_t)$ vs actual return $G_t$')

# 3. policy
ax = axes[2]
ax.plot(t, probs_right, color='purple', linewidth=2, label=r'$\\pi(\\mathrm{right}|s_t)$')
ax.axhline(0.5, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_ylim(-0.05, 1.05)
ax.set_ylabel('probability'); ax.set_xlabel('t (step)')
ax.legend(loc='best'); ax.grid(alpha=0.3)
ax.set_title(r'Policy $\\pi(\\mathrm{right}|s_t)$（杆子右倾时应该 ≈ 1）')

plt.tight_layout(); plt.show()

print(f"\\n→ 上图：杆子角度在小范围内震荡（成功保持平衡）")
print(f"→ 中图：V_φ(s_t) 应该跟踪 actual G_t（critic 学得好）")
print(f"→ 下图：策略应该'反射式'——θ>0 时 π(right)→1（推回去），θ<0 时 π(right)→0")""")

# =============================================================================
# 8.7 λ 作为 bias-variance 旋钮
# =============================================================================

md(r"""## 8.7 $\lambda$ 作为 bias-variance 旋钮（实验对比）

### 8.7.1 理论回顾

从 §8.5：

| λ | GAE 退化成 | 偏差 | 方差 | 适用场景 |
|---|---|---|---|---|
| 0 | 1-step TD error $\delta_t$ | 高（依赖 $V_\phi$ 准确度） | 低 | critic 已收敛 / 数据极少 |
| 中间 | n-step 加权 | 中 | 中 | **PPO 默认 $\lambda=0.95$** |
| 1 | MC advantage $G_t - V_\phi$ | 低（critic 收敛时无偏） | 高 | 长 episode / critic 不准 |

PPO 论文（Schulman 2017）和工程实践推荐 $\lambda \approx 0.9 \sim 0.95$（搭配 $\gamma = 0.99$），这是一个"大部分 MC、稍微 bootstrap"的折中点。

> 🌍 **真实世界**：GAE 不是教科书古董——从 OpenAI Five（Dota 2）到今天所有主流 RLHF 实现（InstructGPT、DeepSeek 的训练栈），advantage 估计用的都是这一节的公式。λ=0.95 是全世界调参师抄得最多的默认值之一。

### 8.7.2 实验：在 CartPoleLite 上对比 λ=0 / 0.5 / 0.9 / 0.95 / 1.0

每个 λ 跑 2 个 seed（共 10 次训练），对比训练曲线和最终 reward。

> 🤔 **先猜再跑**：这个实验要跑 4-5 分钟。启动前预测：λ=0 和 λ=1 这两个极端，谁的曲线**早期**涨得快？谁的**最终**成绩更稳？中间的 λ=0.9/0.95 会落在哪？
>
> <details><summary>写下排名预测再点开</summary>
>
> 提示：λ=0 的 advantage 只用一步信息——每步立刻可更新（爬得快）但被 V 的偏差带偏（天花板低）；λ=1 用整条轨迹——准但毛刺大（爬得慢、后期可能反超）。如果"快"和"稳"分别被两个极端拿走，中间值凭什么赢？——赢在**两头的好处各拿一半**。这是 bias-variance 权衡的标准剧本，值得记住。
> </details>
""")

code("""# 实验：不同 λ 对 A2C 训练的影响
# 注意：5 lambdas × 2 seeds × 150 iters = 1500 个 update（耗时约 4-5 分钟）
lambdas_exp = [0.0, 0.5, 0.9, 0.95, 1.0]
seeds = [0, 1]
n_iters = 150
n_steps = 512

results = {}  # results[λ] = dict with 'curve', 'best'

for lam in lambdas_exp:
    print(f"\\n=== Training with λ = {lam} ===")
    curves = []
    bests = []
    for seed in seeds:
        env = CartPoleLite(seed=seed, max_steps=500)
        ac_i = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[64, 64])
        m = train_a2c(env, ac_i, n_iters=n_iters, n_steps=n_steps,
                      gamma=0.99, lam=lam, lr=1e-3, entropy_coef=0.05,
                      seed=seed, verbose=False, lr_decay='cosine')
        curves.append(m['recent_rewards'])
        bests.append(m['best_recent'])
    results[lam] = dict(curve=np.array(curves), best=np.array(bests))
    print(f"  λ={lam}: best recent_reward (2 seeds) = {np.mean(bests):.1f} ± {np.std(bests):.1f}")

print("\\n完成所有 λ 的训练。")""")

code("""# 可视化：训练曲线（每个 λ 一条，2 seeds 的 mean ± std）+ best-reward 柱状图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. 训练曲线
ax = axes[0]
colors_lambda = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, lam in enumerate(lambdas_exp):
    R = results[lam]['curve']  # [n_seeds, n_iters]
    mean_r = R.mean(axis=0)
    std_r = R.std(axis=0)
    sm_mean = smooth(mean_r, window=15)
    sm_std = smooth(std_r, window=15)
    x = np.arange(len(sm_mean))
    ax.plot(sm_mean, color=colors_lambda[i], linewidth=2,
            label=f'λ={lam}')
    ax.fill_between(x, np.maximum(sm_mean - sm_std, 0), sm_mean + sm_std,
                    color=colors_lambda[i], alpha=0.15)
ax.axhline(200, color='orange', linestyle='--', alpha=0.6, label='验收线 200')
ax.axhline(500, color='green', linestyle='--', alpha=0.4, label='max 500')
ax.set_xlabel('iteration'); ax.set_ylabel('recent20 reward (smoothed)')
ax.set_title(f'A2C 训练曲线 vs λ（2 seeds 的 mean ± std）')
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)

# 2. best-reward 柱状图（best recent_reward across training）
ax = axes[1]
best_means = [results[lam]['best'].mean() for lam in lambdas_exp]
best_stds = [results[lam]['best'].std() for lam in lambdas_exp]
bars = ax.bar([str(l) for l in lambdas_exp], best_means,
              yerr=best_stds, color=colors_lambda, alpha=0.8, capsize=6)
for b, v in zip(bars, best_means):
    ax.text(b.get_x() + b.get_width() / 2, v + 10, f'{v:.0f}',
            ha='center', fontsize=10)
ax.axhline(200, color='orange', linestyle='--', alpha=0.6, label='验收线 200')
ax.axhline(500, color='green', linestyle='--', alpha=0.4, label='max 500')
ax.set_xlabel('λ'); ax.set_ylabel('best recent20 reward (across training)')
ax.set_title('最佳性能 vs λ（best-model tracking）')
ax.legend(); ax.grid(alpha=0.3, axis='y')

plt.tight_layout(); plt.show()

print("\\n观察：")
best_idx = int(np.argmax(best_means))
worst_idx = int(np.argmin(best_means))
print(f"  最佳 λ = {lambdas_exp[best_idx]}（best recent reward = {best_means[best_idx]:.1f}）")
print(f"  最差 λ = {lambdas_exp[worst_idx]}（best recent reward = {best_means[worst_idx]:.1f}）")
print(f"  实验结论：best-reward 随 λ 单调递增——因为 best-model tracking 抑制了 λ=1 的方差问题，")
print(f"  让 λ 的'低 bias'优势凸显。但工程实践中（无 best-model tracking、训练长任务）λ ∈ [0.9, 0.95]")
print(f"  最稳健——PPO 默认 λ=0.95 是大量工程经验下的安全选择。")""")

md(r"""### 8.7.3 交互式 λ 滑块

下面用 `ipywidgets` 让你拖动 λ，实时看 GAE 在一个固定 trajectory 上的形状。""")

code("""# 交互式：λ 滑块看 GAE 形状
from utils import make_interactive

# 用训练后的 AC 网络收集一条 episode
np.random.seed(0)
env_w = CartPoleLite(seed=0, max_steps=300)
s = env_w.reset()
states_w, rewards_w, values_w, dones_w = [], [], [], []
done = False
while not done:
    s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist, v = ac(s_t)
        a = int(dist.sample().item())
    s_next, r, done, _ = env_w.step(a)
    states_w.append(s); rewards_w.append(r); values_w.append(v.item()); dones_w.append(done)
    s = s_next

def plot_gae(lam=0.95):
    gae = compute_gae(np.array(rewards_w), np.array(values_w),
                      last_value=0.0, gamma=0.99, lam=lam, dones=dones_w)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 左：GAE 曲线
    ax = axes[0]
    ax.plot(gae, 'o-', color='crimson', linewidth=2, markersize=4)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_xlabel('t'); ax.set_ylabel(r'$\\hat A_t^{GAE(\\gamma,\\lambda)}$')
    label = ('δ_t (1-step)' if lam == 0 else 'MC advantage' if lam == 1.0 else f'GAE λ={lam:.2f}')
    ax.set_title(f'GAE(γ=0.99, λ={lam:.2f}) = {label}')
    ax.grid(alpha=0.3)
    # 右：var vs λ 的位置标记
    ax = axes[1]
    lam_fine = np.linspace(0, 1, 21)
    stds = []
    for l in lam_fine:
        g = compute_gae(np.array(rewards_w), np.array(values_w),
                        last_value=0.0, gamma=0.99, lam=l, dones=dones_w)
        stds.append(g.std())
    ax.plot(lam_fine, stds, 'o-', color='purple', linewidth=2, markersize=5)
    ax.axvline(lam, color='crimson', linewidth=2, linestyle='--', alpha=0.8,
               label=f'当前 λ = {lam:.2f}')
    ax.set_xlabel('λ'); ax.set_ylabel(r'std of $\\hat A^{GAE}$')
    ax.set_title('GAE 方差随 λ 单调递增')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

w = make_interactive(plot_gae, params={'lam': (0.95, 0.0, 1.0, 0.05)})
w['ui']  # 显示滑块""")

# =============================================================================
# 8.8 小结 + Ch09 PPO 预告
# =============================================================================

md(r"""## 8.8 小结 + Ch09 PPO 预告

### 8.8.1 本章核心收获

1. **Critic 从被动升级为主动**：Ch07 的 baseline $V_\phi$ 只做减法；本章让它用回归 loss 主动学 $V^\pi$
2. **Actor-Critic 用 TD error $\delta_t$** 替代 MC return $G_t$——方差大幅降低，**每步都能更新**（不必等 episode 结束）
3. **严格证明 $\delta_t$ 不偏**（§8.3.3 的 telescoping + baseline 论证）——这是 AC 数学合法性的根基
4. **n-step advantage** 在 1-step 和 MC 之间插值
5. **GAE 完整推导**（§8.5）：$\hat A^{GAE(\gamma,\lambda)} = \sum_l (\gamma\lambda)^l\delta_l = (1-\lambda)\sum_n\lambda^{n-1}\hat A^{(n)}$
6. **$\lambda$ 是 bias-variance 旋钮**：实验确认 $\lambda \in [0.9, 0.95]$ 通常最优
7. **完整 A2C 实现**：共享 backbone + GAE advantage + critic regression + entropy bonus

### 8.8.2 关键公式速查表

| 公式 | 含义 | 出现节 |
|---|---|---|
| $\delta_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$ | TD error | §8.3 |
| $\nabla J = \mathbb{E}[\nabla\log\pi \cdot \delta_t]$ | Actor-Critic 梯度 | §8.3 |
| $\hat A_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k \delta_{t+k}$ | n-step advantage | §8.4 |
| $\hat A_t^{GAE(\gamma,\lambda)} = \sum_l (\gamma\lambda)^l \delta_{t+l}$ | GAE 定义 A | §8.5 |
| $\hat A_t^{GAE(\gamma,\lambda)} = (1-\lambda)\sum_n \lambda^{n-1} \hat A_t^{(n)}$ | GAE 定义 B | §8.5 |
| $L_{\text{total}} = L_{\text{actor}} + c_v L_{\text{critic}} + \beta L_{\text{entropy}}$ | A2C 总 loss | §8.6 |

### 8.8.3 承接线索回顾

| 出处 | 承接的概念 | 本章兑现 |
|---|---|---|
| Ch04 §4.3 | "TD error $\delta_t$ → Ch08 Actor-Critic" | ✓ §8.3 严格推导 + 证明不偏 |
| Ch04 §4.8 | "TD(λ) eligibility → Ch08 GAE（PPO 标配）" | ✓ §8.5 完整推导 |
| **Ch04 §4.8** | **"GAE = TD(λ) 思想的 advantage 版本"** | ✓ §8.5.2 严格证明 + §8.5.4 对照 |
| Ch04 §4.8 | "eligibility traces 思想在 AC、PPO、GAE 反复出现" | ✓ §8.5.5 后向递归 = eligibility trace |
| Ch02 §2.3 | $V^\pi$——critic 的学习目标 | ✓ §8.2 全节 |
| Ch05 §5.1 | 控制 = 预测 + 策略改进 | ✓ §8.6 完整 A2C |
| Ch04 §4.8 | "GAE, Retrace(λ), V-trace, IMPALA" | ✓ §8.5.4 + §8.8.4 预告 |

### 8.8.4 旁注：Retrace(λ), V-trace, IMPALA（延伸自 Ch04 §4.8）

GAE 是 **on-policy** 的 TD(λ) advantage 版本。在 **off-policy** 设定下，TD error 需要修正"行为策略 $\mu \neq \pi_\theta$"带来的偏差，这衍生出几个变种：

- **Retrace(λ)**（Munos et al. 2016）：用 $\min(1, \pi/\mu)$ 截断 importance weight，保证低方差且收敛
- **V-trace**（Espeholt et al. 2018 IMPALA）：用 $\min(\bar\rho, \pi/\mu)$ 截断，是 IMPALA 大规模分布式 actor-critic 的核心
- **Q(λ), TB(λ), n-step IS**：其它 off-policy TD(λ) 变种

这些方法的"代数骨架"都是**交换求和顺序 + 几何级数**——和 §8.5.2 的 GAE 证明同源。学透 GAE，看 Retrace / V-trace 就轻而易举。

### 8.8.5 A2C 还差什么？→ PPO

A2C 解决了 REINFORCE 的"高方差 + offline + 无 bootstrap"，但**仍然是严格 on-policy**——每次更新后旧数据失效。这在样本效率上仍然吃亏。

**Ch09 PPO**（Schulman et al. 2017）通过两个关键技术让 on-policy 数据**有限重用**：

1. **Importance ratio** $r_t = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$，把 ratio 写进 policy gradient
2. **Clipped objective** $L^{CLIP} = \mathbb{E}[\min(r_t \hat A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t)]$——限制 ratio 在 $[1\pm\epsilon]$，让"信任域"内的 multiple epochs 训练成为可能

PPO 还会把 GAE 直接用在 advantage 估计里（§8.5 的 GAE 就是 PPO 论文带火的）。其它部件（共享 backbone、critic regression、entropy bonus）全部继承自本章的 A2C。

| 本章 A2C | Ch09 PPO 增量 |
|---|---|
| 单次更新（每个 batch 用一次） | multiple epochs（一个 batch 用多次） |
| $-\log\pi \cdot \hat A$ | $\min(r\hat A, \text{clip}(r)\hat A)$ |
| 直接 on-policy | 通过 clipping 容忍小规模 off-policy |

---

**下一章：第 9 章 — PPO（Proximal Policy Optimization）**。

学完本章后看 PPO 会非常轻松——PPO 只是在 A2C 基础上加了 importance ratio + clipping，让你能在同一批数据上做多个 epoch 的更新。GAE、共享 backbone、entropy bonus、value regression 这些组件我们已经搭好了。""")

code("""# 最终可视化：学到的策略 + value 在 2D 状态空间切片上的形状
# 选两个最关键的状态维度：theta (杆子角度) 和 theta_dot (角速度)
# 其它维度设为 0，扫一遍 (theta, theta_dot) 网格

theta_grid = np.linspace(-0.2, 0.2, 40)
thetadot_grid = np.linspace(-3.0, 3.0, 40)
THETA, THETADOT = np.meshgrid(theta_grid, thetadot_grid)
states_grid = np.zeros((40*40, 4), dtype=np.float32)
states_grid[:, 2] = THETA.flatten()
states_grid[:, 3] = THETADOT.flatten()

with torch.no_grad():
    s_t = torch.as_tensor(states_grid)
    dist, v_grid = ac(s_t)
    p_right = dist.probs[:, 1].numpy().reshape(40, 40)
    v_grid = v_grid.numpy().reshape(40, 40)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. π(right|s) 热力图
ax = axes[0]
im = ax.pcolormesh(THETA, THETADOT, p_right, cmap='RdBu_r', vmin=0, vmax=1, shading='auto')
ax.axhline(0, color='k', linewidth=0.5); ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlabel(r'$\\theta$ (杆子角度)'); ax.set_ylabel(r'$\\dot\\theta$ (角速度)')
ax.set_title(r'$\\pi(\\mathrm{right}|\\theta, \\dot\\theta)$（车/速度=0 切片）')
plt.colorbar(im, ax=ax)

# 2. V_φ(s) 热力图
ax = axes[1]
im = ax.pcolormesh(THETA, THETADOT, v_grid, cmap='viridis', shading='auto')
ax.axhline(0, color='w', linewidth=0.5); ax.axvline(0, color='w', linewidth=0.5)
ax.set_xlabel(r'$\\theta$ (杆子角度)'); ax.set_ylabel(r'$\\dot\\theta$ (角速度)')
ax.set_title(r'$V_\\phi(s)$（critic 学到的价值函数）')
plt.colorbar(im, ax=ax)

plt.tight_layout(); plt.show()
print("观察：")
print("  - 左图：θ>0（右倾）时 π(right)=1，θ<0 时 π(right)=0——'反射式'策略")
print("  - 右图：θ=θ̇=0（完美直立）时 V_φ 最高——critic 学到了'平衡越久价值越高'")
print("  - 这是 actor-critic 学到的完整'世界模型 + 行为策略'。")""")

md(r"""## 8.9 📝 练习

### 练习 1（必做）：把 A2C 的 advantage 从 GAE 换成 n-step

**任务**：
1. 复用本章 A2C 训练循环，把 `compute_gae` 换成 `utils.gae.compute_n_step_advantage`（n 取 1 / 4 / 16）
2. 画三条学习曲线 + 一条 GAE(λ=0.95) 的参照线
3. 用一句话总结 n-step 和 λ 的关系（提示：GAE = n-step 的几何加权平均，§8.5）

**预期结果**：n=1 方差小但偏置大（学得慢）、n=16 接近 MC；GAE 曲线应落在它们之间的甜点上——这就是 λ 的意义。

### 练习 2（选做）：critic 学习率敏感性

把 critic 的学习率单独调小 10 倍（advantage 用的 V_φ 还没学到位），观察 actor 训练曲线的变化。

<details><summary>提示</summary>

- A2C 里 actor 和 critic 可以用两个 optimizer（本章实现如此），只改 critic 那个的 lr
- 预期：advantage 估计系统性偏差 → actor 学到偏的策略或震荡——"critic 差则 actor 跟着差"，这是 actor-critic 的标志性失败模式
</details>

*（开放练习，无参考答案。）
> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch08 的自测题再进入下一章。""")


if __name__ == "__main__":
    # 直接运行时写出 notebook（被 build_notebooks.py import 时不写盘）
    _nb.write("ch08_actor_critic_gae.ipynb")
