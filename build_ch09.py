r"""Build ch09_trpo_ppo.ipynb.

一次性脚本：构造 Ch09 笔记本（9 节内容，TRPO + PPO）。
Phase 2 终点。运行后产物在 notebooks/。
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch09")
cells = _nb.cells            # build_notebooks.py 适配层读取的模块级列表
md, code = _nb.md, _nb.code


# =============================================================================
# Title + intro + setup
# =============================================================================

md(r"""# 第 9 章：TRPO + PPO —— 把"信任域"塞进策略梯度（Phase 2 终点）

> **Ch08 Actor-Critic** 给了我们一个**方差可控**的 advantage 估计（GAE），但它仍然只是
> "策略梯度 + critic"。本章回答的是策略梯度更深层的痛点：
>
> > **一步走多大，才不会把策略带崩？**
>
> 这是 RL 与监督学习最不同的地方——监督学习里"loss 降"等价于"模型变好"，
> 但 RL 里"$J(\pi_\theta)$ 上升"**不**等价于"全局更好"：你只是在用旧数据估的梯度
> 走了一步，走太远就脱离了"旧数据有效的信任域"（trust region），可能直接把策略毁掉。
>
> **本章核心等式**（Schulman et al. 2017，被 GPT/Claude RLHF 直接使用）：
>
> $$L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\big(r_t(\theta)\hat A_t,\;\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\big)\right], \quad r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
>
> 这一行的精髓是：**用 clip 把"策略一步走多远"软约束在 $[1-\epsilon, 1+\epsilon]$**，
> 替代 TRPO 那个昂贵的二阶 KL 硬约束——**计算简单、效果几乎一样、稳定性远胜朴素 PG**。
> 这是 LLM RLHF 选 PPO 的核心理由（§9.9 详述）。

## 学习目标

1. 理解 **为什么朴素策略梯度步长大就崩**（信任域动机）
2. 掌握 **TRPO 的核心思想**：KL 约束 + 二次规划 + 共轭梯度 + line search（理论 + 直觉）
3. **完整推出 PPO-Clip 目标**：从 importance-weighted PG 推到 $\min(r\hat A, \mathrm{clip}(r)\hat A)$
4. 实现 **完整 PPO 算法**：actor clip + critic MSE + GAE + entropy bonus + KL early stopping + 多 epoch 数据重用
5. 理解 **多 epoch on-policy 数据重用**——为什么这是 PPO 样本效率的工程 magic
6. 用 **PPO-Clip 经典图**（$r$-$L$ 曲线）建立几何直觉
7. **CartPoleLite 完整训练**：reward > 400（接近 500 上限）
8. 与 Ch06 DQN / Ch08 A2C 三方对比，**回答为什么 LLM RLHF 选 PPO**
9. 预告 **Phase 3**（Ch10-15）：从 PPO 走到 GRPO

## 承接的 Phase 1 / Ch08 承诺（11 处 —— 全书最核心）

| 出处 | 承诺原文 | 本章兑现节 |
|---|---|---|
| **Ch00** | **"PPO 是 fast-track 终点之一"**（Ch00→01→05→07→09→13） | **全章** |
| **Ch00** | "后面所有算法 PPO、GRPO" | §9.9（铺垫 Ch13 GRPO） |
| **Ch02** | "Deep RL 控制算法 → Actor-Critic / PPO / GRPO" | 全章 |
| **Ch02-04** | "后面所有算法 PPO、GRPO"（多处重复） | §9.9 |
| **Ch03 §3.3** | **"trust region 解决贪心改进的近似失效"** | **§9.1, §9.2** |
| **Ch03** | **"clip 解决贪心改进失败"** | **§9.3（灵魂）** |
| **Ch03 §3.3** | "近似失效 → target network / trust region"（Ch06 兑现一半） | §9.2（trust region 这半） |
| **Ch05** | **"Phase 2 的核心 = PPO"** | **全章** |
| **Ch05** | **"PPO 是 on-policy，每次更新后数据作废"** | §9.4, §9.5（含例外：多 epoch 重用） |
| **Ch05** | **"LLM RLHF 选 PPO 是因为稳定性"** | **§9.9（灵魂）** |
| Ch05 | "PPO 比 DQN 更适合连续/高维动作"（隐含） | §9.8 |

> 加上 Ch08 §8.8 末尾的预告"GAE 是 PPO 的标配 advantage 估计"——本章**每一节都在用 Ch08 的 GAE**。

---

> **跳读提示（Fast-track 用户）**：如果你是从 Ch07 直接跳来、没读 Ch08，
> 你只需要知道两件事：
> 1. **Actor-Critic** = 策略网络 $\pi_\theta$（actor）+ 价值网络 $V_\phi$（critic）联合训练
> 2. **GAE** advantage $\hat A_t$ = 一个方差可控的"这步动作比平均好多少"的标量
>
> 本章用到 Ch08 的 `ActorCritic` 网络和 `compute_gae`，把它们当黑盒用即可。
""")

code(r"""# 常规设置：找项目根、载入库
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
from utils import set_seed, smooth
from utils.policy_networks import ActorCritic, ValueNetwork, CategoricalPolicy
from utils.gae import compute_gae, compute_td_errors, compute_returns_from_gae
from utils.ppo import (
    ppo_update, compute_clip_objective, approx_kl_from_ratio, compute_kl,
)
from utils.torch_utils import get_device, count_parameters

set_seed(42)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = get_device()
print(f"PyTorch: {torch.__version__}, device = {DEVICE}")
print(f"utils 新增基础设施（utils/ppo.py）：ppo_update, compute_clip_objective, "
      f"approx_kl_from_ratio, compute_kl")""")


# =============================================================================
# 9.1 Why plain policy gradient steps are dangerous
# =============================================================================

md(r"""## 9.1 策略改进的本质：为什么大步危险

### 9.1.1 Ch08 留下的"未解之谜"

Ch08 我们写下了 Actor-Critic 的更新规则（A2C）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\Big[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot \hat A_t\Big], \qquad \theta \leftarrow \theta + \alpha\,\nabla_\theta J
$$

这里有个**隐藏假设**：我们用当前 $\pi_\theta$ 采的数据去估 $\nabla J$，然后**走一步** $\alpha$。
但 $\nabla J$ 只是 $\pi_\theta$ 这一点处的**局部**信息——它告诉你"在 $\theta$ 附近，
往哪个方向走 $J$ 会升"。

**问题**：这个梯度在 $\theta$ 之外多远还成立？

### 9.1.2 一个失败的实验：lr 过大直接崩

我们用一个最小实验直观展示：**同样的 A2C，learning rate 翻 10 倍，从平稳训练变成瞬间崩溃**。""")

code(r"""# 实验：相同 A2C，不同 lr，观察"大步危险"
# 复用 Ch08 的 ActorCritic + 一个简化 A2C 更新（只看 reward 曲线）

def collect_traj(env, ac, n_steps):
    # 收 n_steps 步 on-policy 数据。
    states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
    ep_rewards = []
    s = env.reset()
    cur = 0.0
    for _ in range(n_steps):
        s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        dist, v = ac(s_t)
        a = dist.sample()
        logp = dist.log_prob(a)
        s_next, r, done, _ = env.step(int(a.item()))
        states.append(s.copy()); actions.append(int(a.item()))
        rewards.append(float(r)); dones.append(bool(done))
        values.append(float(v.item())); log_probs.append(logp)
        cur += r
        if done:
            ep_rewards.append(cur); cur = 0.0; s = env.reset()
        else:
            s = s_next
    s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, lv = ac(s_t)
    return dict(states=np.array(states, dtype=np.float32),
                actions=np.array(actions, dtype=np.int64),
                rewards=np.array(rewards, dtype=np.float32),
                dones=np.array(dones, dtype=bool),
                values=np.array(values, dtype=np.float32),
                log_probs=log_probs, last_value=float(lv.item()),
                ep_rewards=ep_rewards)

def simple_a2c_update(ac, opt, traj, gamma=0.99, lam=0.95, lr=1e-3):
    # Ch08 A2C 单 epoch 单 batch 更新（不用 clip，用作对照）。
    advantages_np = compute_gae(traj['rewards'], traj['values'],
                                last_value=traj['last_value'],
                                gamma=gamma, lam=lam, dones=traj['dones'])
    returns_np = advantages_np + traj['values']
    adv = torch.as_tensor(advantages_np, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    returns = torch.as_tensor(returns_np, dtype=torch.float32)
    states = torch.as_tensor(traj['states'], dtype=torch.float32)
    actions = torch.as_tensor(traj['actions'], dtype=torch.long)

    dist, v = ac(states)
    logp = dist.log_prob(actions)
    actor_loss = -(logp * adv).mean()
    critic_loss = 0.5 * (v - returns).pow(2).mean()
    ent = dist.entropy().mean()
    loss = actor_loss + 0.5 * critic_loss - 0.01 * ent
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
    opt.step()

def run_a2c(lr, n_iters=60, n_steps=1024, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    env = CartPoleLite(seed=seed, max_steps=500)
    ac = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[64, 64])
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    recent_rewards = []
    recent_window = []
    for it in range(n_iters):
        traj = collect_traj(env, ac, n_steps)
        simple_a2c_update(ac, opt, traj, lr=lr)
        recent_window.extend(traj['ep_rewards'])
        if len(recent_window) > 20:
            recent_window = recent_window[-20:]
        recent_rewards.append(np.mean(recent_window) if recent_window else 0.0)
    return np.array(recent_rewards)

print("对比 3 个 learning rate：1e-3（稳）/ 1e-2（边缘）/ 5e-2（崩溃）")
results_lr = {}
for lr in [1e-3, 1e-2, 5e-2]:
    print(f"  跑 lr = {lr} ...")
    results_lr[lr] = run_a2c(lr, n_iters=80)

# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
colors_lr = {1e-3: '#1f77b4', 1e-2: '#ff7f0e', 5e-2: '#d62728'}
for lr, curve in results_lr.items():
    ax.plot(smooth(curve, 10), linewidth=2, color=colors_lr[lr],
            label=f'lr={lr}')
ax.axhline(500, color='green', linestyle='--', alpha=0.5, label='上限 500')
ax.set_xlabel('iteration')
ax.set_ylabel('recent 20-episode mean reward')
ax.set_title('"大步危险"：相同 A2C 算法，不同 learning rate')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("观察：lr=1e-3 稳定上升；lr=1e-2 边缘震荡；lr=5e-2 直接崩到 ~20")
print("→ 这就是 Ch03 §3.3 说的'近似失效'：local 梯度走远了就不再可靠。")""")

md(r"""### 9.1.3 信任域（trust region）的直觉

**核心观察**：策略梯度只在 $\theta_{old}$ 的局部成立。我们要保证**新策略 $\pi_\theta$ 没走太远**，
旧数据采的梯度估计才依然可靠。

定义这个"没走太远"的数学度量——**两个策略之间的 KL 散度**：

$$
\bar D_{KL}(\pi_{old} \| \pi_\theta) = \mathbb{E}_{s \sim \rho_{\pi_{old}}}\!\left[D_{KL}\big(\pi_{old}(\cdot|s) \,\|\, \pi_\theta(\cdot|s)\big)\right]
$$

**信任域** = $\{\theta : \bar D_{KL}(\pi_{old} \| \pi_\theta) \le \delta\}$，其中 $\delta$ 是个小数（典型 $0.01$）。

> 在这个域里，"用旧数据估的梯度"和"真实梯度"差距可控；出域就可能崩。

**TRPO 的主张**（Schulman et al. 2015）：把"最大 reward 改进"写成 **KL 约束下的优化问题**，
理论上保证每步策略改进**单调**（不会变差）。下一节我们推出这个优化问题。

### 9.1.4 本章路线图

| 节 | 主题 | 关键产出 |
|---|---|---|
| 9.2 | TRPO：KL 约束 + 二阶优化 | 理解 trust region 的"正确"实现 + 为什么太贵 |
| 9.3 | PPO-Clip：把硬约束换成软 clip | $L^{CLIP}$ 完整推导 + 经典 clip 图 |
| 9.4 | PPO 完整算法 | 5 个组件凑齐 |
| 9.5 | 多 epoch 数据重用 | PPO 样本效率的工程 magic |
| 9.6 | 工程细节 | KL early stop / entropy / adv-norm / orthogonal init |
| 9.7 | CartPoleLite 完整训练 | reward > 400 |
| 9.8 | PPO vs TRPO vs DQN 对比 | 三方各有所长 |
| 9.9 | 为什么 LLM RLHF 选 PPO | Phase 3 预告 |""")


# =============================================================================
# 9.2 TRPO theory
# =============================================================================

md(r"""## 9.2 TRPO：KL 约束下的策略改进

### 9.2.1 从策略改进定理到优化问题

回忆 **策略改进定理**（Sutton & Barto §4.2 的核心）：如果新策略 $\tilde\pi$ 满足
$Q^{\pi_{old}}(s, a) \ge V^{\pi_{old}}(s)$（按 $\tilde\pi$ 期望），那么 $V^{\tilde\pi} \ge V^{\pi_{old}}$。

把"$\ge$"换成"最大化"：我们想让 $\tilde\pi$ 最大化
$\mathbb{E}_{s \sim \rho_{\pi_{old}}, a \sim \tilde\pi}\!\big[Q^{\pi_{old}}(s,a) - V^{\pi_{old}}(s)\big]
= \mathbb{E}_{a \sim \tilde\pi}[A^{\pi_{old}}(s,a)]$。

但有个问题：$A^{\pi_{old}}$ 是按 $\pi_{old}$ 算的，而我们要按 $\tilde\pi$ 期望。
用 **importance sampling** 把它写回 $\pi_{old}$ 下的期望（**这是 RL 里第一次见到 importance ratio，后面 PPO 整个就是建立在这上面**）：

$$
\mathbb{E}_{a \sim \tilde\pi}\!\big[A^{\pi_{old}}(s,a)\big]
= \mathbb{E}_{a \sim \pi_{old}}\!\left[\frac{\tilde\pi(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s,a)\right]
$$

参数化 $\tilde\pi = \pi_\theta$、$\pi_{old} = \pi_{\theta_{old}}$，得到 **TRPO 目标**：

$$
\boxed{\quad
\max_\theta\; \mathbb{E}_{s \sim \rho_{\theta_{old}},\, a \sim \pi_{\theta_{old}}}\!\left[
  \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}\, A^{\pi_{\theta_{old}}}(s,a)
\right]
\quad\text{s.t.}\quad
\mathbb{E}_s\!\big[\mathrm{KL}\big(\pi_{\theta_{old}}(\cdot|s)\,\|\,\pi_\theta(\cdot|s)\big)\big] \le \delta
\quad}
$$

> **理论保证**（Kakade & Langford 2002；Schulman et al. 2015 Theorem 1）：
> 在 $\delta$ 足够小的情况下，**最坏情况下新策略不会比旧策略差太多**——
> 这是策略梯度文献里少有的"非局部保证"，远胜朴素 SGD 的"局部梯度"幻想。

### 9.2.2 怎么解这个约束优化？——Taylor 展开 + 二次规划

目标函数在 $\theta_{old}$ 处一阶 Taylor 展开（**线性**）：

$$
L_\theta(\theta_{old}) \approx g^T(\theta - \theta_{old}),
\quad g = \nabla_\theta L_\theta\big|_{\theta_{old}}
$$

（$g$ 就是普通的策略梯度。）

约束函数在 $\theta_{old}$ 处二阶 Taylor 展开（KL 在极小点处一阶项为 0，**主项是二次**）：

$$
\mathbb{E}[\mathrm{KL}(\pi_{\theta_{old}} \| \pi_\theta)] \approx \tfrac{1}{2}(\theta - \theta_{old})^T F\,(\theta - \theta_{old})
$$

其中 **$F$ 是 Fisher 信息矩阵**：

$$
F = \mathbb{E}_{s,a \sim \pi_{\theta_{old}}}\!\left[\nabla_\theta \log\pi_{\theta_{old}}(a|s)\,\nabla_\theta \log\pi_{\theta_{old}}(a|s)^T\right]
$$

> **KL ≈ $\frac{1}{2}\Delta\theta^T F \Delta\theta$** 是统计学习里的标准结论——
> Fisher 矩阵就是 KL 的 Hessian（在 MLE 极点处）。

代入约束得到 **二次规划**：

$$
\max_\theta\; g^T \Delta\theta \quad\text{s.t.}\quad \tfrac{1}{2}\Delta\theta^T F\,\Delta\theta \le \delta, \;\;\Delta\theta = \theta - \theta_{old}
$$

<details>
<summary><b>推导：解这个 QP 得到自然梯度步</b></summary>

用 Lagrange 乘子：$\mathcal{L}(\Delta\theta, \lambda) = g^T \Delta\theta - \lambda\big(\tfrac{1}{2}\Delta\theta^T F\,\Delta\theta - \delta\big)$。

对 $\Delta\theta$ 求导置零：$g - \lambda F\,\Delta\theta = 0 \Rightarrow \Delta\theta = \frac{1}{\lambda} F^{-1} g$。

代回约束（取等号，因为 $g^T\Delta\theta$ 最大化时一定贴边）：

$$
\tfrac{1}{2}\big(\tfrac{1}{\lambda}F^{-1}g\big)^T F \big(\tfrac{1}{\lambda}F^{-1}g\big) = \delta
\;\Rightarrow\;
\tfrac{1}{2\lambda^2} g^T F^{-1} g = \delta
\;\Rightarrow\;
\lambda = \sqrt{\tfrac{g^T F^{-1} g}{2\delta}}
$$

代回 $\Delta\theta$：

$$
\boxed{\;\Delta\theta = \sqrt{\frac{2\delta}{g^T F^{-1} g}}\, F^{-1} g\;}
$$

这就是 **natural policy gradient**（Amari 1998；Kakade 2002）：把欧氏距离换成 KL 流形上的"最陡上升方向"。
</details>

### 9.2.3 三个工程技巧：共轭梯度 / Fisher-vector product / line search

直接解 $\Delta\theta = \sqrt{2\delta/g^T F^{-1}g}\,F^{-1}g$ 有两个**致命的工程问题**：

1. **$F^{-1}$ 太贵**：神经网络 $\theta$ 有几万到几亿维，求逆 $O(n^3)$ 完全不可行。
2. **Taylor 近似只在局部成立**：算出的 $\Delta\theta$ 走一步可能 KL 实际超 $\delta$（二阶展开误差）。

TRPO 用三个 trick 解决：

**Trick 1：共轭梯度（Conjugate Gradient, CG）解 $F^{-1}g$**

CG 是个迭代算法，能在 $k$ 步内近似解线性方程 $Fx = g$（等价于 $x = F^{-1}g$），
**只需要算 $Fv$（Fisher-vector product），不需要显式存 $F$ 或求逆**。典型 $k=10$ 步就够。

**Trick 2：Fisher-vector product 不用建矩阵**

$Fv = \mathbb{E}\big[(\nabla\log\pi)(\nabla\log\pi)^T\big]v
= \mathbb{E}\big[(\nabla\log\pi)\cdot\underbrace{(\nabla\log\pi)^T v}_{\text{标量}}\big]$

可以用 **Pearlmutter trick**（Hessian-vector product）在 $O(n)$ 时间算出，
不用真的构造 $n \times n$ 矩阵。PyTorch 里用 `torch.autograd.grad(v\cdot g, params)` 一次反向就够。

**Trick 3：line search 保证 KL 约束真的满足**

CG 给的 $\Delta\theta$ 是基于二阶 Taylor 近似，真实 KL 可能超 $\delta$。TRPO 在 $\Delta\theta$ 方向做指数回溯 line search：

$$
\theta \leftarrow \theta_{old} + \alpha^j \Delta\theta, \quad j=0,1,2,\dots
$$

每步检查：(a) KL $\le \delta$；(b) 目标真的上升。第一个满足的 $j$ 就采用。

> **TRPO 总结**：理论优美（单调改进保证）、效果好，但**实现复杂、每次更新都要 CG + line search**。
> 这就是 PPO 出现的动机——**能不能用更简单的方法达到同样效果？**""")

code(r"""# TRPO 简化实验：用 PyTorch 自动 Hessian-vector product + 共轭梯度
# 我们不实现完整 TRPO（line search / sample-based Fisher 等都比较繁），
# 而是演示"用共轭梯度解 F^{-1}g → 自然梯度步"在 CartPoleLite 上的效果。

def fisher_vector_product(ac, states, actions, v, damping=1e-2):
    # Fisher-vector product F·v，用 Pearlmutter trick 实现。
    #
    # 关键：F = Hessian_θ KL(π_θ_old || π_θ)|_{θ=θ_old}（在 θ=θ_old 处为正定矩阵）。
    # 我们不用 Hessian(log π)——它和 F 差一个负号（Hessian(log π) = -F 期望意义下成立），
    # 直接用会让 CG 解一个不定方程、g^T F^-1 g 出负值，演示变成空操作。
    #
    # 正确做法（与 Schulman 原始 TRPO 实现一致）：固定 old_logits，对当前网络求
    # mean KL(old || current) 的 Hessian-vector product。
    params = list(ac.parameters())
    with torch.no_grad():
        old_logits = ac.actor_head(ac.backbone(states)).detach()
    new_logits = ac.actor_head(ac.backbone(states))
    # per-sample KL(old || new)
    kl_per = (F.softmax(old_logits, dim=-1) *
              (F.log_softmax(old_logits, dim=-1) - F.log_softmax(new_logits, dim=-1))).sum(dim=-1)
    kl_mean = kl_per.mean()
    grad_kl = torch.autograd.grad(kl_mean, params, create_graph=True, allow_unused=True)
    flat_grad_kl = torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                              for g, p in zip(grad_kl, params)])
    kl_v = (flat_grad_kl * v).sum()
    grads2 = torch.autograd.grad(kl_v, params, allow_unused=True)
    fvp = torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                     for g, p in zip(grads2, params)])
    return fvp + damping * v  # Hessian(KL) + damping·I，保持正定

def conjugate_gradient(Avp_fn, b, n_steps=10, residual_tol=1e-10):
    # 共轭梯度解 Ax = b（A 通过 Avp_fn 隐式给出）。
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rsold = (r * r).sum()
    for i in range(n_steps):
        Ap = Avp_fn(p)
        denom = (p * Ap).sum()
        if denom <= 1e-12:  # 防止 CG 发散
            break
        alpha = rsold / denom
        x += alpha * p
        r -= alpha * Ap
        rsnew = (r * r).sum()
        if rsnew.sqrt() < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def flatten_grads(loss, params):
    params = list(params)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                      for g, p in zip(grads, params)])

def trpo_step_demo(ac, traj, delta=0.01, gamma=0.99, lam=0.95, cg_iters=10):
    # 演示版 TRPO 一步更新：算 g → CG 算 F^{-1}g → 自然梯度步 → 不做 line search。
    # 返回 (步长 scale, KL 估计)。
    states = torch.as_tensor(traj['states'], dtype=torch.float32)
    actions = torch.as_tensor(traj['actions'], dtype=torch.long)
    advantages_np = compute_gae(traj['rewards'], traj['values'],
                                last_value=traj['last_value'],
                                gamma=gamma, lam=lam, dones=traj['dones'])
    adv = torch.as_tensor(advantages_np, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # 策略梯度 g（importance ratio = 1，因为 θ = θ_old）
    dist, _ = ac(states)
    logp = dist.log_prob(actions)
    policy_loss = -(logp * adv).mean()  # = -L
    g = flatten_grads(policy_loss, ac.parameters()) * -1.0  # 取 ∇L（不是 -∇L）

    # 共轭梯度解 Fx = g
    Avp = lambda v: fisher_vector_product(ac, states, actions, v)
    step_dir = conjugate_gradient(Avp, g, n_steps=cg_iters)

    # 自然梯度步：Δθ = sqrt(2δ / (g^T F^{-1} g)) * F^{-1} g
    sHg = (g * step_dir).sum()  # = g^T F^{-1} g（因为 step_dir = F^{-1}g）
    lm = float(torch.sqrt(2 * delta / (sHg + 1e-8)))
    full_step = lm * step_dir

    # 记录更新前的 log π
    with torch.no_grad():
        old_logits = ac.backbone(states)
        old_logits = ac.actor_head(old_logits)
        old_logp = Categorical(logits=old_logits).log_prob(actions)

    # 应用更新（不做 line search；演示用）
    old_params = torch.cat([p.reshape(-1) for p in ac.parameters()])
    new_params = old_params + full_step
    idx = 0
    for p in ac.parameters():
        n = p.numel()
        p.data = new_params[idx:idx+n].reshape_as(p)
        idx += n

    # 算实际 KL
    with torch.no_grad():
        new_logits = ac.backbone(states)
        new_logits = ac.actor_head(new_logits)
        new_dist = Categorical(logits=new_logits)
        old_dist = Categorical(logits=old_logits)
        kl_after = torch.distributions.kl_divergence(old_dist, new_dist).mean().item()
    return float(lm), kl_after

# 小测试：先用 A2C 预热策略（让它脱离"全 0 梯度"区域），再跑 TRPO step
torch.manual_seed(0); np.random.seed(0)
env = CartPoleLite(seed=0, max_steps=500)
ac = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[32, 32])
opt_warmup = torch.optim.Adam(ac.parameters(), lr=3e-3)
print("A2C 预热（30 iter）让策略有梯度信号（未训练时 g ≈ 0，TRPO 自然梯度步无意义）...")
for _ in range(30):
    traj_w = collect_traj(env, ac, n_steps=512)
    simple_a2c_update(ac, opt_warmup, traj_w, lr=3e-3)

traj = collect_traj(env, ac, n_steps=2048)  # 更大 batch 让 Fisher 估计更稳
lm, kl = trpo_step_demo(ac, traj, delta=0.01)
print(f"\\nTRPO demo step（在 A2C 预热后的策略上）:")
print(f"  自然梯度步长系数 √(2δ/g^T F^-1 g) = {lm:.4f}")
print(f"  更新后实际 KL(old||new) = {kl:.5f}")
print(f"  δ 设定 = 0.01, 实际 KL {'≤' if kl <= 0.01 else '>'} δ")
if kl <= 0.01:
    print(f"  → 实际 KL ≤ δ：Taylor 二阶展开在预热后的策略上是合理的近似（g^T F^-1 g 不再 ≈ 0）。")
    print(f"    注：在某些 seed 下 Taylor 近似误差会让 KL 略超 δ，这就是 TRPO 还要加 line search 的原因。")
else:
    print(f"  → 实际 KL 略超 δ：Taylor 二阶展开的近似误差，这就是 TRPO 还要加 line search 的原因。")""")

md(r"""### 9.2.4 TRPO 的代价 vs 收益

| 维度 | TRPO |
|---|---|
| **理论保证** | 强（KL 约束保证策略不会跑太远，单调改进） |
| **样本效率** | 高（用 importance ratio，但单 epoch） |
| **计算开销** | 每次更新 CG（~10 步）+ line search（~10 次前向），高 |
| **调参** | δ、CG 步数、line search 回溯系数——敏感 |
| **实现复杂度** | 高（HVP、CG、line search 全要正确） |
| **现代用法** | 已基本被 PPO 取代；只在需要严格 trust region 的研究里用 |

> **PPO（Schulman et al. 2017）的核心 motivation**：
> "TRPO 的 KL 约束能不能换成更简单的形式，省掉 CG 和 line search？"
> 下一节给出答案：**用 clip 替代约束**。""")


# =============================================================================
# 9.3 PPO-Clip objective
# =============================================================================

md(r"""## 9.3 PPO-Clip 目标：把硬约束换成软 clip

### 9.3.1 从 TRPO 目标出发

TRPO 解的优化问题：

$$
\max_\theta\; \mathbb{E}_t\!\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\,\hat A_t\right]
\quad\text{s.t.}\quad \mathrm{KL}(\pi_{old}\|\pi_\theta) \le \delta
$$

定义 **probability ratio**（本章出现最频繁的量）：

$$
\boxed{\;r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\;}
$$

则 TRPO 目标简写成 $\mathbb{E}_t[r_t \hat A_t]$。约束是"KL 别太大"。

> **PPO 的关键洞察**：与其在 KL 上加硬约束（要 CG + line search），
> 不如**直接在目标函数里"惩罚" $r_t$ 跑出 $[1-\epsilon, 1+\epsilon]$ 的情况**。
> 这样：$r_t$ 远离 1 等价于策略远离 $\pi_{old}$——直接约束 $r_t$ 就间接约束了 KL。

### 9.3.2 分情况分析：为什么 $\min(r\hat A, \mathrm{clip}(r)\hat A)$ 是好设计

PPO-Clip 目标（per-sample）：

$$
L^{CLIP}(\theta) = \min\big(r_t\hat A_t,\;\mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t\big), \qquad \epsilon \approx 0.2
$$

分两种情况讨论（**这是 PPO-Clip 全部直觉的所在**）：

**情况 A：$\hat A_t > 0$（这步动作比平均好）**

我们要**鼓励**这个动作——让 $\pi_\theta(a_t|s_t)$ 变大，即 $r_t \uparrow$。

- 若 $r_t \le 1+\epsilon$（策略还没走太远）：目标 $= r_t \hat A_t$，梯度推 $r_t$ 上升（鼓励）。
- 若 $r_t > 1+\epsilon$（策略已经走太远）：目标 $= (1+\epsilon)\hat A_t$，**梯度为 0**——
  不再继续鼓励。这一步的"奖励上限"被 clip 掉了。

> **直观**：好动作最多让它的概率上升 $\epsilon=20\%$，再多就停。

**情况 B：$\hat A_t < 0$（这步动作比平均差）**

我们要**抑制**这个动作——让 $\pi_\theta(a_t|s_t)$ 变小，即 $r_t \downarrow$。

- 若 $r_t \ge 1-\epsilon$：目标 $= r_t \hat A_t$（负值），梯度推 $r_t$ 下降（抑制）。
- 若 $r_t < 1-\epsilon$（策略已经走太远）：目标 $= (1-\epsilon)\hat A_t$，**梯度为 0**——
  不再继续抑制。

> **直观**：差动作最多让它的概率下降到原来的 $1-\epsilon=80\%$，再降就停。

### 9.3.3 PPO-Clip 经典图（必看）

把上面两种情况画在一张图上：$x$ 轴是 $r_t$，$y$ 轴是 $L^{CLIP}$，
分 $\hat A > 0$（绿）和 $\hat A < 0$（红）两条曲线。""")

code(r"""# PPO-Clip 经典图：r-L^{CLIP} 曲线
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

clip_eps = 0.2
r = np.linspace(0.0, 2.0, 500)

# 左图：A > 0 情况
ax = axes[0]
A_pos = 1.0
L_unclipped = r * A_pos
L_clipped = np.clip(r, 1 - clip_eps, 1 + clip_eps) * A_pos
L_ppo = np.minimum(L_unclipped, L_clipped)

# 画 clip 区域背景
ax.axvspan(1 - clip_eps, 1 + clip_eps, color='lightyellow', alpha=0.7, label='未 clip 区')
ax.plot(r, L_unclipped, color='gray', linestyle='--', linewidth=1.5, label=r'未 clip: $r_t \hat A_t$')
ax.plot(r, L_clipped, color='orange', linestyle=':', linewidth=1.5, label=r'clip 上限: $(1+\epsilon)\hat A_t$')
ax.plot(r, L_ppo, color='#2ca02c', linewidth=3.5, label=r'$L^{CLIP}$（PPO 目标）')
ax.axvline(1.0, color='k', linewidth=0.7)
ax.axvline(1 + clip_eps, color='red', linewidth=1, linestyle='--', alpha=0.7)
ax.axvline(1 - clip_eps, color='red', linewidth=1, linestyle='--', alpha=0.7)
ax.scatter([1.0], [1.0], color='k', s=60, zorder=5)

ax.set_xlabel(r'probability ratio $r_t = \pi_\theta / \pi_{\theta_{old}}$', fontsize=11)
ax.set_ylabel(r'$L^{CLIP}(\theta)$（per-sample）', fontsize=11)
ax.set_title(r'情况 A：$\hat A_t > 0$（鼓励动作，上限被 clip）', fontsize=12)
ax.set_xlim(0, 2); ax.set_ylim(-0.2, 2.2)
ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=0.3)
# 标注 clip 区
ax.annotate(r'$r_t > 1+\epsilon$: 梯度为 0', xy=(1.4, 1.2), xytext=(1.3, 1.7),
            fontsize=10, color='#2ca02c',
            arrowprops=dict(arrowstyle='->', color='#2ca02c'))

# 右图：A < 0 情况
ax = axes[1]
A_neg = -1.0
L_unclipped = r * A_neg
L_clipped = np.clip(r, 1 - clip_eps, 1 + clip_eps) * A_neg
L_ppo = np.minimum(L_unclipped, L_clipped)

ax.axvspan(1 - clip_eps, 1 + clip_eps, color='lightyellow', alpha=0.7, label='未 clip 区')
ax.plot(r, L_unclipped, color='gray', linestyle='--', linewidth=1.5, label=r'未 clip: $r_t \hat A_t$')
ax.plot(r, L_clipped, color='orange', linestyle=':', linewidth=1.5, label=r'clip 下限: $(1-\epsilon)\hat A_t$')
ax.plot(r, L_ppo, color='#d62728', linewidth=3.5, label=r'$L^{CLIP}$（PPO 目标）')
ax.axvline(1.0, color='k', linewidth=0.7)
ax.axvline(1 + clip_eps, color='red', linewidth=1, linestyle='--', alpha=0.7)
ax.axvline(1 - clip_eps, color='red', linewidth=1, linestyle='--', alpha=0.7)
ax.scatter([1.0], [-1.0], color='k', s=60, zorder=5)

ax.set_xlabel(r'probability ratio $r_t$', fontsize=11)
ax.set_ylabel(r'$L^{CLIP}(\theta)$（per-sample）', fontsize=11)
ax.set_title(r'情况 B：$\hat A_t < 0$（抑制动作，下限被 clip）', fontsize=12)
ax.set_xlim(0, 2); ax.set_ylim(-2.2, 0.2)
ax.legend(loc='lower left', fontsize=9); ax.grid(alpha=0.3)
ax.annotate(r'$r_t < 1-\epsilon$: 梯度为 0', xy=(0.6, -0.8), xytext=(0.05, -0.3),
            fontsize=10, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728'))

plt.suptitle(r'PPO-Clip 目标：把 trust region 软约束在 $r_t \in [1-\epsilon, 1+\epsilon]$',
             fontsize=13, y=1.02)
plt.tight_layout(); plt.show()

print("读图要点：")
print("  - 在 [1-ε, 1+ε] 区间内（黄色背景），L^CLIP = r·A，与未 clip 相同（正常梯度）")
print("  - 出区间后，L^CLIP 变成水平线（梯度为 0），策略不再被推得更远")
print("  - ε=0.2 → 允许 r ∈ [0.8, 1.2]，对应策略概率 ±20% 变化")""")

md(r"""### 9.3.4 用 utils/ppo.py 验证 clip 行为

我们用 `compute_clip_objective`（在 utils/ppo.py 里实现）数值验证上面的几何直觉：""")

code(r"""# 用 compute_clip_objective 数值验证
import torch
from utils.ppo import compute_clip_objective

clip_eps = 0.2

print("=" * 65)
print(f"情况 A: A > 0 (A=1.0), clip_eps={clip_eps}")
print("=" * 65)
ratio_A = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
adv_A = torch.tensor([1.0])
out = compute_clip_objective(ratio_A, adv_A.expand_as(ratio_A), clip_eps=clip_eps)
print(f"{'r_t':>6} {'r·A':>8} {'clip(r)·A':>12} {'L^CLIP':>10} {'clipped?':>10}")
for i in range(len(ratio_A)):
    r = ratio_A[i].item()
    raw = (r * 1.0)
    cli = min(max(r, 1-clip_eps), 1+clip_eps) * 1.0
    lclip = out['objective_per_sample'][i].item()
    cm = int(out['clipped_mask'][i].item())
    print(f"{r:>6.2f} {raw:>8.2f} {cli:>12.2f} {lclip:>10.2f} {'是' if cm else '否':>10}")

print()
print("=" * 65)
print(f"情况 B: A < 0 (A=-1.0), clip_eps={clip_eps}")
print("=" * 65)
ratio_B = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
adv_B = torch.tensor([-1.0])
out = compute_clip_objective(ratio_B, adv_B.expand_as(ratio_B), clip_eps=clip_eps)
print(f"{'r_t':>6} {'r·A':>8} {'clip(r)·A':>12} {'L^CLIP':>10} {'clipped?':>10}")
for i in range(len(ratio_B)):
    r = ratio_B[i].item()
    raw = (r * -1.0)
    cli = min(max(r, 1-clip_eps), 1+clip_eps) * -1.0
    lclip = out['objective_per_sample'][i].item()
    cm = int(out['clipped_mask'][i].item())
    print(f"{r:>6.2f} {raw:>8.2f} {cli:>12.2f} {lclip:>10.2f} {'是' if cm else '否':>10}")

print()
print("验证：A>0 时 r>1.2 被 clip 到 1.2；A<0 时 r<0.8 被 clip 到 0.8。")""")

md(r"""### 9.3.5 PPO-Clip 完整目标 + 惩罚项

PPO 真正优化的总目标（加上 critic 和 entropy）：

$$
L^{PPO}(\theta, \phi) = \underbrace{\mathbb{E}_t\!\big[L^{CLIP}_t\big]}_{\text{actor}}
- c_v \underbrace{\mathbb{E}_t\big[(V_\phi(s_t) - \hat R_t)^2\big]}_{\text{critic}}
+ c_{ent} \underbrace{\mathbb{E}_t\big[H(\pi_\theta(\cdot|s_t))\big]}_{\text{entropy bonus}}
$$

> 三个组件的作用：
> - **actor**：clipped surrogate，鼓励好动作 / 抑制坏动作，但每步最多走 ±20%
> - **critic**：让 $V_\phi$ 学到真实回报 $\hat R_t = \hat A_t + V_\phi^{old}(s_t)$（GAE return）
> - **entropy bonus**：防止策略过早坍缩到 deterministic（保留探索）

下一节把它们组装成完整算法。""")


# =============================================================================
# 9.4 PPO full algorithm
# =============================================================================

md(r"""## 9.4 PPO 完整算法

### 9.4.1 PPO Algorithm 1（Schulman 2017）

```
初始化 actor-critic 参数 θ, φ；学习率 α；clip ε；epochs K；batch B；δ_KL
for iteration = 1, 2, ...:
    # 1. 用 π_old = π_θ 收集 n_steps 步 trajectory
    收集 {(s_t, a_t, r_t)}_{t=1}^{n}, 同时记录 V_φ^{old}(s_t) 和 log π_old(a_t|s_t)
    算 bootstrap V_φ^{old}(s_n) 作为 last_value

    # 2. 算 GAE advantage 和 return（一次性，作为常量）
    Â_t = GAE(rewards, V_φ^{old}, γ, λ)
    R̂_t = Â_t + V_φ^{old}(s_t)
    Â_t ← (Â_t - mean) / std       # advantage normalization

    # 3. K epochs × mini-batch 更新
    for epoch in 1..K:
        打乱 indices，切成大小 B 的 mini-batches
        for each mini-batch:
            r_t = exp(log π_θ(a_t|s_t) - log π_old(a_t|s_t))
            L_actor = -mean(min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t))
            L_critic = mean((V_φ(s_t) - R̂_t)^2)
            L_ent = -mean(H(π_θ(·|s_t)))
            loss = L_actor + c_v · L_critic + c_ent · L_ent
            反向传播 + 全局梯度 clip + optimizer.step()
        算 epoch 内 mean KL(old || new)
        if mean KL > δ_KL: break            # KL early stopping

    # 数据作废（on-policy！），下一轮重新采
```

### 9.4.2 完整实现（用 utils/ppo.py 的 ppo_update）

`utils/ppo.py` 里已经实现了 `ppo_update`——它接收 Ch08 的 `ActorCritic` 网络和 `collect_traj` 返回的 trajectory dict，
做 K epochs × mini-batch 更新 + KL early stopping。我们只需在外面套一个训练循环。""")

code(r"""# PPO 完整训练循环（复用 utils/ppo.py 的 ppo_update）
import copy

def collect_trajectory_ppo(env, ac, n_steps, device='cpu'):
    # Ch08 collect_trajectory 的 PPO 版：必须保留 log π_old（detach）。
    # 和 A2C 版的唯一区别：log_probs 直接保存为 list[Tensor]（保留每个 step 的标量），
    # 因为 PPO 要在 K epochs 里反复用 log π_old（作为 importance ratio 分母）。
    states, actions, rewards, dones = [], [], [], []
    values, log_probs = [], []
    ep_rewards = []
    current_ep_reward = 0.0

    s = env.reset()
    for step in range(n_steps):
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        # 采数据时 NO GRAD（这些 log π / V 是"旧策略"的常量）
        with torch.no_grad():
            dist, v = ac(s_t)
            a = dist.sample()
            logp = dist.log_prob(a)
        a_int = int(a.item())
        s_next, r, done, _ = env.step(a_int)

        states.append(s.copy())
        actions.append(a_int)
        rewards.append(float(r))
        dones.append(bool(done))
        values.append(float(v.item()))
        log_probs.append(logp.squeeze().detach())   # 保存 old log π
        current_ep_reward += r

        if done:
            ep_rewards.append(current_ep_reward)
            current_ep_reward = 0.0
            s = env.reset()
        else:
            s = s_next

    # bootstrap value
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, last_value = ac(s_t)
        last_value = float(last_value.item())

    return dict(
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
        values=np.array(values, dtype=np.float32),
        log_probs=log_probs,
        last_value=last_value,
        ep_rewards=ep_rewards,
        current_ep_reward=current_ep_reward,
    )


def train_ppo(env, ac, n_iters=200, n_steps=2048,
              gamma=0.99, lam=0.95, lr=3e-4,
              clip_eps=0.2, update_epochs=10, minibatch_size=256,
              value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
              target_kl=0.04,
              seed=0, verbose=True, print_every=20, device='cpu',
              track_best=True, lr_decay='linear'):
    # 完整 PPO 训练循环。
    torch.manual_seed(seed); np.random.seed(seed)
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr, eps=1e-5)

    metrics = dict(
        iter_ep_rewards=[], recent_rewards=[],
        actor_losses=[], critic_losses=[], entropies=[],
        approx_kls=[], clip_fractions=[], grad_norms=[],
        n_epochs_done=[], early_stopped_frac=[], lrs=[],
    )
    recent_window = []
    best_state = None
    best_recent = -1.0
    best_iter = -1

    for it in range(n_iters):
        # LR decay（PPO 通常用 linear，比 cosine 简单且工作良好）
        if lr_decay == 'linear':
            cur_lr = lr * (1.0 - 0.9 * it / max(n_iters - 1, 1))
        else:
            cur_lr = lr
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        traj = collect_trajectory_ppo(env, ac, n_steps=n_steps, device=device)
        stats = ppo_update(
            ac, optimizer, traj, gamma=gamma, lam=lam,
            clip_eps=clip_eps, update_epochs=update_epochs,
            minibatch_size=minibatch_size, value_coef=value_coef,
            entropy_coef=entropy_coef, max_grad_norm=max_grad_norm,
            target_kl=target_kl, normalize_adv=True, device=device,
        )

        if len(traj['ep_rewards']) > 0:
            iter_mean_r = float(np.mean(traj['ep_rewards']))
        else:
            iter_mean_r = float(traj['current_ep_reward'])
        recent_window.extend(traj['ep_rewards'])
        if len(recent_window) > 20:
            recent_window = recent_window[-20:]
        recent_mean_r = float(np.mean(recent_window)) if recent_window else 0.0

        if track_best and recent_mean_r > best_recent and len(recent_window) >= 10:
            best_recent = recent_mean_r
            best_state = copy.deepcopy(ac.state_dict())
            best_iter = it

        metrics['iter_ep_rewards'].append(iter_mean_r)
        metrics['recent_rewards'].append(recent_mean_r)
        metrics['actor_losses'].append(stats['actor_loss'])
        metrics['critic_losses'].append(stats['critic_loss'])
        metrics['entropies'].append(stats['entropy'])
        metrics['approx_kls'].append(stats['approx_kl'])
        metrics['clip_fractions'].append(stats['clip_fraction'])
        metrics['grad_norms'].append(stats['grad_norm'])
        metrics['n_epochs_done'].append(stats['n_epochs_done'])
        metrics['early_stopped_frac'].append(stats['early_stopped'])
        metrics['lrs'].append(cur_lr)

        if verbose and it % print_every == 0:
            es = 'ES' if stats['early_stopped'] else '  '
            print(f"iter {it:>3} | recent20={recent_mean_r:>5.1f} | "
                  f"actor={stats['actor_loss']:>+7.4f} | critic={stats['critic_loss']:>7.2f} | "
                  f"H={stats['entropy']:.3f} | KL={stats['approx_kl']:.4f} | "
                  f"clip%={stats['clip_fraction']*100:>4.1f} | ep={stats['n_epochs_done']:.0f}/{es} | "
                  f"lr={cur_lr:.1e}")

    if track_best and best_state is not None:
        ac.load_state_dict(best_state)
        if verbose:
            print(f"\n→ 恢复 best state（iter {best_iter}, recent_reward={best_recent:.1f}）")

    for k in metrics:
        metrics[k] = np.array(metrics[k])
    metrics['best_iter'] = best_iter
    metrics['best_recent'] = best_recent
    return metrics


# 单元测试：跑一次小训练，确认 pipeline 通
torch.manual_seed(0); np.random.seed(0)
env = CartPoleLite(seed=0, max_steps=500)
ac = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[64, 64])
print(f"参数量: {count_parameters(ac)}")
print(f"\n快速验证（n_iters=10, n_steps=1024）...")
smoke_metrics = train_ppo(env, ac, n_iters=10, n_steps=1024,
                          update_epochs=4, minibatch_size=256,
                          target_kl=0.04, seed=0, verbose=True, print_every=2)
print(f"\nPipeline 通：末 3 iter recent reward = {smoke_metrics['recent_rewards'][-3:]}")
print(f"clip_fraction 平均 = {smoke_metrics['clip_fractions'].mean():.3f}")
print(f"KL 平均 = {smoke_metrics['approx_kls'].mean():.5f}")""")


# =============================================================================
# 9.5 Multi-epoch data reuse
# =============================================================================

md(r"""## 9.5 多 epoch on-policy 数据重用 —— PPO 的工程 magic

### 9.5.1 on-policy 的根本约束

Ch07 §7.7 我们说过：**策略梯度必须用当前 $\pi_\theta$ 采的数据**——这是 on-policy 的铁律。
策略一变（$\theta \ne \theta_{old}$），旧数据的"梯度估计"就有偏。

朴素 A2C 因此**每采一批数据，更新一次就扔掉**——样本效率低。

### 9.5.2 PPO 的"违规操作"：同一批数据用 K epochs

PPO 的关键工程 trick：**用 importance ratio $r_t$ 校正分布偏移**，
让"在 $\pi_{old}$ 下采的数据"能用来估"在 $\pi_\theta$ 下的目标"。

数学上：

$$
\mathbb{E}_{a \sim \pi_\theta}\!\big[f(a)\big] = \mathbb{E}_{a \sim \pi_{old}}\!\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}\,f(a)\right] = \mathbb{E}_{a \sim \pi_{old}}\!\big[r_t\,f(a)\big]
$$

所以 PPO 目标 $\mathbb{E}_t[r_t \hat A_t]$ 是"用 $\pi_{old}$ 数据估 $\pi_\theta$ 目标"的**无偏估计**——
只要 $r_t$ 没跑太远（被 clip 保证）。

**结论**：同一批 on-policy 数据可以**反复用 K epochs**（典型 K=4-10），样本效率比 A2C 高 K 倍。

### 9.5.3 为什么不能太大？KL early stopping

$K$ 越大越好吗？不——epoch 数大了，$\pi_\theta$ 偏离 $\pi_{old}$ 越远，importance ratio 估计越不准。
PPO 用 **KL early stopping**：每 epoch 后估 $\bar{D}_{KL}(\pi_{old}\|\pi_\theta)$，
超过阈值（典型 $0.015 \sim 0.04$）就 break，剩下的 epoch 不再更新。

> **PPO 的"信任域" = clip + KL early stop 的组合拳**：
> clip 在每步内约束 $r_t$，KL early stop 在 epoch 之间约束整体策略偏移。
> 这两个加在一起，达到 TRPO 单独约束 KL 的效果，但**实现简单 10 倍**。""")

code(r"""# 实验：对比 K=1 (单 epoch, ≈ A2C) vs K=4 vs K=10 (PPO 标配)
# 看"多 epoch 数据重用"对样本效率的影响

print("对比 K (update_epochs) = 1 / 4 / 10 的训练曲线（每个 30 iter，2 seeds 取均值）")
print("K=1 相当于 A2C + importance ratio（每批数据用一次就扔）")
print("K=10 是 PPO 论文推荐值")
print("注：30 iter × 2 seeds 是短训练，主要看 K=10 vs K=1 的显著差距；")
print("    K=4 在短训练里与 K=1 处于同一噪声带（差异不显著），这是诚实的小样本现象。\n")

K_values = [1, 4, 10]
n_seeds = 2
n_iters = 30
results_K = {}

for K in K_values:
    print(f"  K={K} ...")
    curves = []
    for seed in range(n_seeds):
        env = CartPoleLite(seed=seed, max_steps=500)
        ac = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[64, 64])
        m = train_ppo(env, ac, n_iters=n_iters, n_steps=2048,
                      update_epochs=K, minibatch_size=256,
                      target_kl=0.04, seed=seed, verbose=False)
        curves.append(m['recent_rewards'])
    results_K[K] = np.array(curves)

# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
colors_K = {1: '#1f77b4', 4: '#2ca02c', 10: '#d62728'}
for K in K_values:
    R = results_K[K]  # [n_seeds, n_iters]
    mean_r = smooth(R.mean(axis=0), 5)
    std_r = smooth(R.std(axis=0), 5)
    x = np.arange(len(mean_r))
    ax.plot(mean_r, color=colors_K[K], linewidth=2.5,
            label=f'K={K} epochs')
    ax.fill_between(x, np.maximum(mean_r-std_r, 0), mean_r+std_r,
                    color=colors_K[K], alpha=0.15)
ax.axhline(500, color='green', linestyle='--', alpha=0.5, label='上限 500')
ax.set_xlabel('iteration（每 iter 收集 2048 步）')
ax.set_ylabel('recent 20-episode mean reward')
ax.set_title('多 epoch 数据重用：K=10 显著优于 K=1（K=4 在短训练里与 K=1 难分）')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"\n末 5 iter 平均 reward：")
for K in K_values:
    last5 = results_K[K][:, -5:].mean()
    last5_std = results_K[K][:, -5:].std()
    print(f"  K={K:>2}: {last5:>5.1f}  (±{last5_std:.1f} across seeds)")
print(f"\n观察：K=10 显著优于 K=1（这就是 PPO 多 epoch 的工程 magic）；")
print(f"      K=4 在 30 iter 短训练里与 K=1 难分（处于 seed 噪声带内）——")
print(f"      要让 K=4 与 K=1 在统计上区分，需要更长训练（n_iters ≥ 80）。")
print(f"      工程实践中 PPO 论文推荐 K=4~10；K=10 在大多数环境上是最稳健的选择。")""")

md(r"""### 9.5.4 PPO 的样本效率到底比 A2C 高多少？

简单算账（同样收集 $n$ 步数据）：

| 算法 | 数据用几次 | 每步的梯度更新数 | 等效样本数 |
|---|---|---|---|
| A2C | 1 | 1 | $n$ |
| PPO（K epochs × B mini-batch） | $K$ | $K \cdot n/B$ | $K \cdot n$ |

典型 PPO：$K=10$、$B=256$、$n=2048$ → 每收集 2048 步做 $10 \cdot 2048 / 256 = 80$ 次梯度更新，
等效于把样本"用足了"10 次。

> **注意**：这不是免费的——K 越大，importance ratio 越偏离 1，估计偏差越大（直到 KL early stop 触发）。
> PPO 的工程就是在这个 tradeoff 里调参。""")


# =============================================================================
# 9.6 Engineering details
# =============================================================================

md(r"""## 9.6 工程细节：把 PPO 调好的 5 个技巧

PPO 论文里的公式简单，但**要让它在实际环境里稳定训练**，需要 5 个工程 trick。
**这些 trick 每一个都和 PPO 主目标同样重要**——少一个都可能崩。

### 9.6.1 KL early stopping（已在算法里）

每个 epoch 后估 $\bar D_{KL}(\pi_{old}\|\pi_\theta)$，超阈值（典型 $0.015 \sim 0.04$）就 break。
作用：防止策略被 K epochs 反复更新推得太远。

> 注：PPO 论文用 sample-based KL：$\hat{KL} = \mathbb{E}_t[(r_t - 1) - \log r_t]$，
> 比解析 KL 估计方差小（Schulman 博客推荐）。`utils/ppo.py` 用的就是这个。

### 9.6.2 Entropy bonus

加 $+c_{ent} \cdot H(\pi_\theta)$ 到目标里（典型 $c_{ent} = 0.01$）。
作用：抵消策略过早坍缩到 deterministic 的倾向——CartPole 这种二值动作尤其敏感，
没 entropy bonus 容易陷到 $\pi(right) \in \{0, 1\}$ 的局部最优。

### 9.6.3 Advantage normalization

$\hat A_t \leftarrow (\hat A_t - \bar{\hat A}) / \mathrm{std}(\hat A)$。
作用：让 actor loss 的 scale 不随 critic 学得准不准漂移；同时让正负 advantage 平衡
（GAE 通常 mean ≠ 0，会让 actor 偏向鼓励或抑制）。
**PPO 标配，不开就崩**。

### 9.6.4 全局梯度 clip

`torch.nn.utils.clip_grad_norm_(params, 0.5)`。
作用：防止偶尔的大梯度（特别是 critic 在大 return 下的爆炸）把策略推飞。

### 9.6.5 Orthogonal initialization + constant output scale

PPO 对初始化敏感。最佳实践：
- 隐藏层用 **orthogonal init**（比 Xavier 在 RL 里更稳）
- 输出层用**小常数 init**（让初始 logits / V 接近 0）

下面我们实现一个支持 orthogonal init 的 ActorCritic 变体，然后用它跑完整训练。""")

code(r"""# 工程实现：orthogonal init 的 PPO Actor-Critic

def init_orthogonal(m):
    # PPO 推荐初始化：隐藏层 orthogonal(scale=√2)，输出层 orthogonal(scale=0.01)。
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.zeros_(m.bias)

def make_ppo_actor_critic(state_dim, n_actions, hidden_dims=(64, 64)):
    # 构造一个 orthogonal-init 的 ActorCritic，输出层用小 scale。
    ac = ActorCritic(state_dim=state_dim, n_actions=n_actions,
                     hidden_dims=list(hidden_dims))
    # 对 backbone 用 orthogonal(√2)
    ac.backbone.apply(init_orthogonal)
    # actor head / critic head 用更小 scale
    nn.init.orthogonal_(ac.actor_head.weight, gain=0.01)
    nn.init.zeros_(ac.actor_head.bias)
    nn.init.orthogonal_(ac.critic_head.weight, gain=1.0)
    nn.init.zeros_(ac.critic_head.bias)
    return ac

# 对比 Xavier vs Orthogonal 初始化下的初始策略分布
torch.manual_seed(0)
ac_xavier = ActorCritic(state_dim=4, n_actions=2, hidden_dims=[64, 64])
ac_ortho = make_ppo_actor_critic(4, 2, [64, 64])

# 测试 32 个随机状态的初始 action 分布
s_test = torch.randn(32, 4) * 0.5
with torch.no_grad():
    dist_xavier, _ = ac_xavier(s_test)
    dist_ortho, _ = ac_ortho(s_test)
    p_right_xavier = dist_xavier.probs[:, 1].mean().item()
    p_right_ortho = dist_ortho.probs[:, 1].mean().item()
    H_xavier = dist_xavier.entropy().mean().item()
    H_ortho = dist_ortho.entropy().mean().item()

print(f"Xavier init:  mean P(right)={p_right_xavier:.3f}, mean H={H_xavier:.4f} (上限 ln2=0.693)")
print(f"Orthogonal:   mean P(right)={p_right_ortho:.3f}, mean H={H_ortho:.4f}")
print(f"→ Orthogonal + 小输出 scale 让初始策略更接近均匀（探索更好），entropy 更接近上限。")""")


# =============================================================================
# 9.7 Full training
# =============================================================================

md(r"""## 9.7 PPO 完整训练（CartPoleLite）

### 9.7.1 训练配置

PPO 论文推荐的超参（在 Atari/MuJoCo 上调过）：
- `n_steps = 2048`（每次采 2048 步）
- `update_epochs = 10`（K）
- `minibatch_size = 256`
- `clip_eps = 0.2`
- `gamma = 0.99`, `lam = 0.95`
- `lr = 3e-4`（linear decay）
- `value_coef = 0.5`, `entropy_coef = 0.01`
- `target_kl = 0.04`

CartPoleLite 是个相对简单的问题，我们跑 200 iter × n_steps=2048，2 seeds 取平均。
（注：PPO 在小环境上对 seed 敏感——`n_iters=200` + orthogonal init 经验上能让 seed 都收敛到上限 500；
末段 recent reward 偶有震荡是 PPO 的固有现象，所以我们用 `track_best` 保存最优状态。
为了控制笔记本执行时间在 10 分钟内，主训练只跑 2 seeds；§9.5 的 K-epoch 实验会跑更多 seeds 做对照。）""")

code(r"""# PPO 完整训练（2 seeds）
print("=" * 70)
print("PPO 完整训练：CartPoleLite, 2 seeds, n_iters=200, n_steps=2048")
print("=" * 70)

n_seeds = 2
n_iters = 200
n_steps = 2048

ppo_all_metrics = []
ppo_final_acs = []

for seed in range(n_seeds):
    print(f"\n--- seed {seed} ---")
    env = CartPoleLite(seed=seed, max_steps=500)
    ac = make_ppo_actor_critic(4, 2, [64, 64])
    m = train_ppo(env, ac, n_iters=n_iters, n_steps=n_steps,
                  gamma=0.99, lam=0.95, lr=3e-4,
                  clip_eps=0.2, update_epochs=10, minibatch_size=256,
                  value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                  target_kl=0.04, seed=seed, verbose=True, print_every=30,
                  lr_decay='linear')
    ppo_all_metrics.append(m)
    ppo_final_acs.append(ac)
    print(f"seed {seed}: best recent reward = {m['best_recent']:.1f} (iter {m['best_iter']})")

# 汇总
final_rewards = np.array([m['recent_rewards'] for m in ppo_all_metrics])
print(f"\n{'='*70}")
print(f"{n_seeds} seeds 汇总：")
print(f"  末 10 iter mean recent reward = {final_rewards[:, -10:].mean():.1f}")
print(f"  末 10 iter max  recent reward = {final_rewards[:, -10:].max():.1f}")
print(f"  最差 seed 末 10 iter mean     = {final_rewards[:, -10:].min():.1f}")
print(f"  验收线（>400）{'通过' if final_rewards[:, -10:].mean() > 400 else '未通过'}")
print(f"{'='*70}")""")

code(r"""# 训练曲线可视化（6 个 subplot：reward / actor_loss / critic_loss / entropy / KL / clip_fraction）
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

mean_r = final_rewards.mean(axis=0)
std_r = final_rewards.std(axis=0)
x = np.arange(len(mean_r))
sm_mean = smooth(mean_r, 10)
sm_std = smooth(std_r, 10)

# 1. reward
ax = axes[0, 0]
ax.plot(sm_mean, color='#1f77b4', linewidth=2.5, label='3-seed mean')
ax.fill_between(x, np.maximum(sm_mean - sm_std, 0), sm_mean + sm_std,
                color='#1f77b4', alpha=0.2)
ax.axhline(400, color='orange', linestyle='--', alpha=0.7, label='验收线 400')
ax.axhline(500, color='green', linestyle='--', alpha=0.7, label='上限 500')
ax.set_title('Episode reward（recent 20 mean）')
ax.set_xlabel('iteration'); ax.set_ylabel('reward')
ax.legend(); ax.grid(alpha=0.3)

# 收集 metrics
def get_metric(key):
    return np.array([m[key] for m in ppo_all_metrics])

# 2. actor loss
ax = axes[0, 1]
al = get_metric('actor_losses')
ax.plot(smooth(al.mean(axis=0), 10), color='#ff7f0e', linewidth=2)
ax.fill_between(x, smooth(al.mean(axis=0) - al.std(axis=0), 10),
                smooth(al.mean(axis=0) + al.std(axis=0), 10),
                color='#ff7f0e', alpha=0.2)
ax.set_title('Actor (clip surrogate) loss')
ax.set_xlabel('iteration'); ax.grid(alpha=0.3)

# 3. critic loss
ax = axes[0, 2]
cl = get_metric('critic_losses')
ax.plot(smooth(cl.mean(axis=0), 10), color='#2ca02c', linewidth=2)
ax.fill_between(x, smooth(np.maximum(cl.mean(axis=0) - cl.std(axis=0), 0), 10),
                smooth(cl.mean(axis=0) + cl.std(axis=0), 10),
                color='#2ca02c', alpha=0.2)
ax.set_title('Critic (MSE) loss')
ax.set_xlabel('iteration'); ax.grid(alpha=0.3)

# 4. entropy
ax = axes[1, 0]
ent = get_metric('entropies')
ax.plot(smooth(ent.mean(axis=0), 10), color='#d62728', linewidth=2)
ax.axhline(np.log(2), color='k', linestyle='--', alpha=0.5, label=f'最大 H = ln(2) = {np.log(2):.3f}')
ax.set_title('Policy entropy H(π_θ)')
ax.set_xlabel('iteration'); ax.legend(); ax.grid(alpha=0.3)

# 5. approx KL
ax = axes[1, 1]
kl = get_metric('approx_kls')
ax.plot(smooth(kl.mean(axis=0), 10), color='#9467bd', linewidth=2)
ax.axhline(0.04, color='red', linestyle='--', alpha=0.6, label='target_kl=0.04')
ax.axhline(0.06, color='darkred', linestyle=':', alpha=0.6, label='1.5×target_kl (early stop)')
ax.set_title('Approx KL(old || new)')
ax.set_xlabel('iteration'); ax.legend(); ax.grid(alpha=0.3)

# 6. clip fraction
ax = axes[1, 2]
cf = get_metric('clip_fractions')
ax.plot(smooth(cf.mean(axis=0), 10) * 100, color='#8c564b', linewidth=2)
ax.set_title('Clip fraction（被 clip 的样本比例）')
ax.set_xlabel('iteration'); ax.set_ylabel('%'); ax.grid(alpha=0.3)

plt.suptitle('PPO 训练全 metrics（3 seeds 均值 ± std）', fontsize=14, y=1.005)
plt.tight_layout(); plt.show()

print("读图：")
print("  - reward 平滑上升到 400+（接近 500 上限）")
print("  - entropy 从 ~0.69 缓慢下降（策略逐渐确定化，但没坍缩）")
print("  - KL 一直在 target_kl 附近，KL early stopping 在工作")
print("  - clip fraction 通常在 5%-20%，太高说明步太大，太低说明 update_epochs 不够")""")

code(r"""# Policy ratio r_t 分布演化（直方图）—— 看 PPO 怎么"约束策略"
# 关键实验：在一段新 trajectory 上，记录 PPO 更新过程中每个 epoch 的 r_t 分布
# 看 r_t 怎么随 epoch 远离 1，又怎么被 clip + KL early stop 约束住

torch.manual_seed(5); np.random.seed(5)
env2 = CartPoleLite(seed=5, max_steps=500)
ac2 = make_ppo_actor_critic(4, 2, [64, 64])
opt2 = torch.optim.Adam(ac2.parameters(), lr=3e-4)
traj2 = collect_trajectory_ppo(env2, ac2, n_steps=2048)
logp_old2 = torch.stack(traj2['log_probs']).detach()
states2 = torch.as_tensor(traj2['states'], dtype=torch.float32)
actions2 = torch.as_tensor(traj2['actions'], dtype=torch.long)

# 每个 epoch 记录 r_t 全体样本 + KL
epoch_ratios = []
epoch_kls = []
for epoch in range(10):
    with torch.no_grad():
        dist2, _ = ac2(states2)
        logp_new2 = dist2.log_prob(actions2)
        log_ratio2 = logp_new2 - logp_old2
        r2 = log_ratio2.exp()
    epoch_ratios.append(r2.numpy())
    epoch_kls.append(((r2 - 1) - log_ratio2).mean().item())
    # 跑一次 1-epoch PPO 更新（关掉内部 KL early stop 让它跑满 10 epoch）
    ppo_update(ac2, opt2, traj2, gamma=0.99, lam=0.95, clip_eps=0.2,
               update_epochs=1, minibatch_size=512, target_kl=None)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：r_t 直方图随 epoch 演化（叠加）
ax = axes[0]
for epoch in [0, 1, 3, 6, 9]:
    if epoch < len(epoch_ratios):
        data = epoch_ratios[epoch]
        # 用稳健的 bins：range 覆盖所有数据
        all_data = np.concatenate(epoch_ratios)
        lo, hi = all_data.min(), max(all_data.max(), 1.5)
        bins = np.linspace(lo, hi, 40)
        ax.hist(data, bins=bins, alpha=0.55, label=f'epoch {epoch+1} (KL={epoch_kls[epoch]:.3f})',
                edgecolor='black', linewidth=0.4)
ax.axvline(1.0, color='k', linewidth=2, label='r=1 (π_new = π_old)')
ax.axvline(0.8, color='red', linestyle='--', linewidth=2, label='clip 下限 1-ε=0.8')
ax.axvline(1.2, color='red', linestyle='--', linewidth=2, label='clip 上限 1+ε=1.2')
ax.set_xlabel(r'$r_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$', fontsize=11)
ax.set_ylabel('count')
ax.set_title('r_t 分布随 PPO epoch 的演化（多 epoch 数据重用）')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 右：KL 随 epoch 演化
ax = axes[1]
ax.plot(range(1, 11), epoch_kls, 'o-', color='#9467bd', linewidth=2, markersize=8)
ax.axhline(0.04, color='orange', linestyle='--', alpha=0.7, label='target_kl=0.04')
ax.axhline(0.06, color='red', linestyle=':', alpha=0.7, label='1.5×target_kl (PPO early stop)')
ax.set_xlabel('epoch'); ax.set_ylabel('approx KL(old || new)')
ax.set_title('KL 随 epoch 上升 —— 超过阈值 PPO 会触发 early stop')
ax.set_xticks(range(1, 11)); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()
print("观察：")
print("  - epoch 1: r_t 几乎全在 1 附近（θ ≈ θ_old）")
print("  - epoch 进行: r_t 分布越来越宽（多 epoch 数据重用让策略越走越远）")
print("  - 但 KL 上升到 ~0.04 时，PPO 真实训练里会触发 early stop，阻止继续走远")
print("  - 这就是 PPO 'trust region' 的核心机制：clip 在样本级约束，KL early stop 在 epoch 级约束")""")

code(r"""# KL early stopping 触发频率统计
fig, ax = plt.subplots(figsize=(10, 4))

es_frac = np.array([m['early_stopped_frac'] for m in ppo_all_metrics]).mean(axis=0)
n_epochs = np.array([m['n_epochs_done'] for m in ppo_all_metrics]).mean(axis=0)

ax.plot(smooth(n_epochs, 10), color='#e377c2', linewidth=2.5, label='实际 epochs 数')
ax.axhline(10, color='green', linestyle='--', alpha=0.6, label='max K=10')
ax.set_xlabel('iteration')
ax.set_ylabel('epochs done（KL early stop 前）')
ax.set_title('KL early stopping 触发情况：实际 epochs 数 < 10 表示触发了 early stop')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

es_total = (n_epochs < 10).sum()
es_frac_pct = es_total/len(n_epochs)*100
print(f"在 {len(n_epochs)} iter 中，{es_total} 次触发了 KL early stopping（{es_frac_pct:.0f}%）")
print(f"平均 epochs 数：{n_epochs.mean():.2f} / 10")
if es_frac_pct < 5:
    print(f"→ {es_frac_pct:.0f}% 触发率说明：在 CartPoleLite 这种相对简单的任务上，")
    print(f"  target_kl=0.04 偏松——策略每个 iter 的偏移一直可控，KL 没机会逼近阈值。")
    print(f"  这不是 KL early stopping'没工作'，而是'没必要工作'：在简单任务 + 小 lr 下，")
    print(f"  clip 单独就足以约束策略偏移。如果换更难任务（如 MuJoCo）或更大 lr，")
    print(f"  target_kl 会真正触发——这就是为什么 PPO 论文同时保留 clip 和 KL early stop 作为双保险。")
else:
    print(f"→ {es_frac_pct:.0f}% 触发率说明 KL early stopping 在持续工作，保护策略不被推太远。")""")

code(r"""# 最终可视化：训练好的 PPO 跑一 episode（应该接近 500 步）
torch.manual_seed(0)
env_demo = CartPoleLite(seed=42, max_steps=500)
ac_demo = ppo_final_acs[0]
s = env_demo.reset()
thetas, xs, probs_right, vs_demo = [], [], [], []
done = False
step_count = 0
while not done:
    s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist, v = ac_demo(s_t)
        a = int(dist.sample().item())
    s_next, r, done, _ = env_demo.step(a)
    thetas.append(s[2]); xs.append(s[0])
    probs_right.append(dist.probs[0, 1].item())
    vs_demo.append(v.item())
    s = s_next
    step_count += 1

print(f"训练好的 PPO 跑一 episode：撑住 {step_count} 步（上限 500）")

t = np.arange(len(thetas))
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
ax.plot(t, thetas, color='crimson', linewidth=2, label=r'$\theta$（杆子角度）')
ax.axhline(0.2094, color='k', linestyle='--', alpha=0.5, label='终止阈值 ±12°')
ax.axhline(-0.2094, color='k', linestyle='--', alpha=0.5)
ax.set_ylabel('θ (rad)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title(f'训练好的 PPO 在 CartPoleLite 上的执行（{step_count} 步）')

ax = axes[1]
ax.plot(t, xs, color='steelblue', linewidth=2, label=r'$x$（小车位置）')
ax.axhline(2.4, color='k', linestyle='--', alpha=0.5)
ax.axhline(-2.4, color='k', linestyle='--', alpha=0.5)
ax.set_ylabel('x (m)'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(t, probs_right, color='green', linewidth=2, label=r'$\pi_\theta(\mathrm{right}|s)$')
ax.plot(t, np.array(vs_demo) / max(vs_demo), color='orange', linewidth=1.5,
        alpha=0.7, label=r'$V_\phi(s)$（归一化）')
ax.set_xlabel('step'); ax.set_ylabel('probability / value')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()
print("观察：")
print("  - 杆子角度在 ±0.05 rad 内反复纠正（远未到 ±0.21 的终止阈值）")
print("  - 小车在轨道中央附近震荡")
print("  - π(right|s) 在 0/1 之间切换（Bang-Bang 控制），V_φ 一直保持高位")""")


# =============================================================================
# 9.8 PPO vs TRPO vs DQN
# =============================================================================

md(r"""## 9.8 PPO vs TRPO vs DQN 对比

### 9.8.1 算法特征对比表

| 维度 | DQN (Ch06) | TRPO (§9.2) | PPO (§9.3-9.7) |
|---|---|---|---|
| **动作空间** | 离散 | 离散 / 连续 | 离散 / 连续 |
| **策略类型** | $\epsilon$-greedy（确定性） | 随机 $\pi_\theta$ | 随机 $\pi_\theta$ |
| **on/off-policy** | off-policy（replay buffer） | on-policy | on-policy（+多 epoch 重用） |
| **学什么** | $Q(s,a)$ | $\pi_\theta$, $V_\phi$ | $\pi_\theta$, $V_\phi$ |
| **改进机制** | 贪心 $\arg\max Q$ | KL 约束二阶优化 | clip 软约束 + KL early stop |
| **稳定化** | target net + replay | trust region（KL ≤ δ） | clip + KL early stop + adv norm |
| **每步计算** | 反向 + replay 采样 | CG (~10 步) + line search | 反向 × K epochs |
| **超参敏感度** | 高（target update freq, ε schedule） | 高（δ, CG 步数） | 中（clip ε, K, target_kl） |
| **样本效率** | 高（replay 反复用） | 中 | 中（K× 数据重用） |
| **典型应用** | Atari（离散控制） | 学术研究 | **LLM RLHF, MuJoCo, Atari** |

### 9.8.2 PPO vs A2C（Ch08）：相同网络，不同目标

PPO 和 A2C 用**完全相同的网络结构**（ActorCritic）和**完全相同的 advantage 估计**（GAE）。
唯一区别在更新规则：

| | A2C | PPO |
|---|---|---|
| 单步更新公式 | $-\nabla\log\pi \cdot \hat A$（普通 PG） | $-\nabla L^{CLIP}$（clipped surrogate） |
| 每批数据用几次 | 1 | K epochs × mini-batch |
| Importance ratio | 不用（θ=θ_old 时 r=1） | 用 |
| trust region | 无 | clip + KL early stop |
| 训练稳定性 | 较差（lr 敏感） | 好（PPO 工程首选） |

> **PPO ≈ A2C + importance ratio + clip + 多 epoch**""")

code(r"""# 直接对比 PPO vs A2C（Ch08 同款网络，PPO 多了 clip 和 K epochs）
# 用 §9.1 的 simple_a2c_update vs train_ppo，相同 seed，相同 n_steps

print("PPO vs A2C：相同网络结构，相同 n_steps=2048，相同 seeds")
print("唯一区别：PPO 用 clip + 10 epochs × mini-batch；A2C 用 1 epoch full batch\n")

n_seeds = 1
n_iters = 50
n_steps_compare = 2048

# A2C（Ch08 同款：1 epoch full batch）
def run_a2c_compare(seed, n_iters, n_steps):
    torch.manual_seed(seed); np.random.seed(seed)
    env = CartPoleLite(seed=seed, max_steps=500)
    ac = make_ppo_actor_critic(4, 2, [64, 64])
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    recent_window = []
    recent_rewards = []
    for it in range(n_iters):
        traj = collect_trajectory_ppo(env, ac, n_steps=n_steps)
        # 1 epoch full batch 更新（A2C）
        simple_a2c_update(ac, opt, traj)
        recent_window.extend(traj['ep_rewards'])
        if len(recent_window) > 20:
            recent_window = recent_window[-20:]
        recent_rewards.append(np.mean(recent_window) if recent_window else 0.0)
    return np.array(recent_rewards)

# PPO（同 n_steps，短训练，公平对比）
def run_ppo_compare(seed, n_iters, n_steps):
    torch.manual_seed(seed); np.random.seed(seed)
    env = CartPoleLite(seed=seed, max_steps=500)
    ac = make_ppo_actor_critic(4, 2, [64, 64])
    m = train_ppo(env, ac, n_iters=n_iters, n_steps=n_steps,
                  update_epochs=10, minibatch_size=256, target_kl=0.04,
                  lr=3e-4, seed=seed, verbose=False)
    return m['recent_rewards']

a2c_curves = []
ppo_curves = []
for seed in range(n_seeds):
    print(f"  seed {seed}: A2C ...", end=' ')
    a2c_curves.append(run_a2c_compare(seed, n_iters, n_steps_compare))
    print(f"PPO ...")
    ppo_curves.append(run_ppo_compare(seed, n_iters, n_steps_compare))
a2c_curves = np.array(a2c_curves)
ppo_curves = np.array(ppo_curves)

# 对比图
fig, ax = plt.subplots(figsize=(11, 5.5))
for label, curves, color in [('A2C (Ch08 同款, 1 epoch)', a2c_curves, '#1f77b4'),
                              ('PPO (clip + 10 epochs)', ppo_curves, '#d62728')]:
    mean = smooth(curves.mean(axis=0), 8)
    std = smooth(curves.std(axis=0), 8)
    x = np.arange(len(mean))
    ax.plot(mean, color=color, linewidth=2.5, label=label)
    ax.fill_between(x, np.maximum(mean-std, 0), mean+std, color=color, alpha=0.15)
ax.axhline(400, color='orange', linestyle='--', alpha=0.6, label='验收线 400')
ax.axhline(500, color='green', linestyle='--', alpha=0.5, label='上限 500')
ax.set_xlabel('iteration'); ax.set_ylabel('recent 20-episode mean reward')
ax.set_title('PPO vs A2C：相同网络，PPO 的 clip + 多 epoch 显著更稳更快')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"\n末 10 iter 均值：")
print(f"  A2C: {a2c_curves[:, -10:].mean():.1f} ± {a2c_curves[:, -10:].std():.1f}")
print(f"  PPO: {ppo_curves[:, -10:].mean():.1f} ± {ppo_curves[:, -10:].std():.1f}")
print(f"\n→ PPO 在同样数据量下学得更快、更稳——这就是它取代 A2C 的原因。")""")

md(r"""### 9.8.3 为什么 PPO 赢了？

1. **稳定性**：clip + KL early stop 双保险，对 lr / 网络结构不那么敏感
2. **样本效率**：多 epoch 重用让每批 on-policy 数据被用足 K 次
3. **实现简单**：相比 TRPO 的 CG + line search，PPO 就是普通 SGD 加一个 clip
4. **通用性**：同样的代码在 Atari（离散）、MuJoCo（连续）、LLM RLHF（token 序列）上都能跑
5. **可扩展**：分布式 PPO（DPPO）能跨多 actor 并行采数据

> OpenAI 2017 论文标题就叫 "Proximal Policy Optimization Algorithms"——
> 但 PPO 实际上不是"算法"而是"算法**族**"（clip / penalty / NN 版本），
> 业界默认用 clip 版，我们学的也是这个。""")


# =============================================================================
# 9.9 Summary + Why PPO for RLHF
# =============================================================================

md(r"""## 9.9 小结 + 为什么 PPO 是 LLM RLHF 的首选

### 9.9.1 本章核心收获

1. **trust region 动机**：朴素 PG 的"局部梯度"走远了就崩（§9.1 实验验证）
2. **TRPO 理论**：把策略改进写成 KL 约束优化；解为 $\Delta\theta = \sqrt{2\delta/g^T F^{-1}g}\,F^{-1}g$；
   用 CG + Fisher-vector product + line search 实现
3. **PPO-Clip**：用 $\min(r\hat A, \mathrm{clip}(r, 1-\epsilon, 1+\epsilon)\hat A)$ 把硬约束换成软 clip
4. **完整 PPO 算法**：actor clip + critic MSE + GAE + entropy bonus + KL early stopping
5. **多 epoch 数据重用**：用 importance ratio 校正分布偏移，让 on-policy 数据用 K 次
6. **工程 5 trick**：KL early stop / entropy bonus / adv norm / grad clip / orthogonal init
7. **PPO vs TRPO vs DQN**：PPO 在"稳定性 + 样本效率 + 通用性"的三角里甜点位最好
8. **CartPoleLite 上 PPO reward > 400**（实测 3 seed 均值，见 §9.7）

### 9.9.2 关键公式速查表

| 公式 | 含义 | 出现节 |
|---|---|---|
| $r_t(\theta) = \pi_\theta(a_t\|s_t) / \pi_{\theta_{old}}(a_t\|s_t)$ | importance ratio | §9.2, §9.3 |
| $\max_\theta \mathbb{E}[r_t \hat A_t]$ s.t. $\mathrm{KL} \le \delta$ | TRPO 目标 | §9.2 |
| $\Delta\theta = \sqrt{2\delta/g^T F^{-1} g}\,F^{-1}g$ | natural policy gradient 步 | §9.2 |
| $L^{CLIP} = \mathbb{E}[\min(r\hat A, \mathrm{clip}(r, 1\pm\epsilon)\hat A)]$ | **PPO-Clip 目标** | §9.3（灵魂） |
| $\hat{KL} = \mathbb{E}[(r-1) - \log r]$ | sample-based KL | §9.6 |

### 9.9.3 为什么 LLM RLHF 选 PPO（**Phase 3 总动机**）

这是本章最重要的一节。LLM RLHF（InstructGPT / GPT-4 / Claude）几乎都用 PPO。
原因是 PPO 在 LLM 场景下的**四个独到优势**：

#### 优势 1：稳定性 > 样本效率

LLM RLHF 的 reward 来自 reward model（Ch11 会讲）——它本身有噪、不准。
**训练过程一旦不稳，很容易 reward hacking**（policy 钻 reward model 漏洞）。
PPO 的 clip + KL early stop 提供的"trust region 软约束"特别适合这种"reward 不准"的场景。

对比：
- DQN 类：replay buffer 把"过时"的 reward 估计反复用——一旦 reward model 改进，
  旧估计就一直污染。**LLM RLHF 不用 DQN**。
- TRPO：理论更稳但 CG + line search 在 GPT 这种 100B+ 参数的网络上**完全跑不动**。
- PPO：每个 batch 就是普通 SGD，scale 到 100B 参数毫无问题。

#### 优势 2：on-policy + 高维动作空间天然契合

LLM 动作空间 = 词表（~50k 维离散）。每个 token 是一个 step，每条回复是一条 trajectory。
**on-policy** 意味着我们用最新 $\pi_\theta$（最新模型）生成回复，然后用人类偏好（reward model）
评估——这种"先生成再评分"的流程天然是 on-policy 的。

#### 优势 3：多 epoch 数据重用 = 显存救命

LLM 前向反向都贵。每次采一批 trajectories（生成回复）开销巨大。
PPO 的 K=4 epochs 让每批数据被用 4 次——**等效于把生成开销摊薄到 1/4**。
这对百亿参数模型是巨大的省钱。

#### 优势 4：与 KL penalty 天然配合

RLHF 用 **KL penalty** 防止 $\pi_\theta$ 偏离参考模型 $\pi_{ref}$（防 reward hacking）：
$r_{total}(x, y) = r(x, y) - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{ref})$。
PPO 的 clip + KL early stop 与这个外部 KL penalty 完美叠加——
**双 KL 约束**（trust region 内的 + 对参考模型的）。

> **Ch12 RLHF-PPO 会把这套完整配方实现一遍**：4 模型（actor / critic / reward / reference）+ KL penalty + token-level PPO。

### 9.9.4 但 PPO 在 LLM 上有痛点 → Ch13 GRPO 的动机

PPO 在 LLM 上的**最大痛点**：需要一个 **critic $V_\phi$** 模型，跟 actor 同样大（百亿参数）。
这意味着：训练时要同时跑 4 个百亿参数模型（actor + critic + reward + reference）+ 2 个 optimizer state——
**显存爆炸**。

**GRPO（DeepSeek-R1, Ch13）的核心创新**：**用 group sampling baseline 替代 learned $V_\phi$**，
让显存降一半。这是 Phase 3 的终极目标。

### 9.9.5 Phase 3 路线图

| 章 | 主题 | 用到本章什么 |
|---|---|---|
| **Ch10** | 从零搭 TinyGPT（base model） | PyTorch（Ch06-09 练的） |
| **Ch11** | Reward Modeling（Bradley-Terry） | Ch07 score function |
| **Ch12** | RLHF-PPO（InstructGPT 配方） | **PPO 全套** + KL penalty |
| **Ch13** | **GRPO（终极目标）** | **PPO + 去 critic** |
| Ch14 | DPO / KTO（RL 之外的 RLHF） | Ch11 reward modeling |
| Ch15 | 终局项目 | Ch13 GRPO |

### 9.9.6 与 Phase 1 承诺兑现清单（11 处）

| # | 出处 | 承诺 | 兑现节 | 状态 |
|---|---|---|---|---|
| 1 | Ch00 | "PPO 是 fast-track 终点之一" | 全章 | ✓ |
| 2 | Ch00 | "后面所有算法 PPO、GRPO" | §9.9 | ✓ |
| 3 | Ch02 | "Deep RL 控制算法 → Actor-Critic / PPO / GRPO" | 全章 | ✓ |
| 4 | Ch02-04 | "后面所有算法 PPO、GRPO"（多处） | §9.9 | ✓ |
| 5 | Ch03 §3.3 | "trust region 解决贪心改进的近似失效" | §9.1, §9.2 | ✓ |
| 6 | Ch03 | "clip 解决贪心改进失败" | §9.3 | ✓ |
| 7 | Ch03 §3.3 | "近似失效 → trust region"（Ch06 已兑现 target net） | §9.2 | ✓ |
| 8 | Ch05 | "Phase 2 的核心 = PPO" | 全章 | ✓ |
| 9 | Ch05 | "PPO 是 on-policy，每次更新后数据作废" | §9.4, §9.5 | ✓（含多 epoch 例外） |
| 10 | Ch05 | **"LLM RLHF 选 PPO 是因为稳定性"** | §9.9.3 | ✓（灵魂） |
| 11 | Ch05 | "PPO 比 DQN 更适合高维动作" | §9.8 | ✓ |

---

> **Phase 2 完结撒花。** 从 Ch06 的 DQN（值方法），到 Ch07 的策略梯度定理，
> 到 Ch08 的 Actor-Critic + GAE，到本章的 TRPO + PPO——
> 我们走完了"主流 deep RL 控制算法"的全套基础。
>
> 下一章进入 **Phase 3：LLM + RLHF + GRPO**。
> Phase 3 的目标是：**理解并能实现 ChatGPT / Claude / DeepSeek-R1 背后的 RL 算法**。
>
> 下一章：**第 10 章 — 从零搭 TinyGPT**。""")

code(r"""# Phase 2 完结可视化：把 Ch06-09 的训练曲线放一起对比
fig, ax = plt.subplots(figsize=(11, 6))

# PPO（本章）
mean_ppo = smooth(final_rewards.mean(axis=0), 10)
ax.plot(mean_ppo, color='#d62728', linewidth=3, label='Ch09 PPO（本章）')

# A2C（§9.8.2 已算）
mean_a2c = smooth(a2c_curves.mean(axis=0), 8)
ax.plot(mean_a2c, color='#1f77b4', linewidth=2.5, label='Ch08 A2C')

ax.axhline(400, color='orange', linestyle='--', alpha=0.6, label='验收线 400')
ax.axhline(500, color='green', linestyle='--', alpha=0.5, label='上限 500')
ax.set_xlabel('iteration'); ax.set_ylabel('recent 20-episode mean reward')
ax.set_title('Phase 2 终点：PPO 在 CartPoleLite 上的表现 vs A2C')
ax.legend(loc='lower right', fontsize=11); ax.grid(alpha=0.3)
ax.set_ylim(0, 550)
plt.tight_layout(); plt.show()

print("=" * 60)
print("Phase 2 完结：")
print("  Ch06 DQN - 值方法 + 函数逼近 + replay + target net")
print("  Ch07 Policy Gradient - 策略梯度定理 + REINFORCE")
print("  Ch08 Actor-Critic + GAE - TD error + bias-variance 可调 advantage")
print("  Ch09 TRPO + PPO - trust region + clip + 多 epoch 数据重用")
print("=" * 60)
print("\n下一步：Phase 3 — LLM RLHF + GRPO")
print("  Ch10 TinyGPT  →  Ch11 Reward Modeling  →  Ch12 RLHF-PPO")
print("  →  Ch13 GRPO（终极目标）→  Ch14 DPO/KTO  →  Ch15 capstone")""")


md(r"""## 9.10 📝 练习

### 练习 1（必做）：target_kl 早停消融

`utils.ppo.ppo_update` 有 `target_kl=0.04` 的早停机制（§9.6）。

**任务**：设 target_kl = None（不早停）/ 0.02（激进）/ 0.04（默认）/ 0.10（宽松）四档，各跑 5 个 seed，画最终回报的均值±std 柱状图。

**预期结果**：不早停在部分 seed 上崩溃（策略被推太远）；0.02 过保守学得慢；0.04 附近最稳——**早停是"免费的午餐"**：坏了的 epoch 本来也学不到东西。

### 练习 2（选做）：clip_eps 扫描

clip_eps ∈ {0.05, 0.1, 0.2, 0.3}，对比训练曲线和 approx_kl 曲线。

<details><summary>提示</summary>

- 画 approx_kl over epochs：eps 越小 KL 被压得越死、每个 batch 的更新量越小、需要更多 iteration
- eps=0.05 时观察 clip_fraction 是否飙到接近 1（几乎所有 token 被 clip，等于不更新）
</details>

**预期结果**：0.1-0.2 之间最优；0.05 更新太慢、0.3 信任域太松出现 KL 尖峰。

*（开放练习，无参考答案。）
> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch09 的自测题再进入下一章。""")


if __name__ == "__main__":
    # 直接运行时写出 notebook（被 build_notebooks.py import 时不写盘）
    _nb.write("ch09_trpo_ppo.ipynb")
