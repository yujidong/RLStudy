r"""DPO（Direct Preference Optimization）/ KTO（Kahneman-Tversky Optimization）
工具集（Ch14）—— Phase 3 收尾章的核心算法。

本章兑现 Ch00 章节图的最后一块拼图：

    **"Ch14 DPO / KTO" —— RL 之外的 RLHF 替代方案。**

--------------------------------------------------------------------
核心思想（一句话）

RLHF（Ch12）/ GRPO（Ch13）需要：reward model + critic（PPO 时）+ KL penalty +
PPO clipping + 采样 rollout …… 工程上很重。

DPO（Rafailov et al. 2023, NeurIPS Best Paper）的洞察：

    **RLHF 的 KL-constrained 最优解 $\pi^*$ 和 reward $r$ 之间有闭式关系，
    把它代回 Bradley-Terry 偏好模型，RL 目标被重写为一个简单的二分类 loss，
    不需要 reward model，不需要 critic，不需要 PPO——纯监督学习！**

KTO（Ethayarajh et al. 2024, ICML）走得更远：

    **用 prospect theory（前景理论，Kahneman-Tversky 1979）替代 Bradley-Terry，
    只需要"good/bad"二元标签（不需要成对偏好数据）。**

--------------------------------------------------------------------
DPO 推导（4 步，详见 Ch14 §14.2）

**Step 1**（Ch12 已证）：RLHF 的 KL-constrained 最优策略

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(r(x,y)/\beta)$ 是配分函数（与 $\pi$ 无关）。

**Step 2**（反解 reward）：对上式取 log

$$
\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} = \frac{r(x,y)}{\beta} - \log Z(x)
\;\;\Longrightarrow\;\;
r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)
$$

**Step 3**（代入 Bradley-Terry）：$P(y_w \succ y_l | x) = \sigma(r(x,y_w) - r(x,y_l))$
（Ch11 §11.2）。把 Step 2 代入：

$$
P(y_w \succ y_l | x) = \sigma\!\left(
    \beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)}
    - \beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
$$

**注意**：$\beta\log Z(x)$ 在相减中**消掉**了（与 $y$ 无关）——这是 DPO 能去掉
reward model 的关键。

**Step 4**（最大似然）：把 $\pi^*$ 替换为我们想学的 $\pi_\theta$，最大化对数似然
= 最小化：

$$
\boxed{\;\mathcal{L}_{DPO}(\theta) =
-\mathbb{E}_{(x, y_w, y_l)}\!\left[\log\sigma\!\left(
    \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
    - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)\right]\;}
$$

**这就是 DPO 的精髓**——RL 目标被重写为简单的二分类 loss。
- 不需要 reward model（被 $\log\pi/\log\pi_{ref}$ 隐式编码）
- 不需要 critic（没有 advantage）
- 不需要 PPO clipping（直接最大似然，离线数据）
- 不需要 rollout（在静态偏好数据上训）

--------------------------------------------------------------------
KTO（Kahneman-Tversky Optimization）

DPO 用 Bradley-Terry；KTO 用 **prospect theory**：

- Bradley-Terry：建模 $P(y_w \succ y_l)$ —— **需要成对偏好**
- prospect theory：建模单条 response 的 **效用**（loss aversion + diminishing
  sensitivity），只需要"good / bad"二元标签

KTO loss（核心公式见 §14.3）：

$$
\mathcal{L}_{KTO}(\theta) = \mathbb{E}_{y \sim \text{good}}\!\big[\sigma(\beta\cdot\text{KL} - z_0)\big]
                          + w \cdot \mathbb{E}_{y \sim \text{bad}}\!\big[\sigma(\beta\cdot\text{KL} - z_0)\big]
$$

其中 $\text{KL} = \log(\pi_\theta(y|x)/\pi_{ref}(y|x))$，$z_0$ 是一个参考点（reference
point，通常用 KL 的 log-likelihood 均值估计），$w$ 反映 loss aversion（人对损失
比收益更敏感，Kahneman-Tversky 测得 $w \approx 2.25$）。

工程优势：
- 数据更便宜（thumbs up/down vs pairwise comparison）
- 实现和 DPO 几乎一样（同一个 trainer 换个 loss）

--------------------------------------------------------------------
模块组成

- :func:`sequence_log_probs`  —— 算 (prompt, response) 的 $\log\pi(y|x)$（sum over tokens）
- :func:`dpo_loss`           —— DPO loss（核心 1/2）
- :func:`kto_loss`           —— KTO loss（核心 2/2）
- :class:`DPOConfig`         —— 超参 dataclass（β、lr、batch size 等）
- :class:`DPOTrainer`        —— **2 模型**协调器（actor + frozen reference）
- :class:`KTOTrainer`        —— KTO 版本（同 trainer 换 loss）
- :func:`kto_points_to_loss` —— prospect theory value function（教学/可视化用）

设计原则（与 Ch12 / Ch13 一致）：
- **教学优先**：每一步把数学展开，便于读者对照 §14.2 / §14.3 公式
- **不重复造轮子**：复用 :class:`rlenvs.tiny_gpt.TinyGPT` 作 actor / reference backbone；
  评估时复用 :class:`utils.reward_model.RewardModel`
- **关键差异（vs RLHFTrainer / GRPOTrainer）**：
    - **只有 2 个模型**（actor + frozen reference），**没有 reward model 在训练 loop 里**
      （reward model 只用于事后评估 §14.4 / §14.5）
    - **没有 rollout**：在静态偏好数据上训（off-policy to the extreme）
    - **没有 advantage / critic / clipping**：纯监督分类 loss
    - **没有 PPO inner loop**：每个 minibatch 一次梯度步
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. sequence_log_probs: 算 log π(y | x) = sum_t log π(y_t | x, y_<t)
# =============================================================================
def sequence_log_probs(
    model: nn.Module,
    prompt_ids: torch.Tensor,   # [B, T_p]
    response_ids: torch.Tensor, # [B, T_r]
    pad_id: int,
    response_mask: Optional[torch.Tensor] = None,  # [B, T_r], 1 = real token
) -> torch.Tensor:
    r"""算每条 (prompt, response) 的对数似然 $\log\pi(y|x)$（sum over response tokens）。

    数学::

        $\log\pi(y|x) = \sum_{t=0}^{T_r-1} \log\pi(y_t | x, y_{<t})$

    对每个 response token，我们需要：

    1. 把 prompt + response 拼起来喂给 model
    2. 由于 causal mask，position $p$ 的 logits 预测的是 position $p+1$ 的 token
    3. 所以 response token $j$（在 full sequence 中位置 $T_p + j$）的预测 logits
       在位置 $T_p + j - 1$

    **为什么逐样本 forward**（与 RLHFTrainer / GRPOTrainer 同样的考量）：
    batch 内不同样本的 prompt 长度不同，left-pad 会让 response token attend 到
    prompt 的 pad token 上，导致 $\log\pi$ ≠ 真实概率。这里逐样本 forward 真实
    （未 pad）的 prompt + response，保证数学严格。

    Parameters
    ----------
    model : nn.Module
        ``forward(input_ids) -> logits [B, T, V]`` 的语言模型（TinyGPT 或带 backbone 的 wrapper）。
    prompt_ids : LongTensor [B, T_p]
        可能含 pad（左 / 右），用 ``prompt_lens`` 推断真实长度。
    response_ids : LongTensor [B, T_r]
        可能含 pad（右填充），用 ``response_mask`` 或 ``!= pad_id`` 推断真实长度。
    pad_id : int
    response_mask : Optional[FloatTensor [B, T_r]]
        1 = real token, 0 = pad。None 时用 ``response_ids != pad_id`` 推断。

    Returns
    -------
    logp : FloatTensor [B]
        每条样本的 $\sum_t \log\pi(y_t)$（pad 位置不计）。
    """
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    B, T_p = prompt_ids.shape
    T_r = response_ids.size(1)

    if response_mask is None:
        response_mask = (response_ids != pad_id).float()
    # 真实 prompt 长度：从左数连续非 pad 数量（假设右 pad 或无 pad；
    # 如果是 left pad 我们也支持——用 argmax 反向找最后一个非 pad）
    prompt_lens = _compute_prompt_lens(prompt_ids, pad_id)  # [B]

    logp_out = torch.zeros(B, device=prompt_ids.device)
    for b in range(B):
        Lp = int(prompt_lens[b].item())
        Lr = int(response_mask[b].sum().item())
        if Lr == 0:
            continue
        # 真实 prompt + 真实 response（无 padding）
        real_full = torch.cat([
            prompt_ids[b, :Lp],
            response_ids[b, :Lr],
        ]).unsqueeze(0)  # [1, Lp + Lr]
        logits = model(real_full)  # [1, Lp+Lr, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [1, Lp+Lr, V]
        # 抽出每个 response token 的 log π（向量化 gather，一次取全部 Lr 个）
        # response token j (j=0..Lr-1) 在 input 中位于位置 Lp + j
        # 预测它的 logits 在位置 (Lp + j) - 1
        pos = torch.arange(Lp - 1, Lp + Lr - 1, device=logits.device)  # [Lr]
        a_t = response_ids[b, :Lr]                                       # [Lr]
        logp_out[b] = log_probs[0, pos, a_t].sum()
    return logp_out


def _compute_prompt_lens(prompt_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """算每行的真实 prompt 长度（支持 left-pad 和 right-pad / no-pad）。

    规则：找每行**最后一个非 pad** 的位置 +1。如果整行都是 pad（不该发生），返回 0。
    """
    B, T_p = prompt_ids.shape
    lens = torch.zeros(B, dtype=torch.long, device=prompt_ids.device)
    for b in range(B):
        row = prompt_ids[b]
        nonpad = (row != pad_id).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            lens[b] = 0
        else:
            lens[b] = int(nonpad[-1].item()) + 1
    return lens


# =============================================================================
# 2. dpo_loss: DPO loss（核心 1/2）
# =============================================================================
def dpo_loss(
    actor_logp_w: torch.Tensor,    # [B]  log π_θ(y_w | x)
    actor_logp_l: torch.Tensor,    # [B]  log π_θ(y_l | x)
    ref_logp_w: torch.Tensor,      # [B]  log π_ref(y_w | x)
    ref_logp_l: torch.Tensor,      # [B]  log π_ref(y_l | x)
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""DPO loss（§14.2 核心）::

        $\mathcal{L}_{DPO} = -\log\sigma\!\left(
            \beta\big(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
                  - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big)
        \right)$

    记 $\Delta_w = \log\pi_\theta(y_w) - \log\pi_{ref}(y_w)$（actor 相对 ref 在 winner 上
    的 log-prob 增量），$\Delta_l$ 同理。则

        $\mathcal{L}_{DPO} = -\log\sigma\!\big(\beta(\Delta_w - \Delta_l)\big)$
                          $= \text{softplus}\!\big(-\beta(\Delta_w - \Delta_l)\big)$

    我们用 `F.softplus` 数值稳定（不会 exp 上溢），与 :func:`bradley_terry_loss` 同。

    Parameters
    ----------
    actor_logp_w / actor_logp_l : Tensor [B]
        actor 模型在 winner / loser response 上的 $\sum_t\log\pi$（含 batch dim）。
        必须是**有梯度**的 tensor（actor forward 出来的）。
    ref_logp_w / ref_logp_l : Tensor [B]
        reference 模型（冻结）在同样样本上的 $\log\pi$。**无梯度**（detached）。
    beta : float
        KL penalty 系数 β。β 大 → actor 不能离 ref 太远（更保守）；
        β 小 → 可以更激进地拉开 winner/loser。

    Returns
    -------
    loss : scalar Tensor
        DPO loss（mean over batch）。call ``.backward()`` 即可反传到 actor。
    stats : dict
        含标量 metrics（用于日志/可视化）：
            - ``dpo_loss``: 本函数返回的 loss
            - ``chosen_reward``: $\beta\Delta_w$（DPO "隐式 reward" of winner）
            - ``rejected_reward``: $\beta\Delta_l$
            - ``reward_margin``: $\beta(\Delta_w - \Delta_l)$
            - ``reward_accuracy``: reward_margin > 0 的比例（actor 是否在偏好对上"对了"）
            - ``reward_margin_raw``: $\Delta_w - \Delta_l$（不含 β，便于跨 β 比较）

    Notes
    -----
    **"隐式 reward"**（DPO 论文 §4.4）：

        由 Step 2 的反解 $r(x, y) = \beta\log(\pi^*/\pi_{ref}) + \beta\log Z(x)$，
        训练中我们用 $\hat r(x, y; \theta) := \beta\log(\pi_\theta(y|x)/\pi_{ref}(y|x))$
        当 DPO 隐式 reward。

        这个量不需要训练 reward model 就能从 actor 和 reference 算出来，
        可以**事后**评估 actor 学得好不好（reward_margin 应该 > 0 且随训练上升）。
    """
    # Δ_w, Δ_l: log π_θ - log π_ref
    delta_w = actor_logp_w - ref_logp_w  # [B]
    delta_l = actor_logp_l - ref_logp_l

    # DPO 的 logit（输入 sigmoid 的那个值）
    reward_margin = beta * (delta_w - delta_l)  # [B]
    # -log σ(z) = softplus(-z)
    loss = F.softplus(-reward_margin).mean()

    with torch.no_grad():
        chosen_reward = (beta * delta_w).mean().item()
        rejected_reward = (beta * delta_l).mean().item()
        acc = float((reward_margin > 0).float().mean().item())
    stats = {
        "dpo_loss": float(loss.item()),
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "reward_margin": float(reward_margin.mean().item()),
        "reward_accuracy": acc,
        "reward_margin_raw": float((delta_w - delta_l).mean().item()),
    }
    return loss, stats


# =============================================================================
# 3. KTO loss + prospect theory value function（核心 2/2）
# =============================================================================
def prospect_value(
    x: torch.Tensor,
    lambda_aversion: float = 0.5,
    gamma_gain: float = 0.9,
    gamma_loss: float = 0.9,
) -> torch.Tensor:
    r"""Prospect theory 的 **value function**（Kahneman-Tversky 1979）。

    $$
    v(x) = \begin{cases}
        x^{\gamma_+} & x \ge 0 \quad \text{(gain)} \\
        -\lambda \cdot (-x)^{\gamma_-} & x < 0 \quad \text{(loss)}
    \end{cases}
    $$

    两个关键现象（行为经济学经典发现）：

    1. **Diminishing sensitivity**（$\gamma < 1$）：从 0 到 1 的"感觉"比从 100 到 101
       大。所以 v 在原点附近陡，远处平。
    2. **Loss aversion**（$\lambda > 1$）：失去 100 元的痛苦 > 得到 100 元的快乐。
       KT 实验测得 $\lambda \approx 2.25$。

    在 KTO 里：

        response 是 "good"（$\pi_\theta$ 给的 implicit reward > reference point）→ v > 0
        response 是 "bad"  → v < 0（被 $\lambda$ 放大）

    Parameters
    ----------
    x : Tensor 任意 shape
        "reward" 信号（在 KTO 里 = $\beta\log(\pi_\theta/\pi_{ref})$ 相对于 reference point）
    lambda_aversion : float
        损失厌恶系数 $\lambda$。KT 实测 ≈ 2.25；KTO 论文里默认 0.5（因为他们的 x 已经
        是 logit，量级不同）。
    gamma_gain, gamma_loss : float in (0, 1]
        gains / losses 的曲率。

    Returns
    -------
    v : Tensor, same shape as ``x``
    """
    # 分 gains (x>=0) 和 losses (x<0) 用不同公式
    # 注意：当 gamma != 1 且 x 为负时不能直接 x**gamma（NaN），所以分开处理
    pos = x.clamp(min=0.0)
    neg = (-x).clamp(min=0.0)  # -x，但只对 x<0 的部分有效
    v_pos = torch.sign(x).clamp(min=0.0) * pos.pow(gamma_gain)
    v_neg = -lambda_aversion * torch.sign(x).clamp(max=0.0).abs() * neg.pow(gamma_loss)
    return v_pos + v_neg


def kto_points_to_loss(
    points: torch.Tensor,  # [B]
    is_good: torch.Tensor, # [B] in {0, 1}
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    tau: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""简化版 KTO pointwise loss（教学用，对应 §14.3 公式）。

    输入：
        points: $\beta \cdot \log(\pi_\theta(y|x)/\pi_{ref}(y|x))$ —— DPO 隐式 reward
        is_good: 1 = thumbs up, 0 = thumbs down

    KTO 把"good"和"bad"分开建模：
        good response: 想让 points 越大越好（推高 π_θ 对该 response 的概率）
            loss_good = softplus(tau - points)   # 当 points >> tau，loss → 0
        bad response:  想让 points 越小越好
            loss_bad  = softplus(points - tau)   # 当 points << tau，loss → 0

    其中 ``tau`` 是 reference point（"中性" reward 的阈值）。

    完整版 KTO（Kahneman-Tversky）会把 loss_bad 乘上 $\lambda > 1$（loss aversion），
    本函数用 ``undesirable_weight`` 实现。

    最终 loss = desirable_weight * mean(loss_good) + undesirable_weight * mean(loss_bad)
    （good / bad 分别 mean，然后加权求和——避免类别不平衡问题）

    Parameters
    ----------
    points : Tensor [B]
        $\beta\log(\pi_\theta/\pi_{ref})$，每条样本一个。
    is_good : Tensor [B]
        1 = good response, 0 = bad response.
    beta : float
        （未使用，保留为参数对齐 DPO 接口；points 已经乘过 β 了）
    desirable_weight, undesirable_weight : float
        good / bad 的权重（undesirable > desirable 体现 loss aversion）。
    tau : float
        reference point（中性阈值）。

    Returns
    -------
    loss : scalar Tensor
    stats : dict
    """
    good = is_good.float()
    bad = 1.0 - good
    n_good = good.sum().clamp(min=1.0)
    n_bad = bad.sum().clamp(min=1.0)

    # good: softplus(tau - points) → 当 points ↑，loss ↓
    loss_good = F.softplus(tau - points)
    # bad:  softplus(points - tau) → 当 points ↓，loss ↓
    loss_bad = F.softplus(points - tau)

    loss = (
        desirable_weight * (loss_good * good).sum() / n_good
        + undesirable_weight * (loss_bad * bad).sum() / n_bad
    )

    with torch.no_grad():
        # 统计：good 样本的平均 points（应该 > tau）、bad 样本的平均 points（应该 < tau）
        good_pts = (points * good).sum().item() / float(n_good)
        bad_pts = (points * bad).sum().item() / float(n_bad)
        # accuracy: good 样本 points > tau 且 bad 样本 points < tau 的比例
        good_correct = ((points > tau).float() * good).sum().item()
        bad_correct = ((points < tau).float() * bad).sum().item()
        total_correct = good_correct + bad_correct
        acc = total_correct / max(float(n_good + n_bad), 1.0)

    stats = {
        "kto_loss": float(loss.item()),
        "good_points": float(good_pts),
        "bad_points": float(bad_pts),
        "kto_accuracy": float(acc),
        "tau": float(tau),
    }
    return loss, stats


def kto_loss(
    actor_logp: torch.Tensor,   # [B]  log π_θ(y | x)
    ref_logp: torch.Tensor,     # [B]  log π_ref(y | x)
    is_good: torch.Tensor,      # [B]  in {0, 1}
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    tau: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""KTO loss（§14.3 核心）—— 包装 :func:`kto_points_to_loss`。

    Parameters
    ----------
    actor_logp / ref_logp : Tensor [B]
        每条 (prompt, response) 样本的 $\sum_t\log\pi$。
        ``actor_logp`` 必须有梯度，``ref_logp`` 必须 detached。
    is_good : Tensor [B] in {0, 1}
        1 = good response（thumbs up）, 0 = bad response（thumbs down）。
    beta : float
        KL penalty 系数 β（同 DPO）。
    desirable_weight, undesirable_weight : float
        good / bad 的权重（loss aversion：undesirable > desirable）。
    tau : Optional[float]
        reference point。None 时用 ``beta * 0``（即 0，等价于"中性 response 的
        隐式 reward = 0"）。

    Returns
    -------
    loss : scalar Tensor
    stats : dict
    """
    if tau is None:
        tau = 0.0
    points = beta * (actor_logp - ref_logp)
    return kto_points_to_loss(
        points=points,
        is_good=is_good,
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        tau=tau,
    )


# =============================================================================
# 4. DPOConfig: 所有 DPO 超参的 dataclass
# =============================================================================
@dataclass
class DPOConfig:
    r"""DPO / KTO 训练的所有超参。

    与 :class:`utils.rlhf.RLHFConfig` / :class:`utils.grpo.GRPOConfig` 的**核心差异**：

    - **没有 clip_eps / update_epochs / target_kl / entropy_coef**：DPO 是纯监督，
      没有 PPO inner loop。
    - **没有 critic_lr / value_coef / gamma / lam**：DPO 没有 critic。
    - **没有 response_max_len / temperature**：DPO 不做 rollout，直接用静态数据集。
    - **新增 reference_logprob_batch_size**：reference 模型对所有训练数据预计算一次
      $\log\pi_{ref}(y|x)$（因为 ref 冻结，可以缓存），训练时只用 actor。

    Attributes
    ----------
    beta : float
        KL penalty 系数 β（同 RLHF / GRPO）。DPO 论文推荐 0.1-0.5；toy 实验用 0.05-0.5。
    actor_lr : float
        actor 的 learning rate。DPO 通常用比 SFT 更小的 lr（因为 loss 已经是
        监督形式，梯度更"硬"）。
    batch_size : int
        每个 step 的偏好对数（DPO）或样本数（KTO）。
    max_grad_norm : float
        梯度裁剪阈值。
    eval_every : int
        每 ``eval_every`` 步在验证集上评估。
    print_every : int
        每 ``print_every`` 步打印一行日志。
    """

    # DPO / KTO 核心
    beta: float = 0.1

    # Optimizer（注意：只有 actor）
    actor_lr: float = 5e-4

    # Training
    batch_size: int = 32
    max_grad_norm: float = 1.0
    eval_every: int = 50
    print_every: int = 50

    # KTO 专用（DPOTrainer 会忽略）
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0  # 实际使用时设 > 1 体现 loss aversion
    kto_tau: float = 0.0


# =============================================================================
# 5. DPOTrainer: 2 模型协调器（actor + frozen reference）
# =============================================================================
class DPOTrainer:
    r"""协调 **actor + reference** 两个模型跑 DPO（**只有 2 个模型**）。

    与 :class:`utils.rlhf.RLHFTrainer` / :class:`utils.grpo.GRPOTrainer` 的对比：

    ============================  =============================  ==============================
    维度                          RLHFTrainer（Ch12）            DPOTrainer（本章）
    ============================  =============================  ==============================
    模型数                        **4** (actor/critic/reward/ref) **2** (actor/ref)
    reward model 在训练 loop 里？ 需要（每步 forward RM）         **不需要**（被 logπ/logπ_ref 编码）
    Rollout（采 response）        需要                            **不需要**（静态偏好数据）
    Critic / Value function       需要（PPO）                     **不需要**
    Advantage normalization       需要                            **不需要**（纯监督）
    PPO clipping / multi-epoch    需要                            **不需要**
    训练数据                      on-policy rollout               **静态偏好数据集**
    Loss 形式                     policy gradient + clipping      二分类 sigmoid loss
    ============================  =============================  ==============================

    Parameters
    ----------
    actor : TinyGPT-like
        要训练的策略 $\pi_\theta$。``forward(input_ids) -> logits [B, T, V]``。
    reference : TinyGPT-like
        $\pi_{ref}$（冻结的 SFT 模型）。同 actor 接口。
        通常用 ``copy.deepcopy(actor)`` 初始化（保证起点相同）。
    pad_id : int
    cfg : Optional[DPOConfig]
    device : str

    Notes
    -----
    - **reference 始终冻结**（``requires_grad_(False)``、``eval()``）。
    - **没有 reward_model 属性**：reward model 只用于事后评估（在 notebook 里直接
      用 Ch11 的 ``RewardModel``，不属于 trainer 内部状态）。
    """

    def __init__(
        self,
        actor: nn.Module,
        reference: nn.Module,
        pad_id: int,
        cfg: Optional[DPOConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.reference = reference
        self.pad_id = pad_id
        self.cfg = cfg if cfg is not None else DPOConfig()
        self.device = device

        # 冻结 reference（DPO 的核心要求）
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()

        # 只有 actor 的 optimizer（核心：没有 critic_opt / reward_opt）
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.cfg.actor_lr, weight_decay=0.0
        )

        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # 5.1 预计算 reference log π（ref 冻结，缓存一次即可）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def precompute_reference_logps(
        self,
        samples: Sequence[Dict],
        batch_size: int = 64,
    ) -> Dict[str, torch.Tensor]:
        r"""对所有训练样本预计算 reference log prob（$y_w$ 和 $y_l$ 各一份）。

        因为 reference 模型冻结，这两个量在整个 DPO 训练里都是常量，
        预计算一次后续 epoch 反复用，省掉每步两次 forward。

        Returns
        -------
        dict 含 ``logp_w`` [N], ``logp_l`` [N]
        """
        from .reward_model import make_preference_batch
        self.reference.eval()
        N = len(samples)
        logp_w = torch.zeros(N, device=self.device)
        logp_l = torch.zeros(N, device=self.device)
        for start in range(0, N, batch_size):
            batch = samples[start : start + batch_size]
            b = make_preference_batch(batch, pad_id=self.pad_id)
            b = {k: v.to(self.device) for k, v in b.items()}
            logp_w[start : start + len(batch)] = sequence_log_probs(
                self.reference, b["prompt_ids"], b["winner_ids"], self.pad_id,
            )
            logp_l[start : start + len(batch)] = sequence_log_probs(
                self.reference, b["prompt_ids"], b["loser_ids"], self.pad_id,
            )
        return {"logp_w": logp_w, "logp_l": logp_l}

    # ------------------------------------------------------------------
    # 5.2 dpo_update: 单个 batch 的梯度步
    # ------------------------------------------------------------------
    def dpo_update(
        self,
        prompt_ids: torch.Tensor,     # [B, T_p]
        winner_ids: torch.Tensor,     # [B, T_r]
        loser_ids: torch.Tensor,      # [B, T_r]
        ref_logp_w: torch.Tensor,     # [B]  预计算的 log π_ref(y_w)
        ref_logp_l: torch.Tensor,     # [B]  预计算的 log π_ref(y_l)
    ) -> Dict[str, float]:
        """单个 batch 的一次 DPO 梯度步（**只有 actor，没有 critic / RM**）。

        流程::

            1. actor forward (prompt + winner) → log π_θ(y_w)
               actor forward (prompt + loser)  → log π_θ(y_l)
            2. loss = softplus(-β · ((logπ_θ(y_w) - logπ_ref(y_w))
                                    - (logπ_θ(y_l) - logπ_ref(y_l))))
            3. backward + step（只更新 actor）

        Returns
        -------
        dict 含 dpo_loss / chosen_reward / rejected_reward / reward_margin /
        reward_accuracy 等标量 metrics。
        """
        prompt_ids = prompt_ids.to(self.device)
        winner_ids = winner_ids.to(self.device)
        loser_ids = loser_ids.to(self.device)
        ref_logp_w = ref_logp_w.to(self.device).detach()
        ref_logp_l = ref_logp_l.to(self.device).detach()

        # actor forward（两次，winner 和 loser 各一次）
        actor_logp_w = sequence_log_probs(
            self.actor, prompt_ids, winner_ids, self.pad_id,
        )
        actor_logp_l = sequence_log_probs(
            self.actor, prompt_ids, loser_ids, self.pad_id,
        )

        loss, stats = dpo_loss(
            actor_logp_w, actor_logp_l,
            ref_logp_w, ref_logp_l,
            beta=self.cfg.beta,
        )

        self.actor_opt.zero_grad()
        loss.backward()
        actor_gnorm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.cfg.max_grad_norm
        ).item()
        self.actor_opt.step()
        stats["grad_norm"] = float(actor_gnorm)
        return stats

    # ------------------------------------------------------------------
    # 5.3 train: 完整训练 loop（在静态偏好数据上）
    # ------------------------------------------------------------------
    def train(
        self,
        train_samples: Sequence[Dict],
        n_iters: int = 200,
        val_samples: Optional[Sequence[Dict]] = None,
        val_reward_model: Optional[nn.Module] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """完整 DPO 训练 loop。

        Parameters
        ----------
        train_samples : Sequence[Dict]
            静态偏好数据集（来自 :func:`generate_preference_data`）。
            每个 dict 含 ``prompt_ids`` / ``winner_ids`` / ``loser_ids``。
        n_iters : int
            多少个 gradient step。
        val_samples : Optional[Sequence[Dict]]
            验证偏好数据（用于 DPO 自身的 reward_accuracy）。
        val_reward_model : Optional[nn.Module]
            Ch11 的 RewardModel，用于**事后**评估 actor 训练效果
            （mean reward on val set）。**不属于 DPO loop**，只用于验证。
        verbose : bool

        Returns
        -------
        history : List[Dict[str, float]]
            每个 step（或 eval 点）的 metrics dict。
        """
        from .reward_model import make_preference_batch

        cfg = self.cfg
        rng = random.Random(0)

        # 1) 预计算 reference log π（只算一次，所有 epoch 复用）
        if verbose:
            print(f"[DPO] 预计算 reference log π on {len(train_samples)} samples...")
        ref_logps = self.precompute_reference_logps(train_samples)
        ref_logp_w_all = ref_logps["logp_w"]
        ref_logp_l_all = ref_logps["logp_l"]

        # 预先把所有 prompt / winner / loser batch 起来（按 batch 切片用）
        # 为了支持变长，我们每个 step 重新做 batch（小数据集 OK）
        N = len(train_samples)

        t0 = _time_now()
        last_eval = 0
        for it in range(n_iters):
            # 抽一个 mini-batch 的样本 index
            idx = rng.sample(range(N), min(cfg.batch_size, N))
            batch_samples = [train_samples[i] for i in idx]
            b = make_preference_batch(batch_samples, pad_id=self.pad_id)
            b = {k: v.to(self.device) for k, v in b.items()}
            ref_w = ref_logp_w_all[torch.tensor(idx, device=self.device)]
            ref_l = ref_logp_l_all[torch.tensor(idx, device=self.device)]

            # DPO 梯度步
            stats = self.dpo_update(
                b["prompt_ids"], b["winner_ids"], b["loser_ids"], ref_w, ref_l,
            )
            stats["step"] = float(it)
            self.history.append(stats)

            if verbose and (it % cfg.print_every == 0 or it == n_iters - 1):
                elapsed = _time_now() - t0
                print(
                    f"iter {it:>4} | dpo_loss={stats['dpo_loss']:.4f} | "
                    f"chosen_r={stats['chosen_reward']:+.3f} | "
                    f"rejected_r={stats['rejected_reward']:+.3f} | "
                    f"margin={stats['reward_margin']:+.3f} | "
                    f"acc={stats['reward_accuracy']:.3f} | "
                    f"|g|={stats['grad_norm']:.3f} | "
                    f"({elapsed:.1f}s)"
                )

            # 定期 eval
            if (it % cfg.eval_every == 0 or it == n_iters - 1) and val_samples is not None:
                eval_stats = self.evaluate(val_samples, val_reward_model)
                self.history[-1].update(eval_stats)
                last_eval = it

        return self.history

    # ------------------------------------------------------------------
    # 5.4 evaluate: 在验证集上算 DPO + RM-based 指标
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        val_samples: Sequence[Dict],
        val_reward_model: Optional[nn.Module] = None,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """在验证偏好集上评估。

        返回：
        - ``val_dpo_acc``: DPO 自身的 reward_accuracy（margin > 0 比例）
        - ``val_dpo_margin``: 平均 reward margin
        - ``val_rm_mean_reward_w`` / ``val_rm_mean_reward_l``: 用 Ch11 RM 评估 actor
          生成的隐式 reward（如果传了 val_reward_model）

        Parameters
        ----------
        val_samples : Sequence[Dict]
            偏好验证集。
        val_reward_model : Optional[nn.Module]
            Ch11 的 RewardModel（可选）。如果传了，会算 RM 给 actor 当前 winner/loser
            的 reward（评估 DPO 训练后 actor 是否真的"在 RM 眼中变得更好"）。
        """
        from .reward_model import make_preference_batch

        self.actor.eval()
        N = len(val_samples)
        margins = []
        rm_w, rm_l = [], []
        for start in range(0, N, batch_size):
            batch = val_samples[start : start + batch_size]
            b = make_preference_batch(batch, pad_id=self.pad_id)
            b = {k: v.to(self.device) for k, v in b.items()}

            actor_w = sequence_log_probs(
                self.actor, b["prompt_ids"], b["winner_ids"], self.pad_id,
            )
            actor_l = sequence_log_probs(
                self.actor, b["prompt_ids"], b["loser_ids"], self.pad_id,
            )
            ref_w = sequence_log_probs(
                self.reference, b["prompt_ids"], b["winner_ids"], self.pad_id,
            )
            ref_l = sequence_log_probs(
                self.reference, b["prompt_ids"], b["loser_ids"], self.pad_id,
            )
            margin = self.cfg.beta * ((actor_w - ref_w) - (actor_l - ref_l))
            margins.append(margin.cpu())

            if val_reward_model is not None:
                rw = val_reward_model(b["prompt_ids"], b["winner_ids"])
                rl = val_reward_model(b["prompt_ids"], b["loser_ids"])
                rm_w.append(rw.cpu())
                rm_l.append(rl.cpu())

        margin_all = torch.cat(margins)
        out = {
            "val_dpo_acc": float((margin_all > 0).float().mean().item()),
            "val_dpo_margin": float(margin_all.mean().item()),
        }
        if val_reward_model is not None and rm_w:
            rw_all = torch.cat(rm_w)
            rl_all = torch.cat(rm_l)
            out["val_rm_mean_reward_w"] = float(rw_all.mean().item())
            out["val_rm_mean_reward_l"] = float(rl_all.mean().item())
            out["val_rm_margin"] = float((rw_all - rl_all).mean().item())
            out["val_rm_acc"] = float(((rw_all - rl_all) > 0).float().mean().item())
        self.actor.train()
        return out


# =============================================================================
# 6. KTOTrainer: KTO 版本（同 trainer 换 loss + 支持 pointwise 数据）
# =============================================================================
class KTOTrainer:
    r"""协调 **actor + reference** 两个模型跑 KTO（pointwise，good/bad 标签）。

    与 :class:`DPOTrainer` 的**核心差异**：
    - DPO 用**成对**偏好数据 (winner, loser)
    - KTO 用**单条**数据 + good/bad 二元标签

    本 trainer 把同一份成对偏好数据**拆**成单条：winner 标 good，loser 标 bad。
    这样便于在同一数据集上对比 DPO / KTO（不用单独造 pointwise 数据）。

    Parameters
    ----------
    actor, reference, pad_id, cfg, device
        同 DPOTrainer。
    """

    def __init__(
        self,
        actor: nn.Module,
        reference: nn.Module,
        pad_id: int,
        cfg: Optional[DPOConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.reference = reference
        self.pad_id = pad_id
        self.cfg = cfg if cfg is not None else DPOConfig()
        self.device = device

        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()

        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.cfg.actor_lr, weight_decay=0.0
        )
        self.history: List[Dict[str, float]] = []

    @torch.no_grad()
    def precompute_reference_logps_unpacked(
        self,
        samples: Sequence[Dict],
        batch_size: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """把成对偏好数据拆成 pointwise（每个 winner/loser 各一条），算 ref log π。

        Returns
        -------
        dict 含：
            - ``logp`` [2N]  ref log π for each pointwise sample
            - ``is_good`` [2N]  1 = good, 0 = bad
            - ``prompt_ids`` list of [T_p]
            - ``response_ids`` list of [T_r]
        """
        from .reward_model import pad_to_length

        self.reference.eval()
        prompts: List[torch.Tensor] = []
        responses: List[torch.Tensor] = []
        is_good: List[int] = []
        for s in samples:
            prompts.append(s["prompt_ids"])
            responses.append(s["winner_ids"])
            is_good.append(1)
            prompts.append(s["prompt_ids"])
            responses.append(s["loser_ids"])
            is_good.append(0)

        N = len(prompts)
        logp = torch.zeros(N, device=self.device)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            p_batch = pad_to_length(prompts[start:end], self.pad_id).to(self.device)
            r_batch = pad_to_length(responses[start:end], self.pad_id).to(self.device)
            logp[start:end] = sequence_log_probs(
                self.reference, p_batch, r_batch, self.pad_id,
            )
        return {
            "logp": logp,
            "is_good": torch.tensor(is_good, dtype=torch.float32, device=self.device),
            "prompts": prompts,
            "responses": responses,
        }

    def kto_update(
        self,
        prompt_ids: torch.Tensor,    # [B, T_p]
        response_ids: torch.Tensor,  # [B, T_r]
        is_good: torch.Tensor,       # [B]
        ref_logp: torch.Tensor,      # [B]
    ) -> Dict[str, float]:
        """单个 batch 的 KTO 梯度步。"""
        prompt_ids = prompt_ids.to(self.device)
        response_ids = response_ids.to(self.device)
        is_good = is_good.to(self.device)
        ref_logp = ref_logp.to(self.device).detach()

        actor_logp = sequence_log_probs(
            self.actor, prompt_ids, response_ids, self.pad_id,
        )
        loss, stats = kto_loss(
            actor_logp, ref_logp, is_good,
            beta=self.cfg.beta,
            desirable_weight=self.cfg.kto_desirable_weight,
            undesirable_weight=self.cfg.kto_undesirable_weight,
            tau=self.cfg.kto_tau,
        )

        self.actor_opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.cfg.max_grad_norm
        ).item()
        self.actor_opt.step()
        stats["grad_norm"] = float(gnorm)
        return stats

    def train(
        self,
        train_samples: Sequence[Dict],
        n_iters: int = 200,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """KTO 训练 loop。"""
        from .reward_model import pad_to_length

        cfg = self.cfg
        rng = random.Random(0)

        if verbose:
            print(f"[KTO] 预计算 reference log π on {len(train_samples)} pairs "
                  f"(= {len(train_samples)*2} pointwise samples)...")
        ref = self.precompute_reference_logps_unpacked(train_samples)
        ref_logp_all = ref["logp"]
        is_good_all = ref["is_good"]
        prompts_all = ref["prompts"]
        responses_all = ref["responses"]
        N = len(prompts_all)

        t0 = _time_now()
        for it in range(n_iters):
            idx = rng.sample(range(N), min(cfg.batch_size, N))
            p_batch = pad_to_length([prompts_all[i] for i in idx], self.pad_id).to(self.device)
            r_batch = pad_to_length([responses_all[i] for i in idx], self.pad_id).to(self.device)
            ig = is_good_all[torch.tensor(idx, device=self.device)]
            rl = ref_logp_all[torch.tensor(idx, device=self.device)]

            stats = self.kto_update(p_batch, r_batch, ig, rl)
            stats["step"] = float(it)
            self.history.append(stats)

            if verbose and (it % cfg.print_every == 0 or it == n_iters - 1):
                elapsed = _time_now() - t0
                print(
                    f"iter {it:>4} | kto_loss={stats['kto_loss']:.4f} | "
                    f"good_pts={stats['good_points']:+.3f} | "
                    f"bad_pts={stats['bad_points']:+.3f} | "
                    f"acc={stats['kto_accuracy']:.3f} | "
                    f"|g|={stats['grad_norm']:.3f} | "
                    f"({elapsed:.1f}s)"
                )
        return self.history


# =============================================================================
# 7. 工具
# =============================================================================
def _time_now() -> float:
    import time
    return time.time()


__all__ = [
    "sequence_log_probs",
    "dpo_loss",
    "kto_loss",
    "kto_points_to_loss",
    "prospect_value",
    "DPOConfig",
    "DPOTrainer",
    "KTOTrainer",
]
