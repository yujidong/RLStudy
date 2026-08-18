r"""GRPO（Group Relative Policy Optimization）—— Ch13 的核心算法。

本章是整个 RLStudy 项目的**终点**。它兑现的最重要的承诺是：

    **"GRPO 去掉了 value function。"**（CH05 §5.10.3）

--------------------------------------------------------------------
为什么可以去掉 critic？（详见 Ch13 §13.3 完整推导）

PPO（Ch09 / Ch12 RLHF）的 advantage 用 critic $V_\phi$ 作 baseline：
    $\hat A_t = \delta_t + \gamma\lambda \, \hat A_{t+1}$，其中
    $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$。

critic 在 LLM 上有两个工程痛点（§13.1）：
    1. **参数量翻倍**：actor + critic 各一个 LLM。70B actor + 70B critic =
       140B 训练参数，显存爆炸。
    2. **token-level value 难学**：critic 要在每个 prefix $s_t = (x, y_{<t})$
       上学准 $V_\phi(s_t)$，但 LLM 的"reward 稀疏 + 序列长"让 value 学习
       极不稳定（PPO 训练 LLM 时 critic loss 经常不收敛）。

GRPO 的洞察（DeepSeek 2024）：
    **同一个 prompt $x$ 采 $G$ 个 response，用组内相对 reward 当 advantage，
    不需要 critic。**

    $\hat A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}$，其中
    $\bar r = \frac{1}{G}\sum_i r_i$，$\sigma_r = \sqrt{\frac{1}{G}\sum_i (r_i - \bar r)^2}$。

    这是 **REINFORCE with group baseline** 的特例：
    - $\bar r$ 是 $\mathbb{E}_{y \sim \pi}[r(x, y)] \approx V^\pi(x)$ 的
      Monte Carlo 无偏估计（大数律）
    - $\hat A_i = r_i - \bar r \approx Q^\pi(x, y_i) - V^\pi(x) = A^\pi(x, y_i)$
    - 除 $\sigma_r$ 是 advantage normalization（PPO 工程标配），控制梯度量级

    每条 response $y_i$ 内所有 token 共享同一个 $\hat A_i$（sequence-level
    credit assignment），但 PPO clipping 仍然是 per-token 的。

--------------------------------------------------------------------
模块组成

- :func:`compute_group_advantages` —— 纯函数：G 个 reward → G 个标准化 advantage
- :class:`GRPOConfig`              —— 超参 dataclass（G、β、clip ε、K epochs、
                                       KL target 等，**没有 critic_lr / value_coef**）
- :class:`GRPOTrainer`             —— **3 模型**协调器（actor / reward / reference，
                                       **没有 critic**）
    * :meth:`rollout_group`         —— 对每个 prompt 采 G 个 response
    * :meth:`compute_token_rewards` —— reward model + per-token KL penalty
    * :meth:`grpo_update`           —— PPO 更新（**只更新 actor**）
    * :meth:`train`                 —— 完整训练 loop

设计原则（与 Ch12 RLHFTrainer 一致）：
    - **教学优先**：每一步把数学展开，便于读者对照 §13.3-13.4 公式
    - **不重复造轮子**：复用 :class:`rlenvs.tiny_gpt.TinyGPT` 作 actor / reference
      backbone；复用 :class:`utils.reward_model.RewardModel`；复用
      :func:`utils.ppo.compute_clip_objective`
    - **关键差异（vs RLHFTrainer）**：
        - 没有 ValueHead，没有 critic_opt，没有 value_coef
        - advantage 来自 group baseline（dataclass 级别没有 gamma/lam）
        - PPO inner loop 只更新 actor
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppo import compute_clip_objective


# =============================================================================
# 1. compute_group_advantages: G 个 reward → G 个标准化 advantage
# =============================================================================
def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Group baseline advantage：$\hat A_i = (r_i - \bar r) / (\sigma_r + \epsilon)$。

    这是 GRPO 的**数学灵魂**（§13.3）。

    Parameters
    ----------
    rewards : Tensor, shape [G]（或 [B, G]，按最后一维 normalize）
        每个 response 的 scalar reward。
    eps : float
        数值稳定项（避免 $\sigma_r = 0$ 时除零）。

    Returns
    -------
    advantages : Tensor, same shape as ``rewards``
        标准化后的 advantage（均值 0、标准差 ≈ 1）。

    Notes
    -----
    **推导要点**（§13.3 完整版）：

        $V^\pi(x) = \mathbb{E}_{y \sim \pi}[Q^\pi(x, y)]
                  \approx \mathbb{E}_{y \sim \pi}[r(x, y)]$  （RM 学得好时）

        group 均值 $\bar r = \frac{1}{G}\sum_i r_i$ 是
        $\mathbb{E}_{y \sim \pi}[r(x, y)]$ 的 **Monte Carlo 无偏估计**（大数律）。

        所以 $\hat A_i = r_i - \bar r \approx Q^\pi(x, y_i) - V^\pi(x) = A^\pi(x, y_i)$。

    **为什么除 $\sigma_r$**（§13.3.5）：
        - 不同 prompt 的 reward scale 可能差很大（一个 prompt 全是 +2，另一个全是 +0.1）
        - 不除 $\sigma_r$ → 不同 prompt 的梯度量级差很多 → 训练不稳
        - 除 $\sigma_r$ = advantage normalization（PPO 工程标配）
        - 注意：当 G 很小时 $\sigma_r$ 噪声大，加 $\epsilon$ 防爆
    """
    if rewards.dim() == 0:
        raise ValueError("rewards 至少要 1D（一组 G 个 reward）")
    # 沿最后一维（group 维）算 mean / std
    mean = rewards.mean(dim=-1, keepdim=True)
    # unbiased=False 与样本方差一致（除以 G 而非 G-1）；GRPO 原文用样本标准差
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    advantages = (rewards - mean) / (std + eps)
    return advantages


# =============================================================================
# 2. GRPOConfig: 所有 GRPO 超参的 dataclass
# =============================================================================
@dataclass
class GRPOConfig:
    r"""GRPO 训练的所有超参。

    与 :class:`utils.rlhf.RLHFConfig` 的差异（**核心**）：

    - **没有 ``gamma`` / ``lam``**：GRPO 不用 GAE，advantage 是 sequence-level
      的 group baseline（$\hat A_i$ 不沿时间衰减，整条 response 共享一个值）
    - **没有 ``critic_lr`` / ``value_coef``**：没有 critic 要训

    Attributes
    ----------
    group_size : int
        G，每个 prompt 采多少个 response（GRPO 的核心参数）。G 越大 baseline 越准，
        但开销线性增长。DeepSeek-R1 用 64；教学示例用 6-12。
    beta : float
        KL penalty 系数 β（同 RLHF）。β 大 → 更保守。
    clip_eps : float
        PPO clip ε（同 RLHF，per-token）。
    update_epochs : int
        同一批 rollout 数据反复 K epochs（PPO 多 epoch 数据重用）。
    inner_minibatch_size : int
        每个 inner epoch 切成多大的 mini-batch（以 "response" 为单位）。
    entropy_coef : float
        entropy bonus 系数（鼓励探索）。
    max_grad_norm : float
        梯度裁剪阈值。
    target_kl : Optional[float]
        每个 inner epoch 后估 KL(actor_old || actor_new)，超 1.5×target_kl 就 early stop。
    advantage_eps : float
        group advantage 标准化时分母加的小常数，避免 $\sigma_r=0$ 时除零。
    response_max_len : int
        每条 response 最多生成多少 token。
    temperature : float
        rollout 采样温度（>1 更随机，<1 更确定）。**GRPO 对 temperature 敏感**：
        温度太低 → G 个 response 都差不多 → $\sigma_r \approx 0$ → advantage 信号弱。
    top_k : Optional[int]
        rollout 时 top-k 截断（None = 不截断）。
    actor_lr : float
        actor（唯一训练的模型）的 learning rate。
    print_every : int
        每 ``print_every`` 个 outer iter 打印一行训练日志。
    """

    # GRPO 核心
    group_size: int = 8
    advantage_eps: float = 1e-8

    # KL penalty (同 RLHF)
    beta: float = 0.05

    # PPO
    clip_eps: float = 0.2
    update_epochs: int = 4
    inner_minibatch_size: int = 8
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.04

    # Rollout
    response_max_len: int = 16
    temperature: float = 1.0
    top_k: Optional[int] = None

    # Optimizer（注意：只有 actor）
    actor_lr: float = 1e-5

    # Reporting
    print_every: int = 1


# =============================================================================
# 3. GRPOTrainer: 3 模型协调器（actor / reward / reference，没有 critic）
# =============================================================================
class GRPOTrainer:
    r"""协调 **actor / reward / reference** 三个模型跑 GRPO（**没有 critic**）。

    与 :class:`utils.rlhf.RLHFTrainer` 的对比：

    ============================  =============================  ==============================
    维度                          RLHFTrainer（Ch12）            GRPOTrainer（本章）
    ============================  =============================  ==============================
    模型数                        **4** (actor/critic/reward/ref) **3** (actor/reward/ref)
    Advantage 来源                critic $V_\phi$ + GAE          group baseline $\hat A_i$
    Advantage shape               [B, T_r] (per-token)           [B, T_r] (per-token，但
                                                                  每个 response 内所有 token
                                                                  共享同一个 $\hat A_i$)
    gamma / lam                   需要（GAE 用）                 **不需要**
    Critic optimizer              需要                            **不需要**
    PPO clipping                  per-token                      per-token（同）
    KL penalty                    per-token                       per-token（同）
    ============================  =============================  ==============================

    Parameters
    ----------
    actor : TinyGPT-like
        要训练的策略 $\pi_\theta$。``forward(input_ids) -> logits [B, T, V]``。
    reward_model : RewardModel
        $r(x, y)$。``forward(prompt_ids, response_ids) -> reward [B]``。
    reference : TinyGPT-like
        $\pi_{ref}$（冻结的 SFT 模型）。同 actor 接口。
    pad_id : int
        padding token id。
    cfg : Optional[GRPOConfig]
        超参。
    device : str
        ``'cpu'`` 或 ``'cuda'``。

    Notes
    -----
    - **reference 始终冻结**（``requires_grad_(False)``、``eval()``）。
    - **reward_model 在 GRPO 阶段也冻结**（InstructGPT / DeepSeek-R1 配方：
      RM 训好后不再动；DeepSeek-R1 reasoning 阶段甚至完全用规则型 reward）。
    - **只有 actor 接受梯度**。
    """

    def __init__(
        self,
        actor: nn.Module,
        reward_model: nn.Module,
        reference: nn.Module,
        pad_id: int,
        cfg: Optional[GRPOConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.reward_model = reward_model
        self.reference = reference
        self.pad_id = pad_id
        self.cfg = cfg if cfg is not None else GRPOConfig()
        self.device = device

        # 冻结 reference 和 reward_model（与 RLHF 配方一致）
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()
        for p in self.reward_model.parameters():
            p.requires_grad_(False)
        self.reward_model.eval()

        # **只有 actor 的 optimizer**（核心：没有 critic_opt）
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.cfg.actor_lr, weight_decay=0.0
        )

        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # 3.1 rollout_group: 对每个 prompt 采 G 个 response
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout_group(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """对每个 prompt 采 ``cfg.group_size`` 个 response（GRPO 的核心采样）。

        与 :meth:`RLHFTrainer.rollout_responses` 的差异：
        - 同一个 prompt 采 **G 个**（不是 1 个），保证 group 内有 reward 差异
        - **不返回 ``values_old``**（没有 critic）

        Returns
        -------
        dict 含:
            - ``prompts``      : LongTensor [N, T_p]
                其中 ``N = len(prompt_ids_list) * cfg.group_size``
                （每个 prompt 重复 G 次）
            - ``responses``    : LongTensor [N, T_r]
            - ``prompt_lens``  : LongTensor [N]
            - ``response_lens``: LongTensor [N]
            - ``group_ids``    : LongTensor [N]
                每个样本属于哪个 prompt（0..len(prompt_ids_list)-1）。
                用于算 group advantage 时按 group 聚合。
            - ``log_probs_old``: FloatTensor [N, T_r]  log π_actor at rollout
            - ``log_probs_ref``: FloatTensor [N, T_r]  log π_ref    (frozen)
            - ``response_mask``: FloatTensor [N, T_r]
        """
        # rollout 期间切 eval（关 dropout 等随机性），结束后恢复训练模式；
        # reference 是冻结模型，保持 eval 即可
        _actor_was_training = self.actor.training
        self.actor.eval()
        self.reference.eval()

        cfg = self.cfg
        device = self.device
        G = cfg.group_size

        all_prompts: List[torch.Tensor] = []
        all_responses: List[torch.Tensor] = []
        all_group_ids: List[int] = []

        # 对每个 prompt 采 G 次（每次独立采样 → group 内有差异）
        for gid, p in enumerate(prompt_ids_list):
            p_t = p.to(device).long()
            if p_t.dim() == 0:
                continue
            p_b = p_t.unsqueeze(0)  # [1, T_p]
            for _ in range(G):
                full = self._sample_response(
                    self.actor, p_b, cfg.response_max_len,
                    temperature=cfg.temperature, top_k=cfg.top_k,
                    forbidden_ids={self.pad_id},
                )
                resp = full[:, p_t.size(0):].squeeze(0)  # [T_r]
                all_prompts.append(p_t)
                all_responses.append(resp)
                all_group_ids.append(gid)

        N = len(all_prompts)
        if N == 0:
            raise ValueError("rollout_group: 没有 prompt")

        T_p = max(int(p.size(0)) for p in all_prompts)
        T_r = max(int(r.size(0)) for r in all_responses)

        prompts = torch.full((N, T_p), self.pad_id, dtype=torch.long, device=device)
        responses = torch.full((N, T_r), self.pad_id, dtype=torch.long, device=device)
        prompt_lens = torch.zeros(N, dtype=torch.long, device=device)
        response_lens = torch.zeros(N, dtype=torch.long, device=device)
        group_ids = torch.tensor(all_group_ids, dtype=torch.long, device=device)

        for i, (p, r) in enumerate(zip(all_prompts, all_responses)):
            Lp, Lr = int(p.size(0)), int(r.size(0))
            prompts[i, :Lp] = p
            responses[i, :Lr] = r
            prompt_lens[i] = Lp
            response_lens[i] = Lr

        response_mask = (responses != self.pad_id).float()  # [N, T_r]

        # 算每个 response token 的 old log π / ref log π
        # （与 RLHFTrainer 同样的逐样本 forward 策略：不能用 padded batch，
        # 否则 pad 会污染 attention 导致 log_prob ≠ 真实采样概率）
        log_probs_old = torch.zeros(N, T_r, device=device)
        log_probs_ref_resp = torch.zeros(N, T_r, device=device)
        for i in range(N):
            Lp = int(prompt_lens[i].item())
            Lr = int(response_lens[i].item())
            if Lr == 0:
                continue
            real_full = torch.cat([
                prompts[i, :Lp],
                responses[i, :Lr],
            ]).unsqueeze(0)  # [1, Lp + Lr]
            actor_logits = self.actor(real_full)
            ref_logits = self.reference(real_full)

            log_probs_actor = F.log_softmax(actor_logits, dim=-1)
            log_probs_ref = F.log_softmax(ref_logits, dim=-1)

            # 向量化 gather：一次取出全部 Lr 个 response token 的 log π
            pos = torch.arange(Lp - 1, Lp + Lr - 1, device=device)  # [Lr]
            a_t = responses[i, :Lr]                                  # [Lr]
            log_probs_old[i, :Lr] = log_probs_actor[0, pos, a_t]
            log_probs_ref_resp[i, :Lr] = log_probs_ref[0, pos, a_t]

        # rollout 结束，恢复进入本函数前的训练模式（防止 eval 泄漏到后续更新）
        self.actor.train(_actor_was_training)
        return dict(
            prompts=prompts,
            responses=responses,
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            group_ids=group_ids,
            log_probs_old=log_probs_old,
            log_probs_ref=log_probs_ref_resp,
            response_mask=response_mask,
        )

    @staticmethod
    def _sample_response(
        model: nn.Module,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        forbidden_ids: Optional[set] = None,
    ) -> torch.Tensor:
        """自回归采样（与 RLHFTrainer._sample_response 一致，便于对比）。"""
        out = prompt
        for _ in range(max_new_tokens):
            logits = model(out)[:, -1, :]
            logits = logits / max(temperature, 1e-5)
            if forbidden_ids:
                for tid in forbidden_ids:
                    logits[:, tid] = float("-inf")
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, [-1]]
                logits = torch.where(
                    logits < thresh, torch.full_like(logits, float("-inf")), logits
                )
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)
        return out

    # ------------------------------------------------------------------
    # 3.2 compute_token_rewards: reward model + per-token KL penalty
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_token_rewards(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        response_lens: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """算每个 response token 的 reward（与 RLHF 同样的 per-token KL penalty）。

        数学（与 §12.3.5 / §13.4 完全一致）::

            r_t = -β · log( π(a_t|s_t) / π_ref(a_t|s_t) )     t = 0, ..., T-2
            r_{T-1} = r(x, y) - β · log( π(a_{T-1}|s_{T-1}) / π_ref(...) )

        GRPO **保留** per-token KL penalty（和 RLHF 一样），但 advantage 算法不同
        （见 :meth:`_compute_group_advantages`）。

        Returns
        -------
        token_rewards : Tensor [N, T_r]
        rm_rewards    : Tensor [N]         reward model scalar（每条 response）
        kl_per_token  : Tensor [N, T_r]    log(π/π_ref)
        """
        cfg = self.cfg
        N, T_r = responses.shape

        # KL per token: log(π/π_ref) = log π - log π_ref
        kl_per_token = log_probs_old - log_probs_ref  # [N, T_r]

        # reward model 给整段一个标量
        rm_rewards = self.reward_model(prompts, responses)  # [N]

        # 组装 per-token reward：所有位置先加 KL penalty
        token_rewards = -cfg.beta * kl_per_token

        # 在每个 response 的最后一个真实 token 位置加 r(x, y)
        last_idx = response_lens.long() - 1  # [N]
        for i in range(N):
            li = int(last_idx[i].item())
            if li >= 0:
                token_rewards[i, li] = token_rewards[i, li] + rm_rewards[i]

        # pad 位置清零
        token_rewards = token_rewards * (responses != self.pad_id).float()

        return token_rewards, rm_rewards, kl_per_token

    # ------------------------------------------------------------------
    # 3.3 group advantage: 把 reward 按 group 聚合 → group baseline
    # ------------------------------------------------------------------
    def _compute_group_advantages(
        self,
        token_rewards: torch.Tensor,  # [N, T_r]
        group_ids: torch.Tensor,      # [N]
        response_mask: torch.Tensor,  # [N, T_r]
    ) -> torch.Tensor:
        r"""按 group 算 advantage：$\hat A_i = (R_i - \bar R) / \sigma_R$。

        每条 response $i$ 的 **return** $R_i$ = sum of token_rewards（含 KL penalty
        和最后的 RM scalar）。然后**同一 group 内**做标准化。

        把 sequence-level 的 $\hat A_i$ **广播到该 response 的每个 token**
        （GRPO 论文做法：所有 token 共享一个 sequence-level credit，PPO clipping
        仍 per-token）。

        Returns
        -------
        advantages : Tensor [N, T_r]
            ``advantages[i, t] = \hat A_{group(i)}` 对所有真实 token t，
            pad 位置为 0。
        """
        # 1) 每条 response 的 return（token_rewards 已含 pad=0，sum 即可）
        returns = (token_rewards * response_mask).sum(dim=-1)  # [N]

        # 2) 按 group 算 mean / std → group baseline advantage
        #    group_ids 取值 0..K-1（K = 不同 prompt 数）
        device = token_rewards.device
        N = returns.size(0)
        adv_per_response = torch.zeros(N, device=device)
        unique_groups = torch.unique(group_ids)
        for g in unique_groups:
            mask = (group_ids == g)
            r_group = returns[mask]  # [G]
            a_group = compute_group_advantages(r_group, eps=self.cfg.advantage_eps)
            adv_per_response[mask] = a_group

        # 3) 广播：response i 的每个真实 token 都用同一个 \hat A_i
        advantages = adv_per_response.unsqueeze(-1) * response_mask  # [N, T_r]
        return advantages

    # ------------------------------------------------------------------
    # 3.4 grpo_update: PPO 更新（只更新 actor，没有 critic step）
    # ------------------------------------------------------------------
    def grpo_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """对一批 rollout 数据做 PPO 多-epoch 更新（**只更新 actor**）。

        与 :meth:`RLHFTrainer.rlhf_update` 的差异：
        - advantage 用 group baseline（不是 GAE）
        - 没有 critic loss / critic step
        - 没有 advantage normalization（已经在 group 内标准化过了）

        Returns
        -------
        dict 含标量 metrics（actor_loss / entropy / approx_kl / clip_fraction /
        mean_reward / mean_kl_to_ref / n_epochs_done）。
        **注意：没有 ``critic_loss``**（兑现"去掉 value function"承诺）。
        """
        cfg = self.cfg
        device = self.device

        prompts = rollout["prompts"]
        responses = rollout["responses"]
        prompt_lens = rollout["prompt_lens"]
        response_lens = rollout["response_lens"]
        response_mask = rollout["response_mask"]
        log_probs_old = rollout["log_probs_old"].detach()
        log_probs_ref = rollout["log_probs_ref"].detach()
        group_ids = rollout["group_ids"]

        # 1) token rewards（per-token KL penalty + RM scalar at last）
        token_rewards, rm_rewards, kl_per_token = self.compute_token_rewards(
            prompts, responses, response_lens, log_probs_old, log_probs_ref
        )

        # 2) group baseline advantage（不需要 critic！）
        advantages = self._compute_group_advantages(
            token_rewards, group_ids, response_mask
        )
        # 不再做 batch-level advantage normalization——已经在 group 内标准化过

        N = prompts.size(0)

        # PPO inner loop
        acc = dict(
            actor_loss=0.0, entropy=0.0,
            approx_kl=0.0, clip_fraction=0.0, grad_norm=0.0,
        )
        n_updates_total = 0
        epochs_done = 0
        early_stopped = False

        for epoch in range(cfg.update_epochs):
            perm = torch.randperm(N, device=device)
            mb_size = min(cfg.inner_minibatch_size, N)
            epoch_kl_sum = 0.0
            epoch_mb_count = 0

            for start in range(0, N, mb_size):
                idx = perm[start:start + mb_size]
                stats = self._ppo_step(
                    prompts=prompts[idx],
                    responses=responses[idx],
                    prompt_lens=prompt_lens[idx],
                    log_probs_old_mb=log_probs_old[idx],
                    advantages_mb=advantages[idx],
                    response_mask_mb=response_mask[idx],
                )
                for k in ("actor_loss", "entropy",
                          "approx_kl", "clip_fraction", "grad_norm"):
                    acc[k] += stats[k]
                epoch_kl_sum += stats["approx_kl"]
                epoch_mb_count += 1
                n_updates_total += 1

            epochs_done = epoch + 1
            mean_kl_epoch = epoch_kl_sum / max(epoch_mb_count, 1)
            if cfg.target_kl is not None and mean_kl_epoch > 1.5 * cfg.target_kl:
                early_stopped = True
                break

        norm = max(n_updates_total, 1)
        out = {k: v / norm for k, v in acc.items()}
        out["n_epochs_done"] = float(epochs_done)
        out["early_stopped"] = float(early_stopped)
        out["mean_reward"] = float(
            (rm_rewards * response_lens.float()).sum().item() /
            max(float(response_lens.float().sum().item()), 1.0)
        )
        out["mean_kl_to_ref_sample"] = float(
            ((kl_per_token * response_mask).sum() / response_mask.sum().clamp(min=1.0)).item()
        )
        out["mean_kl_to_ref"] = self._compute_analytic_kl_to_ref(
            prompts, responses, prompt_lens, response_lens, response_mask
        )
        out["mean_response_len"] = float(response_lens.float().mean().item())
        # group advantage 统计（教学/调试用）
        out["mean_abs_advantage"] = float(
            (advantages.abs() * response_mask).sum().item() /
            max(float(response_mask.sum().item()), 1.0)
        )
        return out

    @torch.no_grad()
    def _compute_analytic_kl_to_ref(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        prompt_lens: torch.Tensor,
        response_lens: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> float:
        """对每个 response token 位置算解析 KL(actor || ref)（与 RLHFTrainer 一致）。"""
        N = prompts.size(0)
        kl_sum = 0.0
        n_valid = 0.0
        for i in range(N):
            Lp = int(prompt_lens[i].item())
            Lr = int(response_lens[i].item())
            if Lr == 0:
                continue
            real_full = torch.cat([
                prompts[i, :Lp],
                responses[i, :Lr],
            ]).unsqueeze(0)
            actor_logits = self.actor(real_full)
            ref_logits = self.reference(real_full)
            la = F.log_softmax(actor_logits, dim=-1)
            lr_ = F.log_softmax(ref_logits, dim=-1)
            # 向量化：避免逐 token .item()（GPU 上每 token 强制同步一次）
            pos = torch.arange(Lp - 1, Lp + Lr - 1, device=self.device)
            la_pos = la[0, pos]  # [Lr, V]
            lr_pos = lr_[0, pos]
            kl_t = (la_pos.exp() * (la_pos - lr_pos)).sum(dim=-1)  # [Lr]
            kl_sum += float(kl_t.sum())
            n_valid += float(Lr)
        return float(kl_sum / max(n_valid, 1.0))

    def _ppo_step(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        prompt_lens: torch.Tensor,
        log_probs_old_mb: torch.Tensor,
        advantages_mb: torch.Tensor,
        response_mask_mb: torch.Tensor,
    ) -> Dict[str, float]:
        """单个 mini-batch 的 PPO 梯度步（**只更新 actor，没有 critic step**）。"""
        cfg = self.cfg
        B, T_r = responses.shape

        # 逐样本 forward（原因同 RLHFTrainer）
        log_probs_new_list: List[torch.Tensor] = []
        entropies_list: List[torch.Tensor] = []
        for b in range(B):
            Lp = int(prompt_lens[b].item())
            Lr = int(response_mask_mb[b].sum().item())
            if Lr == 0:
                log_probs_new_list.append(torch.zeros(T_r, device=self.device))
                entropies_list.append(torch.zeros(T_r, device=self.device))
                continue
            real_full = torch.cat([
                prompts[b, :Lp],
                responses[b, :Lr],
            ]).unsqueeze(0)
            actor_logits = self.actor(real_full)
            log_probs_full = F.log_softmax(actor_logits, dim=-1)

            # 向量化 gather：一次取出全部真实 token 的 log π / entropy（可微）
            pos = torch.arange(Lp - 1, Lp + Lr - 1, device=self.device)  # [Lr]
            a_t = responses[b, :Lr]                                       # [Lr]
            lp_pos = log_probs_full[0, pos]              # [Lr, V]
            real_lp = lp_pos.gather(1, a_t.unsqueeze(1)).squeeze(1)       # [Lr]
            real_h = -(lp_pos.exp() * lp_pos).sum(dim=-1)                  # [Lr] entropy
            pad_extra = T_r - Lr
            if pad_extra > 0:
                zeros = torch.zeros(pad_extra, device=self.device)
                real_lp = torch.cat([real_lp, zeros])
                real_h = torch.cat([real_h, zeros])
            log_probs_new_list.append(real_lp)
            entropies_list.append(real_h)

        log_probs_new = torch.stack(log_probs_new_list)  # [B, T_r]
        entropies = torch.stack(entropies_list)

        mask = response_mask_mb
        log_ratio = (log_probs_new - log_probs_old_mb) * mask
        ratio = torch.exp(log_ratio.clamp(-30.0, 30.0))

        # PPO-Clip surrogate
        clip_obj = compute_clip_objective(
            ratio=ratio * mask,
            advantages=advantages_mb * mask,
            clip_eps=cfg.clip_eps,
            normalize_adv=False,
        )
        n_valid = mask.sum().clamp(min=1.0)
        actor_loss = -(clip_obj["objective_per_sample"] * mask).sum() / n_valid
        clip_frac = (clip_obj["clipped_mask"] * mask).sum() / n_valid

        entropy = (entropies * mask).sum() / n_valid

        # **没有 critic_loss**（去掉 value function 的核心承诺）
        total_loss = actor_loss - cfg.entropy_coef * entropy

        # **只更新 actor**（没有 critic_opt.zero_grad / step）
        self.actor_opt.zero_grad()
        total_loss.backward()
        actor_gnorm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=cfg.max_grad_norm
        ).item()
        self.actor_opt.step()

        with torch.no_grad():
            r_clamped = ratio.clamp(min=1e-8)
            kl_mb = ((r_clamped - 1.0) - torch.log(r_clamped)) * mask
            kl_mb_mean = (kl_mb.sum() / n_valid).item()

        return dict(
            actor_loss=float(actor_loss.item()),
            entropy=float(entropy.item()),
            approx_kl=kl_mb_mean,
            clip_fraction=float(clip_frac.item()),
            grad_norm=float(actor_gnorm),
        )

    # ------------------------------------------------------------------
    # 3.5 train: 完整训练 loop
    # ------------------------------------------------------------------
    def train(
        self,
        prompt_pool: Sequence[torch.Tensor],
        n_iters: int = 50,
        n_prompts_per_iter: int = 2,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """完整 GRPO 训练 loop。

        Parameters
        ----------
        prompt_pool : Sequence of 1D LongTensor
            每次 iteration 从中**随机有放回**抽 ``n_prompts_per_iter`` 个 prompt。
            每个 prompt 会被采 ``cfg.group_size`` 个 response，所以每次 iter
            共有 ``n_prompts_per_iter * cfg.group_size`` 条 response 参与 PPO 更新。
        n_iters : int
            多少个 outer iteration。
        n_prompts_per_iter : int
            每次 iter 抽几个不同 prompt（每个 prompt 自成一个 group）。
        verbose : bool
            是否打印每个 iter 的 metrics。

        Returns
        -------
        history : List[Dict[str, float]]
        """
        rng = random.Random(0)
        t0 = time_now()
        for it in range(n_iters):
            # 1) 抽若干 prompt（每个 prompt 自成一个 group）
            prompts = [rng.choice(prompt_pool) for _ in range(n_prompts_per_iter)]

            # 2) rollout G 个 response per prompt
            rollout = self.rollout_group(prompts)

            # 3) GRPO update（只更新 actor）
            stats = self.grpo_update(rollout)
            self.history.append(stats)

            if verbose and (it % self.cfg.print_every == 0 or it == n_iters - 1):
                elapsed = time_now() - t0
                es = "ES" if stats["early_stopped"] else "  "
                print(
                    f"iter {it:>3} | reward={stats['mean_reward']:+.3f} | "
                    f"KL(ref)={stats['mean_kl_to_ref']:+.3f} | "
                    f"|A|={stats['mean_abs_advantage']:.3f} | "
                    f"len={stats['mean_response_len']:.1f} | "
                    f"actor={stats['actor_loss']:+.4f} | "
                    f"H={stats['entropy']:.3f} | "
                    f"KL(old)={stats['approx_kl']:.4f} | "
                    f"clip%={stats['clip_fraction'] * 100:.1f} | "
                    f"ep={stats['n_epochs_done']:.0f}/{es} | "
                    f"({elapsed:.1f}s)"
                )
        return self.history


def time_now() -> float:
    """time.time() 的薄包装（方便测试 monkey-patch）。"""
    import time
    return time.time()


__all__ = [
    "compute_group_advantages",
    "GRPOConfig",
    "GRPOTrainer",
    "time_now",
]
