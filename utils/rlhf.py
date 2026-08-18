"""RLHF-PPO（InstructGPT 配方）的 4 模型协调器（Ch12）。

本章兑现 Ch00 的两个核心承诺：
    1. **"InstructGPT 配方"** —— SFT → RM → RLHF-PPO 三阶段全流程
    2. **"4 模型仪表盘"** —— actor π_θ / critic V_φ / reward r / reference π_ref

--------------------------------------------------------------------
核心数学（详见 Ch12 §12.3-12.4）

RLHF 的约束优化目标：

    max_π  E_{x~D, y~π(·|x)}[ r(x, y) ]
    s.t.   E_{x~D}[ KL(π(·|x) || π_ref(·|x)) ] ≤ ε

通过 Lagrangian 转成无约束形式后得到**调整后的 reward**：

    r_total(x, y) = r(x, y) - β · KL(π || π_ref)[x, y]

token-level 形式（per-token KL penalty，最后一步加 RM 的 r(x,y)）：

    r_t = - β · log( π(a_t | s_t) / π_ref(a_t | s_t) )      t < T-1
    r_{T-1} = r(x, y) - β · log( π(a_{T-1} | s_{T-1}) / π_ref(...) )

其中 s_t = (x, y_{<t})（已生成 prefix），a_t = y_t（下一个 token）。

LLM 生成建模成 MDP：
    - state  s_t = (x, y_{<t})          （已生成 prefix）
    - action a_t = y_t                   （下一个 token）
    - transition：确定性 s_{t+1} = (s_t, a_t)
    - γ：单轮对话不长，0.9-0.95（Ch02 §2.5 承诺）

--------------------------------------------------------------------
模块组成

- :class:`ValueHead`          —— 复用 TinyGPT backbone + 一个 scalar value head，
                                 输入 prefix，输出每个位置的 V_φ(s_t)
- :class:`RLHFConfig`         —— 超参 dataclass（β、γ、clip ε、GAE λ、K epochs）
- :class:`RLHFTrainer`        —— 4 模型协调器
    * :meth:`rollout_responses`     —— 用 actor 采 G 个 response
    * :meth:`compute_token_rewards` —— reward model + KL penalty → per-token reward
    * :meth:`compute_token_values`  —— critic 给每个 token 一个 value
    * :meth:`rlhf_update`           —— PPO + GAE 更新（actor + critic）
    * :meth:`train`                 —— 完整训练 loop

设计原则（与项目其它模块一致）：
    - **教学优先**：每一步把数学展开，便于读者对照公式
    - **不重复造轮子**：复用 :class:`rlenvs.tiny_gpt.TinyGPT` 作 actor / reference /
      value backbone；复用 :class:`utils.reward_model.RewardModel`；复用
      :func:`utils.gae.compute_gae` 和 :func:`utils.ppo.compute_clip_objective`
    - **小而能训**：默认配置 < 500k 参数，CPU 上一步 < 1s

--------------------------------------------------------------------
注意：和 CartPole 版 ``utils.ppo.ppo_update`` 的差异

CartPole 版假设状态是定长向量（state_dim），可以堆成 ``[N, state_dim]`` 喂给 MLP。
RLHF 里 "state" 是 token 序列（变长 prefix），不能直接展平。
所以本模块**不直接复用** :func:`utils.ppo.ppo_update`，
但**复用其组件**：
    - GAE（``compute_gae``）逐 trajectory 算 advantage
    - PPO-Clip 目标（``compute_clip_objective``）逐 token 算 surrogate
    - importance ratio / KL early stopping 思想完全一致

PPO 在 token 序列上和 CartPole 上**数学完全一样**，只是数据 shape 不同：
    - "一步" = "一个 token"
    - "一条 trajectory" = "一个 (prompt, response) 对"
    - "batch" = "G 个 response 摊平的所有 token"
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gae import compute_gae
from .ppo import compute_clip_objective


# =============================================================================
# 1. ValueHead：复用 TinyGPT backbone，输出每个位置的 V_φ(s_t)
# =============================================================================
class ValueHead(nn.Module):
    """TinyGPT backbone + 一个 scalar value head，输出**每个 token 位置**的 V。

    结构::

        input_ids [B, T]   (prompt + response tokens)
            ↓ TinyGPT 的 token embedding + PE + N × TransformerBlock + ln_final
            hidden states [B, T, d_model]   ← forward hook 抓 ln_final 输入
            ↓ value_ln + value_head: Linear(d_model → 1)
            values [B, T]   (每个 token 位置一个 V_φ(s_t))

    和 RewardModel 的区别：
        - RewardModel 取**最后一个** token 的 hidden → 整段序列 1 个 reward
        - ValueHead 取**每个** token 的 hidden → 每个位置一个 value

    s_t = 已生成 prefix (x, y_{<t})。我们的约定是：
    ``input_ids[b, t]`` 是 prefix 末尾的下一个 token，hidden states[b, t] "看到"
    了 input_ids[b, :t+1]（causal）。所以 hidden[b, t] = V_φ(s_t) = V_φ((x, y_{<t+1}))。

    在 RLHF 里，我们要 value 估计的 s_t 是"生成 a_t 之前"的 prefix，
    所以 value[b, t] 对应位置 t 用的 prefix 是 input_ids[:, :t]（不含 t）。
    实现细节见 RLHFTrainer 里如何对齐 value 和 token。
    """

    def __init__(self, backbone: nn.Module, d_model: Optional[int] = None) -> None:
        super().__init__()
        self.backbone = backbone
        if d_model is None:
            d_model = getattr(backbone, "d_model", None)
            if d_model is None:
                raise ValueError("无法从 backbone 推断 d_model")
        self.d_model = d_model
        self.value_ln = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        self._hidden: Optional[torch.Tensor] = None
        target = self._find_ln_final(backbone)
        if target is None:
            raise ValueError("backbone 上找不到 'ln_final' 模块")
        target.register_forward_hook(self._hook)

    @staticmethod
    def _find_ln_final(module: nn.Module) -> Optional[nn.Module]:
        if hasattr(module, "ln_final"):
            return module.ln_final
        for name, child in module.named_modules():
            if name == "ln_final":
                return child
        return None

    def _hook(self, module, inputs, output):
        if isinstance(inputs, tuple) and len(inputs) > 0:
            self._hidden = inputs[0].detach()
        else:
            self._hidden = output.detach()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, T] → values [B, T]（每个位置一个 V_φ）。"""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        _ = self.backbone(input_ids)
        hidden = self._hidden  # [B, T, d_model]
        if hidden is None:
            raise RuntimeError("forward hook 没抓到 hidden state")
        # value head: 每个 token 一个标量
        values = self.value_head(self.value_ln(hidden)).squeeze(-1)  # [B, T]
        return values


# =============================================================================
# 2. RLHFConfig：所有 RLHF 超参的 dataclass
# =============================================================================
@dataclass
class RLHFConfig:
    """RLHF-PPO 训练的所有超参。

    Attributes
    ----------
    beta : float
        KL penalty 系数 β。β 大 → 更保守（更靠近 ref）；β 小 → 更激进。
        InstructGPT 用 0.01-0.1；我们的 toy 实验用 0.05-0.5。
    gamma : float
        折扣因子 γ（Ch02 §2.5）。LLM 单轮对话不长，0.9-0.95。
    lam : float
        GAE 的 λ。
    clip_eps : float
        PPO clip ε。
    update_epochs : int
        同一批 rollout 数据反复 K epochs（PPO 多 epoch 数据重用）。
    inner_minibatch_size : int
        每个 inner epoch 切成多大的 mini-batch。
    entropy_coef : float
        entropy bonus 系数（鼓励探索）。
    value_coef : float
        value loss 权重。
    max_grad_norm : float
        梯度裁剪阈值。
    target_kl : Optional[float]
        每个 inner epoch 后估 KL(actor_old || actor_new)，超 1.5×target_kl 就 early stop。
    response_max_len : int
        每条 response 最多生成多少 token。
    temperature : float
        rollout 采样温度（>1 更随机，<1 更确定）。
    normalize_advantage : bool
        是否对 advantage 做 batch 标准化（PPO 工程标配）。
    """

    # KL penalty / discount
    beta: float = 0.1
    gamma: float = 0.95
    lam: float = 0.95

    # PPO
    clip_eps: float = 0.2
    update_epochs: int = 4
    inner_minibatch_size: int = 8
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.04
    normalize_advantage: bool = True

    # Rollout
    response_max_len: int = 16
    temperature: float = 1.0
    top_k: Optional[int] = None

    # Optimizers
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4

    # Reporting
    print_every: int = 1


# =============================================================================
# 3. RLHFTrainer：4 模型协调器
# =============================================================================
class RLHFTrainer:
    """协调 actor / critic / reward / reference 四个模型，跑 RLHF-PPO。

    Parameters
    ----------
    actor : TinyGPT-like
        要训练的策略 π_θ。``forward(input_ids) -> logits [B, T, V]``。
    critic : ValueHead
        V_φ。``forward(input_ids) -> values [B, T]``。
    reward_model : RewardModel
        r(x, y)。``forward(prompt_ids, response_ids) -> reward [B]``。
    reference : TinyGPT-like
        π_ref（冻结的 SFT 模型）。同 actor 接口。
    pad_id : int
        padding token id（用于 batch）。
    cfg : RLHFConfig
        超参。
    device : str
        ``'cpu'`` 或 ``'cuda'``。

    Notes
    -----
    - **reference 始终冻结**（``requires_grad_(False)``、``eval()``）。
    - **reward_model 在 RLHF 阶段也冻结**（InstructGPT 配方：RM 训好后不再动）。
    - 只有 actor 和 critic 接受梯度。
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: ValueHead,
        reward_model: nn.Module,
        reference: nn.Module,
        pad_id: int,
        cfg: Optional[RLHFConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.reference = reference
        self.pad_id = pad_id
        self.cfg = cfg if cfg is not None else RLHFConfig()
        self.device = device

        # 冻结 reference 和 reward_model（RLHF 标准做法）
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()
        for p in self.reward_model.parameters():
            p.requires_grad_(False)
        self.reward_model.eval()

        # Optimizer
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.cfg.actor_lr, weight_decay=0.0
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=self.cfg.critic_lr, weight_decay=0.0
        )

        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # 3.1 Rollout：用 actor 采 G 个 response（同时记 old log π / ref log π）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout_responses(
        self,
        prompt_ids_list: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """用**当前 actor** 对每个 prompt 采一个 response。

        流程::

            for each prompt x:
                y = sample(actor, x, max_new_tokens=response_max_len, T=temperature)
                对每个 token 位置 t：
                    log_pi_old(a_t | s_t)    = log softmax(actor(input_ids[:, :t+1]))[-1, a_t]
                    log_pi_ref(a_t | s_t)    = log softmax(ref(input_ids[:, :t+1]))[-1, a_t]
                values[b, t] = critic(input_ids[:, :t+1])[?, t]  # per-token value

        为了高效（避免对每个 token 单独跑一次 transformer），我们一次性把
        (prompt + response) 整段喂给 actor/ref/critic，然后用 causal 性质
        把"每个位置预测下一个 token 的 logits"取出来。

        Returns
        -------
        dict 含:
            - ``prompts``     : LongTensor [B, T_p]   (left-padded prompt)
            - ``responses``   : LongTensor [B, T_r]   (response tokens)
            - ``prompt_lens`` : LongTensor [B]
            - ``response_lens``: LongTensor [B]
            - ``log_probs_old``: FloatTensor [B, T_r]   log π_actor(a_t | s_t) at rollout time
            - ``log_probs_ref``: FloatTensor [B, T_r]   log π_ref(a_t | s_t)   (frozen)
            - ``values_old``   : FloatTensor [B, T_r]   V_φ(s_t)  at rollout time
                （注意：values_old[b, t] = V_φ(s_t) 其中 s_t = (x, y_{<t})，
                 即"生成 a_t 之前"的 prefix——见 §12.4 中的 state 定义对齐）
            - ``response_mask``: FloatTensor [B, T_r]   1 = real token, 0 = pad
        """
        # rollout 期间切 eval（关 dropout 等随机性），结束后恢复训练模式；
        # reference 是冻结模型，保持 eval 即可
        _modes_were = (self.actor.training, self.critic.training)
        self.actor.eval()
        self.critic.eval()
        self.reference.eval()

        cfg = self.cfg
        device = self.device

        # 1) 对每个 prompt 采 response
        all_prompts: List[torch.Tensor] = []
        all_responses: List[torch.Tensor] = []
        for p in prompt_ids_list:
            p_t = p.to(device).long()
            if p_t.dim() == 0:
                continue
            p_b = p_t.unsqueeze(0)  # [1, T_p]
            full = self._sample_response(self.actor, p_b, cfg.response_max_len,
                                         temperature=cfg.temperature, top_k=cfg.top_k,
                                         forbidden_ids={self.pad_id})
            # full shape [1, T_p + T_r]
            resp = full[:, p_t.size(0):]  # [1, T_r]
            all_prompts.append(p_t)
            all_responses.append(resp.squeeze(0))

        B = len(all_prompts)
        if B == 0:
            raise ValueError("rollout_responses: 没有 prompt")

        T_p = max(int(p.size(0)) for p in all_prompts)
        T_r = max(int(r.size(0)) for r in all_responses)

        # 2) padding 成 batch
        prompts = torch.full((B, T_p), self.pad_id, dtype=torch.long, device=device)
        responses = torch.full((B, T_r), self.pad_id, dtype=torch.long, device=device)
        prompt_lens = torch.zeros(B, dtype=torch.long, device=device)
        response_lens = torch.zeros(B, dtype=torch.long, device=device)
        for b, (p, r) in enumerate(zip(all_prompts, all_responses)):
            Lp, Lr = int(p.size(0)), int(r.size(0))
            prompts[b, :Lp] = p
            responses[b, :Lr] = r
            prompt_lens[b] = Lp
            response_lens[b] = Lr

        response_mask = (responses != self.pad_id).float()  # [B, T_r]

        # 3) 算每步的 old log π / ref log π / value
        # 注意：**不能直接对 padded batch 跑 actor**——因为 prompt 在 batch 内长度不同，
        # 用 pad 填充后，response tokens 会 attend 到 prompt 的 pad 上，
        # 采样时的真实 context（不含 pad）和重算 log_prob 时的 context（含 pad）不一致，
        # 会让 log π_old ≠ 真实采样概率 → importance ratio 失真 → 样本 KL 可能变负。
        #
        # 解决：**逐样本** forward 真实（未 pad）的 prompt+response 序列，
        # 抽出每个 response token 位置的 next-token 分布。
        log_probs_old = torch.zeros(B, T_r, device=device)
        log_probs_ref_resp = torch.zeros(B, T_r, device=device)
        values_old = torch.zeros(B, T_r, device=device)
        for b in range(B):
            Lp = int(prompt_lens[b].item())
            Lr = int(response_lens[b].item())
            if Lr == 0:
                continue
            # 真实 prompt + 真实 response（无 padding）
            real_full = torch.cat([
                prompts[b, :Lp],
                responses[b, :Lr],
            ]).unsqueeze(0)  # [1, Lp + Lr]
            actor_logits = self.actor(real_full)  # [1, Lp+Lr, V]
            ref_logits = self.reference(real_full)
            all_values = self.critic(real_full)  # [1, Lp+Lr]

            log_probs_actor = F.log_softmax(actor_logits, dim=-1)  # [1, T, V]
            log_probs_ref = F.log_softmax(ref_logits, dim=-1)

            # 抽每个 response token 的 log π（向量化：一次 gather 取全部 Lr 个位置）
            # response token j (j=0..Lr-1) 在 input 中位于位置 Lp + j
            # 预测它的 logits 在位置 (Lp + j) - 1
            pos = torch.arange(Lp - 1, Lp + Lr - 1, device=device)  # [Lr]
            a_t = responses[b, :Lr]                                  # [Lr]
            log_probs_old[b, :Lr] = log_probs_actor[0, pos, a_t]
            log_probs_ref_resp[b, :Lr] = log_probs_ref[0, pos, a_t]
            # value[b, j] = V_φ(s_t) where s_t = (x, y_{<j+1})
            # hidden[pos] sees input[:pos+1] = prompt + response[:j+1]
            # = s_t (the prefix when we predict a_t)
            values_old[b, :Lr] = all_values[0, pos]

        # rollout 结束，恢复进入本函数前的训练模式（防止 eval 泄漏到后续 PPO 更新）
        self.actor.train(_modes_were[0])
        self.critic.train(_modes_were[1])
        return dict(
            prompts=prompts,
            responses=responses,
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            log_probs_old=log_probs_old,
            log_probs_ref=log_probs_ref_resp,
            values_old=values_old,
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
        """自回归采样（内部用，不依赖 rlenvs.tiny_gpt.generate 以便控制随机性）。

        prompt: [1, T_p]；返回 [1, T_p + max_new_tokens]。

        forbidden_ids : set of int, optional
            采样时把这些 token 的 logits 设为 -inf（典型：pad token），
            防止 actor 退化成"只生成 pad"。
        """
        out = prompt
        for _ in range(max_new_tokens):
            logits = model(out)[:, -1, :]  # [1, V]
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
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            out = torch.cat([out, next_id], dim=1)
        return out

    # ------------------------------------------------------------------
    # 3.2 compute_token_rewards: reward model + KL penalty → per-token reward
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
        """算每个 response token 的 reward：r_t = -β log(π/π_ref)，最后一步加 r(x,y)。

        数学（§12.3-12.4）::

            r_t = -β · log( π(a_t|s_t) / π_ref(a_t|s_t) )     t = 0, ..., T-2
            r_{T-1} = r(x, y) - β · log( π(a_{T-1}|s_{T-1}) / π_ref(...) )

        其中 r(x, y) 来自 reward model（整段 response 一个标量），加到最后一个
        真实 token 位置（KL penalty 同样加）。

        Returns
        -------
        token_rewards : Tensor [B, T_r]   per-token reward（pad 位置为 0）
        rm_rewards    : Tensor [B]         reward model 标量（每条 response）
        kl_per_token  : Tensor [B, T_r]    每个 token 的 KL contribution = log(π/π_ref)
        """
        cfg = self.cfg
        B, T_r = responses.shape

        # 1) KL per token：log(π/π_ref) = log π - log π_ref
        # 注意 log_probs_old 是 log π_actor (at rollout time, detached)
        kl_per_token = log_probs_old - log_probs_ref  # [B, T_r]

        # 2) reward model 给整段一个标量 r(x, y)
        rm_rewards = self.reward_model(prompts, responses)  # [B]

        # 3) 组装 per-token reward
        token_rewards = -cfg.beta * kl_per_token  # 所有位置先加 KL penalty

        # 在每个 response 的最后一个真实 token 位置加 r(x, y)
        last_idx = response_lens.long() - 1  # [B], 形如 Lr-1
        for b in range(B):
            li = int(last_idx[b].item())
            if li >= 0:
                token_rewards[b, li] = token_rewards[b, li] + rm_rewards[b]

        # pad 位置清零（虽然后续 mask 会处理，但保持干净）
        token_rewards = token_rewards * (responses != self.pad_id).float()

        return token_rewards, rm_rewards, kl_per_token

    # ------------------------------------------------------------------
    # 3.3 Per-trajectory GAE：把每条 response 当作一条 trajectory
    # ------------------------------------------------------------------
    def _compute_advantages_returns(
        self,
        token_rewards: torch.Tensor,  # [B, T_r]
        values_old: torch.Tensor,     # [B, T_r]
        response_lens: torch.Tensor,  # [B]
        response_mask: torch.Tensor,  # [B, T_r]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """逐 trajectory 算 GAE advantage 和 return。

        把每条 response 当一条 trajectory，最后一个真实 token 之后 bootstrap value = 0
        （episode 自然终止，没有后续）。

        Returns
        -------
        advantages : Tensor [B, T_r]    Â_t^{GAE(γ, λ)}
        returns    : Tensor [B, T_r]    G_t = Â_t + V_φ(s_t)  (critic target)
        """
        cfg = self.cfg
        B, T_r = token_rewards.shape
        advantages = torch.zeros_like(token_rewards)
        returns = torch.zeros_like(token_rewards)

        token_rewards_np = token_rewards.cpu().numpy()
        values_old_np = values_old.cpu().numpy()
        lens_np = response_lens.cpu().numpy()

        for b in range(B):
            Lr = int(lens_np[b])
            if Lr == 0:
                continue
            r_seq = token_rewards_np[b, :Lr].tolist()
            v_seq = values_old_np[b, :Lr].tolist()
            # 最后一步后 bootstrap value = 0（terminal）
            adv = compute_gae(
                rewards=r_seq,
                values=v_seq,
                last_value=0.0,
                gamma=cfg.gamma,
                lam=cfg.lam,
                dones=[False] * Lr,
            )
            advantages[b, :Lr] = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            returns[b, :Lr] = advantages[b, :Lr] + values_old[b, :Lr]

        # pad 位置清零
        advantages = advantages * response_mask
        returns = returns * response_mask
        return advantages, returns

    # ------------------------------------------------------------------
    # 3.4 rlhf_update: PPO 更新（actor + critic）
    # ------------------------------------------------------------------
    def rlhf_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """对一批 rollout 数据做 PPO 多-epoch 更新。

        流程::

            1. 从 rollout 算 token_rewards（reward + KL penalty）
            2. 算 advantages / returns（GAE，常量）
            3. 保存 log_probs_old（常量）
            4. for epoch in range(K):
                 for minibatch of responses:
                   # 重算 actor/critic forward（共享 backbone 重跑一遍）
                   new log π(a_t | s_t) = log softmax(actor(prefix))[-1, a_t]
                   new V(s_t) = critic(prefix)
                   ratio = exp(new log π - log π_old)
                   actor_loss  = -PPO-Clip surrogate
                   critic_loss = (V_new - returns)^2
                   entropy bonus
                   backward + step
                 if approx_kl > 1.5 * target_kl: early stop

        Returns
        -------
        dict 含标量 metrics（actor_loss / critic_loss / entropy / approx_kl /
        clip_fraction / mean_reward / mean_kl_to_ref / n_epochs_done）。
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
        values_old = rollout["values_old"].detach()

        # 1) token rewards（KL penalty + RM scalar）
        token_rewards, rm_rewards, kl_per_token = self.compute_token_rewards(
            prompts, responses, response_lens, log_probs_old, log_probs_ref
        )

        # 2) GAE advantages / returns（常量）
        advantages, returns = self._compute_advantages_returns(
            token_rewards, values_old, response_lens, response_mask
        )

        # flatten across batch for normalization / minibatching
        # 这里我们以"每个 response"为单位切 mini-batch（保留 trajectory 结构）
        B = prompts.size(0)

        # advantage normalization（PPO 工程标配，只用真实 token 位置）
        if cfg.normalize_advantage:
            mask = response_mask.bool()
            if mask.any():
                adv_flat = advantages[mask]
                adv_mean = adv_flat.mean()
                adv_std = adv_flat.std()
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                advantages = advantages * response_mask

        # PPO inner loop
        acc = dict(
            actor_loss=0.0, critic_loss=0.0, entropy=0.0,
            approx_kl=0.0, clip_fraction=0.0, grad_norm=0.0,
        )
        n_updates_total = 0
        epochs_done = 0
        early_stopped = False

        for epoch in range(cfg.update_epochs):
            perm = torch.randperm(B, device=device)
            mb_size = min(cfg.inner_minibatch_size, B)
            epoch_kl_sum = 0.0
            epoch_mb_count = 0

            for start in range(0, B, mb_size):
                idx = perm[start:start + mb_size]
                stats = self._ppo_step(
                    prompts=prompts[idx],
                    responses=responses[idx],
                    prompt_lens=prompt_lens[idx],
                    log_probs_old_mb=log_probs_old[idx],
                    values_old_mb=values_old[idx],
                    advantages_mb=advantages[idx],
                    returns_mb=returns[idx],
                    response_mask_mb=response_mask[idx],
                )
                for k in ("actor_loss", "critic_loss", "entropy",
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
        # KL metric：sample-based 是 `E_π_old[log π_old - log π_ref]`，理论 >= 0，
        # 但小样本下方差大；保留作为参考。
        out["mean_kl_to_ref_sample"] = float(
            ((kl_per_token * response_mask).sum() / response_mask.sum().clamp(min=1.0)).item()
        )
        # 解析 KL：直接用 actor / reference 的 logits 算 sum_a π(a)(log π(a) - log π_ref(a))
        # 在每个 response token 位置。保证 >= 0。
        out["mean_kl_to_ref"] = self._compute_analytic_kl_to_ref(
            prompts, responses, prompt_lens, response_lens, response_mask
        )
        out["mean_response_len"] = float(response_lens.float().mean().item())
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
        """对每个 response token 位置算解析 KL(actor || ref) = sum_a π_a (log π_a - log π_ref)。

        保证 >= 0（sample-based 估计在小样本下可能为负，所以这里用解析版本作主指标）。
        """
        B = prompts.size(0)
        T_r = responses.size(1)
        kl_sum = 0.0
        n_valid = 0.0
        for b in range(B):
            Lp = int(prompt_lens[b].item())
            Lr = int(response_lens[b].item())
            if Lr == 0:
                continue
            real_full = torch.cat([
                prompts[b, :Lp],
                responses[b, :Lr],
            ]).unsqueeze(0)
            actor_logits = self.actor(real_full)
            ref_logits = self.reference(real_full)
            la = F.log_softmax(actor_logits, dim=-1)  # [1, T, V]
            lr_ = F.log_softmax(ref_logits, dim=-1)
            # 向量化：一次取出所有 Lr 个位置的 [Lr, V] 切片再求和
            # （避免逐 token .item()——那会在 GPU 上每 token 强制同步一次）
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
        values_old_mb: torch.Tensor,
        advantages_mb: torch.Tensor,
        returns_mb: torch.Tensor,
        response_mask_mb: torch.Tensor,
    ) -> Dict[str, float]:
        """单个 mini-batch 的一次 PPO 梯度步。"""
        cfg = self.cfg

        # 逐样本 forward（不能用 padded batch，原因见 rollout_responses 的注释）。
        # 我们用一个 list 累积可微的 per-token tensors，最后 stack 起来。
        B, T_r = responses.shape
        log_probs_new_list: List[torch.Tensor] = []
        values_new_list: List[torch.Tensor] = []
        entropies_list: List[torch.Tensor] = []
        for b in range(B):
            Lp = int(prompt_lens[b].item())
            Lr = int(response_mask_mb[b].sum().item())  # 真实 response token 数
            if Lr == 0:
                # 该样本没有真实 token，填零占位
                log_probs_new_list.append(torch.zeros(T_r, device=self.device))
                values_new_list.append(torch.zeros(T_r, device=self.device))
                entropies_list.append(torch.zeros(T_r, device=self.device))
                continue
            real_full = torch.cat([
                prompts[b, :Lp],
                responses[b, :Lr],
            ]).unsqueeze(0)  # [1, Lp + Lr]
            actor_logits = self.actor(real_full)  # [1, Lp+Lr, V]
            values_full = self.critic(real_full)  # [1, Lp+Lr]
            log_probs_full = F.log_softmax(actor_logits, dim=-1)  # [1, T, V]

            # 对每个真实 response token 抽取 log π / value / entropy（向量化 gather）
            pos = torch.arange(Lp - 1, Lp + Lr - 1, device=self.device)  # [Lr]
            a_t = responses[b, :Lr]                                       # [Lr]
            lp_pos = log_probs_full[0, pos]              # [Lr, V]，可微
            real_lp = lp_pos.gather(1, a_t.unsqueeze(1)).squeeze(1)       # [Lr]
            real_v = values_full[0, pos]                                   # [Lr]
            real_h = -(lp_pos.exp() * lp_pos).sum(dim=-1)                  # [Lr] entropy
            # pad 到 T_r（剩余位置 0）
            pad_extra = T_r - Lr
            if pad_extra > 0:
                zeros = torch.zeros(pad_extra, device=self.device)
                real_lp = torch.cat([real_lp, zeros])
                real_v = torch.cat([real_v, zeros])
                real_h = torch.cat([real_h, zeros])
            log_probs_new_list.append(real_lp)
            values_new_list.append(real_v)
            entropies_list.append(real_h)

        log_probs_new = torch.stack(log_probs_new_list)  # [B, T_r]
        values_new = torch.stack(values_new_list)
        entropies = torch.stack(entropies_list)

        # mask
        mask = response_mask_mb  # [B, T_r]
        log_ratio = (log_probs_new - log_probs_old_mb) * mask
        # 用 exp() 可能数值不稳，加 clamp
        ratio = torch.exp(log_ratio.clamp(-30.0, 30.0))

        # PPO-Clip surrogate
        clip_obj = compute_clip_objective(
            ratio=ratio * mask,  # 把 pad 位置 ratio 设为 0（adv 也 0，无影响）
            advantages=advantages_mb * mask,
            clip_eps=cfg.clip_eps,
            normalize_adv=False,
        )
        # clip_obj 的 loss 是对所有 B*T 位置 mean，pad 位置会稀释。
        # 按"真实 token 数"重新归一化（n_valid）。
        n_valid = mask.sum().clamp(min=1.0)
        actor_loss = -(clip_obj["objective_per_sample"] * mask).sum() / n_valid
        # clip fraction（只看真实 token）
        clip_frac = (clip_obj["clipped_mask"] * mask).sum() / n_valid

        # critic loss（MSE，只算真实 token）
        critic_loss = 0.5 * ((values_new - returns_mb) ** 2 * mask).sum() / n_valid

        # entropy bonus（鼓励探索）
        entropy = (entropies * mask).sum() / n_valid

        total_loss = (
            actor_loss
            + cfg.value_coef * critic_loss
            - cfg.entropy_coef * entropy  # 减：要 maximize entropy
        )

        # ---- actor + critic backward（两个 optim） ----
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        total_loss.backward()
        # 分别 clip / step（让两个网络各自更新）
        actor_gnorm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=cfg.max_grad_norm
        ).item()
        critic_gnorm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=cfg.max_grad_norm
        ).item()
        self.actor_opt.step()
        self.critic_opt.step()

        # KL estimate (Schulman): mean((r-1) - log r) on valid tokens
        with torch.no_grad():
            r_clamped = ratio.clamp(min=1e-8)
            kl_mb = ((r_clamped - 1.0) - torch.log(r_clamped)) * mask
            kl_mb_mean = (kl_mb.sum() / n_valid).item()

        return dict(
            actor_loss=float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            entropy=float(entropy.item()),
            approx_kl=kl_mb_mean,
            clip_fraction=float(clip_frac.item()),
            grad_norm=float((actor_gnorm + critic_gnorm) / 2.0),
        )

    # ------------------------------------------------------------------
    # 3.5 train: 完整训练 loop
    # ------------------------------------------------------------------
    def train(
        self,
        prompt_pool: Sequence[torch.Tensor],
        n_iters: int = 50,
        group_size: int = 8,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """完整 RLHF-PPO 训练 loop。

        Parameters
        ----------
        prompt_pool : Sequence of 1D LongTensor
            每次 iteration 从中**随机有放回**抽 ``group_size`` 个 prompt 做 rollout。
        n_iters : int
            多少个 outer iteration（每个 iteration 都重新 rollout）。
        group_size : int
            每个 iteration 采多少个 response（G）。
        verbose : bool
            是否打印每个 iter 的 metrics。

        Returns
        -------
        history : List[Dict[str, float]]
            每个 iter 的 metrics dict。
        """
        rng = random.Random(0)
        t0 = time_now()
        for it in range(n_iters):
            # 1) sample G prompts
            prompts = [rng.choice(prompt_pool) for _ in range(group_size)]

            # 2) rollout
            rollout = self.rollout_responses(prompts)

            # 3) PPO update
            stats = self.rlhf_update(rollout)
            self.history.append(stats)

            if verbose and (it % self.cfg.print_every == 0 or it == n_iters - 1):
                elapsed = time_now() - t0
                es = "ES" if stats["early_stopped"] else "  "
                print(
                    f"iter {it:>3} | reward={stats['mean_reward']:+.3f} | "
                    f"KL(ref)={stats['mean_kl_to_ref']:+.3f} | "
                    f"len={stats['mean_response_len']:.1f} | "
                    f"actor={stats['actor_loss']:+.4f} | "
                    f"critic={stats['critic_loss']:.3f} | "
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
    "ValueHead",
    "RLHFConfig",
    "RLHFTrainer",
    "time_now",
]
