r"""PRM (Process Reward Model) —— Ch16 核心基础设施。

本章是 Phase 4 的**起点**。Ch15 §15.6.3 列了 7 个开放研究方向，本章展开其中
第 5 条："**process reward vs outcome reward**——给中间推理步骤打分 (PRM)
vs 只看最终答案 (ORM) 哪个好？"

--------------------------------------------------------------------
为什么 PRM 重要？

- **OpenAI o1（2024.09）** 用 PRM 引导 step-by-step reasoning，
  GSM8K 从 70% → 95%+。Lightman et al. 2023 的 *Let's Verify Step by Step*
  公布了 **PRM800K**（80 万 step-level 标注），是 PRM 路线的奠基工作。
- **DeepSeek-R1**（2025）的 reasoning-RL 阶段也在跟进 step-level reward。
- 与 Ch11-13 的 ORM 形成清晰对照：ORM 只给最终 response 打一个标量，
  PRM 给每个推理 step 打一个标量——credit assignment 更精细。

--------------------------------------------------------------------
与现有模块的关系（**不重复造轮子**）

- 复用 :class:`rlenvs.tiny_gpt.TinyGPT` 作 backbone（同 Ch11 RewardModel 思路）
- 复用 :class:`utils.reward_model.RewardModel` 当作 ORM（§16.1 对照实验）
- 复用 :class:`utils.grpo.GRPOTrainer` 做 PRM-GRPO（§16.5）
- 复用 :func:`utils.reward_model.pad_to_length` 工具

--------------------------------------------------------------------
模块组成

- :class:`ProcessRewardModel` —— TinyGPT backbone + step-level reward head
- :func:`step_level_loss` —— 每个 step 一个分类 loss（good/bad/neutral）
- :func:`prm_best_of_n` —— 用 PRM 对 N 条候选打分，返回分数最高的
- :func:`orm_best_of_n`   —— 对照组：用 ORM 做 Best-of-N（同接口）
- :func:`make_two_step_addition_dataset` —— 简化多步推理任务（两步加法）

forward 约定（与 RewardModel 不同！）::

    RewardModel.forward(prompt_ids, response_ids) -> [B]            # 一个标量
    ProcessRewardModel.forward(prefix_ids)        -> [B, T]         # 每个 token 位置一个 step reward

我们这里把"step"定义在每个 token 上（最细粒度）。一个真正的 step 可以由
task-specific 的规则（如换行符 / CoT sentence boundary）把若干 token 聚合
成一个 step——见 :func:`step_rewards_from_token_rewards`。

为了能直接接 :class:`GRPOTrainer` （它期望 reward_model.forward(prompts,
responses) -> [B]），我们提供 :meth:`ProcessRewardModel.sequence_reward`
返回整条 response 的累加 PRM reward（与 ORM 接口兼容）。
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reward_model import pad_to_length


# =============================================================================
# 1. ProcessRewardModel：TinyGPT backbone + step-level reward head
# =============================================================================
class ProcessRewardModel(nn.Module):
    r"""TinyGPT backbone + 一个 **token-level** reward head。

    结构::

        prefix (prompt + response 已生成的部分)
            ↓ TinyGPT token embedding + PE + N × TransformerBlock + final LayerNorm
            hidden states [B, T, d_model]   ← 通过 forward hook 抓 ``ln_final``
            ↓ reward head: LayerNorm → Linear(d_model → 1) → squeeze
            per-token reward [B, T]   ← 每个位置一个 step-level reward

    设计要点（与 :class:`RewardModel` 对比）：

    - **token-level 输出**：RewardModel 只取**最后一个** response token 的 hidden
      作为整个 response 的汇总，输出标量；PRM 取**每个**位置，输出 [B, T]。
    - **prefix 输入**：PRM 的 forward 输入是完整的 prefix（prompt + 已生成的
      部分 response），不像 RewardModel 分两段 (prompt, response)。
      原因：PRM 的目的是对**任意中间 prefix**打分，prompt/response 边界不重要。
    - **可识别性**：和 RewardModel 一样，PRM 训练用 pairwise / 分类 loss，
      reward 的**绝对值**无意义，只有相对排序/符号有意义。

    数学定义（§16.2）::

        r_PRM : (x, s_{\le i}) → ℝ

    我们把它实例化为 token-level：对 prefix $z = (x, s_{\le i})$ 的每个位置 $t$
    输出一个 reward $r_t$，可以解释为"在该位置结束当前 step 的好坏程度"。

    Parameters
    ----------
    backbone : TinyGPT 或同构 LLM
        要有 ``ln_final`` 子模块和 ``d_model`` 属性（同 :class:`RewardModel`）。
    d_model : Optional[int]
        显式指定 hidden dim（否则从 backbone 推断）。
    """

    def __init__(self, backbone: nn.Module, d_model: Optional[int] = None) -> None:
        super().__init__()
        self.backbone = backbone
        if d_model is None:
            d_model = getattr(backbone, "d_model", None)
            if d_model is None:
                raise ValueError("无法从 backbone 推断 d_model，请显式传入 d_model=")
        self.d_model = d_model

        # step reward head：LayerNorm + Linear（与 RewardModel 同配方）
        self.reward_ln = nn.LayerNorm(d_model)
        self.reward_head = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

        # 通过 forward hook 抓 ln_final 的输入（hidden state）
        # —— 同 RewardModel 的设计，不重写。
        self._hidden: Optional[torch.Tensor] = None
        target = self._find_ln_final(backbone)
        if target is None:
            raise ValueError(
                "backbone 上找不到 'ln_final' 模块；"
                "ProcessRewardModel 当前只支持 TinyGPT 风格的 backbone。"
            )
        target.register_forward_hook(self._hook_capture_hidden)

    @staticmethod
    def _find_ln_final(module: nn.Module) -> Optional[nn.Module]:
        if hasattr(module, "ln_final"):
            return module.ln_final
        for name, child in module.named_modules():
            if name == "ln_final":
                return child
        return None

    def _hook_capture_hidden(self, module, inputs, output) -> None:
        if isinstance(inputs, tuple) and len(inputs) > 0:
            self._hidden = inputs[0].detach()
        else:
            self._hidden = output.detach()

    # ---- 核心 forward：token-level reward ----
    def forward(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """对 prefix 的每个位置输出一个 step-level reward。

        Parameters
        ----------
        prefix_ids : LongTensor [B, T]
            prompt + response 已生成的部分（或纯 prompt 也行）。

        Returns
        -------
        rewards : FloatTensor [B, T]
            每个位置一个 reward 标量。
        """
        if prefix_ids.dim() == 1:
            prefix_ids = prefix_ids.unsqueeze(0)
        _ = self.backbone(prefix_ids)  # hook 会抓 hidden
        hidden = self._hidden
        if hidden is None:
            raise RuntimeError("forward hook 没抓到 hidden state")
        # token-level：每个位置都过 reward head
        rewards = self.reward_head(self.reward_ln(hidden)).squeeze(-1)  # [B, T]
        return rewards

    # ---- 兼容 ORM 接口（给 GRPOTrainer 用）----
    def sequence_reward(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        reduction: str = "sum",
        response_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对 (prompt, response) 算 PRM 累加 reward——与 ORM 接口兼容。

        把 prompt + response 拼成 prefix，过 forward 得到每个 token 的 reward，
        再对 **response 部分**做 reduction（默认 sum）。

        Parameters
        ----------
        prompt_ids : [B, T_p]
        response_ids : [B, T_r]
        reduction : "sum" | "mean"
            sum 给出 R(x, y) = sum_i r(s_i)（§16.2 公式）；
            mean 是平均 step reward。
        response_mask : Optional[FloatTensor [B, T_r]]
            标记 response 中真实（非 pad）位置；不传则全部视为真实。

        Returns
        -------
        rewards : FloatTensor [B]
            每条样本一个标量。
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)
        T_p = prompt_ids.size(1)
        T_r = response_ids.size(1)
        full = torch.cat([prompt_ids, response_ids], dim=1)  # [B, T_p + T_r]
        per_token = self.forward(full)  # [B, T_p + T_r]
        # 取 response 部分
        resp_rewards = per_token[:, T_p:]  # [B, T_r]
        if response_mask is None:
            response_mask = torch.ones_like(resp_rewards)
        else:
            response_mask = response_mask.to(resp_rewards.dtype)
        masked = resp_rewards * response_mask
        if reduction == "sum":
            return masked.sum(dim=-1)
        elif reduction == "mean":
            n = response_mask.sum(dim=-1).clamp(min=1.0)
            return masked.sum(dim=-1) / n
        else:
            raise ValueError(f"未知 reduction: {reduction}")

    # ---- 兼容 RewardModel.forward 的 ORM 风格接口 ----
    def forward_orm_style(self, prompt_ids, response_ids) -> torch.Tensor:
        """让 GRPOTrainer 可以无缝切换：把 PRM 当 ORM 用（sum reduction）。

        GRPOTrainer.compute_token_rewards 调用 ``reward_model(prompts, responses) -> [N]``，
        PRM 通过 sequence_reward 实现同样接口（累加版）。
        """
        return self.sequence_reward(prompt_ids, response_ids, reduction="sum")


# =============================================================================
# 2. Step-level loss：每个 step 一个分类 / pairwise loss
# =============================================================================
def step_level_loss(
    prm: ProcessRewardModel,
    prefix_ids: torch.Tensor,
    step_mask: torch.Tensor,
    step_labels: torch.Tensor,
    loss_type: str = "bce",
) -> torch.Tensor:
    r"""每个 step 一个 loss（pointwise 标注版）。

    数学（§16.3）::

        对 prefix 的每个标注位置 t（step_mask[t]=1）：
            label_t ∈ {0 (bad), 1 (good)}（或连续 [0, 1]）
        loss = BCE(σ(r_t), label_t)

    Lightman et al. 2023 用的是 **3 类分类**（good / bad / neutral），这里我们
    简化成二分类 BCE——足以展示 PRM 训练流程。

    Parameters
    ----------
    prm : ProcessRewardModel
    prefix_ids : LongTensor [B, T]
    step_mask : FloatTensor [B, T]
        1 = 这个位置有标注（要算 loss），0 = 不算。
    step_labels : FloatTensor [B, T]
        每个标注位置的 good (1) / bad (0) 标签。
    loss_type : "bce" | "margin"
        - "bce":    BCE with logits（good/bad 二分类）
        - "margin": max(0, margin - (r_good - r_bad))，对 good/bad 对做 hinge
                    （要求 step_labels 严格 {0, 1}）

    Returns
    -------
    loss : scalar tensor
    """
    per_token_rewards = prm(prefix_ids)  # [B, T]
    if loss_type == "bce":
        # BCE with logits：把 r_t 当 logit
        loss_per = F.binary_cross_entropy_with_logits(
            per_token_rewards, step_labels, reduction="none"
        )
        masked = loss_per * step_mask
        denom = step_mask.sum().clamp(min=1.0)
        return masked.sum() / denom
    elif loss_type == "margin":
        # hinge: good 应该比 bad 高出 margin
        margin = 1.0
        # 对每个样本内、step_mask=1 的位置，good - bad
        # 简化：所有 good 位置 reward 应 > margin，所有 bad 应 < -margin
        # 用 good_mask / bad_mask 分别算 hinge
        good_mask = step_mask * (step_labels > 0.5).float()
        bad_mask = step_mask * (step_labels < 0.5).float()
        loss_good = F.relu(margin - per_token_rewards) * good_mask
        loss_bad = F.relu(margin + per_token_rewards) * bad_mask
        denom = (good_mask + bad_mask).sum().clamp(min=1.0)
        return (loss_good.sum() + loss_bad.sum()) / denom
    else:
        raise ValueError(f"未知 loss_type: {loss_type}")


def step_rewards_from_token_rewards(
    token_rewards: torch.Tensor,
    step_boundaries: torch.Tensor,
) -> torch.Tensor:
    """把 token-level reward 按 step boundary 聚合成 step-level reward。

    Parameters
    ----------
    token_rewards : [B, T]
    step_boundaries : [B, T] (float / bool)
        1 表示该位置是某个 step 的**结束**（如换行符、CoT sentence 末尾）。

    Returns
    -------
    step_rewards : [B, T]
        与 token_rewards 同 shape，但在每个 step 的**结束位置**放上该 step 的聚合 reward
        （这里用 max——代表该 step 的"完成质量"），其他位置 0。
        （这只是聚合策略之一，教学用。）
    """
    # 简化版：直接在每个 boundary 位置放 token reward，便于教学
    out = token_rewards * step_boundaries.float()
    return out


# =============================================================================
# 3. Best-of-N：用 PRM / ORM 对 N 个候选打分，选最好的
# =============================================================================
@torch.no_grad()
def _sample_n_responses(
    actor: nn.Module,
    prompt_ids: torch.Tensor,
    n: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    forbidden_ids: Optional[set] = None,
) -> torch.Tensor:
    """对单个 prompt 采 n 个 response，返回 [n, T_p + T_r]。"""
    out_list = []
    p = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
    # 逐个采样（保持与 GRPOTrainer._sample_response 风格一致）
    for i in range(n):
        out = p
        for _ in range(max_new_tokens):
            logits = actor(out)[:, -1, :]
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
        out_list.append(out)
    # pad 到同长
    max_len = max(o.size(1) for o in out_list)
    padded = torch.full(
        (n, max_len), 0, dtype=out_list[0].dtype, device=out_list[0].device
    )
    for i, o in enumerate(out_list):
        padded[i, : o.size(1)] = o[0] if o.size(0) == 1 else o[i]
    return padded


@torch.no_grad()
def prm_best_of_n(
    prm: ProcessRewardModel,
    actor: nn.Module,
    prompt_ids: torch.Tensor,
    n: int,
    max_new_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    reduction: str = "sum",
) -> Dict[str, torch.Tensor]:
    """PRM Best-of-N：采 n 条 response，用 PRM 打分，返回分数最高的。

    数学（§16.4）::

        对每个候选 y_i：R_PRM(x, y_i) = Σ_t r_PRM(x, y_{i, ≤t}) · mask_t
        选 i* = argmax_i R_PRM

    Returns
    -------
    dict 含:
        - 'best_idx' : int  (0..n-1)
        - 'best_response' : LongTensor [T_r]
        - 'scores' : FloatTensor [n]   每个 candidate 的 PRM 分数
        - 'responses' : LongTensor [n, T_r]  所有候选（pad 后）
        - 'response_lens' : LongTensor [n]
    """
    prm.eval()
    actor.eval()
    p = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
    T_p = p.size(1)

    responses = _sample_n_responses(
        actor, prompt_ids, n, max_new_tokens,
        temperature=temperature, top_k=top_k,
    )  # [n, T_p + T_r]
    # 切出 response 部分
    full_lens = (responses != pad_id).sum(dim=-1).clamp(min=T_p)
    resp_lens = (full_lens - T_p).clamp(min=0)  # [n]
    T_r = int((full_lens - T_p).max().item())
    T_r = max(T_r, 1)
    resp = responses[:, T_p : T_p + T_r]  # [n, T_r]
    # PRM forward：prompt + response
    full = torch.cat([p.expand(n, -1), resp], dim=1)  # [n, T_p + T_r]
    per_token = prm(full)
    resp_rewards = per_token[:, T_p:]  # [n, T_r]
    mask = (resp != pad_id).float()
    if reduction == "sum":
        scores = (resp_rewards * mask).sum(dim=-1)
    elif reduction == "mean":
        scores = (resp_rewards * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    else:
        raise ValueError(f"未知 reduction: {reduction}")
    best_idx = int(scores.argmax().item())
    return dict(
        best_idx=best_idx,
        best_response=resp[best_idx],
        scores=scores,
        responses=resp,
        response_lens=resp_lens,
    )


@torch.no_grad()
def orm_best_of_n(
    orm: nn.Module,
    actor: nn.Module,
    prompt_ids: torch.Tensor,
    n: int,
    max_new_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """ORM Best-of-N：同接口的对照组。

    数学（§16.4）::

        R_ORM(x, y_i) = r_ORM(x, y_i)
        选 i* = argmax_i R_ORM

    与 :func:`prm_best_of_n` 的关键差异：ORM 只看**整条 response**，PRM 看**每个 step**。
    """
    orm.eval()
    actor.eval()
    p = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
    T_p = p.size(1)

    responses = _sample_n_responses(
        actor, prompt_ids, n, max_new_tokens,
        temperature=temperature, top_k=top_k,
    )
    full_lens = (responses != pad_id).sum(dim=-1).clamp(min=T_p)
    resp_lens = (full_lens - T_p).clamp(min=0)
    T_r = int((full_lens - T_p).max().item())
    T_r = max(T_r, 1)
    resp = responses[:, T_p : T_p + T_r]
    # ORM: forward(prompt, response) -> [n]
    scores = orm(p.expand(n, -1), resp)
    best_idx = int(scores.argmax().item())
    return dict(
        best_idx=best_idx,
        best_response=resp[best_idx],
        scores=scores,
        responses=resp,
        response_lens=resp_lens,
    )


# =============================================================================
# 4. 合成多步推理任务：两步加法
# =============================================================================
def make_two_step_addition_dataset(
    n_samples: int = 200,
    max_digit: int = 4,
    seed: int = 0,
) -> List[Dict]:
    r"""生成"两步加法"任务数据集（简化多步推理任务）。

    任务定义::

        输入 prompt:    "a+b+c="
                例:    "2+3+1="
        期望输出（两步）:
            step 1:  "2+3=5;"   (a+b)
            step 2:  "5+1=6"    ((a+b)+c)
            完整:    "2+3=5;5+1=6"

    每个 sample 包含：

    - prompt_ids / full_ids / response_ids
    - step1_correct / step2_correct / final_correct (bool)
    - step_boundaries (哪些 token 位置是 step 结束)

    训练 PRM 用 step1_correct / step2_correct 当标注（合成 step-level label）。

    Parameters
    ----------
    n_samples : int
    max_digit : int
        a, b, c ∈ [0, max_digit]
    seed : int
    """
    rng = random.Random(seed)
    samples: List[Dict] = []
    chars = set("0123456789+=;")
    for ch in "+=;":
        chars.add(ch)
    for d in range(max_digit + 1):
        chars.add(str(d))

    for _ in range(n_samples):
        a = rng.randint(0, max_digit)
        b = rng.randint(0, max_digit)
        c = rng.randint(0, max_digit)
        # 注意：教学简化，两步加法允许 a+b > 9（变成两位数），但 PRM 学起来稍难。
        # 为了让 TinyGPT (~20k 参数) 能学，限制 a+b ≤ 9 且 (a+b)+c ≤ 9
        if a + b > 9 or a + b + c > 9:
            continue
        prompt = f"{a}+{b}+{c}="
        s1 = a + b
        s2 = s1 + c
        # response 格式："<a>+<b>=<s1>;<s1>+<c>=<s2>"
        response = f"{a}+{b}={s1};{s1}+{c}={s2}"
        samples.append({
            "prompt": prompt,
            "response": response,
            "a": a, "b": b, "c": c,
            "step1_answer": s1,
            "step2_answer": s2,
            "step1_correct": True,
            "step2_correct": True,
            "final_correct": True,
            # step 边界：response 里 ';' 和 最后一个字符是 step 结束
            # 具体 boundary mask 由调用方根据 tokenizer 算（这里只存语义信息）
        })
    return samples


def encode_two_step_sample(
    sample: Dict,
    tokenizer,
) -> Dict:
    """把一条两步加法 sample 用 tokenizer 编码，并生成 step-level 标注。

    Returns
    -------
    dict 含:
        - prompt_ids : LongTensor [T_p]
        - response_ids : LongTensor [T_r]
        - full_ids : LongTensor [T_p + T_r]
        - step_mask : FloatTensor [T_p + T_r]  (哪些位置是 step 结束、要算 PRM loss)
        - step_labels : FloatTensor [T_p + T_r]  (每个 step 结束位置的 good(1)/bad(0) label)
        - step1_end_pos : int  (response 内 ';' 的位置)
        - step2_end_pos : int  (response 最后位置)
    """
    prompt = sample["prompt"]
    response = sample["response"]
    p_ids = tokenizer.encode(prompt)
    r_ids = tokenizer.encode(response)

    # response 里 ';' 是 step1 结束，最后一个字符是 step2 结束
    # 找 ';' 在 r_ids 里的位置
    semi_char_id = tokenizer.stoi.get(";", None)
    step1_end = None
    if semi_char_id is not None:
        for i, tid in enumerate(r_ids.tolist()):
            if tid == semi_char_id:
                step1_end = i
                break
    if step1_end is None:
        step1_end = max(len(r_ids) // 2, 0)
    step2_end = len(r_ids) - 1

    # 构建 step_mask 和 step_labels（在 full sequence 上）
    T_p = len(p_ids)
    T_r = len(r_ids)
    full = torch.cat([p_ids, r_ids])
    step_mask = torch.zeros(T_p + T_r)
    step_labels = torch.zeros(T_p + T_r)

    # step1 结束位置（response 内 step1_end，对应 full 内 T_p + step1_end）
    pos1 = T_p + step1_end
    pos2 = T_p + step2_end
    if 0 <= pos1 < T_p + T_r:
        step_mask[pos1] = 1.0
        step_labels[pos1] = 1.0 if sample.get("step1_correct", True) else 0.0
    if 0 <= pos2 < T_p + T_r:
        step_mask[pos2] = 1.0
        step_labels[pos2] = 1.0 if sample.get("step2_correct", True) else 0.0

    return {
        "prompt_ids": p_ids,
        "response_ids": r_ids,
        "full_ids": full,
        "step_mask": step_mask,
        "step_labels": step_labels,
        "step1_end_pos": step1_end,
        "step2_end_pos": step2_end,
    }


def make_wrong_step_variations(
    sample: Dict,
    n_wrong: int = 3,
    seed: int = 0,
) -> List[Dict]:
    """对一个正确 sample 生成若干"中间 step 错"的变体（用于训练 PRM 区分好坏 step）。

    策略：
    - step1 wrong: 把 s1 改成错误值
    - step2 wrong: s1 对，s2 改成错误值
    - both wrong

    Returns list of dict（结构同 :func:`make_two_step_addition_dataset` 的 sample，
    但 step1_correct / step2_correct 反映真实错误）。
    """
    rng = random.Random(seed)
    a, b, c = sample["a"], sample["b"], sample["c"]
    s1_true = a + b
    s2_true = s1_true + c
    variants = []

    def wrong_value(true_val, lo=0, hi=9):
        candidates = [v for v in range(lo, hi + 1) if v != true_val]
        return rng.choice(candidates) if candidates else true_val

    # step1 wrong
    for _ in range(n_wrong):
        s1 = wrong_value(s1_true)
        # 如果 step1 错了，step2 通常也错（除非巧合）
        s2 = s1 + c
        variants.append({
            "prompt": sample["prompt"],
            "response": f"{a}+{b}={s1};{s1}+{c}={s2}",
            "a": a, "b": b, "c": c,
            "step1_answer": s1,
            "step2_answer": s2,
            "step1_correct": False,
            "step2_correct": False,  # 跟着错
            "final_correct": False,
        })

    # step2 wrong (step1 对)
    for _ in range(n_wrong):
        s2 = wrong_value(s2_true)
        variants.append({
            "prompt": sample["prompt"],
            "response": f"{a}+{b}={s1_true};{s1_true}+{c}={s2}",
            "a": a, "b": b, "c": c,
            "step1_answer": s1_true,
            "step2_answer": s2,
            "step1_correct": True,
            "step2_correct": False,
            "final_correct": False,
        })

    return variants


# =============================================================================
# 5. 评估：从 response 解析出 step 答案，判断对错
# =============================================================================
def parse_two_step_response(
    prompt: str,
    response: str,
) -> Dict:
    """从 response 字符串解析出 step1/step2/final 答案，返回对错信息。

    expected response 格式: "<a>+<b>=<s1>;<s1>+<c>=<s2>"
    但实际可能不完整或错误——这个函数尽量解析。

    Returns
    -------
    dict 含:
        - parsed : bool  (能否解析出两个 step)
        - step1_str / step2_str : str
        - step1_correct / step2_correct / final_correct : bool
    """
    # 从 prompt 提取 a, b, c
    try:
        body = prompt.rstrip("=").strip()
        parts = body.split("+")
        a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return dict(parsed=False, step1_str="", step2_str="",
                    step1_correct=False, step2_correct=False, final_correct=False)

    # 尝试按 ';' 切分 response
    if ";" in response:
        s1_part, s2_part = response.split(";", 1)
    else:
        s1_part = response
        s2_part = ""

    # 从 s1_part 找最后一个 '=' 后面的数字
    def extract_answer(s):
        if "=" not in s:
            return None
        after = s.rsplit("=", 1)[-1]
        after = after.strip()
        # 取开头连续数字
        num = ""
        for ch in after:
            if ch.isdigit():
                num += ch
            else:
                break
        try:
            return int(num) if num else None
        except Exception:
            return None

    s1_got = extract_answer(s1_part)
    s2_got = extract_answer(s2_part)

    s1_true = a + b
    s2_true = s1_true + c

    step1_correct = (s1_got == s1_true) if s1_got is not None else False
    step2_correct = (s2_got == s2_true) if s2_got is not None else False
    final_correct = step1_correct and step2_correct

    return dict(
        parsed=True,
        step1_str=str(s1_got),
        step2_str=str(s2_got),
        step1_correct=step1_correct,
        step2_correct=step2_correct,
        final_correct=final_correct,
        s1_got=s1_got, s2_got=s2_got,
        s1_true=s1_true, s2_true=s2_true,
    )


def evaluate_two_step_accuracy(
    actor: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 12,
    temperature: float = 0.0,
    greedy: bool = True,
) -> Dict:
    """对一组 prompt 用 actor 贪心解码，统计 step1/step2/final 准确率。"""
    actor.eval()
    n_step1, n_step2, n_final = 0, 0, 0
    n_total = len(prompts)
    details = []
    with torch.no_grad():
        for prompt in prompts:
            p_ids = tokenizer.encode(prompt).unsqueeze(0)
            # 贪心解码：优先用 rlenvs.tiny_gpt.generate（它需要 actor.max_seq_len）。
            # 如果 actor 是 wrapper（没有 max_seq_len），手动展开 forward。
            try:
                _ = actor.max_seq_len
                from rlenvs.tiny_gpt import generate as tg_generate
                out = tg_generate(actor, p_ids, max_new_tokens=max_new_tokens,
                                  temperature=temperature, greedy=greedy)
            except AttributeError:
                # actor 是 wrapper，手动 greedy
                out = p_ids
                for _ in range(max_new_tokens):
                    cond = out if out.size(1) <= 128 else out[:, -128:]
                    logits = actor(cond)[:, -1, :]
                    if greedy or temperature == 0:
                        next_id = logits.argmax(dim=-1, keepdim=True)
                    else:
                        logits = logits / max(temperature, 1e-5)
                        probs = F.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1)
                    out = torch.cat([out, next_id], dim=1)
            resp_ids = out[0, p_ids.size(1):]
            response = tokenizer.decode(resp_ids.tolist())
            info = parse_two_step_response(prompt, response)
            n_step1 += int(info["step1_correct"])
            n_step2 += int(info["step2_correct"])
            n_final += int(info["final_correct"])
            details.append({"prompt": prompt, "response": response, **info})
    return dict(
        step1_acc=n_step1 / max(1, n_total),
        step2_acc=n_step2 / max(1, n_total),
        final_acc=n_final / max(1, n_total),
        n_total=n_total,
        details=details,
    )


__all__ = [
    "ProcessRewardModel",
    "step_level_loss",
    "step_rewards_from_token_rewards",
    "prm_best_of_n",
    "orm_best_of_n",
    "make_two_step_addition_dataset",
    "encode_two_step_sample",
    "make_wrong_step_variations",
    "parse_two_step_response",
    "evaluate_two_step_accuracy",
]
