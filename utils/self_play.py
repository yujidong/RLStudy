r"""Self-Play + Constitutional AI / RLAIF —— Ch17 核心基础设施。

本章是 Phase 4 第二章。Ch15 §15.6.3 列了 7 个开放研究方向，本章合并展开
其中第 2、3 条：

    **方向 2**：self-play vs human data（AlphaZero 在棋类成功，LLM 能复制吗？）
    **方向 3**：constitutional AI / RLAIF（用大模型当"人类"打分）

两条路径的共同动机是 **减少 RLHF 对人类标注的依赖**：

- **RLHF 的标注 bottleneck**（Ch11）：人类偏好标注贵（~$20/条）、慢（专家限定）、
  有偏（标注者偏好 ≠ 真实用户）、难 scale（训练 100B 模型需要百万级偏好对）。
- **Self-Play**（AlphaZero / SPIN / Self-Rewarding LM）：agent 自己生成数据训自己。
- **Constitutional AI / RLAIF**（Bai 2022 / Lee 2023）：用一个 LLM 当"裁判"，
  替代人类给 response 打分。

--------------------------------------------------------------------
与现有模块的关系（**不重复造轮子**）

- 复用 :class:`rlenvs.tiny_gpt.TinyGPT` 作 actor / judge backbone（同 Ch11/16 思路）
- 复用 :class:`utils.reward_model.RewardModel` 当作 RLAIF 训出的 RM
  （数据换成 AI 偏好对，loss 还是 Bradley-Terry）
- 复用 :func:`utils.reward_model.bradley_terry_loss` / :func:`make_preference_batch`
  / :func:`pad_to_length`
- 复用 :class:`utils.grpo.GRPOTrainer` 做 RLAIF-GRPO（reward 信号换成 AI judge）

--------------------------------------------------------------------
模块组成

- :class:`AIJudge` —— 用 LLM + judge prompt 给 response 打分（核心 RLAIF 组件）
- :class:`Constitution` —— 一组原则（helpful / harmless / honest 等）的容器
- :func:`spin_objective` —— SPIN（Self-Play fIne-tuNing）的分类器目标
- :func:`generate_ai_preferences` —— 用 AI judge 生成偏好对（替代 Ch11 的人类标注）
- :func:`self_reward_score` —— Self-Rewarding LM 风格：LLM 给自己的 response 评分

forward 约定（与 Ch11 RewardModel 兼容）::

    AIJudge.forward(prompt_ids, response_ids) -> FloatTensor [B]
    RewardModel.forward(prompt_ids, response_ids) -> FloatTensor [B]

两者可以**无缝互换**——这就是 RLAIF 的本质：把 Ch11 的 RM 换成一个 AI judge。
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
# 1. Constitution：一组 constitutional principles
# =============================================================================
class Constitution:
    r"""一组 constitutional principles（"宪法"）。

    Constitutional AI（Bai et al. 2022）的核心思想：给 AI 一组**文本原则**，
    让 AI 按这些原则自评。每条原则是一个自然语言描述，例如：

    - "Please identify ways in which the response is **helpful**."
    - "Please identify ways in which the response is **harmless**."
    - "Please identify ways in which the response is **honest**."

    本章简化：每条原则用一个 **judge prompt template** 表示。template 里有一个
    ``{response}`` 占位符，运行时填入具体 response。

    数学上，每条原则 $c_k$ 定义了一个**评估函数**：

    $$r_{AI}^{(k)}(x, y) = \text{LLM}_{\text{judge}}(P_{\text{judge}}^{(k)}(x, y))$$

    其中 $P_{\text{judge}}^{(k)}$ 是原则 $c_k$ 的 judge prompt。
    最终 reward 可以是单原则或加权多原则：

    $$r_{AI}(x, y) = \sum_k w_k \cdot r_{AI}^{(k)}(x, y)$$

    Parameters
    ----------
    principles : List[Dict]
        每个元素是 ``{"name": str, "description": str, "weight": float}``。
        ``description`` 描述这条原则要求什么（如 "the response should be helpful"）。
        ``weight`` 用于加权多原则 reward（默认 1.0）。
    """

    DEFAULT_PRINCIPLES: List[Dict] = [
        {
            "name": "helpful",
            "description": "The response should directly answer the question and be useful.",
            "weight": 1.0,
        },
        {
            "name": "harmless",
            "description": "The response should not contain harmful, toxic, or dangerous content.",
            "weight": 1.0,
        },
        {
            "name": "honest",
            "description": "The response should be truthful and not mislead.",
            "weight": 1.0,
        },
    ]

    def __init__(self, principles: Optional[List[Dict]] = None) -> None:
        if principles is None:
            principles = [dict(p) for p in self.DEFAULT_PRINCIPLES]
        self.principles = principles
        # 校验
        for p in self.principles:
            assert "name" in p and "description" in p, (
                "每条 principle 必须有 'name' 和 'description'"
            )
            p.setdefault("weight", 1.0)

    def __len__(self) -> int:
        return len(self.principles)

    def __iter__(self):
        return iter(self.principles)

    def names(self) -> List[str]:
        return [p["name"] for p in self.principles]

    def weights(self) -> torch.Tensor:
        return torch.tensor([p["weight"] for p in self.principles], dtype=torch.float32)

    def make_judge_prompt(self, response: str, principle_idx: int = 0) -> str:
        """构造一个自然语言 judge prompt（用于真正的 LLM judge）。

        实际 LLM judge 会用 ``LLM(judge_prompt)`` 输出一个评分；本章的
        :class:`AIJudge` 用 token-level 启发式模拟这个过程（见下方）。
        """
        p = self.principles[principle_idx]
        return (
            f"Rate the following response on the '{p['name']}' principle "
            f"(1-5 scale, 5=best). Principle: {p['description']}\n"
            f"Response: {response}\nRating:"
        )


# =============================================================================
# 2. AIJudge：用 LLM + judge prompt 给 response 打分（RLAIF 核心）
# =============================================================================
class AIJudge(nn.Module):
    r"""用一个 LLM（TinyGPT）+ judge prompt 给 response 打分。

    这是 **RLAIF (Lee et al. 2023) / Constitutional AI (Bai et al. 2022)** 的核心组件：
    把 Ch11 的 "人类标注偏好" 换成 "AI judge 打分"。

    ----------------------------------------------------------------
    实现说明（教学简化）

    真实的 RLAIF pipeline 是：

    1. 用 ``LLM`` 生成一个自然语言 critique（如 "This response is helpful
       because..."）
    2. 从 critique 里提取一个标量 reward（如解析 "Rating: 4/5" → 4.0）

    本章简化：不显式生成 critique，而是用一个 **token-level 启发式** 让 TinyGPT
    输出一个标量 reward。具体做法：

    - 把 ``(prompt, response)`` 拼成 prefix
    - 过 TinyGPT backbone 抓 ``ln_final`` 的 hidden state ``[B, T, d_model]``
    - 取**最后一个 response token** 的 hidden vector（同 Ch11 RewardModel）
    - 经 reward head（LayerNorm + Linear）输出标量 reward

    与 Ch11 :class:`RewardModel` 的**唯一区别**：

    - ``RewardModel`` 是**训练出来的**（从人类偏好对学）
    - ``AIJudge`` 是**直接构造的**（用 constitution 的原则"定义"reward，
      不需要人类标注）—— 这是 RLAIF 的灵魂

    为了让两者可以**直接对比**，``AIJudge.forward`` 的接口与 ``RewardModel.forward``
    完全一致：``forward(prompt_ids, response_ids) -> [B]``。

    ----------------------------------------------------------------
    RLAIF 的 bias 问题（§17.4）

    AI judge 有几个已知 bias（Lee 2023 / Zheng 2023 实证）：

    1. **长度偏好**：倾向给长 response 高分（"more is better" 错觉）
    2. **自我偏好**：倾向给自己风格的 response 高分
    3. **位置偏好**：在两个 response 对比时，倾向选先出现的

    本章的简化 AIJudge 也继承了这些 bias（特别是长度偏好）——我们会在 §17.5
    的实验里**量化** AI judge 评分与"真实"reward 的偏差。

    Parameters
    ----------
    backbone : TinyGPT-like
        要有 ``ln_final`` 子模块和 ``d_model`` 属性（同 RewardModel）。
    constitution : Optional[Constitution]
        这条 judge 遵循哪条原则。``None`` 表示用默认 helpful/harmless/honest。
        注意：本章简化版 AIJudge 的 reward 由 backbone 的 hidden state 决定，
        constitution 只影响**输出的解释**（judge prompt 文本），不直接进网络。
    d_model : Optional[int]
        显式指定 hidden dim。
    length_bias : float
        长度偏好系数。``> 0`` 表示给长 response 加分（模拟已知的 judge bias）。
        ``= 0`` 关闭长度偏好。教学用，方便我们对比"AI judge vs 人类"。
    """

    def __init__(
        self,
        backbone: nn.Module,
        constitution: Optional[Constitution] = None,
        d_model: Optional[int] = None,
        length_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if d_model is None:
            d_model = getattr(backbone, "d_model", None)
            if d_model is None:
                raise ValueError("无法从 backbone 推断 d_model，请显式传入 d_model=")
        self.d_model = d_model
        self.constitution = constitution if constitution is not None else Constitution()
        self.length_bias = float(length_bias)

        # reward head：LayerNorm + Linear（与 RewardModel 完全一致）
        self.reward_ln = nn.LayerNorm(d_model)
        self.reward_head = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

        # 通过 forward hook 抓 ln_final 的输入（hidden state）
        self._hidden: Optional[torch.Tensor] = None
        target = self._find_ln_final(backbone)
        if target is None:
            raise ValueError(
                "backbone 上找不到 'ln_final' 模块；"
                "AIJudge 当前只支持 TinyGPT 风格的 backbone。"
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

    def forward(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""对每条 (prompt, response) 算一个 AI judge reward。

        Parameters
        ----------
        prompt_ids : LongTensor [B, T_p]
        response_ids : LongTensor [B, T_r]
        response_mask : Optional[FloatTensor [B, T_r]]
            标记真实（非 pad）位置，用于算 length bias。

        Returns
        -------
        rewards : FloatTensor [B]
            每条样本一个标量 AI judge reward。
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [B, T]
        _ = self.backbone(full_ids)
        hidden = self._hidden
        if hidden is None:
            raise RuntimeError("forward hook 没抓到 hidden state")
        # 取最后一个 token 的 hidden vector（同 RewardModel）
        last_hidden = hidden[:, -1, :]  # [B, d_model]
        reward = self.reward_head(self.reward_ln(last_hidden)).squeeze(-1)  # [B]

        # 长度偏好（可选）：模拟 judge 的已知 bias
        if self.length_bias != 0.0 and response_mask is not None:
            lengths = response_mask.float().sum(dim=-1)  # [B]
            reward = reward + self.length_bias * lengths
        elif self.length_bias != 0.0:
            # 没有 mask 就用 T_r（粗略）
            reward = reward + self.length_bias * float(response_ids.size(1))

        return reward


# =============================================================================
# 3. generate_ai_preferences：用 AI judge 生成偏好对（替代人类标注）
# =============================================================================
@torch.no_grad()
def generate_ai_preferences(
    actor: nn.Module,
    judge: AIJudge,
    tokenizer,
    prompts: Sequence[str],
    n_per_prompt: int = 2,
    max_new_tokens: int = 12,
    temperature: float = 1.0,
    pad_id: Optional[int] = None,
    seed: int = 0,
) -> List[Dict]:
    r"""用 AI judge 给 actor 采的若干 response 打分，组装成 pairwise 偏好对。

    这是 **RLAIF 的数据生成 pipeline**：

    1. 对每个 prompt $x$，用 ``actor`` 采 ``n_per_prompt`` 个 response $\{y_1, \dots, y_n\}$
    2. 用 ``judge`` 给每个 response 打分 $r_{AI}(x, y_i)$
    3. 对每个 prompt，构造 ``n*(n-1)/2`` 个偏好对：
       对于 $(y_i, y_j)$，若 $r_i > r_j$ 则 $(y_i, y_j)$ 是 (winner, loser) 对

    生成的偏好对格式与 :func:`utils.reward_model.generate_preference_data` **完全一致**，
    可以直接喂给 :func:`bradley_terry_loss` 训 RM。

    与 Ch11 的对比：

    =========================  =========================================
    Ch11 (人类标注)             本章 (AI judge)
    =========================  =========================================
    数据来源                   人类标注 winner/loser
    成本                       ~$20/条，慢、有偏
    scale                      难（受标注者人数限制）
    bias                       标注者偏好 ≠ 用户偏好
    =========================  =========================================

    Parameters
    ----------
    actor : nn.Module
        用于采 response 的 actor（TinyGPT wrapper）。
    judge : AIJudge
        评分器。
    tokenizer : CharTokenizer
    prompts : Sequence[str]
    n_per_prompt : int
        每个 prompt 采多少个 response。
    max_new_tokens : int
    temperature : float
    pad_id : Optional[int]
    seed : int

    Returns
    -------
    list of dict，每个 dict 含:
        - 'prompt' / 'winner' / 'loser' : str
        - 'prompt_ids' / 'winner_ids' / 'loser_ids' : LongTensor [T]
        - 'r_w' / 'r_l' / 'r_diff' : float（AI judge 的评分）
        - 'source' : 'ai_judge'
    """
    if pad_id is None:
        pad_id = tokenizer.pad_id
    actor.eval()
    judge.eval()
    rng = random.Random(seed)
    torch.manual_seed(seed)

    pairs: List[Dict] = []
    for prompt_str in prompts:
        prompt_ids = tokenizer.encode(prompt_str)
        p_b = prompt_ids.unsqueeze(0)  # [1, T_p]
        # 采 n 个 response
        responses_ids: List[torch.Tensor] = []
        responses_str: List[str] = []
        for _ in range(n_per_prompt):
            out = _sample_response(
                actor, p_b, max_new_tokens=max_new_tokens,
                temperature=temperature, forbidden_ids={pad_id},
            )
            resp_ids = out[0, p_b.size(1):]  # [T_r]
            responses_ids.append(resp_ids)
            responses_str.append(tokenizer.decode([t for t in resp_ids.tolist() if t != pad_id]))
        # 给每个 response 打分
        T_p = p_b.size(1)
        T_r_max = max(int(r.size(0)) for r in responses_ids)
        T_r_max = max(T_r_max, 1)
        resp_batch = torch.full((len(responses_ids), T_r_max), pad_id, dtype=torch.long)
        resp_mask = torch.zeros((len(responses_ids), T_r_max), dtype=torch.float32)
        prompt_batch = p_b.expand(len(responses_ids), -1).contiguous()
        for i, r in enumerate(responses_ids):
            L = min(int(r.size(0)), T_r_max)
            resp_batch[i, :L] = r[:L]
            resp_mask[i, :L] = 1.0
        scores = judge(prompt_batch, resp_batch, response_mask=resp_mask)  # [n]
        # 构造 pairwise 偏好对
        for i in range(len(responses_ids)):
            for j in range(len(responses_ids)):
                if i == j:
                    continue
                r_i, r_j = float(scores[i].item()), float(scores[j].item())
                if r_i <= r_j:
                    continue
                pairs.append({
                    "prompt": prompt_str,
                    "winner": responses_str[i],
                    "loser": responses_str[j],
                    "prompt_ids": prompt_ids,
                    "winner_ids": responses_ids[i],
                    "loser_ids": responses_ids[j],
                    "r_w": r_i,
                    "r_l": r_j,
                    "r_diff": r_i - r_j,
                    "source": "ai_judge",
                })
    rng.shuffle(pairs)
    return pairs


def _sample_response(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    forbidden_ids: Optional[set] = None,
) -> torch.Tensor:
    """自回归采样（与 GRPOTrainer._sample_response 风格一致）。"""
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


# =============================================================================
# 4. spin_objective：SPIN（Self-Play fIne-tuNing）的分类器目标
# =============================================================================
def spin_objective(
    classifier: nn.Module,
    prompt_ids: torch.Tensor,
    real_response_ids: torch.Tensor,
    fake_response_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""SPIN（Singh et al. 2023, Self-Play fIne-tuNing）的分类器目标。

    ----------------------------------------------------------------
    SPIN 的核心洞察（§17.3）

    当前 LLM $\pi_\theta$ 生成的 response 分布 $\pi_\theta(y|x)$ 与"真人类"分布
    $\pi_{human}(y|x)$ **不同**。如果能训练一个分类器 $f_\phi(x, y)$ 区分
    "真人类 response" vs "agent 自己生成的 response"，并用它做 reward，
    就能让 $\pi_\theta \to \pi_{human}$。

    ----------------------------------------------------------------
    数学（§17.3 形式化）

    SPIN 的分类器 $f_\phi$ 用 logistic loss 训练：

    $$
    \max_\phi \;
    \underbrace{\mathbb{E}_{y \sim \pi_{human}}[\log \sigma(f_\phi(x, y))]}_{\text{real should be high}}
    + \underbrace{\mathbb{E}_{y \sim \pi_\theta}[\log(1 - \sigma(f_\phi(x, y)))]}_{\text{fake should be low}}
    $$

    等价于最小化：

    $$
    \mathcal{L}_{SPIN}(\phi) =
    -\mathbb{E}_{y_{real}}[\log \sigma(f_\phi(x, y_{real}))]
    - \mathbb{E}_{y_{fake}}[\log(1 - \sigma(f_\phi(x, y_{fake})))]
    $$

    用 ``F.softplus`` 数值稳定实现：

    - $\log \sigma(f) = -\text{softplus}(-f)$
    - $\log(1 - \sigma(f)) = -\text{softplus}(f)$

    ----------------------------------------------------------------
    收敛性（§17.3）

    **当 $\pi_\theta = \pi_{human}$ 时**，分类器无法区分两者 → $f_\phi \to 0$ →
    $\sigma(f_\phi) \to 0.5$ → reward 均匀 → 训练停止。这是 SPIN 的自然停止条件
    （类似 GAN 的 Nash 均衡）。

    ----------------------------------------------------------------
    与 GAN / DPO 的关系

    - **GAN**：discriminator 区分 real vs fake，generator 骗 discriminator
    - **SPIN**：classifier 区分 human vs $\pi_\theta$，$\pi_\theta$ 模仿 human
    - **DPO**（Ch14）：用偏好对训 reward = log-ratio，避免显式 RL
    - SPIN 与 DPO 的共同点：都把 RLHF 重写成分类问题；区别是 SPIN 用
      self-play 迭代（每次 $\pi_\theta$ 更新后重新生成 fake 样本）

    ----------------------------------------------------------------
    Parameters
    ----------
    classifier : nn.Module
        分类器，``forward(prompt_ids, response_ids) -> [B]`` 给 logits。
        可以是 RewardModel 或 AIJudge。
    prompt_ids : LongTensor [B, T_p]
    real_response_ids : LongTensor [B, T_r]   来自 human / SFT data
    fake_response_ids : LongTensor [B, T_r]   来自 $\pi_\theta$（当前 actor）

    Returns
    -------
    loss : scalar tensor
        SPIN 分类器 loss（要 minimize）。
    stats : Dict[str, float]
        - 'real_acc' : 分类器把 real 分对的比例（$\sigma(f_{real}) > 0.5$）
        - 'fake_acc' : 分类器把 fake 分对的比例（$\sigma(f_{fake}) < 0.5$）
        - 'real_logit_mean' / 'fake_logit_mean' : 分类器输出的均值
    """
    f_real = classifier(prompt_ids, real_response_ids)  # [B]
    f_fake = classifier(prompt_ids, fake_response_ids)  # [B]

    # -log sigma(f_real) = softplus(-f_real)
    # -log(1 - sigma(f_fake)) = softplus(f_fake)
    loss_real = F.softplus(-f_real).mean()
    loss_fake = F.softplus(f_fake).mean()
    loss = loss_real + loss_fake

    with torch.no_grad():
        real_acc = float((f_real > 0).float().mean().item())
        fake_acc = float((f_fake < 0).float().mean().item())
        stats = {
            "real_acc": real_acc,
            "fake_acc": fake_acc,
            "real_logit_mean": float(f_real.mean().item()),
            "fake_logit_mean": float(f_fake.mean().item()),
            "loss_real": float(loss_real.item()),
            "loss_fake": float(loss_fake.item()),
        }
    return loss, stats


# =============================================================================
# 5. self_reward_score：Self-Rewarding LM（Yuan et al. 2024）
# =============================================================================
@torch.no_grad()
def self_reward_score(
    actor: nn.Module,
    reward_head: nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    r"""Self-Rewarding LM（Yuan et al. 2024）：让 LLM 自己给自己的 response 打分。

    与 AIJudge / RLAIF 的区别：

    - **AIJudge**：用一个**独立的** judge LLM（即使 backbone 相同，judge head 是
      独立训练的或用 constitution 启发式定义的）
    - **Self-Rewarding LM**：让 **同一个** LLM（同一个 forward）输出 reward——
      actor 既负责生成 response，又负责评 response

    Yuan 2024 的做法：在 actor 上加一个 "Judge Head"（与 LM head 并列），
    用一个特殊的 "evaluate" prompt 让模型输出 "Rating: X/5" 这样的 token，
    解析成标量 reward。

    本章简化：直接复用 actor 的 hidden state + 一个独立的 reward_head
    （reward_head 的参数**与 actor 共享 backbone**，但 head 本身独立）。
    这模拟了"actor 自己评自己"——核心是 backbone 共享。

    Parameters
    ----------
    actor : nn.Module
        生成 response 的模型（TinyGPT）。
    reward_head : nn.Module
        把 actor 的 hidden state 压成标量 reward（如 LayerNorm + Linear）。
        通常与 actor 共享 backbone（head 独立）。
    prompt_ids : LongTensor [B, T_p]
    response_ids : LongTensor [B, T_r]

    Returns
    -------
    rewards : FloatTensor [B]
    """
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    full = torch.cat([prompt_ids, response_ids], dim=1)
    # 调 actor forward（reward_head 负责从 hidden state 出 reward）
    # 这里我们假设 reward_head 内部会调 actor.backbone 抓 hidden state；
    # 但为了通用性，我们提供一个 hook-based 的 fallback：
    hidden = getattr(actor, "_last_hidden", None)
    if hidden is None:
        # fallback：直接调 actor，期望它通过 hook 设置 _last_hidden
        _ = actor(full)
        hidden = getattr(actor, "_last_hidden", None)
    if hidden is None:
        raise RuntimeError(
            "self_reward_score 需要 actor 暴露 _last_hidden 属性 "
            "（通过 forward hook 设置）；建议用 AIJudge 包装。"
        )
    last_hidden = hidden[:, -1, :]  # [B, d_model]
    reward = reward_head(last_hidden).squeeze(-1)  # [B]
    return reward


# =============================================================================
# 6. spin_iteration：一轮 SPIN（生成 fake + 训分类器）
# =============================================================================
def spin_iteration(
    classifier: nn.Module,
    actor: nn.Module,
    real_samples: Sequence[Dict],
    tokenizer,
    optimizer: torch.optim.Optimizer,
    max_new_tokens: int = 12,
    temperature: float = 1.0,
    pad_id: Optional[int] = None,
    batch_size: int = 16,
    seed: int = 0,
) -> Dict[str, float]:
    """一轮 SPIN 迭代：(1) actor 生成 fake response  (2) 用 spin_objective 训分类器。

    Parameters
    ----------
    classifier : nn.Module
        SPIN 分类器（要训练）。
    actor : nn.Module
        当前 actor（生成 fake response；这轮不更新）。
    real_samples : Sequence[Dict]
        每个元素含 ``'prompt_ids'`` 和 ``'response_ids'``（real / human response）。
    tokenizer
    optimizer : 优化 classifier 的 optimizer
    batch_size : int

    Returns
    -------
    stats : Dict[str, float]
        SPIN 训练的统计信息（loss / real_acc / fake_acc）。
    """
    if pad_id is None:
        pad_id = tokenizer.pad_id
    classifier.train()
    actor.eval()
    torch.manual_seed(seed)

    # 1) 用 actor 为每个 real_sample 生成 fake response
    fake_data: List[Dict] = []
    with torch.no_grad():
        for s in real_samples:
            p_ids = s["prompt_ids"]
            p_b = p_ids.unsqueeze(0)
            out = _sample_response(
                actor, p_b, max_new_tokens=max_new_tokens,
                temperature=temperature, forbidden_ids={pad_id},
            )
            fake_ids = out[0, p_ids.size(0):]
            fake_data.append({
                "prompt_ids": p_ids,
                "real_ids": s["response_ids"],
                "fake_ids": fake_ids,
            })

    # 2) 切 batch 训分类器
    n = len(fake_data)
    epoch_loss = 0.0
    epoch_real_acc = 0.0
    epoch_fake_acc = 0.0
    n_batches = 0
    indices = torch.randperm(n).tolist()
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = [fake_data[i] for i in batch_idx]
        p_batch = pad_to_length([b["prompt_ids"] for b in batch], pad_id)
        real_batch = pad_to_length([b["real_ids"] for b in batch], pad_id)
        fake_batch = pad_to_length([b["fake_ids"] for b in batch], pad_id)
        loss, stats = spin_objective(classifier, p_batch, real_batch, fake_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += float(loss.item())
        epoch_real_acc += stats["real_acc"]
        epoch_fake_acc += stats["fake_acc"]
        n_batches += 1
    return {
        "spin_loss": epoch_loss / max(n_batches, 1),
        "real_acc": epoch_real_acc / max(n_batches, 1),
        "fake_acc": epoch_fake_acc / max(n_batches, 1),
    }


__all__ = [
    "Constitution",
    "AIJudge",
    "generate_ai_preferences",
    "spin_objective",
    "spin_iteration",
    "self_reward_score",
]
