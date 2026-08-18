"""Reward Modeling 工具集（Ch11）。

本章把 Ch10 的 TinyGPT 当作 **backbone**，加一个标量 reward head，
在 pairwise 偏好数据上训练——兑现 Bradley-Terry / RM 承诺。

设计原则（与项目其它模块一致）：

- **教学优先**：每个组件（reward head、loss、数据生成器）都拆成独立可读的函数。
- **不重复造轮子**：直接复用 :class:`rlenvs.tiny_gpt.TinyGPT` 作为 backbone，
  只在外层加一个 scalar head（通过 forward hook 抓 ``ln_final`` 的输出当 hidden state）。
- **小而能学**：合成偏好数据用简单的"隐含 reward"规则（response 长度 / 关键词），
  让 reward model 能在 CPU 上 < 1 分钟训到 > 70% 准确率。

模块组成：

- :class:`RewardModel` —— TinyGPT backbone + scalar reward head
- :func:`bradley_terry_loss` —— 成对偏好 loss：``-log sigma(r_w - r_l)``
- :func:`generate_preference_data` —— 合成 pairwise 偏好数据生成器
- :func:`make_preference_batch` —— 把 ``(prompt, y_w, y_l)`` 三元组打包成 batch tensor
- :func:`reward_accuracy` —— 在验证偏好对上算 RM 预测正确率
- :func:`true_reward` —— 合成数据的 ground-truth reward（用于过优化曲线）

forward 约定：
    ``RewardModel(prompt_ids, response_ids)`` 拼接 prompt+response，过 TinyGPT，
    取**最后一个 response token** 的 hidden state 作为整个序列的"汇总向量"，
    再经 LayerNorm + Linear → 标量 reward。
"""
from __future__ import annotations

import random
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. RewardModel：TinyGPT backbone + scalar reward head
# =============================================================================
class RewardModel(nn.Module):
    """TinyGPT backbone + 一个 scalar reward head。

    结构::

        prompt + response token 拼接
            ↓ TinyGPT 的 token embedding + PE + N × TransformerBlock + final LayerNorm
            hidden states [B, T, d_model]   ← 通过 forward hook 抓 ``ln_final`` 的输出
            ↓ 取最后一个 response token 的 hidden vector（汇总整段序列）
            ↓ reward_head: LayerNorm → Linear(d_model → 1) → squeeze
            reward scalar [B]

    设计要点：

    - **复用 TinyGPT**：不重写 transformer，直接拿 Ch10 的 backbone，省 90% 代码。
    - **hidden state 抓取**：TinyGPT.forward 返回 logits（[B, T, V]），
      我们用 forward hook 拿 ``ln_final`` 的**输入**（也就是 N 个 block 后、
      LM head 前的 hidden state，shape ``[B, T, d_model]``）作为序列表示。
    - **取最后一个 response token**：自回归 transformer 的最后位置能"看见"
      整个 prompt+response，是最自然的序列汇总点。这是 InstructGPT (2022) 的标准做法。
    - **reward head**：LayerNorm + Linear，把 d_model 维 hidden state 压成 1 个 reward 值。
      不加激活——reward 是无界实数（Bradley-Terry 模型只用 reward **差**，绝对值无意义）。

    可识别性（identifiability）说明：

        Bradley-Terry loss ``-log sigma(r_w - r_l)`` 只看 reward 差，
        所以对任意常数 ``c``，``r → r + c`` 给出同样的 loss。
        训练时 reward 整体平移是自由的，**绝对值不可识别**——只有相对排序有意义。
        （这跟 Dueling Q-Network 的 V/A 平移自由度是同一现象。）
    """

    def __init__(self, backbone: nn.Module, d_model: Optional[int] = None) -> None:
        super().__init__()
        self.backbone = backbone
        # 推断 d_model（兼容 TinyGPT 和任何带 d_model 属性的 backbone）
        if d_model is None:
            d_model = getattr(backbone, "d_model", None)
            if d_model is None:
                raise ValueError(
                    "无法从 backbone 推断 d_model，请显式传入 d_model="
                )
        self.d_model = d_model

        # reward head：LayerNorm + Linear（InstructGPT 标准做法）
        self.reward_ln = nn.LayerNorm(d_model)
        self.reward_head = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

        # ---- 通过 forward hook 抓 ln_final 的输入（hidden state）----
        # TinyGPT.forward 里顺序是：blocks → ln_final → lm_head
        # 所以 ln_final 的"输入"就是 N 个 block 输出后的 hidden state。
        self._hidden: Optional[torch.Tensor] = None
        # TinyGPT 把 ln_final 命名为 'ln_final'
        target_module = self._find_ln_final(backbone)
        if target_module is None:
            raise ValueError(
                "backbone 上找不到 'ln_final' 模块；"
                "RewardModel 当前只支持 TinyGPT 风格的 backbone。"
            )
        target_module.register_forward_hook(self._hook_capture_hidden)

    @staticmethod
    def _find_ln_final(module: nn.Module) -> Optional[nn.Module]:
        """在 backbone 上找名为 'ln_final' 的子模块；找不到返回 None。"""
        if hasattr(module, "ln_final"):
            return module.ln_final
        for name, child in module.named_modules():
            if name == "ln_final":
                return child
        return None

    def _hook_capture_hidden(
        self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        # ln_final 的输入 = (hidden,) tuple，hidden shape [B, T, d_model]
        if isinstance(inputs, tuple) and len(inputs) > 0:
            self._hidden = inputs[0].detach()
        else:
            # 兜底：直接用 ln_final 的输出（也很接近，只是过了 LN）
            self._hidden = output.detach()

    def forward(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """对每条 (prompt, response) 算一个标量 reward。

        Parameters
        ----------
        prompt_ids : LongTensor [B, T_p]
            prompt 的 token ids。
        response_ids : LongTensor [B, T_r]
            对应的 response token ids。T_r ≥ 1。

        Returns
        -------
        rewards : FloatTensor [B]
            每条样本一个标量 reward。
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)
        # 拼成完整序列：prompt + response
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [B, T_p + T_r]
        # 调用 backbone（hook 会自动抓 hidden state）
        _ = self.backbone(full_ids)
        hidden = self._hidden  # [B, T, d_model]
        if hidden is None:
            raise RuntimeError("forward hook 没抓到 hidden state")
        # 取最后一个 token（= 最后一个 response token）的 hidden vector
        last_hidden = hidden[:, -1, :]  # [B, d_model]
        # reward head
        reward = self.reward_head(self.reward_ln(last_hidden)).squeeze(-1)  # [B]
        return reward


# =============================================================================
# 2. Bradley-Terry loss
# =============================================================================
def bradley_terry_loss(
    reward_model: nn.Module,
    prompt_ids: torch.Tensor,
    y_w_ids: torch.Tensor,
    y_l_ids: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry 偏好 loss：``-log sigma(r(x, y_w) - r(x, y_l))``。

    数学（详见 Ch11 §11.2）::

        P(y_w ≻ y_l | x) = sigma(r(x, y_w) - r(x, y_l))

        对数似然 = log sigma(r_w - r_l)
        loss    = -log sigma(r_w - r_l)

    当 ``r_w > r_l``（reward model 排序正确）→ ``r_w - r_l`` 大 → ``sigma`` 接近 1
    → ``-log sigma`` 接近 0（loss 小）。反之 reward model 排错则 loss 大。

    Parameters
    ----------
    reward_model : RewardModel（或任何 ``forward(prompt, response) -> [B]`` 的 nn.Module）
    prompt_ids : [B, T_p]
    y_w_ids    : [B, T_r]   winner response
    y_l_ids    : [B, T_r]   loser response

    Returns
    -------
    loss : scalar tensor
    """
    r_w = reward_model(prompt_ids, y_w_ids)  # [B]
    r_l = reward_model(prompt_ids, y_l_ids)  # [B]
    # -log sigma(r_w - r_l) = softplus(r_l - r_w) = log(1 + exp(r_l - r_w))
    # 用 F.softplus 数值稳定（不会 exp 上溢）。
    loss = F.softplus(r_l - r_w).mean()
    return loss


# =============================================================================
# 3. 合成偏好数据生成器
# =============================================================================
def _default_vocab_tokens(tokenizer) -> List[str]:
    """从 CharTokenizer 取若干可读字符（排除 pad / 控制字符）。"""
    return [c for c in tokenizer.itos if c != tokenizer.PAD_TOKEN]


def true_reward(
    prompt: str,
    response: str,
    target_keyword: str = "good",
    target_len: int = 6,
    keyword_weight: float = 2.0,
    len_weight: float = 0.5,
) -> float:
    """Ground-truth reward（合成偏好数据用，过优化实验也用它）。

    这是一个**人为设计**的 reward 函数，模拟"人类偏好"：
    - 含 target_keyword（默认 'good'）→ +keyword_weight
    - response 长度越接近 target_len（每个字符权重）→ +len_weight * proximity
    - 否则单调下降

    我们会让 RM 从 pairwise 数据里**反推**出这个 reward。

    注意：真实 RLHF 里这个函数不存在（人类偏好是隐式的），只有 pairwise 比较。
    我们这里为了**画过优化曲线**才显式定义它。

    返回：标量 reward（无界，越大越好）。
    """
    # 关键词奖励：含 'good' 加 keyword_weight
    kw_bonus = keyword_weight if target_keyword in response else 0.0
    # 长度奖励：高斯峰在 target_len 上
    L = len(response)
    len_bonus = len_weight * math.exp(-((L - target_len) ** 2) / 8.0)
    # 小惩罚：含 'bad' 关键词扣分
    bad_penalty = -keyword_weight if "bad" in response else 0.0
    return kw_bonus + len_bonus + bad_penalty


def generate_preference_data(
    tokenizer,
    n_samples: int = 200,
    prompts_pool: Optional[Sequence[str]] = None,
    response_pool: Optional[Sequence[str]] = None,
    seed: int = 0,
    target_keyword: str = "good",
    target_len: int = 6,
    keyword_weight: float = 2.0,
    len_weight: float = 0.5,
) -> List[Dict]:
    """合成 pairwise 偏好数据生成器。

    流程::

        for i in range(n_samples):
            prompt  ← 从 prompts_pool 随机选
            resp_A, resp_B ← 从 response_pool 随机选两个不同
            r_A = true_reward(prompt, resp_A)   # 隐含 reward
            r_B = true_reward(prompt, resp_B)
            if r_A > r_B: winner, loser = resp_A, resp_B
            else:         winner, loser = resp_B, resp_A
            sample = {prompt, winner, loser, r_w, r_l, r_w - r_l}

    真实 RLHF 里 winner/loser 由人类标注，这里用规则代替（教学清晰）。

    Parameters
    ----------
    tokenizer : CharTokenizer（只用 vocab 信息编码）
    n_samples : int，生成多少条偏好对
    prompts_pool / response_pool : 可选，自定义 pool
    seed : int，可复现

    Returns
    -------
    list of dict，每个 dict 包含:
        - 'prompt' / 'winner' / 'loser': str
        - 'prompt_ids' / 'winner_ids' / 'loser_ids': LongTensor [T]
        - 'r_w' / 'r_l' / 'r_diff': float（隐含 reward 的 ground truth）
    """
    rng = random.Random(seed)

    if prompts_pool is None:
        prompts_pool = [
            "Q: How is the weather? A:",
            "Q: Is it good? A:",
            "Q: Tell me a word. A:",
            "Q: What do you think? A:",
            "Q: How are you? A:",
        ]
    if response_pool is None:
        # 设计一组 response，让隐含 reward 有差异：
        # - 含 'good' 的 → 高 reward
        # - 长度接近 6 的 → 加分
        # - 含 'bad' 的 → 低 reward
        response_pool = [
            "good",         # len 4, kw+
            "very good",    # len 9, kw+
            "it good",      # len 7, kw+
            "good day",     # len 8, kw+
            "ok",           # len 2
            "fine",         # len 4
            "yes",          # len 3
            "no",           # len 2
            "bad",          # len 3, penalty
            "it is bad",    # len 9, penalty
            "very bad",     # len 8, penalty
            "hello world",  # len 11
            "yes good",     # len 8, kw+
            "no way",       # len 6
            "fine ok",      # len 7
            "great",        # len 5
        ]

    data: List[Dict] = []
    seen = set()
    attempts = 0
    while len(data) < n_samples and attempts < n_samples * 20:
        attempts += 1
        prompt = rng.choice(prompts_pool)
        a, b = rng.sample(response_pool, 2)
        # 跳过完全相同 reward 的对（保证 winner/loser 唯一）
        r_a = true_reward(prompt, a, target_keyword, target_len, keyword_weight, len_weight)
        r_b = true_reward(prompt, b, target_keyword, target_len, keyword_weight, len_weight)
        if abs(r_a - r_b) < 1e-6:
            continue
        key = (prompt, a, b) if a < b else (prompt, b, a)
        if key in seen:
            continue
        seen.add(key)
        if r_a > r_b:
            winner, loser, r_w, r_l = a, b, r_a, r_b
        else:
            winner, loser, r_w, r_l = b, a, r_b, r_a
        data.append({
            "prompt": prompt,
            "winner": winner,
            "loser": loser,
            "prompt_ids": tokenizer.encode(prompt),
            "winner_ids": tokenizer.encode(winner),
            "loser_ids": tokenizer.encode(loser),
            "r_w": float(r_w),
            "r_l": float(r_l),
            "r_diff": float(r_w - r_l),
        })
    return data


# =============================================================================
# 4. Batch 工具
# =============================================================================
def pad_to_length(
    seqs: Sequence[torch.Tensor],
    pad_id: int,
    max_len: Optional[int] = None,
) -> torch.Tensor:
    """把若干 1D LongTensor pad 到同一长度（取最长，或指定 max_len）。

    Returns shape [B, T]。
    """
    if max_len is None:
        max_len = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = min(int(s.numel()), max_len)
        out[i, :L] = s[:L]
    return out


def make_preference_batch(
    samples: Sequence[Dict],
    pad_id: int,
    prompt_max_len: Optional[int] = None,
    response_max_len: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """把若干 preference 样本打包成 batch tensor。

    Returns:
        dict with keys:
          - 'prompt_ids' [B, T_p]
          - 'winner_ids' [B, T_r]
          - 'loser_ids'  [B, T_r]
          - 'r_diff'     [B]   (ground-truth reward diff, 用于过优化曲线/分析)
    """
    prompts = [s["prompt_ids"] for s in samples]
    winners = [s["winner_ids"] for s in samples]
    losers = [s["loser_ids"] for s in samples]
    prompt_batch = pad_to_length(prompts, pad_id, prompt_max_len)
    winner_batch = pad_to_length(winners, pad_id, response_max_len)
    loser_batch = pad_to_length(losers, pad_id, response_max_len)
    r_diff = torch.tensor([s["r_diff"] for s in samples], dtype=torch.float32)
    return {
        "prompt_ids": prompt_batch,
        "winner_ids": winner_batch,
        "loser_ids": loser_batch,
        "r_diff": r_diff,
    }


# =============================================================================
# 5. 评估工具
# =============================================================================
@torch.no_grad()
def reward_accuracy(
    reward_model: nn.Module,
    samples: Sequence[Dict],
    pad_id: int,
    batch_size: int = 64,
) -> float:
    """在偏好对上算 RM 准确率：r(prompt, winner) > r(prompt, loser) 的比例。

    Returns: float in [0, 1]。
    """
    reward_model.eval()
    n_correct = 0
    n_total = 0
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        b = make_preference_batch(batch, pad_id)
        r_w = reward_model(b["prompt_ids"], b["winner_ids"])
        r_l = reward_model(b["prompt_ids"], b["loser_ids"])
        n_correct += int((r_w > r_l).sum().item())
        n_total += len(batch)
    return n_correct / max(1, n_total)


@torch.no_grad()
def predict_rewards(
    reward_model: nn.Module,
    samples: Sequence[Dict],
    pad_id: int,
    which: str = "winner",
    batch_size: int = 64,
) -> torch.Tensor:
    """对一堆样本的指定侧（winner / loser）算 reward，返回 [N] tensor。"""
    reward_model.eval()
    out: List[torch.Tensor] = []
    key = "winner_ids" if which == "winner" else "loser_ids"
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        b = make_preference_batch(batch, pad_id)
        r = reward_model(b["prompt_ids"], b[key])
        out.append(r.detach().cpu())
    return torch.cat(out)


__all__ = [
    "RewardModel",
    "bradley_terry_loss",
    "generate_preference_data",
    "true_reward",
    "make_preference_batch",
    "pad_to_length",
    "reward_accuracy",
    "predict_rewards",
]
