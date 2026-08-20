r"""Build notebooks/ch19_agentic_rl.ipynb —— Agentic RL：把 RL 用到 Agent 上.

Phase 5 第一章（也是最后一章）：单轮 GRPO -> 多轮工具调用的世界观升级，
配一个真正能跑的玩具：带计算器的 LLM agent + 一轮 RAFT（rejection sampling）。
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch19")
cells = _nb.cells
md, code = _nb.md, _nb.code


# =============================================================================
# Title
# =============================================================================
md(r"""# 第 19 章：Agentic RL —— 把强化学习用到 Agent 上

Ch13 结束时你可能有一个隐约的疑问：我们训练的 GRPO，每次都是「一问一答」——prompt 进、response 出、打分、更新。可真正的 agent 不是这样工作的：它会**连续行动多轮**，中间调用搜索、代码、计算器这些工具，看到工具返回的观察再决定下一步。**动作改变世界，世界反馈信息**——这个描述你在哪见过？Ch00 的第一页：ClickWorld 的 agent-environment loop。

所以这一章是一个闭环：**Agent 的循环本来就是 RL 的循环**。你要做的只是把 Ch13 的「response」换成「一条带工具调用的轨迹」。听起来只是格式变化，但它正是当下最热的研究方向——Search-R1（用 RL 训练会搜索的模型）、ToolRL、OpenAI o3 的工具使用、DeepSeek-R1 的长链推理，本质上都是「Agentic RL」。

本章我们做一个最小但完整的版本：

1. 实现一个**带计算器工具的多轮解码循环**（工具结果注入生成上下文——和真实 agent 完全同构）
2. 对比实验：**同一个模型，无工具 vs 有工具**，看能力差距从哪来
3. 亲手跑一轮 **RAFT**（采样-过滤-再训练）：不写一行 RL 代码的「穷人版强化学习」
4. 把 GRPO 从单轮升级到多轮需要改什么（结构级讨论 + 伪代码 + 真实论文地图）

> 前置：Ch10（TinyGPT）、Ch13（GRPO）建议已完成；Ch16（PRM）有助于理解过程奖励部分。

## 学习目标

1. 说清 Agent loop 与 RL loop 的**同构关系**，以及多轮 setting 下 state/action/reward 各是什么
2. 实现**工具增强解码**：模型生成中穿插环境注入的观察
3. 用实验理解「工具 = 能力外置」：为什么小模型 + 工具能赢大模型裸算
4. 跑通 **RAFT**（rejection sampling + SFT），理解它是 GRPO 的一步近似
5. 掌握 Agentic GRPO 相对 Ch13 的**三个结构性变化**和四大挑战
""")

code(r"""# 常规设置：找项目根、载入库
import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rlenvs import CharTokenizer, build_tiny_gpt
from utils import set_seed

set_seed(0)
print(f"torch {torch.__version__} | CPU 线程 {torch.get_num_threads()}")
""")

md(r"""## 19.1 Agent loop 就是 RL loop

先做一次「翻译练习」。你在 Agent 教程里看到的词，和 Ch02 学过的 MDP 术语，一一对上：

| Agent 世界的说法 | RL 世界的说法 | 我们的玩具里是 |
|---|---|---|
| observation（观察：用户问题 / 工具返回） | state $s_t$ | 已生成的 prompt + 历史（含工具结果） |
| action（回复一段文本 / 发起一次工具调用） | action $a_t$ | 生成下一个 token（或触发 `?` 工具段） |
| environment（搜索引擎 / 计算器 / 代码沙盒） | 转移核 $P(s'\mid s,a)$ | 计算器：表达式 → 正确结果（**确定性**） |
| 任务完成信号（答案对了 / 测试通过） | reward $r$ | 最终答案 == 真值 → 1，否则 0 |
| 多轮交互直到结束 | trajectory / episode | 一条带工具调用的完整轨迹 |

两个真正的**新困难**（其余全是旧知识）：

1. **动作是 token 序列，且和环境交错**：模型 emit `?34+28=`，环境 inject `62`，模型继续。生成过程被切成多段——rollout 不再是「一口气采样」，而是一个**交互循环**。
2. **奖励更稀疏了**：一问一答时奖励延迟到句尾；多轮工具下延迟到**任务终点**。中间哪次工具调用是好的？——这正是 Ch16 PRM（过程奖励）和 Ch08 GAE（信用分配）在 agent 时代突然变热的原因。

> 🤔 **先猜再跑**（本章主实验预告）：同一个 TinyGPT（8 万参数、char-level），训练它做三位数连加 `34+28+61`。版本 A 直接输出答案；版本 B 允许调用计算器（先把前两个数喂给工具、拿到结果、再和第三个数求和）。预测两个版本在**没见过的测试题**上的正确率各是多少？先写下两个百分比再看 §19.3 的结果。
>
> <details><summary>写下猜测再点开提示（不是答案）</summary>
>
> 思考角度：版本 A 必须**在参数里学会进位加法**——这对 8 万参数的 char 模型是个硬任务；版本 B 只需要学会三件「小事」：把操作数**抄写**进工具调用、**读回**工具结果、把最终结果**抄写**到答案位。哪一边对网络更友好？如果你的直觉说「工具版应该接近满分、直算版很低」，想想这个结论对真实 LLM 意味着什么——工具不是接口，是**能力的边界扩展**。
> </details>
""")

md(r"""## 19.2 工具增强解码：模型与环境的共舞

给模型三件新「语法」：

- `?` 发起工具调用，之后 emit 一个表达式（数字和 `+`）
- `=` 结束表达式——**此刻环境接管**：计算结果并把数字注入生成流
- `#` 给出最终答案，之后 emit 答案数字
- `.` 轨迹终止符（模型必须学会「说完了」）

一条完整的工具轨迹长这样（prompt 是 `34+28+61`）：

```
34+28+61 | ?34+28= | 62 | ?62+61= | 123 | # | 123
 prompt     模型生成    ↑注入   模型生成    ↑注入      模型生成
```

实现上这是一个**状态机 + 修改版自回归解码**：每步照常从模型拿下一个 token 的分布；一旦模型 emit `=`（处于工具段内），暂停采样、调用工具、把结果 token 逐个拼回上下文，再继续。这和 Search-R1 把搜索结果拼回 context 的做法**结构完全相同**——只是把「搜索引擎」换成了「计算器」。""")

code(r"""# 19.2.1 工具环境 + 交互式解码（本章核心基础设施，~40 行）
def make_calculator():
    # 计算器工具：表达式（数字和 +）-> 正确结果的字符串。非法输入返回 None。
    def calc(expr: str):
        parts = [p for p in expr.split('+') if p != '']
        if not parts or not all(p.isdigit() for p in parts):
            return None
        return str(sum(int(p) for p in parts))
    return calc


@torch.no_grad()
def decode_with_tools(model, tok, prompt, max_new=48, temperature=0.8,
                      greedy=False, verbose=False):
    # 工具增强的自回归解码：
    # 状态机 start -> expr(工具表达式内) -> normal(结果已注入) -> ans(# 之后)
    # 环境（计算器）在模型 emit '=' 时接管，把结果注入上下文。
    # 返回 (完整输出字符串, 最终答案 str 或 '', 工具调用日志)。
    calc = make_calculator()
    ids = tok.encode(prompt).tolist()
    out = prompt
    state, expr_buf, ans_buf = "start", "", ""
    tool_log = []

    for _ in range(max_new):
        ctx = torch.tensor([ids[-model.max_seq_len:]], dtype=torch.long)
        logits = model(ctx)[0, -1]
        if greedy or temperature == 0:
            nid = int(logits.argmax())
        else:
            probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            nid = int(torch.multinomial(probs, 1))
        c = tok.itos[nid]
        out += c
        ids.append(nid)

        if state == "ans":                      # # 之后：收答案数字
            if c.isdigit():
                ans_buf += c
            else:
                break                           # 答案被非数字终止
        elif c == '#':
            state = "ans"
        elif c == '?':
            expr_buf, state = "", "expr"
        elif state == "expr":
            if c.isdigit() or c == '+':
                expr_buf += c
            elif c == '=':
                res = calc(expr_buf)            # ---- 环境接管 ----
                tool_log.append((expr_buf, res))
                if verbose:
                    tag = res if res is not None else "非法!"
                    print(f"    [tool] {expr_buf or '(空)'} -> {tag}")
                if res is not None:
                    for rc in res:              # 结果注入生成流
                        out += rc
                        ids.append(tok.stoi[rc])
                state = "normal"
    return out, ans_buf, tool_log


# 语法演示（不训练，先看交互循环怎么转）
ALPHABET = "0123456789+=?#."
tok = CharTokenizer()
tok.train("0123456789+=?#.")

demo_model = build_tiny_gpt(vocab_size=tok.vocab_size, d_model=64,
                            n_heads=4, n_layers=2, d_ff=256, max_seq_len=96)
print(f"演示模型（随机初始化）vocab={tok.vocab_size}，参数量 "
      f"{sum(p.numel() for p in demo_model.parameters()):,}")
print("随机模型当然什么都不会——但它 emit '=' 时计算器已经会接管：")
out, ans, log = decode_with_tools(demo_model, tok, "34+28+61", max_new=24,
                                  temperature=1.0, verbose=True)
print(f"  完整输出: {out!r}")
print(f"  工具日志: {log}")
""")

md(r"""## 19.3 主实验：无工具 vs 有工具

### 19.3.1 数据

任务：`a+b+c`，三个两位数（10~99），答案最多三位数。两个版本用**完全相同的问题集**训练同构的模型，唯一区别是目标格式：

- **版本 A（直算）**：`34+28+61` → `=123`
- **版本 B（工具）**：`34+28+61` → `?34+28=62?62+61=123#123`（中间结果由数据合成器算好，等价于「专家示范」）""")

code(r"""# 19.3.2 合成训练数据（两个版本共享同一批问题）
def make_problems(n, lo=10, hi=99, rng=None):
    rng = rng or random.Random(0)
    probs = set()
    while len(probs) < n:
        probs.add((rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)))
    return sorted(probs)

def direct_trace(a, b, c):
    return f"={a + b + c}."

def tool_trace(a, b, c):
    s1 = a + b
    return f"?{a}+{b}={s1}?{s1}+{c}={a + b + c}#{a + b + c}."

train_probs = make_problems(80, rng=random.Random(0))
test_probs = make_problems(40, rng=random.Random(1))

A_DATA = [(f"{a}+{b}+{c}", direct_trace(a, b, c)) for a, b, c in train_probs]
B_DATA = [(f"{a}+{b}+{c}", tool_trace(a, b, c)) for a, b, c in train_probs]

print(f"训练 {len(train_probs)} 题 / 测试 {len(test_probs)} 题（不相交）")
print("A 样例:", A_DATA[0][0], "->", A_DATA[0][1])
print("B 样例:", B_DATA[0][0], "->", B_DATA[0][1])
""")

code(r"""# 19.3.3 训练函数：带 prompt 掩码的 SFT（只在 response 部分算 loss）
def sft_train(model, tok, data, n_steps=300, batch_size=32, lr=1.5e-3, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        batch = random.sample(data, min(batch_size, len(data)))
        seqs, masks = [], []
        for prompt, resp in batch:
            full = tok.encode(prompt + resp).tolist()
            plen = len(prompt)
            # 位置 i 预测 full[i+1]；只学 response 部分（i+1 >= plen）
            m = [0.0] * (len(full) - 1)
            for i in range(len(full) - 1):
                if i + 1 >= plen:
                    m[i] = 1.0
            seqs.append(full)
            masks.append(m)
        L = max(len(s) for s in seqs)
        pad = tok.pad_id
        x = torch.full((len(seqs), L - 1), pad, dtype=torch.long)
        y = torch.full((len(seqs), L - 1), pad, dtype=torch.long)
        m = torch.zeros((len(seqs), L - 1))
        for i, (s, mk) in enumerate(zip(seqs, masks)):
            x[i, :len(s) - 1] = torch.tensor(s[:-1])
            y[i, :len(s) - 1] = torch.tensor(s[1:])
            m[i, :len(mk)] = torch.tensor(mk)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                             reduction='none').reshape(y.shape)
        loss = (ce * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    print(f"  [{tag}] {n_steps} 步 完成，loss {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"({time.time() - t0:.0f}s)")
    return losses


def eval_direct(model, tok, problems, max_new=5):
    # 版本 A 评估：贪心生成 = 后的数字。
    ok = 0
    for a, b, c in problems:
        ids = tok.encode(f"{a}+{b}+{c}").tolist()
        ans = ""
        started = False
        for _ in range(max_new):
            ctx = torch.tensor([ids[-model.max_seq_len:]], dtype=torch.long)
            with torch.no_grad():
                nid = int(model(ctx)[0, -1].argmax())
            ch = tok.itos[nid]
            ids.append(nid)
            if ch.isdigit():
                ans += ch; started = True
            elif started:
                break
        ok += int(ans == str(a + b + c))
    return ok / len(problems)


def eval_tool(model, tok, problems):
    # 版本 B 评估：工具增强贪心解码，看 # 后的答案。
    ok = 0
    for a, b, c in problems:
        _, ans, _ = decode_with_tools(model, tok, f"{a}+{b}+{c}",
                                      max_new=48, greedy=True)
        ok += int(ans == str(a + b + c))
    return ok / len(problems)
""")

code(r"""# 19.3.4 训练两个版本（结构完全相同，只有目标格式不同）
model_A = build_tiny_gpt(vocab_size=tok.vocab_size, d_model=64, n_heads=4,
                         n_layers=2, d_ff=256, max_seq_len=96)
model_B = build_tiny_gpt(vocab_size=tok.vocab_size, d_model=64, n_heads=4,
                         n_layers=2, d_ff=256, max_seq_len=96)

print("版本 A（直算）：")
lossA = sft_train(model_A, tok, A_DATA, n_steps=300, tag="A")
print("版本 B（工具）：")
lossB = sft_train(model_B, tok, B_DATA, n_steps=600, tag="B")

accA = eval_direct(model_A, tok, test_probs)
accB = eval_tool(model_B, tok, test_probs)
print(f"\\n测试集正确率（40 道没见过的题）：")
print(f"  A 直算   : {accA:.1%}")
print(f"  B 工具   : {accB:.1%}")

# 看一条 B 的完整轨迹（贪心）
a, b, c = test_probs[0]
out, ans, log = decode_with_tools(model_B, tok, f"{a}+{b}+{c}", max_new=48, greedy=True)
print(f"\\nB 的样例轨迹: {out!r}   (答案 {ans!r}, 真值 {a + b + c})")
""")

md(r"""### 19.3.5 读结果：工具是能力的边界扩展

如果实验符合预期，你会看到 **A 显著低于 B**。停下来想一分钟为什么——两个模型的参数量、架构、训练数据量**完全相同**：

- A 的网络必须把「进位加法」**压缩进 8 万个参数**——这是硬功夫，学不会就是学不会；
- B 的网络只需学会**抄写**（把操作数复制进工具调用）、**转写**（把注入的结果复制进下一次调用和答案）——对序列模型来说这是本行。

> 🌍 **真实世界**：这个结论按比例放大就是今天的 LLM 格局——再大的模型算不定长乘法也不如一行 `python -c`；再强的记忆也比不过一次检索。**工具不是 UI，是能力边界的扩展**。OpenAI o3、DeepSeek-R1 的「先搜再答」、代码沙盒里的反复试错，都是这条原理的工业级版本。而「学会**什么时候**用工具、**怎么**用工具」——这正是下一节 RL 要解决的学习问题。
""")

md(r"""## 19.4 从 best-of-N 到 RAFT：把「推理时的余量」装进权重

先量一个 agent 时代最重要的量。同一个模型 B，对每道测试题**采 8 条轨迹**：

- **oracle best-of-8**：只要有一条对就算对（上限——假设有个完美裁判）
- **多数投票**（self-consistency）：8 条的最终答案投票取众数（不需要裁判！）

马上会看到：greedy 正确率有限 的模型，best-of-8 几乎全对——**采样分布里藏着巨大余量**。这正是「推理时扩展」（inference-time scaling）的原理：o1/R1 的「多想几遍再答」花的就是这笔钱。

但这余量每次推理都要重新付 8 倍计算。能不能**一次性装进权重**？两个方案：

- **RAFT**（Rejection sampling + Fine-Tuning，STaR 的现代版）：采样 K 条 → 只留答案正确的 → 拿这些自产成功再 SFT。它和 GRPO 的关系一句话说清：**GRPO 用 group advantage 连续加权（好 +1σ、差 −1σ），RAFT 把权重粗暴二值化（好 1、差 0，丢掉负样本的梯度信息）**。
- **GRPO**：完整的策略梯度（下一节把它搬到多轮）。

> 🤔 **先猜再跑**：三个量——(1) oracle best-of-8 大概多少？(2) 多数投票比 oracle 高还是低？(3) 最关键的：一轮 RAFT 之后，模型 B 的 **greedy** 正确率会显著上升、小幅上升、还是基本不动？
>
> <details><summary>写下三个预测再跑</summary>
>
> 提示 1/2：单条成功率哪怕只有 70%，8 条至少一条对 ≈ 1−0.3⁸ ≈ 99.99%；投票要「对的比错的多」，介于单条和 oracle 之间。
> 提示 3（诚实预告，这是本节真正的考点）：RAFT 生效需要两个前提——模型「平均水平远低于采样上限」且「还有泛化余量」。想想我们这个玩具：训练题只有 80 道且模型已经把它们**背下来了**（自产轨迹几乎全对），再训练等于在同样分布上加练——泛化天花板撞死之后，加练练的是什么？
> </details>
""")

code(r"""# 19.4.1 量化推理时余量：best-of-8 与多数投票
from collections import Counter

def bon_eval(model, tok, problems, k=8, temperature=1.0):
    # 返回 (oracle best-of-k 正确率, 多数投票正确率)
    oracle_ok, vote_ok = 0, 0
    for a, b, c in problems:
        answers = []
        for _ in range(k):
            _, ans, _ = decode_with_tools(model, tok, f"{a}+{b}+{c}",
                                          max_new=48, temperature=temperature)
            answers.append(ans)
        truth = str(a + b + c)
        if truth in answers:
            oracle_ok += 1
        top, _ = Counter(answers).most_common(1)[0]
        vote_ok += int(top == truth)
    return oracle_ok / len(problems), vote_ok / len(problems)

accB_greedy = eval_tool(model_B, tok, test_probs)
accB_oracle, accB_vote = bon_eval(model_B, tok, test_probs, k=8)
print(f"模型 B 在测试集上：")
print(f"  greedy 单次     : {accB_greedy:.1%}")
print(f"  多数投票 (k=8)  : {accB_vote:.1%}")
print(f"  oracle best-of-8: {accB_oracle:.1%}   <- 采样分布里的余量")
""")

code(r"""# 19.4.2 RAFT 一轮：采样 -> 过滤 -> 再训练（如实观察结果）
def raft_collect(model, tok, problems, k=8, temperature=1.0):
    # 对每道题采 k 条轨迹，留下「至少一次成功工具调用且答案正确」的自产轨迹。
    kept, n_total = [], 0
    for a, b, c in problems:
        prompt = f"{a}+{b}+{c}"
        for _ in range(k):
            out, ans, tool_log = decode_with_tools(
                model, tok, prompt, max_new=48, temperature=temperature)
            n_total += 1
            ok = (ans == str(a + b + c)
                  and len(tool_log) >= 1
                  and all(r is not None for _, r in tool_log))
            if ok:
                kept.append((prompt, out[len(prompt):]))   # 剥掉 prompt
    return kept, n_total

t0 = time.time()
raft_data, n_total = raft_collect(model_B, tok, train_probs, k=8)
print(f"训练题采样 {n_total} 条，过滤保留 {len(raft_data)} 条"
      f"（{len(raft_data) / n_total:.1%}），耗时 {time.time() - t0:.0f}s")
print("（注意：训练题几乎全对——模型把它们背下来了，这正是问题所在）")

accB_before = accB_greedy
lossR = sft_train(model_B, tok, raft_data, n_steps=150, tag="RAFT")
accB_after = eval_tool(model_B, tok, test_probs)

print(f"\\nRAFT 前后（同一测试集，greedy）：{accB_before:.1%} -> {accB_after:.1%} "
      f"({accB_after - accB_before:+.1%})")
""")

code(r"""# 19.4.3 可视化：五个量放一起看
fig, ax = plt.subplots(figsize=(9, 4.5))
names = ["A 直算\\n(无工具)", "B greedy\\n(单次)", "B 多数投票\\n(k=8)",
         "B oracle\\n(best-of-8)", "B +RAFT\\n(greedy)"]
vals = [accA, accB_greedy, accB_vote, accB_oracle, accB_after]
colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#1f77b4"]
bars = ax.bar(names, vals, color=colors, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}",
            ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("测试集正确率")
ax.set_ylim(0, 1.08)
ax.set_title("无工具 vs 工具、单次 vs 采样余量、以及 RAFT 的真实表现")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
""")

md(r"""### 读结果：这次实验教会我们「先量天花板，再选工具」

如实解读你看到的（具体数字每次运行略有波动，但形状一致）：

- **greedy << oracle best-of-8**：采样分布里的余量是真的、巨大——这就是推理时扩展的根据，也是 RL 理论上能吃到的收益上限
- **多数投票 ≈ greedy**：投票治「随机错」，不治「系统错」——B 的错误是固定的抄写滑误（每次都错在同一个位置），8 条里错得整齐划一，投票无药可医。真实推理任务里两种错误混杂，投票通常有效一些
- **RAFT 不仅没吃到余量，还明显倒退**：因为两个前提都不满足——模型在 80 道训练题上已经背熟（自产数据 ≈ 专家数据，过滤没有信息增量，重训只是加练过拟合），而泛化天花板（oracle 那一线）已经贴着（测试题的抄写错误是「容量不够」而不是「练得不够」）
- **什么时候 RAFT 真的有效？** 当任务有真正的大答案空间（推理链而不是抄写）、SFT 水平远低于采样上限、且泛化有余量时——STaR 在 GSM8K 上的经典设定正是如此。真实 LLM 上这些条件常常成立，在 8 万参数的抄写玩具上不成立

> 🌍 **真实世界**：这一节的五个量就是 agent 评估的标准仪表盘——单次性能（便宜）、多数投票（多花 k 倍推理费买稳定）、oracle best-of-k（RL 的理论收益上限）、以及「RL 之后单次性能涨了多少」（把推理费一次性折旧成权重）。**先量 oracle best-of-k，再决定要不要上 RL**——这个工程直觉值回本章票价。

最后一块拼图：RAFT 的二值化过滤丢掉了「差多少」的信息。**用 group advantage 连续加权、且天生适配多轮轨迹**的，正是你已经学过的 GRPO——下一节把它搬进 agent 世界。
""")

md(r"""## 19.5 Agentic GRPO：从单轮到多轮，改哪三处

有了上面的玩具，现在能精确说出 Ch13 的 GRPO 要搬到 agent 世界需要改什么了。**目标函数一个字不用改**（group advantage、clip、KL penalty 原样保留），改的是**rollout 的发生方式**：

| | Ch13 单轮 GRPO | Agentic GRPO（多轮） |
|---|---|---|
| 轨迹 | 一次采样 prompt→response | **交互循环**：模型段 ⇄ 环境段（§19.2 的 `decode_with_tools` 就是雏形） |
| 状态 | prompt 固定 | 每轮包含**工具返回的观察**（上下文随交互增长） |
| 奖励 | RM 打分（或规则） | 通常是**任务结局**（答案对/错、测试通过/失败），更稀疏 |
| log π 的计算 | 对 response token 求和 | 只对**模型 emit 的 token** 求和——**环境注入的 token 不算**（不是模型的动作！） |

最后一点是唯一容易踩的坑：工具返回的 token 属于**观察**（state 的一部分），如果把它们的 log-prob 也算进 π 的分子，策略梯度就错了。伪代码：

```python
for prompt x in batch:                    # 每个任务
    for i in range(G):                    # 组内 G 条轨迹
        traj, logp = [], []
        while not done:
            tok_ids = model.sample(...)   # 模型段：emit token（计入 logp）
            obs = env.step(tok_ids)       # 环境段：工具观察（不计入 logp）
        r_i = task_reward(traj)           # 结局奖励（稀疏）
    A_i = (r - mean(r)) / (std + eps)     # ← Ch13 原封不动
loss = ppo_clip(logp, A_i) + kl_penalty   # ← Ch13 原封不动
```

**四大挑战**（每一条都是 2024-2025 的活跃论文方向）：

1. **长视野信用分配**——10 轮工具调用前的那次搜索，对最终答对有多大功劳？GAE over turns、PRM（Ch16）都是候选答案
2. **奖励稀疏 + 可作弊**——结局奖励太粗，过程奖励又容易被 hack（Ch11 的 Goodhart 在这里复发）
3. **rollout 昂贵**——一条轨迹 = 多次 LLM 推理 + 多次工具调用；训练吞吐被环境卡住
4. **环境不可复现/不可仿真**——真实搜索引擎每次结果不同，同一个动作两次奖励不同，方差暴涨
""")

md(r"""## 19.6 真实世界与前沿地图

> 🌍 **这个方向的名字就叫 Agentic RL**。几个值得认识的坐标（按与我们玩具的关系排）：
>
> - **ReAct**（Yao et al. 2022）——Thought-Action-Observation 循环的开山，我们的 `?expr=结果` 就是它的最小化身；原工作用 prompting 实现，**不学习**
> - **STaR / RAFT**（Zelikman 2022; Dong et al. 2023）——§19.4 的方法本体：采样、过滤、再训练
> - **Search-R1**（Jin et al. 2025）——把搜索引擎当工具、答案正确性当 reward、GRPO 做优化器的完整闭环；我们的玩具换掉工具就是这个工作的骨架
> - **ToolRL / Tool-use RL**（2025 系列）——研究「什么时候调工具」本身作为学习目标：reward 里显式加入工具调用的时机奖惩
> - **OpenAI o3 / DeepSeek-R1**——工业级推理 agent：长链工具调用 + 强化学习，公开报告都描述了「反思、回溯、自我纠错」行为的**涌现**（我们在 ch15 猜过的「先装死、后起飞」的大模型版）
>
> 一个诚实的现状判断（也是你判断研究机会的依据）：**单轮 RLHF 的配方已经标准化（DPO/GRPO），而多轮 Agentic RL 的配方还在混战**——奖励怎么设计、过程奖励怎么防 hack、长视野 credit 怎么分，都还没有共识答案。这正是「刚刚开始」的含义：**入场的好时机**。

## 19.7 小结

> 从 Ch00 网格里「策略决定一切」的懵懂，到 Ch13 让模型学会「对的答案概率更高」，
> 再到今天——模型伸出手，**拿起工具**，在多轮交互中自己试错、自己改进。
> 你手里的小小计算器轨迹，和 o3 的搜索-推理-回溯循环，在结构上是同一个故事。

- ✅ Agent loop 与 RL loop 同构：观察=状态、工具调用=动作、任务结局=奖励
- ✅ 工具增强解码 = 状态机 + 环境注入（观察进上下文，但不进 log π）
- ✅ 实验：同参数模型，工具版碾压直算版——工具是能力边界的扩展
- ✅ RAFT = 权重二值化的 GRPO：从自己的成功中学习
- ✅ Agentic GRPO 的三个结构变化 + 四大开放挑战

## 19.8 📝 练习

### 练习 1（必做）：把 RAFT 变成 iterative

只跑了一轮 RAFT。把它改成 3 轮循环（每轮重新采样、过滤、再训练），画出每轮的测试正确率折线。预期：边际收益递减，最终停在「采样分布覆盖正确答案」的上限附近。

**提示**：每轮 `raft_collect` 用**当前**的 model_B；把 `eval_tool` 的结果存进 list。

### 练习 2（选做）：给工具调用加代价

现实里工具不免费（延迟、配额）。改 `decode_with_tools`：每次工具调用记录一次计数；reward 改成「答案正确 − 0.1 × 工具次数」，重新跑 RAFT。观察模型会不会学会「简单的题直答、难的题才用工具」——这就是 ToolRL 论文的核心设定。

> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch19 的自测题。

## 参考文献

- Yao, S. et al. *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)
- Zelikman, E. et al. *STaR: Bootstrapping Reasoning With Reasoning* (2022)
- Dong, G. et al. *RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment* (2023)
- Jin, B. et al. *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning* (2025)
- DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability via RL* (2025)
""")


if __name__ == "__main__":
    _nb.write("ch19_agentic_rl.ipynb")
