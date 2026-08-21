r"""Build notebooks/ch20_agentic_grpo.ipynb —— Agentic GRPO 实战 + 全书终章.

Phase 5 第二章：把 Ch13 的 GRPO 真正搬进多轮工具世界（不是伪代码，是跑起来的实现），
用两个对照实验揭示 on-policy RL 的信号来源，最后以全书终章收尾。
"""
from nb_helpers import NotebookBuilder

_nb = NotebookBuilder("ch20")
cells = _nb.cells
md, code = _nb.md, _nb.code


# =============================================================================
# Title
# =============================================================================
md(r"""# 第 20 章：Agentic GRPO 实战 —— 多轮强化学习与全书终章

上一章（Ch19）结束在一个悬而未决的问题上：oracle best-of-8 显示采样分布里有 ~20 个百分点的余量，但 RAFT 不仅没吃到、还倒退了。我们当时给出的诊断是「背熟的题集上过滤没有信息增量」。

这一章把这个诊断变成可检验的实验，并请出真正的主角：**把 Ch13 的 GRPO 一字不改地（几乎）搬进 Ch19 的多轮工具世界，亲手跑通它**。过程中会遇到两个反直觉的发现——GRPO 在固定题集上同样学不动，而**换成一池子每轮现生成的新题后它开始稳定涨分**。这两个现象合起来，就是 on-policy 强化学习的本质：**信号来自「模型还做不稳的事」，而 RL 的训练集可以由任务分布无限自生成**——DeepSeek-R1 的无限题海、Search-R1 的无限搜索，都是这个原理的工业版。

这也是 RLStudy 的最后一章。跑完它，你从 Ch00 的 5×5 网格出发的旅程就完整了。

> 前置：Ch13（GRPO）、Ch19（工具循环与五量仪表盘）必须完成。

## 学习目标

1. 实现**带 log π 蒙版的多轮 rollout**：模型 emit 的 token 计入策略、环境注入的观察不计入
2. 实现 **agentic GRPO 更新步**：组优势 + PPO-clip + KL 锚，全在多轮轨迹上
3. 用两个对照实验理解 **RL 的信号来自哪里**（组内方差）与 **on-policy 自生成课程**的含义
4. 掌握 Agentic GRPO 相对 Ch13 的三个结构性变化和四大开放挑战
5. 带走全书的最终能力清单与下一步路线
""")

code(r"""# 常规设置：找项目根、载入库（与 Ch19 相同的玩具：计算器 + 两步加法）
import sys, pathlib, random, time, copy
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rlenvs import CharTokenizer, build_tiny_gpt
from utils import set_seed

set_seed(0)
print(f"torch {torch.__version__} | 本章实验总耗时约 3-4 分钟")
""")

md(r"""## 20.1 复盘与本章地图

先把手里的牌摆清楚（全部来自 Ch19）：

| 组件 | 状态 |
|---|---|
| 工具环境（计算器）+ 语法 `?expr=` / `#ans` / `.` | ✅ Ch19 §19.2 |
| SFT 模型 B（会工具格式，greedy ~65-85%，oracle Bo8 高一线） | ✅ Ch19 §19.3 |
| 五量仪表盘 + RAFT 负结果 | ✅ Ch19 §19.4 |
| **多轮 GRPO** | ❌ 本章的主角 |

GRPO 相对 RAFT 多了两样东西：**负样本的梯度**（错的轨迹概率被压低）和**连续加权**（好 1σ 差 −1σ 而不是 1/0）。理论上它应该能吃到更多余量。但 Ch13 的实现是单轮的——本章要解决的核心工程问题只有一个：

> **多轮轨迹里，哪些 token 算「模型的动作」？**

答案 Ch19 已经埋好：模型 emit 的算，环境注入的不算。 rollout 时记一个**蒙版**，更新时只在蒙版位置算 log π——就这么简单。

> 🤔 **先猜再跑**（本章主实验）：GRPO 训练在两种题源上各跑一段——(a) 那 80 道**已经背熟**的训练题；(b) 每轮**现生成的新题**。预测两者的测试 greedy 曲线：都涨？都不涨？还是一平一涨？如果想不清，回想策略梯度的信号是什么——**advantage 非零**才能有梯度，而组优势 = (r−r̄)/σ 要求组内**有对有错**。
>
> <details><summary>写下预测再点开</summary>
>
> 提示：背熟的题上采样成功率 ≈ 100% → 每组 6 条全对 → σ=0 → advantage 全 0 → **梯度为零，什么都学不到**。新题上成功率 ~65-75% → 组内有对有错 → 信号充沛。所以预期是 (a) 平、(b) 涨。如果你预测到了这一点，你已经理解了 on-policy RL 的一个深刻约束：**RL 教不了模型已经完全掌握的东西**——它只从「不确定」中学习。
> </details>
""")

code(r"""# 20.1.1 复刻 Ch19 的玩具基础设施（紧凑版：tokenizer/问题/工具/解码/评估）
tok = CharTokenizer()
tok.train("0123456789+=?#.")

def make_problems(n, rng):
    s = set()
    while len(s) < n:
        s.add((rng.randint(10, 99), rng.randint(10, 99), rng.randint(10, 99)))
    return sorted(s)

def calc(expr):
    parts = [p for p in expr.split('+') if p]
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return str(sum(int(p) for p in parts))

def tool_trace(a, b, c):
    s1 = a + b
    return f"?{a}+{b}={s1}?{s1}+{c}={a + b + c}#{a + b + c}."

train_probs = make_problems(80, random.Random(0))
test_probs = make_problems(40, random.Random(1))
B_DATA = [(f"{a}+{b}+{c}", tool_trace(a, b, c)) for a, b, c in train_probs]

@torch.no_grad()
def greedy_eval(model, problems):
    # 工具增强贪心解码评估（同 Ch19 eval_tool）
    ok = 0
    for A, B, C in problems:
        ids = list(tok.encode(f"{A}+{B}+{C}"))
        st, eb, ab = "start", "", ""
        for _ in range(48):
            x = torch.tensor([ids[-model.max_seq_len:]], dtype=torch.long)
            nid = int(model(x)[0, -1].argmax())
            ch = tok.itos[nid]; ids.append(nid)
            if st == "ans":
                if not ch.isdigit(): break
                ab += ch
            elif ch == '#': st = "ans"
            elif ch == '?': eb, st = "", "expr"
            elif st == "expr":
                if ch.isdigit() or ch == '+': eb += ch
                elif ch == '=':
                    r = calc(eb)
                    if r:
                        for rc in r: ids.append(tok.stoi[rc])
                    st = "normal"
        ok += int(ab == str(A + B + C))
    return ok / len(problems)

def sft_train(model, data, n_steps, lr=1.5e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(n_steps):
        batch = random.sample(data, 32)
        seqs = [tok.encode(p + r).tolist() for p, r in batch]
        L = max(len(s) for s in seqs); pad = tok.pad_id
        x = torch.full((len(seqs), L - 1), pad, dtype=torch.long)
        y = torch.full_like(x, pad); m = torch.zeros(x.shape)
        for i, s in enumerate(seqs):
            pl = len(batch[i][0])
            x[i, :len(s) - 1] = torch.tensor(s[:-1]); y[i, :len(s) - 1] = torch.tensor(s[1:])
            for j in range(len(s) - 1):
                if j + 1 >= pl: m[i, j] = 1.0
        lg = model(x)
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                             reduction='none').reshape(y.shape)
        loss = (ce * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

model = build_tiny_gpt(vocab_size=tok.vocab_size, d_model=64, n_heads=4,
                       n_layers=2, d_ff=256, max_seq_len=96)
t0 = time.time()
sft_train(model, B_DATA, 600)
acc0 = greedy_eval(model, test_probs)
print(f"SFT 完成（{time.time() - t0:.0f}s）| 测试 greedy: {acc0:.1%}")
""")

md(r"""## 20.2 多轮 rollout：带 log π 蒙版的轨迹采集

对 Ch19 的 `decode_with_tools` 做三处升级，就得到 GRPO 需要的 rollout：

1. 记录**完整 token 序列** `ids`（含环境注入的）——这是后面重算 log π 的输入
2. 记录**蒙版** `emitted`：每个位置是「模型 emit 的」（True）还是「环境注入的」（False）
3. 顺手记下**旧 log π**（采样时的）——PPO ratio 的分母（Ch09 §9.4）

奖励照旧是任务结局：答案对 +1、错 0——**整条轨迹一个标量**，信用分配全部交给 group advantage（Ch13 的哲学原样成立）。""")

code(r"""# 20.2.1 多轮 rollout 实现（Ch19 解码器 + 蒙版 + 旧 logp）
@torch.no_grad()
def rollout_track(model, prompt, max_new=48, temperature=0.9):
    # 返回 (ids, emitted 蒙版, old_logp 列表, 最终答案)
    ids = list(tok.encode(prompt))
    emitted = [False] * len(ids)      # prompt 是环境给的观察，不算动作
    old_logp = [0.0] * len(ids)
    st, eb, ab = "start", "", ""
    for _ in range(max_new):
        x = torch.tensor([ids[-model.max_seq_len:]], dtype=torch.long)
        logp = F.log_softmax(model(x)[0, -1] / max(temperature, 1e-5), dim=-1)
        nid = int(torch.multinomial(logp.exp(), 1))
        ch = tok.itos[nid]
        ids.append(nid); emitted.append(True); old_logp.append(float(logp[nid]))
        if st == "ans":
            if not ch.isdigit(): break
            ab += ch
        elif ch == '#': st = "ans"
        elif ch == '?': eb, st = "", "expr"
        elif st == "expr":
            if ch.isdigit() or ch == '+': eb += ch
            elif ch == '=':
                r = calc(eb)                     # ---- 环境接管 ----
                if r is not None:
                    for rc in r:                 # 注入的观察：emitted=False
                        ids.append(tok.stoi[rc])
                        emitted.append(False); old_logp.append(0.0)
                st = "normal"
    return ids, emitted, old_logp, ab

# 演示：跑一条轨迹，亲眼看「动作」与「观察」怎么交错
ids, emitted, old_logp, ab = rollout_track(model, "34+28+61")
pretty = "".join(ch.upper() if em else f"[{ch}]" for ch, em in zip(tok.decode(ids), emitted))
print("M=模型动作  [] =环境注入的观察：")
print(" ", pretty[:100])
n_act = sum(emitted)
print(f"\\n轨迹总长 {len(ids)} token，其中模型动作 {n_act} 个、环境观察 {len(ids) - n_act} 个")
print(f"（GRPO 的 log π 只对那 {n_act} 个动作计算——观察是 state，不是 action）")
""")

md(r"""## 20.3 GRPO 更新步：Ch13 的公式，多轮的轨迹

更新步和 Ch13 的 `_ppo_step` 一一对应，只有轨迹来源不同：

1. **组优势**：同一 prompt 的 G 条轨迹，`Â = (r − r̄)/(σ_r + ε)`——Ch13 原文公式
2. **PPO-clip**：ratio = π_new/π_old（只在动作 token 上），clip(1±ε)
3. **KL 锚**：对 SFT 参考模型的解析 KL——Ch12 §12.3 的缰绳，防止 RL 把语言能力练坏

```python
for 每条轨迹:                       # 多轮轨迹 = Ch13 的 response
    logπ_new = model(完整序列)[动作位置]   # 蒙版 gather
    ratio = exp(logπ_new - logπ_old)
    loss += -min(ratio·Â, clip(ratio)·Â) + β·KL(π‖π_ref)
```""")

code(r"""# 20.3.1 Agentic GRPO 更新步（~35 行，Ch13 公式的多轮化身）
def grpo_step(model, ref, problems, G=6, clip_eps=0.2, lr=1e-4, beta_kl=0.02):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # 1) 交互式 rollout：每题 G 条轨迹
    trajs = []
    for A, B, C in problems:
        p = f"{A}+{B}+{C}"
        for _ in range(G):
            ids, em, lp, ab = rollout_track(model, p)
            trajs.append((ids, em, lp, 1.0 if ab == str(A + B + C) else 0.0))
    # 2) 组优势（Ch13 原文公式）
    advs = []
    for i in range(0, len(trajs), G):
        grp = trajs[i:i + G]
        rr = [t[3] for t in grp]; mu = sum(rr) / G
        sd = (sum((x - mu) ** 2 for x in rr) / G) ** 0.5
        advs.extend((t[3] - mu) / (sd + 1e-4) for t in grp)
    # 3) 更新：clip 目标 + KL 锚，只在动作 token 上
    opt.zero_grad()
    for (ids, em, lp, r), adv in zip(trajs, advs):
        idx = [t for t in range(1, len(ids)) if em[t]]   # 动作位置（蒙版！）
        if not idx: continue
        ids_t = torch.tensor([ids[:-1]], dtype=torch.long)
        logp = F.log_softmax(model(ids_t)[0], dim=-1)
        pos = torch.tensor([t - 1 for t in idx])
        act = torch.tensor([ids[t] for t in idx])
        new_lp = logp[pos, act]
        old = torch.tensor(lp)[torch.tensor(idx)]
        ratio = torch.exp((new_lp - old).clamp(-20, 20))
        adv_t = torch.tensor(adv)
        obj = torch.min(ratio * adv_t, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t)
        with torch.no_grad():
            rlogp = F.log_softmax(ref(ids_t)[0], dim=-1)
        kl = (rlogp.exp() * (rlogp - logp)).sum(-1).mean()   # 解析 KL（Ch09 k3）
        loss = -obj.sum() + beta_kl * kl * len(idx)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()
    return sum(t[3] for t in trajs) / len(trajs)            # 本轮 rollout 成功率
""")

md(r"""## 20.4 实验一：在背熟的题集上训练（预期：纹丝不动）

用 Ch19 那 80 道**SFT 已经背熟**的训练题做 GRPO——每轮抽 16 道、各采 G=6 条。""")

code(r"""# 20.4.1 实验一：固定（背熟的）题集
ref = copy.deepcopy(model)          # KL 锚 = SFT 参考（Ch12 的缰绳）
acc_fixed = [greedy_eval(model, test_probs)]
t0 = time.time()
for it in range(8):
    bar = grpo_step(model, ref, random.sample(train_probs, 16))
    acc_fixed.append(greedy_eval(model, test_probs))
    print(f"iter {it + 1} | rollout 成功率 {bar:.0%} | 测试 greedy {acc_fixed[-1]:.1%}"
          f" | {time.time() - t0:.0f}s")
""")

md(r"""**读数**：rollout 成功率接近 100%、测试曲线平的——**梯度为零，因为组内没有方差**。

这不是 bug，是策略梯度的数学：`Â = (r−r̄)/σ`，当 σ=0 时 Â≡0。RL 的学习信号**不是来自奖励本身，而是来自组内的不一致**——同一条题一会儿对一会儿错，才有「对的轨迹该加分、错的该减分」可说。模型完全掌握（或完全不会）的任务上，GRPO 一筹莫展。

> 这个「失败」在真实工程里天天出现：RLHF 时对太简单/太难的任务，组内奖励全同、advantage 为零，等于白烧 rollout。DeepSeek-R1 的做法之一是**按难度筛题**（让模型处于「够得着但不稳」的区间）——和给学生布置作业是同一个道理。

## 20.5 实验二：无限题海（自生成课程）

现在换题源：**每轮从任务分布现生成 16 道新题**（我们的任务是可生成的——真实场景对应「无限题库」或「可随机参数化的环境」）。模型没背过它们，组内自然有对有错——信号回来了。

> 🤔 如果你 §20.1 的预测是「一平一涨」，这里就是兑现时刻；顺便猜猜幅度：8 个 iteration 能涨几个点？""")

code(r"""# 20.5.1 实验二：每轮现生成新题（on-policy 自生成课程）
gen_rng = random.Random(42)
def fresh_problems(n):
    return make_problems(n, gen_rng)

acc_fresh = [greedy_eval(model, test_probs)]
t0 = time.time()
for it in range(30):
    bar = grpo_step(model, ref, fresh_problems(16), lr=1.2e-4, beta_kl=0.01)
    if it % 3 == 2 or it == 29:
        acc_fresh.append(greedy_eval(model, test_probs))
        print(f"iter {it + 1:2d} | 新题成功率 {bar:.0%} | 测试 greedy {acc_fresh[-1]:.1%}"
              f" | {time.time() - t0:.0f}s")
""")

code(r"""# 20.5.2 两条曲线放一起：固定题集 vs 无限题海
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(range(len(acc_fixed)), acc_fresh[:len(acc_fixed)] if False else acc_fixed,
             'o-', color='#d62728', label='固定题集（背熟）')
axes[0].set_title('实验一：固定题集 —— 没有信号')
axes[1].plot(range(0, 31, 3), acc_fresh, 'o-', color='#2ca02c', label='无限题海（每轮新题）')
axes[1].set_title('实验二：自生成课程 —— 涨')
for ax in axes:
    ax.axhline(acc0, color='gray', ls='--', alpha=0.7, label=f'SFT 基线 {acc0:.0%}')
    ax.set_xlabel('GRPO iteration'); ax.set_ylabel('测试 greedy 正确率')
    ax.set_ylim(0, 1.0); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md(r"""### 读结果：on-policy RL 的两个本质

（具体数字每次运行略有波动，形状是稳定的。）

1. **信号来自不确定性**。实验一与实验二的唯一区别是题目的「新鲜度」，结果天壤之别——RL 不是「把奖励最大化」这么简单，它优化的是**组内相对差距**，而差距只在「够得着但不稳」的区间存在。练习太简单或太难都学不到东西，对模型和对人一样。
2. **训练集可以无限自生成**。SFT 和 RAFT 被固定数据集锁死（80 道题背完就到头）；GRPO 每轮从任务分布现抽题——**任务分布无限，课程就无限**。这就是 on-policy 的深意：模型在自己的能力边缘上持续给自己出题、持续从对错对比中改进。

> 🌍 **真实世界**：DeepSeek-R1-Zero 训练时面对的是可以无限生成的数学/代码题；Search-R1 的环境是每次查询都不同的真实搜索；AlphaZero（Ch17）的自我对弈更是彻底的无限课程——**RL 三十年的主线之一，就是把「训练数据」变成「训练环境」**。你刚在 8 万参数的玩具上亲手复现了这条主线。

还有一个值得注意的细节：实验二涨到的水平，大致就是 Ch19 里 oracle best-of-8 标出的余量上限附近——**GRPO 确实吃到了 RAFT 吃不到的那份余量**（靠负样本梯度 + 连续加权），兑现了上一章的悬念。
""")

md(r"""## 20.6 从单轮到多轮：三个结构性变化与四大挑战

现在回头看，Ch13 的 GRPO 搬到 agent 世界**目标函数一个字没改**，改的只是三件事：

| | Ch13 单轮 GRPO | Agentic GRPO（本章） |
|---|---|---|
| 轨迹 | 一次采样 prompt→response | **交互循环**：模型段 ⇄ 环境段（`rollout_track`） |
| 状态 | prompt 固定 | 每轮包含**工具返回的观察**（上下文随交互增长） |
| log π 的计算 | 对 response token 求和 | 只对**模型 emit 的 token** 求和——**环境注入的观察不算**（`emitted` 蒙版） |

第三条是唯一容易踩的坑：观察是 state 的一部分，把环境注入 token 的 log-prob 算进 π，策略梯度就错了。

**四大挑战**（每一条都是 2024-2025 的活跃论文方向）：

1. **长视野信用分配**——10 轮工具调用前的那次搜索，对最终答对有多大功劳？GAE over turns、PRM（Ch16）都是候选答案
2. **奖励稀疏 + 可作弊**——结局奖励太粗，过程奖励又容易被 hack（Ch11 的 Goodhart 在这里复发）
3. **rollout 昂贵**——一条轨迹 = 多次 LLM 推理 + 多次工具调用；本章玩具上一条轨迹 ~40 次 forward，真实 agent 一次搜索就是几百 ms
4. **环境不可复现/不可仿真**——真实搜索引擎每次结果不同，同一个动作两次奖励不同，方差暴涨

## 20.7 前沿地图 + 全书终章

> 🌍 **这个方向的名字就叫 Agentic RL**。几个值得认识的坐标：
>
> - **ReAct**（Yao et al. 2022）——Thought-Action-Observation 循环开山，Ch19 的 `?expr=结果` 是它的最小化身
> - **STaR / RAFT**（2022-2023）——Ch19 §19.4 的方法本体
> - **Search-R1**（Jin et al. 2025）——搜索引擎当工具、答案正确性当 reward、GRPO 当优化器；本章玩具换掉工具就是它的骨架
> - **ToolRL**（2025 系列）——「什么时候调工具」本身作为学习目标（Ch19 练习 2 的放大版）
> - **OpenAI o3 / DeepSeek-R1**——工业级推理 agent；公开报告描述的「反思、回溯、自我纠错」涌现，就是无限课程 + 结局奖励的规模化版本
>
> 诚实的现状判断：**单轮 RLHF 配方已标准化（DPO/GRPO），多轮 Agentic RL 还在混战**——奖励设计、过程奖励防 hack、长视野 credit，都没有共识答案。「刚刚开始」的另一个名字，是**入场的好时机**。

### 终章：你走完了什么

> 从 Ch00 网格里「策略决定一切」的懵懂出发——你推过贝尔曼方程的递归，踩过 deadly triad 的发散，
> 掷过策略梯度的飞镖，拧过 GAE 的 λ 旋钮，给 PPO 装过刹车片；
> 然后你把一个 token 变成一个动作，让语言模型伸出手拿起工具，
> 最后在今天，让它在无限的题海里自己给自己上课。
> 从 5×5 的网格到 agent 的世界循环，贯穿一切的还是 Ch00 那句话：
> **动作改变世界，世界反馈信息，奖励告诉你做得好不好。**

**能力清单**（可以写进简历/研究陈述的那种）：

- 从零实现并验证了 11+ 个 RL 算法（表格方法 → DQN → 策略梯度 → PPO → RLHF/GRPO/DPO → Agentic GRPO）
- 亲手跑通 SFT → RM → RLHF/GRPO 的完整后训练流水线（CPU 上的忠实缩小版）
- 能读懂并改进 InstructGPT / DeepSeek-R1 / Search-R1 论文的方法部分
- 掌握 agent 评估的五量仪表盘与「先量天花板再选工具」的工程判断

**下一步去哪**：

| 目标 | 路线 |
|---|---|
| 做研究 | 读 Search-R1 / ToolRL + 最近 3 个月 arXiv，挑四大挑战之一进攻 |
| 做工程 | 申请 GPU，把 §15.3 配方在 1.5B 模型 + GSM8K 上跑通；学 vLLM/TRL 做生产化 |
| 做 Agent 产品 | 在 LLM 应用里埋「采样-打分-过滤」回路（本章的仪表盘直接可用） |
| 继续学习 | Datawhale《Hello-Agents》—— 从本章走向完整的多智能体工程 |

## 20.8 📝 练习

### 练习 1（必做）：G 扫描

`grpo_step` 的 G=6 改成 2 / 4 / 12（iteration 数等比调整保持 rollout 预算），对比最终正确率与训练稳定性。预期：G=2 组内方差估计太噪、不稳定；G=12 每组信号更稳但每轮题数变少——这就是 Ch13 练习 1 在多轮世界的重演。

### 练习 2（选做）：ToolRL 设定

给 reward 加工具代价：`r = 答案正确 − 0.1 × 工具调用次数`。改造 `rollout_track`（返回工具计数）和奖励行即可。观察模型会不会学会「两位数直加、需要进位才用工具」的分工——这就是 2025 年 ToolRL 论文的核心实验，你现在有全部零件。

> 📖 做完练习后，去根目录 STUDY_GUIDE.md 做 Ch20 的自测题——然后，合上这本书。

## 参考文献

- Jin, B. et al. *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning* (2025)
- Qian, C. et al. *ToolRL: Reward is All Tool Learning Needs* (2025)
- DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability via RL* (2025)
- Schulman, J. et al. *Proximal Policy Optimization Algorithms* (2017)——本章更新步的直系祖先
""")


if __name__ == "__main__":
    _nb.write("ch20_agentic_grpo.ipynb")
