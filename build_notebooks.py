"""Notebook 生成器：把每个 notebook 的内容用 Python 列表描述，然后写出 .ipynb。

用法：
    python build_notebooks.py                # 重建本脚本注册的 15 个 notebook + 5 个 solutions
    python build_notebooks.py --chapters 06-09   # 只重建 Phase 2
    python build_notebooks.py --chapters 00-05   # 只重建 Phase 1
    python build_notebooks.py --chapter 13       # 只重建单个章节
    python build_notebooks.py --list             # 列出全部章节，不写文件

Ch00-05 + Ch10：在 build_notebooks.py 内 inline 定义（Cell 元组 + build_notebook）。
Ch06-09 + Ch11-14：由独立的 build_chXX.py 生成，本文件通过 import + adapter 调用，
    避免代码重复。各 build_chXX.py 的接口不统一（Ch06-09 用模块级 `cells` 列表 + dict
    风格；Ch10 用 ch10() 函数返回 List[Cell]；Ch11-14 用模块级 `cells` + nbformat dict），
    build_notebooks.py 提供适配层。
Ch15：capstone 项目，目前手维护（无 build_ch15.py），本脚本不动它。
Ch16-18：有各自的 build_chXX.py（Ch17/18 还配 chXX_content.txt），但未注册进
    本脚本，需要单独运行对应脚本重建。

每个 cell 是 (type, source) 元组，type 为 'md' 或 'code'。
source 是字符串（自动按行切）。
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Tuple

# 核心 helper（md/code/build_notebook/save）与全部 build_chXX.py 共享自 nb_helpers.py，
# 保证整套 notebook 的写出格式与 metadata 完全一致。
from nb_helpers import Cell, md, code, build_notebook, save


# =============================================================================
# 适配层：包装外部 build_chXX.py
# =============================================================================
# 各 build_chXX.py 的接口有两种：
#   (1) Ch06-09, Ch11-14：模块级 `cells` 列表，每个 cell 是 nbformat v4 dict
#       （含 cell_type / id / metadata / source / [outputs, execution_count]）
#   (2) Ch10：定义 `ch10()` 函数，返回 List[Cell]（与本文件 inline 风格一致）
#
# 适配策略：把两种都归一成 List[Cell] 元组（type, source_str），再交给本文件
# 已有的 build_notebook() 写出，保证 16 章 notebook 的 metadata / 字段结构完全一致。


def _load_chapter_module(name: str):
    """以 build_notebooks.py 所在目录为 cwd，import 一个 build_chXX 模块。

    多次调用安全（Python 的 sys.modules 缓存会保证只执行一次模块体）。
    各 build_chXX.py 的写盘都在 `if __name__ == "__main__"` 守卫内，
    因此 import **没有写盘副作用**（历史上的"双重写盘"已随 nb_helpers.py
    统一重构消除）。
    """
    # 把本文件所在目录加到 sys.path 头部（保证 import build_chXX / nb_helpers 能找到）
    root = Path(__file__).resolve().parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module(name)


# ---- 各章 wrapper -----------------------------------------------------------
# 每个 wrapper 返回 List[Cell] 元组，统一接口，便于加入 NOTEBOOK_BUILDERS。

def _cells_from_module_dict(module_name: str) -> List[Cell]:
    """从模块级 `cells` 列表（dict 风格）转成 List[Cell] 元组。

    用途：Ch06-09, Ch11-14。把 nbformat dict 转回 (type, source_str) 元组，
    再交给本文件的 build_notebook()，保证和 Ch00-05 走同一条写出路径，
    最终产物的 metadata / 字段结构完全一致。
    """
    mod = _load_chapter_module(module_name)
    raw_cells = list(mod.cells)  # 浅拷贝，避免修改模块全局
    tuple_cells: List[Cell] = []
    for c in raw_cells:
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        ctype = c.get("cell_type", "markdown")
        if ctype == "markdown":
            tuple_cells.append(("md", src))
        else:
            tuple_cells.append(("code", src))
    return tuple_cells


def ch06() -> List[Cell]:
    """Ch06 DQN + 函数逼近（来自 build_ch06.py）。"""
    return _cells_from_module_dict("build_ch06")


def ch07() -> List[Cell]:
    """Ch07 策略梯度定理（来自 build_ch07.py）。"""
    return _cells_from_module_dict("build_ch07")


def ch08() -> List[Cell]:
    """Ch08 Actor-Critic + GAE（来自 build_ch08.py）。"""
    return _cells_from_module_dict("build_ch08")


def ch09() -> List[Cell]:
    """Ch09 TRPO + PPO（来自 build_ch09.py）。"""
    return _cells_from_module_dict("build_ch09")


def ch10() -> List[Cell]:
    """Ch10 TinyGPT（来自 build_ch10.py 的 ch10() 函数）。"""
    mod = _load_chapter_module("build_ch10")
    return list(mod.ch10())


def ch11() -> List[Cell]:
    """Ch11 Reward Modeling（来自 build_ch11.py）。"""
    return _cells_from_module_dict("build_ch11")


def ch12() -> List[Cell]:
    """Ch12 RLHF-PPO / InstructGPT（来自 build_ch12.py）。"""
    return _cells_from_module_dict("build_ch12")


def ch13() -> List[Cell]:
    """Ch13 GRPO / DeepSeek-R1（来自 build_ch13.py）。"""
    return _cells_from_module_dict("build_ch13")


def ch14() -> List[Cell]:
    """Ch14 DPO / KTO（来自 build_ch14.py）。"""
    return _cells_from_module_dict("build_ch14")

def ch19() -> List[Cell]:
    """Ch19 Agentic RL（来自 build_ch19.py）。"""
    return _cells_from_module_dict("build_ch19")

def ch20() -> List[Cell]:
    """Ch20 Agentic GRPO 实战（来自 build_ch20.py）。"""
    return _cells_from_module_dict("build_ch20")


# Ch15 没有 build_ch15.py（capstone 项目，手维护）。
# 它的 notebook notebooks/ch15_capstone.ipynb 由人工维护，本脚本不重建。


# =============================================================================
# Notebook 内容定义（Ch00-05 inline）
# =============================================================================

def ch00() -> List[Cell]:
    """Ch00 环境搭建 + RL 全景 + ClickWorld 演示。"""
    return [
        md("""# 第 0 章：环境搭建、学习地图与"什么是强化学习"

欢迎来到强化学习的世界！

无论你的志向是读懂 DeepSeek-R1 那篇论文、亲手微调一个自己的 LLM，还是想真正理解"RLHF 到底在优化什么"而不是停留在名词层面——这套教材都为你铺了一条完整的路：从 5×5 的小网格出发，一路走到能亲手实现 GRPO 训练语言模型。

> 这一章不教任何算法，目的是让你**直观看到一次完整的 RL 过程**，并确认所有环境就绪。

## 学习目标

读完本章后你应该能：

1. 描述强化学习的 **agent-environment loop**
2. 区分**监督学习 / 强化学习 / RLHF** 三种范式
3. 跑通本课程的第一个环境 `ClickWorld`，看到智能体在交互中"学习"
4. 知道接下来 18 章分别讲什么、以及一条"快速通道"
"""),

        code("""# 自动设置 sys.path，让 notebook 能找到根目录下的 rlenvs/ 和 utils/
import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 载入 numpy 和 matplotlib
# 在 Jupyter 里 IPython 自动激活 inline 后端，图会嵌在 cell 输出里。
# 不要用 matplotlib.use() 手动切，那会覆盖 IPython 的默认设置导致图不显示。
import numpy as np
import matplotlib.pyplot as plt

print(f"Python: {sys.version.split()[0]}")
print(f"numpy : {np.__version__}")
print(f"matplotlib: {plt.matplotlib.__version__}")
print(f"backend : {plt.get_backend()}  (应是 inline / widget 才能显示图)")
"""),

        md("""## 0.1 环境自检

下面这个 cell 会检查我们后续章节需要的所有依赖。如果某行显示 **❌**，先在终端执行：

```bash
pip install -r requirements.txt
```

然后再回来运行此 cell。
"""),

        code("""import importlib
checks = [
    ('numpy',       'numpy'),
    ('matplotlib',  'matplotlib'),
    ('scipy',       'scipy'),
    ('ipywidgets',  'ipywidgets'),
    ('ipympl',      'ipympl'),
    ('tqdm',        'tqdm'),
    ('torch',       'torch（Phase 2 起需要，可选）'),
]

ok = True
for mod, desc in checks:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        print(f"  [✓] {desc:<35} {ver}")
    except ImportError:
        print(f"  [✗] {desc:<35} 未安装")
        ok = False

print()
print("全部就绪 ✓" if ok else "存在缺失依赖，请先 pip install")
"""),

        md("""## 0.1b Jupyter 速成（第一次用 notebook 必读）

如果你没用过 Jupyter notebook，花 3 分钟了解下面几件事，能避免后面 90% 的"跑不通"：

| 操作 | 怎么做 | 说明 |
|---|---|---|
| 运行 cell | 选中后按 `Shift + Enter` | 运行当前 cell 并跳到下一个 |
| 在上方/下方插入 cell | 先按 `Esc`，再按 `A` / `B` | A = above, B = below |
| 删除 cell | 先按 `Esc`，再连按 `D D` | |
| 重启并全部重跑 | 菜单 Kernel → Restart Kernel and Run All Cells | **最常见的修复手段** |

三个关键认知：

1. **cell 必须按顺序运行**：后面的 cell 依赖前面 cell 定义的变量。跳着运行报 `NameError`，先回头重跑前面的 cell。
2. **kernel 的记忆 ≠ 你看到的代码**：改了 cell 但没重跑，执行的还是旧代码。出现诡异结果时，Restart Kernel and Run All Cells。
3. **每章开始前建议 Restart & Run All**：这是验证一章真的能在你机器上跑通的标准方法，本教材每章都控制在 10 分钟以内。

> 📖 **怎么检验自己学会了？** 根目录的 **`STUDY_GUIDE.md`** 给全部 19 章各配了自测题（答案可折叠）。建议每学完一章就去做对应的自测，答不上来再回头复习。
"""),

        md("""## 0.2 什么是强化学习？

先从一个你熟悉的场景说起。假设你要教一个完全不懂规则的朋友下五子棋：

- **告诉他规则**——这是编程：写死每一步的判断逻辑；
- **给他看一千盘高手的棋谱让他模仿**——这是监督学习：每一步都有"标准答案"；
- **只让他上桌去下，赢了才有糖吃**——这就是强化学习。

第三种方式看起来最残忍：没有答案、没有人告诉他哪步错了，只有终局的胜负。但也只有这条路，能让他**发现棋谱上没有的下法**——因为模仿永远有一个看不见的天花板：老师水平。

强化学习和监督学习最核心的区别，一句话概括：

> **监督学习**有标准答案（标签），**强化学习**只有延迟的、稀疏的奖励。

| | 监督学习 | 强化学习 |
|---|---|---|
| 数据 | `(x, y)` 标签对 | `(s, a, r, s')` 转移序列 |
| 反馈 | 即时、确定 | 延迟、稀疏、随机 |
| 目标 | 拟合 `y = f(x)` | 最大化累计奖励 `Σ γ^t r_t` |
| 决策 | 一次推理 | 序列决策、当前动作影响未来 |

回到下棋的例子：AI 自我对弈赢了 +1 输了 -1，但**是第 3 步的妙手还是第 87 步的昏招决定了这盘棋？** 棋局结果里没有说。把最终的奖励"分账"回每一个动作——这就是 **credit assignment（信用分配）问题**，它将贯穿这套教材的始终：从 Ch04 的 TD error，到 Ch08 的 GAE，再到 Ch13 GRPO 的 group advantage，全都在用不同的方式回答这一个问题。

### Agent–Environment Loop

```
        action a_t
   ┌─────────────────┐
   │                 ▼
[ Agent ]      [ Environment ]
   ▲                 │
   │                 │
   └─ state s_{t+1}, reward r_{t+1} ─┘
```

每一时刻 $t$：

1. Agent 根据当前状态 $s_t$，按策略 $\\pi(a|s)$ 选择动作 $a_t$
2. Environment 接收 $a_t$，转移为新状态 $s_{t+1}$，并给一个标量奖励 $r_{t+1}$
3. Agent 的目标是最大化**期望累计奖励**：

$$
J(\\pi) = \\mathbb{E}_\\pi\\left[ \\sum_{t=0}^{\\infty} \\gamma^t r_{t+1} \\right]
$$

其中 $\\gamma \\in [0, 1]$ 是**折扣因子**（Ch02 会详细讲为什么要折扣）。
"""),

        md("""## 0.3 玩具演示：ClickWorld

现在让我们真的"动手"看一下这个循环。

**`ClickWorld`** 是一个 $10 \\times 10$ 的网格：
- 智能体（蓝点）从**左上角** $(0, 0)$ 出发
- 目标在**右下角** $(9, 9)$（金色，奖励 +1）
- 中间有一个陷阱 $(5, 5)$（红色，奖励 -1）
- 抵达目标 +1、踩到陷阱 -1、其他 0

我们对比两种"策略"：

1. **纯随机游走**：每步等概率往上/下/左/右走（撞墙就原地不动）
2. **贪心策略**：每步朝目标方向走

下面的演示是**逐帧动画**。matplotlib 的 jshtml 播放器自带这些控件：

| 按钮 | 功能 |
|---|---|
| ▶ / ⏸ | 播放 / 暂停 |
| − / + | 减慢 / 加快播放速度 |
| ↻ | 循环播放（相当于自动重跑） |
| 进度条 | 拖动到任意帧 |

**想换种子重跑**：改下面 cell 里的 `SEED = 0`（任意整数），然后 Shift+Enter 重跑 cell。
**想调初始速度**：改 `INTERVAL_MS = 200`（毫秒，越大越慢）。
"""),

        code("""from rlenvs import ClickWorld
import utils  # 触发中文字体配置（utils/viz.py 顶部设了 Microsoft YaHei 等）
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle
from IPython.display import HTML


# ---------- 你可以调的两个参数 ----------
SEED = 0              # 改成任意整数，重跑 cell 看不同结果
INTERVAL_MS = 200     # 帧间隔（毫秒），越大越慢；推荐 100~500
# ------------------------------------

GRID_SIZE = 10
START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)  # (9, 9)，最远的对角
TRAP = (5, 5)


# ---------- 工具函数：跑轨迹 + 生成动画 ----------
def collect_trajectory(env, policy_fn, n_steps, start=START):
    \"\"\"按 policy_fn 跑 n_steps 步，返回 (states, rewards, reached_goal)。\"\"\"
    env.reset()
    env.state = start  # ClickWorld.state 是普通属性，直接赋值即可（默认 reset 是随机起点）
    env.t = 0
    env.reward_history = []
    env.trajectory = [env.state]
    states = [env.state]
    rewards = [0.0]
    reached = False
    for _ in range(n_steps):
        a = policy_fn(env.state)
        _, r, done, _ = env.step(a)
        states.append(env.state)
        rewards.append(r)
        if done:
            reached = True
            for _ in range(3):  # 终止后多停 3 帧让画面定住
                states.append(env.state); rewards.append(0.0)
            break
    return states, rewards, reached


def animate_clickworld(env, states, rewards, interval_ms=200):
    \"\"\"把 (states, rewards) 渲染成 FuncAnimation。\"\"\"
    fig, ax = plt.subplots(figsize=(5, 5))
    size = env.size

    def draw(k):
        ax.clear()
        for i in range(size + 1):
            ax.axhline(i, color='lightgray', linewidth=0.8)
            ax.axvline(i, color='lightgray', linewidth=0.8)
        for (r, c) in env.penalties:
            ax.add_patch(Rectangle((c, r), 1, 1, color='crimson', alpha=0.5))
            ax.text(c + 0.5, r + 0.5, 'X', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
        if env.goal is not None:
            gr, gc = env.goal
            ax.add_patch(Rectangle((gc, gr), 1, 1, color='gold', alpha=0.7))
            ax.text(gc + 0.5, gr + 0.5, 'G', ha='center', va='center',
                    fontsize=14, fontweight='bold')
        # 起点标记
        sr, sc = START
        ax.text(sc + 0.5, sr + 0.5, 'S', ha='center', va='center',
                fontsize=11, color='green', fontweight='bold')
        # 走过的轨迹
        if k > 0:
            ys = [r + 0.5 for r, c in states[:k + 1]]
            xs = [c + 0.5 for r, c in states[:k + 1]]
            ax.plot(xs, ys, '-', color='steelblue', alpha=0.6, linewidth=1.5)
        # 当前位置
        r, c = states[k]
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.3, color='navy'))
        cum = sum(rewards[:k + 1])
        ax.set_xlim(0, size); ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'step {k}   累计奖励 = {cum:.2f}')

    anim = animation.FuncAnimation(
        fig, draw, frames=len(states),
        interval=interval_ms, blit=False, repeat=True,
    )
    plt.close(fig)
    return anim


def random_policy(s):
    return int(np.random.randint(4))


# ---------- 跑一遍随机游走并显示动画 ----------
np.random.seed(SEED)  # 让随机策略也可复现（ClickWorld 内部有自己的 rng，但 random_policy 用全局）
env = ClickWorld(size=GRID_SIZE, seed=SEED)
env.set_goal(GOAL)
env.set_penalty(TRAP)

N_STEPS = 80  # 给随机游走 80 步看能不能到
states, rewards, reached = collect_trajectory(env, random_policy, n_steps=N_STEPS)
status = f"第 {len(states) - 4} 步到达目标" if reached \\
         else f"{N_STEPS} 步内未到达目标"
print(f"随机游走 {N_STEPS} 步：{status}")
print(f"累计奖励 = {sum(rewards):.2f}")
anim = animate_clickworld(env, states, rewards, interval_ms=INTERVAL_MS)
HTML(anim.to_jshtml())
"""),

        md("""看完了随机游走，你应该看到：**纯随机的智能体在网格里乱转**——有时绕回起点、有时撞墙、偶尔接近目标又被带偏。即便 80 步给它够长，也经常到不了对角的目标。

要让智能体"聪明地"走向目标，我们需要给它一个**策略**——这就是接下来 18 章要研究的全部内容。

下面我们用一个简单的"贪心策略"演示：每步朝目标方向（曼哈顿距离最短的方向）走。

> 🤔 **先猜再跑**：运行下面 cell 之前，写下两个预测——
> 1. 贪心策略大约多少步到达目标？（随机游走 80 步都没到）
> 2. 更关键的问题：它的**净奖励**（累计奖励总和）是多少？
>
> <details><summary>写下猜测后再点开对照直觉</summary>
>
> 提示：回想一下地图上除了目标，还有什么格子会**扣分**？"每步都朝目标走"的贪心路线，会不会路过它？
> 猜完再跑——猜错的地方，就是你接下来要学的东西。
> </details>
"""),

        code("""# 定义一个朝 goal 走的贪心策略
# 注意 ClickWorld 的动作约定：0=上, 1=下, 2=左, 3=右（和 GridWorld 不一样！）
def greedy_policy(state, goal=GOAL):
    \"\"\"
    朝 goal 方向走的贪心策略：优先拉近行/列差距更大的那个轴。
    ClickWorld actions: 0=上, 1=下, 2=左, 3=右
    \"\"\"
    dr = goal[0] - state[0]   # 正：目标在下方
    dc = goal[1] - state[1]   # 正：目标在右方
    if abs(dc) > abs(dr):
        return 3 if dc > 0 else 2   # 朝右 / 朝左
    elif dr != 0:
        return 1 if dr > 0 else 0   # 朝下 / 朝上
    return 0  # 已到目标


# 重新建一个干净的环境（避免上面随机游走污染状态）
env = ClickWorld(size=GRID_SIZE, seed=SEED)
env.set_goal(GOAL)
env.set_penalty(TRAP)

states, rewards, reached = collect_trajectory(env, greedy_policy, n_steps=30)
n_used = len(states) - 4 if reached else len(states) - 1
print(f"贪心策略：{n_used} 步到达目标，累计奖励 = {sum(rewards):.2f}")
anim = animate_clickworld(env, states, rewards, interval_ms=INTERVAL_MS)
HTML(anim.to_jshtml())
"""),

        md("""看，**有策略**比**没策略**强多了——贪心策略 18 步直达目标。

但是！注意一个细节：贪心策略一路上**踩了陷阱 (5, 5)**，扣了 1 分；最后到达目标 +1 分；**净奖励 = 0**。

如果有个策略能**绕开陷阱同时还能到目标**，它的净奖励会是 +1——比贪心更好。

这就引出了 RL 的核心问题：

> **怎么自动找到一个能拿到尽可能多奖励的策略 $\\pi$？**

注意是**"尽可能多奖励"**，不是"能到目标"。也许绕远路避开陷阱比直走更好；也许中间还有更复杂的权衡。**怎么自动找出这种策略**，就是接下来 18 章的全部内容。"""),

        md("""## 0.4 学习地图

先感受一下你即将经历的旅程：

> 从老虎机的「眼前一步」，到 MDP 的「一生长远」；从动态规划的「全知全能」，
> 到 TD 学习的「摸着石头过河」；从表格的「一格一格」，到神经网络的「举一反三」；
> 最后抵达这条路的终点——把「生成一个 token」变成「做一个决策」，
> 用强化学习微调一个语言模型。这就是从 Ch01 到 Ch13 的全部故事线。

我们用 5 个 Phase（共 21 章）带你从零基础到能用 PPO / GRPO 训练 LLM、再到研究前沿与 Agentic RL：

```
Phase 1：经典 RL 基础（你现在在这里）
├── Ch00 环境搭建 + 全景
├── Ch01 多臂老虎机：探索 vs 利用
├── Ch02 MDP + 贝尔曼方程：RL 的数学语言
├── Ch03 动态规划：当你"知道一切"时怎么求解
├── Ch04 TD 学习：从样本中学习
└── Ch05 Q-learning / SARSA：第一个完整的控制算法

Phase 2：策略梯度 + PPO
├── Ch05b PyTorch 速成（没用过 PyTorch 的读者，进 Ch06 前先读）
├── Ch06 DQN + 函数逼近
├── Ch07 策略梯度定理
├── Ch08 Actor-Critic + GAE
└── Ch09 TRPO + PPO：现代 RL 的中流砥柱

Phase 3：LLM RLHF + GRPO（终极目标）
├── Ch10 从零搭 TinyGPT
├── Ch11 Reward Modeling
├── Ch12 RLHF-PPO (InstructGPT 配方)
├── Ch13 GRPO (DeepSeek-R1 的核心)
├── Ch14 DPO / KTO
└── Ch15 终局项目

Phase 4：研究前沿
├── Ch16 PRM（过程奖励模型）
├── Ch17 Self-Play + Constitutional AI / RLAIF
└── Ch18 Offline RL（CQL / IQL / Decision Transformer）

Phase 5：通往 Agentic RL（新兴方向）
├── Ch19 Agent 基础：让 LLM 学会使用工具（工具增强解码 + best-of-N + RAFT）
└── Ch20 Agentic GRPO 实战：多轮强化学习 + 无限题海（全书终章，呼应 Ch00 的 agent loop）
```

### 🏁 Fast-track 路径（如果你赶时间）

如果你已经有 RL 基础、想尽快到 LLM RLHF 部分：

**Ch00 → Ch01 → Ch05 → Ch07 → Ch09 → Ch13**（约 20 小时直达 GRPO）

> 快速通过 Ch06-09 需要会 PyTorch——没写过的话把 **Ch05b** 插进 fast-track 里，多花一小时值得。

### 这条路对应工业界的什么？

Phase 3 学到的东西，正是 GPT / DeepSeek 这类模型出厂前的真实工序。一张表帮你建立方位感（现在看不懂没关系，学完 Ch15 再回来会心一笑）：

| 工业界流水线 | 本教材 | 你将亲手做的事 |
|---|---|---|
| 预训练（海量文本，下一个词预测） | Ch10 TinyGPT | 从零搭一个 mini-GPT 并在语料上预训练 |
| 监督微调 SFT（指令-回答对） | Ch10 §10.7 | 只在 response 上算 loss，教模型「回答问题」 |
| 奖励建模 RM（人类偏好 → 分数） | Ch11 | 用 Bradley-Terry 从偏好对学出打分器 |
| RLHF / GRPO（用 RM 信号做 RL） | Ch12 / Ch13 | PPO 与 GRPO 两套配方各训一遍 |
| 免 RL 的对齐（直接偏好优化） | Ch14 DPO/KTO | 不 rollout 也能对齐的捷径 |
| 前沿（过程奖励 / 自我博弈 / 离线 RL） | Ch16-18 | 打开研究论文的钥匙 |
"""),

        md("""## 0.5 怎么使用这套教材

每个 notebook 都遵循这个结构：

1. **学习目标**：开头 3-5 条 bullet
2. **概念 + 数学推导**：LaTeX 公式，关键证明放在可折叠 `<details>` 块里
3. **数值验证**：关键公式后面有代码 cell，用数值方法验证
4. **交互式 widget**：滑块调参，实时看效果
5. **从零实现**：你自己写代码，我们提供脚手架
6. **可视化**：训练曲线、动画
7. **练习 + 自测**：练习的参考答案在 `solutions/` 目录；每章的自测题集中在根目录 `STUDY_GUIDE.md`

### 一些小贴士

- 每章都能**独立跑通**（10 分钟以内），不会卡死
- 重要概念会反复出现（比如 on-policy vs off-policy），第一次见是引子，第二次见是深入
- 公式不会的，跳过！下一章还会再讲
- **画图/动画显示说明**：本教材的动画用 `to_jshtml()` 生成 HTML 播放器、滑块用 ipywidgets 控件——普通 inline 后端就能用，**不需要** `%matplotlib widget`。如果图不显示，Restart Kernel and Run All Cells 即可
"""),

        md("""## 0.6 一些核心术语速查

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| 状态 | state $s$ | 环境的当前快照 |
| 动作 | action $a$ | agent 能选的操作 |
| 奖励 | reward $r$ | 环境给 agent 的即时反馈 |
| 回报 | return $G$ | 从某时刻起累计的折扣奖励 $\\sum \\gamma^t r$ |
| 策略 | policy $\\pi$ | state → action 的映射 |
| 价值 | value $V^\\pi(s)$ | 在状态 $s$ 下、用策略 $\\pi$ 的期望回报 |
| 动作价值 | $Q^\\pi(s,a)$ | 在 $s$ 选 $a$、之后用 $\\pi$ 的期望回报 |
| 折扣因子 | discount $\\gamma$ | 未来奖励的衰减系数，常 0.9~0.99 |
| Episode | / | 一次完整的轨迹（从开始到结束） |
| On-policy | / | 用当前策略采样的数据训练当前策略 |
| Off-policy | / | 可以用旧策略采的数据训练新策略 |
"""),

        md("""## 0.7 小结

- ✅ 环境已就绪
- ✅ 理解了 agent-environment loop
- ✅ 见识了 `ClickWorld`：策略决定一切
- ✅ 知道接下来 18 章学什么

下一章：**第 1 章 — 多臂老虎机**。我们将用最简单的 RL 问题，引入两个核心矛盾：**探索 vs 利用** 和 **信用分配**。

> 💡 提示：如果上面的 `ClickWorld` 动画没有显示，多半是 cell 没按顺序运行——Restart Kernel and Run All Cells 一次即可。动画是 HTML 播放器（jshtml），不需要 `%matplotlib widget`。
"""),
    ]


def ch01() -> List[Cell]:
    """Ch01 多臂老虎机：探索 vs 利用 + UCB + 乐观初始化 + 非平稳。"""
    return [
        md("""# 第 1 章：多臂老虎机 —— RL 的最简形式

上一章 ClickWorld 告诉我们"策略决定一切"——但**策略是从哪来的**？从这一章起，我们开始亲手造一个。出发地点刻意选在全世界最简单的决策问题：只有动作、没有状态，连"未来"都还没有出场。

这个问题你在生活中早就遇到过：食堂有 5 个窗口，你不知道哪个最好吃。每天只能排一个队。**总去已知最好的窗口，还是偶尔试试别的？**——去新窗口可能发现惊喜（探索），也可能吃到黑暗料理（代价）；总吃老几样又心有不甘（利用的天花板）。这就是 **探索 vs 利用** 的两难，它将陪伴我们到最后一章：GRPO 里"多采几个回答再挑好的"，本质上还是食堂问题。

> **核心矛盾**：你面对 $K$ 台老虎机，每台的期望奖励 $q_*(a)$ 未知。
> 怎么在 $T$ 步内拿到尽可能多的奖励？

## 学习目标

1. 理解 **探索 vs 利用**（exploration vs exploitation）的两难
2. 掌握 **ε-greedy** 和 **UCB1** 两种经典算法
3. 能用数值实验比较不同策略的 **regret** 曲线
4. 学会用 **递推式更新**（constant-α）处理非平稳问题
5. 知道为什么 **Thompson Sampling** 在随机性问题上近乎最优

> 🌍 **真实世界**：这就是 A/B 测试的数学本体——两个按钮版本 = 两根拉杆，每次用户点击 = 一次拉动，转化率 = 期望奖励。药品临床试验是更严肃的版本：每根拉杆是一种疗法，探索的代价是病人的痛苦。
"""),

        code("""# 常规设置：把项目根加入 sys.path，载入常用库
import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed, plot_training_curve, plot_regret_curve, make_interactive
from rlenvs import MultiArmedBandit

set_seed(42)
np.set_printoptions(precision=3)
"""),

        md("""## 1.1 问题陈述

**$K$-臂老虎机** 是最简单的 RL 问题：
- 有 $K$ 个动作 $a \\in \\{1, 2, \\dots, K\\}$，每个动作有一个**固定的真实期望奖励** $q_*(a)$
- 在时刻 $t$，agent 选一个动作 $A_t$，环境返回一个奖励 $R_t \\sim \\text{某分布，均值 } q_*(A_t)$
- 没有状态、没有时间关联。**唯一的难点是 $q_*$ 未知。**

我们用 Gaussian 版本：$R_t | A_t = a \\sim \\mathcal{N}(q_*(a), 1)$。

### 一个玩具实例

下面我们建一个 10 臂老虎机，真实均值从 $\\mathcal{N}(0, 1)$ 采出，最优臂的均值约 1.5。
"""),

        code("""env = MultiArmedBandit(n_arms=10, reward_dist='gaussian', seed=42)
print(f"真实均值 q_*: {env.q_star}")
print(f"最优臂: a* = {env.optimal_arm()}, q_*(a*) = {env.q_star[env.optimal_arm()]:.3f}")

# 画每个臂的奖励分布
fig, ax = plt.subplots(figsize=(8, 4))
for i, q in enumerate(env.q_star):
    samples = np.random.default_rng(0).normal(q, 1, 200)
    ax.scatter(np.full(200, i) + np.random.default_rng(i).uniform(-0.15, 0.15, 200),
               samples, s=8, alpha=0.4)
    ax.plot([i-0.3, i+0.3], [q, q], 'k-', linewidth=2)
ax.set_xlabel('action a')
ax.set_ylabel('reward')
ax.set_title('10-armed bandit: 每臂的真实分布')
plt.show()
"""),

        md("""注意第 6、第 8 号臂的均值最高。如果 **事先知道** $q_*$，每步都选第 8 号就行。但 RL 的难处正在于 $q_*$ **要靠试错估计**。

## 1.2 估计 $Q(a)$ 的两种方法

我们用 $Q_t(a)$ 表示 $t$ 时刻对 $q_*(a)$ 的估计。两种主流做法：

### 方法 A：样本平均（Sample Average）

$$
Q_t(a) = \\frac{\\sum_{i=1}^{t-1} R_i \\cdot \\mathbb{1}[A_i = a]}{N_t(a)}
$$

其中 $N_t(a)$ 是前 $t-1$ 步里选 $a$ 的次数。

**收敛性**：当 $N_t(a) \\to \\infty$ 时，$Q_t(a) \\to q_*(a)$（大数定律）。

### 方法 B：常量 $\\alpha$ 递推

$$
Q_{t+1}(a) = Q_t(a) + \\alpha \\big[ R_t - Q_t(a) \\big]
$$

这其实是**指数加权移动平均**——把所有历史奖励按 $\\alpha(1-\\alpha)^k$ 衰减。

<details>
<summary><b>📝 推导：递推公式 = 指数加权（点开看）</b></summary>

把 $Q_{t+1}$ 展开两步：

$$
Q_{t+1} = (1-\\alpha) Q_t + \\alpha R_t = (1-\\alpha)[(1-\\alpha)Q_{t-1} + \\alpha R_{t-1}] + \\alpha R_t
$$

继续展开，对 $a$ 第 $k$ 次被选（设当时奖励为 $R_k$）：

$$
Q_{k+1} = (1-\\alpha)^k Q_1 + \\sum_{i=1}^{k} \\alpha (1-\\alpha)^{k-i} R_i
$$

权重和为 $\\sum_{i=1}^{k} \\alpha (1-\\alpha)^{k-i} = 1 - (1-\\alpha)^k \\to 1$。
所以越近的奖励权重越大、越远越小，呈指数衰减。
</details>

### 用样本平均在线性时间复杂度内实现

朴素地每次重算 $Q_t$ 是 $O(t)$ 的，但**递推式** 让它变 $O(1)$：

$$
Q_{t+1}(a) = Q_t(a) + \\frac{1}{N_t(a)+1} \\big[ R_t - Q_t(a) \\big]
$$

这等价于样本平均，证明留作思考。
"""),

        code("""# 实现：估计 Q 的通用骨架
class BanditAgent:
    \"\"\"一个抽象的 bandit 智能体：子类只需重写 select_action().\"\"\"
    def __init__(self, n_arms, alpha='1/N', init=0.0):
        self.n_arms = n_arms
        self.alpha = alpha  # '1/N' 表示样本平均；float 表示常量 alpha
        self.init = init
        self.Q = np.full(n_arms, init, dtype=float)
        self.N = np.zeros(n_arms, dtype=int)

    def reset(self):
        self.Q[:] = self.init
        self.N[:] = 0

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        self.N[action] += 1
        alpha = 1.0 / self.N[action] if self.alpha == '1/N' else self.alpha
        self.Q[action] += alpha * (reward - self.Q[action])

# 一个永远"贪心"的 agent（只利用、不探索）
class GreedyAgent(BanditAgent):
    def select_action(self):
        return int(np.argmax(self.Q))


# 跑一个纯贪心 agent，看它能在 1000 步内拿到多少奖励
def run_episode(agent, env, n_steps):
    rewards = []
    opt_actions = []
    a_star = env.optimal_arm()
    env.reset()
    agent.Q[:] = 0
    agent.N[:] = 0
    for t in range(n_steps):
        a = agent.select_action()
        r = env.pull(a)
        agent.update(a, r)
        rewards.append(r)
        opt_actions.append(a == a_star)
    return np.array(rewards), np.array(opt_actions)


env = MultiArmedBandit(n_arms=10, reward_dist='gaussian', seed=42)
greedy = GreedyAgent(10)
rewards, opt = run_episode(greedy, env, n_steps=1000)
print(f"纯贪心：1000 步平均奖励 {rewards.mean():.3f}，选中最优臂的比例 {opt.mean():.1%}")
"""),

        md("""### 纯贪心的致命缺陷

纯贪心很容易**锁死**：第一次试出某个臂给出正向奖励，就一直选它、永远不试别的臂。

正确做法是引入 **探索**。下面我们实现最经典的探索策略：**$\\epsilon$-greedy**。
"""),

        md("""## 1.3 $\\epsilon$-greedy：最经典的探索

策略：

$$
A_t = \\begin{cases}
\\arg\\max_a Q_t(a) & \\text{概率 } 1 - \\epsilon \\\\
\\text{随机一个臂} & \\text{概率 } \\epsilon
\\end{cases}
$$

- 大 $\\epsilon$ → 探索多、收敛快、但天花板低
- 小 $\\epsilon$ → 探索少、收敛慢、但天花板高

> 🤔 **先猜再跑**：下面的基准实验在 200 个 seed、1000 步上比较 ε ∈ {0, 0.01, 0.1, 0.3}。两个预测：
> 1. ε=0（纯贪心，**从不探索**）最后选最优臂的比例是多少？
> 2. 短期（前 100 步）谁领先，长期（1000 步）谁的天花板最高？
>
> <details><summary>写下猜测再点开对照</summary>
>
> 提示 1：纯贪心第一次随机试到哪个臂，就可能**一辈子**认准它——它没有任何机制发现自己错了。
> 提示 2：探索是**投资**：现在亏一点，换以后选得准。投资有回本周期，周期取决于你还要玩多久。
> </details>
"""),

        code("""class EpsilonGreedyAgent(BanditAgent):
    def __init__(self, n_arms, epsilon=0.1, alpha='1/N'):
        super().__init__(n_arms, alpha=alpha)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.Q))


# 在 200 个随机种子上跑 1000 步，比较不同 epsilon
def benchmark(epsilons, n_seeds=200, n_steps=1000):
    results = {}
    for eps in epsilons:
        all_rewards = np.zeros((n_seeds, n_steps))
        all_opt = np.zeros((n_seeds, n_steps), dtype=bool)
        for seed in range(n_seeds):
            env = MultiArmedBandit(n_arms=10, reward_dist='gaussian', seed=seed)
            agent = EpsilonGreedyAgent(10, epsilon=eps)
            rewards, opt = run_episode(agent, env, n_steps)
            all_rewards[seed] = rewards
            all_opt[seed] = opt
        results[eps] = (all_rewards, all_opt)
    return results

results = benchmark([0.0, 0.01, 0.1, 0.3], n_seeds=200, n_steps=1000)

# 画平均奖励曲线
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for eps, (rw, _) in results.items():
    mean_curve = rw.mean(axis=0)
    smoothed = np.convolve(mean_curve, np.ones(50)/50, mode='valid')
    axes[0].plot(smoothed, label=f'ε={eps}')
axes[0].set_xlabel('step')
axes[0].set_ylabel('avg reward (smoothed w=50)')
axes[0].set_title('平均奖励')
axes[0].legend()

for eps, (_, opt) in results.items():
    mean_curve = opt.mean(axis=0)
    smoothed = np.convolve(mean_curve, np.ones(50)/50, mode='valid')
    axes[1].plot(smoothed, label=f'ε={eps}')
axes[1].set_xlabel('step')
axes[1].set_ylabel('% optimal action')
axes[1].set_title('选中最优臂的比例')
axes[1].legend()
plt.tight_layout()
plt.show()
"""),

        md("""你应该观察到：

- $\\epsilon = 0$（纯贪心）：长期来看最差，约 70-75%
- $\\epsilon = 0.1$：约 80%，且后期平台高
- $\\epsilon = 0.3$：上升快但天花板低
- $\\epsilon = 0.01$：上升慢但终值最高

**没有银弹**——最优 $\\epsilon$ 取决于 $T$ 和任务方差。

## 1.4 交互式 widget：调 $\\epsilon$ 看效果

下面这个交互组件让你实时调节 $\\epsilon$，看奖励曲线变化。
"""),

        code("""def plot_eps_demo(epsilon=0.1):
    n_seeds, n_steps = 100, 800
    rw = np.zeros((n_seeds, n_steps))
    opt = np.zeros((n_seeds, n_steps), dtype=bool)
    for seed in range(n_seeds):
        env = MultiArmedBandit(n_arms=10, reward_dist='gaussian', seed=seed)
        agent = EpsilonGreedyAgent(10, epsilon=epsilon)
        r, o = run_episode(agent, env, n_steps)
        rw[seed] = r; opt[seed] = o
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_curve = rw.mean(axis=0)
    sm = np.convolve(mean_curve, np.ones(40)/40, mode='valid')
    ax.plot(sm, color='steelblue', linewidth=2, label='smoothed avg reward')
    ax.axhline(env.q_star.max(), color='crimson', linestyle='--', label='最优 q*')
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel('step')
    ax.set_ylabel('reward')
    ax.set_title(f'ε-greedy (ε={epsilon:.2f})')
    ax.legend()
    plt.show()

w = make_interactive(plot_eps_demo,
                     params={'epsilon': (0.1, 0.0, 0.5, 0.01)},
                     layout='hbox')
"""),

        md("""## 1.5 UCB1：基于不确定性的探索

$\\epsilon$-greedy 是**盲目探索**——它随机选臂。能不能更"聪明地"探索？

直觉：如果一个臂**很久没被试过**，我们对它的估计就不确定，应该去试一下。

### UCB1 公式

$$
A_t = \\arg\\max_a \\left[ Q_t(a) + c \\sqrt{\\frac{\\ln t}{N_t(a)}} \\right]
$$

- 第一项 $Q_t(a)$：当前估计（**利用**）
- 第二项 $c \\sqrt{\\frac{\\ln t}{N_t(a)}}$：**不确定性奖励**（**探索**）。$N_t(a)$ 越小（越没试过），不确定性越大，越值得探索。

### UCB1 的理论保证（regret 上界）

<details>
<summary><b>📝 完整证明：UCB1 的对数 regret 上界（点开看）</b></summary>

**定理**（Auer et al. 2002）：对任何 $T \\geq 1$，UCB1 的期望累计 regret 满足：

$$
\\mathbb{E}[R_T] \\leq 8 \\sum_{a: q_*(a) < q_*(a^*)} \\frac{\\ln T}{\\Delta_a} + \\left(1 + \\frac{\\pi^2}{3}\\right) \\sum_{a \\neq a^*} \\Delta_a
$$

其中 $\\Delta_a = q_*(a^*) - q_*(a)$ 是次优臂的差距。

**证明思路**：

1. **Hoeffding 不等式**：对有界随机变量 $X \\in [0, 1]$，$N$ 次独立采样的均值 $\\bar{X}$ 偏离 $\\mathbb{E}[X]$ 超过 $\\epsilon$ 的概率至多 $2e^{-2N\\epsilon^2}$。所以：

$$
|Q_t(a) - q_*(a)| \\leq \\sqrt{\\frac{\\ln t}{2 N_t(a)}} \\quad \\text{w.h.p.}
$$

2. **"选错" 次数上界**：对次优臂 $a$，若它在 $t$ 时刻被选且 $Q_t(a) > Q_t(a^*)$，意味着要么 $a$ 的估计偏高、要么 $a^*$ 的估计偏低。两种情况都用 Hoeffding：

$$
\\Pr\\left[Q_t(a) > Q_t(a^*) \\mid q_*(a) < q_*(a^*)\\right] \\leq t^{-4}
$$

3. **几何级数**：$\\sum_{t=1}^{\\infty} t^{-4}$ 收敛，所以次优臂被选的"期望错误次数"是 $O(\\ln T / \\Delta_a)$。

4. **每错一次贡献 $\\Delta_a$ 到 regret**，总和就是定理的形式。

完整证明见 Auer, Cesa-Bianchi & Fischer 2002《Finite-time Analysis of the Multiarmed Bandit Problem》。
</details>

**关键信息**：UCB1 的 regret 是 $O(\\ln T)$ 的，比 $\\epsilon$-greedy 的 $O(\\sqrt{T})$ 渐进更好。
"""),

        code("""class UCBAgent(BanditAgent):
    def __init__(self, n_arms, c=2.0):
        super().__init__(n_arms, alpha='1/N')
        self.c = c
        self.t = 0

    def select_action(self):
        self.t += 1
        # 如果有臂没试过，优先试
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            return int(untried[0])
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(ucb))

# 在同样 200 个种子上对比 ε-greedy 和 UCB
def run_with_agent(agent, env, n_steps):
    env.reset()
    agent.Q[:] = 0; agent.N[:] = 0
    agent.t = 0
    rewards = np.zeros(n_steps)
    opt = np.zeros(n_steps, dtype=bool)
    a_star = env.optimal_arm()
    for t in range(n_steps):
        a = agent.select_action()
        r = env.pull(a)
        agent.update(a, r)
        rewards[t] = r
        opt[t] = (a == a_star)
    return rewards, opt


n_seeds, n_steps = 200, 1000
methods = {
    'ε=0.1':  lambda: EpsilonGreedyAgent(10, epsilon=0.1),
    'ε=0.01': lambda: EpsilonGreedyAgent(10, epsilon=0.01),
    'UCB c=2': lambda: UCBAgent(10, c=2.0),
}
all_rw = {k: np.zeros((n_seeds, n_steps)) for k in methods}
all_opt = {k: np.zeros((n_seeds, n_steps), dtype=bool) for k in methods}
for seed in range(n_seeds):
    env = MultiArmedBandit(n_arms=10, reward_dist='gaussian', seed=seed)
    for name, fac in methods.items():
        r, o = run_with_agent(fac(), env, n_steps)
        all_rw[name][seed] = r
        all_opt[name][seed] = o

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for name in methods:
    mean_curve = all_rw[name].mean(axis=0)
    sm = np.convolve(mean_curve, np.ones(50)/50, mode='valid')
    axes[0].plot(sm, label=name)
    mean_opt = all_opt[name].mean(axis=0)
    sm_opt = np.convolve(mean_opt, np.ones(50)/50, mode='valid')
    axes[1].plot(sm_opt, label=name)
axes[0].set_title('平均奖励'); axes[0].legend(); axes[0].set_xlabel('step')
axes[1].set_title('最优臂选中比例'); axes[1].legend(); axes[1].set_xlabel('step')
plt.tight_layout(); plt.show()
"""),

        md("""你应该看到 **UCB 在早期就迅速接近最优**，因为它的探索是"有目的"的，不是盲目随机。

## 1.6 乐观初始化：另一种探索的"奇技淫巧"

一个意外的发现：**把 $Q_1(a)$ 初始化为一个很大的正数**（比如 +5），用纯贪心也能探索。

直觉：每个臂都看起来"特别好"，纯贪心会去试每一个，每次试都会"失望"（$Q$ 下降），直到试过所有臂。

注意：这种方法在**平稳**问题里和 $\\epsilon$-greedy 一样好；但在**非平稳**问题里它会**失去探索能力**（所有 $Q$ 都已收敛到合理值后，纯贪心再也不探索了）。
"""),

        code("""class OptimisticGreedyAgent(BanditAgent):
    def __init__(self, n_arms, init=5.0, alpha='1/N'):
        super().__init__(n_arms, alpha=alpha, init=init)
        self.Q[:] = init

    def select_action(self):
        return int(np.argmax(self.Q))

# 通用 runner，调用 agent.reset()
def run_optimistic(env, agent, n_steps):
    env.reset()
    agent.reset()
    rewards, opt = np.zeros(n_steps), np.zeros(n_steps, dtype=bool)
    a_star = env.optimal_arm()
    for t in range(n_steps):
        a = agent.select_action()
        r = env.pull(a); agent.update(a, r)
        rewards[t] = r; opt[t] = (a == a_star)
    return rewards, opt

n_seeds, n_steps = 200, 1000
greedy_rw = np.zeros((n_seeds, n_steps))
optim_rw = np.zeros((n_seeds, n_steps))
for seed in range(n_seeds):
    env = MultiArmedBandit(n_arms=10, reward_dist='gaussian', seed=seed)
    r1, _ = run_optimistic(env, GreedyAgent(10, init=0.0), n_steps)
    r2, _ = run_optimistic(env, OptimisticGreedyAgent(10, init=5.0), n_steps)
    greedy_rw[seed] = r1
    optim_rw[seed] = r2

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.convolve(greedy_rw.mean(0), np.ones(50)/50, mode='valid'), label='greedy, init=0')
ax.plot(np.convolve(optim_rw.mean(0), np.ones(50)/50, mode='valid'), label='greedy, init=5')
ax.set_xlabel('step'); ax.set_ylabel('avg reward (w=50)')
ax.set_title('Optimistic init 把纯贪心变成探索器'); ax.legend()
plt.tight_layout(); plt.show()
"""),

        md("""## 1.7 非平稳问题：为什么需要常量 $\\alpha$

到目前为止我们假设 $q_*$ 不变。但现实常常是**非平稳**的：用户的口味会变、市场会变、模型部署的环境会变。

我们用 `MultiArmedBandit(non_stationary=True)` 让 $q_*$ 每步加一个高斯噪声漂移。

**为什么样本平均会失败？** 因为它对所有历史一视同仁，过去 1000 步的奖励和最近的 1 步一样重要。

**为什么常量 $\\alpha$ 行？** 因为它的权重指数衰减——最近的奖励最重要。
"""),

        code("""n_seeds, n_steps = 1000, 2000
results = {}
for name, alpha in [('sample avg', '1/N'), ('α=0.1', 0.1)]:
    all_rw = np.zeros((n_seeds, n_steps))
    for seed in range(n_seeds):
        env = MultiArmedBandit(n_arms=10, reward_dist='gaussian',
                               non_stationary=True, drift_std=0.01, seed=seed)
        agent = EpsilonGreedyAgent(10, epsilon=0.1, alpha=alpha)
        r, _ = run_with_agent(agent, env, n_steps)
        all_rw[seed] = r
    results[name] = all_rw

fig, ax = plt.subplots(figsize=(8, 4))
for name, rw in results.items():
    ax.plot(np.convolve(rw.mean(0), np.ones(100)/100, mode='valid'), label=name, linewidth=2)
ax.set_xlabel('step'); ax.set_ylabel('avg reward (w=100)')
ax.set_title('非平稳问题：常量 α 完胜样本平均'); ax.legend()
plt.tight_layout(); plt.show()
"""),

        md("""## 1.8 小结

| 方法 | 探索机制 | regret | 非平稳友好 |
|---|---|---|---|
| 纯贪心 | 无 | $O(T)$ | ✗（锁死） |
| $\\epsilon$-greedy | 随机扰动 | $O(\\sqrt{T})$ | ✓（常量 $\\alpha$） |
| UCB1 | 不确定性 | $O(\\ln T)$ | 需改造 |
| 乐观初始化 | 假装每个臂都很好 | 同 $\\epsilon$-greedy | ✗（初期有效） |

**核心收获**：

1. RL 的本质是 **在探索和利用之间找平衡**
2. **递推更新** $Q \\leftarrow Q + \\alpha(R - Q)$ 是 RL 中反复出现的母题
3. **算法不能光看理论曲线**，还要考虑方差、非平稳、超参敏感性

## 1.9 📝 练习

### 练习 1（必做）：实现 Thompson Sampling

Thompson Sampling 的核心思想：对每个臂 $a$ 维护一个**后验分布** $p(q_a | \\text{history})$，每步从这个后验采样 $\\hat{q}_a$，选 $\\arg\\max_a \\hat{q}_a$。

对 Bernoulli 奖励（每臂真实概率 $\\theta_a \\in [0, 1]$，给奖励 0/1），用 **Beta 分布**作为共轭先验：

- 先验：$\\text{Beta}(1, 1)$（均匀分布）
- 观测 $r=1$ 后：$\\alpha \\leftarrow \\alpha + 1$
- 观测 $r=0$ 后：$\\beta \\leftarrow \\beta + 1$
- 采样：$\\hat{\\theta}_a \\sim \\text{Beta}(\\alpha_a, \\beta_a)$

**任务**：实现 `ThompsonBernoulliAgent`，在 200 个 Bernoulli 10-臂 bandit 上对比它和 UCB1。

> 参考答案：`solutions/ch01_thompson_sampling.ipynb`

---

> 📖 学完本章，先做 `STUDY_GUIDE.md` 里 Ch01 的自测题（4 题），全对再进下一章。

下一章：**第 2 章 — MDP 与贝尔曼方程**。
我们将引入**状态**和**未来**的概念，把单步的 bandit 推广到序列决策的 MDP。
"""),
    ]


def ch02() -> List[Cell]:
    """Ch02 MDP + 贝尔曼方程：RL 的数学语言。"""
    return [
        md("""# 第 2 章：MDP 与贝尔曼方程 —— RL 的数学语言

> 这一章是整本书的**地基**。把贝尔曼方程吃透，后面所有算法（DQN、PPO、GRPO）都能看懂。

## 学习目标

1. 理解 **马尔可夫决策过程（MDP）** 的五元组 $(S, A, P, R, \\gamma)$
2. 掌握 **状态价值 $V^\\pi$** 和 **动作价值 $Q^\\pi$** 的定义
3. **逐步推导贝尔曼期望方程**（最重要！）
4. 通过交互式 widget 直观感受 $\\gamma$ 对 $V$ 的影响
5. 看"值传播"动画：奖励如何从终点倒着传回起点
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from utils import set_seed, plot_value_heatmap, make_interactive
from rlenvs import GridWorld, small_grid_5x5, bridge_grid

set_seed(0)
"""),

        md("""## 2.1 从 bandit 到 MDP：引入"状态"

第 1 章的老虎机已经能平衡探索与利用了，但它有一个被我们悄悄绕过的假设：**今天拉哪根拉杆，不会改变明天有哪些拉杆**。每次面对的都是同一个问题、同一组选择。

下围棋可就不是这样了——这一步落在哪里，直接决定了棋盘接下来长什么样、你还有哪些选择。一旦**动作开始改变世界**，"眼前这一步拿多少奖励"就远远不够了：一个贪吃子的棋手，每步的即时奖励都很高，然后在二十手后被围死。

所以我们需要一种新的语言，能够回答两个 bandit 回答不了的问题：

1. **状态怎么转移？** 选了动作 $a$ 之后，世界会变成什么样？
2. **奖励怎么和长期挂钩？** 当前一步的甜头和未来的收益，如何权衡？

这就是 MDP。在动形式化定义之前，先在三个熟悉场景里找找"状态"的影子：

- 围棋：当前棋盘就是状态，不同棋盘对应不同最优动作
- 开车：位置 + 速度 + 路况构成状态
- LLM：当前已生成的 token 序列就是状态

### MDP 五元组

一个**马尔可夫决策过程**由五元组 $\\mathcal{M} = (\\mathcal{S}, \\mathcal{A}, P, R, \\gamma)$ 刻画：

| 符号 | 含义 |
|---|---|
| $\\mathcal{S}$ | 状态集合（state space） |
| $\\mathcal{A}$ | 动作集合 |
| $P(s' \\| s, a)$ | **转移概率**：在 $s$ 选 $a$，到 $s'$ 的概率 |
| $R(s, a)$（或 $R(s'a s)$） | 在 $s$ 选 $a$（再到 $s'$）的期望奖励 |
| $\\gamma \\in [0, 1]$ | **折扣因子** |

### 马尔可夫性

**下一个状态和奖励只依赖当前状态和动作，不依赖更早的历史**：

$$
P(s_{t+1} = s' \\mid s_t, a_t, s_{t-1}, a_{t-1}, \\dots) = P(s_{t+1} = s' \\mid s_t, a_t)
$$

这是个**假设**——很多现实问题严格说不满足马尔可夫性（比如打牌时弃牌堆的信息），但**通常能找到一个充分包含信息的 state 表征**让马尔可夫性近似成立。LLM 中"上下文窗口"就是为近似马尔可夫性服务的。
"""),

        md("""## 2.2 回报与折扣：为什么要 $\\gamma$

agent 的目标是最大化 **期望累计奖励**：

$$
G_t = R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\dots = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}
$$

### 为什么必须 $\\gamma < 1$？

三个理由：

1. **数学**：避免无穷大。如果 episode 无限长、$\\gamma = 1$，则 $G_t$ 可能发散
2. **人类直觉**：今天的 1 元 > 明天的 1 元（货币的时间价值）
3. **算法收敛**： discounted MDP 一定有有限的最优值（Ch03 会用到）

### $\\gamma$ 的现实意义

- $\\gamma = 0$：只看眼前奖励（**目光短浅**），$V^\\pi(s) = \\mathbb{E}[R_{t+1}]$
- $\\gamma = 1$：未来奖励和眼前一样重要（**无限远见**），数学上危险
- $\\gamma \\in [0.9, 0.99]$：典型取值。**越大越难学，但策略越优**

### $G_t$ 的自递推（递归形式）

**关键观察**：回报可以递归定义。

$$
G_t = R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\dots = R_{t+1} + \\gamma \\big[ R_{t+2} + \\gamma R_{t+3} + \\dots \\big] = R_{t+1} + \\gamma G_{t+1}
$$

这个简单的等式 $G_t = R_{t+1} + \\gamma G_{t+1}$ 几乎是所有 RL 算法的根源。
"""),

        md("""## 2.3 策略、状态价值、动作价值

### 策略 $\\pi(a|s)$

策略是从状态到动作分布的映射：$\\pi(a|s) = \\Pr[A_t = a \\mid S_t = s]$。

它可以是确定性的（$\\pi(s)$ 总返回同一动作），也可以是随机的。

### 状态价值函数 $V^\\pi(s)$

**在状态 $s$ 出发、之后按策略 $\\pi$ 行动，期望累计奖励**：

$$
V^\\pi(s) = \\mathbb{E}_\\pi[G_t \\mid S_t = s]
$$

### 动作价值函数 $Q^\\pi(s, a)$

**在状态 $s$ 选 $a$、之后按 $\\pi$ 行动，期望累计奖励**：

$$
Q^\\pi(s, a) = \\mathbb{E}_\\pi[G_t \\mid S_t = s, A_t = a]
$$

**两者关系**：

$$
V^\\pi(s) = \\sum_a \\pi(a|s) Q^\\pi(s, a)
$$

即 $V$ 是 $Q$ 在策略下的期望。
"""),

        md("""## 2.4 贝尔曼期望方程（**最重要！**）

现在手上有了 $V^\\pi$ 的定义，但按定义算它要把**所有可能的未来**都加一遍——围棋有 $10^{170}$ 种未来，一次都枚举不动。怎么办？

> 🤔 **先自己猜 30 秒**：$V^\\pi(s)$（在 $s$ 出发的期望总回报）和 $V^\\pi(s')$（在下一个状态出发的期望总回报）之间，应该存在什么关系？
>
> <details><summary>想好了点开对照</summary>
>
> 直觉：「一个状态的价值 = 马上能拿的奖励 + 打了折之后的下一个状态的价值」。
> 如果这个递归成立，我们就不需要枚举整棵未来树——每个状态只跟它的**后继**有关。
> 把这个直觉写成等式并严格证明它，就是本节的内容。
> </details>

我们想把 $V^\\pi$ 和 $Q^\\pi$ 写成递归形式——这就是**贝尔曼方程**。

### 直觉

$V^\\pi(s)$ 是从 $s$ 出发的期望回报。它等于：

> 即时奖励 + $\\gamma$ × 下一状态的价值（期望）

### 推导 $V^\\pi$ 的贝尔曼方程

$$
\\begin{aligned}
V^\\pi(s) &= \\mathbb{E}_\\pi[G_t \\mid S_t = s] \\\\
         &= \\mathbb{E}_\\pi[R_{t+1} + \\gamma G_{t+1} \\mid S_t = s]  \\quad (\\text{用 } G_t = R_{t+1} + \\gamma G_{t+1}) \\\\
         &= \\mathbb{E}_\\pi[R_{t+1}] + \\gamma \\, \\mathbb{E}_\\pi[G_{t+1} \\mid S_t = s]
\\end{aligned}
$$

**第一项**：$\\mathbb{E}_\\pi[R_{t+1} \\mid S_t = s]$。需要先选 $a$（概率 $\\pi(a|s)$），然后从 $(s, a)$ 采样的奖励期望是 $R(s, a)$：

$$
\\mathbb{E}_\\pi[R_{t+1} \\mid S_t = s] = \\sum_a \\pi(a|s) R(s, a) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a) r(s, a, s')
$$

（最后一项在 $r$ 与 $s'$ 也有关时用）

**第二项**：$\\mathbb{E}_\\pi[G_{t+1} \\mid S_t = s]$。需要选 $a$（$\\pi$），转移到 $s'$（$P$），然后从 $s'$ 出发的期望回报是 $V^\\pi(s')$：

$$
\\mathbb{E}_\\pi[G_{t+1} \\mid S_t = s] = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a) V^\\pi(s')
$$

合起来：

$$
\\boxed{\\; V^\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a) \\big[ r(s,a,s') + \\gamma V^\\pi(s') \\big] \\;}
$$

这就是**贝尔曼期望方程**。"期望"是因为我们对**策略 $\\pi$** 和**转移 $P$** 都取期望。

### 推导 $Q^\\pi$ 的贝尔曼方程

类似地：

$$
\\boxed{\\; Q^\\pi(s, a) = \\sum_{s'} P(s'|s,a) \\big[ r(s,a,s') + \\gamma \\sum_{a'} \\pi(a'|s') Q^\\pi(s', a') \\big] \\;}
$$

或者用 $V$ 表达：

$$
Q^\\pi(s, a) = \\sum_{s'} P(s'|s,a) \\big[ r(s,a,s') + \\gamma V^\\pi(s') \\big]
$$

### 矩阵形式（为 Ch03 做铺垫）

若有 $n$ 个状态，$V^\\pi$ 是 $n$ 维向量，则：

$$
V^\\pi = R^\\pi + \\gamma P^\\pi V^\\pi
$$

其中 $P^\\pi_{ss'} = \\sum_a \\pi(a|s) P(s'|s,a)$、$R^\\pi_s = \\sum_a \\pi(a|s) R(s,a)$。

这是一个**线性方程组**！Ch03 我们会看到怎么解。
"""),

        md("""## 2.5 看一眼真实 MDP：small_grid_5x5

我们用一个 $5 \\times 5$ 的网格：
- 起点 $(4, 0)$（左下角）
- 终点 $(0, 4)$（右上角，奖励 $+1$）
- 一个陷阱 $(1, 2)$（奖励 $-0.5$）
- 两堵墙 $(2, 2), (2, 3)$
- 每步默认 $-0.05$（让 agent 别磨蹭）

我们环境**完全暴露** $P[s, a, s']$ 和 $R[s, a]$，可以直接用它们算 $V^\\pi$。
"""),

        code("""env = small_grid_5x5(seed=0)
print(f"shape: {env.shape}, n_states={env.nS}, n_actions={env.nA}")
print(f"动作: 0=↑, 1=→, 2=↓, 3=←")
print(f"终点: {env.terminals}")
print(f"墙:   {env.walls}")
print(f"特殊奖励: {env.rewards}")
print()
print("P[s, a, s'] 的形状：", env.P.shape)
print("R[s, a] 的形状：    ", env.R.shape)
print()
# 验证 P 行和为 1
print("P 每行和：", env.P.sum(axis=2).min(), "~", env.P.sum(axis=2).max())
"""),

        md("""## 2.6 手动计算均匀随机策略的 $V^\\pi$

我们让策略 $\\pi(a|s) = 1/4$ 对每个动作（均匀随机）。

### 用矩阵形式解 $V^\\pi$

$V^\\pi = R^\\pi + \\gamma P^\\pi V^\\pi \\Rightarrow (I - \\gamma P^\\pi) V^\\pi = R^\\pi \\Rightarrow V^\\pi = (I - \\gamma P^\\pi)^{-1} R^\\pi$

这是**精确解**，不用迭代。
"""),

        code("""def compute_uniform_random_V(env, gamma=0.9):
    \"\"\"用矩阵求逆精确解 V^π，π 是均匀随机策略。\"\"\"
    nS, nA = env.nS, env.nA
    # 构造 π(a|s) = 1/nA 的矩阵 [nS, nA]
    pi = np.full((nS, nA), 1.0 / nA)
    # R^π[s] = Σ_a π(a|s) R(s, a)
    R_pi = (pi * env.R).sum(axis=1)
    # P^π[s, s'] = Σ_a π(a|s) P(s'|s, a)
    P_pi = np.einsum('sa,saq->sq', pi, env.P)
    # 解线性方程组
    A = np.eye(nS) - gamma * P_pi
    V = np.linalg.solve(A, R_pi)
    return V, R_pi, P_pi


V, R_pi, P_pi = compute_uniform_random_V(env, gamma=0.9)
print("均匀随机策略下的 V^π（精确解）：")
print(V.reshape(env.shape).round(2))
print()

# 画热力图
fig, ax = plt.subplots(figsize=(5, 5))
plot_value_heatmap(
    V, env.shape, cell_text=True,
    walls=list(env.walls), terminals=list(env.terminals),
    ax=ax, title='V^π (random policy, γ=0.9)',
)
plt.tight_layout(); plt.show()
"""),

        md("""注意几个现象：

1. **终点 (0, 4) 处 V=0**：终止态不再产生奖励（这一步没拿奖励）
2. **越靠近终点 V 越高**：因为可以更快到达 +1
3. **陷阱 (1, 2) 处 V 是负数**：踩上去 -0.5
4. **墙没价值**：不可达

### 数值验证：蒙特卡洛估计

矩阵法得到的 $V$ 对不对？我们用蒙特卡洛验证一下：从状态 $s$ 出发、按均匀随机策略走，直到终点，记录累计奖励的均值。
"""),

        code("""def uniform_random_policy(s):
    return np.random.randint(4)

def mc_estimate_V(env, gamma=0.9, n_episodes=2000, max_steps=200):
    \"\"\"用 MC 估计每个非终止、非墙状态的 V^π。\"\"\"
    V_sum = np.zeros(env.nS)
    V_cnt = np.zeros(env.nS)
    for s0 in range(env.nS):
        # 跳过终止态和墙
        if env.is_terminal(s0):
            continue
        if env.state_to_xy(s0) in env.walls:
            continue
        for _ in range(n_episodes):
            env._state = s0
            G = 0.0
            t = 0
            done = False
            while not done and t < max_steps:
                a = uniform_random_policy(env.state)
                _, r, done, _ = env.step(a)
                G += (gamma ** t) * r
                t += 1
            V_sum[s0] += G
            V_cnt[s0] += 1
    V_mc = np.where(V_cnt > 0, V_sum / np.maximum(V_cnt, 1), 0.0)
    return V_mc, V_cnt


V_mc, V_cnt = mc_estimate_V(env, gamma=0.9, n_episodes=2000, max_steps=200)

# 比较（只看有效状态）
print(f"{'state':<6}{'V_exact':<12}{'V_mc':<12}{'diff':<10}")
for s in [0, 5, 10, 15, 20, 24]:
    if V_cnt[s] > 0:
        print(f"{s:<6}{V[s]:<12.3f}{V_mc[s]:<12.3f}{abs(V[s]-V_mc[s]):<10.3f}")

# 只统计有效状态的最大误差
mask = V_cnt > 0
err = np.abs(V - V_mc)[mask].max()
print(f"\\n有效状态的最大绝对误差: {err:.4f}（应在 0.05 以内）")
"""),

        md("""### 数值验证的小问题

如果你看到误差比较大，可能是：
1. MC 方差大（2000 episodes 不够）→ 增加 `n_episodes`
2. 步数限制截断（200 不够，γ=0.9 时尾巴贡献有限）

后面 Ch04 我们会用 TD(0) 给出更高效的估计方法。

## 2.7 交互式 widget：$\\gamma$ 如何影响 $V$
"""),

        code("""def plot_V_gamma(gamma=0.9):
    V, _, _ = compute_uniform_random_V(env, gamma=gamma)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_value_heatmap(
        V, env.shape, cell_text=True,
        walls=list(env.walls), terminals=list(env.terminals),
        ax=ax, title=f'V^π (random policy, γ={gamma:.2f})',
    )
    plt.tight_layout(); plt.show()

w = make_interactive(plot_V_gamma,
                     params={'gamma': (0.9, 0.0, 0.99, 0.01)})
"""),

        md("""拖动 $\\gamma$ 从 0 到 0.99，观察：

- $\\gamma = 0$：所有非终点状态 V ≈ 即时奖励（多为 -0.05 或 -0.5）
- $\\gamma$ 增大：值"传播"更远，远离终点的格子也开始变正
- $\\gamma \\to 1$：值普遍变大（因为终点 +1 的影响传得很远）

### 这个 widget 的本质

**$\\gamma$ 控制 agent 的"远见程度"**。这正是 RL 中调 $\\gamma$ 的核心权衡：

- $\\gamma$ 小：agent 短视，只看下一步（容易学，但策略可能次优）
- $\\gamma$ 大：agent 远见，看长期（更难学，但策略更优）

## 2.8 值传播动画：奖励如何"逆向"流回起点

我们做这样一个实验：
1. 初始化 $V_0(s) = 0$ 对所有 $s$
2. 迭代 $V_{k+1}(s) = R^\\pi(s) + \\gamma P^\\pi V_k(s)$
3. 看 $V_k$ 怎么随 $k$ 变化

这正是 Ch03 的**迭代策略评估** 的核心。这里我们提前展示，让你看"值是怎么传播的"。
"""),

        code("""from utils import animate_agent

def value_propagation_animation(env, gamma=0.9, n_iters=30, fps=2):
    \"\"\"迭代策略评估，每一帧是一次 sweep 的 V_k。\"\"\"
    nS, nA = env.nS, env.nA
    pi = np.full((nS, nA), 1.0 / nA)
    R_pi = (pi * env.R).sum(axis=1)
    P_pi = np.einsum('sa,saq->sq', pi, env.P)
    V = np.zeros(nS)
    Vs = [V.copy()]
    for _ in range(n_iters):
        V = R_pi + gamma * P_pi @ V
        Vs.append(V.copy())

    fig, ax = plt.subplots(figsize=(5, 5))
    def update(k):
        ax.clear()
        plot_value_heatmap(
            Vs[k], env.shape, cell_text=True,
            walls=list(env.walls), terminals=list(env.terminals),
            ax=ax, title=f'sweep {k}',
        )
    anim = animation.FuncAnimation(fig, update, frames=len(Vs), interval=1000//fps, blit=False, repeat=True)
    plt.close(fig)
    return anim

anim = value_propagation_animation(env, gamma=0.9, n_iters=25, fps=2)
HTML(anim.to_jshtml())  # 在 notebook 内显示
"""),

        md("""你应该看到：

- **Sweep 0**：所有 $V = 0$
- **Sweep 1**：只有终点的邻居获得了非零值（"信号到达了邻居"）
- **Sweep 2, 3, ...**：值像水波一样传到更远的格子
- **Sweep ~20**：基本收敛到精确 $V^\\pi$

**这就是"值传播"——奖励信号沿 $P$ 反向传播的过程**。理解了它，你就理解了 RL 中"学习"的本质。

## 2.9 📝 练习

### 练习 1：Bridge Grid 中 $\\gamma$ 如何翻转策略

`bridge_grid()` 是一个 $3 \\times 5$ 网格：
- 起点 $(1, 0)$，终点 $(1, 4)$
- 中间 $(1, 1..3)$ 是"桥"，桥下 $(2, 1..3)$ 是深渊（$-1$）
- 桥上方 $(0, *)$ 也是通路，绕远但安全

**问题**：找一个 $\\gamma^*$ 阈值，使得：
- $\\gamma < \\gamma^*$：最优策略"绕远"（走 row 0）
- $\\gamma > \\gamma^*$：最优策略"抄近道"（走桥）

**提示**：
1. 对不同 $\\gamma$，用矩阵法算最优 $V^*$
2. 找出"桥起点" $(1, 0)$ 和"绕远起点" $(0, 0)$ 的 $V$ 何时翻转

> 参考答案：`solutions/ch02_bridge_gamma.ipynb`

---

## 2.10 小结

| 概念 | 一句话 |
|---|---|
| MDP | $(S, A, P, R, \\gamma)$ 五元组——RL 世界的完整数学描述 |
| 回报 $G_t$ | 从 $t$ 起的折扣累计奖励 $\\sum_k \\gamma^k r_{t+k+1}$ |
| 贝尔曼期望方程 | $V^\\pi(s) = \\mathbb{E}[r + \\gamma V^\\pi(s')]$——一切 RL 算法的根源 |
| 矩阵解 | $V = (I - \\gamma P_\\pi)^{-1} r_\\pi$，模型已知时的精确解 |
| $Q^\\pi$ | 多一个动作维度的价值；$\\arg\\max_a Q^* = \\pi^*$ |

三个带走的东西：

1. **γ 不只是工程参数，是数学必需**：它让无限长轨迹的回报收敛，也编码了"未来多重要"
2. **贝尔曼方程是递归**：当前的值 = 即时奖励 + 折扣 × 后继的值——后面每一章都在用不同方式解这一个方程
3. **解析解 vs MC 验证**：两者在容差内一致，建立了"采样可信"的信心——这是后面所有无模型方法的地基

> 📖 学完本章，先做 `STUDY_GUIDE.md` 里 Ch02 的自测题（4 题），全对再进下一章。

---

下一章：**第 3 章 — 动态规划**。
我们将用本章的贝尔曼方程，去**精确求解 MDP**——当你"知道一切"（即知道 $P$ 和 $R$）时。
"""),
    ]


def ch03() -> List[Cell]:
    """Ch03 动态规划：策略评估 + 策略迭代 + 值迭代。"""
    return [
        md("""# 第 3 章：动态规划 —— 当你"知道一切"时

Ch02 结尾我们用矩阵求逆一步解出了 V^π——但那是因为 5×5 网格只有 25 个状态。围棋有 10^170 个状态，矩阵求逆？连矩阵都存不下。**好在贝尔曼方程本身就是递归的**——递归的东西可以迭代着算：从一个随机的猜测出发，反复代入方程，直到不再变化。这就是动态规划（DP）。

先说清楚一个前提：DP 假设你**知道一切**（完整的 P 和 R）。这在真实世界几乎从不成立——所以有读者会问"学它干嘛"。两个理由：第一，它给后面所有近似算法提供了**标准答案**——你写的 TD/DQN/PPO 学得对不对，先跟 DP 的解对表；第二，DP 的两个核心操作（策略评估 + 贪心改进）会被后续算法原样继承，只是"精确"换成"采样"、"表格"换成"网络"。

> 🌍 **真实世界**：DP 并没有退场——电梯调度、库存管理、无人机路径规划这些**模型已知**的运筹问题今天仍在用 DP 求解；AlphaGo 的 MCTS 里也藏着 DP 的影子。

## 学习目标

1. 用**迭代策略评估** 解 $V^\\pi$（不再依赖矩阵求逆）
2. 掌握**策略改进定理**（含完整证明）
3. 实现完整的 **策略迭代（Policy Iteration）**
4. 推导**贝尔曼最优性方程** 并实现**值迭代（Value Iteration）**
5. 看动画理解"扫描"过程
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from utils import set_seed, plot_value_heatmap, make_interactive
from rlenvs import GridWorld, small_grid_5x5

set_seed(0)
"""),

        md("""## 3.1 DP 的假设：完美模型

动态规划（Dynamic Programming, DP）假设你**完全知道 MDP 的所有要素**：
- 转移概率 $P(s'|s,a)$
- 奖励函数 $R(s,a)$
- 状态集 $\\mathcal{S}$、动作集 $\\mathcal{A}$

这在现实中通常**不成立**——但你下棋知道规则、迷宫知道布局就是 DP 能直接处理的场景。
更重要的：**DP 是所有 model-free RL 算法（Ch04+）的理论基础**。

DP 的两大用途：

1. **预测（Prediction）**：给定 $\\pi$，求 $V^\\pi$
2. **控制（Control）**：求最优 $\\pi^*$ 和 $V^*$

## 3.2 迭代策略评估（Iterative Policy Evaluation）

**问题**：给定 $\\pi$，求 $V^\\pi$。

Ch02 用了矩阵求逆 $V^\\pi = (I - \\gamma P^\\pi)^{-1} R^\\pi$，但矩阵求逆在状态空间大时（$10^5$ 状态以上）不可行。

**迭代法**：用一个近似序列 $V_0, V_1, V_2, \\dots$ 收敛到 $V^\\pi$。初始 $V_0$ 任意，每一步做"贝尔曼 backup"：

$$
V_{k+1}(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a) \\big[ r(s,a,s') + \\gamma V_k(s') \\big]
$$

**直觉**：每一步把"当前对未来的估计"代回贝尔曼方程，得到更好的估计。

### 收敛性

可以证明（Ch02 矩阵形式 $V^\\pi = R^\\pi + \\gamma P^\\pi V^\\pi$），$V_k \\to V^\\pi$ 当 $\\gamma < 1$，收敛速率 $O(\\gamma^k)$。
"""),

        code("""def iterative_policy_eval(env, pi, gamma=0.9, theta=1e-6, max_iters=10000, record_history=False):
    \"\"\"迭代策略评估。
    pi : [nS, nA] 的概率矩阵
    theta : 收敛阈值（最大变化 < theta 时停止）
    \"\"\"
    nS, nA = env.nS, env.nA
    V = np.zeros(nS)
    history = [V.copy()] if record_history else None
    for it in range(max_iters):
        delta = 0.0
        new_V = np.zeros(nS)
        for s in range(nS):
            if env.is_terminal(s):
                continue
            v_old = V[s]
            v_new = 0.0
            for a in range(nA):
                if pi[s, a] == 0:
                    continue
                # Σ_{s'} P(s'|s,a) [r + γ V(s')]
                for s2 in range(nS):
                    p = env.P[s, a, s2]
                    if p == 0:
                        continue
                    # 注意：我们环境的 R 是 R[s, a] 形式，不依赖 s'
                    v_new += pi[s, a] * p * (env.R[s, a] + gamma * V[s2])
            new_V[s] = v_new
            delta = max(delta, abs(v_new - v_old))
        V = new_V
        if record_history:
            history.append(V.copy())
        if delta < theta:
            break
    return V, it + 1, history


env = small_grid_5x5(seed=0)
nS, nA = env.nS, env.nA
pi_uniform = np.full((nS, nA), 1.0 / nA)

V_dp, n_iters, history = iterative_policy_eval(
    env, pi_uniform, gamma=0.9, theta=1e-6, record_history=True
)
print(f"迭代策略评估在 {n_iters} 步收敛")

# 用矩阵法对比
pi = pi_uniform
R_pi = (pi * env.R).sum(axis=1)
P_pi = np.einsum('sa,saq->sq', pi, env.P)
V_exact = np.linalg.solve(np.eye(nS) - 0.9 * P_pi, R_pi)
print(f"DP vs 矩阵法 最大误差: {np.abs(V_dp - V_exact).max():.2e}")
"""),

        md("""### 向量化版本（快很多）

上面那个三重循环慢且啰嗦。用 numpy einsum 一下：
"""),

        code("""def iterative_policy_eval_vec(env, pi, gamma=0.9, theta=1e-7, max_iters=10000, record=False):
    \"\"\"向量化版本，比循环快 100 倍。\"\"\"
    nS, nA = env.nS, env.nA
    R_sa = env.R  # [nS, nA]
    P_sas = env.P  # [nS, nA, nS]
    V = np.zeros(nS)
    history = [V.copy()] if record else None
    for it in range(max_iters):
        # Q_k[s, a] = Σ_{s'} P(s'|s,a) [R(s,a) + γ V_k(s')]
        Q = R_sa + gamma * np.einsum('saq,q->sa', P_sas, V)
        # 注意：上式 R[s,a] 已经平均了 s'，但我们的 env 实现里 R[s,a] 不依赖 s'，直接用
        new_V = (pi * Q).sum(axis=1)
        # 终止态强制为 0
        for s in range(nS):
            if env.is_terminal(s):
                new_V[s] = 0.0
        delta = np.abs(new_V - V).max()
        V = new_V
        if record:
            history.append(V.copy())
        if delta < theta:
            break
    return V, it + 1, history


V_dp2, n_iters2, history2 = iterative_policy_eval_vec(env, pi_uniform, gamma=0.9, record=True)
print(f"向量化版本：{n_iters2} 步收敛，误差 {np.abs(V_dp2 - V_exact).max():.2e}")
"""),

        md("""## 3.3 策略改进：从 $V^\\pi$ 找到更好的 $\\pi'$

给定 $V^\\pi$，我们能找一个更好的策略 $\\pi'$ 吗？

### 贪心策略

$$
\\pi'(s) = \\arg\\max_a \\sum_{s'} P(s'|s,a) \\big[ r(s,a,s') + \\gamma V^\\pi(s') \\big]
$$

也就是"看哪个动作的短期奖励 + 长期价值最大"。这叫**对 $V^\\pi$ 贪心**。

### 策略改进定理（核心！）

**定理**：设 $\\pi'$ 是对 $V^\\pi$ 贪心的策略。则对所有 $s$：

$$
V^{\\pi'}(s) \\geq V^\\pi(s)
$$

<details>
<summary><b>📝 完整证明（点开看）</b></summary>

**目标**：证 $V^{\\pi'}(s) \\geq V^\\pi(s), \\forall s$。

记 $q_\\pi(s, a) = \\sum_{s'} P(s'|s,a)[r + \\gamma V^\\pi(s')]$。

**Step 1**：因为 $\\pi'$ 对 $V^\\pi$ 贪心，所以

$$
V^\\pi(s) \\leq q_\\pi(s, \\pi'(s)) = \\sum_{s'} P(s'|s,\\pi'(s)) [r + \\gamma V^\\pi(s')]
$$

**Step 2**：把右边那个 $V^\\pi(s')$ 用同样办法展开（递归）

$$
\\begin{aligned}
V^\\pi(s) &\\leq \\mathbb{E}_{s' \\sim P, \\cdot | s, \\pi'(s)} [r + \\gamma V^\\pi(s')] \\\\
         &\\leq \\mathbb{E}_{\\pi'} [r_1 + \\gamma r_2 + \\gamma^2 V^\\pi(s_2)] \\quad \\text{(再用一次贪心)} \\\\
         &\\leq \\mathbb{E}_{\\pi'} \\left[ \\sum_{k=0}^{T} \\gamma^k r_{k+1} + \\gamma^{T+1} V^\\pi(s_{T+1}) \\right]
\\end{aligned}
$$

**Step 3**：令 $T \\to \\infty$，$\\gamma^{T+1} V^\\pi(s_{T+1}) \\to 0$（$\\gamma < 1$ 保证），右边变成 $V^{\\pi'}(s)$。所以

$$
V^\\pi(s) \\leq V^{\\pi'}(s) \\quad \\blacksquare
$$

</details>

**意义**：贪心改进**永远不变差**。这是 RL 算法设计的一个基石。
"""),

        code("""def greedy_policy_from_V(env, V, gamma=0.9):
    \"\"\"对 V 贪心，返回确定性策略 [nS, nA]（one-hot）。\"\"\"
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))
    for s in range(nS):
        if env.is_terminal(s):
            continue
        for a in range(nA):
            for s2 in range(nS):
                p = env.P[s, a, s2]
                if p > 0:
                    Q[s, a] += p * (env.R[s, a] + gamma * V[s2])
    pi = np.zeros((nS, nA))
    pi[np.arange(nS), Q.argmax(axis=1)] = 1.0
    # 终止态任选（反正不动作）
    return pi, Q


# 用 V_uniform 贪心，看新策略是否更好
pi_greedy, Q_greedy = greedy_policy_from_V(env, V_dp2, gamma=0.9)
V_greedy, _, _ = iterative_policy_eval_vec(env, pi_greedy, gamma=0.9)

print("改进后比改进前更优？", (V_greedy >= V_dp2 - 1e-6).all())
print(f"  V^π_uniform 平均: {V_dp2.mean():.3f}")
print(f"  V^π_greedy 平均: {V_greedy.mean():.3f}")

# 画两个 V 对比
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, V_, title in [(axes[0], V_dp2, 'V (uniform)'), (axes[1], V_greedy, 'V (greedy improved)')]:
    plot_value_heatmap(
        V_, env.shape, cell_text=True,
        walls=list(env.walls), terminals=list(env.terminals),
        ax=ax, title=title,
    )
plt.tight_layout(); plt.show()
"""),

        md("""## 3.4 策略迭代（Policy Iteration）

把"评估"和"改进"交替进行：

```
初始化 π_0
repeat:
    V^π_k ← 迭代策略评估(π_k)
    π_{k+1} ← 对 V^π_k 贪心
until π_{k+1} == π_k
```

由策略改进定理，$V^{\\pi_{k+1}} \\geq V^{\\pi_k}$，且当 $\\pi$ 是最优时严格收敛。
"""),

        code("""def policy_iteration(env, gamma=0.9, theta=1e-7, max_outer=50, record=False):
    nS, nA = env.nS, env.nA
    pi = np.full((nS, nA), 1.0 / nA)  # 初始均匀随机
    history_pi = [pi.copy()] if record else None
    history_V = [] if record else None
    for k in range(max_outer):
        V, _, _ = iterative_policy_eval_vec(env, pi, gamma=gamma, theta=theta)
        if record:
            history_V.append(V.copy())
        pi_new, _ = greedy_policy_from_V(env, V, gamma=gamma)
        if record:
            history_pi.append(pi_new.copy())
        if np.array_equal(pi_new.argmax(axis=1), pi.argmax(axis=1)):
            return V, pi, k + 1, history_V, history_pi
        pi = pi_new
    return V, pi, max_outer, history_V, history_pi


V_star_pi, pi_star, n_outer, hist_V, hist_pi = policy_iteration(env, gamma=0.9, record=True)
print(f"策略迭代在 {n_outer} 次外层迭代收敛")
print(f"V* 平均: {V_star_pi.mean():.3f}, max: {V_star_pi.max():.3f}")

# 画最优 V 和最优策略
arrows = ['↑','→','↓','←']
optimal_actions = pi_star.argmax(axis=1)
fig, ax = plt.subplots(figsize=(5, 5))
plot_value_heatmap(
    V_star_pi, env.shape, policy=optimal_actions, action_labels=arrows,
    cell_text=True, walls=list(env.walls), terminals=list(env.terminals),
    ax=ax, title='V* + π* (Policy Iteration)',
)
plt.tight_layout(); plt.show()
"""),

        md("""## 3.5 值迭代（Value Iteration）

策略迭代的问题：每次外层迭代都要做一次完整的策略评估（收敛到 $V^\\pi$）——开销大。

**值迭代**的洞察：评估不必完全收敛！只要做一次"贝尔曼 backup"就够了：

$$
V_{k+1}(s) = \\max_a \\sum_{s'} P(s'|s,a) \\big[ r(s,a,s') + \\gamma V_k(s') \\big]
$$

注意和迭代策略评估的区别：多了个 $\\max_a$。这就是**贝尔曼最优性方程**的迭代形式。

### 贝尔曼最优性方程

最优价值函数 $V^*$ 满足：

$$
\\boxed{\\; V^*(s) = \\max_a \\sum_{s'} P(s'|s,a) \\big[ r(s,a,s') + \\gamma V^*(s') \\big] \\;}
$$

这是一个**非线性**方程（因为 max），所以不能直接矩阵求逆。但**值迭代**可以收敛到它。

### 策略迭代 vs 值迭代

| | 策略迭代 | 值迭代 |
|---|---|---|
| 外层 | 改进策略 | 改进 V |
| 内层 | 完整策略评估 | 一次 backup |
| 复杂度 | 每个外层多步内层 | 每步都是一次 backup |
| 实际效率 | 状态多时慢 | 通常更快 |

两者最终都收敛到 $V^*$。
"""),

        code("""def value_iteration(env, gamma=0.9, theta=1e-7, max_iters=10000, record=False):
    \"\"\"值迭代。\"\"\"
    nS, nA = env.nS, env.nA
    V = np.zeros(nS)
    history = [V.copy()] if record else None
    for it in range(max_iters):
        # Q[s, a] = Σ_{s'} P(s'|s,a) [R(s,a) + γ V(s')]
        Q = env.R + gamma * np.einsum('saq,q->sa', env.P, V)
        new_V = Q.max(axis=1)
        # 终止态为 0
        for s in range(nS):
            if env.is_terminal(s):
                new_V[s] = 0.0
        delta = np.abs(new_V - V).max()
        V = new_V
        if record:
            history.append(V.copy())
        if delta < theta:
            break
    # 提取最优策略
    Q_final = env.R + gamma * np.einsum('saq,q->sa', env.P, V)
    pi = np.zeros((nS, nA))
    pi[np.arange(nS), Q_final.argmax(axis=1)] = 1.0
    return V, pi, it + 1, history


V_star_vi, pi_star_vi, n_iters_vi, hist_V_vi = value_iteration(env, gamma=0.9, record=True)
print(f"值迭代在 {n_iters_vi} 次迭代收敛")
print(f"V* 与策略迭代结果的差异: {np.abs(V_star_vi - V_star_pi).max():.2e}")
"""),

        md("""## 3.6 值迭代动画：看 V 怎么收敛

> 🤔 **先猜再跑**：动画从 V₀ = 0（一片黑）开始播放值迭代的每一轮扫描。预测一下**第 1 轮扫描后**，网格里哪些格子已经有值、哪些还是 0？
>
> <details><summary>想好了再点开</summary>
>
> 第 1 轮只有**终点的直接邻居**非零：贝尔曼 backup 一次只能"看到一步远"的奖励。像水波从终点荡开，每扫一轮多传一格——这个"波速"就是后面 TD 学习（Ch04）里 credit 沿时间回传的速度上限。
> </details>

下面动画展示值迭代从 $V_0 = 0$ 开始的演化。你应该看到 **奖励信号从终点 +1 倒着传回起点**。
"""),

        code("""# 因为值迭代收敛快（~30 步），我们直接画前 20 步
fig, ax = plt.subplots(figsize=(5, 5))
n_frames = min(20, len(hist_V_vi))

def update(k):
    ax.clear()
    plot_value_heatmap(
        hist_V_vi[k], env.shape,
        cell_text=True, walls=list(env.walls), terminals=list(env.terminals),
        ax=ax, title=f'value iteration  iter={k}',
    )

anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=400, blit=False, repeat=True)
plt.close(fig)
HTML(anim.to_jshtml())
"""),

        md("""## 3.7 比较两个算法的效率

我们让两个算法都收敛到 $\\theta = 10^{-7}$，看哪个快。
"""),

        code("""import time

# 值迭代
t0 = time.time()
V_vi, pi_vi, n_vi, _ = value_iteration(env, gamma=0.9, theta=1e-7)
t_vi = time.time() - t0

# 策略迭代
t0 = time.time()
V_pi, pi_pi, n_pi_outer, _, _ = policy_iteration(env, gamma=0.9, theta=1e-7)
t_pi = time.time() - t0

print(f"{'算法':<14}{'总迭代数':<14}{'时间(s)':<10}{'最终 max V':<10}")
print(f"{'值迭代':<14}{n_vi:<14}{t_vi:<10.3f}{V_vi.max():<10.3f}")
print(f"{'策略迭代':<14}{n_pi_outer:<14}{t_pi:<10.3f}{V_pi.max():<10.3f}")
print(f"\\n两者 V 的差异: {np.abs(V_vi - V_pi).max():.2e}")
"""),

        md("""## 3.8 数值验证：策略迭代的单调改进

策略改进定理说每步 $V^{\\pi_{k+1}} \\geq V^{\\pi_k}$。我们看一下记录的 history 是不是这样。
"""),

        code("""# 我们重跑一次，记录每个外层迭代的 V
print(f"{'iter':<6}{'mean(V^π_k)':<14}{'max(V^π_k)':<14}{'Δ':<12}")
prev_mean = -np.inf
for k, Vk in enumerate(hist_V):
    cur_mean = Vk.mean()
    delta = cur_mean - prev_mean if k > 0 else 0
    print(f"{k:<6}{cur_mean:<14.4f}{Vk.max():<14.4f}{delta:<+12.4f}")
    prev_mean = cur_mean
"""),

        md("""你应该看到 $V$ 的均值**单调递增**，且增量很快变小——这就是策略改进定理的数值体现。

## 3.9 📝 练习

### 练习 1：Modified Policy Iteration

策略迭代每次评估到收敛 $\\theta = 10^{-7}$，开销大。一个折中方案叫 **Modified Policy Iteration**：

- 每次评估只做 $k$ 次 sweep（不等到收敛）
- 然后立即贪心改进

**任务**：
1. 实现 `modified_policy_iteration(env, k=5, ...)`
2. 对比 $k \\in \\{1, 2, 5, 10, 20, 50\\}$，找出能"以最少总运算量收敛到接近最优"的 $k$
3. 画出 $k$ vs 总 sweep 数 + 最终 $V$ 误差

**预期结果**：$k \\approx 5$ 通常比纯策略迭代（$k \\to \\infty$）和纯值迭代（$k=1$）都好。

> 参考答案：`solutions/ch03_modified_policy_iteration.ipynb`

---

## 3.10 小结

| 算法 | 每轮做什么 | 收敛到 |
|---|---|---|
| 策略迭代 | 完整评估 $V^\\pi$ → 贪心改进 | $\\pi^*$ |
| 值迭代 | 一次贝尔曼最优回扫 $V \\leftarrow \\max_a \\dots$ | $V^*$ |
| Modified PI | $k$ 次 sweep 评估 + 改进（练习 1） | $\\pi^*$ |

三个带走的东西：

1. **DP = 已知模型时的精确解法**：它并不实用（需要完整 $P$、状态数爆炸），但给后面所有近似算法提供了"标准答案"来对照
2. **策略改进定理**是迭代收敛的数学保证：贪心改进永不变差、有限策略数内必达最优
3. **γ<1 ⇒ 压缩映射**：误差每轮以 γ 几何衰减——这套收敛分析在 Ch04 的 TD 学习里会原样复用

> 📖 学完本章，先做 `STUDY_GUIDE.md` 里 Ch03 的自测题（3 题），全对再进下一章。

---

下一章：**第 4 章 — TD 学习**。
我们将放弃 "知道 P 和 R" 的假设——agent 必须**从交互样本中学习**。这是 RL 真正走入现实的起点。
"""),
    ]


def ch04() -> List[Cell]:
    """Ch04 TD 学习：MC、TD(0)、n-step、TD(λ)、eligibility traces。"""
    return [
        md("""# 第 4 章：TD 学习 —— 从样本中学习

上一章的 DP 有个让人不放心的前提：**P 和 R 你全知道**。可真实世界谁来告诉你转移概率？学开车的乘客不会给你一张「踩油门 0.3 秒后速度分布表」——你只能**自己开、自己感受、自己总结**。这一章，agent 终于走进现实：不再有模型，只有一条条真实经历过的轨迹样本。

从样本里估计期望，最朴素的想法是蒙特卡洛（MC）：把回报平均一下就行。但 MC 有个致命的别扭之处——**必须等整局结束**才能更新。下了一半的棋，前面那步的好坏要等终局才知道？我们将在本章造出 RL 最重要的一个思想：**TD（时序差分）**——不等终局，每走一步就用「下一步的猜测」修正「这一步的猜测」。你在生活里早就在用它：考试交卷前检查出一道错题时，你会立刻修正对「这类题我会不会」的判断——而不是等到出分那天。

> 🌍 **真实世界**：TD 的第一个成名作是 1992 年的 TD-Gammon——一个**从零开始**、只靠自我对弈的输赢信号学会西洋双陆棋的程序，棋力达到人类世界冠军水平。它是 AlphaGo 的精神祖先，证明了「不用人类棋谱、只靠 TD + 自我对弈」这条路走得通。

## 学习目标

1. 理解 **蒙特卡洛（MC）** 和 **TD(0)** 的本质区别
2. 推导 TD(0) 的更新规则与**期望收敛性**
3. 复现 Sutton-Barto 的经典 **Random Walk** 实验
4. 掌握 **n-step TD** 和 **TD(λ)**（eligibility traces）
5. 直观理解 **bias/variance tradeoff**
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed, make_interactive, plot_bar_compare
from rlenvs import RandomWalk, GridWorld, small_grid_5x5

set_seed(0)
"""),

        md("""## 4.1 放弃"知道 P 和 R"的假设

Ch03 的 DP 算法用到了：

- $P(s'|s,a)$：从 $s$ 选 $a$ 到 $s'$ 的概率
- $R(s,a)$：奖励函数

但现实中你**不知道**这些。比如开车时：
- 你不知道前面那辆车下一秒会怎么开
- 你不知道某个路况会不会打滑

**Model-free RL** 只用采到的轨迹 $(S_0, A_0, R_1, S_1, A_1, R_2, \\dots)$ 学。

## 4.2 蒙特卡洛（MC）估计

最直接的想法：**多次采样，求平均**。

### First-visit MC

对每个状态 $s$，记录所有 "首次访问 $s$" 之后的回报 $G_t$，取平均作为 $V(s)$。

### Every-visit MC

不要求"首次"，每次访问 $s$ 都记录 $G_t$。

两者在 episode 长、状态访问次数有限时**有差异**，但都收敛到 $V^\\pi$。

### MC 的特点

- **无偏**：估计的 $V$ 不带系统性偏差
- **高方差**：每次 $G_t$ 是从 $t$ 到终止的累计奖励，方差大
- **不需要 bootstrap**：不依赖其他 $V$ 估计
- **必须等 episode 结束**：无法在线学习
"""),

        code("""def run_episode_uniform(env_grid, start_state, gamma=1.0, max_steps=500):
    \"\"\"在 GridWorld 上从 start_state 出发，用均匀随机策略走完一个 episode。\"\"\"
    env_grid._state = start_state
    trajectory = []
    done = False
    t = 0
    s = start_state
    while not done and t < max_steps:
        a = np.random.randint(env_grid.nA)
        s_next, r, done, _ = env_grid.step(a)
        trajectory.append((s, a, r, s_next))
        s = s_next
        t += 1
    return trajectory


def mc_estimate_first_visit(env_grid, gamma=0.9, n_episodes=5000):
    \"\"\"first-visit MC，估计 V^π（π 是均匀随机策略）。\"\"\"
    nS = env_grid.nS
    returns_sum = np.zeros(nS)
    returns_cnt = np.zeros(nS)
    for _ in range(n_episodes):
        start = env_grid.reset()
        traj = run_episode_uniform(env_grid, start, gamma=gamma)
        # 反向累计 G，每步存 (s, G_t)
        G = 0.0
        Gs_rev = []
        for s, a, r, s_next in reversed(traj):
            G = r + gamma * G
            Gs_rev.append((s, G))
        # Gs_rev 是反向的，转回来
        Gs = list(reversed(Gs_rev))
        # first-visit：只取每个 s 第一次出现的 G
        seen = set()
        for s, G in Gs:
            if s in seen:
                continue
            seen.add(s)
            returns_sum[s] += G
            returns_cnt[s] += 1
    V = np.where(returns_cnt > 0, returns_sum / np.maximum(returns_cnt, 1), 0.0)
    return V, returns_cnt


env = small_grid_5x5(seed=0)
V_mc, cnt = mc_estimate_first_visit(env, gamma=0.9, n_episodes=3000)
print(f"MC 估计的 V^π（用了 {cnt.sum()} 个 first-visit 样本）")
print(V_mc.reshape(env.shape).round(2))
"""),

        md("""## 4.3 TD(0)：把 MC 和 DP 结合起来

TD(0) 的洞察：**不必等 episode 结束**。每一步都用 **当前一步奖励 + 对下一步的估计** 来更新。

### TD(0) 更新规则

$$
V(S_t) \\leftarrow V(S_t) + \\alpha \\big[ \\underbrace{R_{t+1} + \\gamma V(S_{t+1})}_{\\text{TD target}} - V(S_t) \\big]
$$

记号：

- **TD target**：$R_{t+1} + \\gamma V(S_{t+1})$，新的 $G_t$ 估计
- **TD error**：$\\delta_t = R_{t+1} + \\gamma V(S_{t+1}) - V(S_t)$，更新方向

直觉：**朝 TD target 走一小步**。

### 和 MC、DP 的关系

| | MC | DP | TD(0) |
|---|---|---|---|
| 用样本？ | ✓ | ✗ | ✓ |
| 用 bootstrap（依赖其他 V）？ | ✗ | ✓ | ✓ |

TD(0) 是两者的混血：**采样** + **bootstrap**。

### TD(0) 的特点

- **有偏**：因为 target 用了其他 $V$ 估计（这些估计不准）
- **低方差**：只看一步奖励
- **在线**：每步都能更新，不用等 episode 结束
- **在 continuing task（无终止）也能用**

## 4.4 TD(0) 的收敛性

<details>
<summary><b>📝 TD(0) 在期望下的等价更新（点开看）</b></summary>

考虑 $\\alpha \\to 0$、无穷多样本的极限。对 $V(S_t)$ 的更新量为：

$$
\\mathbb{E}[\\delta_t | S_t = s] = \\mathbb{E}[R_{t+1} + \\gamma V(S_{t+1}) - V(s) | S_t = s]
$$

$$
= \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a) [r(s,a,s') + \\gamma V(s')] - V(s)
$$

$$
= (T^\\pi V)(s) - V(s)
$$

其中 $T^\\pi$ 是贝尔曼 backup 算子。当 $V = V^\\pi$ 时 $(T^\\pi V)(s) = V(s)$，所以 $V^\\pi$ 是这个动力系统的不动点。

由 $T^\\pi$ 的 $\\gamma$-压缩性（$\\|T^\\pi V_1 - T^\\pi V_2\\| \\leq \\gamma \\|V_1 - V_2\\|$），TD(0) 在 step-size 满足 Robbins-Monro 条件（$\\sum \\alpha = \\infty, \\sum \\alpha^2 < \\infty$）时几乎必然收敛到 $V^\\pi$。

</details>

**直觉**：TD(0) 是用采样实现的 Ch03 迭代策略评估。
"""),

        code("""def td0_estimate(env_grid, gamma=0.9, alpha=0.1, n_episodes=5000, max_steps=200):
    \"\"\"TD(0) 估计 V^π（π 是均匀随机）。\"\"\"
    nS = env_grid.nS
    V = np.zeros(nS)
    for _ in range(n_episodes):
        s = env_grid.reset()
        done = False
        t = 0
        while not done and t < max_steps:
            a = np.random.randint(env_grid.nA)
            s_next, r, done, _ = env_grid.step(a)
            # TD(0) 更新
            td_target = r + gamma * (0.0 if done else V[s_next])
            V[s] += alpha * (td_target - V[s])
            s = s_next
            t += 1
    return V


V_td = td0_estimate(env, gamma=0.9, alpha=0.05, n_episodes=5000)

# 用 DP 算精确解对比
pi_uniform = np.full((env.nS, env.nA), 1.0 / env.nA)
R_pi = (pi_uniform * env.R).sum(axis=1)
P_pi = np.einsum('sa,saq->sq', pi_uniform, env.P)
V_exact = np.linalg.solve(np.eye(env.nS) - 0.9 * P_pi, R_pi)

print(f"{'state':<6}{'V_exact':<12}{'V_td0':<12}{'V_mc':<12}")
for s in [0, 5, 10, 15, 20, 24]:
    print(f"{s:<6}{V_exact[s]:<12.3f}{V_td[s]:<12.3f}{V_mc[s]:<12.3f}")

print(f"\\nTD(0) vs exact 最大误差: {np.abs(V_td - V_exact).max():.3f}")
print(f"MC vs exact 最大误差: {np.abs(V_mc - V_exact).max():.3f}")
"""),

        md("""## 4.5 经典实验：Sutton-Barto Random Walk

**这是 RL 教材最经典的图之一**——Sutton & Barto 教科书图 6.2 的复现，几乎每门 RL 课都会让学生亲手跑一遍。

> 🤔 **先猜再跑**：下面的实验跑 α = 0.05/0.10/0.15/0.20 四档的 TD(0)，画「各状态的 V 估计随 episode 变化」的折线，真值是灰色横线。预测：**episode 数很少时（比如 3 个），哪一档 α 的估计离真值最远？猜一个方向（整体偏高/偏低/乱跳）再跑。**
>
> <details><summary>写下猜测再点开</summary>
>
> 提示：TD target 里有 V(s')——初期 V 全是 0，target 系统性偏小，估计**从下往上爬**。α 越大爬得越快、但每一步都被单条轨迹的噪声拽得越狠。你会同时看到 bias 的「爬升」和 variance 的「毛刺」——这张图就是 §4.6 bias/variance 分析的实物版。
> </details>

Sutton-Barto 19 状态随机游走：

- 19 个状态排一条直线（编号 1..19）
- 从中间 (state 10) 出发
- 每步 50% 往左、50% 往右
- 落到 state 0（左端）奖励 -1
- 落到 state 20（右端）奖励 +1

真实 $V^\\pi$ 是线性插值：$V(s) = -1 + 2s/20$，从 -0.9 到 +0.9。

我们看 TD(0) 在不同 $\\alpha$ 下能多快收敛到真值。
"""),

        code("""env_rw = RandomWalk(n_states=19, left_reward=-1.0, right_reward=1.0, seed=0)
true_V = env_rw.true_values()
print(f"真实 V^π：{true_V.round(2)}")

def td0_random_walk(env_rw, alpha=0.1, n_episodes=10):
    \"\"\"TD(0) 在 RandomWalk 上估计 V^π。\"\"\"
    V = np.zeros(env_rw.nS)  # 内部状态 1..n_states
    V_history = [V.copy()]
    for ep in range(n_episodes):
        s = env_rw.reset()  # 返回 1..n_states
        done = False
        while not done:
            s_next, r, done, _ = env_rw.step()
            # 转回内部 0-indexed
            s_idx = s - 1
            s_next_idx = s_next - 1 if s_next != 0 and s_next != env_rw.nS + 1 else 0
            td_target = r if done else r + 0.9 * V[s_next_idx]
            V[s_idx] += alpha * (td_target - V[s_idx])
            s = s_next
        V_history.append(V.copy())
    return V, V_history


# 在前几次 episode 的几个 α 下画 V
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
alphas = [0.05, 0.1, 0.15]
episodes_to_show = [0, 1, 10, 50, 100]

for col, alpha in enumerate(alphas):
    np.random.seed(0)
    V, hist = td0_random_walk(env_rw, alpha=alpha, n_episodes=100)
    for row, ep_idx in enumerate([1, 100]):
        ax = axes[row][col]
        x = np.arange(1, 20)
        ax.plot(x, true_V, 'k-', linewidth=2, label='true $V^\\pi$')
        ax.plot(x, hist[min(ep_idx, len(hist)-1)], 'r.--', linewidth=1.5, label=f'TD est (ep={ep_idx})')
        ax.set_title(f'α={alpha}, episode={ep_idx}')
        ax.set_xlabel('state')
        ax.set_ylabel('V')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),

        md("""你应该看到：

- **Episode 1 后**：TD 估计只在中间几个状态有非零值（因为只走过那些）
- **Episode 100 后**：TD 几乎完全对齐真值
- **α 越大收敛越快**，但太大会震荡

## 4.6 MC vs TD 的 bias/variance

我们做一个对比实验：在 RandomWalk 上，比较 MC（first-visit）和 TD(0) 估计的**偏差**（系统性偏离真值）和**方差**。
"""),

        code("""def mc_random_walk(env_rw, gamma=1.0, n_episodes=100, alpha=0.01):
    \"\"\"first-visit MC，每步用 alpha 增量更新。\"\"\"
    V = np.zeros(env_rw.nS)
    for ep in range(n_episodes):
        # 跑一个 episode
        s = env_rw.reset()
        trajectory = []
        done = False
        while not done:
            s_next, r, done, _ = env_rw.step()
            trajectory.append((s, r))
            s = s_next
        # 算 G_t 反向
        G = 0.0
        visited_G = {}
        for s, r in reversed(trajectory):
            G = r + gamma * G
            visited_G[s] = G
        for s, G in visited_G.items():
            V[s - 1] += alpha * (G - V[s - 1])
    return V


# 跑 100 次实验，统计 bias 和 variance
n_runs = 100
n_eps = 100
V_td_runs = np.zeros((n_runs, 19))
V_mc_runs = np.zeros((n_runs, 19))
for run in range(n_runs):
    np.random.seed(run)
    env_rw = RandomWalk(n_states=19, seed=run)
    V_td_runs[run] = td0_random_walk(env_rw, alpha=0.1, n_episodes=n_eps)[0]
    V_mc_runs[run] = mc_random_walk(env_rw, n_episodes=n_eps, alpha=0.01)

td_bias = (V_td_runs.mean(axis=0) - true_V).mean()
mc_bias = (V_mc_runs.mean(axis=0) - true_V).mean()
td_var = V_td_runs.var(axis=0).mean()
mc_var = V_mc_runs.var(axis=0).mean()

print(f"{'':<10}{'TD(0)':<12}{'MC':<12}")
print(f"{'bias':<10}{td_bias:<12.4f}{mc_bias:<12.4f}")
print(f"{'variance':<10}{td_var:<12.4f}{mc_var:<12.4f}")

# 画一个柱状图
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_bar_compare([abs(td_bias), abs(mc_bias)], ['TD(0)', 'MC'],
                 colors=['steelblue', 'crimson'], title='|bias|', ax=axes[0])
plot_bar_compare([td_var, mc_var], ['TD(0)', 'MC'],
                 colors=['steelblue', 'crimson'], title='variance', ax=axes[1])
plt.tight_layout(); plt.show()
"""),

        md("""你应该看到：

- **TD(0) 偏差更大**（因为 bootstrap 引入偏差）
- **MC 方差更大**（因为累计奖励方差叠加）
- 这是经典的 **bias-variance tradeoff**

## 4.7 n-step TD：MC 和 TD(0) 之间

TD(0) 只看 1 步。MC 看到终止。**n-step TD** 看中间任意 $n$ 步：

$$
G_t^{(n)} = R_{t+1} + \\gamma R_{t+2} + \\dots + \\gamma^{n-1} R_{t+n} + \\gamma^n V(S_{t+n})
$$

直觉：

- $n = 1$：TD(0)，**低方差、高偏差**
- $n = \\infty$（或到终止）：MC，**高方差、无偏差**
- $n$ 中间：折中
"""),

        code("""def n_step_td_random_walk(env_rw, n=3, alpha=0.1, gamma=1.0, n_episodes=100):
    \"\"\"n-step TD on RandomWalk。\"\"\"
    V = np.zeros(env_rw.nS)
    for ep in range(n_episodes):
        s = env_rw.reset()
        # 缓存 rewards 和 states
        states = [s]
        rewards = [0]  # 占位，让索引对齐
        T = float('inf')
        t = 0
        while True:
            if t < T:
                s_next, r, done, _ = env_rw.step()
                states.append(s_next)
                rewards.append(r)
                if done:
                    T = t + 1
            tau = t - n + 1  # 现在 update 的时刻
            if tau >= 0:
                # G_tau^{(n)}
                G = 0.0
                upper = min(tau + n, T)
                for k in range(tau + 1, upper + 1):
                    G += gamma ** (k - tau - 1) * rewards[k]
                if tau + n < T:
                    s_end_idx = states[tau + n] - 1
                    if 0 <= s_end_idx < env_rw.nS:
                        G += gamma ** n * V[s_end_idx]
                s_tau_idx = states[tau] - 1
                if 0 <= s_tau_idx < env_rw.nS:
                    V[s_tau_idx] += alpha * (G - V[s_tau_idx])
            t += 1
            if tau == T - 1:
                break
            if t > 1000:  # 安全 break
                break
    return V


# 比较 n=1, 3, ∞（MC）
n_steps_to_test = [1, 2, 3, 5, 10, 30]
n_runs = 30
rmses = []
for n in n_steps_to_test:
    errs = []
    for run in range(n_runs):
        np.random.seed(run)
        env_rw = RandomWalk(n_states=19, seed=run)
        V = n_step_td_random_walk(env_rw, n=n, alpha=0.1 if n <= 5 else 0.05, n_episodes=10)
        err = np.sqrt(np.mean((V - true_V) ** 2))
        errs.append(err)
    rmses.append(np.mean(errs))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(n_steps_to_test, rmses, 'o-', linewidth=2, markersize=10)
ax.set_xlabel('n (steps)')
ax.set_ylabel('RMS error over 10 episodes')
ax.set_title('n-step TD：n 越大方差越大')
ax.set_xscale('log')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),

        md("""## 4.8 TD(λ)：用 eligibility traces 平滑所有 n

$n$-step TD 让你**选一个 n**。但如果我想**同时用所有 n 的平均**呢？

**TD(λ)** 用 backward view（eligibility traces）实现这个想法：

### Eligibility trace

每个状态 $s$ 有一个 trace $e_t(s)$，每步更新：

$$
e_t(s) = \\gamma \\lambda e_{t-1}(s) + \\mathbb{1}[S_t = s]
$$

- $\\lambda = 0$：只有当前状态有 trace，等价于 TD(0)
- $\\lambda = 1$：trace 不衰减，等价于 MC（在 episodic 任务里）
- $\\lambda \\in (0, 1)$：在两者之间

### TD(λ) 更新

每步用**同一个 TD error** 更新**所有状态**，按它们的 trace 加权：

$$
\\delta_t = R_{t+1} + \\gamma V(S_{t+1}) - V(S_t)
$$

$$
V(s) \\leftarrow V(s) + \\alpha \\delta_t e_t(s), \\quad \\forall s
$$

直觉：最近访问过的状态、且权重没衰减太多（$\\gamma \\lambda$）的状态，应该承担这次误差更多。
"""),

        code("""def td_lambda_random_walk(env_rw, lam=0.5, alpha=0.1, gamma=1.0, n_episodes=100):
    \"\"\"TD(λ) with eligibility traces on RandomWalk。\"\"\"
    V = np.zeros(env_rw.nS)
    for ep in range(n_episodes):
        e = np.zeros(env_rw.nS)  # eligibility trace
        s = env_rw.reset()
        done = False
        while not done:
            s_next, r, done, _ = env_rw.step()
            s_idx = s - 1
            s_next_idx = max(0, min(env_rw.nS - 1, s_next - 1))
            td_target = r if done else r + gamma * V[s_next_idx]
            delta = td_target - V[s_idx]
            # 更新 trace：所有状态的 trace 都衰减
            e = gamma * lam * e
            e[s_idx] += 1.0
            # 用同一个 delta 更新所有状态
            V += alpha * delta * e
            s = s_next
    return V


# 比较 λ = 0, 0.3, 0.5, 0.8, 1
lambdas = [0.0, 0.3, 0.5, 0.8, 0.9, 1.0]
n_runs = 30
rmses = []
for lam in lambdas:
    errs = []
    for run in range(n_runs):
        np.random.seed(run)
        env_rw = RandomWalk(n_states=19, seed=run)
        V = td_lambda_random_walk(env_rw, lam=lam, alpha=0.1, n_episodes=10)
        err = np.sqrt(np.mean((V - true_V) ** 2))
        errs.append(err)
    rmses.append(np.mean(errs))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lambdas, rmses, 's-', linewidth=2, markersize=10, color='purple')
ax.set_xlabel('λ')
ax.set_ylabel('RMS error over 10 episodes')
ax.set_title('TD(λ)：中间的 λ 往往最好')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),

        md("""**典型结果**：$\\lambda \\approx 0.5$ 附近最好——既不太偏、又不太噪。

## 4.9 交互式 widget：调 $\\alpha$ 和 $\\lambda$
"""),

        code("""def td_lambda_demo(alpha=0.1, lam=0.5):
    np.random.seed(0)
    env_rw = RandomWalk(n_states=19, seed=0)
    V = td_lambda_random_walk(env_rw, lam=lam, alpha=alpha, n_episodes=20)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, 20)
    ax.plot(x, true_V, 'k-', linewidth=2, label='true $V^\\pi$')
    ax.plot(x, V, 'r.--', linewidth=1.5, label=f'TD(λ) est')
    ax.set_title(f'α={alpha}, λ={lam}, 20 episodes')
    ax.set_xlabel('state'); ax.set_ylabel('V')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-1.1, 1.1)
    plt.tight_layout(); plt.show()

w = make_interactive(td_lambda_demo,
                     params={'alpha': (0.1, 0.01, 0.5, 0.01),
                             'lam':   (0.5, 0.0, 1.0, 0.05)},
                     layout='hbox')
"""),

        md("""## 4.10 小结

| 方法 | 看几步 | 偏差 | 方差 | 在线 |
|---|---|---|---|---|
| MC | 全部 | 无 | 高 | ✗（等 episode 结束） |
| TD(0) | 1 步 | 有 | 低 | ✓ |
| n-step | n 步 | 中 | 中 | ✓ |
| TD(λ) | 全部加权（按 λ） | 可调 | 可调 | ✓ |

**核心收获**：

1. **bootstrap**（用估计更新估计）是 RL 的关键设计选择
2. TD 把 MC 和 DP 的优势结合：**采样** + **bootstrap**
3. **bias-variance tradeoff** 是 RL 调参的核心
4. **TD(λ) 的 eligibility trace 思想**会在后面的 Actor-Critic、PPO 里反复出现

## 4.11 📝 练习

### 练习 1：在 GridWorld 上比较 MC vs TD(0)

用 `small_grid_5x5`，对比 MC 和 TD(0) 估计的 $V^\\pi$：

1. 固定 $\\gamma = 0.9$，跑 100 个 seed
2. 画两条 RMS-error-vs-episodes 曲线
3. 哪个收敛更快？为什么？

### 练习 2：online TD(λ) for control

把本章的 TD(λ) 推广到 **SARSA(λ)**：用 $Q$ 替代 $V$，结合 Ch05 的 SARSA 思想，在 GridWorld 上学习 $\\pi^*$。

**提示**：eligibility trace 从 $e(s)$ 变成 $e(s, a)$——每次 step 后全体 `e *= γλ`、刚访问的 `(s, a)` 加 1；更新式 `Q += α · δ · e`，其中 δ 用 SARSA 的 target（$r + \\gamma Q(s', a') - Q(s, a)$，$a'$ 是实际采样的下一个动作）。

> 参考答案：练习 1 → `solutions/ch04_mc_vs_td_gridworld.ipynb`；练习 2 是开放练习（无参考答案）——写完后可以和 Ch05 的 Q-learning 对比收敛速度。
>
> 📖 两章都完成后，做 `STUDY_GUIDE.md` 里 Ch04 的自测题（4 题）。

---

下一章：**第 5 章 — Q-learning 和 SARSA**。
我们将从 **预测** 跳到 **控制**——学习最优策略 $\\pi^*$，并区分 **on-policy vs off-policy** 这个 PPO/GRPO 都绕不开的核心概念。
"""),
    ]


def ch05() -> List[Cell]:
    """Ch05 Q-learning / SARSA：on-policy vs off-policy 的核心。"""
    return [
        md("""# 第 5 章：Q-learning 和 SARSA —— 第一个完整控制算法

> Ch04 学了**预测**（给定 $\\pi$，估 $V^\\pi$）。这一章学**控制**：找最优 $\\pi^*$。
> 我们将用两个看起来几乎一样、本质却截然不同的算法，引出 RL 中最重要的概念之一：
> **on-policy vs off-policy**。

## 学习目标

1. 理解 **SARSA（on-policy）** 和 **Q-learning（off-policy）** 的更新规则
2. 推导两者的更新公式，并证明期望形式的等价性
3. 复现 Sutton-Barto **CliffWalk** 的经典对比图：SARSA 保守、Q-learning 激进
4. 理解 **maximization bias** 并掌握 **Double Q-learning**
5. 牢记 on/off-policy 的区别——它是 PPO / GRPO 设计的核心
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed, plot_training_curve, plot_q_table, make_interactive
from rlenvs import CliffWalk, GridWorld

set_seed(0)
"""),

        md("""## 5.1 从预测到控制：为什么要用 $Q$ 而不是 $V$

Ch04 用 TD 学 $V$。但**控制**（找最优策略）需要比较**所有动作**的优劣，$V$ 不够用。

我们改用 **$Q(s, a)$**：在 $s$ 选 $a$、之后按 $\\pi$ 行动的期望回报。一旦有了 $Q$，最优策略就是

$$
\\pi^*(s) = \\arg\\max_a Q^*(s, a)
$$

### ε-greedy 探索

Ch01 的 ε-greedy 在这里复用：

$$
\\pi(a|s) = \\begin{cases} 1 - \\epsilon + \\epsilon/|\\mathcal{A}| & a = \\arg\\max_{a'} Q(s, a') \\\\ \\epsilon/|\\mathcal{A}| & \\text{otherwise} \\end{cases}
$$

## 5.2 SARSA：on-policy TD 控制

### 算法

每一步采一个 transition $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$（注意需要 $A_{t+1}$），然后更新：

$$
Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha \\big[ R_{t+1} + \\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \\big]
$$

名字来源：$S, A, R, S', A'$。

### 关键性质

**target 里的 $Q(S_{t+1}, A_{t+1})$ 中的 $A_{t+1}$ 是从 $\\pi$ 采出来的**——也就是和 $A_t$ 同一个策略。

这就是 **on-policy**：用 $\\pi$ 采的数据，去评估/改进 $\\pi$ 本身。
"""),

        md("""## 5.3 Q-learning：off-policy TD 控制

### 算法

每步用 $(S_t, A_t, R_{t+1}, S_{t+1})$ 更新（**不需要 $A_{t+1}$**）：

$$
Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha \\big[ R_{t+1} + \\gamma \\max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \\big]
$$

**关键区别**：target 里是 $\\max_{a'} Q(S_{t+1}, a')$，**不需要按 $\\pi$ 采样**。

### off-policy 解释

虽然行为策略（用来采数据的）还是 ε-greedy，但 target 假设下一步会**走最大 Q 的动作**——也就是假设下一步是 **greedy 策略**。

所以 Q-learning 是 **off-policy**：行为策略 ≠ 目标策略。

### 期望形式：SARSA 和 Q-learning 的微妙差异

<details>
<summary><b>📝 期望形式等价：SARSA(λ=0) ≈ Expected SARSA ≈ Q-learning（点开看）</b></summary>

考虑 SARSA 在期望下（对 $A_{t+1}$ 求期望）：

$$
\\mathbb{E}_{A_{t+1} \\sim \\pi}[Q(S_{t+1}, A_{t+1})] = \\sum_a \\pi(a | S_{t+1}) Q(S_{t+1}, a)
$$

这叫 **Expected SARSA**。它的方差比 SARSA 小（不用采样 $A_{t+1}$）。

如果 $\\pi$ 是 greedy（即 $\\pi(a|s) = \\mathbb{1}[a = \\arg\\max Q]$），那么：

$$
\\sum_a \\pi(a|s) Q(s, a) = \\max_a Q(s, a)
$$

**Expected SARSA 在 greedy 策略下 = Q-learning**。

所以 Q-learning ≈ SARSA 在"目标策略是 greedy"的特例。
</details>
"""),

        md("""## 5.4 实现：SARSA 和 Q-learning
"""),

        code("""def epsilon_greedy_action(Q, s, epsilon):
    \"\"\"ε-greedy 动作选择。\"\"\"
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[s]))


def sarsa(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    \"\"\"SARSA 算法。\"\"\"
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))
    episode_rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        a = epsilon_greedy_action(Q, s, epsilon)
        done = False
        ep_reward = 0.0
        while not done:
            s_next, r, done, _ = env.step(a)
            ep_reward += r
            a_next = epsilon_greedy_action(Q, s_next, epsilon) if not done else 0
            td_target = r + (0 if done else gamma * Q[s_next, a_next])
            Q[s, a] += alpha * (td_target - Q[s, a])
            s, a = s_next, a_next
        episode_rewards.append(ep_reward)
    return Q, np.array(episode_rewards)


def q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    \"\"\"Q-learning 算法。\"\"\"
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))
    episode_rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            a = epsilon_greedy_action(Q, s, epsilon)
            s_next, r, done, _ = env.step(a)
            ep_reward += r
            td_target = r + (0 if done else gamma * np.max(Q[s_next]))
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next
        episode_rewards.append(ep_reward)
    return Q, np.array(episode_rewards)


# 验证两个算法在简单 GridWorld 上能学到东西
env = CliffWalk(seed=0)
Q_sarsa, rw_sarsa = sarsa(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1)
Q_ql, rw_ql = q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1)

print(f"SARSA 最后 50 episodes 平均奖励: {rw_sarsa[-50:].mean():.2f}")
print(f"Q-learning 最后 50 episodes 平均奖励: {rw_ql[-50:].mean():.2f}")
"""),

        md("""## 5.5 CliffWalk：SARSA 保守、Q-learning 激进

**这是 Sutton-Barto 最经典的图之一**。

4×12 CliffWalk：
- 起点 $(3, 0)$、终点 $(3, 11)$
- 中间 row=3, col=1..10 是悬崖，落入 -100，回到起点
- 每步 -1（鼓励尽快到达）

**直觉预期**：
- **Q-learning 学到 "贴着悬崖边走最短路"**——因为 target 假设下一步 greedy（不会掉下去），但行为 ε-greedy 偶尔会掉下去
- **SARSA 学到 "远离悬崖的更安全路径"**——因为它考虑 ε-greedy 下可能掉下去的事实
"""),

        code("""# 跑 30 个 seed，对比
n_seeds = 30
n_eps = 500
all_rw_sarsa = np.zeros((n_seeds, n_eps))
all_rw_ql = np.zeros((n_seeds, n_eps))
for seed in range(n_seeds):
    env_s = CliffWalk(seed=seed)
    _, rw_s = sarsa(env_s, n_episodes=n_eps, alpha=0.5, epsilon=0.1)
    env_q = CliffWalk(seed=seed)
    _, rw_q = q_learning(env_q, n_episodes=n_eps, alpha=0.5, epsilon=0.1)
    all_rw_sarsa[seed] = rw_s
    all_rw_ql[seed] = rw_q

# 画两条曲线
fig, ax = plt.subplots(figsize=(9, 5))
sm_sarsa = np.convolve(all_rw_sarsa.mean(0), np.ones(20)/20, mode='valid')
sm_ql = np.convolve(all_rw_ql.mean(0), np.ones(20)/20, mode='valid')
ax.plot(sm_sarsa, label='SARSA (on-policy)', linewidth=2)
ax.plot(sm_ql, label='Q-learning (off-policy)', linewidth=2)
ax.set_xlabel('episode (smoothed w=20)')
ax.set_ylabel('reward')
ax.set_title('CliffWalk：SARSA 更稳，Q-learning 更激进')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"SARSA 渐进奖励: {all_rw_sarsa[:, -50:].mean():.2f}")
print(f"Q-learning 渐进奖励: {all_rw_ql[:, -50:].mean():.2f}")
print(f"最优值（贴悬崖走）: -12")
"""),

        md("""你应该看到：

- **训练期**：Q-learning 经常掉悬崖（reward -100 频繁），SARSA 更稳
- **渐进 reward**：Q-learning ≈ -12（最优路径）但方差大；SARSA ≈ -25~−30（绕远更稳）

**核心洞察**：
- Q-learning 学的 $\\pi^*$ 是"如果 ε=0 的最优"
- SARSA 学的 $\\pi^*$ 是"考虑 ε 探索成本的最优"
- ε 越小，两者越接近

## 5.6 看一眼 SARSA 学到的策略

我们把学到的 $\\pi$ 画出来，看 SARSA 是不是真的"绕开了悬崖"。
"""),

        code("""def render_cliff_policy(Q, ax, title='Learned policy'):
    \"\"\"在 CliffWalk 上画策略。\"\"\"
    ax.clear()
    n_rows, n_cols = 4, 12
    # 悬崖
    from matplotlib.patches import Rectangle, Circle
    for c in range(1, 11):
        ax.add_patch(Rectangle((c, 3), 1, 1, color='crimson', alpha=0.4))
    # 终点
    ax.add_patch(Rectangle((11, 3), 1, 1, color='gold', alpha=0.7))
    # 起点
    # 画策略箭头
    arrows = {0: (0, 0.4), 1: (0.4, 0), 2: (0, -0.4), 3: (-0.4, 0)}
    for s in range(48):
        r, c = divmod(s, 12)
        if r == 3 and 1 <= c <= 10:  # 悬崖
            continue
        if s == 47:  # 终点
            continue
        a = int(np.argmax(Q[s]))
        dr, dc = arrows[a]
        ax.arrow(c + 0.5, r + 0.5, dc, -dr, head_width=0.15, head_length=0.1, fc='navy', ec='navy')
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)

env = CliffWalk(seed=0)
Q_sarsa, _ = sarsa(env, n_episodes=2000, alpha=0.5, epsilon=0.1)
env = CliffWalk(seed=0)
Q_ql, _ = q_learning(env, n_episodes=2000, alpha=0.5, epsilon=0.1)

fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
render_cliff_policy(Q_sarsa, axes[0], title='SARSA 学到的策略（绕远路）')
render_cliff_policy(Q_ql, axes[1], title='Q-learning 学到的策略（贴悬崖）')
plt.tight_layout(); plt.show()
"""),

        md("""## 5.7 Maximization Bias：为什么 Q-learning 会"过度乐观"

Q-learning 的 target 里有一个 $\\max$。**max 操作会引入正偏差**。

### 直觉

假设真值 $Q^*(s, a) = 0$ 对所有 $a$，但你的估计 $Q(s, a)$ 有噪声（围绕 0 抖）。

$\\max_a Q(s, a)$ 的期望 = $\\mathbb{E}[\\max_a Q(s, a)] > \\max_a \\mathbb{E}[Q(s, a)] = 0$

也就是"max 的期望 > 期望的 max"。这叫 **Jensen 不等式**。

后果：Q-learning 系统性高估某些 $(s, a)$，可能反复去试那些被高估的次优动作。

### 一个最小例子

考虑一个状态 $s$ 有两个动作 $a_1, a_2$：
- $a_1$：奖励 $\\sim N(0, 1)$（其实 0 期望）
- $a_2$：奖励 $\\sim N(-0.1, 1)$（其实稍微负）

**真实最优**是 $a_1$（期望 0）。但 Q-learning 可能在某次采样里看到 $a_2$ 偶然给个大正数，把它的 $Q$ 高估，然后反复去试 $a_2$。

> 🤔 **先猜再跑**：下面这个实验会跑 200 个 seed、每个 300 episode，统计 Q-learning **最后 50 个 episode 里选 $a_2$（次优动作）的频率**。你猜是接近 0%（学明白了）、5%（偶尔犯迷糊）、还是 30%+（系统性偏差）？
>
> <details><summary>写下你的百分比再点开</summary>
>
> 关键在于 max：两个动作的 Q 都有噪声，而 `max` 永远挑**当下看起来更高**的那个——噪声里"虚高"的一侧更容易被选中。这不是偶尔犯迷糊，是结构性偏袒。猜 5% 的读者，准备被结果惊讶。
> </details>
"""),

        code("""class OneStateTrap:
    \"\"\"单状态、两动作的 trap。\"\"\"
    def __init__(self):
        self.nS = 1
        self.nA = 2
        self._rng = np.random.default_rng()
    def reset(self):
        return 0
    def step(self, a):
        # a=0: N(0,1)；a=1: N(-0.1, 1)
        r = self._rng.normal(0, 1) if a == 0 else self._rng.normal(-0.1, 1)
        return 0, r, True, {}


def q_learning_trap(n_episodes=300, alpha=0.1, gamma=0.0, epsilon=0.1):
    env = OneStateTrap()
    Q = np.zeros((1, 2))
    a2_frequencies = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        actions_taken = []
        while not done:
            a = epsilon_greedy_action(Q, s, epsilon)
            actions_taken.append(a)
            s_next, r, done, _ = env.step(a)
            td_target = r  # γ=0, 没有下一状态
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next
        a2_frequencies.append(np.mean(actions_taken))
    return Q, np.array(a2_frequencies)


# 跑 1000 个 seed
n_seeds = 500
freq_a2 = np.zeros((n_seeds, 300))
for seed in range(n_seeds):
    np.random.seed(seed)
    Q, freq = q_learning_trap(n_episodes=300, alpha=0.1, epsilon=0.1)
    freq_a2[seed] = freq

print(f"最优 a=0 的频率应该是 0.5（ε=0.1 + 90% greedy 选 a=0）")
print(f"实际 Q-learning 选 a=1 的平均频率（最后 50 episodes）：{freq_a2[:, -50:].mean():.3f}")
print(f"Q[0, 0] 期望 ~ 0, Q[0, 1] 期望 ~ -0.1")
print(f"实际 Q[0]: {Q}")

# 画频率曲线
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.convolve(freq_a2.mean(0), np.ones(20)/20, mode='valid'),
        label='Q-learning: 频率 a=1', linewidth=2)
ax.axhline(0.1 * 0.5, color='gray', linestyle='--', label='理论下界 ε * 0.5')
ax.set_xlabel('episode (smoothed)')
ax.set_ylabel('P(a=1)')
ax.set_title('Maximization Bias：Q-learning 高估了 a=1')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),

        md("""## 5.8 Double Q-learning：解决方案

**思路**：把经验随机分两半，维护两个独立的 $Q_1, Q_2$。每步用其中一个评估、用另一个选择动作：

```
以 0.5 概率更新 Q_1：
    a* = argmax_a Q_1[s', a]
    Q_1[s, a] += α [r + γ Q_2[s', a*] - Q_1[s, a]]
否则更新 Q_2：
    a* = argmax_a Q_2[s', a]
    Q_2[s, a] += α [r + γ Q_1[s', a*] - Q_2[s, a]]
```

**关键**：用 $Q_1$ 选动作、用 $Q_2$ 评估，两者独立→期望上消除了 max 的偏差。

## 5.9 📝 练习：实现 Double Q-learning

**任务**：

1. 实现 `double_q_learning(env, ...)` 函数
2. 在上面的 OneStateTrap 上验证它消除了 maximization bias（选 a=1 的频率应该接近 ε/2 = 0.05）
3. 在 CliffWalk 上对比 Q-learning 和 Double Q-learning

**预期结果**：
- Double Q-learning 在 trap 上选 a=1 的频率从 ~25% 降到 ~5%
- 在 CliffWalk 上 Double Q-learning 性能略好（方差更小）

> 参考答案：`solutions/ch05_double_q_learning.ipynb`

## 5.10 小结

| | SARSA | Q-learning |
|---|---|---|
| 策略 | on-policy | off-policy |
| Target | $R + \\gamma Q(s', a')$ | $R + \\gamma \\max_{a'} Q(s', a')$ |
| 行为策略 | $\\epsilon$-greedy（与目标同） | $\\epsilon$-greedy（与目标不同） |
| CliffWalk | 保守绕远 | 激进贴崖 |
| 偏差 | 较小 | 有 maximization bias |

### on-policy vs off-policy（**重要！**）

> **on-policy**：用 $\\pi$ 采的数据，训练 $\\pi$ 自己
> **off-policy**：用一个 behavior 策略采的数据，训练另一个 target 策略

这个区别会在后面的 PPO / GRPO 中反复出现：

- **PPO 是 on-policy**：必须用当前 $\\pi_\\theta$ 采的数据更新 $\\pi_\\theta$。每次更新后数据"作废"。
- **Q-learning / DQN 是 off-policy**：可以用任意策略采的数据训练目标策略。**经验回放（replay buffer）成为可能**——大幅提高样本效率。

**为什么 PPO 在 LLM 上更好？** Ch13 我们会看到，GRPO 之所以取代 PPO 的"LLM 版本"，正是因为它进一步简化了 on-policy 训练的复杂度。

---

## Phase 1 结束！

你已经学完了经典 RL 的核心：

- ✅ Ch00 RL 全景
- ✅ Ch01 多臂老虎机：探索 vs 利用
- ✅ Ch02 MDP + 贝尔曼方程
- ✅ Ch03 动态规划：精确求解
- ✅ Ch04 TD 学习：从样本中学习
- ✅ Ch05 SARSA / Q-learning：on-policy vs off-policy

接下来 **Phase 2**：

- **Ch05b PyTorch 速成（没用过 PyTorch 的读者先读这个，1 小时）**
- Ch06 DQN + 函数逼近（神经网络 + 经验回放）
- Ch07 策略梯度定理
- Ch08 Actor-Critic + GAE
- Ch09 TRPO + PPO（**整个 Phase 2 的核心**）

然后 **Phase 3**：

- Ch13 GRPO（DeepSeek-R1 的核心算法）

请你**在跳到 Phase 2 之前**，确保：

1. 能默写出贝尔曼期望方程（Ch02）
2. 能用一两句话解释 on-policy vs off-policy（Ch05）
3. 知道 TD(0) 和 MC 的核心区别（Ch04）

这三个知识点是后面所有内容的钥匙。更完整的门槛自测见 `STUDY_GUIDE.md` 的"Phase 1 → 2 门槛"。
"""),
    ]


# =============================================================================
# Ch05b：PyTorch 速成（Phase 1 → Phase 2 的过渡章）
# =============================================================================
def ch05b() -> List[Cell]:
    """Ch05b PyTorch 速成：只用过 numpy 的读者，进 Ch06 前的 1 小时过渡。

    Phase 1（Ch00-05）全部用 numpy；Ch06 的 DQN 起切入 PyTorch。
    本章只讲后面章节真正用到的那部分 PyTorch，不讲深度学习理论。
    """
    return [
        md("""# 第 5b 章：PyTorch 速成——从 numpy 到 `loss.backward()`

> **写给谁**：Phase 1 我们全程用 numpy（把认知负载留给 RL 本身）；从 Ch06 的 DQN 开始要用 PyTorch。
> 如果你**写过 PyTorch / 熟悉 autograd**，直接跳到本章末尾的"5b.6 揭开 Ch06 的黑盒"即可。
> 如果你只用过 numpy——这一章就是为你写的，花 1 小时，后面 13 章都会顺很多。

## 学习目标

读完本章后你应该能：

1. 用 `torch.Tensor` 做你熟悉的一切 numpy 操作
2. 说清楚 `requires_grad` / `backward()` / `.grad` 三者的关系
3. 默写**标准训练五步循环**：`zero_grad → forward → loss → backward → step`
4. 定义一个 `nn.Module`，并解释 `train()` / `eval()` 的区别
5. 看懂 Ch06 会用到的三个操作：`gather`、`torch.no_grad`、target network 拷贝

## 为什么从 Ch06 起切换 PyTorch？

Ch05 的 Q-learning 用**表格**存 $Q(s,a)$——状态太多时表格爆炸（CartPole 的连续状态就存不下）。
Ch06 起我们用**神经网络**逼近 $Q$，而"网络参数 $\\theta$ 关于 loss 的梯度"
交给 PyTorch 的 autograd 自动算：**你写前向，它算反向**。我们手写 replay buffer、训练循环、
PPO clip——只有求导这一步交给框架（这是本教材"不依赖黑盒 RL 库"原则的边界）。

> 🌍 **真实世界**：PyTorch 是当下 LLM 训练的事实标准——GPT 系列、LLaMA、DeepSeek、Qwen 的训练代码都是 PyTorch。你在这里学的 `loss.backward()` 五步循环，和那些千亿参数模型的训练脚本里的是同一套动词。
"""),

        code("""# 自动设置 sys.path（和 Ch00 一样）
import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    raise SystemExit(
        "本章需要 PyTorch：请先 pip install torch（CPU 版即可），再重启 kernel 重跑。"
    )

import numpy as np
print(f"torch: {torch.__version__}  |  numpy: {np.__version__}")
"""),

        md("""## 5b.1 Tensor：会 numpy 就会一半

`torch.Tensor` 和 `np.ndarray` 的 API 高度一致——下面这张对照表覆盖了本教材用到的全部操作：

| 你在 numpy 里写的 | PyTorch 里写 | 备注 |
|---|---|---|
| `np.zeros((3,4))` | `torch.zeros(3,4)` | shape 不再是 tuple |
| `np.ones(5)` | `torch.ones(5)` | |
| `A @ B` | `A @ B` | 矩阵乘完全一样 |
| `A.sum(axis=1)` | `A.sum(dim=1)` | `axis` 改叫 `dim` |
| `np.argmax(A)` | `A.argmax()` | |
| `A[0:2]` | `A[0:2]` | 切片一样 |
| `A[[0,2]]` | `A[[0,2]]` 或 `A.index_select(0, idx)` | |
| `rng.uniform(...)` | `torch.rand(...)` | 随机数见下 |
"""),

        code("""import torch

# 创建 + 运算：和 numpy 几乎逐字对应
A = torch.rand(2, 3)            # 均匀分布 [0,1)，相当于 rng.uniform(0,1,(2,3))
B = torch.randn(2, 3)           # 标准正态
print("A =", A)
print("A.sum(dim=1) =", A.sum(dim=1))
print("A.argmax(dim=1) =", A.argmax(dim=1))

# 和 numpy 互转（共享内存，零拷贝——改一边另一边也变！）
a_np = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(a_np)      # numpy -> torch
back = t.numpy()                # torch -> numpy
print("torch.from_numpy:", t, "| .numpy():", back)

# 默认 dtype 是 float32（不是 numpy 的 float64）——神经网络的标准精度
print("默认 dtype:", torch.ones(1).dtype)
"""),

        md("""**两个和 numpy 不同的习惯**要提前记住：

1. **随机性**：`torch.manual_seed(0)` 对应 `np.random.seed(0)`；本教材统一用 `from utils import set_seed`，它会同时 seed numpy / random / torch。
2. **广播（broadcasting）规则与 numpy 完全一致**：`[B, T, V] * [B, T, 1]` 会自动广播到最后一个维度。后面章节大量依赖这个。

## 5b.2 autograd：PyTorch 的灵魂

一句话：**你用 tensor 算出 loss，调 `loss.backward()`，PyTorch 自动算出 loss 关于所有"标记了 `requires_grad=True` 的 tensor"的梯度**，存进它们的 `.grad` 属性。

$$
\\text{设 } f(x) = 3x^2 + 2x \\quad \\Rightarrow \\quad \\frac{df}{dx} = 6x + 2
$$
"""),

        code("""x = torch.tensor([1.0], requires_grad=True)   # 标记：我要对 x 求导
f = 3 * x**2 + 2 * x
f.backward()                                    # 反向传播：自动算 df/dx
print(f"f(x=1) = {f.item()}")
print(f"解析梯度 df/dx = 6*1+2 = 8.0")
print(f"autograd 算出 x.grad = {x.grad.item()}")

# 教材传统：数值验证——用有限差分 (f(x+h) - f(x-h)) / (2h) 对比
# 注意用 float64：float32 下 x±h 的舍入误差被 /2h 放大，差分会不准
xd = torch.tensor([1.0], dtype=torch.float64)
h = 1e-6
fd = ((3*(xd+h)**2 + 2*(xd+h)) - (3*(xd-h)**2 + 2*(xd-h))) / (2*h)
print(f"有限差分梯度          = {fd.item():.8f}")
print(f"解析 / autograd       = 8.0 / {x.grad.item()}")
print("两者一致 ✓" if abs(fd.item() - x.grad.item()) < 1e-4 else "不一致 ✗")
"""),

        md("""注意上面第一次出现的 `with torch.no_grad():`——**在块内的运算不记录梯度**。
为什么需要它：算"验证用的数值"或"target 值"时我们不需要梯度，关掉能省内存、避免误更新。

### 训练五步循环（背下来）

```python
optimizer.zero_grad()   # ① 清上一次的梯度（梯度默认累加，不清零会越加越大）
loss = criterion(model(x), y)   # ② 前向：算预测 + loss
loss.backward()         # ③ 反向：算 d loss / d 所有参数
optimizer.step()        # ④ 用梯度更新参数（如 θ -= lr * grad）
# ⑤ 重复 ①-④，直到 loss 收敛
```

> 最常见的初学者 bug：**忘写 ①**。PyTorch 的梯度是累加的（`.grad +=`），不清零等于用了一个越来越大的错误梯度。
"""),

        code("""# 完整最小示例：用神经网络拟合 y = sin(x)（CPU 上几秒）
from utils import set_seed
set_seed(0)

# 待拟合的数据
x = torch.linspace(-3, 3, 200).unsqueeze(1)    # [200, 1]
y = torch.sin(x)

# 一个两层 MLP（5b.3 会详细讲 nn.Module）
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64), torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)   # Adam：自适应步长
criterion = torch.nn.MSELoss()

losses = []
for step in range(500):
    optimizer.zero_grad()          # ①
    pred = model(x)                # ② 前向
    loss = criterion(pred, y)
    loss.backward()                # ③
    optimizer.step()               # ④
    losses.append(loss.item())     # ⑤

print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
"""),

        code("""# 看看拟合效果
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(losses)
axes[0].set_title("training loss")
axes[0].set_yscale("log")
with torch.no_grad():               # 画图不需要梯度
    axes[1].scatter(x, y, s=6, label="data: sin(x)")
    axes[1].plot(x, model(x), color="red", label="network")
axes[1].legend()
axes[1].set_title("fit result")
plt.show()
"""),

        md("""## 5b.3 `nn.Module`：所有网络的基类

后面章节的 Q 网络、策略网络、TinyGPT 全是 `nn.Module` 的子类。它替你做三件事：

1. **登记参数**：`self.linear = nn.Linear(...)` 里的权重会自动进 `model.parameters()`，optimizer 拿这个列表去更新
2. **`model(x)` 自动调 `forward`**：`__call__` 会挂钩子（hook），别直接调 `model.forward(x)`
3. **模式切换**：`model.train()` / `model.eval()` 控制 dropout、batchnorm 等层的行为

> ⚠️ **第三点是本教材代码里真实踩过的坑**：eval() 之后忘了切回 train()，后续训练就在错误的模式下跑。
> 记住配对规则——**rollout / 采样 / 评估时 eval()，用完立刻 train() 回来**。
"""),

        code("""class TinyMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()                          # 必须先调父类构造
        self.net = torch.nn.Sequential(             # 子模块也会被登记
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x):                           # 只定义前向
        return self.net(x)

m = TinyMLP(4, 32, 2)
print(m)                                            # 自动生成的结构描述
print(f"参数量: {sum(p.numel() for p in m.parameters())}")   # (4*32+32) + (32*2+2) = 226

# train/eval 模式查看
print("默认 training 模式:", m.training)   # True
m.eval()
print("eval() 后:", m.training)            # False
m.train()
print("train() 后:", m.training)           # True
"""),

        md("""## 5b.4 Ch06 特供：三个马上要用的操作

**① `gather`——按索引取 Q 值**。DQN 的 loss 需要 $Q(s_t, a_t)$：网络输出**所有**动作的 Q `[B, n_actions]`，但我们只要**实际执行的那个动作**的 Q。这就是 gather：

```python
q = q_net(states)                      # [B, n_actions]
q_sa = q.gather(1, actions)            # [B, 1]  取第 b 行第 actions[b] 个
```

**② `torch.no_grad()`——构造 TD target**。target $= r + \\gamma \\max_{a'} Q_{\\text{target}}(s', a')$ 是**监督信号**，不该有梯度流过（否则会"追自己的尾巴"，即 Ch06 §6.x 的半梯度概念）。

**③ target network 拷贝**：`target_net.load_state_dict(online_net.state_dict())` 把在线网络整套参数复制给目标网络（隔 N 步做一次，稳定训练）。
"""),

        code("""from utils.networks import QNetwork   # Ch06 用的真实 Q 网络（5b.6 揭晓内部）

set_seed(0)
q_net = QNetwork(state_dim=4, n_actions=2)     # CartPoleLite 的形状

states  = torch.rand(8, 4)                     # batch = 8 条 transition
actions = torch.randint(0, 2, (8, 1))          # 每条实际执行的动作 [B, 1]

# ① gather
q_all = q_net(states)                          # [8, 2]
q_sa = q_all.gather(1, actions)                # [8, 1]
print("q_all[0] =", q_all[0].tolist(), "| action[0] =", actions[0].item(),
      "| gather 对得上:", abs(q_all[0, actions[0].item()].item() - q_sa[0].item()) < 1e-7)

# ② no_grad 构造 target（模仿 dqn_utils.dqn_update_step 里的关键两行）
rewards, dones = torch.ones(8), torch.zeros(8)
with torch.no_grad():
    q_next_max = q_net(states).max(dim=1).values       # 假装这是 target_net
    target = rewards + 0.99 * q_next_max * (1 - dones)
print("target 无梯度（detach 于 no_grad）:", not target.requires_grad)

# ③ target network 拷贝
target_net = QNetwork(state_dim=4, n_actions=2)       # 初始随机（和 q_net 不同）
target_net.load_state_dict(q_net.state_dict())        # 现在完全一致
with torch.no_grad():
    diff = (q_net(states) - target_net(states)).abs().max()
print(f"拷贝后两网络输出最大差异: {diff.item():.2e}  (应为 0)")
"""),

        md("""## 5b.5 揭开 Ch06 的黑盒：`utils/networks.py` 的 `QNetwork`

Ch06 直接 `from utils.networks import QNetwork`——现在你已经能完全读懂它了。逐行对照上面的知识：

- `make_mlp(...)`：把 `[Linear, ReLU, Linear, ...]` 叠成 `nn.Sequential`（就是 5b.3 的 TinyMLP 换个写法）
- `forward`：`x -> self.net(x)`，输入 state `[B, 4]`，输出 Q 值 `[B, n_actions]`
"""),

        code("""import inspect
from utils.networks import make_mlp, QNetwork
print(inspect.getsource(QNetwork))
# 练习（不用写代码）：说出 QNetwork(state_dim=4, n_actions=2) 有多少参数、
# forward 的输入输出 shape——答不上来就回 5b.3 / 5b.4 再看一遍。
"""),

        md("""## 5b.6 常见坑速查（后面章节踩到时回来翻）

| 症状 | 原因 | 修复 |
|---|---|---|
| loss 不降反升 / 梯度爆炸 | 忘了 `optimizer.zero_grad()` | 五步循环①别省 |
| 显存越跑越满 | 存了带梯度的张量（如整个 loss 历史） | 用 `.item()` / `.detach()` 只存数值 |
| `RuntimeError: element 0 of tensors does not require grad` | 在 `no_grad` 块里做训练 | 检查缩进，训练代码别放进 no_grad |
| 评估指标随机跳变 | dropout 还开着（没 eval()） | 评估前 `model.eval()`，用完 `model.train()` |
| numpy 和 tensor 混用报错 | `np.ndarray` 直接喂网络 | `torch.as_tensor(x, dtype=torch.float32)` |

## 小结

- ✅ tensor ≈ numpy（`axis`→`dim`，默认 float32）
- ✅ autograd：`requires_grad` 标记 → `backward()` 计算 → `.grad` 读取；数值验证用有限差分
- ✅ 训练五步循环：`zero_grad → forward → loss → backward → step`
- ✅ `nn.Module` 管参数 / train / eval 模式
- ✅ DQN 三件套：`gather` 取 $Q(s,a)$、`no_grad` 造 target、`load_state_dict` 拷贝网络

下一章：**第 6 章 — DQN + 函数逼近**。神经网络进场，replace 表格——但探索、replay、target network 这些 RL 的核心难题，全部手写。

> 📖 学完 Ch06 后记得做 `STUDY_GUIDE.md` 里 Ch06 的自测题。
"""),
    ]


# =============================================================================
# 主入口
# =============================================================================

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"

# 全部注册 notebook 的注册表。
# Ch00-05b：在本文件内 inline 定义（List[Cell] 元组）。
# Ch06-09 + Ch11-14：通过 import build_chXX.py 包装（见上方 ch06()-ch14()）。
# Ch15：capstone，无 build_ch15.py，notebooks/ch15_capstone.ipynb 手维护、不重建。
NOTEBOOK_BUILDERS = [
    ("ch00_setup_and_overview.ipynb",           ch00),
    ("ch01_multi_armed_bandits.ipynb",          ch01),
    ("ch02_mdps_and_bellman.ipynb",             ch02),
    ("ch03_dynamic_programming.ipynb",          ch03),
    ("ch04_td_learning.ipynb",                  ch04),
    ("ch05_q_learning_sarsa.ipynb",             ch05),
    ("ch05b_pytorch_primer.ipynb",              ch05b),
    ("ch06_dqn_function_approximation.ipynb",   ch06),
    ("ch07_policy_gradient.ipynb",              ch07),
    ("ch08_actor_critic_gae.ipynb",             ch08),
    ("ch09_trpo_ppo.ipynb",                     ch09),
    ("ch10_tiny_gpt.ipynb",                     ch10),
    ("ch11_reward_modeling.ipynb",              ch11),
    ("ch12_rlhf_ppo.ipynb",                     ch12),
    ("ch13_grpo.ipynb",                         ch13),
    ("ch14_dpo_kto.ipynb",                      ch14),
    ("ch19_agentic_rl.ipynb",                   ch19),
    ("ch20_agentic_grpo.ipynb",                 ch20),
    # Ch15-18: 手维护/独立脚本，不在此自动重建（见文件头注释）
    # Ch15: capstone 项目
    # Ch16: PRM（utils/prm.py + 手维护 notebook）
    # Ch17: Self-Play + Constitutional AI / RLAIF（utils/self_play.py + 手维护 notebook）
    # Ch18: Offline RL（utils/offline_rl.py + 手维护 notebook）
]

# 给 --chapters / --chapter 参数用的章节号 -> builder 映射。
# 章节号用 00-14（与文件名 chXX_ 前缀对应）；05b 是 PyTorch 过渡章。
_CHAPTER_INDEX = {
    "00": ch00, "01": ch01, "02": ch02, "03": ch03, "04": ch04, "05": ch05,
    "05b": ch05b,
    "06": ch06, "07": ch07, "08": ch08, "09": ch09,
    "10": ch10, "11": ch11, "12": ch12, "13": ch13, "14": ch14,
    "19": ch19, "20": ch20,
}


def _parse_range(spec: str) -> List[str]:
    """把 '06-09' / '00-14' / '13' 解析成 ['06','07','08','09'] 这种列表。

    支持带字母后缀的单章（如 '05b'）；范围语法只支持纯数字。
    """
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        lo_i, hi_i = int(lo), int(hi)
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        return [f"{i:02d}" for i in range(lo_i, hi_i + 1)]
    else:
        # 单个章节号（可能是 '05b' 这种带后缀的）
        try:
            return [f"{int(spec):02d}"]
        except ValueError:
            return [spec]


def _filter_builders(chapters: List[str]) -> List[Tuple[str, callable]]:
    """根据章节号列表过滤 NOTEBOOK_BUILDERS。"""
    out = []
    for ch_id in chapters:
        if ch_id == "15":
            print(f"  [skip] ch15 capstone notebook 是手维护的，跳过；"
                  f"如需修改请直接编辑 notebooks/ch15_capstone.ipynb")
            continue
        if ch_id == "16":
            print(f"  [skip] ch16 PRM notebook 是手维护的，跳过；"
                  f"如需修改请直接编辑 notebooks/ch16_prm.ipynb")
            continue
        if ch_id == "17":
            print(f"  [skip] ch17 Self-Play + RLAIF notebook 是手维护的，跳过；"
                  f"如需修改请直接编辑 notebooks/ch17_self_play_cai.ipynb "
                  f"(重建请用 build_ch17.py + ch17_content.txt)")
            continue
        if ch_id == "18":
            print(f"  [skip] ch18 Offline RL notebook 是手维护的，跳过；"
                  f"如需修改请直接编辑 notebooks/ch18_offline_rl.ipynb "
                  f"(重建请用 build_ch18.py + ch18_content.txt)")
            continue
        if ch_id not in _CHAPTER_INDEX:
            print(f"  [warn] 未知章节号 ch{ch_id}，跳过")
            continue
        target_fname = next(
            (fname for fname, _ in NOTEBOOK_BUILDERS
             if fname.startswith(f"ch{ch_id}_")),
            None,
        )
        if target_fname is None:
            print(f"  [warn] ch{ch_id} 在 NOTEBOOK_BUILDERS 中找不到，跳过")
            continue
        out.append((target_fname, _CHAPTER_INDEX[ch_id]))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="重建 RLStudy notebooks。默认重建注册的 15 章 + 5 个 solutions。"
    )
    parser.add_argument(
        "--chapters",
        type=str,
        default=None,
        help="只重建指定章节范围，如 '06-09'（Phase 2）、'00-05'（Phase 1）、'10-14'（Phase 3）。"
             "不指定则重建全部。注意 ch15-18 永远跳过（手维护/独立脚本）。",
    )
    parser.add_argument(
        "--chapter",
        type=str,
        default=None,
        help="只重建单个章节，如 '13'。和 --chapters 互斥（--chapter 优先）。",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出全部已注册的 notebook，不写文件。",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("已注册的 notebook：")
        for fname, builder in NOTEBOOK_BUILDERS:
            print(f"  {fname}  ({builder.__name__})")
        print(f"（另：ch15_capstone.ipynb 手维护；ch16-18 用各自的 build_chXX.py 重建，均不在此注册）")
        return

    # 选择要构建的章节
    if args.chapter is not None:
        chapters = _parse_range(args.chapter)
    elif args.chapters is not None:
        chapters = _parse_range(args.chapters)
    else:
        chapters = None  # 全部

    if chapters is None:
        builders = NOTEBOOK_BUILDERS
    else:
        builders = _filter_builders(chapters)
        if not builders:
            print("没有匹配的章节，退出。")
            return

    print(f"生成 notebooks（{len(builders)} 章）...")
    for fname, builder in builders:
        cells = builder()
        nb = build_notebook(cells)
        save(nb, NOTEBOOKS_DIR / fname)

    # 只有不指定 --chapter / --chapters 时才一并重建 solutions，避免每次单章构建
    # 都把 5 个 solution 也跑一遍。
    if chapters is None:
        print("生成 solutions...")
        for fname, builder in SOLUTION_BUILDERS:
            cells = builder()
            nb = build_notebook(cells)
            save(nb, SOLUTIONS_DIR / fname)
    print("完成。")


# =============================================================================
# Solutions
# =============================================================================

SOLUTIONS_DIR = Path(__file__).parent / "solutions"


def sol_ch01_thompson() -> List[Cell]:
    """Ch01 练习参考答案：Thompson Sampling for Bernoulli bandits。"""
    return [
        md("""# Ch01 练习参考答案：Thompson Sampling

> Bernoulli 多臂老虎机 + Beta 共轭先验 + 与 UCB1 对比。
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed
from rlenvs import MultiArmedBandit

set_seed(0)


class ThompsonBernoulliAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Beta(1,1) 先验
        self.beta = np.ones(n_arms)

    def reset(self):
        self.alpha[:] = 1; self.beta[:] = 1

    def select_action(self):
        # 从每个臂的后验采样
        theta = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(theta))

    def update(self, action, reward):
        # Bernoulli 奖励是 0/1
        if reward == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1


class UCBAgent:
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms, dtype=int)
        self.t = 0

    def reset(self):
        self.Q[:] = 0; self.N[:] = 0; self.t = 0

    def select_action(self):
        self.t += 1
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            return int(untried[0])
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(ucb))

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


def run_compare(agent_factory, env_factory, n_steps=1000, n_seeds=200):
    n_arms = None
    all_cum_regret = np.zeros((n_seeds, n_steps))
    all_opt = np.zeros((n_seeds, n_steps), dtype=bool)
    for seed in range(n_seeds):
        env = env_factory(seed)
        agent = agent_factory(env.n_arms)
        agent.reset()
        env.reset()
        a_star = env.optimal_arm()
        cum = 0.0
        for t in range(n_steps):
            a = agent.select_action()
            r = env.pull(a)
            agent.update(a, r)
            cum += env.q_star[a_star] - env.q_star[a]
            all_cum_regret[seed, t] = cum
            all_opt[seed, t] = (a == a_star)
    return all_cum_regret, all_opt


n_seeds, n_steps = 200, 1000
methods = {
    'Thompson': lambda: ThompsonBernoulliAgent,
    'UCB1 c=2': lambda: UCBAgent,
}
results = {}
for name, fac in methods.items():
    cum, opt = run_compare(fac(),
                            lambda seed: MultiArmedBandit(n_arms=10, reward_dist='bernoulli', seed=seed),
                            n_steps=n_steps, n_seeds=n_seeds)
    results[name] = (cum, opt)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for name, (cum, opt) in results.items():
    axes[0].plot(cum.mean(0), label=name, linewidth=2)
    axes[0].fill_between(np.arange(n_steps), cum.mean(0) - cum.std(0), cum.mean(0) + cum.std(0), alpha=0.15)
    sm = np.convolve(opt.mean(0), np.ones(50)/50, mode='valid')
    axes[1].plot(sm, label=name, linewidth=2)
axes[0].set_title('Cumulative regret')
axes[0].set_xlabel('step')
axes[0].legend()
axes[1].set_title('Optimal action % (smoothed w=50)')
axes[1].set_xlabel('step')
axes[1].legend()
plt.tight_layout(); plt.show()

print(f"最终累计 regret (均值 ± std):")
for name, (cum, _) in results.items():
    print(f"  {name:<12}: {cum[:, -1].mean():.1f} ± {cum[:, -1].std():.1f}")
print("\\nThompson 应该明显小于 UCB1（O(√T) vs O(ln T) 在 Bernoulli 上都好但 TS 更稳）")
"""),
    ]


def sol_ch02_bridge() -> List[Cell]:
    """Ch02 练习参考答案：Bridge Grid γ 阈值。"""
    return [
        md("""# Ch02 练习参考答案：Bridge Grid 中 γ 如何翻转策略

> 我们找 γ* 使得最优策略从"绕远"切到"抄近道"。
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed
from rlenvs import bridge_grid

set_seed(0)


def value_iteration(env, gamma, theta=1e-8, max_iters=10000):
    V = np.zeros(env.nS)
    for it in range(max_iters):
        Q = env.R + gamma * np.einsum('saq,q->sa', env.P, V)
        new_V = Q.max(axis=1)
        for s in range(env.nS):
            if env.is_terminal(s):
                new_V[s] = 0.0
        delta = np.abs(new_V - V).max()
        V = new_V
        if delta < theta:
            break
    return V


env = bridge_grid(seed=0)
print(f"shape: {env.shape}, 终点: {env.terminals}")
# 终点 (1, 4) = 1*5+4 = 9
# 桥起点 (1, 0) = 1*5+0 = 5
# 绕远起点 (0, 0) = 0

# 扫描 γ 找 V*(bridge_start) vs V*(roundabout_start)
gammas = np.linspace(0.0, 0.99, 100)
v_bridge = []
v_round = []
for g in gammas:
    V = value_iteration(env, gamma=g)
    v_bridge.append(V[5])   # 桥起点
    v_round.append(V[0])    # 绕远起点

v_bridge = np.array(v_bridge)
v_round = np.array(v_round)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gammas, v_bridge, label='V*(1,0) bridge 起点（直接走桥）', linewidth=2)
ax.plot(gammas, v_round, label='V*(0,0) 绕远起点', linewidth=2)
ax.set_xlabel('γ')
ax.set_ylabel('V*')
ax.set_title('Bridge Grid：γ 越大，绕远越值')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# 找最优动作在每个 γ 下从 (1, 0) 出发是什么
print(f"\\n从起点 (1, 0) 在不同 γ 下的最优策略：")
for g in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
    V = value_iteration(env, gamma=g)
    s = 5  # (1, 0)
    # 计算 Q[s, a]
    Q = env.R + g * np.einsum('saq,q->sa', env.P, V)
    best_a = int(np.argmax(Q[s]))
    names = ['↑', '→', '↓', '←']
    print(f"  γ={g:.2f}: 最优动作 = {names[best_a]}, V*(1,0) = {V[s]:.3f}")
"""),
    ]


def sol_ch03_mpi() -> List[Cell]:
    """Ch03 练习参考答案：Modified Policy Iteration。"""
    return [
        md("""# Ch03 练习参考答案：Modified Policy Iteration

> 对比不同 k 的 modified policy iteration 在收敛速度和精度上的权衡。
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import set_seed
from rlenvs import small_grid_5x5

set_seed(0)


def modified_policy_iteration(env, k=5, gamma=0.9, max_outer=100):
    nS, nA = env.nS, env.nA
    pi = np.full((nS, nA), 1.0 / nA)
    total_sweeps = 0
    for outer in range(max_outer):
        # k 次 sweep 的策略评估
        V = np.zeros(nS)
        for _ in range(k):
            Q = env.R + gamma * np.einsum('saq,q->sa', env.P, V)
            new_V = (pi * Q).sum(axis=1)
            for s in range(nS):
                if env.is_terminal(s):
                    new_V[s] = 0.0
            V = new_V
            total_sweeps += 1
        # 贪心改进
        Q = env.R + gamma * np.einsum('saq,q->sa', env.P, V)
        new_pi = np.zeros((nS, nA))
        new_pi[np.arange(nS), Q.argmax(axis=1)] = 1.0
        if np.array_equal(new_pi.argmax(axis=1), pi.argmax(axis=1)):
            return V, pi, outer + 1, total_sweeps
        pi = new_pi
    return V, pi, max_outer, total_sweeps


env = small_grid_5x5(seed=0)

# 真值（用纯策略迭代收敛）
def policy_iteration_exact(env, gamma=0.9, theta=1e-12):
    nS, nA = env.nS, env.nA
    pi = np.full((nS, nA), 1.0 / nA)
    while True:
        V = np.zeros(nS)
        while True:
            Q = env.R + gamma * np.einsum('saq,q->sa', env.P, V)
            new_V = (pi * Q).sum(axis=1)
            for s in range(nS):
                if env.is_terminal(s):
                    new_V[s] = 0.0
            if np.abs(new_V - V).max() < theta:
                break
            V = new_V
        Q = env.R + gamma * np.einsum('saq,q->sa', env.P, V)
        new_pi = np.zeros((nS, nA))
        new_pi[np.arange(nS), Q.argmax(axis=1)] = 1.0
        if np.array_equal(new_pi.argmax(axis=1), pi.argmax(axis=1)):
            return V, pi
        pi = new_pi

V_true, _ = policy_iteration_exact(env, gamma=0.9)

ks = [1, 2, 3, 5, 10, 20, 50, 100]
results = []
for k in ks:
    t0 = time.time()
    V, pi, n_outer, total_sweeps = modified_policy_iteration(env, k=k, gamma=0.9)
    t = time.time() - t0
    err = np.abs(V - V_true).max()
    results.append((k, n_outer, total_sweeps, err, t))
    print(f"k={k:<4}  outer iters={n_outer:<3}  total sweeps={total_sweeps:<4}  "
          f"V 误差={err:.2e}  时间={t*1000:.1f}ms")

print()
print("观察：")
print("- k=1（值迭代）通常 total sweeps 较大，但每次 sweep 便宜")
print("- k=5~10 通常 total sweeps 最少")
print("- k→∞ 接近纯策略迭代")
"""),
    ]


def sol_ch04_mc_td() -> List[Cell]:
    """Ch04 练习参考答案：MC vs TD 在 GridWorld。"""
    return [
        md("""# Ch04 练习参考答案：MC vs TD(0) 在 GridWorld
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed
from rlenvs import small_grid_5x5

set_seed(0)


def td0_estimate(env, gamma=0.9, alpha=0.05, n_episodes=2000):
    V = np.zeros(env.nS)
    rms_history = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = np.random.randint(env.nA)
            s_next, r, done, _ = env.step(a)
            td_target = r + (0 if done else gamma * V[s_next])
            V[s] += alpha * (td_target - V[s])
            s = s_next
        rms_history.append(np.sqrt(np.mean(V ** 2)))
    return V, np.array(rms_history)


def mc_first_visit(env, gamma=0.9, alpha=0.01, n_episodes=2000):
    V = np.zeros(env.nS)
    rms_history = []
    for ep in range(n_episodes):
        s = env.reset()
        traj = []
        done = False
        while not done:
            a = np.random.randint(env.nA)
            s_next, r, done, _ = env.step(a)
            traj.append((s, r))
            s = s_next
        G = 0
        Gs = []
        for s, r in reversed(traj):
            G = r + gamma * G
            Gs.append((s, G))
        seen = set()
        for s, G in reversed(Gs):
            if s in seen:
                continue
            seen.add(s)
            V[s] += alpha * (G - V[s])
        rms_history.append(np.sqrt(np.mean(V ** 2)))
    return V, np.array(rms_history)


env = small_grid_5x5(seed=0)
n_seeds = 30
n_eps = 1500
td_runs = np.zeros((n_seeds, n_eps))
mc_runs = np.zeros((n_seeds, n_eps))
for seed in range(n_seeds):
    np.random.seed(seed)
    env = small_grid_5x5(seed=seed)
    _, td_hist = td0_estimate(env, alpha=0.05, n_episodes=n_eps)
    _, mc_hist = mc_first_visit(env, alpha=0.01, n_episodes=n_eps)
    td_runs[seed] = td_hist
    mc_runs[seed] = mc_hist

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(td_runs.mean(0), label='TD(0) α=0.05', linewidth=2)
ax.plot(mc_runs.mean(0), label='MC α=0.01', linewidth=2)
ax.set_xlabel('episode')
ax.set_ylabel('RMS(V) over runs')
ax.set_title('TD vs MC：TD 早期下降快，MC 后期更准')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("观察：")
print("- TD 早期下降快（bootstrap 让信号快速传播）")
print("- MC 后期更接近真值（无偏）")
print("- 这正是 Ch04 讲的 bias-variance tradeoff")
"""),
    ]


def sol_ch05_dql() -> List[Cell]:
    """Ch05 练习参考答案：Double Q-learning。"""
    return [
        md("""# Ch05 练习参考答案：Double Q-learning
"""),

        code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / 'rlenvs').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed
from rlenvs import CliffWalk

set_seed(0)


def epsilon_greedy_action(Q, s, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[s]))


def double_q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    nS, nA = env.nS, env.nA
    Q1 = np.zeros((nS, nA))
    Q2 = np.zeros((nS, nA))
    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = epsilon_greedy_action(Q1 + Q2, s, epsilon)
            s_next, r, done, _ = env.step(a)
            ep_r += r
            if np.random.random() < 0.5:
                a_star = int(np.argmax(Q1[s_next])) if not done else 0
                td_target = r + (0 if done else gamma * Q2[s_next, a_star])
                Q1[s, a] += alpha * (td_target - Q1[s, a])
            else:
                a_star = int(np.argmax(Q2[s_next])) if not done else 0
                td_target = r + (0 if done else gamma * Q1[s_next, a_star])
                Q2[s, a] += alpha * (td_target - Q2[s, a])
            s = s_next
        rewards.append(ep_r)
    return Q1 + Q2, np.array(rewards)


def q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))
    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = epsilon_greedy_action(Q, s, epsilon)
            s_next, r, done, _ = env.step(a)
            ep_r += r
            td_target = r + (0 if done else gamma * np.max(Q[s_next]))
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next
        rewards.append(ep_r)
    return Q, np.array(rewards)


# 1) 在 OneStateTrap 上验证消除 maximization bias
class OneStateTrap:
    def __init__(self):
        self.nS = 1
        self.nA = 2
        self._rng = np.random.default_rng()
    def reset(self):
        return 0
    def step(self, a):
        r = self._rng.normal(0, 1) if a == 0 else self._rng.normal(-0.1, 1)
        return 0, r, True, {}


def run_trap_double(n_episodes=300, alpha=0.1, epsilon=0.1, n_seeds=200):
    freqs = np.zeros((n_seeds, n_episodes))
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = OneStateTrap()
        Q1 = np.zeros((1, 2))
        Q2 = np.zeros((1, 2))
        for ep in range(n_episodes):
            s = env.reset()
            done = False
            actions = []
            while not done:
                a = epsilon_greedy_action(Q1 + Q2, s, epsilon)
                actions.append(a)
                _, r, done, _ = env.step(a)
                if np.random.random() < 0.5:
                    Q1[s, a] += alpha * (r - Q1[s, a])
                else:
                    Q2[s, a] += alpha * (r - Q2[s, a])
            freqs[seed, ep] = np.mean(actions)
    return freqs


freqs_double = run_trap_double(n_episodes=300, alpha=0.1, epsilon=0.1, n_seeds=200)
print(f"Double Q-learning 在 trap 上选 a=1 的频率（最后 50 ep）: {freqs_double[:, -50:].mean():.3f}")
print(f"理论值 ε/2 = 0.05；纯 Q-learning 约 0.25")
print(f"→ Double Q-learning 几乎消除了 maximization bias！")

# 2) 在 CliffWalk 上对比
n_seeds, n_eps = 30, 500
dql_rw = np.zeros((n_seeds, n_eps))
ql_rw = np.zeros((n_seeds, n_eps))
for seed in range(n_seeds):
    env1 = CliffWalk(seed=seed)
    _, r1 = double_q_learning(env1, n_episodes=n_eps, alpha=0.5, epsilon=0.1)
    env2 = CliffWalk(seed=seed)
    _, r2 = q_learning(env2, n_episodes=n_eps, alpha=0.5, epsilon=0.1)
    dql_rw[seed] = r1
    ql_rw[seed] = r2

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np.convolve(ql_rw.mean(0), np.ones(20)/20, mode='valid'), label='Q-learning', linewidth=2)
ax.plot(np.convolve(dql_rw.mean(0), np.ones(20)/20, mode='valid'), label='Double Q-learning', linewidth=2)
ax.set_xlabel('episode (smoothed w=20)'); ax.set_ylabel('reward')
ax.set_title('CliffWalk: Double Q-learning 方差更小、更稳')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"\\n最终 50 episodes 平均奖励：")
print(f"  Q-learning:        {ql_rw[:, -50:].mean():.2f}")
print(f"  Double Q-learning: {dql_rw[:, -50:].mean():.2f}")
"""),
    ]


SOLUTION_BUILDERS = [
    ("ch01_thompson_sampling.ipynb", sol_ch01_thompson),
    ("ch02_bridge_gamma.ipynb", sol_ch02_bridge),
    ("ch03_modified_policy_iteration.ipynb", sol_ch03_mpi),
    ("ch04_mc_vs_td_gridworld.ipynb", sol_ch04_mc_td),
    ("ch05_double_q_learning.ipynb", sol_ch05_dql),
]


if __name__ == "__main__":
    main()

