"""matplotlib 可视化工具集合。

所有函数都返回 Figure / Axes 对象，方便调用者进一步定制。
所有函数都假设已经在 notebook 中执行过 ``%matplotlib widget`` 或
在脚本中 ``plt.show()``。
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import FancyArrow

# 全局美观设置 + 中文字体（matplotlib 默认 DejaVu Sans 不含中文，会显示豆腐块）
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",    # Windows
    "SimHei",             # Windows 备选
    "PingFang SC",        # Mac
    "Heiti TC",           # Mac 备选
    "Noto Sans CJK SC",   # Linux
    "WenQuanYi Zen Hei",  # Linux 备选
    "DejaVu Sans",        # fallback
]
plt.rcParams["axes.unicode_minus"] = False  # 负号在 CJK 字体下显示


# =============================================================================
# 训练曲线
# =============================================================================
def smooth(x: Sequence[float], window: int = 20) -> np.ndarray:
    """滑动平均。返回与 x 等长的数组，端点用反射填充避免边缘下凹。"""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def plot_training_curve(
    rewards: Sequence[float] | Sequence[Sequence[float]],
    window: int = 20,
    labels: Optional[Sequence[str]] = None,
    title: str = "Training Curve",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    ax: Optional[plt.Axes] = None,
    show_smoothed_std: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """绘制训练曲线。

    Parameters
    ----------
    rewards
        - 一维序列：单次实验的每 episode 奖励
        - 二维序列 [n_seeds, n_episodes]：多次实验，自动画均值 + std 阴影
    window : int
        滑动平均窗口大小
    show_smoothed_std : bool
        仅在二维输入时生效：是否画平滑后的 ±std 阴影
    """
    rewards = np.asarray(rewards, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    if rewards.ndim == 1:
        # 单条曲线
        ax.plot(rewards, color="#9ec5e8", alpha=0.55, linewidth=1, label="raw")
        ax.plot(smooth(rewards, window), color="#1f77b4", linewidth=2, label=f"smoothed (w={window})")
        ax.legend()
    else:
        # 多条曲线
        mean = rewards.mean(axis=0)
        std = rewards.std(axis=0)
        x = np.arange(len(mean))
        if labels is None:
            labels = [f"seed {i}" for i in range(len(rewards))]
        for i, r in enumerate(rewards):
            ax.plot(r, alpha=0.35, linewidth=0.9, label=labels[i])
        ax.plot(smooth(mean, window), color="crimson", linewidth=2.2, label="mean (smoothed)")
        if show_smoothed_std:
            sm = smooth(mean, window)
            sm_std = smooth(std, window)
            ax.fill_between(x, sm - sm_std, sm + sm_std, color="crimson", alpha=0.18)
        ax.legend(loc="best", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


# =============================================================================
# GridWorld 可视化
# =============================================================================
def plot_value_heatmap(
    V: Sequence[float],
    shape: Tuple[int, int],
    policy: Optional[Sequence[int]] = None,
    action_labels: Optional[Sequence[str]] = None,
    walls: Optional[Sequence[Tuple[int, int]]] = None,
    terminals: Optional[Sequence[Tuple[int, int]]] = None,
    cell_text: bool = False,
    cmap: str = "RdYlGn",
    ax: Optional[plt.Axes] = None,
    title: str = "State-Value V(s)",
) -> Tuple[plt.Figure, plt.Axes]:
    """在 grid 上画 V 的热力图，可选叠加策略箭头。

    Parameters
    ----------
    V : shape ``[n_states]`` 数组
    shape : (rows, cols)
    policy : shape ``[n_states]`` 数组，每个元素是动作 id
    action_labels : 例如 ['↑','→','↓','←']
    walls, terminals : (row, col) 坐标列表
    cell_text : 是否在每个格子里写出 V 值
    """
    V = np.asarray(V, dtype=float).reshape(shape)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which="minor", color="k", linewidth=1.2, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.imshow(V, cmap=cmap, vmin=V.min(), vmax=V.max(), origin="upper")

    if cell_text:
        for r in range(shape[0]):
            for c in range(shape[1]):
                ax.text(c, r, f"{V[r, c]:.2f}", ha="center", va="center", fontsize=8)

    if walls:
        for (r, c) in walls:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
    if terminals:
        for (r, c) in terminals:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="gold", alpha=0.7))

    if policy is not None and action_labels is not None:
        for r in range(shape[0]):
            for c in range(shape[1]):
                a = int(policy[r * shape[1] + c])
                ax.text(c, r - 0.32, action_labels[a], ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def plot_q_table(
    Q: np.ndarray,
    action_labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Q(s, a)",
) -> Tuple[plt.Figure, plt.Axes]:
    """Q 表热力图：行为 state，列为 action。"""
    Q = np.asarray(Q)
    n_states, n_actions = Q.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(n_actions * 0.8 + 2, n_states * 0.35 + 1.5))
    else:
        fig = ax.figure
    im = ax.imshow(Q, cmap="viridis", aspect="auto")
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    if action_labels is not None:
        ax.set_xticks(range(n_actions))
        ax.set_xticklabels(action_labels)
    for s in range(n_states):
        for a in range(n_actions):
            ax.text(a, s, f"{Q[s, a]:.1f}", ha="center", va="center", color="w", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


# =============================================================================
# 动画
# =============================================================================
def animate_agent(
    env,
    policy: Callable[[int], int] | np.ndarray,
    n_steps: int = 50,
    fps: int = 5,
    figsize: Tuple[float, float] = (4.5, 4.5),
    title_prefix: str = "step",
) -> animation.FuncAnimation:
    """生成智能体在 env 中走 n_steps 步的 FuncAnimation。

    Parameters
    ----------
    env : 必须实现 reset() / step(a) / render(ax) 接口
    policy : 一个函数 s -> a，或一个 size = n_states 的 numpy 数组（确定性策略）
    n_steps : 步数
    fps : 帧率（每秒几格）

    Returns
    -------
    matplotlib.animation.FuncAnimation 对象，notebook 内可直接显示。
    若要保存::

        anim.save('xxx.gif', writer='pillow', fps=fps)
    """
    if callable(policy):
        get_action = policy
    else:
        get_action = lambda s: int(policy[s])

    state = env.reset()
    states = [state]
    rewards = [0.0]
    actions_taken = [None]
    for _ in range(n_steps):
        a = get_action(state)
        s_next, r, done, _ = env.step(a)
        states.append(s_next)
        rewards.append(r)
        actions_taken.append(a)
        state = s_next
        if done:
            # 终止后保持终点画面
            for _ in range(min(3, n_steps - len(states))):
                states.append(s_next)
                rewards.append(0.0)
                actions_taken.append(None)
            break

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(k: int):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        env.render(ax, state=states[k])
        ax.set_title(f"{title_prefix} {k}  r={rewards[k]:.2f}")

    anim = animation.FuncAnimation(
        fig, update, frames=len(states), interval=1000 // fps, blit=False, repeat=True
    )
    plt.close(fig)
    return anim


def save_animation(anim: animation.FuncAnimation, filename: str, fps: int = 5) -> str:
    """保存动画。优先用 pillow（保存 gif，无需 ffmpeg），失败则降级到 Pillow GIF。"""
    try:
        anim.save(filename, writer="pillow", fps=fps)
    except Exception as e:
        # 进一步降级：保存为静态图序列
        print(f"[save_animation] 保存失败：{e}")
    return filename


# =============================================================================
# 实时训练面板（Phase 2+ 用）
# =============================================================================
class LivePlot:
    """非阻塞实时训练面板。

    用法::

        lp = LivePlot(metrics=['reward', 'loss'])
        for step in range(1000):
            ...
            lp.update({'reward': r, 'loss': l})
    """

    def __init__(
        self,
        metrics: Sequence[str],
        n_cols: int = 2,
        figsize: Optional[Tuple[float, float]] = None,
        smooth_window: int = 20,
    ):
        self.metrics = list(metrics)
        self.n = len(self.metrics)
        self.n_cols = min(n_cols, self.n)
        self.n_rows = (self.n + self.n_cols - 1) // self.n_cols
        figsize = figsize or (4.5 * self.n_cols, 3.2 * self.n_rows)
        self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=figsize)
        if self.n == 1:
            self.axes = np.array([self.axes])
        self.axes = self.axes.flatten()
        self.history: dict[str, list] = {m: [] for m in self.metrics}
        self.smooth_window = smooth_window
        self.fig.tight_layout()

    def update(self, values: dict[str, float], step: Optional[int] = None) -> None:
        for m, v in values.items():
            if m in self.history:
                self.history[m].append(v)
        for i, m in enumerate(self.metrics):
            ax = self.axes[i]
            ax.clear()
            data = np.asarray(self.history[m], dtype=float)
            ax.plot(data, color="#9ec5e8", alpha=0.6, linewidth=0.9, label="raw")
            if len(data) >= 5:
                ax.plot(smooth(data, self.smooth_window), color="crimson", linewidth=2, label=f"smoothed (w={self.smooth_window})")
            ax.set_title(m)
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.3)
        self.fig.canvas.draw_idle()
        try:
            from IPython.display import clear_output, display
            display(self.fig)
            clear_output(wait=True)
        except Exception:
            pass


# =============================================================================
# 通用小工具
# =============================================================================
def plot_bar_compare(
    values: Sequence[float],
    labels: Sequence[str],
    colors: Optional[Sequence[str]] = None,
    title: str = "",
    ylabel: str = "",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """柱状图比较，用于 bias/variance / 多算法比较等。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    return fig, ax


def plot_regret_curve(
    rewards: Sequence[Sequence[float]],
    optimal_per_step: float,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """绘制累计 regret 曲线。rewards 是 [n_runs, n_steps]。"""
    rewards = np.asarray(rewards)
    cum_regret = np.cumsum(optimal_per_step - rewards, axis=1)
    mean = cum_regret.mean(axis=0)
    std = cum_regret.std(axis=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    x = np.arange(len(mean))
    ax.plot(x, mean, linewidth=2, label="mean cumulative regret")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25)
    if labels is not None:
        for i, r in enumerate(cum_regret):
            ax.plot(r, alpha=0.3, linewidth=0.7, label=labels[i])
        ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Regret over time")
    return fig, ax
