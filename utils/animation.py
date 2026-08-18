"""FuncAnimation 辅助工具 + GIF/MP4 导出。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from matplotlib import animation


def make_frame_getter(env, policy, n_steps: int):
    """返回 ``(get_frames, run)`` 两个函数。

    - ``run()``：执行一次完整 episode，记录每一步的 ``state``、``action``、``reward``
    - ``get_frames()``：返回所有缓存的状态列表，供 FuncAnimation 使用

    用途：当策略和环境本身不变时，先跑一次再播放，比每帧重跑更稳定。
    """
    state = env.reset()
    frames = [state]
    actions: List[int] = []
    rewards: List[float] = [0.0]
    s = state
    for _ in range(n_steps):
        a = int(policy(s))
        s_next, r, done, _ = env.step(a)
        frames.append(s_next)
        actions.append(a)
        rewards.append(r)
        s = s_next
        if done:
            for _ in range(min(3, n_steps - len(frames) + 1)):
                frames.append(s_next)
                rewards.append(0.0)
                actions.append(-1)
            break

    return frames, actions, rewards


def save_gif(anim: animation.FuncAnimation, path: str | Path, fps: int = 5) -> Path:
    """保存 FuncAnimation 为 .gif。优先用 pillow（纯 Python，无 ffmpeg 依赖）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer="pillow", fps=fps)
    return path


def save_mp4(anim: animation.FuncAnimation, path: str | Path, fps: int = 15) -> Path:
    """保存为 .mp4，需要 ffmpeg。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(path), writer="ffmpeg", fps=fps)
    except Exception as e:
        print(f"[save_mp4] 保存失败（缺少 ffmpeg？）：{e}")
        # 降级到 gif
        gif_path = path.with_suffix(".gif")
        save_gif(anim, gif_path, fps=fps)
        return gif_path
    return path
