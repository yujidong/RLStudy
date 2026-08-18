"""RLStudy 自研环境集合。

所有环境都遵循统一的 reset()/step() 接口（OpenAI Gym 风格），
方便后续 Ch06+ 在不同章节间切换。
"""
from .bandit import MultiArmedBandit
from .cart_pole_lite import CartPoleLite
from .click_world import ClickWorld
from .cliff_walk import CliffWalk
from .grid_world import (
    ACTIONS,
    N_ACTIONS,
    ACTION_NAMES,
    GridWorld,
    bridge_grid,
    cliff_world_4x12,
    small_grid_5x5,
)
from .random_walk import RandomWalk

# Ch10+ TinyGPT：纯 PyTorch 实现的 mini-GPT。延迟导入，避免 Phase 1 强依赖 torch。
# 注意：tiny_gpt 的名字**只有导入成功才**进 __all__——否则在没装 torch 的机器上
# `from rlenvs import *` 会因 __all__ 引用未定义名字而 NameError。
_TINY_GPT_NAMES = [
    "CharTokenizer",
    "PositionalEncoding",
    "CausalSelfAttention",
    "TransformerBlock",
    "TinyGPT",
    "build_tiny_gpt",
    "compute_loss",
    "make_lm_batch",
    "generate",
    "sft_loss",
]
try:
    from .tiny_gpt import (  # noqa: F401
        CharTokenizer,
        PositionalEncoding,
        CausalSelfAttention,
        TransformerBlock,
        TinyGPT,
        build_tiny_gpt,
        compute_loss,
        make_lm_batch,
        generate,
        sft_loss,
    )
except ImportError:
    pass  # 未安装 torch（Phase 1）

__all__ = [
    # bandit
    "MultiArmedBandit",
    # cart pole (Ch06+: 连续状态、离散动作)
    "CartPoleLite",
    # click world
    "ClickWorld",
    # cliff walk
    "CliffWalk",
    "cliff_world_4x12",
    # grid world
    "GridWorld",
    "small_grid_5x5",
    "bridge_grid",
    "ACTIONS",
    "ACTION_NAMES",
    "N_ACTIONS",
    # random walk
    "RandomWalk",
] + (_TINY_GPT_NAMES if "TinyGPT" in globals() else [])
