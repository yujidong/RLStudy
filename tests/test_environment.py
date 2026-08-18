"""环境冒烟测试：验证 Phase 1 所有依赖都能导入。"""
import sys


def test_python_version():
    assert sys.version_info >= (3, 10), "需要 Python >= 3.10"


def test_numpy():
    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6


def test_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境也能跑
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close(fig)


def test_ipywidgets():
    import ipywidgets
    assert ipywidgets.__version__


def test_scipy():
    import scipy
    assert scipy.__version__


def test_rlenvs_import():
    import rlenvs
    assert hasattr(rlenvs, "GridWorld")
    assert hasattr(rlenvs, "MultiArmedBandit")
    assert hasattr(rlenvs, "RandomWalk")
    assert hasattr(rlenvs, "CliffWalk")
    assert hasattr(rlenvs, "ClickWorld")


def test_utils_import():
    import utils
    for fn in ["set_seed", "plot_training_curve", "make_interactive"]:
        assert hasattr(utils, fn), f"utils 缺少 {fn}"


def test_torch_optional():
    """torch 是可选的（Phase 2 起）。"""
    try:
        import torch
    except ImportError:
        return  # 不强求
    # 若装了，验证基本可用
    x = torch.tensor([1.0, 2.0])
    assert float(x.sum()) == 3.0
