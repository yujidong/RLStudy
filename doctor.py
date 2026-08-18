"""RLStudy 环境体检脚本 —— 学生在本地运行，检查环境是否就绪。

用法（在项目根目录）::

    python doctor.py

会逐项检查 Python 版本 / 依赖 / PyTorch / Jupyter / 项目包 / 冒烟测试，
打印 ✓/✗ 报告，并写出 doctor_report.txt（有问题时直接把这个文件发给老师）。

全部通过退出码 0；任何一项失败退出码 1（便于脚本化判断）。
"""
from __future__ import annotations

import io
import os
import platform
import shutil
import sys
import time
from pathlib import Path

# Windows 控制台默认编码可能是 GBK：先切 UTF-8 代码页，再统一输出编码
if os.name == "nt":
    os.system("chcp 65001 >nul")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPORT_PATH = Path(__file__).parent / "doctor_report.txt"
_report: list[str] = []
_problems: list[str] = []


def log(line: str = "") -> None:
    print(line)
    _report.append(line)


def ok(msg: str) -> None:
    log(f"    [✓] {msg}")


def fail(msg: str, fix: str) -> None:
    log(f"    [✗] {msg}")
    log(f"        修复建议: {fix}")
    _problems.append(msg)


def check_import(name: str, fix: str | None = None) -> str | None:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "") or "（已安装）"
        ok(f"{name} {ver}")
        return ver
    except Exception as e:  # ImportError 或损坏安装的其它异常
        fix = fix or f"pip install {name}"
        fail(f"{name} 不可用（{type(e).__name__}: {e}）", fix)
        return None


# ---------------------------------------------------------------- 1. Python
log("=" * 62)
log("RLStudy 环境体检")
log("=" * 62)
log()
log("[1/7] Python 解释器")
log(f"    版本   : {sys.version.split()[0]}")
log(f"    实现   : {platform.python_implementation()} @ {sys.executable}")
log(f"    系统   : {platform.system()} {platform.release()} ({platform.machine()})")
if sys.version_info >= (3, 10):
    ok(f"Python {'.'.join(map(str, sys.version_info[:3]))} ≥ 3.10")
else:
    fail(f"Python {sys.version_info[:3]} 版本过低（需要 ≥ 3.10）",
         "到 python.org 下载 Python 3.10+，或用 setup_windows.bat / setup-mac-linux.sh 重建环境")

# 提醒：.venv 存在但当前不是它 → 学生忘了激活
if (Path(__file__).parent / ".venv").exists() and sys.base_prefix == sys.prefix:
    log()
    log("    [!] 注意：项目里已有 .venv 虚拟环境，但当前解释器不是它——")
    log(r"        Windows 先执行  .venv\Scripts\activate   再跑 python doctor.py")
    log("        macOS/Linux 先执行  source .venv/bin/activate")
    log("        （或者直接用 start_windows.bat / 一键脚本，它们会自动激活）")

# ---------------------------------------------------------------- 2. 核心依赖
log()
log("[2/7] 核心依赖（Phase 1 必需）")
for name in ["numpy", "scipy", "matplotlib", "ipywidgets", "tqdm", "pytest"]:
    check_import(name)

# ---------------------------------------------------------------- 3. PyTorch
log()
log("[3/7] PyTorch（Phase 2/3 必需）")
torch_ver = check_import("torch", fix="pip install torch --index-url https://download.pytorch.org/whl/cpu")
if torch_ver:
    import torch
    n = torch.get_num_threads()
    ok(f"CPU 可用（{n} 线程）")
    if torch.cuda.is_available():
        log(f"    GPU     : {torch.cuda.get_device_name(0)}（本教材用不到，纯加分项）")

# ---------------------------------------------------------------- 4. Jupyter
log()
log("[4/7] Jupyter（打开教材必备）")
for name, pretty in [("jupyterlab", "jupyterlab"), ("jupyter", "jupyter 核心"), ("nbconvert", "nbconvert")]:
    check_import(name, fix=f"pip install {name}")

# ---------------------------------------------------------------- 5. 项目包
log()
log("[5/7] 项目自带包（rlenvs / utils）")
root = Path(__file__).parent
try:
    sys.path.insert(0, str(root))
    import rlenvs
    import utils  # noqa: F401
    envs = [n for n in ("GridWorld", "CliffWalk", "MultiArmedBandit", "RandomWalk", "ClickWorld", "CartPoleLite")]
    missing = [n for n in envs if not hasattr(rlenvs, n)]
    ok(f"rlenvs 导入成功（{len(envs) - len(missing)}/{len(envs)} 个环境就位）")
    ok("utils 导入成功")
except Exception as e:
    fail(f"项目包导入失败（{type(e).__name__}: {e}）",
         "确认你在 RLStudy 项目根目录下运行本脚本，且没有移动 rlenvs/ utils/ 目录")

# ---------------------------------------------------------------- 6. 冒烟测试
log()
log("[6/7] 冒烟测试（真的跑一小段）")
try:
    from rlenvs import GridWorld, MultiArmedBandit
    env = GridWorld(shape=(4, 4), seed=0)
    env.reset()
    env.step(0)
    ok("GridWorld 创建 + step 正常")
    b = MultiArmedBandit(n_arms=5, seed=0)
    b.pull(0)
    ok("MultiArmedBandit 正常")
except Exception as e:
    fail(f"环境冒烟测试失败（{type(e).__name__}: {e}）",
         "把 doctor_report.txt 发给老师，附上你运行本脚本前做过什么")
if torch_ver:
    try:
        import torch
        y = torch.nn.Linear(4, 2)(torch.randn(3, 4))
        assert y.shape == (3, 2)
        ok("torch 前向计算正常")
    except Exception as e:
        fail(f"torch 前向失败（{e}）", "重装: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(io.BytesIO(), format="png")
    plt.close(fig)
    ok("matplotlib 离屏渲染正常")
except Exception as e:
    fail(f"matplotlib 渲染失败（{e}）", "pip install --force-reinstall matplotlib")

# ---------------------------------------------------------------- 7. 性能抽测
log()
log("[7/7] 性能抽测（确认机器跑得动）")
try:
    from rlenvs import GridWorld
    t0 = time.perf_counter()
    env = GridWorld(shape=(12, 12), seed=0)
    env.reset()
    for _ in range(2000):
        env.step(env.nA - 1)
    dt = time.perf_counter() - t0
    if dt < 5.0:
        ok(f"2000 步 GridWorld 耗时 {dt * 1000:.0f} ms（< 5s，完全够用）")
    else:
        ok(f"2000 步 GridWorld 耗时 {dt * 1000:.0f} ms（偏慢但可用；每章训练会久一点）")
except Exception as e:
    fail(f"性能抽测失败（{e}）", "发报告给老师")

# ---------------------------------------------------------------- 总结
log()
log("=" * 62)
if not _problems:
    log("全部通过 ✓ —— 环境就绪，直接运行:  jupyter lab notebooks/")
else:
    log(f"发现 {len(_problems)} 个问题 —— 按上面每条 ✗ 的「修复建议」处理后重跑本脚本")
    log(f"仍未解决就把 {REPORT_PATH.name} 发给老师")
log(f"报告已写入: {REPORT_PATH}")
log("=" * 62)

REPORT_PATH.write_text("\n".join(_report) + "\n", encoding="utf-8")
sys.exit(0 if not _problems else 1)
