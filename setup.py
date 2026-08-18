"""RLStudy 一键配置驱动脚本（三端共用：Windows / macOS / Linux）。

`setup_windows.bat` 和 `setup-mac-linux.sh` 都只负责找到 Python，然后运行本文件。
所有逻辑（虚拟环境、依赖安装、PyTorch CPU 版、环境体检、启动 JupyterLab）
集中在这里——为什么不用纯 .bat/.sh 写？因为 cmd.exe 解析含中文的批处理文件
会乱码断行（编码坑），而 Python 的输出编码完全可控。

用法::

    python setup.py               # 完整流程：建 venv -> 装依赖 -> 体检 -> 启动 JupyterLab
    python setup.py --no-launch   # 同上，但装完不启动（用于测试/自动化）
    python setup.py --doctor-only # 不装任何东西，只用当前 Python 跑一遍体检
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
IS_WINDOWS = os.name == "nt"


def utf8_console() -> None:
    """把 Windows 控制台切到 UTF-8 代码页，中文与 ✓/✗ 才能正常显示。"""
    if IS_WINDOWS:
        os.system("chcp 65001 >nul")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run(cmd: list, quiet: bool = False, **kw) -> int:
    """执行子进程并原样透传输出（学生能看到 pip 下载进度条）。"""
    cmd = [str(c) for c in cmd]
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, **kw).returncode


def main() -> int:
    utf8_console()
    print("=" * 52)
    print(" RLStudy 一键环境配置")
    print("=" * 52)

    # ---- 1/5 版本检查 ----
    print(f"\n[1/5] Python: {sys.version.split()[0]}  @ {sys.executable}")
    if sys.version_info < (3, 10):
        print("[X] Python 版本低于 3.10，跑不动本教材。")
        print("    请到 https://www.python.org/downloads/ 安装最新版（Windows 记得勾选")
        print("    Add Python to PATH），装完重新运行一键脚本。")
        return 1

    doctor_only = "--doctor-only" in sys.argv
    if doctor_only:
        return subprocess.run([sys.executable, ROOT / "doctor.py"]).returncode

    # ---- 2/5 虚拟环境 ----
    venv_py = ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if venv_py.exists():
        print("\n[2/5] 复用已有的 .venv 虚拟环境（不污染系统 Python）")
    else:
        print("\n[2/5] 创建虚拟环境 .venv（不污染系统 Python）...")
        if run([sys.executable, "-m", "venv", ROOT / ".venv"]) != 0 or not venv_py.exists():
            print("[X] 创建虚拟环境失败。常见原因与解法：")
            print("    Ubuntu/Debian: sudo apt install python3-venv 后重跑")
            print("    Windows: 删除 .venv 文件夹后重跑；仍失败把报错截图发老师")
            return 1

    # ---- 3/5 CPU 版 PyTorch（约 200MB，比默认 CUDA 版小很多）----
    print("\n[3/5] 安装 PyTorch（CPU 版，约 200MB，请耐心等待下载）...")
    torch_ok = run([venv_py, "-m", "pip", "install", "torch",
                    "--index-url", TORCH_CPU_INDEX]) == 0

    # ---- 4/5 其余依赖（torch 失败时绝不从默认源误装 2GB+ 的 CUDA 版）----
    print("\n[4/5] 安装其余依赖...")
    if torch_ok:
        rc = run([venv_py, "-m", "pip", "install", "-r", ROOT / "requirements.txt", "-q"])
    else:
        print("[!] PyTorch 下载失败——先装其余依赖继续，前五章纯 numpy 不受影响。")
        print("    网络好时再单独执行：")
        print(f"    pip install torch --index-url {TORCH_CPU_INDEX}")
        lines = [l for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().lower().startswith("torch")
                 and not l.strip().startswith("#")]
        tmp = Path(tempfile.gettempdir()) / "rlstudy-req.txt"
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rc = run([venv_py, "-m", "pip", "install", "-r", tmp, "-q"])
    if rc != 0:
        print("[X] 依赖安装失败。可先重跑本脚本重试；网络差可加国内镜像：")
        print("    -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return rc

    # ---- 5/5 环境体检 ----
    print("\n[5/5] 环境体检...")
    drc = run([venv_py, ROOT / "doctor.py"])
    if drc != 0:
        print("\n有项目未通过：按上面每条 ✗ 的修复建议处理后重跑本脚本，")
        print(f"或把 {ROOT / 'doctor_report.txt'} 发给老师。")
        return drc

    # ---- 启动 ----
    if "--no-launch" in sys.argv:
        print("\n全部就绪 ✓（--no-launch：跳过启动 JupyterLab）")
        return 0
    print("\n全部就绪！正在打开 JupyterLab——浏览器没弹出就手动访问 http://localhost:8888")
    print("（关闭本窗口或 Ctrl+C 即可停止；下次学习用 start_windows.bat / source .venv/bin/activate）")
    os.chdir(ROOT)
    return run([venv_py, "-m", "jupyter", "lab", "notebooks/"])


if __name__ == "__main__":
    sys.exit(main())
