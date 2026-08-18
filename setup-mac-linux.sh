#!/usr/bin/env bash
# RLStudy 一键环境配置（macOS / Linux）
# 用法：在项目根目录执行  bash setup-mac-linux.sh
set -e
cd "$(dirname "$0")"

echo "============================================"
echo " RLStudy 一键环境配置 - macOS / Linux"
echo "============================================"
echo

# ---- 1/5 找一个能用的 Python ----
if ! command -v python3 >/dev/null 2>&1; then
    echo "[X] 没找到 python3。"
    echo "    macOS : brew install python   （没有 brew 就到 python.org 下载安装包）"
    echo "    Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "[X] python3 版本低于 3.10，跑不动本教材，请先升级（macOS: brew upgrade python; Ubuntu 见 PPA deadsnakes 或官网安装包）"
    exit 1
}
echo "找到 Python:"
python3 --version

# ---- 2/5 创建独立虚拟环境 .venv（不污染系统 Python）----
if [ -d .venv ]; then
    echo "[2/5] 复用已有的 .venv 虚拟环境"
else
    echo "[2/5] 创建虚拟环境 .venv ..."
    python3 -m venv .venv || {
        echo "[X] 创建虚拟环境失败——Ubuntu/Debian 常见原因是缺 venv 组件："
        echo "    sudo apt install python3-venv   然后重跑本脚本"
        exit 1
    }
fi
source .venv/bin/activate

# ---- 3/5 安装 CPU 版 PyTorch（约 200MB，比默认 CUDA 版小很多）----
echo "[3/5] 安装 PyTorch（CPU 版），请耐心等待下载..."
python -m pip install --upgrade pip -q
TORCH_OK=1
pip install torch --index-url https://download.pytorch.org/whl/cpu -q || TORCH_OK=0

# ---- 4/5 安装其余依赖（torch 失败时绝不从默认源补装 2GB+ 的 CUDA 版）----
if [ "$TORCH_OK" = "1" ]; then
    echo "[4/5] 安装其余依赖..."
    pip install -r requirements.txt -q
else
    echo "[!] PyTorch 下载失败——先装其余依赖继续，前五章纯 numpy 不受影响。"
    echo "    之后网络好时再单独执行: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    grep -vi '^torch' requirements.txt > "${TMPDIR:-/tmp}/rlstudy-req.txt"
    pip install "${TMPDIR:-/tmp}/rlstudy-req.txt" -q
fi

# ---- 5/5 环境体检 ----
echo "[5/5] 环境体检..."
python doctor.py

echo
echo "全部就绪！正在打开 JupyterLab（浏览器没弹出就手动访问 http://localhost:8888）"
jupyter lab notebooks/
