#!/usr/bin/env bash
# RLStudy 一键环境配置（macOS / Linux）
# 用法：在项目根目录执行  bash setup-mac-linux.sh
set -e
cd "$(dirname "$0")"

echo "============================================"
echo " RLStudy 一键环境配置 - macOS / Linux"
echo "============================================"
echo

# ---- 1/5 找 Python ----
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "[X] 没找到 python3。请先安装 Python 3.10+（macOS: brew install python; Ubuntu: sudo apt install python3 python3-venv python3-pip）"
    exit 1
fi
$PY --version

# ---- 2/5 创建独立虚拟环境 .venv ----
if [ -d .venv ]; then
    echo "[2/5] 复用已有的 .venv 虚拟环境"
else
    echo "[2/5] 创建虚拟环境 .venv ..."
    $PY -m venv .venv
fi
source .venv/bin/activate

# ---- 3/5 安装 CPU 版 PyTorch（约 200MB，比默认 CUDA 版小很多）----
echo "[3/5] 安装 PyTorch（CPU 版），请耐心等待下载..."
python -m pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cpu -q \
    || echo "[!] PyTorch 安装失败——前五章用不到它，可稍后单独重试本步"

# ---- 4/5 安装其余依赖 ----
echo "[4/5] 安装其余依赖..."
pip install -r requirements.txt -q

# ---- 5/5 环境体检 ----
echo "[5/5] 环境体检..."
python doctor.py

echo
echo "全部就绪！正在打开 JupyterLab（浏览器没弹出就手动访问 http://localhost:8888）"
jupyter lab notebooks/
