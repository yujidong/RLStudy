@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  RLStudy 一键环境配置 - Windows
echo ============================================
echo.

REM ---- 1/5 找 Python ----
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
    echo [X] 没找到 Python。请到 https://www.python.org/downloads/ 安装 3.10 或更新版本，
    echo     安装第一屏务必勾选 "Add Python to PATH"，装好后重新双击本脚本。
    pause
    exit /b 1
)
%PY% --version

REM ---- 2/5 创建独立虚拟环境 .venv ----
if exist .venv (
    echo [2/5] 复用已有的 .venv 虚拟环境
) else (
    echo [2/5] 创建虚拟环境 .venv ...
    %PY% -m venv .venv || (echo [X] 创建虚拟环境失败 & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"

REM ---- 3/5 安装 CPU 版 PyTorch - 约 200MB，比默认 CUDA 版小很多 ----
echo [3/5] 安装 PyTorch - CPU 版，请耐心等待下载...
python -m pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
if errorlevel 1 echo [!] PyTorch 安装失败 - 前五章用不到它，可稍后单独重试本步

REM ---- 4/5 安装其余依赖 ----
echo [4/5] 安装其余依赖...
pip install -r requirements.txt -q

REM ---- 5/5 环境体检 ----
echo [5/5] 环境体检...
python doctor.py
if errorlevel 1 (
    echo.
    echo 有项目未通过：按上面每条 X 的修复建议处理后重跑本脚本，
    echo 或把 doctor_report.txt 发给老师。
    pause
    exit /b 1
)

echo.
echo 全部就绪！正在打开 JupyterLab - 浏览器没弹出就手动访问 http://localhost:8888
jupyter lab notebooks/
pause
