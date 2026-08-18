@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  RLStudy 一键环境配置 - Windows
echo ============================================
echo.

REM ---- 1/5 找一个能用的 Python ----
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :no_python

REM 防两类坑：Microsoft Store 的假 python 别名（实际打不开）、能跑但版本低于 3.10
%PY% -c "print('ok')" >nul 2>nul
if errorlevel 1 goto :no_python
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto :bad_version
echo 找到 Python:
%PY% --version
goto :venv

:no_python
echo [X] 没找到可用的 Python。两种常见原因：
echo     1. 还没安装 - 到 https://www.python.org/downloads/ 装 3.10 或更新版本，
echo        第一屏务必勾选 "Add Python to PATH"，装完重开本脚本
echo     2. 装过但被 Microsoft Store 假别名挡住 - 打开「设置 - 应用 - 高级应用设置
echo        - 应用执行别名」，把 python.exe / python3.exe 两项关掉，重开本脚本
pause
exit /b 1

:bad_version
echo [X] 找到的 Python 版本低于 3.10，跑不动本教材。
echo     到 https://www.python.org/downloads/ 安装最新版 - 勾选 "Add Python to PATH" -
echo     装完重开本脚本 - 它会自动优先使用新版本。
pause
exit /b 1

:venv
REM ---- 2/5 创建独立虚拟环境 .venv（不污染系统 Python）----
if exist .venv (
    echo [2/5] 复用已有的 .venv 虚拟环境
) else (
    echo [2/5] 创建虚拟环境 .venv ...
)
if not exist .venv %PY% -m venv .venv
if not exist ".venv\Scripts\activate.bat" (
    echo [X] 创建虚拟环境失败。可删除 .venv 文件夹后重试；仍失败就把报错截图发老师。
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

REM ---- 3/5 安装 CPU 版 PyTorch（约 200MB，比默认 CUDA 版小很多）----
echo [3/5] 安装 PyTorch - CPU 版，请耐心等待下载...
python -m pip install --upgrade pip -q
set TORCH_OK=1
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
if errorlevel 1 set TORCH_OK=0

REM ---- 4/5 安装其余依赖（torch 失败时绝不从默认源补装 2GB+ 的 CUDA 版）----
if "%TORCH_OK%"=="1" (
    echo [4/5] 安装其余依赖...
    pip install -r requirements.txt -q
) else (
    echo [!] PyTorch 下载失败 - 先装其余依赖继续，前五章纯 numpy 不受影响。
    echo     之后网络好时再单独执行：pip install torch --index-url https://download.pytorch.org/whl/cpu
    findstr /V /I /B "torch" requirements.txt > "%TEMP%\rlstudy-req.txt"
    pip install "%TEMP%\rlstudy-req.txt" -q
)

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
