@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
    echo 还没有配置环境 - 请先双击 setup_windows.bat
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
echo 正在打开 JupyterLab - 浏览器没弹出就手动访问 http://localhost:8888
jupyter lab notebooks/
pause
