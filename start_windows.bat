@echo off
REM RLStudy daily launcher (Windows). Pure ASCII on purpose.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Environment not configured yet. Run setup_windows.bat first.
    pause
    exit /b 1
)

echo Starting JupyterLab ... if no browser opens, visit http://localhost:8888
".venv\Scripts\python.exe" -m jupyter lab notebooks/
pause
