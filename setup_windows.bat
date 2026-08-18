@echo off
REM RLStudy one-click setup (Windows). Pure ASCII on purpose:
REM cmd.exe cannot reliably parse batch files containing non-ASCII text.
REM All real logic (and Chinese output) lives in setup.py, run below.
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :no_python

%PY% setup.py
pause
exit /b %errorlevel%

:no_python
echo [ERROR] Python not found.
echo.
echo   Fix 1: install Python 3.10+ from https://www.python.org/downloads/
echo          CHECK the box "Add Python to PATH" on the first screen,
echo          then run this script again.
echo   Fix 2: if already installed but still not found - open Windows Settings,
echo          go to Apps, Advanced app settings, App execution aliases,
echo          turn OFF python.exe and python3.exe, then run this script again.
echo.
pause
exit /b 1
