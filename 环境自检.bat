@echo off
setlocal
title Runtime-LivePortrait - environment check
cd /d "%~dp0"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo   .venv not found - run the one-click setup first.
    echo.
    pause
    exit /b 1
)
"%VENV_PY%" -u "%~dp0tools\check_env.py"
echo.
pause
