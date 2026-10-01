@echo off
setlocal
title Runtime-LivePortrait - one-click setup
cd /d "%~dp0"

echo.
echo ============================================================
echo   Runtime-LivePortrait  -  one-click setup
echo ============================================================
echo.
echo   This will:
echo     1. find Python 3.10, or install it for you (per-user, no admin)
echo     2. create a virtual environment in .venv
echo     3. download the Python dependencies once (about 3 GB)
echo     4. install them from the local copy, offline
echo     5. download the model weights (about 667 MB)
echo.
echo   Nothing is installed system-wide, and no other Python on
echo   this PC is touched. Roughly 9 GB of disk is needed in total.
echo.
pause

set "PS=powershell"
where pwsh >nul 2>&1 && set "PS=pwsh"

%PS% -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup.ps1"
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
if "%RC%"=="0" (
    echo   Setup finished. Start the app with Start_WebUI.bat
) else (
    echo   Setup did NOT finish - exit code %RC%
    echo   See the messages above, or docs\DEPLOY.md
)
echo ============================================================
echo.
pause
exit /b %RC%
