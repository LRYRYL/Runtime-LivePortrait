@echo off
REM ============================================================================
REM  Runtime-LivePortrait  -  launch the web UI
REM
REM  Starts a local page (nothing leaves this machine) and opens it in the
REM  browser. Pick a portrait, press Start, watch the live result.
REM
REM  If .venv does not exist yet, run the one-click setup first.
REM
REM  Optional argument: port.   e.g.  Start_WebUI.bat 7861
REM ============================================================================
setlocal
cd /d "%~dp0"

set "PORT=%1"
if not defined PORT set "PORT=7860"

echo.
echo ============================================================
echo   Runtime-LivePortrait  -  web UI
echo ============================================================
echo.

REM ---- locate the virtual environment --------------------------------------
REM  Order: explicit override, then the venv this project's setup creates.
REM  No absolute path and no conda assumption -- the setup script makes .venv
REM  next to this file, so that is what we look for first.
set "PY="
if defined LPR_PYTHON if exist "%LPR_PYTHON%" set "PY=%LPR_PYTHON%"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"

if not defined PY (
    echo   [ERROR] no Python environment found for this project.
    echo.
    echo           Run the one-click setup first:
    echo             the install launcher in this folder
    echo.
    echo           Or, if you already have an environment with the
    echo           dependencies installed, point straight at it:
    echo             set LPR_PYTHON=D:\somewhere\python.exe
    echo.
    pause
    exit /b 1
)

echo   interpreter : %PY%
echo   port        : %PORT%
echo.
echo   A browser tab will open. The first Start loads the models
echo   and takes 5-7 seconds before any video appears.
echo.
echo   Keep this window open while you use the page.
echo   Press Ctrl+C here to stop the server.
echo ------------------------------------------------------------
echo.

"%PY%" -u app.py --port %PORT%
set "RC=%ERRORLEVEL%"

echo.
echo ------------------------------------------------------------
if "%RC%"=="0" (
    echo Server stopped.
) else (
    echo The server exited with code %RC%.
    echo.
    echo Common causes:
    echo   * the webcam is held by another app ^(WeChat / DingTalk / a browser tab^)
    echo   * the page said "no face found" - use a frontal portrait photo
    echo   * for a full diagnostic, run the check launcher in this folder
)
echo.
pause
endlocal
