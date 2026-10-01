"""Build the PORTABLE variant of Runtime-LivePortrait.

A portable package is the project plus a complete Python runtime, so that nothing at all
has to be installed on the target machine -- no registry writes, no Start-menu entries,
no PATH changes, and deleting the folder leaves no trace.

Why ship a whole runtime rather than the tiny embeddable zip
-----------------------------------------------------------
Measured, not assumed:

    python-3.10.11-embed-amd64.zip     8.23 MB   NO venv, NO ensurepip
    python-build-standalone                      download blocked on this network
    copy of the working runtime         6.2 GB   fully relocatable (verified)

The embeddable zip is far smaller, but its python310._pth mechanism overrides sys.path,
which stops a venv from seeing its own site-packages. Rebuilding it against another
distribution's stdlib then broke _sqlite3 / _ssl / _ctypes. Copying a runtime that is
already known to work avoids all of that, and relocation was verified: from a copied
folder torch reported CUDA available and the engine measured the same frame rate.

Usage
-----
    python build_portable.py --from-runtime <path to a python env>
    python build_portable.py --from-runtime %USERPROFILE%\\.conda\\envs\\lpv --out D:\\pkg
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# files that belong to development, not to a shipped package
PROJECT_SKIP_FILES = [
    ".gitignore", "Start_WebUI.bat", "一键安装.bat", "环境自检.bat",
    "python-3.10.11-amd64.exe",
]
PROJECT_SKIP_DIRS = [".git", ".venv", ".venv-wheels", "__pycache__", "output", ".cache"]
RUNTIME_SKIP_DIRS = ["__pycache__", ".git"]

# the project files that MUST exist afterwards; without these the package is useless
REQUIRED_PROJECT = [
    "app.py",
    "fastportrait/__init__.py",
    "fastportrait/realtime.py",
    "tools/check_env.py",
    "README.md",
]


def is_python_env(p: Path) -> bool:
    return (p / "python.exe").is_file() or (p / "Scripts" / "python.exe").is_file()


def python_in(p: Path) -> Path:
    for c in (p / "Scripts" / "python.exe", p / "python.exe"):
        if c.is_file():
            return c
    raise SystemExit(f"no python.exe under {p}")


def robocopy(src: Path, dst: Path, skip_dirs: list[str], skip_files: list[str] | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["robocopy", str(src), str(dst), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"]
    if skip_dirs:
        cmd += ["/XD"] + skip_dirs
    if skip_files:
        cmd += ["/XF"] + skip_files
    p = subprocess.run(cmd, capture_output=True)
    # robocopy: 0-7 are success codes, 8+ are real failures
    return p.returncode


def assert_not_a_venv(root: Path) -> None:
    """Refuse to package a virtual environment.

    A venv is not relocatable: `pyvenv.cfg` records the absolute path of the
    interpreter that created it, e.g. `home = C:\\Users\\someone\\.conda\\envs\\lpv`.
    Copying that folder to another PC makes the launcher fail with

        No Python at '"C:\\Users\\someone\\.conda\\envs\\lpv\\python.exe'

    because the recorded path does not exist there. A portable runtime must therefore
    be a full directory copy, which has python.exe and DLLs\\ but no pyvenv.cfg.
    """
    cfg = root / "pyvenv.cfg"
    if cfg.is_file():
        home = ""
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("home"):
                home = line.split("=", 1)[-1].strip()
        raise SystemExit(
            f"refusing to package a virtual environment: {root}\n"
            f"  pyvenv.cfg says home = {home}\n"
            "  a venv only works on the machine that created it.\n"
            "  copy a full python environment instead (one with python.exe and DLLs\\)."
        )
    if not (root / "python.exe").is_file():
        raise SystemExit(
            f"{root} has no top-level python.exe, so it is not a self-contained runtime"
        )
    if not (root / "DLLs").is_dir():
        raise SystemExit(
            f"{root} has no DLLs directory, so it is not a self-contained runtime"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-runtime", required=True,
                    help="a python environment to copy (venv, conda env, or python dir)")
    ap.add_argument("--project", default=str(HERE.parent),
                    help="project root to copy (default: the folder containing this script)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-old", action="store_true",
                    help="do not delete an existing output folder first")
    args = ap.parse_args()

    runtime = Path(args.from_runtime)
    project = Path(args.project)
    out = Path(args.out)

    if not is_python_env(runtime):
        raise SystemExit(f"not a python environment: {runtime}")

    # Catch a venv here, before 6 GB gets copied and shipped.
    assert_not_a_venv(runtime if (runtime / "python.exe").is_file()
                      else runtime.parent)
    if not (project / "app.py").is_file():
        raise SystemExit(f"not the project root (no app.py): {project}")

    py = python_in(runtime)
    probe = ("import sys, torch, cv2, numpy, onnxruntime;"
             "print(sys.version.split()[0], torch.__version__,"
             "torch.cuda.is_available(), cv2.__version__, onnxruntime.__version__)")
    r = subprocess.run([str(py), "-c", probe], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"source runtime is not usable:\n{r.stderr[-600:]}")
    print(f"  source runtime : {runtime}")
    print(f"  source check   : {r.stdout.strip()}")
    print(f"  project        : {project}")
    print(f"  output         : {out}")

    if out.exists() and not args.keep_old:
        print("\n  removing the previous output ...")
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) runtime ---------------------------------------------------------
    print(f"\n=== 1) runtime -> {out / '_runtime'} ===")
    t0 = time.time()
    rc = robocopy(runtime, out / "_runtime", RUNTIME_SKIP_DIRS)
    size = sum(f.stat().st_size for f in (out / "_runtime").rglob("*") if f.is_file())
    print(f"    {size/1024**3:.2f} GB in {time.time()-t0:.0f}s (rc={rc})")
    if rc >= 8:
        raise SystemExit(f"robocopy failed with {rc}")

    # ---- 2) project ---------------------------------------------------------
    print(f"\n=== 2) project -> {out / 'LivePortrait-Runtime'} ===")
    proj_dst = out / "LivePortrait-Runtime"
    rc = robocopy(project, proj_dst, PROJECT_SKIP_DIRS, PROJECT_SKIP_FILES)
    print(f"    rc={rc}")

    # The earlier version of this script trusted robocopy and then only checked that
    # python.exe ran, so a silently failed project copy still reported success -- the
    # package launched and immediately died on a missing app.py. Verify instead.
    missing = [f for f in REQUIRED_PROJECT if not (proj_dst / f).is_file()]
    if missing:
        print("    FAILED: the project copy is incomplete, missing:")
        for m in missing:
            print(f"      {m}")
        return 1
    n_files = sum(1 for f in proj_dst.rglob("*") if f.is_file())
    print(f"    {n_files} files, app.py present")

    # weights must be supplied separately (they are not in the repository)
    wdir = proj_dst / "LivePortrait" / "pretrained_weights"
    n_w = len([p for p in wdir.rglob("*") if p.suffix in (".pth", ".onnx")]) if wdir.is_dir() else 0
    print(f"    weights: {n_w}/8")
    if n_w < 8:
        print("    NOTE: copy the weights in, or run tools/download_weights.py once")

    # ---- 3) launchers ------------------------------------------------------
    print("\n=== 3) launchers ===")
    START = r"""@echo off
setlocal
title Runtime-LivePortrait (portable)
cd /d "%~dp0"

set "PY=%~dp0_runtime\python.exe"
if not exist "%PY%" (
    echo   [ERROR] _runtime\python.exe is missing - the package is incomplete.
    echo.
    pause
    exit /b 1
)
set "APP=%~dp0LivePortrait-Runtime\app.py"
if not exist "%APP%" (
    echo   [ERROR] LivePortrait-Runtime\app.py is missing - the package is incomplete.
    echo.
    pause
    exit /b 1
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=7860"

REM PYTHONHOME/PYTHONPATH from the host can break a relocated runtime
set "PYTHONHOME="
set "PYTHONPATH="

pushd "%~dp0LivePortrait-Runtime"
"%PY%" -u "%APP%" --port %PORT%
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" (
    echo.
    echo   The server exited with code %RC%
    pause
)
exit /b %RC%
"""
    DIAG = r"""@echo off
setlocal
title Runtime-LivePortrait - check (portable)
cd /d "%~dp0"

set "PY=%~dp0_runtime\python.exe"
set "CHK=%~dp0LivePortrait-Runtime\tools\check_env.py"
if not exist "%PY%" (
    echo   [ERROR] _runtime\python.exe is missing - the package is incomplete.
    pause
    exit /b 1
)
if not exist "%CHK%" (
    echo   [ERROR] LivePortrait-Runtime\tools\check_env.py is missing.
    pause
    exit /b 1
)

set "PYTHONHOME="
set "PYTHONPATH="

pushd "%~dp0LivePortrait-Runtime"
"%PY%" -u "%CHK%"
popd
echo.
pause
"""
    for name, text in (("启动-便携版.bat", START), ("环境自检-便携版.bat", DIAG)):
        (out / name).write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))
        print(f"    {name}")

    # ---- 4) verify ---------------------------------------------------------
    print("\n=== 4) verification ===")
    ok = True

    py = out / "_runtime" / "python.exe"
    r = subprocess.run([str(py), "-c", probe], cwd=str(proj_dst),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"  runtime relocates and imports : {'OK' if r.returncode == 0 else 'FAILED'}")
    if r.returncode == 0:
        print(f"    {r.stdout.strip()}")
    else:
        print(f"    {r.stderr.strip()[-300:]}")
        ok = False

    # app.py must actually load from the new location -- this is what the first version
    # of this script failed to check, so a broken package still reported success.
    # Only assert on attributes that every version of app.py has; the page markup
    # constants differ between revisions, and a missing one is not a broken package.
    app = proj_dst / "app.py"
    loader = (
        "import importlib.util as u;"
        f"spec = u.spec_from_file_location('lpr_app', r'{app}');"
        "m = u.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        "assert hasattr(m, 'Handler'), 'app.py has no Handler class';"
        "print('app.py loaded (Handler ok, PAGE %d bytes)' % len(getattr(m, 'PAGE', '')))"
    )
    r = subprocess.run([str(py), "-c", loader], cwd=str(proj_dst),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"  app.py loads from new location : {'OK' if r.returncode == 0 else 'FAILED'}")
    if r.returncode == 0:
        print(f"    {r.stdout.strip()}")
    else:
        print(f"    {r.stderr.strip()[-400:]}")
        ok = False

    # every referenced file must exist, using only relative paths
    print("  referenced files:")
    for rel in ("_runtime/python.exe", "LivePortrait-Runtime/app.py",
                "LivePortrait-Runtime/tools/check_env.py",
                "LivePortrait-Runtime/portraits/sample.jpg"):
        exists = (out / rel).exists()
        if not exists:
            ok = False
        print(f"    {'ok  ' if exists else 'MISS'} {rel}")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n  package size : {total/1024**3:.2f} GB")
    print("  RESULT:", "PORTABLE PACKAGE READY" if ok else "PROBLEMS -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
