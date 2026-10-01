"""Host-side client for the frame worker.

This is the piece Deep-Live-Cam will use: spawn `tools/frame_worker.py` with the
`lpv` interpreter, then hand it frames and get animated frames back. It is written
and tested standalone first so the integration is not debugged through a GUI.

The class deliberately mirrors the shape of a Deep-Live-Cam frame processor's
needs: `start()`, `process(frame_bgr) -> frame_bgr`, `stop()`, plus a `ready`
flag.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "tools" / "frame_worker.py"
MAGIC_STOP = 0xFFFFFFFF


def lpv_python() -> Path | None:
    """Interpreter that has torch + LivePortrait's pinned deps.

    No path is hardcoded. Discovery order:
      1. $LPR_PYTHON / $LPV_PYTHON       (explicit override, always wins)
      2. .venv created by the one-click setup   (the normal case now)
      3. $CONDA_PREFIX\\python.exe        (set when a conda env is activated)
      4. any conda "envs/lpv/python.exe" under the usual conda roots
      5. envs named lpv* next to this repo  (handles a portable layout)
    """
    def exe(p: Path) -> bool:
        return p.is_file() and p.name.lower().startswith("python")

    for var in ("LPR_PYTHON", "LPV_PYTHON"):
        env = os.environ.get(var, "").strip()
        # An unset/empty env var must not become Path(""), which resolves to the current
        # directory and would "exist" -- that silently selects a directory as the
        # interpreter and the worker never starts.
        if env and exe(Path(env)):
            return Path(env)

    cands: list[Path] = []

    # the venv the one-click setup creates, right beside the project
    here = Path(__file__).resolve().parent.parent
    cands.append(here / ".venv" / "Scripts" / "python.exe")

    prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if prefix:
        cands.append(Path(prefix) / "python.exe")

    cands.append(Path.home() / ".conda" / "envs" / "lpv" / "python.exe")

    # search the conda installs that actually exist, rather than assuming one
    roots: list[Path] = []
    for var in ("CONDA_ROOT", "MAMBA_ROOT_PREFIX"):
        v = os.environ.get(var, "").strip()
        if v:
            roots.append(Path(v))
    home = Path.home()
    roots += [home / "miniconda3", home / "anaconda3", home / "miniforge3",
              home / ".conda"]
    for drive_letter in ("C:", "D:", "E:"):
        roots.append(Path(f"{drive_letter}/My_App/miniconda3"))
    for r in roots:
        cands.append(r / "envs" / "lpv" / "python.exe")

    # a portable layout: an lpv* env sitting beside this project
    here = Path(__file__).resolve().parent.parent
    for parent in (here, here.parent):
        if parent.is_dir():
            for d in sorted(parent.glob("envs/lpv*")):
                cands.append(d / "python.exe")

    for c in cands:
        if exe(c):
            return c
    return None


class FrameWorker:
    """Run LivePortrait in the lpv interpreter and pipe frames to it."""

    def __init__(self, source: str | Path, quality: int = 90,
                 startup_timeout: float = 180.0, quiet: bool = True) -> None:
        self.source = str(source)
        self.quality = quality
        self.startup_timeout = startup_timeout
        self.quiet = quiet
        self.proc: subprocess.Popen | None = None
        self.ready = False
        self.error = ""
        self._last_ms = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        py = lpv_python()
        if py is None:
            self.error = "lpv interpreter not found (set LPV_PYTHON)"
            return False
        if not WORKER.exists():
            self.error = f"worker missing: {WORKER}"
            return False
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        args = [str(py), str(WORKER), "--source", self.source,
                "--quality", str(self.quality)]
        if self.quiet:
            args.append("--quiet")
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if self.quiet else None,
            cwd=str(ROOT), env=env, bufsize=0)

        # The worker loads its models before announcing readiness, and LivePortrait
        # logs those loads to stdout via `rich`. So the banner is NOT the first line:
        # skip everything that is not the READY line, otherwise it gets mistaken for
        # a startup failure.
        deadline = time.time() + self.startup_timeout
        line = ""
        while time.time() < deadline:
            got = self._read_line()
            if got is None:
                self.error = "worker exited before signalling READY"
                return False
            if got.startswith("READY"):
                line = got
                break
        if not line:
            self.error = "timed out waiting for READY"
            return False
        if not line.startswith("READY 1"):
            self.error = line.strip() or "worker failed to start"
            return False
        self.ready = True
        return True

    def _read_line(self) -> str | None:
        assert self.proc and self.proc.stdout
        buf = b""
        while True:
            ch = self.proc.stdout.read(1)
            if not ch:
                # EOF: return whatever is left, or None if there was nothing
                return buf.decode("utf-8", "replace") if buf else None
            if ch == b"\n":
                return buf.decode("utf-8", "replace")
            buf += ch

    def _read_exact(self, n: int) -> bytes | None:
        assert self.proc and self.proc.stdout
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write(struct.pack("<I", MAGIC_STOP))
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        self.proc = None
        self.ready = False

    # -- per frame ---------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Send one frame, return the animated frame (or None on failure)."""
        if not self.ready or self.proc is None or self.proc.stdin is None:
            return None
        ok, enc = cv2.imencode(".jpg", frame_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return None
        blob = enc.tobytes()
        try:
            self.proc.stdin.write(struct.pack("<I", len(blob)))
            self.proc.stdin.write(blob)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.ready = False
            self.error = "worker pipe closed"
            return None

        hdr = self._read_exact(4)
        if hdr is None:
            self.ready = False
            self.error = "worker stopped responding"
            return None
        (n,) = struct.unpack("<I", hdr)
        if n == MAGIC_STOP:
            self.ready = False
            self.error = "worker reported a fatal error"
            return None
        body = self._read_exact(n)
        if body is None:
            self.ready = False
            return None
        out = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        return out
