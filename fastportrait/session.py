"""End-to-end live portrait compositor: one object that owns the whole webcam path.

This sits above `fastportrait.realtime.LivePortraitFast` and adds the parts a real
session needs, none of which belong in the model driver:

  * open the camera and drain the frames Windows gives you on startup
  * skip frames when no face is present instead of animating garbage
  * reuse one detection result for both the crop and the size check
  * report honest statistics -- only frames that were actually animated count
    towards the frame rate, because a skipped frame costs almost nothing and would
    otherwise inflate the number
  * a clean `run()` loop plus a `step()` for embedding in another host

Kept separate from `realtime.py` so the model driver stays testable without a
camera, and so the Deep-Live-Cam processor (or any other host) can drive `step()`
one frame at a time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .realtime import FastConfig, LivePortraitFast


@dataclass
class SessionStats:
    frames: int = 0
    applied: int = 0
    skipped: int = 0
    times: list = field(default_factory=list)

    def add(self, ms: float, applied: bool) -> None:
        self.frames += 1
        if applied:
            self.applied += 1
            self.times.append(ms)
        else:
            self.skipped += 1

    @property
    def median_ms(self) -> float:
        return float(np.median(self.times)) if self.times else 0.0

    @property
    def fps(self) -> float:
        m = self.median_ms
        return 1000.0 / m if m > 0 else 0.0

    def summary(self) -> str:
        if not self.times:
            return (f"no animated frames ({self.skipped}/{self.frames} skipped -- "
                    f"was a face visible to the camera?)")
        a = np.array(self.times)
        return (f"{self.applied}/{self.frames} animated, {self.skipped} skipped | "
                f"median {self.median_ms:.1f} ms -> {self.fps:.1f} FPS | "
                f"p90 {np.percentile(a,90):.1f} ms")


class LivePortraitSession:
    """A source portrait plus a camera, animated in real time."""

    def __init__(self, source_bgr: np.ndarray,
                 config: FastConfig | None = None) -> None:
        self.engine = LivePortraitFast(config or FastConfig())
        self.source_ok = self.engine.set_source(source_bgr)
        self.stats = SessionStats()
        self._cap: cv2.VideoCapture | None = None

    # -- camera ------------------------------------------------------------
    def open_camera(self, index: int = 0, width: int = 1280,
                    height: int = 720, fps: int = 30, warm: int = 8) -> bool:
        # CAP_DSHOW keeps Windows from spending seconds on the MSMF backend.
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(max(0, warm)):     # the first frames are usually black
            self._cap.read()
            time.sleep(0.02)
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- one frame ---------------------------------------------------------
    def step(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Animate one frame. Returns (output, was_animated)."""
        t = time.perf_counter()
        out = self.engine.process(frame_bgr)
        applied = self.engine.last.applied
        self.stats.add((time.perf_counter() - t) * 1000.0, applied)
        return out, applied

    def run(self, seconds: float = 0.0, display: bool = False,
            save_dir: Path | None = None, save_every: int = 25,
            max_saves: int = 3, quiet: bool = False) -> SessionStats:
        """Drive the camera until `seconds` elapse (0 = until quit)."""
        if self._cap is None:
            raise RuntimeError("open_camera() first")
        if not self.source_ok:
            raise RuntimeError("source portrait has no detectable face")

        # Frames captured here can show the user's face; default to a cache directory
        # that the app wipes on close, so nothing private is left in the project.
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent.parent / ".cache"
        saved = 0
        t_end = time.time() + seconds if seconds > 0 else None
        while True:
            if t_end is not None and time.time() > t_end:
                break
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue
            out, applied = self.step(frame)

            if (save_dir is not None and applied and saved < max_saves
                    and self.stats.frames % save_every == 0):
                save_dir.mkdir(parents=True, exist_ok=True)
                # side by side, scaled to the same height
                h = out.shape[0]
                w = int(frame.shape[1] * h / frame.shape[0])
                cv2.imwrite(str(save_dir / f"session_{saved:02d}.png"),
                            np.hstack([cv2.resize(frame, (w, h)), out]))
                saved += 1

            if not quiet and self.stats.frames % 30 == 0:
                print("  " + self.stats.summary())

            if display:
                cv2.imshow("Runtime-LivePortrait", out)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("s") and save_dir is not None:
                    cv2.imwrite(str(save_dir / f"snap_{int(time.time())}.png"), out)
        if display:
            cv2.destroyAllWindows()
        return self.stats
