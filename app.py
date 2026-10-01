"""Web UI for LivePortrait-Runtime.

Three panes: the raw webcam (what the driver sees), the animated portrait, and a
live readout of the detected head pose and expression motion.

Why the standard library instead of Gradio/FastAPI
--------------------------------------------------
The `lpv` environment has no web framework, and installing Gradio would drag in a
large dependency tree (including a second numpy) into an environment whose versions
are deliberately pinned to what LivePortrait needs. `http.server` plus a small MJPEG
stream does everything this UI needs and adds no dependencies:

    GET  /                the page
    POST /api/portrait    upload the portrait to animate
    POST /api/settings    apply settings (restarts the session)
    POST /api/stop        stop the session
    GET  /api/stream      MJPEG, ?src=driver for the raw webcam, ?src=out for the result
    GET  /api/stats       FPS, stage timings, detected pose/expression

Threading model
---------------
One worker thread owns the camera and the engine and always leaves the newest raw and
processed JPEGs in `Session`. Every connected browser reads those slots, so the camera
is opened exactly once no matter how many pages are open and a slow client can never
stall the capture loop.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fastportrait  # noqa: F401  (applies the CUDA DLL PATH fix before onnxruntime)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

PORTRAITS = ROOT / "portraits"
# Bundled sample / default portrait: a frontal, neutral photo of the intended
# subject, so "use the bundled sample" and the initial default are the same person
# instead of an earlier placeholder.
SAMPLE = PORTRAITS / "sample.jpg"
DEFAULT_PORTRAIT = SAMPLE   # bundled samples are discovered, see sample_portraits()

# Anything the user explicitly saves lands here and is NEVER touched by the privacy
# cleanup -- results the user asked to keep are not temporary files.
OUTPUT = ROOT / "output"

# --- privacy cleanup ---------------------------------------------------------
# Everything the app writes that could contain the user's own face, and therefore
# must not survive a close. Deletion is limited to IMAGE files: a first version
# removed every non-keep file in portraits/, which also destroyed a user's
# `notes.md` and `data.json`. Only photos are in scope.
#
# Which files survive is a PATTERN, not a list of two names: any image called
# `sample*` is a bundled sample (sample.jpg, sample2.jpg, sample3.PNG, ...) and is
# never deleted. Adding another sample therefore needs no code change.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
SAMPLE_STEM_PREFIX = "sample"
# guards against running the sweep twice (handler + finally + atexit)
_CLEANUP_DONE = False


def is_bundled_sample(name: str) -> bool:
    """True for bundled sample portraits: `sample*` with an image extension.

    Case-insensitive on both parts, because the files come from various sources --
    `sample3.PNG` must be kept just like `sample.jpg`.
    """
    p = Path(name)
    return (p.suffix.lower() in IMAGE_SUFFIXES
            and p.stem.lower().startswith(SAMPLE_STEM_PREFIX))


def sample_portraits() -> list[Path]:
    """All bundled sample portraits, sorted (sample.jpg, sample2.jpg, ...)."""
    if not PORTRAITS.is_dir():
        return []
    return sorted((p for p in PORTRAITS.iterdir()
                   if p.is_file() and is_bundled_sample(p.name)),
                  key=lambda p: p.name.lower())


def cleanup_private_files(verbose: bool = False) -> list[str]:
    """Delete the user's uploaded photos and this app's temporary frames.

    Called on shutdown so that closing the app does not leave the user's face on
    disk. Also called once at startup, so a previous crash (which skips the shutdown
    path) cannot leave photos behind either.

    Scope, stated explicitly because over-deleting would be worse than under-deleting:

      portraits/   delete IMAGE files only, except the two bundled synthetic samples.
                   Non-image files (notes, data) are left alone on purpose.
      ROOT/.cache/ delete entirely (webcam snapshots written by the tools)
      docs/        delete only this app's own capture patterns
                   (lpv_webcam_*.png, lpv_snap_*.png, session_*.png, snap_*.png);
                   the documentation images are NOT touched
      ROOT/*.txt   transcripts written by the console tools

    NOT touched, deliberately: output/ (things the user asked to save) and .git.
    """
    removed: list[str] = []

    def drop(p: Path) -> None:
        try:
            if p.is_file():
                p.unlink()
                removed.append(p.name)
        except OSError:
            pass

    def protected(p: Path) -> bool:
        """Hard guard: never delete anything under output/.

        output/ is only touched by the two explicit save buttons. Keeping the guard
        here (rather than only in the patterns below) means a future change to those
        patterns still cannot delete a video the user chose to keep.
        """
        try:
            p.resolve().relative_to(OUTPUT.resolve())
            return True
        except (ValueError, OSError):
            return False

    def drop_unless_protected(p: Path) -> None:
        if not protected(p):
            drop(p)

    # 0. nothing in output/ is ever swept, whatever the patterns below say
    # 1. user photos (images only -- see the note above)
    if PORTRAITS.is_dir():
        for f in PORTRAITS.iterdir():
            if (f.is_file() and not is_bundled_sample(f.name)
                    and f.suffix.lower() in IMAGE_SUFFIXES):
                drop_unless_protected(f)

    # 2. temp frames written by the tools
    cache = ROOT / ".cache"
    if cache.is_dir():
        for f in cache.rglob("*"):
            if f.is_file():
                drop_unless_protected(f)

    # 3. this app's capture patterns in docs/ (never the documentation images)
    docs = ROOT / "docs"
    if docs.is_dir():
        for pat in ("lpv_webcam_*.png", "lpv_snap_*.png",
                    "session_*.png", "snap_*.png"):
            for f in docs.glob(pat):
                drop_unless_protected(f)

    # 4. transcripts the console tools write
    for f in ROOT.glob("*.txt"):
        drop_unless_protected(f)

    if verbose and removed:
        print(f"  privacy cleanup: removed {len(removed)} file(s)")
        for n in removed[:20]:
            print(f"    - {n}")
    return removed


def imread_unicode(path: Path):
    """cv2.imread fails on non-ASCII Windows paths; imdecode does not."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# --- saving results ----------------------------------------------------------
# cv2.VideoWriter with the MJPG codec writes an .avi that Windows Media Player,
# VLC, PotPlayer and browsers all play without extra codecs. Probed encodings are
# listed in order: MJPG first (most compatible), then mp4v/XVID as fallbacks for
# OpenCV builds without the MJPEG encoder.
VIDEO_FOURCCS = ("MJPG", "mp4v", "XVID")
VIDEO_SUFFIX = {".avi", ".mp4"}


class Recorder:
    """Append the latest composite frame to a video file at a steady rate.

    Why a ticker instead of writing on every frame: the engine runs slower than the
    video's nominal frame rate and its speed varies. Writing one frame per processed
    frame would make the saved video play at the wrong speed, so a thread samples the
    most recent frame at a fixed interval and the nominal fps is therefore truthful.
    """

    def __init__(self, path: Path, frame_source, fps: int = 25) -> None:
        self.path = path
        self.frame_source = frame_source        # callable -> BGR ndarray or None
        self.fps = int(fps)
        self.frames = 0
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: cv2.VideoWriter | None = None

    # -- lifecycle --
    def start(self) -> bool:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="lpv-record", daemon=True)
        self._thread.start()
        # wait briefly so a failure to open the writer is reported to the click
        for _ in range(40):
            if self.error or self._writer is not None:
                break
            time.sleep(0.05)
        return not self.error

    def stop(self) -> dict:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=10)
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"file": self.path.name, "frames": self.frames,
                "size_mb": round(size / 1048576, 2), "error": self.error}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- worker --
    def _run(self) -> None:
        probe = None
        for fourcc in VIDEO_FOURCCS:
            w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*fourcc),
                                float(self.fps), probe if probe else (1280, 720))
            if w.isOpened():
                self._writer = w
                break
            w.release()
        if self._writer is None:
            self.error = "no usable video codec in this OpenCV build"
            return

        period = 1.0 / self.fps
        nxt = time.perf_counter()
        try:
            while not self._stop.is_set():
                frame = self.frame_source()
                if frame is not None:
                    h, w = frame.shape[:2]
                    if probe is None:
                        # Real size is unknown until the first frame arrives, so the
                        # writer was opened with a guess. Reopen once at the true size
                        # if it differs, otherwise VideoWriter silently drops frames.
                        if (w, h) != (1280, 720):
                            self._writer.release()
                            fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCCS[0])
                            self._writer = cv2.VideoWriter(str(self.path), fourcc,
                                                           float(self.fps), (w, h))
                        probe = (w, h)
                    self._writer.write(frame)
                    self.frames += 1
                nxt += period
                delay = nxt - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    nxt = time.perf_counter()   # fell behind; do not accumulate
        finally:
            if self._writer is not None:
                self._writer.release()


class Session:
    """Owns the camera and engine; publishes the newest raw and processed frames."""

    # The engine holds ~2.5 GB of weights and never changes, so it is built once for
    # the whole process and shared by every session. Reloading it per Start was both
    # slow and the cause of a hang (see _get_engine).
    _ENGINE = None
    _SOURCE_KEY = None

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.running = False
        # ONE camera thread for the whole app lifetime, driven by `mode`. Having a
        # separate preview thread and session thread both open the camera is what
        # froze the webcam pane -- see the note on CAM_* below.
        self.cam_thread: threading.Thread | None = None
        self.cam_stop = threading.Event()
        self.mode = "preview"
        # depth-1 handoff from the camera thread to the engine
        self.slot_lock = threading.Lock()
        self.frame_slot = None
        self.slot_ready = threading.Event()
        self.raw: bytes | None = None
        self.out: bytes | None = None          # JPEG for the MJPEG stream
        self.last_out = None                  # BGR frame for snapshot/recording
        self.portrait_path: Path | None = None
        self.status = "idle"
        self.error = ""
        self.fps = 0.0
        self.frame_ms = 0.0
        self.applied = 0
        self.skipped = 0
        self.pose: dict = {}
        self.motion = 0.0
        self.settings = {
            "camera": 0,
            "region": "all",
            "pasteback": False,
            "stitching": True,
            "multiplier": 0.7,
            "jpeg": 80,
        }

    # -- camera: ONE owner, two modes --------------------------------------
    #
    # The preview used to be its own thread with its own VideoCapture, while the
    # session opened a second one. They raced, and the bug was concretely this:
    # `_preview_loop` ran `while self.previewing and not self.running`, but `start()`
    # set `self.running = True` BEFORE calling `stop_preview()`. The preview loop's
    # condition therefore failed on the spot and the thread ended -- its `finally:
    # cap.release()` never ran, so the camera handle leaked and the session's handle
    # was starved. Result: the webcam pane froze the moment Start was pressed.
    #
    # Rather than patch the flag ordering (easy to regress), a single thread now owns
    # the only camera handle for the whole lifetime of the app and just switches mode.
    # There is no handover, so there is nothing to race.
    CAM_IDLE, CAM_PREVIEW, CAM_SESSION = "idle", "preview", "session"

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def start_preview(self) -> None:
        """Show the webcam before Start, without loading the models."""
        if self.running:
            return
        self.mode = self.CAM_PREVIEW

    def stop_preview(self) -> None:
        if self.mode == self.CAM_PREVIEW:
            self.mode = self.CAM_IDLE

    def _ensure_camera(self) -> None:
        """Start the single capture thread. Idempotent."""
        if self.cam_thread is not None and self.cam_thread.is_alive():
            return
        self.cam_stop.clear()
        self.cam_thread = threading.Thread(target=self._capture_loop,
                                           name="lpv-camera", daemon=True)
        self.cam_thread.start()

    def _capture_loop(self) -> None:
        """The only place the camera is opened.

        Publishes a raw JPEG on EVERY frame (so the webcam pane runs at camera speed)
        and hands the newest frame to the engine through a depth-1 slot. Frames the
        engine cannot keep up with are dropped, which is right for live video -- a
        backlog would only add latency.
        """
        cam = int(self.settings["camera"])
        quality = int(self.settings["jpeg"])
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error = f"cannot open camera {cam}"
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(8):                       # first DSHOW frames are often black
            cap.read()
            time.sleep(0.02)
        try:
            while not self.cam_stop.is_set():
                if self.mode == self.CAM_IDLE:
                    time.sleep(0.02)
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
                ok_j, buf = cv2.imencode(".jpg", frame,
                                         [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok_j:
                    with self.lock:
                        self.raw = buf.tobytes()
                if self.mode == self.CAM_SESSION:
                    with self.slot_lock:
                        self.frame_slot = frame
                    self.slot_ready.set()
        finally:
            cap.release()

    def stop_camera(self) -> None:
        self.mode = self.CAM_IDLE
        self.cam_stop.set()
        t = self.cam_thread
        if t is not None and t.is_alive():
            t.join(timeout=5)
        self.cam_thread = None

    # -- session (engine) --------------------------------------------------
    def _get_engine(self, cfg):
        """Build the engine once and keep it; only the source is re-set after that.

        Measured bug this fixes: `_engine_loop` used to construct a `LivePortraitFast`
        on EVERY Start, which reloads all four base models (~2.5 GB) and warms up the
        detectors. Start #1 reached `running` in 8.7s, but #2 and #3 NEVER did -- the
        rebuild hung partway (only 2 of 3 `appearance_feature_extractor` loads in the
        log completed), so the page sat on "starting" forever and Start looked dead
        after switching portraits a few times.

        The weights never change, so rebuilding them was pure waste. The engine is now
        a singleton and `set_source` is the only per-portrait work.
        """
        if type(self)._ENGINE is None:
            from fastportrait.realtime import LivePortraitFast
            type(self)._ENGINE = LivePortraitFast(cfg)
        return type(self)._ENGINE

    def _apply_source(self, engine) -> bool:
        """Point the (cached) engine at the currently selected portrait."""
        if self.portrait_path is None:
            return False
        key = str(self.portrait_path)
        if key == type(self)._SOURCE_KEY:
            return True                       # already driving this portrait
        src = imread_unicode(self.portrait_path)
        if src is None:
            self.error = f"cannot read portrait: {self.portrait_path.name}"
            return False
        if not engine.set_source(src):
            self.error = "no face found in the portrait - try a frontal photo"
            return False
        type(self)._SOURCE_KEY = key
        return True

    def _engine_loop(self) -> None:
        """Consume the newest camera frame, animate it, publish the result."""
        try:
            from fastportrait.realtime import FastConfig
        except Exception as exc:                                    # noqa: BLE001
            self.error = f"import failed: {exc}"
            self.running = False
            return

        cfg = FastConfig(animation_region=self.settings["region"],
                         pasteback=self.settings["pasteback"],
                         stitching=self.settings["stitching"],
                         driving_multiplier=float(self.settings["multiplier"]))
        try:
            engine = self._get_engine(cfg)
        except Exception as exc:                                    # noqa: BLE001
            self.error = f"engine init failed: {exc}"
            self.running = False
            return
        if not self._apply_source(engine):
            if not self.error:
                self.error = "no portrait selected"
            self.running = False
            return

        quality = int(self.settings["jpeg"])
        self.status = "running"
        times: list[float] = []
        try:
            while self.running:
                if not self.slot_ready.wait(timeout=0.5):
                    continue
                self.slot_ready.clear()
                with self.slot_lock:
                    frame = self.frame_slot
                    self.frame_slot = None
                if frame is None:
                    continue

                t0 = time.perf_counter()
                try:
                    out = engine.process(frame)
                except Exception as exc:                            # noqa: BLE001
                    self.error = f"frame failed: {exc}"
                    break
                ms = (time.perf_counter() - t0) * 1000
                if engine.last.applied:
                    self.applied += 1
                    times.append(ms)
                else:
                    self.skipped += 1
                if times:
                    recent = times[-30:]
                    self.frame_ms = sum(recent) / len(recent)
                    self.fps = 1000.0 / self.frame_ms if self.frame_ms else 0.0
                self.pose = dict(engine.last_pose)
                self.motion = engine.last_motion

                ok_o, buf_o = cv2.imencode(".jpg", out,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok_o:
                    with self.lock:
                        self.out = buf_o.tobytes()
                        # keep the raw composite too: the recorder needs actual pixels
                        # (re-encoding the JPEG every tick would lose quality and cost
                        # CPU for no reason)
                        self.last_out = out
        finally:
            self.running = False
            if not self.error:
                self.status = "stopped"

    # -- saving ------------------------------------------------------------
    def out_frame(self):
        """Newest composite frame as BGR ndarray, or None."""
        with self.lock:
            return None if self.last_out is None else self.last_out.copy()

    def save_snapshot(self) -> dict:
        """Write the current composite frame to output/ and report what was written."""
        frame = self.out_frame()
        if frame is None:
            return {"ok": False, "error": "no composite frame yet — press 开始 first"}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / f"snapshot_{_stamp()}.png"
        n = 1
        while path.exists():                    # two clicks within the same second
            path = OUTPUT / f"snapshot_{_stamp()}_{n}.png"
            n += 1
        try:
            ok = cv2.imwrite(str(path), frame)
        except Exception as exc:                                    # noqa: BLE001
            return {"ok": False, "error": f"write failed: {exc}"}
        if not ok:
            return {"ok": False, "error": "write failed"}
        return {"ok": True, "file": path.name, "dir": str(OUTPUT),
                "size_kb": round(path.stat().st_size / 1024, 1)}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Switch the camera to session mode and start the engine thread."""
        self.stop_engine()
        if self.portrait_path is None:
            self.error = "no portrait selected"
            return
        self.error = ""
        self.status = "starting (models load once, 5-7 s)"
        self.applied = self.skipped = 0
        self._ensure_camera()          # single owner; no handover to race
        self.mode = self.CAM_SESSION
        self.running = True
        self.thread = threading.Thread(target=self._engine_loop,
                                       name="lpv-engine", daemon=True)
        self.thread.start()

    def stop_engine(self) -> None:
        """Stop only the engine; the camera thread keeps running for the preview."""
        self.running = False
        t = self.thread
        if t is not None and t.is_alive():
            t.join(timeout=5)
        self.thread = None

    def stop(self) -> None:
        self.stop_engine()
        if self.mode == self.CAM_SESSION:
            self.mode = self.CAM_PREVIEW     # fall back to showing the camera
        self.status = "stopped"

    def snapshot(self, which: str) -> bytes | None:
        with self.lock:
            return self.raw if which == "driver" else self.out


SESSION = Session()
# Active Recorder, or None. One at a time.
RECORDER: "Recorder | None" = None


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LivePortrait-Runtime</title>
<style>
  :root { color-scheme: dark; --bg:#14161a; --panel:#1b1e24; --line:#262a32; --dim:#8b93a3; }
  * { box-sizing: border-box; }
  html, body { height:100%; }
  body { margin:0; background:var(--bg); color:#e7e9ee;
         font:14px/1.6 "Microsoft YaHei",system-ui,Segoe UI,sans-serif;
         display:flex; flex-direction:column; overflow:hidden; }

  header { flex:0 0 auto; padding:8px 16px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header span { color:var(--dim); font-size:12px; }

  main { flex:1 1 auto; min-height:0; display:grid; gap:12px; padding:12px 16px;
         grid-template-columns: minmax(230px, 288px) minmax(0, 1fr); }

  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px;
           padding:12px 14px; overflow-y:auto; min-height:0; }
  .panel h2 { font-size:11px; letter-spacing:.06em; color:var(--dim); margin:0 0 8px; }

  label { display:block; margin:9px 0 3px; font-size:12.5px; color:#b6bdcb; }
  input[type=file], select { width:100%; background:#12141a; color:#e7e9ee;
        border:1px solid #2e333d; border-radius:6px; padding:6px 7px; font-size:13px; }
  input[type=range] { width:100%; }
  button { width:100%; margin-top:9px; padding:9px; border-radius:7px; border:0;
           font-size:14px; font-weight:600; cursor:pointer; background:#3b82f6; color:#fff; }
  button.ghost { background:#2a2f39; color:#cfd5e0; font-weight:500; }
  button:disabled { opacity:.45; cursor:default; }
  .inline { display:flex; align-items:center; gap:8px; margin-top:7px;
            font-size:12.5px; color:#b6bdcb; }
  .inline input { width:auto; }
  .inline label { margin:0; }

  /* --- stage: the portrait is the main view, the webcam sits BESIDE it ---
     It used to be an absolutely-positioned inset floating on top of the result and
     covering part of the face. It is now a sibling column, so nothing overlaps. */
  .stage { display:flex; flex-direction:column; min-height:0; gap:10px; }
  .views { flex:1 1 auto; min-height:0; display:flex; gap:10px;
           align-items:stretch; }

  /* The result panel spans the full column, as it always has. */
  /* The result panel holds a title bar plus the media underneath. The bar is a normal
     flex row now, NOT an overlay: attach() and stop() both replace the media element's
     innerHTML, so anything inside it (including a button) was being destroyed every
     time the stream was re-attached -- which is what made the "new tab" button
     disappear on Stop and never come back. Keeping the bar outside the cleared element
     is what makes it survive. */
  .main-view { flex:1 1 auto; min-width:0; min-height:0; background:#000;
               border:1px solid var(--line); border-radius:10px; overflow:hidden;
               display:flex; flex-direction:column; }
  .main-head { flex:0 0 auto; display:flex; align-items:center;
               justify-content:space-between; gap:6px; padding:5px 8px;
               background:#14161a; border-bottom:1px solid var(--line); }
  .main-head .cap { font-size:10.5px; color:var(--dim); }
  .main-media { flex:1 1 auto; min-height:0; position:relative; overflow:hidden; }

  .main-media img { position:absolute; inset:0; width:100%; height:100%;
                    object-fit:contain; object-position:center; }

  .main-media #hint { position:absolute; left:50%; top:50%;
                      transform:translate(-50%, -50%);
                      width:max-content; max-width:min(40ch, 90%); }

  /* webcam column: fixed width, pinned to the TOP, and sized to the camera's own
     16:9 aspect ratio (the user asked for the vertical extent to be smaller and the
     shape to match the image). It never overlaps the portrait. */
  .side { flex:0 0 auto; align-self:flex-start; width:240px; display:flex;
          flex-direction:column; }
  .side-head { flex:0 0 auto; display:flex; align-items:center;
               justify-content:space-between; gap:6px; padding:0 0 4px; }
  .side .cap { flex:1 1 auto; font-size:10.5px; color:var(--dim); }
  /* small inline button, deliberately not the full-width style used for Start/Stop */
  button.mini { width:auto; margin:0; padding:3px 8px; font-size:11px;
                font-weight:500; border-radius:5px; background:#2a2f39;
                color:#cfd5e0; white-space:nowrap; }
  button.mini.on { background:#b45309; color:#fff; }

  /* save row: snapshot + record, side by side under Start/Stop */
  .save-row { display:flex; gap:6px; margin-top:6px; }
  .save-row button { flex:1 1 0; width:auto; margin:0; font-size:12px;
                     padding:8px 6px; }

  /* Recording indicator. "Slowly pulsing" on purpose -- a fast blink reads as an
     error, a slow one reads as "in progress". Colour AND motion change so it is
     still obvious with animations disabled. */
  @keyframes recPulse {
    0%, 100% { background:#7f1d1d; color:#fff; box-shadow:0 0 0 0 rgba(239,68,68,.55); }
    50%      { background:#dc2626; color:#fff; box-shadow:0 0 0 7px rgba(239,68,68,0); }
  }
  button.recording { animation:recPulse 2s ease-in-out infinite;
                     border-color:#dc2626; }
  button.recording::before { content:"\25CF "; }
  @media (prefers-reduced-motion: reduce) {
    button.recording { animation:none; background:#dc2626; color:#fff; }
  }

  /* Privacy blur: applied ONLY to the element on screen, in the browser.
     Nothing is re-uploaded or re-processed, so the driver frame the model sees is
     untouched -- the animation keeps working normally while the preview is hidden.

     No `transform` here on purpose. It used to carry `scale(1.12)` to hide the dark
     edge bleed that a blur makes at the borders, and that scaling cropped ~12% off
     every side. Because the tile is then a different crop of the same frame, an
     asymmetric scene (the face off to one side, a door behind you) reads as though
     the picture had shifted or flipped relative to the un-blurred view. Measured: the
     blur class itself never mirrors -- both tiles kept the marker on the same side --
     so the zoom was purely cosmetic and is not worth the confusion.
     Orientation therefore stays identical with and without the blur. */
  /* Privacy mask for the INPUT window.
     The camera tile is the only place a real person appears, so it is masked beyond
     recovery: the image is scaled down to a few pixels and smoothed back up, which
     destroys facial structure entirely. A plain Gaussian blur was the earlier
     approach, but a blur is a reversible convolution -- given a lossless frame it can
     in principle be undone. This cannot. The composite window is deliberately left
     clear, since it shows the selected portrait rather than the person at the camera. */
  .side-frame img.blur {
    filter: blur(16px) url(#lprPixelate);
    transform: scale(1.02);   /* hides the soft edge the blur leaves at the borders */
  }

  /* a subtle badge so it is obvious the mask is a display filter, not a failure */
  .side-frame { overflow:hidden; }
  .side-frame.blurred { position:relative; }
  .side-frame.blurred::after {
    content:"隐私模糊";
    position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
    font-size:11px; color:#fff; background:rgba(0,0,0,.55);
    padding:3px 9px; border-radius:10px; pointer-events:none;
  }
  .side-frame { aspect-ratio:16 / 9; background:#000;
                border:1px solid var(--line); border-radius:9px; overflow:hidden;
                display:flex; align-items:center; justify-content:center; }
  .side-frame img { display:block; width:100%; height:100%; object-fit:cover; }

  #hint { color:var(--dim); font-size:13px; padding:20px; text-align:center; max-width:40ch; }

  .stats { flex:0 0 auto; display:grid; gap:8px;
           grid-template-columns:repeat(auto-fit, minmax(86px, 1fr)); }
  .stat { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:6px 10px; }
  .stat b { display:block; font-size:clamp(14px, 1.8vw, 18px); font-weight:600; }
  .stat i { font-style:normal; font-size:10.5px; color:var(--dim); }

  #status { margin-top:8px; font-size:12.5px; color:#9fb4d8; min-height:18px; }
  #err { margin-top:7px; font-size:12.5px; color:#ff8f8f; min-height:15px; white-space:pre-wrap; }
  .tip { margin-top:9px; font-size:11.5px; color:var(--dim); line-height:1.5;
         border-top:1px solid var(--line); padding-top:9px; }
  .tip b { color:#b6bdcb; font-weight:600; }

  @media (max-width: 820px) {
    body { overflow:auto; }
    main { grid-template-columns:1fr; overflow:visible; }
    .panel { overflow:visible; }
    .side { width:130px; }
    .views { flex-direction:column; min-height:52vh; }
    .side { width:100%; }
  }
</style></head><body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><filter id="lprPixelate" x="0" y="0" width="100%" height="100%">
  <feFlood x="1" y="1" width="1" height="1"/>
  <feComposite width="14" height="14"/>
  <feTile result="tiled"/>
  <feComposite in="SourceGraphic" in2="tiled" operator="in"/>
  <feMorphology operator="dilate" radius="7"/>
</filter></svg>
<header>
  <h1>LivePortrait-Runtime</h1>
  <span>LivePortrait &middot; 全部在本机运行 &middot; 不上传任何数据</span>
</header>
<main>
  <div class="panel">
    <h2>肖 像</h2>
    <label>要驱动的照片</label>
    <input type="file" id="portrait" accept="image/*">
    <div class="inline">
      <input type="checkbox" id="useSample">
      <label for="useSample">改用内置样张</label>
    </div>
    <select id="samplePick" style="margin-top:4px"></select>
    <div id="portraitName" class="tip" style="border:0;padding-top:3px;margin-top:1px">
      尚未选择照片 &mdash; 将使用内置样张</div>
    <div class="tip"><b>选图要点：</b>正脸看镜头、整颗头都在画面里（别裁掉下巴）、
      光线均匀、<b>表情自然</b>。避免侧脸、低头、墨镜、夸张表情。</div>

    <h2 style="margin-top:14px">设 置</h2>
    <label>摄像头</label>
    <select id="camera"><option value="0">0（默认）</option><option value="1">1</option><option value="2">2</option></select>

    <label>驱动范围</label>
    <select id="region">
      <option value="all">全部（姿态 + 表情）</option>
      <option value="exp">仅表情</option>
      <option value="pose">仅头部姿态</option>
      <option value="lip">仅嘴部</option>
      <option value="eyes">仅眼睛</option>
    </select>

    <label>动作强度 <span id="multv">0.7</span> 倍</label>
    <input type="range" id="multiplier" min="0.3" max="2.0" step="0.1" value="0.7">
    <div class="tip" style="border:0;padding-top:2px;margin-top:2px">
      调大 → 转头和表情更明显；调小 → 更收敛。表情夸张时调小。</div>

    <label>把结果贴回摄像头画面</label>
    <select id="pasteback">
      <option value="0" selected>关 &mdash; 显示动画后的肖像（推荐）</option>
      <option value="1">开 &mdash; 叠加到摄像头画面</option>
    </select>

    <label>头部拼接</label>
    <select id="stitching"><option value="1" selected>开（默认）</option><option value="0">关</option></select>

    <label>画面质量 <span id="jpegv">80</span></label>
    <input type="range" id="jpeg" min="50" max="95" value="80">

    <button id="start">开始</button>
    <button id="stop" class="ghost">停止</button>
    <div class="save-row">
      <button id="snapBtn" class="ghost" type="button"
              title="把当前合成画面存成 PNG 到 output/">保存截图</button>
      <button id="recBtn" class="ghost" type="button"
              title="开始/停止录制合成画面到 output/">保存视频</button>
    </div>
    <div id="status">未启动</div>
    <div id="err"></div>
    <div id="saveMsg" class="tip" style="display:none"></div>
  </div>

  <div class="stage">
    <div class="views">
      <div class="main-view">
        <div class="main-head">
          <span class="cap">合成画面</span>
          <button id="popOutBtn" class="mini" type="button"
                  title="在新标签页中全屏观看合成画面">在新标签页打开</button>
        </div>
        <div class="main-media" id="frameOut">
          <div id="hint">点 <b>开始</b>。首次启动要加载模型，5&ndash;7 秒黑屏是正常的。</div>
        </div>
      </div>
      <div class="side">
        <div class="side-head">
          <span class="cap">摄像头（驱动源）</span>
          <button id="popCamBtn" class="mini" type="button"
                  title="在新标签页中观看摄像头画面">新标签页</button>
          <button id="blurBtn" class="mini" type="button" title="仅模糊网页上的预览，不影响识别">隐私模糊</button>
        </div>
        <div class="side-frame" id="frameIn">
          <div id="hint" style="padding:12px;font-size:11px">等待中</div>
        </div>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><b id="fps">-</b><i>动画帧率</i></div>
      <div class="stat"><b id="ms">-</b><i>毫秒/帧</i></div>
      <div class="stat"><b id="ok">0</b><i>已动画帧</i></div>
      <div class="stat"><b id="skip">0</b><i>无人脸（保持画面）</i></div>
      <div class="stat"><b id="pitch">-</b><i>俯仰&deg;</i></div>
      <div class="stat"><b id="yaw">-</b><i>左右&deg;</i></div>
      <div class="stat"><b id="motion">-</b><i>表情幅度</i></div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const nameBox = $('portraitName');
const TXT_SAMPLE = '正在使用内置样张';
const TXT_NONE = '尚未选择照片 \u2014 将使用内置样张';

// ---- bundled sample picker -----------------------------------------------
// The bundled samples are whatever matches `sample*` in portraits/, so sample3 etc.
// appear automatically. Selecting one drives that photo.
const samplePick = $('samplePick');
let sampleNames = [];

function sampleLabel(n) {
  return n === 'sample.jpg' ? n + '（默认）' : n;
}

async function loadSamples() {
  try {
    const j = await (await fetch('/api/samples')).json();
    sampleNames = j.samples || [];
    samplePick.innerHTML = '';
    for (const n of sampleNames) {
      const o = document.createElement('option');
      o.value = n; o.textContent = sampleLabel(n);
      samplePick.appendChild(o);
    }
    if (j.current && sampleNames.includes(j.current)) samplePick.value = j.current;
    samplePick.style.display = sampleNames.length > 1 ? '' : 'none';
  } catch (e) {}
}

// Selecting a file must override the sample checkbox, and vice versa.
$('portrait').addEventListener('change', () => {
  const f = $('portrait').files[0];
  if (f) { $('useSample').checked = false; nameBox.textContent = '将驱动：' + f.name; }
  else { nameBox.textContent = $('useSample').checked ? TXT_SAMPLE : TXT_NONE; }
  // a new file means a new source; make Start available straight away
  lastStatus = 'stopped';
  $('start').disabled = false;
});
$('useSample').addEventListener('change', () => {
  const f = $('portrait').files[0];
  nameBox.textContent = $('useSample').checked ? TXT_SAMPLE
    : (f ? '将驱动：' + f.name : TXT_NONE);
});
// picking a different bundled sample implies "use the bundled sample"
samplePick.addEventListener('change', () => {
  $('useSample').checked = true;
  $('portrait').value = '';
  nameBox.textContent = '将驱动：' + samplePick.value;
  // Deliberately does NOT post to /api/portrait: Start sends the selection itself.
  // Posting here as well meant the server saw the same portrait twice, treated the
  // second one as a change, and stopped the session that Start had just launched --
  // which is why the button clicked but produced no picture.
  // The server ends a running session when the portrait genuinely changes, so Start
  // is free to press either way.
  $('status').textContent = '已选择源图，点「开始」生效';
});
loadSamples();
$('jpeg').addEventListener('input', e => $('jpegv').textContent = e.target.value);
$('multiplier').addEventListener('input', e => $('multv').textContent = (+e.target.value).toFixed(1));

// ---- Start button state -------------------------------------------------
// Helpers so the button's enabled/disabled state has exactly one definition.
// Bug this fixes: the poll below disables Start whenever the status starts with
// "running", and nothing re-enabled it when the source portrait changed -- so after
// switching portraits the button stayed grey and Stop had to be pressed first.
let lastStatus = '';
function refreshStart() {
  const busy = (lastStatus || '').indexOf('running') === 0;
  $('start').disabled = busy;
}

// ---- save snapshot / record video --------------------------------------
// Both write into output/, which the privacy cleanup never touches.
let recording = false;

function saveMsg(text, ok) {
  const el = $('saveMsg');
  el.style.display = '';
  el.textContent = text;
  el.style.color = ok ? '' : '#f87171';
}

$('snapBtn').addEventListener('click', async () => {
  $('snapBtn').disabled = true;
  try {
    const j = await (await fetch('/api/snapshot', { method: 'POST' })).json();
    if (j.ok) saveMsg(`已保存截图：${j.file}（${j.size_kb} KB）→ output/`, true);
    else saveMsg('截图失败：' + j.error, false);
  } catch (e) { saveMsg('截图失败：' + e, false); }
  $('snapBtn').disabled = false;
});

$('recBtn').addEventListener('click', async () => {
  const btn = $('recBtn');
  btn.disabled = true;
  try {
    const j = await (await fetch('/api/record', { method: 'POST' })).json();
    if (j.error) { saveMsg('录制失败：' + j.error, false); }
    else if (j.recording) {
      recording = true;
      btn.classList.add('recording');
      btn.textContent = '停止录制';
      saveMsg(`录制中：${j.file} → output/`, true);
    } else {
      recording = false;
      btn.classList.remove('recording');
      btn.textContent = '保存视频';
      const secs = j.frames ? (j.frames / 25).toFixed(1) : '0';
      saveMsg(`已保存视频：${j.file}（${j.frames} 帧 / 约 ${secs} 秒，`
              + `${j.size_mb} MB）→ output/`, true);
    }
  } catch (e) { saveMsg('录制失败：' + e, false); }
  btn.disabled = false;
});

// ---- privacy blur ------------------------------------------------------
// Purely a CSS filter on the <img> in this page. The JPEG stream from the server is
// never modified, so the frame the model is driven by stays sharp and the animation
// is unaffected -- only what is displayed is obscured. Useful when screen-sharing.
let blurOn = false;

function applyBlur() {
  const box = $('frameIn');
  const img = box.querySelector('img');
  if (img) img.classList.toggle('blur', blurOn);
  box.classList.toggle('blurred', blurOn);
  $('blurBtn').classList.toggle('on', blurOn);
  $('blurBtn').textContent = blurOn ? '取消模糊' : '隐私模糊';
}

// Open a stream full-screen in its own tab. A plain <img> is used on the viewer page
// rather than this page's element, because a second document needs its own
// connection -- which is exactly why attach() reuses one <img> per element here.
$('popOutBtn').addEventListener('click', () => {
  window.open('/viewer?src=out', '_blank', 'noopener');
});
$('popCamBtn').addEventListener('click', () => {
  window.open('/viewer?src=driver', '_blank', 'noopener');
});

$('blurBtn').addEventListener('click', () => {
  blurOn = !blurOn;
  applyBlur();
  try { localStorage.setItem('lpv_blur', blurOn ? '1' : '0'); } catch (e) {}
});

// ---- MJPEG stream plumbing ---------------------------------------------
// Every /api/stream response is an ENDLESS multipart reply, so it holds a browser
// connection open for as long as the page lives. `attach` used to build a brand-new
// <img> on every Start and drop the old element on the floor:
//
//   page load          -> 1 open connection (driver)
//   each click of 开始  -> +1 more (out)
//
// A browser allows only about six concurrent connections per host, so after four or
// five Starts every later request -- including /api/stats and Start itself -- queued
// behind them and never completed. That is why the button looked dead after a few
// source switches: nothing was failing, the requests were never being sent.
//
// Fix: keep ONE <img> per element and only ever change its `src`. Pointing an existing
// <img> at a new URL makes the browser tear down the previous multipart connection as
// part of replacing the image, so exactly one stream per element is ever open.
// (fetch()+blob() cannot be used here: a multipart stream never ends, so the promise
// would never resolve and no image would ever appear.)
const streamImgs = new Map();       // element -> its single <img>
// `mainStopped` mirrors the Start/Stop button. While stopped the composite stream is
// dropped (there is nothing to show), but the webcam preview keeps running so the face
// can still be framed before restarting.
let mainStopped = false;

function attach(el, src, opts) {
  const always = !!(opts && opts.always);   // the camera preview is never gated
  let img = streamImgs.get(el);
  if (!img) {
    img = new Image();
    img.alt = '';
    streamImgs.set(el, img);
  }
  img.onload = () => {
    if (el.firstChild !== img) {
      el.innerHTML = '';
      el.appendChild(img);
    }
    applyBlur();   // the blur class lives on the <img>, so re-apply after attaching
  };
  // Reusing the element and only changing `src` is the crux of the fix: the browser
  // cancels the previous multipart response when the source is replaced, so only one
  // stream per element is ever open. While stopped, leave the stream closed -- `stop()`
  // blanks the element anyway, so there is nothing to show.
  if (mainStopped && !always) return img;
  img.src = '/api/stream?src=' + src + '&t=' + Date.now();
  return img;
}

// remember the setting across reloads
try { blurOn = localStorage.getItem('lpv_blur') === '1'; } catch (e) {}
// ?mask=1 forces the camera mask on regardless of the saved preference. Useful for
// recording a demo or taking a screenshot without exposing whoever is at the camera.
try {
  if (new URLSearchParams(location.search).get('mask') === '1') blurOn = true;
} catch (e) {}
if (blurOn) { $('blurBtn').classList.add('on'); $('blurBtn').textContent = '取消模糊'; }

// The camera preview starts as soon as the page opens, so the face can be framed
// before pressing Start. The server runs a camera-only preview thread for this;
// it hands the camera over to the session when Start is pressed.
attach($('frameIn'), 'driver', { always: true });

async function start() {
  $('err').textContent = ''; $('status').textContent = '启动中…';
  $('start').disabled = true;

  const f = $('portrait').files[0];
  const wantSample = $('useSample').checked || !f;
  let r, j;
  if (wantSample) {
    const pick = $('samplePick').value;
    const q = pick ? '&sample=' + encodeURIComponent(pick) : '';
    r = await fetch('/api/portrait?sample=1' + q, { method: 'POST' });
    j = await r.json();
  } else {
    const data = await f.arrayBuffer();
    r = await fetch('/api/portrait?name=' + encodeURIComponent(f.name),
                    { method: 'POST', headers: {'Content-Type':'application/octet-stream'}, body: data });
    j = await r.json();
  }
  if (!j.ok) { $('err').textContent = j.error; $('start').disabled = false; return; }

  const s = { camera: +$('camera').value, region: $('region').value,
              pasteback: $('pasteback').value === '1',
              stitching: $('stitching').value === '1',
              multiplier: +$('multiplier').value,
              jpeg: +$('jpeg').value };
  r = await fetch('/api/settings', { method:'POST',
        headers: {'Content-Type':'application/json'}, body: JSON.stringify(s) });
  j = await r.json();
  if (!j.ok) { $('err').textContent = j.error; $('start').disabled = false; return; }

  mainStopped = false;

  attach($('frameOut'), 'out');
}

async function stop() {
  mainStopped = true;
  $('frameOut').innerHTML = '<div id="hint">已停止。</div>';
  await fetch('/api/stop', { method: 'POST' });
  // The server restarts the camera-only preview on stop, so re-attach the view
  // rather than blanking it -- the face can then be re-framed straight away.
  attach($('frameIn'), 'driver', { always: true });
  $('status').textContent = '已停止';
  $('start').disabled = false;
}
$('start').onclick = start;
$('stop').onclick = stop;

setInterval(async () => {
  try {
    const j = await (await fetch('/api/stats')).json();
    $('status').textContent = j.status || '';
    if (j.error) $('err').textContent = j.error;
    $('fps').textContent = j.fps ? j.fps.toFixed(1) : '-';
    $('ms').textContent = j.frame_ms ? j.frame_ms.toFixed(0) : '-';
    $('ok').textContent = j.applied;
    $('skip').textContent = j.skipped;
    const p = j.pose || {};
    $('pitch').textContent = p.pitch !== undefined ? p.pitch.toFixed(1) : '-';
    $('yaw').textContent   = p.yaw   !== undefined ? p.yaw.toFixed(1)   : '-';
    $('motion').textContent = j.motion ? j.motion.toFixed(3) : '-';
    lastStatus = j.status || '';
    if (j.error) { $('start').disabled = false; } else { refreshStart(); }
  } catch (e) {}
}, 1000);
</script>
</body></html>
"""


# A bare full-window viewer for one stream, opened by the two "new tab" buttons.
# The stream itself already exists at /api/stream; what those buttons need is a page
# that fills the window with it, and that behaves sensibly when opened while the
# engine is stopped (the composite stream produces nothing until Start is pressed, so
# the page polls /api/stats and swaps in the stream once it is actually running).
VIEWER = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · Runtime-LivePortrait</title>
<style>
  html,body { margin:0; height:100%; background:#000; overflow:hidden; }
  body { display:flex; align-items:center; justify-content:center; }
  img { max-width:100vw; max-height:100vh; object-fit:contain; display:block; }
  #wait { color:#8b93a7; font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;
          text-align:center; padding:24px; }
  #wait b { color:#cfd5e0; }
</style></head><body>
<div id="wait">__WAIT__</div>
<script>
// `flow` = the composite; the camera is always live, so only the composite waits.
var which = "__SRC__", flow = "__FLOW__" === "1";
var timer = null;

function showStream() {
  if (document.querySelector('img')) return;
  var img = new Image();
  img.src = '/api/stream?src=' + which + '&t=' + Date.now();
  document.getElementById('wait').remove();
  document.body.appendChild(img);
}

function poll() {
  fetch('/api/stats').then(function (r) { return r.json(); }).then(function (j) {
    var started = (j.status || '').indexOf('running') === 0;
    if (started) { clearInterval(timer); showStream(); }
  }).catch(function () {});
}

if (flow) { timer = setInterval(poll, 1000); poll(); } else { showStream(); }
</script>
</body></html>
"""

# titles / placeholders for the two panes the buttons open
VIEWER_TITLES = {
    "out": ("合成画面",
            "合成画面尚未开始。<br>回到主页面点 <b>开始</b>，这里会自动显示。"),
    "driver": ("摄像头（驱动源）", "正在连接摄像头…"),
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # keep the console readable
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self) -> None:                                   # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/viewer":
            # Standalone full-window view of one stream, opened by the two buttons.
            which = "driver" if "src=driver" in self.path else "out"
            if which not in VIEWER_TITLES:
                which = "out"
            title, wait = VIEWER_TITLES[which]
            body = (VIEWER
                    .replace("__TITLE__", title)
                    .replace("__WAIT__", wait)
                    .replace("__SRC__", which)
                    .replace("__FLOW__", "1" if which == "out" else "0")
                    ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/samples":
            # The bundled samples, so the page can offer them in a dropdown. Any file
            # matching `sample*` shows up here automatically.
            self._json({
                "samples": [p.name for p in sample_portraits()],
                "current": (SESSION.portrait_path.name
                            if SESSION.portrait_path else None),
            })
        elif path == "/api/stats":
            self._json({"status": SESSION.status, "error": SESSION.error,
                        "fps": SESSION.fps, "frame_ms": SESSION.frame_ms,
                        "applied": SESSION.applied, "skipped": SESSION.skipped,
                        "running": SESSION.running, "pose": SESSION.pose,
                        "motion": SESSION.motion})
        elif path == "/api/stream":
            which = "driver" if "src=driver" in self.path else "out"
            self._stream(which)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def _stream(self, which: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            # The camera thread lives for the whole app session, so stream until it is
            # told to stop -- this covers both preview and session modes, and there is
            # no mode-change race to lose frames over.
            while not SESSION.cam_stop.is_set():
                buf = SESSION.snapshot(which)
                if buf:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                    self.wfile.write(buf)
                    self.wfile.write(b"\r\n")
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    @staticmethod
    def _switch_portrait(path: Path) -> bool:
        """Point the session at `path`, stopping a live session only if it CHANGED.

        Returns True when a stop happened.

        Why the "only if it changed" guard is essential: the page's Start button sends
        the selected sample along with its request, and the sample dropdown has already
        sent it. So Start was re-applying the same portrait, this function stopped the
        session that was starting up, and the result was exactly the reported symptom --
        the button clicks, the request succeeds, and nothing ever appears.

        Stopping on a genuine change is still wanted: while the engine runs the page
        greys out Start, so without stopping, switching the source left the button
        unusable until Stop was pressed manually.

        `_SOURCE_KEY` is invalidated so the next Start really re-runs `set_source`
        instead of reusing the previous portrait's appearance features.
        """
        if path is None:
            return False
        same = (SESSION.portrait_path is not None
                and str(SESSION.portrait_path) == str(path))
        if same:
            return False
        stopped = False
        if SESSION.running or SESSION.thread is not None:
            if RECORDER is not None:
                RECORDER.stop()
            SESSION.stop()
            stopped = True
        SESSION.portrait_path = path
        Session._SOURCE_KEY = None
        return stopped

    def do_POST(self) -> None:                                  # noqa: N802
        path = self.path.split("?")[0]
        query = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&")) \
            if "?" in self.path else {}

        if path == "/api/portrait":
            if "sample" in query:
                # Pick a bundled sample. `?sample=1` (or no name) keeps the default;
                # `?sample=sample3.PNG` selects a specific one. Only names returned by
                # sample_portraits() are accepted, so a crafted name cannot reach
                # outside portraits/ or pick a non-sample image.
                want = (query.get("sample") or "").strip()
                avail = sample_portraits()
                if not avail:
                    return self._json({"ok": False,
                                       "error": "no bundled sample portrait found"})
                chosen = None
                if want and want != "1":
                    for p in avail:
                        if p.name.lower() == want.lower():
                            chosen = p
                            break
                    if chosen is None:
                        return self._json({"ok": False,
                                           "error": f"unknown sample: {want}"})
                else:
                    chosen = DEFAULT_PORTRAIT if DEFAULT_PORTRAIT in avail else avail[0]
                stopped = self._switch_portrait(chosen)
                return self._json({"ok": True, "portrait": chosen.name,
                                   "session_stopped": stopped})
            raw = self._body()
            name = query.get("name", "portrait.jpg")
            PORTRAITS.mkdir(parents=True, exist_ok=True)
            suffix = Path(name).suffix.lower() or ".jpg"
            if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                suffix = ".jpg"
            dest = PORTRAITS / ("uploaded" + suffix)
            dest.write_bytes(raw)
            img = imread_unicode(dest)
            if img is None:
                return self._json({"ok": False, "error": "that file is not a readable image"})
            stopped = self._switch_portrait(dest)
            return self._json({"ok": True, "portrait": dest.name,
                               "size": f"{img.shape[1]}x{img.shape[0]}",
                               "session_stopped": stopped})

        if path == "/api/settings":
            try:
                s = json.loads(self._body() or b"{}")
            except Exception:
                return self._json({"ok": False, "error": "bad json"})
            SESSION.settings.update({k: s[k] for k in s if k in SESSION.settings})
            SESSION.applied = SESSION.skipped = 0
            SESSION.start()
            for _ in range(60):        # let obvious failures surface in the UI
                if SESSION.error or SESSION.status == "running":
                    break
                time.sleep(0.1)
            if SESSION.error:
                return self._json({"ok": False, "error": SESSION.error})
            return self._json({"ok": True})

        if path == "/api/snapshot":
            return self._json(SESSION.save_snapshot())

        if path == "/api/record":
            # The same endpoint starts and stops, matching the single toggle button.
            global RECORDER
            if RECORDER is not None:
                info = RECORDER.stop()
                RECORDER = None
                info["ok"] = not info.get("error")
                info["recording"] = False
                info["dir"] = str(OUTPUT)
                return self._json(info)
            if SESSION.out_frame() is None:
                return self._json({"ok": False, "recording": False,
                                   "error": "no composite frame yet - press Start first"})
            OUTPUT.mkdir(parents=True, exist_ok=True)
            path = OUTPUT / f"recording_{_stamp()}.avi"
            n = 1
            while path.exists():
                path = OUTPUT / f"recording_{_stamp()}_{n}.avi"
                n += 1
            rec = Recorder(path, SESSION.out_frame, fps=25)
            if not rec.start():
                RECORDER = None
                return self._json({"ok": False, "recording": False,
                                   "error": rec.error or "could not start recording"})
            RECORDER = rec
            return self._json({"ok": True, "recording": True, "file": path.name,
                               "dir": str(OUTPUT)})

        if path == "/api/stop":
            # stop() falls the single camera thread back to preview mode, so the
            # webcam keeps showing and the face can be re-framed.
            if RECORDER is not None:            # never leave a recording half-written
                RECORDER.stop()
                RECORDER = None
            SESSION.stop()
            return self._json({"ok": True})

        self._json({"ok": False, "error": "not found"}, 404)


def free_port(preferred: int) -> int:
    for p in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def _install_close_cleanup() -> None:
    """Run the privacy cleanup on console signals, and make Ctrl+C actually quit.

    This went through three measured revisions, because the obvious implementations
    are each wrong in a different way:

    1. Plain `try/finally` around serve_forever(): Ctrl+C terminated the process with
       0xC000013A (STATUS_CONTROL_C_EXIT) and the `finally` NEVER ran -- the uploaded
       photo was still on disk. So a handler really is required.

    2. Registering a pywin32 handler that returns True: this told Windows "handled",
       which suppressed termination -- **Ctrl+C stopped working entirely** and the
       only way out was Task Manager.

    3. Registering a handler that returns False for CTRL_C/BREAK: still broke it.
       Measured: with ANY pywin32 handler registered, Python's KeyboardInterrupt
       never fires and the process dies with 0xC000013A anyway. (Verified against a
       no-handler baseline, which behaves the same way when stdin/stdout are not an
       interactive console.)

    So the handler cleans up itself and then exits cleanly with `os._exit(0)`
    (returncode 0, measured). Raising is not an option from a console control
    handler, and doing the work here also covers CTRL_CLOSE_EVENT -- closing the
    console window -- which Python can never turn into an exception.

    `cleanup_private_files()` is idempotent and guarded by `_CLEANUP_DONE`, so it
    does not matter that the normal path may also have called it.
    """
    global _CLEANUP_DONE

    def _handler(event):
        global _CLEANUP_DONE
        try:
            if not _CLEANUP_DONE:
                _CLEANUP_DONE = True
                cleanup_private_files()
                # Only safe to print for the close/logoff events; these
                # 0/1 = CTRL_C / CTRL_BREAK, 2 = CTRL_CLOSE, 5/6 = logoff/shutdown.
                if event not in (0, 1):
                    print("\n  privacy cleanup on close done.", flush=True)
        except Exception:
            pass
        # Exit cleanly. Without this the process dies with 0xC000013A and, more
        # importantly, a returning-True handler would leave it running forever.
        os._exit(0)

    try:
        import win32api                      # type: ignore
        win32api.SetConsoleCtrlHandler(_handler, True)
        return
    except Exception:
        pass

    # pywin32 not available: atexit is weaker (it cannot run on a hard close) but it
    # still covers an ordinary interpreter shutdown.
    import atexit

    def _atexit():
        global _CLEANUP_DONE
        if not _CLEANUP_DONE:
            _CLEANUP_DONE = True
            cleanup_private_files()

    atexit.register(_atexit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    SESSION.settings["camera"] = args.camera
    # so the Start button works immediately, without choosing a file
    for cand in (DEFAULT_PORTRAIT, SAMPLE):
        if cand.exists():
            SESSION.portrait_path = cand
            break

    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True

    print("=" * 62)
    print("  LivePortrait-Runtime  -  web UI")
    print("=" * 62)
    print(f"  open: {url}")
    print(f"  portrait default: {SESSION.portrait_path.name if SESSION.portrait_path else 'none'}")
    print("  Ctrl+C here to quit (closing this window also works)")
    print("  closing this window removes your uploaded photos and temp frames")
    print()
    # Sweep anything a previous crash left behind (the shutdown path below does not
    # run if the process is killed).
    cleanup_private_files()
    _install_close_cleanup()
    # Show the camera right away, before Start is pressed, so the face can be
    # framed up front. Camera-only: the models are not loaded until Start.
    SESSION._ensure_camera()
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        SESSION.stop()
        SESSION.stop_camera()
        srv.server_close()
        # Privacy: do not leave the user's own photos or captured frames on disk.
        cleanup_private_files(verbose=True)
        print("  your uploaded photos and temp frames were removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
