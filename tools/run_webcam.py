"""Live webcam driver: animate a source portrait with your camera in real time.

Usage (from Runtime-LivePortrait/):
    <lpv python> tools\\run_webcam.py --source <portrait.jpg> [--seconds 10]

Keys while running: q / ESC quits, s saves a frame.

Why this is a separate tool from verify_speed.py: that one measures the per-frame
cost on a fixed pair of images, which is the right way to compare optimisations.
This one runs the actual camera loop, so it also exposes the costs that only appear
live -- capture rate, detection on real poses, and end-to-end latency.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Captured frames can contain the user's face, so they go to a cache dir the
# app wipes on close rather than into docs/ next to the documentation images.
_CACHE = ROOT / ".cache"
_CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def read_unicode(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="portrait to animate")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until quit")
    ap.add_argument("--display", action="store_true", help="show a window")
    ap.add_argument("--save-frames", type=int, default=3)
    args = ap.parse_args()

    docs = ROOT / "portraits"
    source = Path(args.source) if args.source else None
    if source is None:
        c = docs / "sample.jpg"
        source = c if c.exists() else None
    if source is None or not source.exists():
        print("no source portrait; pass --source <file>")
        return 2

    from fastportrait.realtime import FastConfig, LivePortraitFast

    print("=" * 74)
    print(" Runtime-LivePortrait -- live webcam")
    print("=" * 74)
    print(f"source: {source.name}")

    t0 = time.perf_counter()
    eng = LivePortraitFast(FastConfig())
    if not eng.set_source(read_unicode(source)):
        print("RESULT: FAIL (no face in the source portrait)")
        return 1
    print(f"source prepared in {time.perf_counter()-t0:.2f}s on {eng.device}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"cannot open camera {args.camera}")
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"camera {args.camera} at "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    for _ in range(8):          # DSHOW's first frames are often black
        cap.read()
        time.sleep(0.02)
    eng.warmup(runs=2)

    times, saved, n, noface = [], 0, 0, 0
    t_end = time.time() + args.seconds if args.seconds > 0 else None
    print("running (q/ESC to quit) ..." if args.display else "running ...")
    try:
        while True:
            if t_end is not None and time.time() > t_end:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            n += 1
            t = time.perf_counter()
            out = eng.process(frame)
            dt = (time.perf_counter() - t) * 1000
            if eng.last.applied:
                times.append(dt)
            else:
                # A frame with no face returns the input untouched and costs almost
                # nothing, so including it would report a flattering and meaningless
                # frame rate. Only applied frames count.
                noface += 1

            if n % 30 == 0 and times:
                recent = times[-30:]
                print(f"  frame {n:5d}  {np.mean(recent):6.1f} ms  "
                      f"({1000/np.mean(recent):5.1f} FPS)  "
                      f"missed {noface}/{n}")

            if saved < args.save_frames and n % 25 == 0:
                cv2.imwrite(str(_CACHE / f"lpv_webcam_{saved:02d}.png"),
                            np.hstack([cv2.resize(frame, (out.shape[1], out.shape[0])), out]))
                saved += 1

            if args.display:
                cv2.imshow("Runtime-LivePortrait", out)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("s"):
                    cv2.imwrite(str(_CACHE / f"lpv_snap_{int(time.time())}.png"), out)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    if not times:
        print("no frames processed")
        return 3
    a = np.array(times)
    med = float(np.median(a))
    print("-" * 74)
    print(f"frames processed : {len(a)}")
    print(f"face missed      : {noface}/{n}")
    print(f"per-frame median : {med:.1f} ms -> {1000/med:.1f} FPS")
    print(f"mean / p90       : {a.mean():.1f} / {np.percentile(a,90):.1f} ms")
    print(f"VRAM peak        : {torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
          if torch.cuda.is_available() else "")
    print(f"saved {saved} frame pairs to {_CACHE}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
