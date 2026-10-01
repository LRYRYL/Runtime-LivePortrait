"""Run the full capture -> bridge -> animate path on a video file.

Why this exists
---------------
`run_webcam.py` needs a face in front of the camera to report a real frame rate;
when the lens pointed at an empty room it reported a meaningless number (every frame
was skipped). This tool removes that dependency: it feeds a *file* through exactly
the same path the live host uses -- JPEG frames over the subprocess bridge -- so the
end-to-end pipeline (capture, encode, IPC, decode, animate) can be verified and
timed deterministically, with no one sitting in front of the camera.

It also reports the bridge overhead separately from the animation, which is the
number that decides whether the out-of-process integration is viable.

Usage:
    <lpv python> tools/run_video.py --video clip.mp4 [--limit 120] [--in-process]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file to use as the driver")
    ap.add_argument("--source", default=None, help="portrait to animate")
    ap.add_argument("--limit", type=int, default=120, help="frames to process")
    ap.add_argument("--in-process", action="store_true",
                    help="skip the bridge; measure the model path alone")
    ap.add_argument("--save", default=None, help="write an mp4 of the result")
    args = ap.parse_args()

    docs = ROOT / "portraits"
    source = Path(args.source) if args.source else None
    if source is None:
        c = docs / "sample.jpg"
        source = c if c.exists() else None
    if source is None or not source.exists():
        print("no source portrait; pass --source")
        return 2

    video = Path(args.video)
    if not video.exists():
        print(f"no such video: {video}")
        return 2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open {video}")
        return 2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("=" * 74)
    print(" Runtime-LivePortrait -- video-driven end-to-end run")
    print("=" * 74)
    print(f"source : {source.name}")
    print(f"video  : {video.name}  ({total} frames)")
    print(f"mode   : {'in-process' if args.in_process else 'subprocess bridge'}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 25.0, (512, 512))

    times, applied, skipped, produced = [], 0, 0, None

    if args.in_process:
        from fastportrait.realtime import FastConfig, LivePortraitFast
        eng = LivePortraitFast(FastConfig())
        if not eng.set_source(cv2.imdecode(np.fromfile(str(source), np.uint8),
                                          cv2.IMREAD_COLOR)):
            print("no face in source")
            return 1
    else:
        from frame_client import FrameWorker
        w = FrameWorker(source)
        if not w.start():
            print(f"worker failed: {w.error}")
            return 1
        eng = w

    n = 0
    try:
        while n < args.limit:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            t = time.perf_counter()
            out = eng.process(frame) if args.in_process else eng.process(frame)
            dt = (time.perf_counter() - t) * 1000
            if out is None:
                print(f"  frame {n}: pipeline failed")
                break
            # A skipped frame is returned unchanged (same object in-process, same
            # pixels over the bridge), so detect it by comparing shapes: the
            # animated output is the 512x512 source portrait.
            animated = out.shape[0] == 512 and out.shape[1] == 512
            if animated:
                applied += 1
                times.append(dt)
                produced = out
            else:
                skipped += 1
            if writer is not None and produced is not None:
                writer.write(produced if animated else cv2.resize(out, (512, 512)))
            if n % 30 == 0:
                med = float(np.median(times[-30:])) if times else 0.0
                msg = f"{1000/med:5.1f} FPS" if med else "no animated frame yet"
                print(f"  frame {n:4d}  {med:6.1f} ms  {msg}  "
                      f"(applied {applied}, skipped {skipped})")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.in_process:
            eng.stop()

    print("-" * 74)
    print(f"frames read      : {n}")
    print(f"animated         : {applied}")
    print(f"skipped          : {skipped}")
    if times:
        a = np.array(times)
        print(f"per-frame median : {np.median(a):.1f} ms -> {1000/np.median(a):.1f} FPS")
        print(f"p90              : {np.percentile(a,90):.1f} ms")
    if args.save:
        print(f"wrote {args.save}")
    print("RESULT: PASS" if times else "RESULT: FAIL (nothing animated)")
    return 0 if times else 1


if __name__ == "__main__":
    raise SystemExit(main())
