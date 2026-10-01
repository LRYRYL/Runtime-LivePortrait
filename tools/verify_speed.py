"""Measure LivePortrait's real per-frame cost on this machine.

The point of Runtime-LivePortrait is speed, so this reports measured numbers rather
than repeating the paper's 4090 figures. It separates the two very different costs:

  source preparation  once per portrait (crop + appearance feature + keypoints)
  per-frame           crop + motion + warp/SPADE decode + paste-back

and reports percentiles, because a live session is judged by its slow frames, not
its mean. Also reports VRAM and, if a webcam frame is available, writes a
before/after image so quality can be judged rather than asserted.

Run from Runtime-LivePortrait/ with the lpv env:
    <lpv python> tools\\verify_speed.py --source <portrait.jpg>
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def read_unicode(p: Path):
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="portrait to animate")
    ap.add_argument("--driving", default=None, help="frame to drive it with")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--compile", action="store_true", help="enable torch.compile")
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--region", default="all",
                    choices=["all", "exp", "pose", "lip", "eyes"])
    args = ap.parse_args()

    docs = ROOT / "docs"
    # Self-contained defaults: the bundled synthetic portrait drives itself, so this
    # tool no longer depends on any archived folder or on a real person's photo.
    source = Path(args.source) if args.source else None
    if source is None:
        cands = [ROOT / "portraits" / "sample.jpg", ROOT / "portraits" / "my_face.jpg"]
        source = next((c for c in cands if c.exists()), None)
    if source is None or not source.exists():
        print("no source portrait available")
        return 2
    driving = Path(args.driving) if args.driving else None
    if driving is None:
        driving = source          # drive the portrait with itself: a clean self-test
    if not driving.exists():
        print("no driving frame available")
        return 2

    print("=" * 78)
    print(" LivePortrait speed verification (Runtime-LivePortrait)")
    print("=" * 78)
    print(f"source : {source.name}")
    print(f"driving: {driving.name}")

    from fastportrait.realtime import FastConfig, LivePortraitFast

    cfg = FastConfig(half=not args.no_half, torch_compile=args.compile,
                     animation_region=args.region)
    print(f"config : half={cfg.half} compile={cfg.torch_compile} "
          f"region={cfg.animation_region}")

    t0 = time.perf_counter()
    eng = LivePortraitFast(cfg)
    init_s = time.perf_counter() - t0
    print(f"\n[1] model load            : {init_s:6.2f} s")
    print(f"    device                : {eng.device}")
    if torch.cuda.is_available():
        print(f"    gpu                   : {torch.cuda.get_device_name(0)}")

    src = read_unicode(source)
    drv = read_unicode(driving)
    print(f"    source {src.shape[1]}x{src.shape[0]}  "
          f"driving {drv.shape[1]}x{drv.shape[0]}")

    t0 = time.perf_counter()
    ok = eng.set_source(src)
    prep_s = time.perf_counter() - t0
    if not ok:
        print("\nRESULT: FAIL (no face in the source portrait)")
        return 1
    print(f"\n[2] source preparation    : {prep_s:6.3f} s  (once per portrait)")

    # one warm pass, then measure
    eng.process(drv)
    times, stages = [], []
    for _ in range(args.frames):
        t = time.perf_counter()
        out = eng.process(drv)
        times.append((time.perf_counter() - t) * 1000)
        stages.append(eng.last)

    a = np.array(times)
    med = float(np.median(a))
    print(f"\n[3] per-frame (n={args.frames})")
    print(f"    median                : {med:6.2f} ms  -> {1000/med:5.1f} FPS")
    print(f"    mean                  : {a.mean():6.2f} ms  -> {1000/a.mean():5.1f} FPS")
    print(f"    p90 / p99             : {np.percentile(a,90):6.2f} / "
          f"{np.percentile(a,99):6.2f} ms")
    print(f"    min / max             : {a.min():6.2f} / {a.max():6.2f} ms")

    def avg(attr):
        return statistics.mean(getattr(s, attr) for s in stages)

    print(f"\n[4] stage breakdown (mean)")
    for attr, label in (("detect_ms", "detection"), ("crop_ms", "crop"),
                        ("motion_ms", "motion extract"),
                        ("warp_ms", "warp + SPADE decode"),
                        ("paste_ms", "paste back")):
        v = avg(attr)
        print(f"    {label:22s}: {v:6.2f} ms  ({100*v/med:4.1f}% of frame)")
    print(f"    {'measured total':22s}: {med:6.2f} ms")

    print(f"\n[5] VRAM peak             : {vram_gb():5.2f} GB")

    # quality: before/after so it can be judged, not asserted
    #
    # The two images are NOT the same size: the driving frame is whatever the camera
    # gives us (1280x720) while the animated output is the 512x512 source portrait.
    # They must be matched before hstacking, or numpy raises "all the input array
    # dimensions except for the concatenation axis must match exactly".
    h = out.shape[0]
    w = max(1, int(drv.shape[1] * h / drv.shape[0]))
    side = np.hstack([cv2.resize(drv, (w, h), interpolation=cv2.INTER_AREA), out])
    dst = ROOT / "docs"
    dst.mkdir(parents=True, exist_ok=True)
    outp = dst / "lpv_realtime.png"
    cv2.imwrite(str(outp), side)
    print(f"\n[6] wrote {outp}  (left = driving frame, right = animated)")

    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
