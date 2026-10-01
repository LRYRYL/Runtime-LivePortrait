"""Per-stage timing for a frame, so the slow part can be identified.

The summary in speed_probe.py reports one total. This prints the breakdown the pipeline
already records, plus the CPU-side costs that sit between the stages, so it is possible
to tell apart:

  * GPU stages that are simply slow on this card
  * CPU work that leaves the GPU idle (which is what shows up as low GPU utilisation)

Run from the project folder:
    .venv\\Scripts\\python.exe tools\\stage_timing.py
or, in a portable package:
    _runtime_cu128\\python.exe tools\\stage_timing.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 66)
print("  Runtime-LivePortrait - per-stage timing")
print("=" * 66)

import cv2                                                          # noqa: E402
import torch                                                        # noqa: E402
from fastportrait.realtime import FastConfig, LivePortraitFast      # noqa: E402

portrait = next((ROOT / "portraits" / n for n in
                 ("sample.jpg", "sample2.jpg", "sample3.PNG")
                 if (ROOT / "portraits" / n).is_file()), None)
if portrait is None:
    raise SystemExit("no portrait found in portraits/")
src = cv2.imdecode(np.fromfile(str(portrait), np.uint8), cv2.IMREAD_COLOR)

print(f"\n  portrait : {portrait.name}")
print(f"  torch    : {torch.__version__}   cuda {torch.version.cuda}")
print(f"  device   : {torch.cuda.get_device_name(0)}")

eng = LivePortraitFast(FastConfig(driving_multiplier=0.7))
t0 = time.perf_counter()
ok = eng.set_source(src)
print(f"  set_source: {ok}  ({time.perf_counter()-t0:.1f}s)")

for _ in range(8):                      # warm up so the first frames are not measured
    eng.process(src)
torch.cuda.synchronize()

N = 40
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    eng.process(src)
torch.cuda.synchronize()
total = (time.perf_counter() - t0) / N * 1000

print(f"\n  total per frame : {total:.1f} ms   ->  {1000/total:.1f} FPS")

# the pipeline keeps a running breakdown; average it over fresh frames
print("\n  per-stage breakdown (averaged over the measured frames):")
try:
    acc: dict[str, float] = {}
    for _ in range(N):
        eng.process(src)
        rep = eng.last            # the pipeline stores the latest FrameStats here
        for key in ("detect_ms", "crop_ms", "motion_ms", "warp_ms", "paste_ms",
                    "total_ms"):
            acc[key] = acc.get(key, 0.0) + float(getattr(rep, key, 0.0))
    named = {
        "detect_ms": "face detect (onnxruntime)",
        "crop_ms": "crop / align",
        "motion_ms": "motion extractor",
        "warp_ms": "warp + SPADE generate",
        "paste_ms": "paste back",
    }
    s = 0.0
    for key, label in named.items():
        v = acc[key] / N
        s += v
        pct = v / total * 100
        bar = "#" * max(1, int(pct / 2))
        print(f"    {label:<28} {v:6.1f} ms  {pct:5.1f}%  {bar}")
    print(f"    {'sum of stages':<28} {s:6.1f} ms  {s/total*100:5.1f}%")
    print(f"    {'unaccounted (overhead)':<28} {total-s:6.1f} ms"
          f"  {(total-s)/total*100:5.1f}%")
except Exception as exc:                                            # noqa: BLE001
    print(f"    breakdown unavailable: {exc}")

# CPU-side costs, measured independently of the model
print("\n  CPU-side costs (independent of the GPU):")
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
for q in (80, 60, 40):
    t = time.perf_counter()
    for _ in range(20):
        cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    print(f"    jpeg encode 1280x720 q{q:<3}  {(time.perf_counter()-t)/20*1000:6.2f} ms")
frame_small = cv2.resize(frame, (512, 512))
t = time.perf_counter()
for _ in range(20):
    cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(f"    jpeg encode  512x512 q80   {(time.perf_counter()-t)/20*1000:6.2f} ms")

try:
    import onnxruntime as ort
    print(f"\n  onnxruntime {ort.__version__}   providers: {ort.get_available_providers()}")
except Exception:                                                   # noqa: BLE001
    pass
