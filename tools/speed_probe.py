"""Per-stage speed probe.

Answers "is this machine slow because of the GPU, or because of something fixable".

It reports:
  * which GPU, which runtime, whether CUDA is actually active
  * whether onnxruntime landed on the CUDA provider or silently fell back to CPU
    (a CPU fallback is a very common cause of a large, unexpected slowdown)
  * per-stage timings for the engine, so the slow part is identifiable
  * GPU clocks and power limit, since a laptop on battery or in a quiet fan profile
    runs far below its rated speed

Usage (from the LivePortrait-Runtime folder):
    ..\\_runtime_cu128\\python.exe tools\\speed_probe.py
    ..\\_runtime\\python.exe      tools\\speed_probe.py
Or just run it without arguments to use whichever python starts it.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 66)
print("  Runtime-LivePortrait - speed probe")
print("=" * 66)

# ---------------------------------------------------------------- 1) machine
print("\n[1] machine")
print(f"    python      : {sys.version.split()[0]}  ({sys.executable})")
print(f"    platform    : {platform.platform()}")
try:
    import torch
    print(f"    torch       : {torch.__version__}   built for CUDA {torch.version.cuda}")
    print(f"    arch_list   : {torch.cuda.get_arch_list()}")
    print(f"    cuda_ok     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"    device      : {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
        props = torch.cuda.get_device_properties(0)
        print(f"    memory      : {props.total_memory/1024**3:.1f} GB")
        # A build rarely carries a cubin for every possible card. When the exact one is
        # missing the driver uses the nearest lower cubin or JIT-compiles from PTX, which
        # is normal and often costs nothing measurable -- the reference machine runs its
        # fastest runtime without an exact match. So report it, do not call it a fault.
        exact = f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()
        below = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
        lower = [a for a in below
                 if int(a.split("_")[1]) // 10 == cap[0]
                 and int(a.split("_")[1]) % 10 <= cap[1]]
        note = "" if exact else "   (nearest lower: %s; normal)" % (
            lower[-1] if lower else "none")
        print("    exact cubin for this card  : %s%s" % (exact, note))
except Exception as exc:
    print(f"    torch FAILED: {exc}")

# ------------------------------------------------------- 2) nvidia-smi detail
print("\n[2] GPU state (clocks / power)")
try:
    q = ("name,driver_version,clocks.sm,clocks.max.sm,power.draw,power.limit,"
         "temperature.gpu,utilization.gpu")
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"],
                         capture_output=True, text=True, timeout=20)
    for line in (out.stdout or "").strip().splitlines():
        print("    " + line.strip())
    print("    (clocks.sm well below clocks.max.sm means it is throttling or capped)")
except Exception as exc:
    print(f"    nvidia-smi unavailable: {exc}")

# --------------------------------------------------- 3) onnxruntime provider
print("\n[3] onnxruntime execution provider")
try:
    import onnxruntime as ort
    print(f"    onnxruntime : {ort.__version__}")
    print(f"    available   : {ort.get_available_providers()}")
    import tempfile
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])])
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    tmp = Path(tempfile.gettempdir()) / "_lpr_probe.onnx"
    onnx.save(model, tmp)
    sess = ort.InferenceSession(str(tmp),
                                providers=["CUDAExecutionProvider",
                                           "CPUExecutionProvider"])
    active = sess.get_providers()
    print(f"    ACTIVE      : {active}")
    if "CUDAExecutionProvider" not in active:
        print("    *** onnxruntime fell back to CPU - this is a major slowdown ***")
    tmp.unlink(missing_ok=True)
except Exception as exc:
    print(f"    onnxruntime probe failed: {exc}")

# ------------------------------------------------------------ 4) engine rate
print("\n[4] engine throughput")
try:
    import cv2
    import numpy as np
    from fastportrait.realtime import FastConfig, LivePortraitFast

    portrait = next((ROOT / "portraits" / n for n in
                     ("sample.jpg", "sample2.jpg", "sample3.PNG")
                     if (ROOT / "portraits" / n).is_file()), None)
    src = cv2.imdecode(np.fromfile(str(portrait), np.uint8), cv2.IMREAD_COLOR)
    print(f"    portrait    : {portrait.name}")

    t0 = time.perf_counter()
    eng = LivePortraitFast(FastConfig(driving_multiplier=0.7))
    ok = eng.set_source(src)
    print(f"    set_source  : {ok}   ({time.perf_counter()-t0:.1f}s)")

    for _ in range(5):                      # warm-up
        eng.process(src)

    ts = []
    for _ in range(40):
        t = time.perf_counter()
        eng.process(src)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"\n    frames      : {len(ts)}")
    print(f"    median      : {med:.1f} ms   ->  {1000/med:.1f} FPS")
    print(f"    min / p10   : {ts[0]:.1f} / {ts[len(ts)//10]:.1f} ms")
    print(f"    p90 / max   : {ts[len(ts)*9//10]:.1f} / {ts[-1]:.1f} ms")
    if ts[-1] > med * 2:
        print("    NOTE: max is far above median - something is stalling "
              "periodically (thermal, power, or another process)")
except Exception as exc:
    import traceback
    print(f"    engine probe failed: {exc}")
    traceback.print_exc()

# ------------------------------------------------------- 5) torch op benchmark
print("\n[5] raw torch benchmark (isolates GPU vs pipeline)")
try:
    import torch
    if torch.cuda.is_available():
        for shape, label in (((512, 512, 3), "512x512x3"),
                             ((256, 256, 64), "256x256x64 conv-ish")):
            a = torch.randn(1, shape[2] if len(shape) == 3 else shape[2],
                            shape[0], shape[1], device="cuda")
            w = torch.randn(shape[2] if len(shape) == 3 else shape[2],
                            shape[2] if len(shape) == 3 else shape[2], 3, 3,
                            device="cuda")
            for _ in range(3):
                torch.nn.functional.conv2d(a, w, padding=1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                torch.nn.functional.conv2d(a, w, padding=1)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1000
            print(f"    conv2d {label:<22} {ms:7.2f} ms")
        # a pure memory-bandwidth test, since these models are bandwidth-heavy
        x = torch.randn(4096, 4096, device="cuda")
        for _ in range(3):
            y = x * 2.0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y = x * 2.0
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1000
        gb = 4096 * 4096 * 4 * 2 / 1024 ** 3        # read + write
        print(f"    elementwise 4096x4096  {ms:7.2f} ms  "
              f"({gb / (ms/1000):.0f} GB/s effective)")
except Exception as exc:
    print(f"    torch benchmark failed: {exc}")

print("\n" + "=" * 66)
print("  Reference: RTX 4080 Laptop, cu118 runtime -> 82.5 ms / 12.1 FPS")
print("             RTX 4080 Laptop, cu128 runtime -> 92.5 ms / 10.8 FPS")
print("=" * 66)
