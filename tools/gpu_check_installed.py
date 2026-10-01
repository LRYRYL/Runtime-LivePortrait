"""Verify the installed torch can actually target this machine's GPU.

`torch.cuda.is_available()` is not enough. On a card whose architecture the build was
never compiled for, torch still reports CUDA as available and only warns at run time:

    UserWarning: NVIDIA GeForce RTX 5070 Laptop GPU with CUDA capability sm_120 is not
    compatible with the current PyTorch installation.
    The current PyTorch install supports CUDA capabilities sm_37 ... sm_90

This turns that into an explicit pass or fail. The rule itself lives in gpu_compat.py so
that this, check_env.py and the installer all agree.

    python gpu_check_installed.py --expect cu118

Exit status 0 when the installed build covers this GPU, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_compat import evaluate                                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", default="", help="the stack that was installed")
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:                                                # noqa: BLE001
        print(f"FAIL|torch could not be imported: {exc}")
        return 1

    version = torch.__version__
    cuda_ver = torch.version.cuda or "none"

    if not torch.cuda.is_available():
        print(f"FAIL|torch {version} (CUDA {cuda_ver}) reports no usable CUDA device")
        return 1

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    usable, reason, detail = evaluate(list(torch.cuda.get_arch_list()), cap)

    if usable:
        # "no exact cubin" is worth saying out loud, because torch prints a warning for
        # it even though it is harmless. Without this the warning looks like a problem.
        suffix = "" if reason == "native" else " (torch warns about this; it is harmless)"
        print(f"OK|{name} sm_{cap[0]}{cap[1]}: {detail}{suffix}  [torch {version}]")
        return 0

    print(
        f"FAIL|torch {version} (CUDA {cuda_ver}) cannot target {name} "
        f"sm_{cap[0]}{cap[1]}: {detail}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
