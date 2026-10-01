"""Verify the installed torch can actually target this machine's GPU.

Run after installing, given the stack that was installed. Answers the question the old
check could not: `torch.cuda.is_available()` returns True in some broken situations, and
more importantly a stack that cannot target the GPU's architecture fails only later,
with a warning the user has to interpret:

    UserWarning: NVIDIA GeForce RTX 5070 Laptop GPU with CUDA capability sm_120 is not
    compatible with the current PyTorch installation.
    The current PyTorch install supports CUDA capabilities sm_37 ... sm_90

This turns that into an explicit failure that names the fix.

    python gpu_check_installed.py --expect cu118
Exit status 0 when the installed build covers this GPU, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", default="", help="the stack that was installed")
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:                                    # noqa: BLE001
        print(f"FAIL|torch could not be imported: {exc}")
        return 1

    version = torch.__version__
    cuda_ver = torch.version.cuda or "none"
    available = torch.cuda.is_available()
    arch = list(torch.cuda.get_arch_list())

    if not available:
        print(f"FAIL|torch {version} (CUDA {cuda_ver}) reports no usable CUDA device")
        return 1

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    mine = cap[0] * 10 + cap[1]

    cudins = sorted({int(a.split("_")[1]) for a in arch if a.startswith("sm_")})
    best = max(cudins) if cudins else 0

    # A build is usable when its highest compiled cubin is at least this card's
    # capability: either there is one for exactly this architecture, or one for a lower
    # architecture that the driver runs as-is. When every cubin is older than the card,
    # torch emits the "not compatible with the current PyTorch installation" warning,
    # and for a card this new that means the kernels simply were not compiled for it.
    if best >= mine:
        if mine in cudins:
            print(f"OK|{name} {sm} has a native cubin in torch {version}")
        else:
            lower = max(c for c in cudins if c <= mine)
            print(f"OK|{name} {sm} runs on the sm_{lower} cubin"
                  f" (torch {version}, no exact cubin; normal and expected)")
        return 0

    print(
        f"FAIL|torch {version} (CUDA {cuda_ver}) cannot target {name} {sm}: every kernel "
        f"in this build is compiled for sm_{best} or lower."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
