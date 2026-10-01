"""Decide which CUDA stack this machine's GPU needs.

Why this exists
---------------
No single PyTorch build covers every NVIDIA architecture well:

    cu118   sm_50..sm_90   native sm_86 / sm_89   cannot target sm_120 at all
    cu128   sm_50..sm_120  native sm_86 / sm_120  no sm_89 cubin (RTX 40 pays ~12%)

So the CUDA 11.8 stack is the fastest choice for an RTX 20/30/40 card, while an RTX 50
card (Blackwell, sm_120) can only run on cu128 -- under cu118 it fails outright with

    CUDA capability sm_120 is not compatible with the current PyTorch installation

This module reads the GPU's compute capability and returns the right stack name, so the
installer and the wheel downloader always agree.

Used both as a library (`choose_cuda()`) and as a command (`python gpu_choice.py`) so
PowerShell can call it without duplicating the rule.
"""
from __future__ import annotations

import subprocess
import sys


def query_compute_caps() -> list[tuple[int, int, str]]:
    """Returns [(major, minor, name), ...] for every NVIDIA GPU, or [] if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    found = []
    for line in (out.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        cap = parts[1]
        if "." not in cap:
            continue
        try:
            major, minor = (int(x) for x in cap.split(".", 1))
        except ValueError:
            continue
        found.append((major, minor, name))
    return found


def choose_cuda(caps: list[tuple[int, int, str]]) -> tuple[str, str]:
    """Pick the CUDA stack and explain why.

    Rule: cu118 for anything at or below sm_89, cu128 for sm_90 and above.
    sm_89 (RTX 40) deliberately stays on cu118 -- cu128 has no sm_89 cubin, so it would
    fall back to the sm_86 binary and measure about 12% slower.
    """
    if not caps:
        return "cu118", "no NVIDIA GPU detected; defaulting to cu118"

    # the card that matters is the weakest one we must support
    major, minor = min((c[0], c[1]) for c in caps)
    names = ", ".join(c[2] for c in caps)

    if major < 9:
        return "cu118", f"sm_{major}{minor} ({names}) -> cu118"
    return "cu128", f"sm_{major}{minor} ({names}) -> cu128 (CUDA 11.8 cannot target this)"


def main() -> int:
    caps = query_compute_caps()
    stack, reason = choose_cuda(caps)
    # command mode: print just the stack, so a shell can capture it
    if len(sys.argv) > 1 and sys.argv[1] == "--name":
        print(stack)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--why":
        print(reason)
        return 0
    for major, minor, name in caps:
        print(f"  gpu        : {name}  sm_{major}{minor}")
    if not caps:
        print("  gpu        : none detected")
    print(f"  cuda stack : {stack}")
    print(f"  reason     : {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
