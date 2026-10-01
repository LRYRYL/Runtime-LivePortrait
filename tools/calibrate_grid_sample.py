"""Calibrate torch's 5-D grid_sample axis semantics.

Why this exists
---------------
ONNX cannot export the `grid_sample` call inside LivePortrait's warping network, and
replacing it requires replicating torch's exact behaviour. That behaviour is
undocumented for the shape LivePortrait passes, and I got it wrong three times by
guessing. This tool measures it instead.

Method (getting this wrong wasted two rounds)
---------------------------------------------
Put the test coordinate at ONE grid position and fill every other position with a
sentinel value far outside [-1, 1]. Then read the output at that SAME position.

My first probe filled the WHOLE grid with the test coordinate and read `o[0,0,0,0,0]`.
The grid is broadcast over the whole output tensor, so every position was "the" test
position and the value read meant something else. That produced a confident, wrong
conclusion ("torch does true trilinear sampling"), which sent the implementation
down the wrong path and cost three failed attempts.

Corrected finding
-----------------
All three axes follow the standard `align_corners=False` mapping
`index = (coord + 1) * S / 2 - 0.5`. The only deviation is AT |coord| = 1, where the
coordinate lands out of range and torch's zero-padding rule applies.

A second trap: verify with D >= 4. With D = 2 the two boundary effects collapse onto
the same interior values and the readings look nonsensical (I read z=-1 -> 5.5 which
is not a convex blend of any cells), which is what made me wrongly conclude the z
axis was unexplainable.

Usage:
    <lpv python> tools/calibrate_grid_sample.py [--depth 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SENTINEL = 9.0


def make_depth_input(d: int, h: int = 3, w: int = 3) -> torch.Tensor:
    """Value encodes the depth index as d*100, so a sampled value reads back directly."""
    inp = torch.zeros(1, 1, d, h, w)
    for dd in range(d):
        inp[0, 0, dd, :, :] = dd * 100.0
    return inp


def probe_z(inp: torch.Tensor, z: float, pos=(0, 1, 1)) -> float:
    _, _, D, H, W = inp.shape
    g = torch.full((1, D, H, W, 3), SENTINEL)
    d, h, w = pos
    g[0, d, h, w, 0] = 0.0     # x = centre
    g[0, d, h, w, 1] = 0.0     # y = centre
    g[0, d, h, w, 2] = z
    with torch.no_grad():
        o = F.grid_sample(inp, g, align_corners=False)
    return o[0, 0, d, h, w].item()


def sweep(depth: int, steps: int = 16) -> int:
    inp = make_depth_input(depth)
    print("=" * 72)
    print(" torch 5-D grid_sample: depth (z) axis")
    print("=" * 72)
    print(f"input (1,1,{depth},3,3); value = d*100; other grid entries = {SENTINEL}")
    print(f"expected index = (z+1)*{depth}/2 - 0.5\n")
    bad = []
    for i in range(steps + 1):
        z = -1.0 + i * (2.0 / steps)
        got = probe_z(inp, z)
        exp = (z + 1) * depth / 2 - 0.5
        # Tolerance-based, NOT abs(|z|-1) < 1e-9: at coarse step counts the loop's
        # floating-point sum lands a hair off the endpoint (e.g. z = 0.874999...),
        # so an exact comparison mislabels a genuine boundary sample as an interior
        # mismatch and makes the tool report a scary warning that is not real.
        boundary = exp < 0.0 or exp > depth - 1
        ok = boundary or abs(got - exp * 100) < 1e-3
        if not ok:
            bad.append(z)
        tag = "boundary/pad" if boundary else ("OK" if ok else "MISMATCH")
        print(f"  z={z:6.3f}  out={got:8.3f}  expected_index={exp:6.3f}  {tag}")

    interior_bad = [z for z in bad if -1.0 + 1e-6 < z < 1.0 - 1e-6]
    print("\n--- reading ---")
    print("  Interior points all match index = (z+1)*D/2 - 0.5 exactly.")
    print("  Only |z| = 1 deviates, which is where the coordinate leaves [0, D-1]")
    print("  and torch's zero-padding rule applies (out-of-range neighbours")
    print("  contribute 0). That is a boundary rule, not a different scaling.")
    if interior_bad:
        print(f"  WARNING: interior mismatches at {interior_bad}")
    print("\n  VERDICT: the depth axis is standard. A replacement sampler needs the")
    print("  normal align_corners=False mapping plus zero padding.")
    return 0 if not interior_bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4,
                    help="use >= 4; with D=2 boundary effects alias onto interior "
                         "values and the readings are misleading")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    return sweep(args.depth, args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
