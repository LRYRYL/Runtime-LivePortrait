"""A 3-D trilinear sampler that ONNX can export, replacing torch's grid_sample.

    *** STATUS: NOT CORRECT -- STOP CHANGING sample() BY TRIAL ***

Round 14 state: the gather RULES are fully established, but reshaping the grid-shaped
index (B, Dout, Hout, Wout) into the axis order each gather needs defeated three
consecutive attempts. Errors, in order:

    unsqueeze(1).expand(B,C,Dout,Hout,Wout)  -> broadcasting filled dim 1 with the
                                                index VALUES instead of 1
    permute(0,2,1,3).unsqueeze(3)            -> expanded size (4) must match (5)
    permute(0,1,3,2).unsqueeze(3)            -> expanded size (6) must match (5)

Measured gather rules (torch 2.1), reliable and worth keeping:

  * index must have the SAME RANK as the input
  * every non-gathered axis of the index must match the INPUT's size on that axis
  * so, for input (B,C,D,H,W) and output (B,C,Dout,Hout,Wout):
        dim=2 -> idx (B, 1,  Dout, Hout, Wout)
        dim=3 -> idx (B, C,  Dout, 1,    Wout)
        dim=4 -> idx (B, C,  Dout, Hout, 1   )
    Each verified in isolation, with both constant and per-position indices.

RECOMMENDED NEXT STEP -- stop deriving axis orders, avoid the problem entirely:

  Flatten instead of chaining three gathers.
    1. view input as (B, C, D*H*W)
    2. compute each corner's LINEAR index directly: z*(H*W) + y*W + x
    3. one take_along_dim / gather along the last axis
    4. reshape to (B, C, Dout, Hout, Wout)

  One gather, one index dimension, no axis order to get wrong -- and simpler to
  export to ONNX anyway. Do not touch sample() again until main() passes.

---- original notes below ----

Why this is needed
------------------
LivePortrait's warping network calls `F.grid_sample` with a shape combination that
torch accepts but the ONNX exporter cannot lower:

    OnnxExporterError: Unsupported: ONNX export of operator GridSample
    with 5D volumetric input

Semantics, measured not guessed
-------------------------------
`tools/calibrate_grid_sample.py` established, on real tensors:

  * all three axes follow the standard align_corners=False mapping
        index = (coord + 1) * S / 2 - 0.5
  * coordinates landing outside [0, S-1] get ZERO padding
  * an earlier claim that the depth axis used an unexplained scaling was an
    artifact of calibrating with D=2, where the boundary effects alias onto the
    same interior values

Do NOT use advanced indexing (`inp[:, :, zc, yc, xc]`) anywhere in here: it silently
produces a tensor with an EXTRA dimension, which then mis-broadcasts downstream.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def grid_sample_3d(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Trilinear sample, matching torch's 5-D grid_sample with zero padding.

    input: (B, C, D, H, W)
    grid:  (B, Dout, Hout, Wout, 3), last axis = (x, y, z), each in [-1, 1]
           x indexes W, y indexes H, z indexes D.
    returns (B, C, Dout, Hout, Wout)
    """
    B, C, D, H, W = input.shape
    Dout, Hout, Wout = grid.shape[1:4]
    dtype = input.dtype

    # [-1, 1] -> index space (align_corners=False)
    x = ((grid[..., 0] + 1) * W - 1) / 2
    y = ((grid[..., 1] + 1) * H - 1) / 2
    z = ((grid[..., 2] + 1) * D - 1) / 2

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    z0 = torch.floor(z)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    wx = (x - x0).clamp(0, 1)
    wy = (y - y0).clamp(0, 1)
    wz = (z - z0).clamp(0, 1)

    # ------------------------------------------------------------------
    # Flattened single-gather implementation.
    #
    # Chaining three gathers needs each index permuted into a different axis order,
    # and I got that wrong three times running (see the docstring). Flattening removes
    # the problem: one index, one gather, no axis order to reason about.
    #
    #   input      (B, C, D, H, W) -> (B, C, D*H*W)
    #   linear idx  z*(H*W) + y*W + x      shape (B, 1, Dout*Hout*Wout)
    #   gather along the last axis         -> (B, C, Dout*Hout*Wout)
    #   reshape                            -> (B, C, Dout, Hout, Wout)
    # ------------------------------------------------------------------
    flat = input.reshape(B, C, D * H * W)
    n_out = Dout * Hout * Wout

    def corner(zc, yc, xc, valid):
        """Gather one corner via a linear index. Returns (B, C, Dout, Hout, Wout)."""
        zc = zc.clamp(0, D - 1).to(torch.long).reshape(B, n_out)
        yc = yc.clamp(0, H - 1).to(torch.long).reshape(B, n_out)
        xc = xc.clamp(0, W - 1).to(torch.long).reshape(B, n_out)
        lin = (zc * (H * W) + yc * W + xc).unsqueeze(1).expand(B, C, n_out)
        got = torch.gather(flat, 2, lin).reshape(B, C, Dout, Hout, Wout)
        return got * valid.unsqueeze(1).to(dtype)

    def in_range(t, n):
        return (t >= 0) & (t <= n - 1)

    acc = torch.zeros(B, C, Dout, Hout, Wout, dtype=dtype, device=input.device)
    for zc, w_z in ((z0, 1 - wz), (z1, wz)):
        vz = in_range(zc, D)
        for yc, w_y in ((y0, 1 - wy), (y1, wy)):
            vy = in_range(yc, H)
            for xc, w_x in ((x0, 1 - wx), (x1, wx)):
                vx = in_range(xc, W)
                valid = vz & vy & vx
                weight = (w_z * w_y * w_x).unsqueeze(1).to(dtype)
                acc = acc + corner(zc, yc, xc, valid) * weight
    return acc


def make_test_pair(B=1, C=8, D=4, H=5, W=6, seed=0):
    torch.manual_seed(seed)
    inp = torch.randn(B, C, D, H, W)
    grid = torch.rand(B, D, H, W, 3) * 2 - 1
    return inp, grid


def main() -> int:
    print("=" * 72)
    print(" 3-D trilinear sampler vs torch.grid_sample")
    print("=" * 72)
    ok_all = True
    # several shapes, including LivePortrait's real one
    for tag, (B, C, D, H, W) in (
        ("small", (1, 2, 4, 5, 6)),
        ("deep ", (1, 4, 16, 8, 8)),
        ("LIVEPORTRAIT (B,C,D,H,W)", (1, 32, 16, 64, 64)),
    ):
        inp, grid = make_test_pair(B, C, D, H, W, seed=1)
        with torch.no_grad():
            ref = F.grid_sample(inp, grid, align_corners=False)
            got = grid_sample_3d(inp, grid)
        assert ref.shape == got.shape, (ref.shape, got.shape)
        diff = (ref - got).abs().max().item()
        rel = diff / max(float(ref.abs().max()), 1e-6)
        ok = rel < 1e-4
        ok_all &= ok
        print(f"  {tag:26s} shapes match {tuple(ref.shape)}  "
              f"max|diff| {diff:.3e}  rel {rel:.2e}  {'PASS' if ok else 'FAIL'}")
    print()
    print(f"  RESULT: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
