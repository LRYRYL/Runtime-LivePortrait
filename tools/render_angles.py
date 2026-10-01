"""Render a range of yaw angles for the source portrait.

Exists to document the head-turn fix: before it, `process()` built the driver's
rotation matrix and never applied it, so a 40 degree yaw changed the output by only
0.95/255. With the rotation applied the head actually turns, and this renders the
evidence.

Rendering works by rebuilding the driving keypoints the same way
`LivePortraitWrapper.transform_keypoint` does -- `scale * (kp @ R + exp) + t` -- with an
explicit yaw, then decoding. `headpose_pred_to_degree` is an identity function, so the
pitch/yaw/roll values are already DEGREES; feeding radians yields a rotation of ~0,
which is what caused an earlier, wrong conclusion that the model could not turn heads.

Usage:
    <lpv python> tools/render_angles.py [--source PATH] [--out PATH] [--mult 0.7]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fastportrait  # noqa: F401,E402  (CUDA DLL PATH fix, must precede onnxruntime)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fastportrait.realtime import FastConfig, LivePortraitFast  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402


def read_unicode(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "portraits" / "sample.jpg"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "testD_deg.jpg"))
    ap.add_argument("--mult", type=float, default=0.7)
    ap.add_argument("--yaws", default="0,20,40,60")
    args = ap.parse_args()

    src = read_unicode(Path(args.source))
    if src is None:
        print(f"cannot read {args.source}")
        return 2

    eng = LivePortraitFast(FastConfig(driving_multiplier=args.mult))
    if not eng.set_source(src):
        print("no face in the source portrait")
        return 1
    xs, fs, si = eng.x_s, eng.f_s, eng.x_s_info
    dev = xs.device

    def render(yaw_deg: float):
        R = get_rotation_matrix(
            torch.tensor([[si["pitch"].item()]], device=dev),
            torch.tensor([[yaw_deg]], device=dev),
            torch.tensor([[si["roll"].item()]], device=dev))
        kp = si["kp"]
        kp_new = kp @ R + si["exp"].view(kp.shape)
        kp_new = kp_new * si["scale"][..., None]
        kp_new[:, :, 0:2] += si["t"][:, None, 0:2]
        x = eng.wrapper.stitching(xs, kp_new)
        x = xs + (x - xs) * args.mult
        with eng.wrapper.inference_ctx():
            o = eng.wrapper.warp_decode(fs, xs, x)
        return np.ascontiguousarray(eng.wrapper.parse_output(o["out"])[0])

    tiles = []
    for y in [float(v) for v in args.yaws.split(",")]:
        img = render(y)
        cv2.putText(img, "yaw %d" % int(y), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        tiles.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    grid = np.hstack(tiles)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    d = float(np.abs(tiles[0].astype(int) - tiles[-1].astype(int)).mean())
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})")
    print(f"yaw {args.yaws.split(',')[0]} vs {args.yaws.split(',')[-1]}: "
          f"mean|diff| = {d:.2f}  (>5 means the head really turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
