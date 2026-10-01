"""Export LivePortrait's per-frame networks to ONNX.

Why export only these
---------------------
Of the four networks, only two are on the per-frame critical path:

  appearance_feature_extractor (F)   once per source portrait -- NOT exported
  motion_extractor (M)               every frame, ~10.7 ms   -- exported
  warping_module (W) + spade_generator (G)
                                     every frame, ~41 ms     -- exported as ONE graph

W and G are fused into a single ONNX graph on purpose: they always run back to
back on the same tensors, so exporting them separately would add an inference call
plus a device round-trip per frame. Fusing keeps the intermediate in-graph.

Measured on this machine, W+G is 61% of the frame, so it is the only place where
ONNX Runtime can move the needle.

Usage (from Runtime-LivePortrait/):
    <lpv python> tools/export_onnx.py [--fp16] [--verify]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

LP = ROOT / "LivePortrait"
sys.path.insert(0, str(LP))

from src.config.inference_config import InferenceConfig  # noqa: E402
from src.live_portrait_wrapper import LivePortraitWrapper  # noqa: E402

OUT = ROOT / "onnx"


class WarpingWithFlatSampler(nn.Module):
    """Wraps the warping network so its 5-D grid_sample becomes a flat sampler.

    Two approaches failed before this one, both worth recording:

      * `warping.deform_input = lambda ...` on the INSTANCE -- works in eager mode,
        but torch.onnx.export traces the class-level forward and never sees it, so the
        export still raised "Unsupported: ONNX export of operator GridSample with 5D
        volumetric input".
      * reassigning `self.warping.__class__` INSIDE forward -- also too late, because
        the tracer has already resolved the call by then.

    What works is patching `torch.nn.functional.grid_sample` itself, since
    `WarpingNetwork.deform_input` calls it by that name and the exporter resolves the
    symbol while tracing. The patch is scoped to the export call and always restored.
    """

    def __init__(self, warping):
        super().__init__()
        self.warping = warping

    def forward(self, feature_3d, kp_source, kp_driving):
        return self.warping(feature_3d, kp_source=kp_source, kp_driving=kp_driving)


def export_one(module: nn.Module, args: tuple, names: list[str],
               out_names: list[str], path: Path, fp16: bool,
               dynamic: bool = False) -> Path:
    module = module.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(dev)
    args = tuple(a.to(dev) for a in args)

    # opset 17 covers everything these nets use and is what ORT 1.18 handles best.
    # dynamo=False keeps the legacy exporter, which is far more predictable for
    # these hand-written architectures.
    import torch.nn.functional as F
    from fastportrait.grid_sample_3d import grid_sample_3d

    _orig = F.grid_sample

    def _patched(inp, grid, mode="bilinear", padding_mode="zeros",
                 align_corners=None, **kw):
        # Only the volumetric call is replaced; 2-D/4-D sampling keeps torch's own
        # implementation so nothing else in the graph changes behaviour.
        if inp.dim() == 5:
            return grid_sample_3d(inp, grid)
        return _orig(inp, grid, mode=mode, padding_mode=padding_mode,
                     align_corners=align_corners, **kw)

    F.grid_sample = _patched
    try:
        with torch.no_grad():
            torch.onnx.export(
                module, args, str(path),
                input_names=names, output_names=out_names,
                opset_version=17, do_constant_folding=True,
                dynamic_axes=({n: {0: "batch"} for n in names} if dynamic else None),
            )
    finally:
        F.grid_sample = _orig
    return path


class WarpDecode(nn.Module):
    """warping_module + spade_generator as one graph.

    Mirrors `LivePortraitWrapper.warp_decode` for the single-frame, single-batch
    case, minus the fp16 autocast context (precision is handled inside ONNX
    Runtime instead).
    """

    def __init__(self, warping, spade):
        super().__init__()
        self.warping = WarpingWithFlatSampler(warping)
        self.spade = spade

    def forward(self, feature_3d, kp_source, kp_driving):
        ret = self.warping(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
        return self.spade(feature=ret["out"])


def verify_deform(wrapper: LivePortraitWrapper) -> bool:
    """Prove the folded 4-D sampling equals the original call.

    Uses the real shapes from the wrapper rather than invented ones, so a shape
    assumption that happens to hold for random tensors cannot pass by accident.
    """
    import torch.nn.functional as F

    dev = wrapper.device
    inp = torch.randn(1, 32, 16, 64, 64, device=dev)
    grid = torch.rand(1, 16, 64, 64, 3, device=dev) * 2 - 1

    with torch.no_grad():
        ref = F.grid_sample(inp, grid, align_corners=False)

        b, c, d, h, w = inp.shape
        src = inp.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        g = grid.reshape(b * d, h, w, 3)[..., :2]
        got = F.grid_sample(src, g, align_corners=False)
        got = got.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)

    diff = (ref - got).abs().max().item()
    print(f"    deform 5D-vs-4D max |diff| = {diff:.3e}  "
          f"{'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


class MotionExtract(nn.Module):
    def __init__(self, motion):
        super().__init__()
        self.motion = motion

    def forward(self, x):
        return self.motion(x)


def references(wrapper: LivePortraitWrapper):
    """Build the real input shapes from the wrapper's own config."""
    dev = wrapper.device
    B, N = 1, 21          # 21 keypoints for the human model
    feat = torch.randn(B, 32, 16, 64, 64, device=dev)
    kp_s = torch.randn(B, N, 3, device=dev)
    kp_d = torch.randn(B, N, 3, device=dev)
    img = torch.rand(B, 3, 256, 256, device=dev)
    return feat, kp_s, kp_d, img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true",
                    help="also export fp16 variants (ORT handles the cast)")
    ap.add_argument("--verify", action="store_true",
                    help="compare ONNX output against PyTorch")
    ap.add_argument("--dynamic", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = InferenceConfig(flag_use_half_precision=False)
    wrapper = LivePortraitWrapper(inference_cfg=cfg)
    print(f"device: {wrapper.device}")

    feat, kp_s, kp_d, img = references(wrapper)

    print("\n[1] exporting warping_module + spade_generator (61% of the frame)")
    wd = WarpDecode(wrapper.warping_module, wrapper.spade_generator)
    t0 = time.perf_counter()
    p_wd = export_one(wd, (feat, kp_s, kp_d),
                      ["feature_3d", "kp_source", "kp_driving"], ["out"],
                      OUT / "warp_decode.onnx", args.fp16, args.dynamic)
    print(f"    {p_wd.name}  {p_wd.stat().st_size/1024**2:.1f} MB  "
          f"({time.perf_counter()-t0:.1f}s)")

    if args.fp16:
        print("\n[2] exporting fp16 variant")
        wd16 = WarpDecode(wrapper.warping_module, wrapper.spade_generator).half()
        t0 = time.perf_counter()
        p16 = export_one(wd16, (feat.half(), kp_s.half(), kp_d.half()),
                         ["feature_3d", "kp_source", "kp_driving"], ["out"],
                         OUT / "warp_decode_fp16.onnx", True, args.dynamic)
        print(f"    {p16.name}  {p16.stat().st_size/1024**2:.1f} MB  "
              f"({time.perf_counter()-t0:.1f}s)")

    print("\n[3] exporting motion_extractor (14% of the frame)")
    t0 = time.perf_counter()
    p_m = export_one(MotionExtract(wrapper.motion_extractor), (img,), ["img"],
                     ["pitch", "yaw", "roll", "t", "exp", "scale", "kp"],
                     OUT / "motion_extractor.onnx", args.fp16, args.dynamic)
    print(f"    {p_m.name}  {p_m.stat().st_size/1024**2:.1f} MB  "
          f"({time.perf_counter()-t0:.1f}s)")

    if not args.verify:
        print(f"\nwrote to {OUT}")
        return 0

    print("\n[4] numeric verification against PyTorch")
    import onnxruntime as ort
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(p_wd), providers=prov)
    print(f"    providers: {sess.get_providers()}")

    with torch.no_grad():
        ref = wd(feat, kp_s, kp_d)
    got = sess.run(["out"], {
        "feature_3d": feat.cpu().numpy(),
        "kp_source": kp_s.cpu().numpy(),
        "kp_driving": kp_d.cpu().numpy(),
    })[0]

    r = ref.cpu().numpy()
    diff = np.abs(r - got)
    denom = max(float(np.abs(r).max()), 1e-6)
    print(f"    max |diff| {diff.max():.3e}   rel {diff.max()/denom:.3e}")
    print(f"    shapes ref {r.shape} onnx {got.shape}")
    ok = diff.max() / denom < 1e-3
    print(f"    {'PASS' if ok else 'FAIL'} (relative tolerance 1e-3)")

    print("\n[5] timing")
    for tag, fn in (
        ("pytorch", lambda: wd(feat, kp_s, kp_d)),
        ("onnx", lambda: sess.run(["out"], {
            "feature_3d": feat.cpu().numpy(),
            "kp_source": kp_s.cpu().numpy(),
            "kp_driving": kp_d.cpu().numpy()})),
    ):
        for _ in range(3):
            fn()
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
        print(f"    {tag:8s} median {np.median(ts):6.1f} ms")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
