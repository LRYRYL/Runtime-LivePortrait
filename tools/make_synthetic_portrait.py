"""Generate a synthetic portrait with StyleGAN2-FFHQ.

Why synthetic
-------------
The default portrait shipped with the web UI has to be something that cannot create a
rights problem. A StyleGAN2-FFHQ sample is **not a real person at all**: no
personality rights, no model release, no dataset redistribution question for a single
generated image. SFHQ documents the same reasoning ("all images ... synthetically
generated so there are no privacy issues or license issues") but is Kaggle-hosted in
large archives, so generating one locally is simpler and fully reproducible from the
checkpoint already on this machine.

Everything uses the pure-Python `upfirdn2d_native` from the HairFastGAN repo. The
compiled CUDA ops cannot be built in the `lpv` env (ninja absent, nvcc 13.1 vs the
cu118 torch build), so `torch.utils.cpp_extension.load` is stubbed out before import.
`upfirdn2d_native` needs `kernel.flip([0,1])`; a hand-written replacement that flipped
only the last axis produced wrong channel counts -- use the repo's own kernel.

Usage:
    <lpv python> tools/make_synthetic_portrait.py --seed 7 [--psi 0.7] [--out PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parent.parent
# The StyleGAN2 checkpoint and generator source are not part of this project; they came
# from an earlier HairFastGAN checkout. Nothing is hardcoded -- the location is resolved
# at run time (see find_hairgan) so the tool keeps working wherever that checkout lives.
HAIRGAN_ENV = "LPV_HAIRGAN"


def find_hairgan() -> Path:
    """Locate the HairFastGAN checkout holding the StyleGAN2 generator + weights.

    Resolution order, most explicit first:
      1. $LPV_HAIRGAN                        (set this if it lives somewhere unusual)
      2. a sibling of this project's parent  (e.g. faceAI_archived_*/PersonaLive-LiveEdit/hairgan)
      3. any */PersonaLive-LiveEdit/hairgan  one or two levels up
    """
    def ok(p: Path) -> bool:
        return ((p / "models" / "stylegan2" / "model.py").is_file()
                and (p / "pretrained_models" / "StyleGAN" / "ffhq.pt").is_file())

    env = os.environ.get(HAIRGAN_ENV, "").strip()
    if env:
        # An explicit setting must not be silently ignored: a typo used to fall
        # through to auto-discovery and "work", which hides the mistake.
        p = Path(env)
        if ok(p):
            return p
        raise SystemExit(
            f"${HAIRGAN_ENV} is set but does not look like a HairFastGAN checkout:\n"
            f"  {p}\n"
            "Expected models/stylegan2/model.py and "
            "pretrained_models/StyleGAN/ffhq.pt under it.\n"
            f"Fix the path, or unset {HAIRGAN_ENV} to auto-discover.")

    for base in (ROOT.parent, ROOT.parent.parent):
        if not base.is_dir():
            continue
        direct = base / "PersonaLive-LiveEdit" / "hairgan"
        if ok(direct):
            return direct
        for cand in sorted(base.glob("*/PersonaLive-LiveEdit/hairgan")):
            if ok(cand):
                return cand

    raise SystemExit(
        "Could not find the HairFastGAN checkout that holds the StyleGAN2 weights.\n"
        "Point at it explicitly, e.g.:\n"
        f'  $env:{HAIRGAN_ENV} = "D:\\somewhere\\PersonaLive-LiveEdit\\hairgan"\n'
        "It must contain models/stylegan2/model.py and "
        "pretrained_models/StyleGAN/ffhq.pt.\n"
        "(The portraits/sample*.jpg already generated do NOT need this tool.)")


HG = find_hairgan()


def build_generator(device: str = "cuda"):
    """Return (G, latent_avg) with the repo's native ops substituted in."""
    # Neutralise the extension build before anything imports it.
    import torch.utils.cpp_extension as _ce
    _ce.load = lambda *a, **k: types.SimpleNamespace()
    _ce.load_inline = lambda *a, **k: types.SimpleNamespace()

    sys.path.insert(0, str(HG))
    import models.stylegan2 as _sg

    spec = importlib.util.spec_from_file_location(
        "_u2d", str(HG / "models" / "stylegan2" / "op" / "upfirdn2d.py"))
    u2d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(u2d)
    native = u2d.upfirdn2d_native

    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
        return native(input, kernel, up_x=up, up_y=up, down_x=down, down_y=down,
                      pad_x0=pad[0], pad_x1=pad[1], pad_y0=pad[0], pad_y1=pad[1])

    def fused_leaky_relu(inp, bias=None, negative_slope=0.2, scale=2 ** 0.5):
        if bias is not None:
            inp = inp + bias.view(1, -1, 1, 1)
        return Fn.leaky_relu(inp, negative_slope) * scale

    class FusedLeakyReLU(torch.nn.Module):
        def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(channel))
            self.negative_slope, self.scale = negative_slope, scale

        def forward(self, input):
            return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

    op = types.ModuleType("models.stylegan2.op")
    op.__path__ = []
    op.upfirdn2d = upfirdn2d
    op.fused_leaky_relu = fused_leaky_relu
    op.FusedLeakyReLU = FusedLeakyReLU
    sys.modules["models.stylegan2.op"] = op
    setattr(_sg, "op", op)

    from models.stylegan2.model import Generator

    ck = torch.load(str(HG / "pretrained_models" / "StyleGAN" / "ffhq.pt"),
                    map_location="cpu")
    G = Generator(1024, 512, 8, channel_multiplier=2)
    missing, unexpected = G.load_state_dict(ck["g_ema"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected")
    G = G.eval().to(device)
    return G, ck["latent_avg"].to(device)


def generate(seed: int, psi: float, device: str = "cuda"):
    import cv2
    import numpy as np

    G, latent_avg = build_generator(device)
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(1, 512, generator=g).to(device)
        img = G([z], input_is_latent=False, randomize_noise=True,
                truncation=psi, truncation_latent=latent_avg.unsqueeze(0))[0]
        arr = ((img.clamp(-1, 1) + 1) / 2 * 255)[0].permute(1, 2, 0)
        arr = arr.cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--psi", type=float, default=0.7,
                    help="truncation: lower = more typical/frontal and cleaner")
    ap.add_argument("--out", default=str(ROOT / "portraits" / "sample.jpg"))
    ap.add_argument("--grid", default=None, help="also write a contact sheet")
    ap.add_argument("--seeds", default="", help="comma list for the contact sheet")
    args = ap.parse_args()

    import cv2
    import numpy as np

    if args.grid and args.seeds:
        tiles = []
        for s in [int(x) for x in args.seeds.split(",")]:
            a = generate(s, args.psi)
            a = cv2.resize(a, (300, 300), interpolation=cv2.INTER_AREA)
            cv2.putText(a, "s%d" % s, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)
            tiles.append(a)
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(args.grid, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote grid", args.grid)

    img = generate(args.seed, args.psi)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"wrote {args.out}  {img.shape[1]}x{img.shape[0]}  seed={args.seed} psi={args.psi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
