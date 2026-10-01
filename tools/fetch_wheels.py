"""Download every Python dependency for the right CUDA stack into a local wheel folder.

Two things this solves:

  1. offline installs -- the network is touched once, here, and the install itself then
     runs with --no-index. This also keeps ~3 GB of third-party binaries out of the
     repository, which is both smaller and simpler to license.

  2. GPU compatibility -- the CUDA stack must match the card. torch cu118 supports only
     up to sm_90, so it cannot run on an RTX 50 card at all, while cu128 is about 12%
     slower on an RTX 40 card. tools/gpu_choice.py reads the GPU and picks, and this
     script fetches only that stack (rather than all of them).

    python tools/fetch_wheels.py                     # auto-detect the GPU
    python tools/fetch_wheels.py --cuda cu128        # force a stack
    python tools/fetch_wheels.py --source pypi       # official indexes (default: CN mirror)
    python tools/fetch_wheels.py --check             # report, download nothing

Run it with a **Python 3.10** interpreter: `pip download` resolves wheels for whichever
interpreter runs it, so another version would fetch the wrong ones.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(HERE))
from gpu_choice import choose_cuda, query_compute_caps  # noqa: E402

MIRRORS = {
    "cn": ("https://pypi.tuna.tsinghua.edu.cn/simple", "Tsinghua mirror"),
    "pypi": ("https://pypi.org/simple", "official PyPI"),
}
TORCH_INDEX = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu128": "https://download.pytorch.org/whl/cu128",
}


def run(cmd: list[str]) -> bool:
    t0 = time.time()
    import subprocess
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-2:]
    for l in tail:
        print("      ", l, flush=True)
    if p.returncode != 0:
        print(f"      FAILED after {time.time()-t0:.0f}s", flush=True)
        for l in (p.stderr or "").splitlines()[-10:]:
            print("      !", l, flush=True)
        return False
    print(f"      ok ({time.time()-t0:.0f}s)", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=None,
                    help="where to put the wheels (default: .venv-wheels/<stack>)")
    ap.add_argument("--source", choices=list(MIRRORS), default="cn")
    ap.add_argument("--cuda", choices=["auto", "cu118", "cu128"], default="auto",
                    help="force a CUDA stack instead of detecting the GPU")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    index, label = MIRRORS[args.source]

    # ---- decide the stack ---------------------------------------------------
    caps = query_compute_caps()
    auto_stack, reason = choose_cuda(caps)
    stack = auto_stack if args.cuda == "auto" else args.cuda
    dest = Path(args.dest) if args.dest else (ROOT / ".venv-wheels" / stack)

    print("=" * 66)
    print("  Runtime-LivePortrait - dependency wheels")
    print("=" * 66)
    print(f"  interpreter : {sys.executable}")
    print(f"  version     : {sys.version.split()[0]}")
    for major, minor, name in caps:
        print(f"  gpu         : {name}  sm_{major}{minor}")
    if not caps:
        print("  gpu         : none detected")
    print(f"  cuda stack  : {stack}"
          f"{'   (forced)' if args.cuda != 'auto' else ''}")
    print(f"  reason      : {reason}")
    print(f"  packages    : {label}")
    print(f"  torch from  : {TORCH_INDEX[stack]}")
    print(f"  wheels to   : {dest}")

    if sys.version_info[:2] != (3, 10):
        print("\n  WARNING: this is not Python 3.10; pip would fetch wheels for the"
              " wrong version.")

    have = list(dest.glob("*.whl")) if dest.is_dir() else []
    if args.check:
        print(f"\n  {len(have)} wheel file(s) present"
              f" ({sum(w.stat().st_size for w in have)/1024**2:.0f} MB)")
        if not have:
            print(f"  nothing in {dest} yet; run without --check to download")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    print("\n  NOTE: a few GB will be downloaded. One-time step.\n")

    base = [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
            "--dest", str(dest)]
    # torch has to come from PyTorch's index (the +cuXXX local version is not on PyPI),
    # while onnxruntime-gpu comes from the mirror. extra-index-url lets one resolve see
    # both, which matters because pip downloads a package's dependencies too.
    both = ["--index-url", TORCH_INDEX[stack], "--extra-index-url", index]

    ok = True
    print(f"  [1/2] CUDA stack ({stack})")
    ok &= run(base + ["-r", str(HERE / f"requirements-{stack}.txt")] + both)
    print("\n  [2/2] shared runtime dependencies")
    ok &= run(base + ["-r", str(HERE / "requirements-common.txt")] + both)

    whls = sorted(dest.glob("*.whl"))
    total = sum(w.stat().st_size for w in whls) / 1024 ** 3
    print(f"\n  {len(whls)} wheels, {total:.2f} GB in {dest}")

    if not ok:
        print("\n  RESULT: INCOMPLETE - some downloads failed. Re-run to retry,"
              " or try --source pypi.")
        return 1
    print("\n  RESULT: OK")
    print(f"  Next: tools\\setup.ps1 will install these with no network access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
