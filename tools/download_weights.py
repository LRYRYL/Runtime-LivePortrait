"""Download the LivePortrait model weights.

The weights are NOT part of this repository: they total ~667 MB, and one of the two
model sets (InsightFace `buffalo_l`) is licensed for NON-COMMERCIAL RESEARCH ONLY.
Fetching them here keeps that separation explicit instead of burying 667 MB of
third-party binaries in git history.

    python tools/download_weights.py              # normal
    python tools/download_weights.py --mirror     # use hf-mirror.com (faster in CN)
    python tools/download_weights.py --check      # only report what is missing

Every file is verified against the SHA-256 published by the official Hugging Face
repository (KwaiVGI/LivePortrait), so a truncated or substituted download is caught
rather than silently producing a broken model.

Source: https://huggingface.co/KwaiVGI/LivePortrait  (mirror: https://hf-mirror.com)

Licences
--------
* liveportrait/*  -- LivePortrait, MIT (code) / see the upstream repository
* insightface/*   -- InsightFace models, NON-COMMERCIAL RESEARCH ONLY
                     https://github.com/deepinsight/insightface  ("the models trained
                     with these data are available for non-commercial research
                     purposes only")
By downloading these files you agree to those terms.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import requests                     # declared dependency; used so 308 redirects work
except ImportError:                     # pragma: no cover
    requests = None

USER_AGENT = "Runtime-LivePortrait/1.0"

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "LivePortrait" / "pretrained_weights"

REPO_PATH = "KwaiVGI/LivePortrait"
HOSTS = {
    "hf": "https://huggingface.co",
    "mirror": "https://hf-mirror.com",
}

# (relative path under pretrained_weights, size in bytes, sha256 from the HF LFS metadata)
FILES: list[tuple[str, int, str]] = [
    ("liveportrait/base_models/appearance_feature_extractor.pth", 3_387_959,
     "5279bb8654293dbdf327030b397f107237dd9212fb11dd75b83dfb635211ceb5"),
    ("liveportrait/base_models/motion_extractor.pth", 112_545_506,
     "251e6a94ad667a1d0c69526d292677165110ef7f0cf0f6d199f0e414e8aa0ca5"),
    ("liveportrait/base_models/spade_generator.pth", 221_813_590,
     "4780afc7909a9f84e24c01d73b31a555ef651521a1fe3b2429bd04534d992aee"),
    ("liveportrait/base_models/warping_module.pth", 182_180_086,
     "2f61a6f265fe344f14132364859a78bdbbc2068577170693da57fb96d636e282"),
    ("liveportrait/retargeting_models/stitching_retargeting_module.pth", 2_393_098,
     "3652d5a3f95099141a56986aaddec92fadf0a73c87a20fac9a2c07c32b28b611"),
    ("liveportrait/landmark.onnx", 114_666_491,
     "31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db"),
    # InsightFace buffalo_l -- NON-COMMERCIAL RESEARCH ONLY
    ("insightface/models/buffalo_l/2d106det.onnx", 5_030_888,
     "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"),
    ("insightface/models/buffalo_l/det_10g.onnx", 16_923_827,
     "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256_of(path) == digest


def human(n: float) -> str:
    return f"{n / 1048576:.1f} MB"


def fetch(url: str, dest: Path) -> None:
    """Stream to a .part file, then rename, so an interrupted run cannot look complete.

    Uses `requests` rather than urllib because hf-mirror.com answers with a
    **308 Permanent Redirect**, and urllib's default opener raises
    `HTTP Error 308` instead of following it -- for a 3.2 MB file as much as a 211 MB
    one, so the whole mirror was unusable. requests follows 308 transparently.

    Downloaded by hand rather than with `urlretrieve` so the caller gets real progress
    output for a 628 MB transfer.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": USER_AGENT}

    if requests is not None:
        with requests.get(url, headers=headers, stream=True, timeout=(30, 300),
                          allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            last = -1
            with tmp.open("wb") as out:
                for chunk in r.iter_content(1 << 20):
                    if not chunk:
                        continue
                    out.write(chunk)
                    got += len(chunk)
                    pct = int(got * 100 / total) if total else 0
                    if pct // 5 != last // 5 or (total and got >= total):
                        last = pct
                        sys.stdout.write(f"\r      {human(got)} / {human(total)}  {pct:3d}%")
                        sys.stdout.flush()
    else:
        # requests is a declared dependency, so this path is only a safety net; follow
        # redirects explicitly since that is exactly what breaks on the mirror.
        class Follow(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                return urllib.request.Request(newurl, headers=headers)

        opener = urllib.request.build_opener(Follow)
        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=60) as r, tmp.open("wb") as out:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            last = -1
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                out.write(b)
                got += len(b)
                pct = int(got * 100 / total) if total else 0
                if pct // 5 != last // 5 or got == total:
                    last = pct
                    sys.stdout.write(f"\r      {human(got)} / {human(total)}  {pct:3d}%")
                    sys.stdout.flush()

    sys.stdout.write("\r" + " " * 44 + "\r")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", choices=list(HOSTS), default=None,
                    help="which source to use; default is the hf-mirror.com mirror")
    ap.add_argument("--official", action="store_true",
                    help="use huggingface.co instead of the hf-mirror.com mirror")
    ap.add_argument("--mirror", action="store_true",
                    help="(kept for compatibility; the mirror is already the default)")
    ap.add_argument("--check", action="store_true",
                    help="only report which files are present/missing")
    args = ap.parse_args()

    # The mirror is the default: it is reachable from mainland China, where
    # huggingface.co is frequently blocked, and it carries the identical files (the
    # SHA-256 checks below prove that on every run).
    if args.host:
        choice = args.host
    elif args.official:
        choice = "hf"
    else:
        choice = "mirror"
    host = HOSTS[choice]
    print("=" * 66)
    print("  LivePortrait weights")
    print("=" * 66)
    print(f"  destination : {DEST}")
    print(f"  source      : {host}")
    print("  NOTE: the insightface/* files are NON-COMMERCIAL RESEARCH ONLY.")
    print()

    missing: list[tuple[str, int, str]] = []
    for rel, size, digest in FILES:
        dest = DEST / rel
        if verify(dest, size, digest):
            print(f"  [ ok ]  {rel}  ({human(size)})")
        elif dest.exists():
            print(f"  [BAD ]  {rel}  -- size/hash mismatch, will re-download")
            missing.append((rel, size, digest))
        else:
            print(f"  [ -- ]  {rel}  ({human(size)})")
            missing.append((rel, size, digest))

    if args.check:
        print()
        print(f"  {len(FILES) - len(missing)}/{len(FILES)} present")
        return 1 if missing else 0

    if not missing:
        print("\n  nothing to do -- all files verified")
        return 0

    total = sum(s for _, s, _ in missing)
    print(f"\n  downloading {len(missing)} file(s), {human(total)} total\n")
    failed: list[str] = []
    for i, (rel, size, digest) in enumerate(missing, 1):
        url = f"{host}/{REPO_PATH}/resolve/main/{rel}"
        print(f"  [{i}/{len(missing)}] {rel}")
        try:
            fetch(url, DEST / rel)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"      FAILED: {exc}")
            failed.append(rel)
            continue
        if verify(DEST / rel, size, digest):
            print(f"      verified ok")
        else:
            print("      FAILED: downloaded file failed verification")
            failed.append(rel)

    print()
    if failed:
        print("  Some files could not be downloaded:")
        for f in failed:
            print(f"    - {f}")
        print("\n  Retry, or try --mirror if huggingface.co is slow/blocked.")
        return 1
    print("  All weights downloaded and verified.")
    print("  Next:  python tools/check_env.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
