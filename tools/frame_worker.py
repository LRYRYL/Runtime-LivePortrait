"""Frame-pipe worker: animate frames sent by a host process.

Why a subprocess exists at all
------------------------------
LivePortrait needs torch 2.1 and numpy 1.26. Deep-Live-Cam's venv has no torch and
numpy 2.5, and the two cannot be reconciled in one interpreter (numpy 2.x breaks
LivePortrait's dependencies, and installing torch into Deep-Live-Cam's env would
violate the constraint of not disturbing the existing projects). So Deep-Live-Cam
starts this worker with the `lpv` interpreter and talks to it over stdio.

Protocol (all little-endian)
----------------------------
  host -> worker   uint32 length | JPEG bytes        (a BGR frame)
                   uint32 0xFFFFFFFF                  (shutdown)
  worker -> host   uint32 length | JPEG bytes        (the animated frame)
                   uint32 0xFFFFFFFF | uint32 status  (fatal error)

JPEG rather than raw pixels because a 1280x720 BGR frame is 2.7 MB raw and ~60 KB
encoded; at 15-40 FPS the pipe would otherwise become the bottleneck. It also
costs ~1 ms to encode/decode, well under the ~68 ms animation.

The first frame after startup is preceded by a one-line text banner on stdout so
the host can tell "still loading models" from "hung"; the host reads that line
before entering the binary protocol.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MAGIC_STOP = 0xFFFFFFFF


def _read_exact(stream, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_blob(stream) -> bytes | None:
    hdr = _read_exact(stream, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack("<I", hdr)
    if n == MAGIC_STOP:
        return None
    return _read_exact(stream, n)


def _write_blob(stream, data: bytes) -> None:
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


def decode_frame(blob: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def encode_frame(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="portrait to animate")
    ap.add_argument("--quality", type=int, default=90,
                    help="JPEG quality of frames the host sends us")
    ap.add_argument("--out-quality", type=int, default=82,
                    help="JPEG quality of frames we send back. The return path is "
                         "the one on the critical path, so it can be lower.")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src_path = Path(args.source)
    src = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if src is None:
        print(f"READY 0 cannot decode {src_path}", flush=True)
        return 2

    from fastportrait.realtime import FastConfig, LivePortraitFast

    eng = LivePortraitFast(FastConfig())
    if not eng.set_source(src):
        print("READY 0 no face in source portrait", flush=True)
        return 2
    # Tell the host we are up; it must not send frames before this line.
    print("READY 1", flush=True)

    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    n_in = n_out = 0
    times: list[float] = []
    try:
        while True:
            blob = _read_blob(inp)
            if blob is None:
                break
            frame = decode_frame(blob)
            if frame is None:
                _write_blob(out, encode_frame(np.zeros((8, 8, 3), np.uint8), 50))
                continue
            n_in += 1
            t = time.perf_counter()
            animated = eng.process(frame)
            dt = (time.perf_counter() - t) * 1000
            applied = eng.last.applied
            if applied:
                times.append(dt)
                n_out += 1
                payload = animated
            else:
                # Not animated: echo the input so the host always gets one frame
                # back per frame sent and never desynchronises.
                payload = frame
            _write_blob(out, encode_frame(payload, args.out_quality))
            if not args.quiet and n_in % 30 == 0:
                if times:
                    med = float(np.median(times[-60:]))
                    msg = (f"  [lpv] {n_in} in / {n_out} animated | "
                           f"median {med:.1f} ms -> {1000/med:.1f} FPS\n")
                else:
                    msg = f"  [lpv] {n_in} in / {n_out} animated | no face\n"
                sys.stderr.write(msg)
                sys.stderr.flush()
            if args.display:
                cv2.imshow("lpv worker", animated)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        if args.display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
