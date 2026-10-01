"""Environment self-check for the launcher.

Lives in Python rather than inline `python -c` inside the .bat because percent
signs are special to cmd: a format string like '%dx%d' gets mangled when the batch
file is parsed, and the failure is silent (the check printed nothing at all, on both
the success and the failure path). Keeping the logic here removes that whole class of
quoting bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# files that must all be present for a live session
WEIGHTS = [
    "LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
    "LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth",
    "LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth",
    "LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth",
    "LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
    "LivePortrait/pretrained_weights/liveportrait/landmark.onnx",
    "LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx",
    "LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx",
]


def _compute_caps() -> list[tuple[int, int, str]]:
    """[(major, minor, name)] for each NVIDIA GPU, via the shared GPU chooser.

    Used to explain *why* CUDA is unavailable, and which stack would fix it. Importing
    the chooser keeps one source of truth for the sm_89 / sm_90 boundary.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gpu_choice import query_compute_caps
        return query_compute_caps()
    except Exception:                                           # noqa: BLE001
        return []


def main() -> int:
    fails = 0

    print("  packages:")
    try:
        import cv2
        import numpy
        import onnxruntime as ort
        import torch
        print(f"    [ OK ] torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
        print(f"    [ OK ] numpy        {numpy.__version__}")
        print(f"    [ OK ] opencv       {cv2.__version__}")
        print(f"    [ OK ] onnxruntime  {ort.__version__}")
        print(f"           CUDA build  {torch.version.cuda}")
    except Exception as exc:                                    # noqa: BLE001
        print(f"    [FAIL] import failed: {exc}")
        return 1

    print("  gpu:")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"    [ OK ] {props.name}")
        major, minor = props.major, props.minor
        print(f"           compute capability sm_{major}{minor}")
        print(f"           {total / 1024**3:.1f} GB total, {free / 1024**3:.1f} GB free")

        # A build that has no cubin for this card still usually runs (the driver falls
        # back to the nearest lower cubin or JITs from PTX), so this is informational.
        arch = torch.cuda.get_arch_list()
        exact = f"sm_{major}{minor}" in arch
        if exact:
            print(f"           native cubin present: yes")
        else:
            lower = [a for a in arch if a.startswith("sm_")
                     and int(a.split("_")[1]) // 10 == major
                     and int(a.split("_")[1]) % 10 <= minor]
            print(f"           native cubin present: no"
                  f"  (nearest lower {lower[-1] if lower else 'none'}; normal)")
    else:
        print("    [FAIL] no CUDA device visible to torch")
        # The usual cause is a CUDA build that cannot target this card's architecture:
        # torch cu118 stops at sm_90, so an RTX 50 (sm_120) card is invisible to it.
        caps = _compute_caps()
        if caps:
            sm = f"sm_{caps[0][0]}{caps[0][1]}"
            need = "cu128" if caps[0][0] >= 9 else "cu118"
            print(f"           this machine has {caps[0][2]} ({sm})")
            print(f"           a working install needs the {need} stack"
                  f"{'' if need == 'cu118' else '; cu118 cannot target sm_120'}")
            print(f"           fix:  $env:LPR_CUDA='{need}';"
                  f" powershell -ExecutionPolicy Bypass -File tools\\setup.ps1 -Reinstall")
        fails += 1

    print("  camera:")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"    [ OK ] camera 0 at {w}x{h}")
        else:
            print("    [FAIL] camera opened but returned no frame")
            fails += 1
    else:
        print("    [FAIL] camera 0 unavailable - another app may hold it")
        print("           (WeChat / DingTalk / a browser tab all grab the webcam)")
        fails += 1
    cap.release()

    print("  weights (8 files, ~628 MB):")
    missing = 0
    for rel in WEIGHTS:
        p = ROOT / rel
        if p.exists():
            print(f"    [ OK ] {p.name}")
        else:
            print(f"    [FAIL] missing {rel}")
            missing += 1
    fails += missing

    print("  portrait:")
    # Prefer the user's own photo; the bundled sample is a synthetic StyleGAN2 face
    # (not a real person), so it needs no attribution and carries no rights issue.
    mine = [ROOT / "portraits" / n
            for n in ("my_face.jpg", "my_face.png", "sample.jpg")]
    found = next((p for p in mine if p.exists()), None)
    if found:
        print(f"    [ OK ] {found.relative_to(ROOT)}  (your photo)")
    else:
        print("    [FAIL] no portrait found; save one as portraits/my_face.jpg")
        print("           (or restore the bundled synthetic portraits/sample.jpg)")
        fails += 1

    print()
    if fails == 0:
        print("  RESULT: everything needed for a live session is present.")
    else:
        print(f"  RESULT: {fails} problem(s) found - see the [FAIL] lines above.")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
