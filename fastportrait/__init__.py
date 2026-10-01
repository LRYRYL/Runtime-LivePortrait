"""Runtime-LivePortrait runtime.

Importing this package performs one piece of global setup that has to happen before
onnxruntime loads any execution provider.
"""
from __future__ import annotations

import os
from pathlib import Path


def _fix_cuda_dll_path() -> None:
    """Put the CUDA 11.8 runtime on PATH so onnxruntime's CUDA provider can load.

    Measured failures on this machine, from the CUDA host:

        [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126
        "" when trying to load onnxruntime_providers_cuda.dll

    and, with a different onnxruntime version:

        CUDA_PATH is set but CUDA wasnt able to be loaded

    Cause: the machine has the CUDA **13.1** toolkit installed, so CUDA_PATH points
    at v13.1, but torch -- and therefore this project -- is built against CUDA
    **11.8**. The 11.8 runtime DLLs (cublasLt64_11, cudart64_110, cudnn64_8, ...)
    ship inside `torch/lib` and are not on PATH, so the provider DLL cannot resolve
    its imports.

    Prepending `torch/lib` to PATH fixes it; verified by running a real ONNX
    inference through the CUDA provider. Doing it here means every entry point
    (app.py and the tools) gets it, as long as this package is imported before
    onnxruntime.

    Note: onnxruntime-gpu is pinned to 1.18.0 because 1.19+ requires the CUDA 12
    runtime; that combination also failed, with a missing `cublasLt64_12.dll`.
    """
    try:
        import torch  # noqa: F401  (importing also registers its DLL directories)
        lib = Path(torch.__file__).resolve().parent / "lib"
    except Exception:
        return
    if not lib.is_dir():
        return
    cur = os.environ.get("PATH", "")
    if str(lib) not in cur.split(os.pathsep):
        os.environ["PATH"] = str(lib) + os.pathsep + cur


_fix_cuda_dll_path()
