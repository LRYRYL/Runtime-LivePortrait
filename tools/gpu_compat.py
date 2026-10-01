"""One rule for "can this torch build actually target this GPU?", shared by every tool.

Why this is centralised
-----------------------
The same question is asked in three places, and getting it wrong once produced a
diagnostic that told users everything was normal on a machine where the GPU could not
run at all:

    native cubin present: no  (nearest lower none; normal)

That "normal" was wrong. The card was sm_120 and the build's highest cubin was sm_90, so
there was nothing to fall back to. Two failure modes look identical in
`torch.cuda.get_arch_list()` and must not be confused:

    build max >= card   -> usable. Either an exact cubin, or a lower one the driver
                           runs as-is. cu118 on an RTX 4080 lands here (sm_86 cubin),
                           and the stderr warning about a missing sm_89 cubin is benign.

    build max <  card   -> NOT usable. No cubin was compiled for an architecture this
                           new, which for an RTX 50 card under cu118 means the kernels
                           simply do not exist. torch still reports CUDA as available,
                           so only this comparison catches it.

Returned tuple: (usable, reason, detail)
    reason is one of "native", "fallback", "incompatible", "no-cuda"
"""
from __future__ import annotations


def rank(token: str) -> int:
    """'sm_86' -> 86, so comparisons are plain integer comparisons."""
    return int(token.split("_")[1])


def evaluate(arch_list: list[str], cap: tuple[int, int]) -> tuple[bool, str, str]:
    """Decide whether a build with `arch_list` can run on a card with capability `cap`."""
    major, minor = cap
    mine = major * 10 + minor
    cudins = sorted({rank(a) for a in arch_list if a.startswith("sm_")})

    if not cudins:
        return False, "incompatible", "this build contains no compiled kernels at all"

    best = cudins[-1]

    if mine in cudins:
        return True, "native", f"native cubin for sm_{mine}"

    if best >= mine:
        lower = max(c for c in cudins if c <= mine)
        return True, "fallback", f"runs on the sm_{lower} cubin"

    return (
        False,
        "incompatible",
        f"every kernel in this build is compiled for sm_{best} or lower, "
        f"but this card is sm_{mine}",
    )


def evaluate_torch() -> tuple[bool, str, str, str]:
    """Convenience wrapper: evaluate the torch that is currently importable.

    Returns (usable, reason, detail, version) where version is torch's version string.
    Import failures come back as usable=False.
    """
    try:
        import torch
    except Exception as exc:                                    # noqa: BLE001
        return False, "no-cuda", f"torch could not be imported: {exc}", "?"

    version = torch.__version__
    if not torch.cuda.is_available():
        return False, "no-cuda", "torch reports no usable CUDA device", version

    cap = torch.cuda.get_device_capability(0)
    usable, reason, detail = evaluate(list(torch.cuda.get_arch_list()), cap)
    return usable, reason, detail, version


if __name__ == "__main__":
    usable, reason, detail, version = evaluate_torch()
    print(f"{'OK' if usable else 'FAIL'}|torch {version}: {detail} ({reason})")
    raise SystemExit(0 if usable else 1)
