"""Pick a portrait with a file dialog, then start the live session.

Convenience wrapper so the launcher does not need the path typed on the command
line. Kept as a separate script (rather than importing the webcam tool) so the
webcam tool stays usable without a GUI dependency when driven from a host.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ask_for_portrait() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:                                    # noqa: BLE001
        print(f"tkinter unavailable ({exc}); pass --source instead")
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Choose a portrait to animate (frontal face, whole head visible)",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        initialdir=str(ROOT / "portraits") if (ROOT / "portraits").is_dir() else str(ROOT),
    )
    root.destroy()
    return Path(path) if path else None


def main() -> int:
    portrait = ask_for_portrait()
    if portrait is None:
        print("no portrait chosen - cancelled")
        return 1
    if not portrait.exists():
        print(f"not found: {portrait}")
        return 1
    print(f"portrait: {portrait}")

    seconds = sys.argv[1] if len(sys.argv) > 1 else "60"
    cmd = [sys.executable, "-u", str(ROOT / "tools" / "run_webcam.py"),
           "--source", str(portrait), "--display", "--seconds", seconds]
    # Run in this process's child so the console stays attached and the caller's
    # error handling still sees the exit code.
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
