"""
Timelapse Deflicker Tool — main entry point.
Must be run as __main__ on Windows for multiprocessing to work correctly.
"""
import multiprocessing
import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of the working directory
# from which the script is launched.  This is especially important on Windows
# when double-clicking main.py or running it from a different cwd.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> None:
    # Required for Windows: prevents recursive subprocess spawning.
    multiprocessing.freeze_support()

    # Run the whole process at Below-Normal priority so it doesn't
    # compete with the desktop while crunching large sequences.
    from core.worker_init import set_low_priority
    set_low_priority()

    # Enable per-monitor DPI awareness so the UI is sharp on HiDPI displays.
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    import tkinter as tk
    from gui.main_window import MainWindow

    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
