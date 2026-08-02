"""
Timelapse Deflicker Tool — main entry point.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _hide_console() -> None:
    """
    Hide the Windows console window when launching via python.exe.
    With ThreadPoolExecutor there are no subprocesses, so only the main
    process console needs to be hidden.
    """
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)   # SW_HIDE
            kernel32.FreeConsole()
    except Exception:
        pass


def main() -> None:
    if sys.platform == "win32":
        _hide_console()
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
