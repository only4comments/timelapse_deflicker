"""
Worker process initialiser — called once per subprocess at pool startup,
and once on the main process from main.py.

Sets priority to Below Normal so the tool stays in the background while
the desktop remains responsive.

The pseudo-handle returned by GetCurrentProcess() can be truncated on
64-bit Windows when ctypes uses its default 32-bit return type, causing
SetPriorityClass to silently fail.  Using OpenProcess() with the real PID
gives a proper 64-bit HANDLE and is reliable on all Windows versions.
"""
from __future__ import annotations


def set_low_priority() -> None:
    import sys
    import os

    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_SET_INFORMATION  = 0x0200
            BELOW_NORMAL_PRIORITY    = 0x00004000

            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_SET_INFORMATION, False, os.getpid()
            )
            if handle:
                ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY)
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass
    else:
        try:
            os.nice(10)
        except Exception:
            pass
