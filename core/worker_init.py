"""
Worker process initialiser — called once per subprocess at pool startup.

Sets OS priority to Below Normal so the tool stays in the background.

Also suppresses numpy/BLAS internal threading.  Without this, each worker
process spawns its own OpenBLAS/MKL thread pool (8-16 threads by default).
With N worker processes, that produces N*8 threads competing for N physical
cores — causing massive context switching and cache thrashing.  For the
image-processing work done here (large sequential array ops, not BLAS
matrix math), 1 thread per process is strictly better: each worker gets
exclusive use of its core's L1/L2 cache, and DRAM bandwidth is shared
cleanly between workers rather than wasted on thread coordination.

These env vars must be set BEFORE numpy is first imported in the process,
which is why they live here in the initialiser rather than in the worker
functions themselves.
"""
from __future__ import annotations
import os


def set_low_priority() -> None:
    # ── suppress numpy/BLAS internal threading ────────────────────────────────
    # Must happen before any numpy import in this process.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"

    # ── OS priority ───────────────────────────────────────────────────────────
    import sys

    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_SET_INFORMATION = 0x0200
            BELOW_NORMAL_PRIORITY   = 0x00004000
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
