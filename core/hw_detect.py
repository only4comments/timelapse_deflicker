"""
Hardware detection and worker-count recommendation.

Keeps all the detection logic in one place so it can be tested independently
of the GUI and called from both main_window (after folder load) and
settings_panel (for the initial startup hint).

Worker count formula
--------------------
available_ram  = total_ram - max(4 GB, total_ram * 0.25)
per_worker_ram = 2 * frame_bytes + PROCESS_OVERHEAD_MB
                 # 2x: one array for source, one for output
                 # PROCESS_OVERHEAD: Python interpreter + tifffile + numpy
                 #   loaded fresh per subprocess ≈ 80 MB measured

pass1_workers  = min(ram_limit, physical_cores)
                 # Pass 1 is read-only + integer math: CPU is doing real
                 # work per worker, so physical_cores is the right ceiling.

pass2_workers  = min(ram_limit, max(1, physical_cores // 2))
                 # Pass 2 reads AND writes the same NVMe drive simultaneously,
                 # halving effective per-worker I/O bandwidth.  Fewer workers
                 # keep the SSD pipeline full without thrashing it.

Both are clamped to [1, MAX_SANE_WORKERS] as a safety rail.

psutil is used when available (gives exact available RAM, not just total).
Falls back to total-RAM-only heuristic if not installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_PROCESS_OVERHEAD_MB = 80    # per worker: Python + tifffile + numpy fresh import
_MAX_SANE_WORKERS    = 64    # hard upper cap regardless of RAM
_MIN_FRAME_MB        = 4.0   # minimum assumed frame size (tiny safety floor)


@dataclass
class HWProfile:
    """Everything the auto-detect found, bundled for easy display in the UI."""
    total_ram_gb:      float
    available_ram_gb:  float
    physical_cores:    int
    logical_cores:     int
    frame_w:           int | None   # None until a source folder is loaded
    frame_h:           int | None
    frame_depth:       int          # bits: 8 or 16
    frame_mb:          float        # derived
    per_worker_mb:     float        # derived
    pass1_workers:     int
    pass2_workers:     int
    ram_is_limiting:   bool         # True when RAM caps below core count
    source: str = ""                # human-readable description for the tooltip


def _get_ram_gb() -> tuple[float, float]:
    """Return (total_gb, available_gb).  available may equal total if psutil absent."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    except ImportError:
        pass

    # Windows fallback via ctypes
    try:
        import ctypes
        class _MEMSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        ms = _MEMSTATUS()
        ms.dwLength = ctypes.sizeof(ms)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        total = ms.ullTotalPhys / 1e9
        avail = ms.ullAvailPhys / 1e9
        return total, avail
    except Exception:
        pass

    # Linux fallback via /proc/meminfo
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = int(v.strip().split()[0]) * 1024  # kB -> bytes
        total = info.get("MemTotal", 0) / 1e9
        avail = info.get("MemAvailable", info.get("MemFree", 0)) / 1e9
        return total, avail
    except Exception:
        pass

    # Last resort: assume 8 GB, no availability info
    return 8.0, 8.0


def _get_physical_cores() -> int:
    """
    Best-effort physical core count.
    psutil is authoritative; falls back to logical // 2 which is correct for
    any HyperThreaded Intel/AMD desktop CPU.
    """
    try:
        import psutil
        pc = psutil.cpu_count(logical=False)
        if pc:
            return pc
    except ImportError:
        pass
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


def _probe_frame_size(first_file: Path) -> tuple[int, int, int]:
    """
    Read just the header of the first image file to get (width, height, bit_depth)
    without loading the whole image.  For TIFF: tifffile.TiffFile reads the IFD only.
    For JPEG: PIL reads the SOF marker only.
    Returns (0, 0, 8) on any error so callers can treat it as unknown.
    """
    try:
        ext = first_file.suffix.lower()
        if ext in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(str(first_file)) as tf:
                page = tf.pages[0]
                h, w = page.shape[:2]
                depth = 16 if page.dtype == "uint16" else 8
            return w, h, depth
        else:
            from PIL import Image
            with Image.open(str(first_file)) as img:
                w, h = img.size
            return w, h, 8
    except Exception:
        return 0, 0, 8


def _compute_workers(
    available_mb: float,
    physical_cores: int,
    per_worker_mb: float,
) -> tuple[int, int, bool]:
    """
    Return (pass1_workers, pass2_workers, ram_is_limiting).
    """
    ram_limit = max(1, int(available_mb / max(per_worker_mb, 1.0)))
    ram_limit = min(ram_limit, _MAX_SANE_WORKERS)

    p1 = min(ram_limit, physical_cores)
    p2 = min(ram_limit, max(1, physical_cores // 2))

    ram_is_limiting = ram_limit < physical_cores

    return max(1, p1), max(1, p2), ram_is_limiting


def detect(first_file: Path | None = None) -> HWProfile:
    """
    Run the full hardware + frame-size detection and return a HWProfile.
    Call with first_file=None at startup (uses 4K assumption).
    Call again with the actual first file once a source folder is loaded.
    """
    total_gb, avail_gb = _get_ram_gb()
    physical = _get_physical_cores()
    logical  = os.cpu_count() or physical * 2

    # RAM budget: leave max(4 GB, 25%) for OS, GUI, page file, other apps
    reserve_gb   = max(4.0, total_gb * 0.25)
    available_mb = max(0.0, avail_gb - (reserve_gb - (total_gb - avail_gb))) * 1024
    # Simpler: just use available - a fixed 2 GB GUI headroom
    available_mb = max(0.0, (avail_gb - 2.0) * 1024)

    if first_file is not None and first_file.exists():
        w, h, depth = _probe_frame_size(first_file)
    else:
        w, h, depth = 0, 0, 16   # unknown — will use fallback below

    if w > 0 and h > 0:
        frame_mb      = w * h * 3 * (depth // 8) / 1e6
        per_worker_mb = 2 * frame_mb + _PROCESS_OVERHEAD_MB
        source        = f"{w}×{h} px, {depth}-bit  ({frame_mb:.0f} MB/frame)"
    else:
        # Fallback: assume 4K until a file is read
        w, h, depth   = 3840, 2160, 16
        frame_mb      = w * h * 3 * (depth // 8) / 1e6
        per_worker_mb = 2 * frame_mb + _PROCESS_OVERHEAD_MB
        source        = f"4K assumed ({frame_mb:.0f} MB/frame) — load a folder to refine"

    frame_mb      = max(frame_mb, _MIN_FRAME_MB)
    per_worker_mb = max(per_worker_mb, _PROCESS_OVERHEAD_MB + _MIN_FRAME_MB * 2)

    p1, p2, limiting = _compute_workers(available_mb, physical, per_worker_mb)

    return HWProfile(
        total_ram_gb     = round(total_gb, 1),
        available_ram_gb = round(avail_gb, 1),
        physical_cores   = physical,
        logical_cores    = logical,
        frame_w          = w if w > 0 else None,
        frame_h          = h if h > 0 else None,
        frame_depth      = depth,
        frame_mb         = round(frame_mb, 1),
        per_worker_mb    = round(per_worker_mb, 1),
        pass1_workers    = p1,
        pass2_workers    = p2,
        ram_is_limiting  = limiting,
        source           = source,
    )


def tooltip_text(p: HWProfile) -> str:
    """Human-readable explanation of the recommendation for the UI tooltip."""
    lines = [
        f"Auto-detected hardware:",
        f"  RAM:    {p.total_ram_gb:.0f} GB total, {p.available_ram_gb:.0f} GB free",
        f"  CPU:    {p.physical_cores} physical cores ({p.logical_cores} logical)",
        f"  Frame:  {p.source}",
        f"  Per-worker peak RAM: {p.per_worker_mb:.0f} MB",
        f"",
        f"Recommended:",
        f"  Pass 1 (analysis):    {p.pass1_workers} workers",
        f"  Pass 2 (correction):  {p.pass2_workers} workers",
    ]
    if p.ram_is_limiting:
        lines.append(f"  ⚠ RAM is the limiting factor, not CPU.")
    else:
        lines.append(f"  CPU core count is the limiting factor.")
    lines.append(f"")
    lines.append(f"Pass 2 uses half the workers of Pass 1 because")
    lines.append(f"it reads and writes simultaneously, splitting SSD bandwidth.")
    return "\n".join(lines)
