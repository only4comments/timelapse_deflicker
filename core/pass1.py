"""
Pass 1 — luminance analysis.

Optimisations
-------------
1. mmap-based TIFF reading (_load_raw_tiff_fast) — single OS call instead
   of one seek/read/frombuffer per strip.
2. Mean-metric shortcut — channel means computed directly from uint16,
   no float32 intermediate.
3. Executor passed in from persistent WorkerPool — no re-spawn per pass.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


# ── worker (runs in subprocess) ───────────────────────────────────────────────

def _analyze_worker(args: tuple) -> dict[str, Any]:
    index, filepath_str, metric, percentile_value, need_histogram = args
    try:
        import numpy as np
        from pathlib import Path as _P
        from core.image_io import _load_raw_tiff_fast, load_y_channel_float
        from core.luminance import N_HISTOGRAM_BINS

        fpath = _P(filepath_str)
        ext   = fpath.suffix.lower()

        # ── mean on TIFF — no float32 conversion at all ───────────────────────
        if metric == "mean" and ext in (".tif", ".tiff") and not need_histogram:
            raw   = _load_raw_tiff_fast(fpath)
            scale = 1.0 / (65535.0 if raw.dtype == np.uint16 else 255.0)
            if raw.ndim == 3:
                ch  = raw.mean(axis=(0, 1))
                lum = float(0.2126*ch[0] + 0.7152*ch[1] + 0.0722*ch[2]) * scale
            else:
                lum = float(raw.mean()) * scale
            return {"index": index, "filepath": filepath_str,
                    "luminance": lum, "histogram": None, "error": None}

        # ── general path: median / percentile / histogram ─────────────────────
        Y    = load_y_channel_float(filepath_str)
        flat = Y.ravel()

        if metric == "mean":
            lum = float(np.mean(flat))
        elif metric == "median":
            lum = float(np.median(flat))
        elif metric == "percentile":
            lum = float(np.percentile(flat, percentile_value))
        else:
            lum = float(np.mean(flat))

        hist = None
        if need_histogram:
            counts, _ = np.histogram(flat, bins=N_HISTOGRAM_BINS,
                                     range=(0.0, 1.0))
            total = float(counts.sum())
            hist  = counts.astype(np.float64) / (total or 1.0)

        return {"index": index, "filepath": filepath_str,
                "luminance": lum, "histogram": hist, "error": None}

    except Exception as exc:
        return {"index": index, "filepath": filepath_str,
                "luminance": None, "histogram": None, "error": str(exc)}


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_pass1(
    file_list: list[Path],
    metric: str,
    percentile_value: float,
    need_histogram: bool,
    worker_count: int,
    progress_callback,
    cancel_event,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[list[float | None], list[Any]]:
    import concurrent.futures as cf

    total     = len(file_list)
    task_args = [
        (i, str(f), metric, percentile_value, need_histogram)
        for i, f in enumerate(file_list)
    ]

    luminance_values: list = [None] * total
    histograms:       list = [None] * total

    def _run(ex):
        future_map = {ex.submit(_analyze_worker, a): a for a in task_args}
        done = 0
        for future in cf.as_completed(future_map):
            if cancel_event.is_set():
                for f in future_map:
                    f.cancel()
                break
            result = future.result()
            idx = result["index"]
            luminance_values[idx] = result["luminance"]
            histograms[idx]       = result["histogram"]
            done += 1
            progress_callback(done, total, result)

    if executor is not None:
        _run(executor)
    else:
        from core.worker_init import set_low_priority
        with cf.ProcessPoolExecutor(max_workers=worker_count,
                                    initializer=set_low_priority) as ex:
            _run(ex)

    return luminance_values, histograms
