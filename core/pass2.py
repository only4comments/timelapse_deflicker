"""
Pass 2 — correction and output.

Worker functions at module level for picklability.
The executor is passed in from outside (same warm pool as Pass 1).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


# ── worker (runs in subprocess) ───────────────────────────────────────────────

def _process_worker(args: tuple) -> dict[str, Any]:
    (index, src_str, dst_str, correction_factor, ref_histogram,
     correction_mode, output_format, jpeg_quality, jpeg_resize_1080) = args
    try:
        from core.image_io import load_image_float, save_image_float
        from core.luminance import apply_luminance_scaling, apply_histogram_matching

        img = load_image_float(src_str)
        corrected = (
            apply_luminance_scaling(img, correction_factor)
            if correction_mode == "scaling"
            else apply_histogram_matching(img, ref_histogram)
        )
        save_image_float(corrected, dst_str, output_format,
                         jpeg_quality, jpeg_resize_1080)

        return {"index": index, "src": src_str, "dst": dst_str,
                "correction_factor": correction_factor, "error": None}
    except Exception as exc:
        return {"index": index, "src": src_str, "dst": dst_str,
                "correction_factor": correction_factor, "error": str(exc)}


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_pass2(
    file_list: list[Path],
    dest_folder: Path,
    rolling_lum: np.ndarray,
    measured_lum: list[float | None],
    rolling_hists: list[np.ndarray | None],
    correction_mode: str,
    output_format: str,
    jpeg_quality: int,
    jpeg_resize_1080: bool,
    worker_count: int,
    progress_callback,
    cancel_event,
    executor: ProcessPoolExecutor | None = None,
) -> list[str]:
    """
    Run Pass 2.  *executor* is used and NOT shut down if provided.
    """
    import concurrent.futures as cf
    from core.image_io import output_extension

    ext   = output_extension(output_format)
    total = len(file_list)

    task_args = []
    for i, src in enumerate(file_list):
        dst = dest_folder / (src.stem + ext)
        lum = measured_lum[i]
        factor = float(rolling_lum[i]) / lum if (lum and lum > 1e-9) else 1.0
        task_args.append((
            i, str(src), str(dst), factor,
            rolling_hists[i] if rolling_hists else None,
            correction_mode, output_format, jpeg_quality, jpeg_resize_1080,
        ))

    errors: list[str] = []

    def _run(ex):
        future_map = {ex.submit(_process_worker, a): a for a in task_args}
        done = 0
        try:
            for future in cf.as_completed(future_map):
                if cancel_event.is_set():
                    for f in future_map:
                        f.cancel()
                    break
                result = future.result()
                if result["error"]:
                    errors.append(
                        f"Frame {result['index']} "
                        f"({Path(result['src']).name}): {result['error']}"
                    )
                done += 1
                progress_callback(done, total, result)
        except Exception:
            raise

    if executor is not None:
        _run(executor)
    else:
        from core.worker_init import set_low_priority
        with cf.ProcessPoolExecutor(max_workers=worker_count,
                                    initializer=set_low_priority) as ex:
            _run(ex)

    return errors
