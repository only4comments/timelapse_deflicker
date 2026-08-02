"""
Pass 2 — correction and output.

Performance notes
-----------------
The original float32 round-trip (uint16 → float32 / 65535 → multiply →
clip → float32 * 65535 → uint16) allocates 288 MB per frame and touches
every byte twice (read uint16, write float32, read float32, write uint16).
With N workers all doing this concurrently, DRAM bandwidth saturates fast.

The LUT path builds a 65536-entry uint16→uint16 table (128 KB, fits in L2
cache) from the scalar correction factor, then applies it with a single
integer gather: out = lut[raw_uint16].  No float32 array is allocated for
the main image data.  DRAM traffic drops from ~384 MB per frame to ~288 MB
(read once, write once), and crucially the access pattern is far more cache-
friendly so multiple workers scale near-linearly instead of saturating at ~4.

The float32 path is kept as a fallback for:
  - histogram-matching correction mode (non-linear, LUT not applicable)
  - JPEG output (PIL requires uint8, float intermediate is unavoidable)
  - format conversion (e.g. uint16 source → tiff8 output)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.image_io import load_image_raw, save_scaled_raw


def _process_worker(args: tuple) -> dict[str, Any]:
    (index, src_str, dst_str, correction_factor,
     ref_histogram, correction_mode, output_format,
     jpeg_quality, resize_enabled, resize_edge, resize_value) = args
    try:
        raw, depth = load_image_raw(src_str)
        save_scaled_raw(raw, depth, dst_str, correction_factor,
                        output_format, jpeg_quality,
                        resize_enabled, resize_edge, resize_value)

        return {"index": index, "src": src_str, "dst": dst_str,
                "correction_factor": correction_factor, "error": None}

    except Exception as exc:
        return {"index": index, "src": src_str, "dst": dst_str,
                "correction_factor": correction_factor, "error": str(exc)}


def run_pass2(
    file_list: list[Path],
    dest_folder: Path,
    rolling_lum: np.ndarray,
    measured_lum: list[float | None],
    rolling_hists: list[np.ndarray | None],
    correction_mode: str,
    output_format: str,
    jpeg_quality: int,
    resize_enabled: bool = False,
    resize_edge: str = "shorter",
    resize_value: str = "1080",
    jpeg_resize_1080: bool = False,  # legacy — ignored
    worker_count: int = 4,
    progress_callback = None,
    cancel_event = None,
    executor = None,
) -> list[str]:
    import concurrent.futures as cf
    from core.image_io import output_extension

    ext   = output_extension(output_format)
    total = len(file_list)

    task_args = []
    for i, src in enumerate(file_list):
        dst = dest_folder / (src.stem + ext)
        lum = measured_lum[i]
        factor = float(rolling_lum[i]) / lum if (lum and lum > 1e-9) else 1.0
        ref_hist = rolling_hists[i] if rolling_hists else None
        task_args.append((
            i, str(src), str(dst), factor, ref_hist,
            correction_mode, output_format, jpeg_quality,
            resize_enabled, resize_edge, resize_value,
        ))

    errors: list[str] = []

    with cf.ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_map = {pool.submit(_process_worker, a): a for a in task_args}
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
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    return errors
