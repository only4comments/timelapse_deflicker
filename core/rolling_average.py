"""
Rolling (sliding) window average helpers.

Window is centred on each frame and truncated at the sequence boundaries —
no padding or mirroring.
"""
from __future__ import annotations

import numpy as np


def rolling_average(values: list[float] | np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute a centred rolling average of *values* with *window_size*.
    Edge frames use a truncated window (fewer samples, not padded).
    Returns float64 array of the same length.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    half = window_size // 2
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = arr[lo:hi].mean()
    return result


def rolling_average_histograms(
    histograms: list[np.ndarray],
    window_size: int,
) -> list[np.ndarray]:
    """
    Compute a centred rolling average of histogram arrays.
    Each element of *histograms* must have the same shape.
    Returns a list of float64 arrays (same length as input).
    """
    n = len(histograms)
    half = window_size // 2
    result: list[np.ndarray] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = histograms[lo:hi]
        avg = np.mean(np.stack(window, axis=0), axis=0)
        result.append(avg.astype(np.float64))
    return result
