"""
Luminance extraction and correction routines.

All operations work on float32 RGB images normalised to [0, 1].
Y channel is computed using Rec.709 coefficients throughout.
"""
from __future__ import annotations

import numpy as np

# Rec.709 RGB -> Y (luminance) coefficients
_KR = 0.2126
_KG = 0.7152
_KB = 0.0722

N_HISTOGRAM_BINS = 1024  # used for histogram matching mode


def _y(img: np.ndarray) -> np.ndarray:
    """Compute Rec.709 luminance (Y) plane from float32 RGB image."""
    return _KR * img[:, :, 0] + _KG * img[:, :, 1] + _KB * img[:, :, 2]


# ─────────────────────────────────────────── luminance measurement ────────────

def measure_luminance(
    img: np.ndarray,
    metric: str,
    percentile_value: float = 95.0,
) -> float:
    """
    Return a single luminance scalar for an image.
    metric: "mean" | "median" | "percentile"
    """
    flat = _y(img).ravel()
    if metric == "mean":
        return float(np.mean(flat))
    if metric == "median":
        return float(np.median(flat))
    if metric == "percentile":
        return float(np.percentile(flat, percentile_value))
    raise ValueError(f"Unknown metric: {metric!r}")


def compute_y_histogram(img: np.ndarray) -> np.ndarray:
    """
    Compute a normalised Y-channel histogram with N_HISTOGRAM_BINS bins.
    Returns float64 array of shape (N_HISTOGRAM_BINS,) summing to 1.
    """
    Y = _y(img)
    counts, _ = np.histogram(Y.ravel(), bins=N_HISTOGRAM_BINS, range=(0.0, 1.0))
    total = float(counts.sum())
    return counts.astype(np.float64) / (total or 1.0)


def _hist_median(hist: np.ndarray) -> float:
    """
    Return the median value implied by a normalised histogram.
    i.e. the bin-centre where the CDF first reaches 0.5.
    """
    cdf = np.cumsum(hist)
    cdf /= cdf[-1] + 1e-12
    bin_centers = np.linspace(0.0, 1.0, len(hist))
    return float(np.interp(0.5, cdf, bin_centers))


# ──────────────────────────────────────────── correction routines ─────────────

def apply_luminance_scaling(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Multiply every RGB channel by *factor* and clip to [0, 1].

    A uniform scale is the only brightness operation that leaves white
    balance intact: every channel moves by the same multiplier so the
    R:G:B ratios are preserved exactly at every pixel.
    """
    return np.clip(img * factor, 0.0, 1.0).astype(np.float32)


def apply_histogram_matching(
    img: np.ndarray,
    ref_hist: np.ndarray,
) -> np.ndarray:
    """
    Correct the brightness of *img* to match the luminance level described
    by *ref_hist*, without introducing any colour cast.

    Why a global scale instead of a per-pixel tone curve
    -----------------------------------------------------
    A full CDF-based tone-curve remap (the classic histogram-equalisation
    approach) redistributes values non-uniformly across the tonal range:
    it boosts shadows more than highlights (or vice versa) to force the
    output histogram to match the reference shape.  Because different
    colours live at different tonal levels (e.g. blue sky sits in the
    highlights, green grass in the mid-shadows), this differential tonal
    reshaping amplifies channels at different rates — producing a
    systematic colour cast even when all three channels are scaled by the
    same per-pixel factor Y_new/Y_old.

    The correct operation for deflickering is to change *brightness only*,
    not to reshape the tonal curve.  We therefore extract a single scalar
    from the reference histogram — the median luminance — and divide it by
    the source image's median luminance to get a global scale factor.  That
    scalar is then applied uniformly to all three RGB channels, which
    preserves white balance and R:G:B ratios exactly.

    Using the histogram median (rather than the raw scalar already stored
    in rolling_lum) gives a more robust luminance estimate on high-contrast
    or blown-highlight scenes, which is the sole reason the histogram path
    exists.

    ref_hist : float array of shape (N_HISTOGRAM_BINS,), raw counts or
               normalised — will be normalised internally.
    """
    src_median = float(np.median(_y(img).ravel()))
    ref_median = _hist_median(ref_hist)

    if src_median < 1e-6:
        return img.copy()

    factor = ref_median / src_median
    return apply_luminance_scaling(img, factor)
