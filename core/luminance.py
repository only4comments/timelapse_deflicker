"""
Luminance extraction and correction routines.

All operations work on float32 RGB images normalised to [0, 1].
Y channel is computed using Rec.709 coefficients throughout.
"""
from __future__ import annotations

import numpy as np

# Rec.709 RGB → Y (luminance) coefficients
_KR = 0.2126
_KG = 0.7152
_KB = 0.0722

# Rec.709 RGB → Cb/Cr (used to split and reconstruct colour)
# Cb = -0.1146 R - 0.3854 G + 0.5000 B
# Cr =  0.5000 R - 0.4542 G - 0.0458 B
_CB_R, _CB_G, _CB_B = -0.1146, -0.3854, 0.5000
_CR_R, _CR_G, _CR_B =  0.5000, -0.4542, -0.0458

# Rec.709 YCbCr → RGB
# R = Y                + 1.5748 Cr
# G = Y - 0.1873 Cb   - 0.4681 Cr
# B = Y + 1.8556 Cb
_R_Y,  _R_CR          =  1.0,  1.5748
_G_Y,  _G_CB, _G_CR   =  1.0, -0.1873, -0.4681
_B_Y,  _B_CB           =  1.0,  1.8556

N_HISTOGRAM_BINS = 1024  # used for histogram matching mode


def _to_ycbcr(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split float32 RGB image into (Y, Cb, Cr) planes."""
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y  = _KR * R + _KG * G + _KB * B
    Cb = _CB_R * R + _CB_G * G + _CB_B * B
    Cr = _CR_R * R + _CR_G * G + _CR_B * B
    return Y, Cb, Cr


def _from_ycbcr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """Reconstruct float32 RGB from YCbCr planes. Result is clipped to [0,1]."""
    R = _R_Y * Y                       + _R_CR * Cr
    G = _G_Y * Y + _G_CB * Cb + _G_CR * Cr
    B = _B_Y * Y + _B_CB * Cb
    return np.clip(np.stack([R, G, B], axis=2), 0.0, 1.0).astype(np.float32)


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
    Y, _, _ = _to_ycbcr(img)
    flat = Y.ravel()
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
    Y, _, _ = _to_ycbcr(img)
    counts, _ = np.histogram(Y.ravel(), bins=N_HISTOGRAM_BINS, range=(0.0, 1.0))
    total = float(counts.sum())
    return counts.astype(np.float64) / (total or 1.0)


# ──────────────────────────────────────────── gamut mapping ──────────────────

def _gamut_clip(img: np.ndarray) -> np.ndarray:
    """
    Bring out-of-gamut pixels back into [0, 1] RGB by reducing chroma
    proportionally while keeping luminance (Y) constant, instead of
    hard-clamping each channel independently.

    Hard clamp:   R=0.95, G=1.08, B=0.85  →  R=0.95, G=1.00, B=0.85
                  (green advantage shrinks → bleaches)
    Gamut clip:   reduces Cb/Cr until the most-clipped channel hits 1.0
                  (hue preserved, only saturation gently reduced)

    Pixels already inside gamut pass through unchanged.
    """
    # Identify pixels where any channel exceeds 1.0
    over = np.any(img > 1.0, axis=2)          # (H, W) bool
    if not np.any(over):
        return np.clip(img, 0.0, 1.0)

    result = img.copy()
    R, G, B = img[over, 0], img[over, 1], img[over, 2]

    # Luminance of the out-of-gamut pixels (Rec.709)
    Y_oog = _KR * R + _KG * G + _KB * B       # scalar luma target

    # Maximum channel value — this tells us by how much we need to shrink
    max_ch = np.maximum.reduce([R, G, B])      # (N,)

    # Scale: bring max channel to 1.0; Y is preserved because we scale
    # the full RGB triplet toward the grey point (Y, Y, Y).
    # Derivation: corrected = Y + scale * (RGB - Y)
    #   max(corrected) = 1.0  →  scale = (1 - Y) / (max_ch - Y)
    # When max_ch == Y (already grey) no correction needed.
    denom = max_ch - Y_oog
    safe = denom > 1e-6
    scale = np.where(safe, (1.0 - Y_oog) / denom, 1.0)
    scale = np.clip(scale, 0.0, 1.0)          # never amplify

    # Apply: blend toward the grey point
    result[over, 0] = Y_oog + scale * (R - Y_oog)
    result[over, 1] = Y_oog + scale * (G - Y_oog)
    result[over, 2] = Y_oog + scale * (B - Y_oog)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────── correction routines ─────────────

def apply_luminance_scaling(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale the Y channel by *factor* and also scale Cb/Cr by the same factor
    so the chroma/luma ratio stays constant (no saturation shift).
    Out-of-gamut pixels are brought back using gamut-aware chroma reduction
    rather than hard per-channel clamping.
    """
    Y, Cb, Cr = _to_ycbcr(img)
    Y_new  = Y  * factor
    Cb_new = Cb * factor
    Cr_new = Cr * factor
    raw_rgb = _from_ycbcr(Y_new, Cb_new, Cr_new)   # may be out of gamut
    return _gamut_clip(raw_rgb)


def apply_histogram_matching(
    img: np.ndarray,
    ref_hist: np.ndarray,
) -> np.ndarray:
    """
    Match the Y-channel histogram of *img* to the reference distribution
    given by *ref_hist* (raw counts or normalised; will be normalised here).
    Cb/Cr channels are unchanged, preserving hue.
    Out-of-gamut pixels after reconstruction are handled by gamut-aware
    chroma reduction instead of hard per-channel clamping.

    ref_hist : float array of shape (N_HISTOGRAM_BINS,)
    """
    Y, Cb, Cr = _to_ycbcr(img)

    # Source CDF
    src_counts, _ = np.histogram(Y.ravel(), bins=N_HISTOGRAM_BINS, range=(0.0, 1.0))
    src_cdf = np.cumsum(src_counts.astype(np.float64))
    src_cdf /= src_cdf[-1] + 1e-12

    # Reference CDF
    ref_cdf = np.cumsum(ref_hist.astype(np.float64))
    ref_cdf /= ref_cdf[-1] + 1e-12

    # Map: for each source bin level, find the reference level with matching CDF
    bin_centers = np.linspace(0.0, 1.0, N_HISTOGRAM_BINS)
    lut = np.interp(src_cdf, ref_cdf, bin_centers)

    # Quantise Y to bin indices and apply LUT
    indices = np.clip((Y * N_HISTOGRAM_BINS).astype(np.int32), 0, N_HISTOGRAM_BINS - 1)
    Y_new = lut[indices].astype(np.float32)

    raw_rgb = _from_ycbcr(Y_new, Cb, Cr)      # may be out of gamut
    return _gamut_clip(raw_rgb)
