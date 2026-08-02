"""
Image reading and writing.

All images are normalised to float32 in [0, 1] internally for the general
path.  The hot path for 16-bit TIFF output uses a uint16 LUT to avoid
allocating a 288 MB float32 intermediate array per frame — see
save_image_scaled_u16().

Shape is always (H, W, 3) — grayscale/RGBA inputs are converted to RGB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".tif", ".tiff", ".jpg", ".jpeg"}
)


# ─────────────────────────────────────────────── loading ──────────────────────

def load_image_float(filepath: str | Path) -> np.ndarray:
    """
    Load any supported image and return float32 RGB array normalised to [0, 1].
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".tif", ".tiff"):
        raw = tifffile.imread(str(filepath))
    else:
        pil_img = Image.open(str(filepath)).convert("RGB")
        raw = np.array(pil_img, dtype=np.uint8)

    if raw.dtype == np.uint8:
        img = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        img = raw.astype(np.float32) / 65535.0
    elif raw.dtype in (np.float32, np.float64):
        img = raw.astype(np.float32)
    else:
        try:
            max_val = float(np.iinfo(raw.dtype).max)
        except ValueError:
            max_val = float(raw.max()) or 1.0
        img = raw.astype(np.float32) / max_val

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def load_image_raw(filepath: str | Path) -> tuple[np.ndarray, int]:
    """
    Load a TIFF or JPEG and return (raw_array, bit_depth) without converting
    to float32.  Used by the fast LUT path in Pass 2 to avoid the expensive
    uint16→float32 conversion when writing back to uint16 TIFF.

    Returns:
        raw   : uint8 or uint16 ndarray of shape (H, W, 3)
        depth : 8 or 16
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".tif", ".tiff"):
        raw = tifffile.imread(str(filepath))
        depth = 16 if raw.dtype == np.uint16 else 8
    else:
        pil_img = Image.open(str(filepath)).convert("RGB")
        raw = np.array(pil_img, dtype=np.uint8)
        depth = 8

    # Ensure (H, W, 3)
    if raw.ndim == 2:
        raw = np.stack([raw, raw, raw], axis=2)
    elif raw.shape[2] == 1:
        raw = np.concatenate([raw, raw, raw], axis=2)
    elif raw.shape[2] == 4:
        raw = raw[:, :, :3]

    return raw, depth


def _apply_resize(
    pil_img: "Image.Image",
    resize_enabled: bool,
    resize_edge: str,
    resize_value: str,
) -> "Image.Image":
    """
    Proportionally resize *pil_img* according to the resize settings.

    resize_edge  : "shorter" | "longer"
    resize_value : pixel count as "1080", or percentage as "50%"
    Never upscales.
    """
    if not resize_enabled:
        return pil_img

    from gui.settings_panel import _parse_resize_value
    parsed = _parse_resize_value(resize_value)
    if parsed is None:
        return pil_img

    w, h = pil_img.size
    shorter = min(w, h)
    longer  = max(w, h)

    if isinstance(parsed, float):
        # Percentage — apply to both dimensions, preserving ratio
        factor = parsed / 100.0
        new_w  = max(1, round(w * factor))
        new_h  = max(1, round(h * factor))
    else:
        # Pixel target on the chosen edge
        target = parsed
        ref    = shorter if resize_edge == "shorter" else longer
        if ref <= target:
            return pil_img   # never upscale
        factor = target / ref
        new_w  = max(1, round(w * factor))
        new_h  = max(1, round(h * factor))

    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    filepath = Path(filepath)
    if filepath.suffix.lower() in (".tif", ".tiff"):
        raw = tifffile.imread(str(filepath))
        return 16 if raw.dtype == np.uint16 else 8
    return 8


# ─────────────────────────────────────────────── saving ───────────────────────

def save_image_float(
    img_float: np.ndarray,
    filepath: str | Path,
    output_format: str,
    jpeg_quality: int = 85,
    resize_enabled: bool = False,
    resize_edge: str = "shorter",
    resize_value: str = "1080",
    jpeg_resize_1080: bool = False,  # legacy — ignored
) -> None:
    """Save a float32 [0, 1] RGB image (general path)."""
    filepath  = Path(filepath)
    img_float = np.clip(img_float, 0.0, 1.0)

    if output_format == "jpeg":
        arr     = (img_float * 255.0 + 0.5).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="RGB")
        pil_img = _apply_resize(pil_img, resize_enabled, resize_edge, resize_value)
        pil_img.save(str(filepath), format="JPEG",
                     quality=jpeg_quality, subsampling=0)

    elif output_format == "tiff8":
        arr     = (img_float * 255.0 + 0.5).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="RGB")
        pil_img = _apply_resize(pil_img, resize_enabled, resize_edge, resize_value)
        tifffile.imwrite(str(filepath), np.array(pil_img),
                         compression=None, photometric="rgb")

    elif output_format == "tiff16":
        arr = (img_float * 65535.0 + 0.5).astype(np.uint16)
        if resize_enabled:
            # PIL doesn't support uint16 directly; resize each channel in float32
            from gui.settings_panel import _parse_resize_value as _prv
            parsed = _prv(resize_value)
            if parsed is not None:
                h, w = img_float.shape[:2]
                shorter, longer = min(w, h), max(w, h)
                if isinstance(parsed, float):
                    factor = parsed / 100.0
                    nw, nh = max(1, round(w*factor)), max(1, round(h*factor))
                else:
                    ref = shorter if resize_edge == "shorter" else longer
                    if ref > parsed:
                        factor = parsed / ref
                        nw, nh = max(1, round(w*factor)), max(1, round(h*factor))
                    else:
                        nw, nh = w, h
                if (nw, nh) != (w, h):
                    pil16 = Image.fromarray(arr, mode="I;16") if arr.ndim == 2 else None
                    # Resize via float32 to preserve 16-bit quality
                    resized_f = np.zeros((nh, nw, 3), dtype=np.float32)
                    for c in range(3):
                        ch_pil = Image.fromarray(
                            img_float[:, :, c], mode="F"
                        ).resize((nw, nh), Image.Resampling.LANCZOS)
                        resized_f[:, :, c] = np.array(ch_pil)
                    arr = np.clip(resized_f * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        tifffile.imwrite(str(filepath), arr, compression=None, photometric="rgb")

    else:
        raise ValueError(f"Unknown output format: {output_format!r}")


def build_lut_u8(factor: float) -> np.ndarray:
    """256-entry uint8→uint8 LUT for a brightness scale factor."""
    return np.clip(
        np.arange(256, dtype=np.float32) * factor + 0.5,
        0, 255,
    ).astype(np.uint8)


def build_lut_u16(factor: float) -> np.ndarray:
    """65536-entry uint16→uint16 LUT for a brightness scale factor.

    The LUT fits in 128 KB — comfortably inside L2 cache on any modern CPU.
    Applying it with raw[lut] is a pure integer gather operation, avoiding
    the 288 MB float32 intermediate that the general path allocates.
    This reduces DRAM traffic by ~3x per frame and scales near-linearly
    with worker count, unlike the float32 path which saturates DRAM bandwidth
    at ~4 concurrent workers.
    """
    return np.clip(
        np.arange(65536, dtype=np.float32) * factor + 0.5,
        0, 65535,
    ).astype(np.uint16)


def save_scaled_raw(
    raw: np.ndarray,
    depth: int,
    filepath: str | Path,
    factor: float,
    output_format: str,
    jpeg_quality: int = 85,
    resize_enabled: bool = False,
    resize_edge: str = "shorter",
    resize_value: str = "1080",
    jpeg_resize_1080: bool = False,  # legacy — ignored
) -> None:
    """
    Apply a scalar brightness factor to a raw uint8/uint16 image and save,
    using a LUT to avoid float32 allocation for TIFF→TIFF paths.
    """
    filepath = Path(filepath)

    # ── fast LUT path: uint16 source → uint16 TIFF output ────────────────────
    if depth == 16 and output_format == "tiff16" and raw.dtype == np.uint16:
        lut = build_lut_u16(factor)
        out = lut[raw]
        if resize_enabled:
            pil_img = Image.fromarray(
                (out.astype(np.float32) / 65535.0 * 255.0 + 0.5).astype(np.uint8),
                mode="RGB"
            )
            pil_img = _apply_resize(pil_img, True, resize_edge, resize_value)
            out = (np.array(pil_img).astype(np.float32) / 255.0 * 65535.0 + 0.5
                   ).astype(np.uint16)
        tifffile.imwrite(str(filepath), out, compression=None, photometric="rgb")
        return

    # ── fast LUT path: uint8 source → JPEG or tiff8 output ───────────────────
    if depth == 8 and output_format in ("jpeg", "tiff8") and raw.dtype == np.uint8:
        lut     = build_lut_u8(factor)
        out     = lut[raw]
        pil_img = Image.fromarray(out, mode="RGB")
        pil_img = _apply_resize(pil_img, resize_enabled, resize_edge, resize_value)
        if output_format == "tiff8":
            tifffile.imwrite(str(filepath), np.array(pil_img),
                             compression=None, photometric="rgb")
        else:
            pil_img.save(str(filepath), format="JPEG",
                         quality=jpeg_quality, subsampling=0)
        return

    # ── general float32 fallback (format conversion or edge cases) ────────────
    if raw.dtype == np.uint8:
        img_float = raw.astype(np.float32) / 255.0
    else:
        img_float = raw.astype(np.float32) / 65535.0
    from core.luminance import apply_luminance_scaling
    corrected = apply_luminance_scaling(img_float, factor)
    save_image_float(corrected, filepath, output_format,
                     jpeg_quality, resize_enabled, resize_edge, resize_value)


def output_extension(output_format: str) -> str:
    return {"jpeg": ".jpg", "tiff8": ".tif", "tiff16": ".tif"}[output_format]
