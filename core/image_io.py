"""
Image reading and writing.

Two loading paths
-----------------
load_image_float(filepath)      Full float32 RGB [0,1] — used by Pass 2.
load_y_channel_float(filepath)  Y channel only, float32 [0,1] — used by Pass 1.
                                 Avoids creating the full float32 RGB intermediate,
                                 cutting per-frame memory ~3× and skipping one
                                 large numpy allocation.

Fast TIFF reading
-----------------
Camera TIFFs are typically written with ROWSPERSTRIP=1 (one strip per row).
tifffile's default path reads each strip in a Python loop, causing thousands
of seek/read/frombuffer calls per file — pure Python bytecode at 100% CPU
but very low IPC (no SIMD), which explains high %CPU with low wattage.

_load_raw_tiff_fast() maps the whole file into virtual memory with a single
mmap() call, then slices the pixel data directly — one OS call instead of
thousands.  Falls back to tifffile.imread() for compressed or exotic TIFFs.
"""
from __future__ import annotations

import mmap
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".tif", ".tiff", ".jpg", ".jpeg"}
)

# Rec.709 coefficients (same as luminance.py — duplicated here to avoid
# importing from luminance inside worker subprocesses unnecessarily)
_KR, _KG, _KB = 0.2126, 0.7152, 0.0722


# ─────────────────────────────────────── fast raw TIFF loader ─────────────────

def _load_raw_tiff_fast(filepath: Path) -> np.ndarray:
    """
    Return the raw pixel array (uint8 or uint16) for a single-page TIFF.

    Strategy
    --------
    1. Use tifffile only to parse metadata (IFD tags: offsets, byte counts,
       dtype, shape).  This is unavoidable but cheap compared to data loading.
    2. mmap() the entire file — one system call.
    3. Slice the pixel data directly from the memory map — no Python loops,
       no per-strip seek/read calls, full OS prefetch / cache benefits.
    4. For stripped TIFFs where strips are contiguous on disk, read the whole
       run in one slice.  For non-contiguous strips, slice per-strip but still
       via mmap (no seek overhead).
    5. Fall back to tifffile.imread() for compressed or multi-sample-per-pixel
       exotic layouts.
    """
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            if len(tif.pages) != 1:
                return tifffile.imread(str(filepath))

            page = tif.pages[0]

            # Only take the fast path for uncompressed data
            compression = page.compression
            if compression not in (
                tifffile.COMPRESSION.NONE,
                tifffile.COMPRESSION.RAW,       # alias in some versions
                1,                               # numeric value for no compression
            ):
                return tifffile.imread(str(filepath))

            dtype      = page.dtype
            shape      = page.shape            # (H, W) or (H, W, C)
            offsets    = page.dataoffsets      # tuple of ints
            bytecounts = page.databytecounts   # tuple of ints

            expected_bytes = int(np.prod(shape)) * dtype.itemsize
            actual_bytes   = sum(bytecounts)
            if actual_bytes != expected_bytes:
                # Unexpected layout — let tifffile handle it
                return tifffile.imread(str(filepath))

            with open(str(filepath), "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    if len(offsets) == 1:
                        # Single contiguous block — fastest case
                        start = offsets[0]
                        raw = np.frombuffer(
                            mm[start : start + bytecounts[0]], dtype=dtype
                        ).copy()      # .copy() detaches from the mmap

                    else:
                        # Multiple strips: check if contiguous on disk
                        contiguous = all(
                            offsets[i] + bytecounts[i] == offsets[i + 1]
                            for i in range(len(offsets) - 1)
                        )
                        if contiguous:
                            start = offsets[0]
                            end   = offsets[-1] + bytecounts[-1]
                            raw   = np.frombuffer(
                                mm[start:end], dtype=dtype
                            ).copy()
                        else:
                            # Non-contiguous strips — still use mmap slices
                            # (avoids seek overhead, keeps OS cache hot)
                            parts = [
                                np.frombuffer(mm[off : off + bc], dtype=dtype)
                                for off, bc in zip(offsets, bytecounts)
                            ]
                            raw = np.concatenate(parts)

            return raw.reshape(shape)

    except Exception:
        return tifffile.imread(str(filepath))


# ─────────────────────────────────────── raw array → normalised shape ─────────

def _raw_to_rgb_float32(raw: np.ndarray) -> np.ndarray:
    """Convert a raw uint8/uint16 array to float32 RGB [0,1], shape (H,W,3)."""
    if raw.dtype == np.uint8:
        scale = 1.0 / 255.0
    elif raw.dtype == np.uint16:
        scale = 1.0 / 65535.0
    elif raw.dtype in (np.float32, np.float64):
        img = raw.astype(np.float32)
        scale = None
    else:
        try:
            scale = 1.0 / float(np.iinfo(raw.dtype).max)
        except ValueError:
            scale = 1.0 / (float(raw.max()) or 1.0)

    img = raw.astype(np.float32)
    if scale is not None:
        img *= scale

    # Ensure (H, W, 3)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


# ─────────────────────────────────────── public loading API ───────────────────

def load_image_float(filepath: str | Path) -> np.ndarray:
    """
    Full float32 RGB [0,1] image.  Used by Pass 2 (needs all channels).
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".tif", ".tiff"):
        raw = _load_raw_tiff_fast(filepath)
    else:
        pil_img = Image.open(str(filepath)).convert("RGB")
        raw = np.array(pil_img, dtype=np.uint8)

    return _raw_to_rgb_float32(raw)


def load_y_channel_float(filepath: str | Path) -> np.ndarray:
    """
    Y (luminance) channel only, float32 [0,1], shape (H, W).
    Used by Pass 1 — avoids creating the full float32 RGB intermediate.

    Memory comparison for a 6000×4000 16-bit RGB TIFF:
      Old path:  144 MB (uint16) + 288 MB (float32 RGB) + 96 MB (Y) = 528 MB
      This path: 144 MB (uint16) + 96 MB (Y) + 96 MB (one channel temp) = 336 MB
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".tif", ".tiff"):
        raw = _load_raw_tiff_fast(filepath)

        if raw.dtype == np.uint16:
            scale = 1.0 / 65535.0
        elif raw.dtype == np.uint8:
            scale = 1.0 / 255.0
        else:
            scale = 1.0 / (float(np.iinfo(raw.dtype).max)
                           if raw.dtype.kind == "u" else 1.0)

        if raw.ndim == 2:
            return raw.astype(np.float32) * scale

        # RGB: reshape to (H*W, 3) so memory is contiguous per pixel,
        # then use a single matmul for the Rec.709 dot product.
        # This avoids strided channel extraction (raw[:,:,0]) which reads
        # every 6th byte and wastes 2/3 of each cache line.
        h, w = raw.shape[:2]
        coeffs = np.array([_KR * scale, _KG * scale, _KB * scale], np.float32)
        flat   = raw.reshape(-1, 3).astype(np.float32)   # (H*W, 3), contiguous
        Y      = (flat @ coeffs).reshape(h, w)            # BLAS matmul
        return Y

    else:
        # JPEG: load full RGB then extract Y (JPEG is always small/fast)
        pil_img = Image.open(str(filepath)).convert("RGB")
        raw = np.array(pil_img, dtype=np.float32) / 255.0
        return (_KR * raw[:, :, 0] + _KG * raw[:, :, 1] + _KB * raw[:, :, 2])


def get_bit_depth(filepath: str | Path) -> int:
    filepath = Path(filepath)
    if filepath.suffix.lower() in (".tif", ".tiff"):
        raw = _load_raw_tiff_fast(Path(filepath))
        return 16 if raw.dtype == np.uint16 else 8
    return 8


# ─────────────────────────────────────────────────── saving ───────────────────

def save_image_float(
    img_float: np.ndarray,
    filepath: str | Path,
    output_format: str,
    jpeg_quality: int = 85,
    jpeg_resize_1080: bool = False,
) -> None:
    """
    Save a float32 [0,1] RGB image.
    output_format : "jpeg" | "tiff8" | "tiff16"
    """
    filepath  = Path(filepath)
    img_float = np.clip(img_float, 0.0, 1.0)

    if output_format == "jpeg":
        arr     = (img_float * 255.0 + 0.5).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="RGB")
        if jpeg_resize_1080:
            w, h    = pil_img.size
            shorter = min(w, h)
            if shorter > 1080:
                scale   = 1080 / shorter
                pil_img = pil_img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS,
                )
        pil_img.save(str(filepath), format="JPEG",
                     quality=jpeg_quality, subsampling=0)

    elif output_format == "tiff8":
        arr = (img_float * 255.0 + 0.5).astype(np.uint8)
        tifffile.imwrite(str(filepath), arr, compression=None, photometric="rgb")

    elif output_format == "tiff16":
        arr = (img_float * 65535.0 + 0.5).astype(np.uint16)
        tifffile.imwrite(str(filepath), arr, compression=None, photometric="rgb")

    else:
        raise ValueError(f"Unknown output format: {output_format!r}")


def output_extension(output_format: str) -> str:
    return {"jpeg": ".jpg", "tiff8": ".tif", "tiff16": ".tif"}[output_format]
