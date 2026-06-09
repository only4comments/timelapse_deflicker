# Timelapse Deflickering Tool — Full Specification

## Overview

A standalone Windows desktop application written in Python 3.11 that removes luminance flicker from timelapse sequences made of individual image files. The tool uses a two-pass workflow: first analysing all frames to build a luminance profile, then applying corrections and writing the output. Processing is parallelised across all available CPU cores. Files are never modified in place.

---

## Supported Input Formats

| Format | Bit Depth | Compression |
|---|---|---|
| TIFF | 16-bit | Uncompressed |
| TIFF | 8-bit | Uncompressed |
| TIFF | 8-bit | LZW |
| JPEG | 8-bit | Standard |

Files are discovered in the source folder alphabetically by filename. Subdirectories are ignored.

**Mixed-format folders:** If the source folder contains more than one file extension (e.g. both `.tif` and `.jpg`), the tool warns the user and asks them to confirm which extension to process, or to abort.

---

## Output Formats

Configured via radio button. The output folder is always a separate directory chosen by the user; source files are never overwritten.

| Option | Details |
|---|---|
| JPEG | Configurable quality slider (1–100, default 85). Chroma subsampling fixed at 4:4:4. |
| TIFF 8-bit uncompressed | Output always 8-bit regardless of input depth. 16→8 conversion is done by proper linear scaling (not byte truncation). |
| TIFF 16-bit uncompressed | Full-depth output. If input is 8-bit, values are scaled up to 16-bit. |

**Bit-depth mismatch warning:** When the user selects TIFF 8-bit output with a 16-bit source, a visible notice is shown in the UI explaining that this is a lossy downgrade.

---

## Algorithm — Two-Pass Workflow

### Pass 1 — Luminance Analysis

For every frame, a single luminance value is extracted from the Y channel (Rec.709 / YCbCr). All colour processing is luminance-only; no per-channel RGB correction is applied to avoid colour casts.

**Luminance metric** (radio button, applies to all frames):

- Mean
- Median
- Percentile — with a numeric input field for the percentile value (default: 95)

The result of Pass 1 is an ordered array of per-frame luminance values.

### Rolling Average

A centred rolling average of the per-frame luminance values is computed using a user-configurable window size (see Controls). This produces a smooth target luminance curve representing the intended gradual brightness change of the scene.

**Edge handling:** At the start and end of the sequence where a full window is not available, the window is truncated to the available frames (no padding, mirroring, or clamping).

### Pass 2 — Correction & Output

For each frame, a multiplicative correction factor is computed:

```
correction_factor = rolling_average_luminance[i] / measured_luminance[i]
```

The correction is applied by multiplying all pixel values by this scalar. Clipping is applied after multiplication to keep values within the valid range for the output bit depth.

**Correction mode** (radio button):

- **Luminance scaling** — apply the multiplicative factor globally (fast, clean, recommended for most cases).
- **Full histogram matching** — reshape the entire tonal distribution of each frame to match its rolling-average reference histogram (more powerful, slower, may introduce artefacts in extreme cases).

---

## GUI Layout

The application has a single main window divided into the following regions:

### Top — Source & Destination

- **Open source folder** button + display of selected path
- **Select destination folder** button + display of selected path
- If the destination folder already contains files when the user starts processing, a dialog asks: Overwrite existing files / Skip existing files / Abort.

### Left Panel — Settings

**File ordering**
- Alphabetical (fixed, not user-configurable)

**Rolling average window size**
- Numeric input field, writable from keyboard
- Adjustable with up/down arrow buttons
- Default: 10
- Minimum: 1

**Luminance metric** (radio group)
- Mean
- Median
- Percentile → inline numeric field (integer 1–100, default 95)

**Correction mode** (radio group)
- Luminance scaling
- Full histogram matching

**Output format** (radio group)
- JPEG → quality slider (1–100, default 85)
- TIFF 8-bit uncompressed
- TIFF 16-bit uncompressed

**Worker threads**
- Numeric input, default = all logical CPU cores
- Minimum: 1

### Centre — Before / After Preview

A side-by-side preview panel optimised for 21:9 displays.

- **Frame selector** — numeric input or slider to jump to any frame by index
- **Before** pane — source frame, scaled to fit the panel (preserving aspect ratio; no upscaling beyond panel size)
- **After** pane — the same frame with correction applied in real time (single-frame preview, does not require a full run)
- Scaling is always fit-to-pane; the original file is never modified for preview purposes
- Preview is regenerated on demand (button or frame change), not live/auto

### Bottom — Progress & Controls

- **Run** button — starts the two-pass processing job
- **Cancel** button — gracefully stops processing after the current batch of workers finishes; partial output files from cancelled frames are deleted
- **Overall progress bar** — shows frames completed / total frames
- **Phase label** — shows current phase: "Pass 1 – Analysing luminance…" / "Pass 2 – Applying corrections…"
- **ETA display** — estimated time remaining, updated every second
- **Frames/sec counter**

### Bottom — Log Console

A scrolling, read-only text area below the progress bar showing:

- Start time and configuration summary
- Any skipped or errored files with reason
- Average correction factor applied per frame (Pass 2)
- Completion summary: total frames, elapsed time, output folder path
- Log is also written to a `.log` file in the output folder automatically

---

## Parallelisation

- Pass 1 (luminance extraction) is fully parallelised across all frames using `concurrent.futures.ProcessPoolExecutor`.
- Pass 2 (correction + write) is similarly parallelised.
- The rolling average computation (between passes) is a fast sequential step on the luminance array and is not parallelised.
- Worker count is user-configurable; default is all logical cores (e.g. 32).
- Memory note: all frames may be loaded into worker processes simultaneously. At 10 workers × ~50–100 MB per 24 Mpix 16-bit TIFF, peak RAM usage can reach several gigabytes. This is acceptable given the target machine's ~40 GB available RAM.

---

## Settings Persistence

On exit, the application saves the following to a `config.json` file in the application directory. On next launch, these values are restored:

- Last source folder path
- Last destination folder path
- Rolling average window size
- Luminance metric and percentile value
- Correction mode
- Output format and JPEG quality
- Worker thread count

---

## File Naming

Output files use the same filename as the source file, with the extension changed to match the output format (e.g. `DSC_0001.tif` → `DSC_0001.jpg` for JPEG output). No suffixes are added.

---

## Error Handling

| Condition | Behaviour |
|---|---|
| Unreadable source file | Logged, frame skipped, processing continues |
| Output write failure | Logged, frame skipped, processing continues |
| Source folder empty or no matching files | Error dialog, run not started |
| Destination folder not set | Run button disabled |
| Source folder not set | Run button disabled |
| Mixed file extensions detected | Warning dialog before run |
| Correction factor is zero or NaN (e.g. black frame) | Frame copied unmodified; warning logged |

---

## Dependencies (Python packages)

| Package | Purpose |
|---|---|
| `Pillow` | JPEG and 8-bit TIFF I/O |
| `tifffile` | 16-bit TIFF I/O |
| `numpy` | Pixel arithmetic, luminance computation, histogram matching |
| `scikit-image` | Histogram matching (`skimage.exposure.match_histograms`) |
| `tkinter` | GUI (stdlib) |
| `concurrent.futures` | Parallelisation (stdlib) |

---

## Out of Scope (explicit non-goals)

- Partial-frame / banding flicker correction
- Per-channel RGB correction
- Video file input or output
- Network / cloud storage paths
- Batch queue (multiple source folders in one session)
