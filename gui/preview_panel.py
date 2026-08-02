"""
Side-by-side Before / After preview panel.

Downscaled thumbnails are cached as JPEG in the temp directory.
Preview is on-demand (requires Pass 1 to be complete).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


_THUMBNAIL_QUALITY = 90   # JPEG quality used for the disk-cached preview thumbnail
_PREVIEW_MAX_H = 680      # max height of each preview pane in pixels


class PreviewPanel(ttk.LabelFrame):
    """
    Public interface used by MainWindow:
        set_file_list(files)
        set_pass1_results(luminance_values, rolling_lum, rolling_hists)
        get_current_frame_index()  -> int
        request_preview(settings)  -> triggers async refresh
    """

    def __init__(self, parent, temp_manager, **kw):
        super().__init__(parent, text="Preview  (requires Pass 1 complete)", **kw)
        self._temp = temp_manager
        self._file_list: list[Path] = []
        self._luminance_values: list[float | None] = []
        self._rolling_lum: list[float] = []
        self._rolling_hists: list | None = None
        self._pass1_done = False
        self._preview_thread: threading.Thread | None = None
        self._tk_before: ImageTk.PhotoImage | None = None
        self._tk_after:  ImageTk.PhotoImage | None = None
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── controls row ──────────────────────────────────────────────────────
        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=0, sticky="ew", padx=6, pady=4)

        ttk.Label(ctrl, text="Frame:").pack(side="left")
        self._frame_var = tk.IntVar(value=0)
        self._frame_spin = ttk.Spinbox(
            ctrl, from_=0, to=0, width=7,
            textvariable=self._frame_var, state="disabled"
        )
        self._frame_spin.pack(side="left", padx=4)

        self._slider_var = tk.IntVar(value=0)
        self._slider = ttk.Scale(
            ctrl, from_=0, to=1, orient="horizontal", length=300,
            variable=self._slider_var, command=self._on_slider,
            state="disabled"
        )
        self._slider.pack(side="left", padx=4)

        self._status_var = tk.StringVar(value="Run Pass 1 first to enable preview.")
        ttk.Label(ctrl, textvariable=self._status_var, foreground="#888").pack(
            side="left", padx=12
        )

        self._btn_preview = ttk.Button(
            ctrl, text="⟳  Refresh Preview",
            command=self._trigger_preview, state="disabled"
        )
        self._btn_preview.pack(side="right", padx=4)

        # ── image panes ───────────────────────────────────────────────────────
        panes = ttk.Frame(self)
        panes.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)
        panes.rowconfigure(0, weight=1)

        # Before
        before_wrap = ttk.LabelFrame(panes, text="Before (source)")
        before_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        before_wrap.rowconfigure(0, weight=1)
        before_wrap.columnconfigure(0, weight=1)
        self._canvas_before = tk.Canvas(
            before_wrap, bg="#111", highlightthickness=0
        )
        self._canvas_before.grid(row=0, column=0, sticky="nsew")

        # After
        after_wrap = ttk.LabelFrame(panes, text="After (corrected)")
        after_wrap.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        after_wrap.rowconfigure(0, weight=1)
        after_wrap.columnconfigure(0, weight=1)
        self._canvas_after = tk.Canvas(
            after_wrap, bg="#111", highlightthickness=0
        )
        self._canvas_after.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------ public

    def set_file_list(self, files: list[Path]) -> None:
        self._file_list = files
        n = max(0, len(files) - 1)
        self._frame_spin.configure(to=n, state="normal" if n >= 0 else "disabled")
        self._slider.configure(to=max(1, n), state="normal" if n >= 0 else "disabled")
        self._frame_var.set(0)
        self._slider_var.set(0)
        self._pass1_done = False
        self._status_var.set("Run Pass 1 first to enable preview.")
        self._btn_preview.configure(state="disabled")
        self._clear_canvases()

    def set_pass1_results(
        self,
        luminance_values: list,
        rolling_lum,
        rolling_hists,
    ) -> None:
        self._luminance_values = luminance_values
        self._rolling_lum = rolling_lum
        self._rolling_hists = rolling_hists
        self._pass1_done = True
        self._btn_preview.configure(state="normal")
        self._status_var.set("Pass 1 done — select a frame and click Refresh.")

    def get_current_frame_index(self) -> int:
        return int(self._frame_var.get())

    def request_preview(self, settings: dict) -> None:
        """Public call from MainWindow when settings change and pass1 is done."""
        if self._pass1_done:
            self._trigger_preview(settings)

    # ------------------------------------------------------------------ events

    def _on_slider(self, value) -> None:
        self._frame_var.set(int(float(value)))

    def _trigger_preview(self, settings: dict | None = None) -> None:
        if not self._file_list or not self._pass1_done:
            return
        if self._preview_thread and self._preview_thread.is_alive():
            return  # already running

        idx = int(self._frame_var.get())
        self._status_var.set(f"Generating preview for frame {idx}…")
        self._btn_preview.configure(state="disabled")

        t = threading.Thread(
            target=self._preview_worker,
            args=(idx, settings),
            daemon=True,
        )
        self._preview_thread = t
        t.start()

    def _preview_worker(self, idx: int, settings: dict | None) -> None:
        try:
            filepath = self._file_list[idx]

            # ── downscaled source (cached) ────────────────────────────────────
            cache_path = self._temp.preview_path(filepath.stem)
            if cache_path.exists():
                before_pil = Image.open(str(cache_path)).convert("RGB")
            else:
                src_pil = Image.open(str(filepath)).convert("RGB")
                src_pil.thumbnail(
                    (4000, _PREVIEW_MAX_H), Image.Resampling.LANCZOS
                )
                src_pil.save(str(cache_path), format="JPEG", quality=_THUMBNAIL_QUALITY)
                before_pil = src_pil

            # ── corrected version (always recomputed, fast on small image) ───
            import numpy as np
            from core.luminance import apply_luminance_scaling, apply_histogram_matching

            arr = np.array(before_pil, dtype=np.float32) / 255.0

            # Compute correction from rolling avg and measured luminance
            lum = self._luminance_values[idx] if self._luminance_values else None
            r_lum = self._rolling_lum[idx] if self._rolling_lum is not None else None
            correction_mode = (settings or {}).get("correction_mode", "scaling")

            if lum and r_lum and lum > 1e-9:
                if correction_mode == "scaling":
                    factor = r_lum / lum
                    corrected = apply_luminance_scaling(arr, factor)
                else:
                    ref_hist = (
                        self._rolling_hists[idx]
                        if self._rolling_hists else None
                    )
                    if ref_hist is not None:
                        corrected = apply_histogram_matching(arr, ref_hist)
                    else:
                        corrected = apply_luminance_scaling(arr, r_lum / lum)
            else:
                corrected = arr

            after_arr = (np.clip(corrected, 0.0, 1.0) * 255.0).astype(np.uint8)
            after_pil = Image.fromarray(after_arr, "RGB")

            # Schedule UI update on main thread
            self.after(0, self._show_images, before_pil, after_pil, idx)

        except Exception as exc:
            self.after(0, self._preview_error, str(exc))

    def _show_images(
        self, before_pil: Image.Image, after_pil: Image.Image, idx: int
    ) -> None:
        self._display_on_canvas(self._canvas_before, before_pil)
        self._tk_before = self._canvas_before.image  # keep reference

        self._display_on_canvas(self._canvas_after, after_pil)
        self._tk_after = self._canvas_after.image

        self._status_var.set(
            f"Frame {idx} of {len(self._file_list) - 1}  |  "
            f"{before_pil.size[0]}×{before_pil.size[1]} px (preview)"
        )
        self._btn_preview.configure(state="normal")

    def _preview_error(self, msg: str) -> None:
        self._status_var.set(f"Preview error: {msg}")
        self._btn_preview.configure(state="normal")

    # ------------------------------------------------------------------ helpers

    def _display_on_canvas(self, canvas: tk.Canvas, pil_img: Image.Image) -> None:
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 600, _PREVIEW_MAX_H

        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih, 1.0)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        canvas.image = photo   # prevent GC

    def _clear_canvases(self) -> None:
        for canvas in (self._canvas_before, self._canvas_after):
            canvas.delete("all")
            canvas.image = None
