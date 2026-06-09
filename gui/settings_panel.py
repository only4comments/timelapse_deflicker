"""
Left-hand settings panel: luminance analysis, correction, output, workers.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Any


class SettingsPanel(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        row = 0

        # ── Luminance Analysis ────────────────────────────────────────────────
        lum_frame = ttk.LabelFrame(self, text="Luminance Analysis", padding=8)
        lum_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=(4, 2))
        lum_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(lum_frame, text="Rolling window:").grid(
            row=0, column=0, sticky="w", pady=2
        )

        win_frame = ttk.Frame(lum_frame)
        win_frame.grid(row=0, column=1, sticky="w")
        self._window_var = tk.IntVar(value=10)
        win_spin = ttk.Spinbox(
            win_frame, from_=1, to=9999, width=6,
            textvariable=self._window_var
        )
        win_spin.pack(side="left")

        ttk.Separator(lum_frame, orient="horizontal").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=4
        )

        ttk.Label(lum_frame, text="Luminance metric:").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )

        self._metric_var = tk.StringVar(value="median")
        for i, (label, value) in enumerate([
            ("Mean", "mean"),
            ("Median", "median"),
            ("Percentile", "percentile"),
        ]):
            rb = ttk.Radiobutton(
                lum_frame, text=label, variable=self._metric_var,
                value=value, command=self._on_metric_change
            )
            rb.grid(row=3 + i, column=0, columnspan=2, sticky="w", padx=8)

        # Percentile sub-row
        pct_frame = ttk.Frame(lum_frame)
        pct_frame.grid(row=6, column=0, columnspan=2, sticky="w", padx=24)
        ttk.Label(pct_frame, text="Percentile (1–100):").pack(side="left")
        self._percentile_var = tk.IntVar(value=95)
        self._pct_spin = ttk.Spinbox(
            pct_frame, from_=1, to=100, width=5,
            textvariable=self._percentile_var
        )
        self._pct_spin.pack(side="left", padx=4)
        self._on_metric_change()   # set initial state

        # ── Correction Mode ───────────────────────────────────────────────────
        corr_frame = ttk.LabelFrame(self, text="Correction Mode", padding=8)
        corr_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        corr_frame.columnconfigure(0, weight=1)
        row += 1

        self._correction_var = tk.StringVar(value="scaling")
        ttk.Radiobutton(
            corr_frame, text="Luminance scaling  (fast)",
            variable=self._correction_var, value="scaling"
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            corr_frame, text="Full histogram matching",
            variable=self._correction_var, value="histogram"
        ).grid(row=1, column=0, sticky="w")

        # ── Output Format ─────────────────────────────────────────────────────
        out_frame = ttk.LabelFrame(self, text="Output Format", padding=8)
        out_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        out_frame.columnconfigure(0, weight=1)
        row += 1

        self._format_var = tk.StringVar(value="jpeg")
        ttk.Radiobutton(
            out_frame, text="JPEG",
            variable=self._format_var, value="jpeg",
            command=self._on_format_change
        ).grid(row=0, column=0, sticky="w")

        # JPEG quality sub-row
        q_frame = ttk.Frame(out_frame)
        q_frame.grid(row=1, column=0, sticky="ew", padx=16)
        ttk.Label(q_frame, text="Quality:").pack(side="left")
        self._jpeg_quality_var = tk.IntVar(value=85)
        self._quality_scale = ttk.Scale(
            q_frame, from_=1, to=100, orient="horizontal", length=100,
            variable=self._jpeg_quality_var, command=self._on_quality_change
        )
        self._quality_scale.pack(side="left", padx=4)
        self._quality_label = ttk.Label(q_frame, text="85", width=3)
        self._quality_label.pack(side="left")

        # JPEG resize sub-row  (row=2 — exclusively)
        self._jpeg_resize_var = tk.BooleanVar(value=False)
        self._jpeg_resize_check = ttk.Checkbutton(
            out_frame,
            text="Resize for quick review  (shorter edge = 1080 px)",
            variable=self._jpeg_resize_var,
        )
        self._jpeg_resize_check.grid(row=2, column=0, sticky="w", padx=16, pady=(2, 4))

        # TIFF options start at row=3 (below the checkbox)
        ttk.Radiobutton(
            out_frame, text="TIFF 8-bit uncompressed",
            variable=self._format_var, value="tiff8",
            command=self._on_format_change
        ).grid(row=3, column=0, sticky="w")

        self._depth_warn_label = ttk.Label(
            out_frame,
            text="⚠ 16→8 bit is a lossy downgrade",
            foreground="#c8a000",
            font=("Segoe UI", 8),
        )
        # shown/hidden by _on_format_change at row=4

        ttk.Radiobutton(
            out_frame, text="TIFF 16-bit uncompressed",
            variable=self._format_var, value="tiff16",
            command=self._on_format_change
        ).grid(row=5, column=0, sticky="w")

        self._on_format_change()

        # ── Performance ───────────────────────────────────────────────────────
        perf_frame = ttk.LabelFrame(self, text="Performance", padding=8)
        perf_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=(2, 4))
        perf_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(perf_frame, text="Worker processes:").grid(
            row=0, column=0, sticky="w"
        )
        default_workers = os.cpu_count() or 4
        self._workers_var = tk.IntVar(value=default_workers)
        ttk.Spinbox(
            perf_frame, from_=1, to=256, width=5,
            textvariable=self._workers_var
        ).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(
            perf_frame,
            text=f"(detected {default_workers} logical cores)",
            foreground="#888",
            font=("Segoe UI", 8),
        ).grid(row=1, column=0, columnspan=2, sticky="w")

    # ------------------------------------------------------------------ events

    def _on_metric_change(self) -> None:
        state = "normal" if self._metric_var.get() == "percentile" else "disabled"
        self._pct_spin.configure(state=state)

    def _on_format_change(self) -> None:
        fmt = self._format_var.get()
        q_state = "normal" if fmt == "jpeg" else "disabled"
        self._quality_scale.configure(state=q_state)
        self._jpeg_resize_check.configure(state=q_state)

        # Show bit-depth warning only for tiff8
        if fmt == "tiff8":
            self._depth_warn_label.grid(row=4, column=0, sticky="w", padx=8)
        else:
            self._depth_warn_label.grid_remove()

    def _on_quality_change(self, _=None) -> None:
        self._quality_label.configure(text=str(int(self._jpeg_quality_var.get())))

    # ------------------------------------------------------------------ public

    def get_settings(self) -> dict[str, Any]:
        return {
            "window_size":       max(1, self._window_var.get()),
            "metric":            self._metric_var.get(),
            "percentile_value":  float(self._percentile_var.get()),
            "correction_mode":   self._correction_var.get(),
            "output_format":     self._format_var.get(),
            "jpeg_quality":      int(self._jpeg_quality_var.get()),
            "jpeg_resize_1080":  self._jpeg_resize_var.get(),
            "worker_count":      max(1, self._workers_var.get()),
        }

    def apply_settings(self, s: dict[str, Any]) -> None:
        self._window_var.set(s.get("window_size", 10))
        self._metric_var.set(s.get("metric", "median"))
        self._percentile_var.set(s.get("percentile_value", 95))
        self._correction_var.set(s.get("correction_mode", "scaling"))
        self._format_var.set(s.get("output_format", "jpeg"))
        self._jpeg_quality_var.set(s.get("jpeg_quality", 85))
        self._jpeg_resize_var.set(s.get("jpeg_resize_1080", False))
        self._workers_var.set(s.get("worker_count", os.cpu_count() or 4))
        self._on_metric_change()
        self._on_format_change()
        self._on_quality_change()
