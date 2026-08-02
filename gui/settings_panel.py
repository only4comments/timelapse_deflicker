"""
Left-hand settings panel: analysis settings, correction mode, output format.
Performance / worker settings are handled by the auto-detect info card in
the source bar and stored internally — not shown here.
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

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        row = 0

        # ── Luminance Analysis ────────────────────────────────────────────────
        lum_frame = ttk.LabelFrame(self, text="Analysis", padding=8)
        lum_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=(4, 2))
        lum_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(lum_frame, text="Rolling window:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        win_frame = ttk.Frame(lum_frame)
        win_frame.grid(row=0, column=1, sticky="w")
        self._window_var = tk.IntVar(value=10)
        ttk.Spinbox(win_frame, from_=1, to=9999, width=6,
                    textvariable=self._window_var).pack(side="left")
        ttk.Label(win_frame, text=" frames", foreground="#888",
                  font=("Segoe UI", 8)).pack(side="left")

        ttk.Separator(lum_frame, orient="horizontal").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=4
        )

        ttk.Label(lum_frame, text="Metric:").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        self._metric_var = tk.StringVar(value="median")
        for i, (label, value) in enumerate([
            ("Mean", "mean"),
            ("Median", "median"),
            ("Percentile", "percentile"),
        ]):
            ttk.Radiobutton(
                lum_frame, text=label, variable=self._metric_var,
                value=value, command=self._on_metric_change
            ).grid(row=3+i, column=0, columnspan=2, sticky="w", padx=8)

        pct_frame = ttk.Frame(lum_frame)
        pct_frame.grid(row=6, column=0, columnspan=2, sticky="w", padx=24)
        ttk.Label(pct_frame, text="Percentile (1–100):").pack(side="left")
        self._percentile_var = tk.IntVar(value=95)
        self._pct_spin = ttk.Spinbox(pct_frame, from_=1, to=100, width=5,
                                      textvariable=self._percentile_var)
        self._pct_spin.pack(side="left", padx=4)
        self._on_metric_change()

        # ── Resize ────────────────────────────────────────────────────────────
        resize_frame = ttk.LabelFrame(self, text="Resize Output", padding=8)
        resize_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        resize_frame.columnconfigure(0, weight=1)
        row += 1

        self._jpeg_resize_var = tk.BooleanVar(value=False)
        resize_check = ttk.Checkbutton(
            resize_frame,
            text="Resize output",
            variable=self._jpeg_resize_var,
            command=self._on_resize_toggle,
        )
        resize_check.grid(row=0, column=0, sticky="w")

        # Resize sub-controls — enabled/disabled by checkbox
        self._resize_sub = ttk.Frame(resize_frame)
        self._resize_sub.grid(row=1, column=0, sticky="ew", padx=12, pady=(2, 4))

        self._resize_edge_var = tk.StringVar(value="shorter")
        ttk.Radiobutton(
            self._resize_sub, text="Shorter edge",
            variable=self._resize_edge_var, value="shorter",
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            self._resize_sub, text="Longer edge",
            variable=self._resize_edge_var, value="longer",
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        val_frame = ttk.Frame(self._resize_sub)
        val_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(val_frame, text="Size:").pack(side="left")
        self._resize_value_var = tk.StringVar(value="1080")
        self._resize_entry = ttk.Entry(
            val_frame, textvariable=self._resize_value_var, width=8
        )
        self._resize_entry.pack(side="left", padx=(4, 4))
        ttk.Label(
            val_frame, text="px  or  %",
            foreground="#555", font=("Segoe UI", 8),
        ).pack(side="left")

        self._resize_hint_var = tk.StringVar(value="")
        ttk.Label(
            self._resize_sub, textvariable=self._resize_hint_var,
            foreground="#555", font=("Segoe UI", 8),
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self._resize_value_var.trace_add("write", self._on_resize_value_change)
        self._on_resize_toggle()

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

        ttk.Radiobutton(
            out_frame, text="TIFF 8-bit",
            variable=self._format_var, value="tiff8",
            command=self._on_format_change
        ).grid(row=2, column=0, sticky="w")

        self._depth_warn_label = ttk.Label(
            out_frame,
            text="⚠  16→8 bit is lossy",
            foreground="#c8a000", font=("Segoe UI", 8),
        )

        ttk.Radiobutton(
            out_frame, text="TIFF 16-bit",
            variable=self._format_var, value="tiff16",
            command=self._on_format_change
        ).grid(row=4, column=0, sticky="w")

        self._on_format_change()

        # ── Hardware (hidden spinboxes, managed by auto-detect) ───────────────
        # These are not shown in the UI but hold the worker counts so that
        # get_settings() / apply_settings() can persist them via config.
        self._workers_p1_var = tk.IntVar(value=max(1, (os.cpu_count() or 4) // 2))
        self._workers_p2_var = tk.IntVar(value=max(1, (os.cpu_count() or 4) // 4))

    # ──────────────────────────────────────────────────── events ─────────────

    def _on_metric_change(self) -> None:
        state = "normal" if self._metric_var.get() == "percentile" else "disabled"
        self._pct_spin.configure(state=state)

    def _on_format_change(self) -> None:
        fmt = self._format_var.get()
        q_state = "normal" if fmt == "jpeg" else "disabled"
        self._quality_scale.configure(state=q_state)
        # Resize is available for all formats, not just JPEG
        if fmt == "tiff8":
            self._depth_warn_label.grid(row=3, column=0, sticky="w", padx=8)
        else:
            self._depth_warn_label.grid_remove()

    def _on_resize_toggle(self) -> None:
        """Enable/disable resize sub-controls based on checkbox state."""
        enabled = self._jpeg_resize_var.get()
        state = "normal" if enabled else "disabled"
        for child in self._resize_sub.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
            for grandchild in child.winfo_children():
                try:
                    grandchild.configure(state=state)
                except tk.TclError:
                    pass
        if enabled:
            self._on_resize_value_change()
        else:
            self._resize_hint_var.set("")

    def _on_resize_value_change(self, *_) -> None:
        """Validate the resize value and update the hint label."""
        if not self._jpeg_resize_var.get():
            return
        raw = self._resize_value_var.get().strip()
        parsed = _parse_resize_value(raw)
        if parsed is None:
            self._resize_hint_var.set("⚠  Enter a number (pixels) or value% (e.g. 1080 or 50%)")
        elif isinstance(parsed, float):
            self._resize_hint_var.set(f"→ {parsed:.0f}% of original")
        else:
            self._resize_hint_var.set(f"→ {parsed} px on the selected edge")

    def _on_quality_change(self, _=None) -> None:
        self._quality_label.configure(text=str(int(self._jpeg_quality_var.get())))

    # ──────────────────────────────────────────────────── public ─────────────

    def apply_hw_profile(self, profile) -> None:
        """Called by main_window after folder load — stores worker counts silently."""
        self._workers_p1_var.set(profile.pass1_workers)
        self._workers_p2_var.set(profile.pass2_workers)

    def get_settings(self) -> dict[str, Any]:
        return {
            "window_size":       max(1, self._window_var.get()),
            "metric":            self._metric_var.get(),
            "percentile_value":  float(self._percentile_var.get()),
            "correction_mode":   "scaling",
            "output_format":     self._format_var.get(),
            "jpeg_quality":      int(self._jpeg_quality_var.get()),
            "resize_enabled":    self._jpeg_resize_var.get(),
            "resize_edge":       self._resize_edge_var.get(),   # "shorter" | "longer"
            "resize_value":      self._resize_value_var.get(),  # e.g. "1080" or "50%"
            # Legacy key kept for config compatibility
            "jpeg_resize_1080":  False,
            "worker_count":      max(1, self._workers_p1_var.get()),
            "worker_count_p2":   max(1, self._workers_p2_var.get()),
        }

    def apply_settings(self, s: dict[str, Any]) -> None:
        self._window_var.set(s.get("window_size", 10))
        self._metric_var.set(s.get("metric", "median"))
        self._percentile_var.set(s.get("percentile_value", 95))
        self._format_var.set(s.get("output_format", "jpeg"))
        self._jpeg_quality_var.set(s.get("jpeg_quality", 85))

        # Migrate legacy jpeg_resize_1080 → new resize fields
        if s.get("resize_enabled") is not None:
            self._jpeg_resize_var.set(s["resize_enabled"])
        elif s.get("jpeg_resize_1080"):
            self._jpeg_resize_var.set(True)
        else:
            self._jpeg_resize_var.set(False)

        self._resize_edge_var.set(s.get("resize_edge", "shorter"))
        self._resize_value_var.set(s.get("resize_value", "1080"))

        pc = max(1, (os.cpu_count() or 4) // 2)
        self._workers_p1_var.set(s.get("worker_count",    pc))
        self._workers_p2_var.set(s.get("worker_count_p2", max(1, pc // 2)))

        self._on_metric_change()
        self._on_format_change()
        self._on_quality_change()
        self._on_resize_toggle()


def _parse_resize_value(raw: str) -> "int | float | None":
    """
    Parse a resize value string.
    Returns:
      int   — pixel count  (e.g. "1080")
      float — percentage   (e.g. 50.0 from "50%")
      None  — invalid input
    """
    raw = raw.strip()
    if raw.endswith("%"):
        try:
            pct = float(raw[:-1])
            return pct if 1.0 <= pct <= 400.0 else None
        except ValueError:
            return None
    else:
        try:
            px = int(raw)
            return px if px >= 1 else None
        except ValueError:
            return None
