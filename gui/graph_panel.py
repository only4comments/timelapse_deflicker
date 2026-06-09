"""
Pass 1 luminance analysis graph — embedded matplotlib figure.

Three modes
-----------
idle      – placeholder text, no figure
live      – figure exists; measured line fills in as Pass 1 progresses;
            rolling average appears after finish_live()
complete  – all three series visible; Recalculate button enabled

Live-update strategy
--------------------
Results arrive out of order (parallel workers).  We pre-allocate a NaN
array of length == total frames and fill slots as each result arrives.
A 250 ms timer redraws via set_ydata() — no full re-plot on every frame.
The x-axis is fixed to the full frame count from the start.

Recalculate
-----------
After Pass 1, the user may change the rolling-window size in Settings and
click "↻ Recalculate" to recompute only the rolling average and correction
factor from the already-stored luminance values — no Pass 1 re-run.

Hover tooltip
-------------
Mouse-over shows: Frame N | Measured Y | Rolling avg | Correction factor.
Tooltip follows the cursor and flips side when near the right edge.
"""
from __future__ import annotations

from typing import Callable, Sequence

import tkinter as tk
from tkinter import ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── palette ───────────────────────────────────────────────────────────────────
_C_MEASURED = "#5b9bd5"
_C_ROLLING  = "#ed7d31"
_C_FACTOR   = "#70ad47"
_BG         = "#1a1a1a"
_AX_BG      = "#1e1e1e"
_GRID       = "#333333"
_TEXT       = "#cccccc"
_SPINE      = "#444444"

_REDRAW_MS  = 250


class GraphPanel(ttk.LabelFrame):
    """
    Constructor args
    ----------------
    on_recalculate : callable()  — called when the Recalculate button is clicked.
                                   main_window supplies the logic.
    """

    def __init__(self, parent, on_recalculate: Callable, **kw):
        super().__init__(parent, text="Pass 1 — Luminance Analysis", **kw)
        self._on_recalculate = on_recalculate

        # Figure objects
        self._fig    = None
        self._canvas: FigureCanvasTkAgg | None = None
        self._ax1    = None
        self._ax2    = None

        # Data arrays (set at finish_live / refresh_rolling)
        self._meas_arr:   np.ndarray | None = None
        self._roll_arr:   np.ndarray | None = None
        self._factor_arr: np.ndarray | None = None

        # File list for tooltip filenames
        self._file_list: list = []

        # Line artists
        self._line_meas = None
        self._line_roll = None
        self._line_fact = None

        # Live-mode state
        self._live_mode  = False
        self._live_data: np.ndarray | None = None
        self._dirty      = False
        self._n_frames   = 0

        # Stored settings for Recalculate label
        self._window_size = 0
        self._metric      = ""

        # Hover annotation
        self._annot = None

        self._build()

    # ───────────────────────────────────────────────────── build ──────────────

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)   # row 0 = toolbar, row 1 = canvas/placeholder

        # ── toolbar ──────────────────────────────────────────────────────────
        tb = ttk.Frame(self, padding=(6, 4))
        tb.grid(row=0, column=0, sticky="ew")

        self._info_var = tk.StringVar(value="")
        ttk.Label(tb, textvariable=self._info_var,
                  foreground="#888", font=("Segoe UI", 8)).pack(side="left")

        self._btn_recalc = ttk.Button(
            tb,
            text="↻  Recalculate rolling average",
            command=self._on_recalculate,
            state="disabled",
        )
        self._btn_recalc.pack(side="right")

        # ── placeholder ───────────────────────────────────────────────────────
        self._placeholder = ttk.Label(
            self,
            text="Run Pass 1 to see the luminance graph.",
            foreground="#666",
            anchor="center",
        )
        self._placeholder.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)

    # ───────────────────────────────────────────────── public API ─────────────

    def start_live(self, n_frames: int, window_size: int, metric: str,
                   file_list: list | None = None) -> None:
        """Called when Pass 1 begins.  Creates figure; starts periodic redraw."""
        self._teardown_canvas()
        self._placeholder.grid_remove()
        self._btn_recalc.configure(state="disabled")

        self._n_frames    = n_frames
        self._window_size = window_size
        self._metric      = metric
        self._file_list   = file_list or []
        self._live_data   = np.full(n_frames, np.nan, dtype=np.float64)
        self._meas_arr    = self._live_data   # shared reference
        self._roll_arr    = None
        self._factor_arr  = None
        self._dirty       = False
        self._live_mode   = True

        x = np.arange(n_frames)

        fig = Figure(figsize=(10, 4), dpi=100, facecolor=_BG)
        self._fig = fig
        ax1 = fig.add_subplot(111)
        ax1.set_facecolor(_AX_BG)
        self._ax1 = ax1

        (self._line_meas,) = ax1.plot(
            x, self._live_data,
            color=_C_MEASURED, linewidth=0.8, alpha=0.85,
            label="Measured luminance",
        )

        ax1.set_xlabel("Frame index", color=_TEXT, fontsize=9)
        ax1.set_ylabel("Luminance  (Y, Rec.709)", color=_TEXT, fontsize=9)
        ax1.set_xlim(0, max(1, n_frames - 1))
        ax1.set_ylim(0.0, 1.05)
        ax1.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor(_SPINE)
        ax1.grid(True, color=_GRID, linewidth=0.5, linestyle="--")

        self._progress_text = ax1.text(
            0.01, 0.97, f"Analysing…  0 / {n_frames}",
            transform=ax1.transAxes,
            color=_TEXT, fontsize=8, va="top",
        )

        ax2 = ax1.twinx()
        ax2.set_facecolor(_AX_BG)
        ax2.set_ylabel("Correction factor  (×)", color=_TEXT, fontsize=9)
        ax2.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor(_SPINE)
        self._ax2 = ax2

        ax1.legend(loc="upper right", fontsize=8,
                   facecolor="#2a2a2a", edgecolor=_SPINE, labelcolor=_TEXT)

        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._canvas = canvas

        self._setup_hover()
        self.after(_REDRAW_MS, self._live_redraw)

        self._info_var.set(
            f"window: {window_size}  |  metric: {metric}  |  {n_frames} frames"
        )

    def add_point(self, index: int, value: float) -> None:
        """Feed one Pass 1 result into the live graph. Main-thread only."""
        if self._live_data is None or index >= len(self._live_data):
            return
        self._live_data[index] = value
        self._dirty = True

    def finish_live(
        self,
        measured: Sequence[float | None],
        rolling: Sequence[float],
    ) -> None:
        """
        Called when Pass 1 is complete.
        Adds rolling average + correction factor; enables Recalculate button.
        """
        self._live_mode = False

        if self._fig is None:
            # Tab was never shown — fall back to full static render
            self.update(measured, rolling, self._window_size, self._metric)
            return

        n = len(measured)
        x = np.arange(n)
        meas_arr   = np.array(
            [v if v is not None else np.nan for v in measured], dtype=np.float64
        )
        roll_arr   = np.asarray(rolling, dtype=np.float64)
        factor_arr = self._compute_factors(meas_arr, roll_arr)

        self._meas_arr   = meas_arr
        self._roll_arr   = roll_arr
        self._factor_arr = factor_arr

        # Finalise measured line
        self._line_meas.set_ydata(meas_arr)

        # Add rolling average
        if self._line_roll:
            self._line_roll.remove()
        (self._line_roll,) = self._ax1.plot(
            x, roll_arr, color=_C_ROLLING, linewidth=2.0,
            label=f"Rolling average  (window {self._window_size})",
        )

        # Add correction factor
        if self._line_fact:
            self._line_fact.remove()
        (self._line_fact,) = self._ax2.plot(
            x, factor_arr,
            color=_C_FACTOR, linewidth=1.0, linestyle="--", alpha=0.85,
            label="Correction factor  (right axis)",
        )
        self._ax2.axhline(
            1.0, color=_C_FACTOR, linewidth=0.5, linestyle=":", alpha=0.5
        )

        self._tighten_axes(meas_arr, factor_arr)
        self._rebuild_legend()

        n_ok = int(np.sum(~np.isnan(meas_arr)))
        self._progress_text.set_text(
            f"Complete  —  {n_ok} / {n} frames analysed"
        )

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()
        self._btn_recalc.configure(state="normal")

    def refresh_rolling(
        self,
        rolling: Sequence[float],
        window_size: int,
    ) -> None:
        """
        Recompute only the rolling average and factor lines using new window.
        The measured luminance line is unchanged.  No Pass 1 re-run needed.
        """
        if self._meas_arr is None or self._fig is None:
            return

        self._window_size = window_size
        roll_arr   = np.asarray(rolling, dtype=np.float64)
        factor_arr = self._compute_factors(self._meas_arr, roll_arr)
        self._roll_arr   = roll_arr
        self._factor_arr = factor_arr

        n = len(roll_arr)
        x = np.arange(n)

        if self._line_roll:
            self._line_roll.remove()
        (self._line_roll,) = self._ax1.plot(
            x, roll_arr, color=_C_ROLLING, linewidth=2.0,
            label=f"Rolling average  (window {window_size})",
        )

        if self._line_fact:
            self._line_fact.remove()
        (self._line_fact,) = self._ax2.plot(
            x, factor_arr,
            color=_C_FACTOR, linewidth=1.0, linestyle="--", alpha=0.85,
            label="Correction factor  (right axis)",
        )

        self._tighten_axes(self._meas_arr, factor_arr)
        self._rebuild_legend()
        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()

        self._info_var.set(
            f"window: {window_size}  |  metric: {self._metric}  |  {n} frames"
        )

    def update(
        self,
        measured: Sequence[float | None],
        rolling: Sequence[float],
        window_size: int = 0,
        metric: str = "",
    ) -> None:
        """Full static render (fallback / future re-render)."""
        self._teardown_canvas()
        self._placeholder.grid_remove()
        self._live_mode  = False
        self._window_size = window_size
        self._metric      = metric

        n = len(measured)
        x = np.arange(n)
        meas_arr   = np.array(
            [v if v is not None else np.nan for v in measured], dtype=np.float64
        )
        roll_arr   = np.asarray(rolling, dtype=np.float64)
        factor_arr = self._compute_factors(meas_arr, roll_arr)

        self._meas_arr   = meas_arr
        self._roll_arr   = roll_arr
        self._factor_arr = factor_arr

        fig = Figure(figsize=(10, 4), dpi=100, facecolor=_BG)
        self._fig = fig
        ax1 = fig.add_subplot(111)
        ax1.set_facecolor(_AX_BG)
        self._ax1 = ax1

        (self._line_meas,) = ax1.plot(
            x, meas_arr, color=_C_MEASURED, linewidth=0.8, alpha=0.85,
            label="Measured luminance",
        )
        (self._line_roll,) = ax1.plot(
            x, roll_arr, color=_C_ROLLING, linewidth=2.0,
            label=f"Rolling average  (window {window_size})",
        )

        ax1.set_xlabel("Frame index", color=_TEXT, fontsize=9)
        ax1.set_ylabel("Luminance  (Y, Rec.709)", color=_TEXT, fontsize=9)
        ax1.set_xlim(0, max(1, n - 1))
        ax1.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor(_SPINE)
        ax1.grid(True, color=_GRID, linewidth=0.5, linestyle="--")

        ax2 = ax1.twinx()
        ax2.set_facecolor(_AX_BG)
        (self._line_fact,) = ax2.plot(
            x, factor_arr, color=_C_FACTOR, linewidth=1.0,
            linestyle="--", alpha=0.85, label="Correction factor  (right axis)",
        )
        ax2.axhline(1.0, color=_C_FACTOR, linewidth=0.5, linestyle=":", alpha=0.5)
        ax2.set_ylabel("Correction factor  (×)", color=_TEXT, fontsize=9)
        ax2.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor(_SPINE)
        self._ax2 = ax2

        self._tighten_axes(meas_arr, factor_arr)
        self._rebuild_legend()

        ax1.text(
            0.01, 0.97,
            f"Complete  —  {n} frames  |  metric: {metric}  |  window: {window_size}",
            transform=ax1.transAxes, color=_TEXT, fontsize=8, va="top",
        )
        self._progress_text = None   # not in live mode

        fig.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._canvas = canvas

        self._setup_hover()
        self._btn_recalc.configure(state="normal")
        self._info_var.set(
            f"window: {window_size}  |  metric: {metric}  |  {n} frames"
        )

    def clear(self) -> None:
        self._live_mode = False
        self._meas_arr = self._roll_arr = self._factor_arr = None
        self._teardown_canvas()
        self._placeholder.grid()
        self._btn_recalc.configure(state="disabled")
        self._info_var.set("")

    # ───────────────────────────────────────────── live redraw ────────────────

    def _live_redraw(self) -> None:
        if not self._live_mode:
            return
        if self._dirty and self._line_meas is not None:
            self._line_meas.set_ydata(self._live_data)
            valid = self._live_data[~np.isnan(self._live_data)]
            if len(valid) >= 2:
                self._ax1.set_ylim(
                    max(0.0,  valid.min() - 0.05),
                    min(1.05, valid.max() + 0.05),
                )
            n_done = int(np.sum(~np.isnan(self._live_data)))
            self._progress_text.set_text(
                f"Analysing…  {n_done} / {self._n_frames} frames"
            )
            self._canvas.draw_idle()
            self._dirty = False
        self.after(_REDRAW_MS, self._live_redraw)

    # ───────────────────────────────────────────── hover tooltip ─────────────

    def _setup_hover(self) -> None:
        """Create the hidden annotation and connect mouse events."""
        if self._ax1 is None or self._canvas is None:
            return

        self._annot = self._ax1.annotate(
            "",
            xy=(0, 0),
            xytext=(14, 14),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc="#2a2a2a", ec=_SPINE, alpha=0.92,
            ),
            color=_TEXT,
            fontsize=8,
            fontfamily="monospace",
            visible=False,
            zorder=10,
        )
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._canvas.mpl_connect("axes_leave_event",    self._on_axes_leave)

    def _on_hover(self, event) -> None:
        if event.inaxes not in (self._ax1, self._ax2):
            if self._annot and self._annot.get_visible():
                self._annot.set_visible(False)
                self._canvas.draw_idle()
            return

        if self._meas_arr is None or self._annot is None:
            return

        # Nearest frame index
        x_data = event.xdata
        if x_data is None:
            return
        idx = int(round(x_data))
        idx = max(0, min(idx, len(self._meas_arr) - 1))

        meas = self._meas_arr[idx]
        roll = self._roll_arr[idx]   if self._roll_arr   is not None else np.nan
        fact = self._factor_arr[idx] if self._factor_arr is not None else np.nan

        lines = [f"Frame {idx}"]
        if self._file_list and idx < len(self._file_list):
            lines.append(self._file_list[idx].name)
        if not np.isnan(meas):
            lines.append(f"Measured   {meas:.4f}")
        if not np.isnan(roll):
            lines.append(f"Average    {roll:.4f}")
        if not np.isnan(fact):
            lines.append(f"Factor     ×{fact:.4f}")

        # Flip tooltip to left side when near right edge (> 70 % of x range)
        xlim  = self._ax1.get_xlim()
        frac  = (idx - xlim[0]) / max(xlim[1] - xlim[0], 1)
        offset = (-110, 14) if frac > 0.70 else (14, 14)

        self._annot.set_text("\n".join(lines))
        self._annot.xy        = (idx, meas if not np.isnan(meas) else 0)
        self._annot.xytext    = offset
        self._annot.set_visible(True)
        self._canvas.draw_idle()

    def _on_axes_leave(self, event) -> None:
        if self._annot and self._annot.get_visible():
            self._annot.set_visible(False)
            self._canvas.draw_idle()

    # ───────────────────────────────────────────── helpers ────────────────────

    @staticmethod
    def _compute_factors(
        meas_arr: np.ndarray, roll_arr: np.ndarray
    ) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(meas_arr > 1e-6, roll_arr / meas_arr, np.nan)

    def _tighten_axes(
        self, meas_arr: np.ndarray, factor_arr: np.ndarray
    ) -> None:
        valid_m = meas_arr[~np.isnan(meas_arr)]
        if len(valid_m):
            self._ax1.set_ylim(
                max(0.0,  valid_m.min() - 0.05),
                min(1.05, valid_m.max() + 0.05),
            )
        valid_f = factor_arr[~np.isnan(factor_arr)]
        if len(valid_f):
            self._ax2.set_ylim(
                max(0.0, valid_f.min() - 0.05),
                valid_f.max() + 0.05,
            )

    def _rebuild_legend(self) -> None:
        lines1, labels1 = self._ax1.get_legend_handles_labels()
        lines2, labels2 = self._ax2.get_legend_handles_labels()
        self._ax1.legend(
            lines1 + lines2, labels1 + labels2,
            loc="upper right", fontsize=8,
            facecolor="#2a2a2a", edgecolor=_SPINE, labelcolor=_TEXT,
        )

    def _teardown_canvas(self) -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._canvas = None
            self._fig    = None
            self._ax1    = None
            self._ax2    = None
            self._line_meas = self._line_roll = self._line_fact = None
            self._live_data = None
            self._annot     = None
