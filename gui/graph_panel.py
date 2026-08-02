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

Keyframes
---------
After Pass 1 the user can click on or near the orange rolling-average line
to place a keyframe (white diamond marker).  Keyframes can be dragged
freely on both axes.  Each keyframe stores its own blend radius so tight
and wide corrections can coexist in the same sequence.

Clicking a keyframe selects it (gold outline); the "KF radius" spinner in
the toolbar then controls that keyframe's radius live.  Clicking empty
space deselects.  Right-click a keyframe to delete it.

Keyframes are accessible via get_keyframes() / set_keyframes() for
persistence in config.
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
_C_MEASURED   = "#5b9bd5"
_C_ROLLING    = "#ed7d31"
_C_ADJUSTED   = "#00d4ff"
_C_FACTOR     = "#70ad47"
_C_KF_NORMAL  = "#ffffff"   # unselected keyframe diamond
_C_KF_SEL     = "#ffd700"   # selected keyframe diamond (gold)
_BG           = "#1a1a1a"
_AX_BG        = "#1e1e1e"
_GRID         = "#333333"
_TEXT         = "#cccccc"
_SPINE        = "#444444"

_REDRAW_MS    = 250
_KF_PICK_PX   = 10    # pixel radius for keyframe hit-test
_KF_SNAP_PX   = 12    # pixel radius to snap new click onto rolling avg
_DEFAULT_KF_R = 30    # default radius for newly created keyframes


def blend_keyframes(
    roll_arr: np.ndarray,
    keyframes: dict[int, tuple[float, int]],
    default_radius: int = _DEFAULT_KF_R,
) -> np.ndarray:
    """
    Apply keyframe adjustments as *additive offsets* onto the rolling average,
    each tapered by its own per-keyframe cosine window.

    Data model
    ----------
    keyframes : {frame_index: (target_luminance, radius_in_frames)}

    Each keyframe records:
      - The luminance value the user dragged it to (absolute, stored as-is).
      - Its own blend radius (independent per keyframe).

    The *offset* applied to the rolling average is derived at blend time:
        offset = target - rolling[frame]
    so the adjusted curve inherits the rolling average's shape and trends —
    the keyframe just lifts or lowers it locally.

    Formula per keyframe at frame F, target T, radius R:
        offset = T - rolling[F]
        w(i)   = 0.5 * (1 + cos(π * |i−F| / R))  for |i−F| ≤ R, else 0
        contribution to total_offset[i] += w(i) * offset

    Multiple overlapping keyframes: contributions are summed directly.
    The result is added to the rolling average and clipped to [0, 1].

    Pure function — no Tk state — safe to call from a background thread.
    """
    if not keyframes:
        return roll_arr.copy()

    n = len(roll_arr)
    total_offset = np.zeros(n, dtype=np.float64)

    for frame, kf_data in keyframes.items():
        # Support both the new tuple format and a plain float (legacy/migration)
        if isinstance(kf_data, (tuple, list)):
            target, radius = float(kf_data[0]), int(kf_data[1])
        else:
            target, radius = float(kf_data), default_radius

        frame  = max(0, min(int(frame), n - 1))
        R      = max(1, radius)
        offset = target - roll_arr[frame]
        lo = max(0, frame - R)
        hi = min(n - 1, frame + R)
        indices = np.arange(lo, hi + 1)
        delta   = np.abs(indices - frame).astype(np.float64)
        w       = 0.5 * (1.0 + np.cos(np.pi * delta / R))
        total_offset[lo:hi+1] += w * offset

    return np.clip(roll_arr + total_offset, 0.0, 1.0)


class GraphPanel(ttk.LabelFrame):
    """
    on_recalculate        : callable() — Recalculate button.
    on_keyframes_changed  : callable(keyframes: dict) — fired on any KF change.
    """

    def __init__(
        self,
        parent,
        on_recalculate: Callable,
        on_keyframes_changed: Callable | None = None,
        **kw,
    ):
        super().__init__(parent, text="Pass 1 — Luminance Analysis", **kw)
        self._on_recalculate       = on_recalculate
        self._on_keyframes_changed = on_keyframes_changed or (lambda kf: None)

        # Figure objects
        self._fig    = None
        self._canvas: FigureCanvasTkAgg | None = None
        self._ax1    = None
        self._ax2    = None

        # Data arrays
        self._meas_arr:   np.ndarray | None = None
        self._roll_arr:   np.ndarray | None = None
        self._adj_arr:    np.ndarray | None = None
        self._factor_arr: np.ndarray | None = None

        self._file_list: list = []

        # Line artists
        self._line_meas = None
        self._line_roll = None
        self._line_adj  = None
        self._line_fact = None

        # Live-mode state
        self._live_mode  = False
        self._live_data: np.ndarray | None = None
        self._dirty      = False
        self._n_frames   = 0
        self._window_size = 0
        self._metric      = ""

        # Hover annotation
        self._annot = None

        # ── keyframe state ────────────────────────────────────────────────────
        # {frame_index (int): (target_luminance (float), radius (int))}
        self._keyframes: dict[int, tuple[float, int]] = {}
        self._default_radius = _DEFAULT_KF_R   # used for newly created KFs
        self._kf_markers: list = []
        self._drag_kf_idx:     int | None = None
        self._selected_kf_idx: int | None = None   # which KF the spinner edits

        self._build()

    # ─────────────────────────────────────────────────── build ────────────────

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        tb = ttk.Frame(self, padding=(6, 4))
        tb.grid(row=0, column=0, sticky="ew")

        self._info_var = tk.StringVar(value="")
        ttk.Label(tb, textvariable=self._info_var,
                  foreground="#888", font=("Segoe UI", 8)).pack(side="left")

        # ── per-keyframe radius (right side of toolbar, right-to-left) ────────
        self._btn_recalc = ttk.Button(
            tb, text="↻  Recalculate rolling average",
            command=self._on_recalculate, state="disabled",
        )
        self._btn_recalc.pack(side="right")

        self._btn_clear_kf = ttk.Button(
            tb, text="✕  Clear keyframes",
            command=self._clear_all_keyframes, state="disabled",
        )
        self._btn_clear_kf.pack(side="right", padx=(0, 8))

        # "KF radius" — edits the *selected* keyframe's radius
        ttk.Label(tb, text="frames", foreground="#888",
                  font=("Segoe UI", 8)).pack(side="right", padx=(0, 2))
        self._kf_radius_var = tk.IntVar(value=_DEFAULT_KF_R)
        self._kf_radius_spin = ttk.Spinbox(
            tb, from_=1, to=9999, width=5,
            textvariable=self._kf_radius_var,
            command=self._on_radius_change,
            state="disabled",
        )
        self._kf_radius_spin.pack(side="right")
        self._kf_radius_spin.bind("<Return>",   lambda _: self._on_radius_change())
        self._kf_radius_spin.bind("<FocusOut>", lambda _: self._on_radius_change())

        self._kf_radius_label = ttk.Label(
            tb, text="KF radius:", foreground="#888", font=("Segoe UI", 8)
        )
        self._kf_radius_label.pack(side="right", padx=(8, 2))

        # "Default radius" — used when a NEW keyframe is created
        ttk.Label(tb, text="frames", foreground="#888",
                  font=("Segoe UI", 8)).pack(side="right", padx=(0, 2))
        self._default_radius_var = tk.IntVar(value=_DEFAULT_KF_R)
        default_spin = ttk.Spinbox(
            tb, from_=1, to=9999, width=5,
            textvariable=self._default_radius_var,
            command=self._on_default_radius_change,
        )
        default_spin.pack(side="right")
        default_spin.bind("<Return>",   lambda _: self._on_default_radius_change())
        default_spin.bind("<FocusOut>", lambda _: self._on_default_radius_change())
        ttk.Label(tb, text="New KF radius:", foreground="#888",
                  font=("Segoe UI", 8)).pack(side="right", padx=(8, 2))

        # placeholder
        self._placeholder = ttk.Label(
            self, text="Run Pass 1 to see the luminance graph.",
            foreground="#666", anchor="center",
        )
        self._placeholder.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)

    # ─────────────────────────────────────────────── public API ───────────────

    def start_live(self, n_frames: int, window_size: int, metric: str,
                   file_list: list | None = None) -> None:
        self._teardown_canvas()
        self._placeholder.grid_remove()
        self._btn_recalc.configure(state="disabled")

        self._n_frames    = n_frames
        self._window_size = window_size
        self._metric      = metric
        self._file_list   = file_list or []
        self._live_data   = np.full(n_frames, np.nan, dtype=np.float64)
        self._meas_arr    = self._live_data
        self._roll_arr    = None
        self._adj_arr     = None
        self._factor_arr  = None
        self._dirty       = False
        self._live_mode   = True
        self._kf_markers  = []

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
            transform=ax1.transAxes, color=_TEXT, fontsize=8, va="top",
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
        if self._live_data is None or index >= len(self._live_data):
            return
        self._live_data[index] = value
        self._dirty = True

    def finish_live(
        self,
        measured: Sequence[float | None],
        rolling: Sequence[float],
    ) -> None:
        self._live_mode = False
        if self._fig is None:
            self.update(measured, rolling, self._window_size, self._metric)
            return

        n = len(measured)
        x = np.arange(n)
        meas_arr   = np.array([v if v is not None else np.nan for v in measured],
                               dtype=np.float64)
        roll_arr   = np.asarray(rolling, dtype=np.float64)
        adj_arr    = self._blend_keyframes(roll_arr)
        factor_arr = self._compute_factors(meas_arr, adj_arr)

        self._meas_arr   = meas_arr
        self._roll_arr   = roll_arr
        self._adj_arr    = adj_arr
        self._factor_arr = factor_arr

        self._line_meas.set_ydata(meas_arr)

        if self._line_roll:
            self._line_roll.remove()
        (self._line_roll,) = self._ax1.plot(
            x, roll_arr, color=_C_ROLLING, linewidth=2.0,
            label=f"Rolling average  (window {self._window_size})",
        )

        self._draw_adjusted_line(x, adj_arr)

        if self._line_fact:
            self._line_fact.remove()
        (self._line_fact,) = self._ax2.plot(
            x, factor_arr, color=_C_FACTOR, linewidth=1.0,
            linestyle="--", alpha=0.85, label="Correction factor  (right axis)",
        )
        self._ax2.axhline(1.0, color=_C_FACTOR, linewidth=0.5,
                          linestyle=":", alpha=0.5)

        self._tighten_axes(meas_arr, factor_arr)
        self._rebuild_legend()
        self._redraw_kf_markers()

        n_ok = int(np.sum(~np.isnan(meas_arr)))
        self._progress_text.set_text(
            f"Complete  —  {n_ok} / {n} frames analysed  |  "
            f"click orange line to add keyframes"
        )
        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()
        self._btn_recalc.configure(state="normal")
        self._setup_click_and_drag()

    def refresh_rolling(self, rolling: Sequence[float], window_size: int) -> None:
        if self._meas_arr is None or self._fig is None:
            return
        self._window_size = window_size
        roll_arr   = np.asarray(rolling, dtype=np.float64)
        adj_arr    = self._blend_keyframes(roll_arr)
        factor_arr = self._compute_factors(self._meas_arr, adj_arr)
        self._roll_arr   = roll_arr
        self._adj_arr    = adj_arr
        self._factor_arr = factor_arr

        n = len(roll_arr)
        x = np.arange(n)

        if self._line_roll:
            self._line_roll.remove()
        (self._line_roll,) = self._ax1.plot(
            x, roll_arr, color=_C_ROLLING, linewidth=2.0,
            label=f"Rolling average  (window {window_size})",
        )

        self._draw_adjusted_line(x, adj_arr)

        if self._line_fact:
            self._line_fact.remove()
        (self._line_fact,) = self._ax2.plot(
            x, factor_arr, color=_C_FACTOR, linewidth=1.0,
            linestyle="--", alpha=0.85, label="Correction factor  (right axis)",
        )

        self._tighten_axes(self._meas_arr, factor_arr)
        self._rebuild_legend()
        self._redraw_kf_markers()
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
        self._teardown_canvas()
        self._placeholder.grid_remove()
        self._live_mode   = False
        self._window_size = window_size
        self._metric      = metric
        self._kf_markers  = []

        n = len(measured)
        x = np.arange(n)
        meas_arr   = np.array([v if v is not None else np.nan for v in measured],
                               dtype=np.float64)
        roll_arr   = np.asarray(rolling, dtype=np.float64)
        adj_arr    = self._blend_keyframes(roll_arr)
        factor_arr = self._compute_factors(meas_arr, adj_arr)

        self._meas_arr   = meas_arr
        self._roll_arr   = roll_arr
        self._adj_arr    = adj_arr
        self._factor_arr = factor_arr

        fig = Figure(figsize=(10, 4), dpi=100, facecolor=_BG)
        self._fig = fig
        ax1 = fig.add_subplot(111)
        ax1.set_facecolor(_AX_BG)
        self._ax1 = ax1

        (self._line_meas,) = ax1.plot(x, meas_arr, color=_C_MEASURED,
                                       linewidth=0.8, alpha=0.85,
                                       label="Measured luminance")
        (self._line_roll,) = ax1.plot(x, roll_arr, color=_C_ROLLING,
                                       linewidth=2.0,
                                       label=f"Rolling average  (window {window_size})")
        self._line_adj = None
        self._draw_adjusted_line(x, adj_arr)

        ax1.set_xlabel("Frame index", color=_TEXT, fontsize=9)
        ax1.set_ylabel("Luminance  (Y, Rec.709)", color=_TEXT, fontsize=9)
        ax1.set_xlim(0, max(1, n - 1))
        ax1.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor(_SPINE)
        ax1.grid(True, color=_GRID, linewidth=0.5, linestyle="--")

        ax2 = ax1.twinx()
        ax2.set_facecolor(_AX_BG)
        (self._line_fact,) = ax2.plot(x, factor_arr, color=_C_FACTOR,
                                       linewidth=1.0, linestyle="--", alpha=0.85,
                                       label="Correction factor  (right axis)")
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
            f"Complete  —  {n} frames  |  metric: {metric}  |  window: {window_size}"
            f"  |  click orange line to add keyframes",
            transform=ax1.transAxes, color=_TEXT, fontsize=8, va="top",
        )
        self._progress_text = None
        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._canvas = canvas

        self._setup_hover()
        self._setup_click_and_drag()
        self._redraw_kf_markers()
        self._btn_recalc.configure(state="normal")
        self._info_var.set(f"window: {window_size}  |  metric: {metric}  |  {n} frames")

    def clear(self) -> None:
        self._live_mode = False
        self._meas_arr = self._roll_arr = self._adj_arr = self._factor_arr = None
        self._teardown_canvas()
        self._placeholder.grid()
        self._btn_recalc.configure(state="disabled")
        self._info_var.set("")

    def get_adjusted_rolling(self) -> np.ndarray | None:
        return self._adj_arr

    # ── keyframe persistence ──────────────────────────────────────────────────

    def get_keyframes(self) -> dict:
        """Return {str(frame): [luminance, radius]} — JSON-serialisable."""
        return {str(k): [float(v[0]), int(v[1])] for k, v in self._keyframes.items()}

    def get_keyframe_radius(self) -> int:
        """Return the default radius (used for new keyframes and config save)."""
        return self._default_radius

    def set_keyframes(self, kf: dict, radius: int | None = None) -> None:
        """Restore keyframes from config."""
        restored = {}
        for k, v in kf.items():
            frame = int(k)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                restored[frame] = (float(v[0]), int(v[1]))
            else:
                # legacy plain float — use stored default radius
                r = radius if radius is not None else _DEFAULT_KF_R
                restored[frame] = (float(v), r)
        self._keyframes = restored
        if radius is not None:
            self._default_radius = radius
            self._default_radius_var.set(radius)
        self._update_radius_spinner()
        if self._roll_arr is not None:
            self._apply_keyframes_and_redraw()

    # ─────────────────────────────────────────────── live redraw ──────────────

    def _live_redraw(self) -> None:
        if not self._live_mode:
            return
        if self._dirty and self._line_meas is not None:
            self._line_meas.set_ydata(self._live_data)
            valid = self._live_data[~np.isnan(self._live_data)]
            if len(valid) >= 2:
                self._ax1.set_ylim(max(0.0,  valid.min() - 0.05),
                                   min(1.05, valid.max() + 0.05))
            n_done = int(np.sum(~np.isnan(self._live_data)))
            self._progress_text.set_text(
                f"Analysing…  {n_done} / {self._n_frames} frames"
            )
            self._canvas.draw_idle()
            self._dirty = False
        self.after(_REDRAW_MS, self._live_redraw)

    # ─────────────────────────────────────────────── hover tooltip ────────────

    def _setup_hover(self) -> None:
        if self._ax1 is None or self._canvas is None:
            return
        self._annot = self._ax1.annotate(
            "",
            xy=(0, 0), xytext=(14, 14), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="#2a2a2a",
                      ec=_SPINE, alpha=0.92),
            color=_TEXT, fontsize=8, fontfamily="monospace",
            visible=False, zorder=10,
        )
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._canvas.mpl_connect("axes_leave_event",    self._on_axes_leave)

    def _on_hover(self, event) -> None:
        if self._drag_kf_idx is not None:
            return
        if event.inaxes not in (self._ax1, self._ax2):
            if self._annot and self._annot.get_visible():
                self._annot.set_visible(False)
                self._canvas.draw_idle()
            return
        if self._meas_arr is None or self._annot is None:
            return

        x_data = event.xdata
        if x_data is None:
            return
        idx = int(round(x_data))
        idx = max(0, min(idx, len(self._meas_arr) - 1))

        meas = self._meas_arr[idx]
        roll = self._roll_arr[idx]  if self._roll_arr  is not None else np.nan
        adj  = self._adj_arr[idx]   if self._adj_arr   is not None else np.nan
        fact = self._factor_arr[idx] if self._factor_arr is not None else np.nan

        lines = [f"Frame {idx}"]
        if self._file_list and idx < len(self._file_list):
            lines.append(self._file_list[idx].name)
        if not np.isnan(meas): lines.append(f"Measured   {meas:.4f}")
        if not np.isnan(roll): lines.append(f"Rolling    {roll:.4f}")
        if not np.isnan(adj) and self._keyframes:
            lines.append(f"Adjusted   {adj:.4f}")
        if not np.isnan(fact): lines.append(f"Factor     ×{fact:.4f}")
        if idx in self._keyframes:
            lum, r = self._keyframes[idx]
            lines.append(f"● keyframe  r={r}")

        xlim   = self._ax1.get_xlim()
        frac   = (idx - xlim[0]) / max(xlim[1] - xlim[0], 1)
        offset = (-110, 14) if frac > 0.70 else (14, 14)

        self._annot.set_text("\n".join(lines))
        self._annot.xy     = (idx, meas if not np.isnan(meas) else 0)
        self._annot.xytext = offset
        self._annot.set_visible(True)
        self._canvas.draw_idle()

    def _on_axes_leave(self, event) -> None:
        if self._annot and self._annot.get_visible():
            self._annot.set_visible(False)
            self._canvas.draw_idle()

    # ──────────────────────────────── click & drag (keyframes) ───────────────

    def _setup_click_and_drag(self) -> None:
        if self._canvas is None:
            return
        self._canvas.mpl_connect("button_press_event",   self._on_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_drag)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    def _resolve_ax1_coords(self, event) -> tuple[float, float] | None:
        """
        Return (x_data, y_data) in ax1's coordinate space regardless of
        which twin axis matplotlib assigned event.inaxes to.
        """
        if event.inaxes not in (self._ax1, self._ax2):
            return None
        x_data = event.xdata
        if x_data is None:
            return None
        if event.inaxes is self._ax1:
            y_data = event.ydata
        else:
            _, y_data = self._ax1.transData.inverted().transform(
                (event.x, event.y)
            )
        if y_data is None:
            return None
        return x_data, y_data

    def _on_press(self, event) -> None:
        if self._roll_arr is None:
            return
        coords = self._resolve_ax1_coords(event)
        if coords is None:
            return
        x_data, y_data = coords
        frame_idx = int(round(x_data))
        frame_idx = max(0, min(frame_idx, len(self._roll_arr) - 1))

        # ── right-click: delete ───────────────────────────────────────────────
        if event.button == 3:
            hit = self._hit_keyframe(x_data, y_data)
            if hit is not None:
                del self._keyframes[hit]
                if self._selected_kf_idx == hit:
                    self._selected_kf_idx = None
                    self._update_radius_spinner()
                self._apply_keyframes_and_redraw()
            return

        # ── left-click ────────────────────────────────────────────────────────
        if event.button == 1:
            hit = self._hit_keyframe(x_data, y_data)
            if hit is not None:
                # Grab existing keyframe for dragging + select it
                self._drag_kf_idx     = hit
                self._selected_kf_idx = hit
                self._update_radius_spinner()
                self._redraw_kf_markers()
                self._canvas.draw_idle()
                return

            # Click near rolling-average line → create new keyframe
            roll_y = self._roll_arr[frame_idx]
            disp_click = self._ax1.transData.transform((x_data, y_data))
            disp_roll  = self._ax1.transData.transform((frame_idx, roll_y))
            dist_px = np.hypot(disp_click[0] - disp_roll[0],
                               disp_click[1] - disp_roll[1])
            if dist_px <= _KF_SNAP_PX:
                r = self._default_radius
                self._keyframes[frame_idx] = (roll_y, r)
                self._drag_kf_idx     = frame_idx
                self._selected_kf_idx = frame_idx
                self._update_radius_spinner()
                self._apply_keyframes_and_redraw()
                return

            # Click on empty space → deselect
            if self._selected_kf_idx is not None:
                self._selected_kf_idx = None
                self._update_radius_spinner()
                self._redraw_kf_markers()
                self._canvas.draw_idle()

    def _on_drag(self, event) -> None:
        if self._drag_kf_idx is None:
            return
        coords = self._resolve_ax1_coords(event)
        if coords is None:
            return
        x_data, y_data = coords

        n = len(self._roll_arr)
        new_frame = int(round(x_data))
        new_frame = max(0, min(new_frame, n - 1))
        new_lum   = float(np.clip(y_data, 0.0, 1.0))

        old_frame = self._drag_kf_idx
        _, radius = self._keyframes.get(old_frame, (new_lum, self._default_radius))

        if new_frame != old_frame:
            del self._keyframes[old_frame]
            self._drag_kf_idx     = new_frame
            self._selected_kf_idx = new_frame

        self._keyframes[new_frame] = (new_lum, radius)
        self._apply_keyframes_and_redraw()

    def _on_release(self, event) -> None:
        if self._drag_kf_idx is not None:
            self._drag_kf_idx = None
            self._on_keyframes_changed(self.get_keyframes())

    # ─────────────────────────────── keyframe blend ───────────────────────────

    def _blend_keyframes(self, roll_arr: np.ndarray) -> np.ndarray:
        return blend_keyframes(roll_arr, self._keyframes, self._default_radius)

    def _apply_keyframes_and_redraw(self) -> None:
        if self._roll_arr is None or self._fig is None:
            return

        adj_arr    = self._blend_keyframes(self._roll_arr)
        factor_arr = self._compute_factors(self._meas_arr, adj_arr)
        self._adj_arr    = adj_arr
        self._factor_arr = factor_arr

        n = len(adj_arr)
        x = np.arange(n)

        self._draw_adjusted_line(x, adj_arr)

        if self._line_fact:
            self._line_fact.remove()
        (self._line_fact,) = self._ax2.plot(
            x, factor_arr, color=_C_FACTOR, linewidth=1.0,
            linestyle="--", alpha=0.85, label="Correction factor  (right axis)",
        )
        self._tighten_axes(self._meas_arr, factor_arr)
        self._rebuild_legend()
        self._redraw_kf_markers()
        self._canvas.draw_idle()

        has_kf = bool(self._keyframes)
        self._btn_clear_kf.configure(state="normal" if has_kf else "disabled")

    # ─────────────────────────────── adjusted line ────────────────────────────

    def _draw_adjusted_line(self, x: np.ndarray, adj_arr: np.ndarray) -> None:
        if self._line_adj is not None:
            self._line_adj.remove()
            self._line_adj = None
        if not self._keyframes or self._ax1 is None:
            return
        (self._line_adj,) = self._ax1.plot(
            x, adj_arr, color=_C_ADJUSTED, linewidth=1.5,
            linestyle="-", alpha=0.9,
            label="Adjusted  (used by Pass 2)", zorder=3,
        )

    # ──────────────────────────────── KF markers ──────────────────────────────

    def _redraw_kf_markers(self) -> None:
        for sc in self._kf_markers:
            try:
                sc.remove()
            except Exception:
                pass
        self._kf_markers = []
        if not self._keyframes or self._ax1 is None:
            return

        sel = self._selected_kf_idx
        for frame, (lum, _) in self._keyframes.items():
            color     = _C_KF_SEL    if frame == sel else _C_KF_NORMAL
            edgecolor = "#ffa500"    if frame == sel else "#000000"
            size      = 90           if frame == sel else 60
            sc = self._ax1.scatter(
                [frame], [lum],
                marker="D", s=size, color=color,
                edgecolors=edgecolor, linewidths=1.2 if frame == sel else 0.8,
                zorder=5, label="_nolegend_",
            )
            self._kf_markers.append(sc)

    # ──────────────────────────────── hit-test ────────────────────────────────

    def _hit_keyframe(self, x_data: float, y_data: float) -> int | None:
        if not self._keyframes or self._ax1 is None:
            return None
        best_dist = _KF_PICK_PX + 1
        best_idx  = None
        disp_click = self._ax1.transData.transform((x_data, y_data))
        for frame, (lum, _) in self._keyframes.items():
            disp_kf = self._ax1.transData.transform((frame, lum))
            dist = np.hypot(disp_kf[0] - disp_click[0],
                            disp_kf[1] - disp_click[1])
            if dist < best_dist:
                best_dist = dist
                best_idx  = frame
        return best_idx

    # ──────────────────────────── toolbar actions ─────────────────────────────

    def _update_radius_spinner(self) -> None:
        """Sync the KF-radius spinner to the currently selected keyframe."""
        sel = self._selected_kf_idx
        if sel is not None and sel in self._keyframes:
            _, r = self._keyframes[sel]
            self._kf_radius_var.set(r)
            self._kf_radius_spin.configure(state="normal")
            self._kf_radius_label.configure(foreground=_C_KF_SEL)
        else:
            self._kf_radius_spin.configure(state="disabled")
            self._kf_radius_label.configure(foreground="#888")

    def _on_radius_change(self) -> None:
        """Apply spinner value to the currently selected keyframe."""
        sel = self._selected_kf_idx
        if sel is None or sel not in self._keyframes:
            return
        try:
            r = max(1, int(self._kf_radius_var.get()))
        except (ValueError, tk.TclError):
            return
        lum, _ = self._keyframes[sel]
        self._keyframes[sel] = (lum, r)
        if self._roll_arr is not None:
            self._apply_keyframes_and_redraw()
        self._on_keyframes_changed(self.get_keyframes())

    def _on_default_radius_change(self) -> None:
        try:
            self._default_radius = max(1, int(self._default_radius_var.get()))
        except (ValueError, tk.TclError):
            pass

    def _clear_all_keyframes(self) -> None:
        self._keyframes.clear()
        self._selected_kf_idx = None
        self._update_radius_spinner()
        self._apply_keyframes_and_redraw()
        self._on_keyframes_changed({})

    # ─────────────────────────────── helpers ──────────────────────────────────

    @staticmethod
    def _compute_factors(meas_arr, roll_arr):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(meas_arr > 1e-6, roll_arr / meas_arr, np.nan)

    def _tighten_axes(self, meas_arr, factor_arr):
        valid_m = meas_arr[~np.isnan(meas_arr)]
        if len(valid_m):
            self._ax1.set_ylim(max(0.0,  valid_m.min() - 0.05),
                               min(1.05, valid_m.max() + 0.05))
        valid_f = factor_arr[~np.isnan(factor_arr)]
        if len(valid_f):
            self._ax2.set_ylim(max(0.0, valid_f.min() - 0.05),
                               valid_f.max() + 0.05)

    def _rebuild_legend(self):
        lines1, labels1 = self._ax1.get_legend_handles_labels()
        lines2, labels2 = self._ax2.get_legend_handles_labels()
        combined = [(l, lb) for l, lb in zip(lines1 + lines2, labels1 + labels2)
                    if lb != "_nolegend_"]
        if combined:
            ls, lbs = zip(*combined)
            self._ax1.legend(ls, lbs, loc="upper right", fontsize=8,
                             facecolor="#2a2a2a", edgecolor=_SPINE,
                             labelcolor=_TEXT)

    def _teardown_canvas(self):
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._canvas    = None
            self._fig       = None
            self._ax1       = None
            self._ax2       = None
            self._line_meas = self._line_roll = self._line_adj = self._line_fact = None
            self._live_data = None
            self._annot     = None
            self._kf_markers = []
