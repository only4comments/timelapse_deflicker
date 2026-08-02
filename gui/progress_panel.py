"""
Progress bar, phase label, ETA, frames/sec.
Run-mode selector (Pass 1+2 / Pass 1 only / Pass 2 only).
Run and Cancel buttons.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import time


class ProgressPanel(ttk.Frame):
    def __init__(self, parent, on_run, on_cancel, **kw):
        super().__init__(parent, **kw)
        self._on_run = on_run
        self._on_cancel = on_cancel
        self._start_time: float | None = None
        self._last_update: float = 0.0
        self._build()

    def _build(self) -> None:
        self.columnconfigure(2, weight=1)

        # ── run mode ─────────────────────────────────────────────────────────
        mode_frame = ttk.LabelFrame(self, text="Run mode", padding=(8, 2))
        mode_frame.grid(row=0, column=0, columnspan=6, sticky="w", padx=(0, 16), pady=(2, 4))

        self._run_mode_var = tk.StringVar(value="both")

        self._rb_both = ttk.Radiobutton(
            mode_frame, text="Pass 1 + Pass 2",
            variable=self._run_mode_var, value="both",
        )
        self._rb_both.pack(side="left", padx=8)

        self._rb_p1 = ttk.Radiobutton(
            mode_frame, text="Pass 1 only  (analyse)",
            variable=self._run_mode_var, value="pass1",
        )
        self._rb_p1.pack(side="left", padx=8)

        self._rb_p2 = ttk.Radiobutton(
            mode_frame, text="Pass 2 only  (apply & export)",
            variable=self._run_mode_var, value="pass2",
            state="disabled",
        )
        self._rb_p2.pack(side="left", padx=8)

        # ── buttons + bar row ────────────────────────────────────────────────
        self._btn_run = ttk.Button(self, text="▶  Run", command=self._on_run, width=12)
        self._btn_run.grid(row=1, column=0, padx=(0, 6), pady=4)

        self._btn_cancel = ttk.Button(
            self, text="■  Cancel", command=self._on_cancel,
            width=12, state="disabled",
        )
        self._btn_cancel.grid(row=1, column=1, padx=(0, 10), pady=4)

        self._progress_var = tk.DoubleVar(value=0.0)
        self._pbar = ttk.Progressbar(
            self, variable=self._progress_var, maximum=100.0,
        )
        self._pbar.grid(row=1, column=2, sticky="ew", padx=(0, 10), pady=4)

        # ── stats ─────────────────────────────────────────────────────────────
        stats = ttk.Frame(self)
        stats.grid(row=1, column=3, sticky="e")

        self._phase_var = tk.StringVar(value="Idle")
        self._eta_var   = tk.StringVar(value="")
        self._fps_var   = tk.StringVar(value="")
        self._count_var = tk.StringVar(value="")

        ttk.Label(stats, textvariable=self._phase_var,
                  foreground="#555", width=32, anchor="w").grid(row=0, column=0, padx=4)
        ttk.Label(stats, textvariable=self._count_var,
                  foreground="#555", width=14, anchor="e").grid(row=0, column=1, padx=4)
        ttk.Label(stats, textvariable=self._eta_var,
                  foreground="#555", width=12, anchor="e").grid(row=0, column=2, padx=4)
        ttk.Label(stats, textvariable=self._fps_var,
                  foreground="#555", width=10, anchor="e").grid(row=0, column=3, padx=4)

    # ------------------------------------------------------------------ public

    def get_run_mode(self) -> str:
        """Returns "both" | "pass1" | "pass2"."""
        return self._run_mode_var.get()

    def set_pass2_enabled(self, enabled: bool) -> None:
        """Enable/disable the Pass-2-only radio button."""
        self._rb_p2.configure(state="normal" if enabled else "disabled")

    def set_running(self, running: bool) -> None:
        if running:
            self._btn_run.configure(state="disabled")
            self._btn_cancel.configure(state="normal")
            for rb in (self._rb_both, self._rb_p1, self._rb_p2):
                rb.configure(state="disabled")
            self._start_time = time.monotonic()
        else:
            self._btn_run.configure(state="normal")
            self._btn_cancel.configure(state="disabled")
            self._rb_both.configure(state="normal")
            self._rb_p1.configure(state="normal")
            # Pass-2-only stays at whatever state it was before running
            self._start_time = None
            self._fps_var.set("")
            self._eta_var.set("")

    def update_progress(self, done: int, total: int, phase: str) -> None:
        if total == 0:
            return
        self._progress_var.set(done / total * 100.0)
        self._phase_var.set(phase)
        self._count_var.set(f"{done} / {total} frames")

        now = time.monotonic()
        if self._start_time and (now - self._last_update) >= 0.5:
            elapsed = now - self._start_time
            fps = done / elapsed if elapsed > 0 else 0.0
            remaining = (total - done) / fps if fps > 0 else 0.0
            self._fps_var.set(f"{fps:.1f} fps")
            self._eta_var.set(f"ETA {_fmt_seconds(remaining)}")
            self._last_update = now

    def set_phase(self, text: str) -> None:
        self._phase_var.set(text)

    def reset(self) -> None:
        self._progress_var.set(0.0)
        self._phase_var.set("Idle")
        self._eta_var.set("")
        self._fps_var.set("")
        self._count_var.set("")
        self._start_time = None


def _fmt_seconds(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"
