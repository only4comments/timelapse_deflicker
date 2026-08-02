"""
Timelapse Deflicker — main window.

Layout (top to bottom, single column)
--------------------------------------
Row 0 : Source bar   — folder picker + auto-detect info card
Row 1 : Graph panel  — always visible, grows to fill space
Row 2 : Action bar   — Analyse | Deflicker | Cancel | progress + ETA
Row 3 : Log panel    — compact, fixed height, dark console

Flow enforced by button states
-------------------------------
  Analyse   — always available when a source folder is loaded
  Deflicker — enabled only after Analyse has completed successfully
              (or after Recalculate)
  Both disable and Cancel appears while processing.

Run modes (internal only, not exposed in UI)
  "pass1" — source → analysis → graph
  "pass2" — stored results → corrections → output files
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from config import Config
from core.scanner import scan_folder
from core.pass1 import run_pass1
from core.pass2 import run_pass2
from core.rolling_average import rolling_average

from gui.settings_panel import SettingsPanel
from gui.graph_panel import GraphPanel, blend_keyframes
from gui.log_panel import LogPanel


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self._root   = root
        self._config = Config()

        self._file_list:     list[Path] = []
        self._source_folder: Path | None = None
        self._dest_folder:   Path | None = None

        self._luminance_values: list[float | None] = []
        self._rolling_lum:      np.ndarray | None = None
        self._pass1_settings:   dict | None = None

        self._processing    = False
        self._cancel_event  = threading.Event()
        self._progress_queue: queue.Queue = queue.Queue()
        self._pool = None
        self._hw_profile    = None   # last HWProfile from auto-detect

        # timing for ETA
        self._run_start_time: float = 0.0
        self._last_eta_update: float = 0.0

        self._build_ui()
        self._restore_config()
        self._poll_queue()

    # ──────────────────────────────────────────────────── UI build ────────────

    def _build_ui(self) -> None:
        root = self._root
        root.title("Timelapse Deflicker")
        root.minsize(1000, 700)
        root.protocol("WM_DELETE_WINDOW", self._on_closing)

        try:
            ttk.Style().theme_use("vista")
        except Exception:
            try:
                ttk.Style().theme_use("clam")
            except Exception:
                pass

        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)   # graph expands

        # ── Row 0: source bar ─────────────────────────────────────────────────
        self._build_source_bar(root)

        # ── Row 1: settings (left) + graph (right) ────────────────────────────
        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 0))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._settings = SettingsPanel(body, width=260)
        self._settings.grid(row=0, column=0, sticky="ns", padx=(0, 6))

        self._graph = GraphPanel(
            body,
            on_recalculate=self._on_recalculate,
            on_keyframes_changed=self._on_keyframes_changed,
        )
        self._graph.grid(row=0, column=1, sticky="nsew")

        # ── Row 2: action bar ─────────────────────────────────────────────────
        self._build_action_bar(root)

        # ── Row 3: log ───────────────────────────────────────────────────────
        self._log = LogPanel(root, padding=4)
        self._log.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 6))

    def _build_source_bar(self, root: tk.Tk) -> None:
        bar = ttk.Frame(root, padding=(8, 6, 8, 4))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(4, weight=1)

        # Source
        ttk.Button(bar, text="📂  Source folder",
                   command=self._open_source).grid(row=0, column=0, padx=(0, 6))
        self._src_label = ttk.Label(bar, text="No folder selected",
                                    foreground="#888", anchor="w")
        self._src_label.grid(row=0, column=1, sticky="ew")

        # Separator
        ttk.Separator(bar, orient="vertical").grid(
            row=0, column=2, sticky="ns", padx=12
        )

        # Output
        ttk.Button(bar, text="📁  Output folder",
                   command=self._open_dest).grid(row=0, column=3, padx=(0, 6))
        self._dst_label = ttk.Label(bar, text="No folder selected",
                                    foreground="#888", anchor="w")
        self._dst_label.grid(row=0, column=4, sticky="ew")

        # Auto-detect info card (row 1, spans full width)
        self._info_card = ttk.Frame(bar, relief="groove", borderwidth=1)
        self._info_card.grid(row=1, column=0, columnspan=5, sticky="ew",
                              pady=(6, 0))
        self._info_card.columnconfigure(1, weight=1)

        self._info_icon = ttk.Label(self._info_card, text="ℹ",
                                     foreground="#555", font=("Segoe UI", 10))
        self._info_icon.grid(row=0, column=0, padx=(8, 4), pady=4)

        self._info_var = tk.StringVar(
            value="Open a source folder — settings will be configured automatically."
        )
        self._info_label = ttk.Label(
            self._info_card, textvariable=self._info_var,
            foreground="#222", font=("Segoe UI", 9), anchor="w",
        )
        self._info_label.grid(row=0, column=1, sticky="ew", pady=4)

        self._btn_perf = ttk.Button(
            self._info_card, text="⚙  Performance options",
            command=self._open_perf_dialog, state="disabled",
        )
        self._btn_perf.grid(row=0, column=2, padx=(4, 8), pady=4)

        ttk.Separator(root, orient="horizontal").grid(
            row=0, column=0, sticky="ew", pady=(72, 0)
        )

    def _build_action_bar(self, root: tk.Tk) -> None:
        bar = ttk.Frame(root, padding=(8, 4))
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(4, weight=1)   # progress bar expands

        self._btn_analyse = ttk.Button(
            bar, text="▶  Analyse",
            command=self._on_analyse, state="disabled", width=14,
        )
        self._btn_analyse.grid(row=0, column=0, padx=(0, 6))

        self._btn_deflicker = ttk.Button(
            bar, text="✦  Deflicker",
            command=self._on_deflicker, state="disabled", width=14,
        )
        self._btn_deflicker.grid(row=0, column=1, padx=(0, 6))

        self._btn_cancel = ttk.Button(
            bar, text="■  Cancel",
            command=self._on_cancel, state="disabled", width=10,
        )
        self._btn_cancel.grid(row=0, column=2, padx=(0, 12))

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0.0)
        self._pbar = ttk.Progressbar(bar, variable=self._progress_var,
                                      maximum=100.0)
        self._pbar.grid(row=0, column=4, sticky="ew", padx=(0, 10))

        # Status / ETA
        self._status_var = tk.StringVar(value="")
        self._eta_var    = tk.StringVar(value="")
        self._fps_var    = tk.StringVar(value="")

        ttk.Label(bar, textvariable=self._status_var,
                  foreground="#666", width=36, anchor="w",
                  ).grid(row=0, column=5)
        ttk.Label(bar, textvariable=self._eta_var,
                  foreground="#666", width=12, anchor="e",
                  ).grid(row=0, column=6)
        ttk.Label(bar, textvariable=self._fps_var,
                  foreground="#666", width=10, anchor="e",
                  ).grid(row=0, column=7)

    # ──────────────────────────────────────────────────── config ─────────────

    def _restore_config(self) -> None:
        self._settings.apply_settings(dict(self._config._data))
        self._graph.set_keyframes(
            self._config.get("keyframes", {}),
            radius=self._config.get("keyframe_radius", 30),
        )
        src = self._config["source_folder"]
        dst = self._config["dest_folder"]
        if src and Path(src).is_dir():
            self._load_source(Path(src), quiet=True)
        if dst:
            self._dest_folder = Path(dst)
            self._dst_label.configure(text=str(dst), foreground="#222")

    def _save_config(self) -> None:
        s = self._settings.get_settings()
        for k, v in s.items():
            self._config[k] = v
        self._config["source_folder"] = str(self._source_folder or "")
        self._config["dest_folder"]   = str(self._dest_folder   or "")
        self._config["keyframes"]        = self._graph.get_keyframes()
        self._config["keyframe_radius"]  = self._graph.get_keyframe_radius()
        self._config.save()

    # ──────────────────────────────────────────────── folder actions ──────────

    def _open_source(self) -> None:
        folder = filedialog.askdirectory(title="Select source image folder")
        if folder:
            self._load_source(Path(folder))

    def _load_source(self, folder: Path, quiet: bool = False) -> None:
        result = scan_folder(folder)
        if not result.files:
            if not quiet:
                messagebox.showwarning(
                    "Empty folder",
                    "No supported image files found.\n"
                    "Supported: TIFF (.tif/.tiff), JPEG (.jpg/.jpeg)",
                )
            return

        if result.is_mixed:
            groups_str = ", ".join(
                f"{g} ({len(result.groups[g])} files)" for g in result.group_names
            )
            choice = _ask_mixed_format(self._root, result.group_names, groups_str)
            if choice is None:
                return
            result = result.filter_to_group(choice)

        self._source_folder = folder
        self._file_list     = result.files
        self._src_label.configure(text=str(folder), foreground="#222")
        self._invalidate_pass1()

        # Auto-detect hardware + frame size
        self._hw_profile = None
        try:
            from core.hw_detect import detect
            profile = detect(first_file=self._file_list[0])
            self._hw_profile = profile
            self._settings.apply_hw_profile(profile)
            self._info_var.set(
                f"{len(self._file_list)} files  ·  {profile.source}  "
                f"·  {profile.per_worker_mb:.0f} MB/worker  "
                f"·  Analysis: {profile.pass1_workers} workers, "
                f"Export: {profile.pass2_workers} workers"
            )
            self._info_icon.configure(text="✓", foreground="#2a7a2a")
            self._info_label.configure(foreground="#111111")
            self._btn_perf.configure(state="normal")
        except Exception:
            self._info_var.set(
                f"{len(self._file_list)} files loaded  —  "
                f"auto-detect unavailable."
            )
            self._info_icon.configure(text="⚠", foreground="#b87800")
            self._info_label.configure(foreground="#222222")
            self._btn_perf.configure(state="disabled")

        self._btn_analyse.configure(state="normal")
        self._log.append(f"Source: {folder}  ({len(self._file_list)} files)", "info")

    def _open_dest(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self._dest_folder = Path(folder)
            self._dst_label.configure(text=folder, foreground="#222")

    def _open_perf_dialog(self) -> None:
        """Open the Performance Options modal dialog."""
        _PerfDialog(self._root, self._hw_profile, self._settings, self._on_perf_applied)

    def _on_perf_applied(self, p1: int, p2: int) -> None:
        """Called by _PerfDialog when the user clicks OK with custom values."""
        self._settings._workers_p1_var.set(p1)
        self._settings._workers_p2_var.set(p2)
        self._info_var.set(
            self._info_var.get().split("  ·  Analysis:")[0]
            + f"  ·  Analysis: {p1} workers, Export: {p2} workers  (manual)"
        )
        self._log.append(
            f"Performance overridden manually: Analysis={p1}, Export={p2}", "info"
        )

    def _invalidate_pass1(self) -> None:
        self._luminance_values = []
        self._rolling_lum      = None
        self._pass1_settings   = None
        self._btn_deflicker.configure(state="disabled")
        self._graph.clear()
        self._graph.set_keyframes({})
        self._config["keyframes"] = {}

    # ──────────────────────────────────────────────── recalculate ────────────

    def _on_keyframes_changed(self, keyframes: dict) -> None:
        adjusted = self._graph.get_adjusted_rolling()
        if adjusted is not None:
            self._rolling_lum = adjusted

        self._config["keyframes"]       = keyframes
        self._config["keyframe_radius"] = self._graph.get_keyframe_radius()
        self._config.save()

        if self._rolling_lum is not None:
            n = len(keyframes)
            if n:
                self._log.append(
                    f"Keyframes updated  —  {n} active  |  "
                    f"will be used on next Deflicker run", "info"
                )
            else:
                self._log.append("All keyframes cleared.", "info")

    def _on_recalculate(self) -> None:
        if not self._luminance_values:
            return
        settings    = self._settings.get_settings()
        window      = settings["window_size"]
        valid_lum   = [v if v is not None else 0.0 for v in self._luminance_values]
        rolling_lum = rolling_average(valid_lum, window)
        self._rolling_lum   = rolling_lum
        self._rolling_hists = None

        self._graph.refresh_rolling(rolling_lum, window)
        adjusted = self._graph.get_adjusted_rolling()
        if adjusted is not None:
            self._rolling_lum = adjusted

        self._log.append(
            f"Rolling average recalculated  —  window={window}", "info"
        )

    # ──────────────────────────────────────────────── action buttons ─────────

    def _on_analyse(self) -> None:
        if not self._file_list:
            messagebox.showwarning("No source", "Please open a source folder first.")
            return

        settings = self._settings.get_settings()
        self._start_run(settings, run_mode="pass1")

    def _on_deflicker(self) -> None:
        if self._rolling_lum is None:
            messagebox.showwarning(
                "Analysis required",
                "Please run Analyse first so the luminance curve is available.",
            )
            return

        if not self._dest_folder:
            messagebox.showwarning(
                "No output folder",
                "Please select an output folder before deflickering.",
            )
            return

        settings = self._settings.get_settings()

        # Check for existing files in output folder
        from core.image_io import output_extension
        ext = output_extension(settings["output_format"])
        existing = (
            [f for f in self._dest_folder.iterdir()
             if f.is_file() and f.suffix.lower() == ext]
            if self._dest_folder.exists() else []
        )
        dest_action = "overwrite"
        if existing:
            dest_action = _ask_dest_conflict(self._root, len(existing))
            if dest_action is None:
                return

        self._dest_folder.mkdir(parents=True, exist_ok=True)
        self._start_run(settings, run_mode="pass2", dest_action=dest_action)

    def _on_cancel(self) -> None:
        if self._processing:
            self._cancel_event.set()
            self._log.append("Cancel requested — finishing current batch…", "warn")

    def _start_run(
        self,
        settings: dict,
        run_mode: str,
        dest_action: str = "overwrite",
    ) -> None:
        self._cancel_event.clear()
        self._processing     = True
        self._run_start_time = time.monotonic()
        self._last_eta_update = 0.0

        self._btn_analyse.configure(state="disabled")
        self._btn_deflicker.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._progress_var.set(0.0)
        self._status_var.set("")
        self._eta_var.set("")
        self._fps_var.set("")

        if run_mode == "pass1":
            self._log.clear()
            self._log.append("═" * 60, "dim")
            self._log.append(
                f"Analysing  —  {len(self._file_list)} frames  |  "
                f"window={settings['window_size']}  |  metric={settings['metric']}  |  "
                f"workers={settings['worker_count']}",
                "info",
            )
            self._graph.start_live(
                n_frames    = len(self._file_list),
                window_size = settings["window_size"],
                metric      = settings["metric"],
                file_list   = self._file_list,
            )
        else:
            self._log.append("═" * 60, "dim")
            self._log.append(
                f"Deflickering  —  {len(self._file_list)} frames  |  "
                f"mode={settings['correction_mode']}  |  "
                f"output={settings['output_format']}  |  "
                f"workers={settings.get('worker_count_p2', settings['worker_count'])}",
                "info",
            )

        executor = None
        threading.Thread(
            target=self._processing_thread,
            args=(settings, run_mode, dest_action, executor),
            daemon=True,
        ).start()

    # ──────────────────────────────────────────────── processing thread ───────

    def _processing_thread(
        self, settings: dict, run_mode: str, dest_action: str, executor
    ) -> None:
        q     = self._progress_queue
        files = self._file_list

        try:
            if run_mode == "pass1":
                q.put(("phase", "Analysing luminance…"))
                lum_values, _ = run_pass1(
                    file_list        = files,
                    metric           = settings["metric"],
                    percentile_value = settings["percentile_value"],
                    need_histogram   = False,
                    worker_count     = settings["worker_count"],
                    progress_callback= lambda done, total, r: q.put(
                        ("pass1_progress", done, total, r)
                    ),
                    cancel_event     = self._cancel_event,
                    executor         = executor,
                )

                if self._cancel_event.is_set():
                    q.put(("cancelled",))
                    return

                q.put(("phase", "Computing rolling average…"))
                valid_lum   = [v if v is not None else 0.0 for v in lum_values]
                rolling_lum = rolling_average(valid_lum, settings["window_size"])

                q.put(("pass1_done", lum_values, None,
                       rolling_lum, None, settings))

            else:  # pass2
                q.put(("phase", "Applying corrections…"))

                lum_values  = self._luminance_values
                rolling_lum = self._rolling_lum

                errors = run_pass2(
                    file_list        = files,
                    dest_folder      = self._dest_folder,
                    rolling_lum      = rolling_lum,
                    measured_lum     = lum_values,
                    rolling_hists    = None,
                    correction_mode  = settings["correction_mode"],
                    output_format    = settings["output_format"],
                    jpeg_quality     = settings["jpeg_quality"],
                    resize_enabled   = settings.get("resize_enabled", False),
                    resize_edge      = settings.get("resize_edge", "shorter"),
                    resize_value     = settings.get("resize_value", "1080"),
                    worker_count     = settings.get("worker_count_p2",
                                                    settings["worker_count"]),
                    progress_callback= lambda done, total, r: q.put(
                        ("pass2_progress", done, total, r)
                    ),
                    cancel_event     = self._cancel_event,
                    executor         = executor,
                )

                if self._cancel_event.is_set():
                    q.put(("cancelled",))
                else:
                    q.put(("done", errors))

        except Exception as exc:
            q.put(("error", str(exc)))

    # ──────────────────────────────────────────────── queue polling ───────────

    def _poll_queue(self) -> None:
        try:
            while True:
                self._handle_message(self._progress_queue.get_nowait())
        except queue.Empty:
            pass
        self._root.after(80, self._poll_queue)

    def _handle_message(self, msg: tuple) -> None:
        kind = msg[0]

        if kind == "phase":
            self._status_var.set(msg[1])

        elif kind == "pass1_progress":
            _, done, total, result = msg
            self._progress_var.set(done / total * 100.0)
            self._status_var.set(f"Analysing  {done} / {total} frames")
            self._update_eta(done, total)
            if result["error"]:
                self._log.append(
                    f"SKIP  [{result['index']:04d}]  "
                    f"{Path(result['filepath']).name}: {result['error']}", "error"
                )
            else:
                self._graph.add_point(result["index"], result["luminance"])
                self._log.append(
                    f"[{result['index']:04d}]  {Path(result['filepath']).name}  "
                    f"Y={result['luminance']:.4f}", "dim"
                )

        elif kind == "pass1_done":
            _, lum_values, _histograms, rolling_lum, _rolling_hists, p1_settings = msg
            self._luminance_values = lum_values
            self._rolling_lum      = rolling_lum
            self._pass1_settings   = p1_settings

            n_ok = sum(1 for v in lum_values if v is not None)
            self._log.append(
                f"Analysis complete  —  {n_ok}/{len(lum_values)} frames", "success"
            )

            valid = np.array([v for v in lum_values if v is not None], dtype=np.float64)
            if len(valid) >= 2:
                lum_cv = float(valid.std() / valid.mean()) if valid.mean() > 1e-6 else 0.0
                self._log.append(
                    f"  Y range {valid.min():.4f} – {valid.max():.4f}  "
                    f"mean={valid.mean():.4f}  std={valid.std():.4f}", "dim"
                )
                if lum_cv < 0.01:
                    self._log.append(
                        "  ⚠  Very low variance — footage may already be deflickered.", "warn"
                    )

            self._graph.finish_live(lum_values, rolling_lum)
            adjusted = self._graph.get_adjusted_rolling()
            if adjusted is not None:
                self._rolling_lum = adjusted

            self._finish_processing(cancelled=False, errors=[], pass1_only=True)

        elif kind == "pass2_progress":
            _, done, total, result = msg
            self._progress_var.set(done / total * 100.0)
            self._status_var.set(f"Deflickering  {done} / {total} frames")
            self._update_eta(done, total)
            if result["error"]:
                self._log.append(
                    f"FAIL  {Path(result['src']).name}: {result['error']}", "error"
                )
            else:
                self._log.append(
                    f"{Path(result['src']).name}  ×{result['correction_factor']:.4f}",
                    "dim"
                )

        elif kind == "done":
            self._finish_processing(cancelled=False, errors=msg[1])

        elif kind == "cancelled":
            self._finish_processing(cancelled=True, errors=[])

        elif kind == "error":
            self._log.append(f"FATAL ERROR: {msg[1]}", "error")
            self._finish_processing(cancelled=False, errors=[msg[1]])

    def _update_eta(self, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last_eta_update < 0.5:
            return
        self._last_eta_update = now
        elapsed = now - self._run_start_time
        if elapsed > 0 and done > 0:
            fps = done / elapsed
            remaining = (total - done) / fps
            self._fps_var.set(f"{fps:.1f} fps")
            self._eta_var.set(f"ETA {_fmt_seconds(remaining)}")

    # ────────────────────────────────────────────────── finish ───────────────

    def _finish_processing(
        self,
        cancelled: bool,
        errors: list[str],
        pass1_only: bool = False,
    ) -> None:
        self._processing = False
        self._btn_cancel.configure(state="disabled")
        self._btn_analyse.configure(
            state="normal" if self._file_list else "disabled"
        )
        self._btn_deflicker.configure(
            state="normal" if self._rolling_lum is not None else "disabled"
        )
        self._fps_var.set("")
        self._eta_var.set("")

        if cancelled:
            self._status_var.set("Cancelled")
            self._log.append("Cancelled by user.", "warn")
        elif pass1_only:
            self._status_var.set(
                "Analysis done — adjust keyframes, then click Deflicker"
            )
            self._log.append(
                "Adjust the curve (rolling window, keyframes), then click Deflicker.",
                "success",
            )
        else:
            if errors:
                self._status_var.set(f"Done with {len(errors)} error(s)")
                self._log.append(
                    f"Finished with {len(errors)} error(s). Output: {self._dest_folder}",
                    "warn",
                )
                for e in errors[:20]:
                    self._log.append(f"  {e}", "error")
            else:
                self._status_var.set("Done")
                self._log.append(
                    f"All {len(self._file_list)} frames exported → {self._dest_folder}",
                    "success",
                )

        self._log.append("═" * 60, "dim")

        if not pass1_only and not cancelled and self._dest_folder:
            try:
                with open(self._dest_folder / "deflicker.log", "a",
                          encoding="utf-8") as fh:
                    fh.write(self._log.get_all())
            except Exception:
                pass

    # ────────────────────────────────────────────────── exit ─────────────────

    def _on_closing(self) -> None:
        if self._processing:
            if not messagebox.askyesno("Quit", "Processing is running. Cancel and quit?"):
                return
            self._cancel_event.set()
        self._save_config()
        pass  # no process pool to shut down
        self._root.destroy()


# ──────────────────────────────────────────────────── perf dialog ────────────

class _PerfDialog:
    """
    Modal 'Performance options' window.
    Shows what was auto-detected, explains the limiting factors,
    and lets the user override the worker counts.
    """

    def __init__(
        self,
        parent: tk.Tk,
        profile,          # HWProfile | None
        settings: "SettingsPanel",
        on_apply,         # callable(p1: int, p2: int)
    ) -> None:
        self._settings  = settings
        self._on_apply  = on_apply
        self._profile   = profile

        win = tk.Toplevel(parent)
        win.title("Performance options")
        win.resizable(False, False)
        win.grab_set()
        win.focus_set()
        self._win = win

        win.columnconfigure(0, weight=1)
        row = 0

        # ── detected info ─────────────────────────────────────────────────────
        info_frame = ttk.LabelFrame(win, text="Auto-detected hardware", padding=10)
        info_frame.grid(row=row, column=0, sticky="ew", padx=16, pady=(12, 4))
        info_frame.columnconfigure(1, weight=1)
        row += 1

        if profile:
            rows = [
                ("RAM",        f"{profile.total_ram_gb:.0f} GB total"
                               f",  {profile.available_ram_gb:.0f} GB free"),
                ("CPU",        f"{profile.physical_cores} physical cores"
                               f"  ({profile.logical_cores} logical / HyperThreading)"),
                ("Frame size", profile.source),
                ("Per-worker", f"{profile.per_worker_mb:.0f} MB peak"
                               "  (2 × frame + process overhead)"),
            ]
            if profile.ram_is_limiting:
                limit = ("Limiting factor",
                         "RAM  — adding more workers would cause page-file pressure")
            else:
                limit = ("Limiting factor",
                         "CPU core count  — RAM has headroom for more workers")
            rows.append(limit)

            for i, (label, value) in enumerate(rows):
                ttk.Label(info_frame, text=label + ":",
                          foreground="#555", font=("Segoe UI", 9)
                          ).grid(row=i, column=0, sticky="w", pady=1, padx=(0, 12))
                ttk.Label(info_frame, text=value,
                          foreground="#111", font=("Segoe UI", 9)
                          ).grid(row=i, column=1, sticky="w", pady=1)
        else:
            ttk.Label(info_frame,
                      text="Auto-detect did not run — open a source folder first.",
                      foreground="#555").grid(row=0, column=0, columnspan=2)

        # ── explanation ───────────────────────────────────────────────────────
        expl = ttk.LabelFrame(win, text="How workers are used", padding=10)
        expl.grid(row=row, column=0, sticky="ew", padx=16, pady=4)
        row += 1

        ttk.Label(
            expl,
            text=(
                "Analysis (Pass 1) reads each frame once and measures brightness.\n"
                "It is CPU-bound: more workers = faster, up to your physical core count.\n"
                "\n"
                "Export (Pass 2) reads AND writes every frame on the same drive.\n"
                "The drive's read+write bandwidth is split, so fewer workers keep it\n"
                "running smoothly. More workers here often makes it slower, not faster."
            ),
            foreground="#333",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w")

        # ── override spinboxes ────────────────────────────────────────────────
        ovr = ttk.LabelFrame(win, text="Override worker counts", padding=10)
        ovr.grid(row=row, column=0, sticky="ew", padx=16, pady=4)
        ovr.columnconfigure(1, weight=1)
        row += 1

        p1_default = settings._workers_p1_var.get()
        p2_default = settings._workers_p2_var.get()

        ttk.Label(ovr, text="Analysis workers:",
                  foreground="#111").grid(row=0, column=0, sticky="w", pady=3)
        self._p1_var = tk.IntVar(value=p1_default)
        ttk.Spinbox(ovr, from_=1, to=256, width=6,
                    textvariable=self._p1_var
                    ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(ovr, text="Export workers:",
                  foreground="#111").grid(row=1, column=0, sticky="w", pady=3)
        self._p2_var = tk.IntVar(value=p2_default)
        ttk.Spinbox(ovr, from_=1, to=256, width=6,
                    textvariable=self._p2_var
                    ).grid(row=1, column=1, sticky="w", padx=8)

        ttk.Button(ovr, text="⟳  Re-run auto-detect",
                   command=self._redetect).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        # ── buttons ───────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(win, padding=(16, 6, 16, 12))
        btn_frame.grid(row=row, column=0, sticky="e")

        ttk.Button(btn_frame, text="OK",
                   command=self._ok, width=10).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Cancel",
                   command=win.destroy, width=10).pack(side="left")

        # Centre on parent
        win.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width()  - win.winfo_width())  // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{px}+{py}")

    def _redetect(self) -> None:
        try:
            from core.hw_detect import detect
            first_file = None
            # Try to get first file from settings panel's parent window context
            # by re-running detect without a file (uses last known profile's frame)
            if self._profile and self._profile.frame_w:
                # Build a dummy path — detect() accepts None safely
                pass
            profile = detect(first_file=None)
            self._profile = profile
            self._p1_var.set(profile.pass1_workers)
            self._p2_var.set(profile.pass2_workers)
        except Exception as exc:
            messagebox.showerror("Auto-detect failed", str(exc), parent=self._win)

    def _ok(self) -> None:
        try:
            p1 = max(1, int(self._p1_var.get()))
            p2 = max(1, int(self._p2_var.get()))
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid value",
                                 "Worker counts must be positive integers.",
                                 parent=self._win)
            return
        self._win.destroy()
        self._on_apply(p1, p2)


# ──────────────────────────────────────────────────── helpers ─────────────────

def _fmt_seconds(s: float) -> str:
    s = int(s)
    if s < 60:   return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:   return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _ask_mixed_format(parent, group_names: list[str], detail: str) -> str | None:
    win = tk.Toplevel(parent)
    win.title("Mixed file formats detected")
    win.resizable(False, False)
    win.grab_set()
    ttk.Label(win,
              text="The source folder contains multiple image formats:\n" + detail,
              padding=12, wraplength=380).pack()
    ttk.Label(win, text="Which format should be processed?",
              padding=(12, 0)).pack()
    choice_var = tk.StringVar(value=group_names[0])
    for g in group_names:
        ttk.Radiobutton(win, text=g.upper(),
                        variable=choice_var, value=g).pack(anchor="w", padx=24)
    result: list[str | None] = [None]
    def ok():
        result[0] = choice_var.get()
        win.destroy()
    btn = ttk.Frame(win, padding=12)
    btn.pack()
    ttk.Button(btn, text="Process selected", command=ok).pack(side="left", padx=6)
    ttk.Button(btn, text="Abort", command=win.destroy).pack(side="left", padx=6)
    parent.wait_window(win)
    return result[0]


def _ask_dest_conflict(parent, count: int) -> str | None:
    win = tk.Toplevel(parent)
    win.title("Output folder not empty")
    win.resizable(False, False)
    win.grab_set()
    ttk.Label(win,
              text=f"The output folder already contains {count} file(s).\n"
                   "What should be done with existing files?",
              padding=12, wraplength=360).pack()
    action_var = tk.StringVar(value="skip")
    ttk.Radiobutton(win, text="Skip existing files",
                    variable=action_var, value="skip").pack(anchor="w", padx=24)
    ttk.Radiobutton(win, text="Overwrite existing files",
                    variable=action_var, value="overwrite").pack(anchor="w", padx=24)
    result: list[str | None] = [None]
    def ok():
        result[0] = action_var.get()
        win.destroy()
    btn = ttk.Frame(win, padding=12)
    btn.pack()
    ttk.Button(btn, text="Continue", command=ok).pack(side="left", padx=6)
    ttk.Button(btn, text="Abort", command=win.destroy).pack(side="left", padx=6)
    parent.wait_window(win)
    return result[0]
