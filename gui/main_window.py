"""
Main application window.

Layout
------
Row 0 : source / destination folder bar
Row 1 : settings panel (left)  +  ttk.Notebook (right)
            Tab 0 – Log   (scrolling dark console)
            Tab 1 – Graph (Pass 1 luminance analysis, live + complete)
Row 2 : progress panel (run-mode selector, progress bar, Run / Cancel)

Run modes
---------
  "both"  – Pass 1 → rolling avg → Pass 2
  "pass1" – Pass 1 → rolling avg → update graph  (no file output)
  "pass2" – use stored Pass 1 results → Pass 2

Recalculate
-----------
After Pass 1 the user may change the rolling window in Settings and press
"↻ Recalculate" in the graph panel.  This recomputes rolling_average()
from the already-stored luminance values and refreshes the graph — no
Pass 1 re-run, no subprocess activity at all.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from config import Config
from core.scanner import scan_folder
from core.pass1 import run_pass1
from core.pass2 import run_pass2
from core.rolling_average import rolling_average, rolling_average_histograms
from core.pool_manager import WorkerPool
from gui.settings_panel import SettingsPanel
from gui.graph_panel import GraphPanel
from gui.progress_panel import ProgressPanel
from gui.log_panel import LogPanel


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self._root   = root
        self._config = Config()

        self._file_list:     list[Path] = []
        self._source_folder: Path | None = None
        self._dest_folder:   Path | None = None

        # Pass 1 results kept in memory between runs
        self._luminance_values: list[float | None] = []
        self._histograms:       list | None = None
        self._rolling_lum:      np.ndarray | None = None
        self._rolling_hists:    list | None = None
        self._pass1_settings:   dict | None = None

        self._processing    = False
        self._cancel_event  = threading.Event()
        self._progress_queue: queue.Queue = queue.Queue()
        self._pool          = WorkerPool()

        self._build_ui()
        self._restore_config()
        self._poll_queue()

    # ─────────────────────────────────────────────────── UI construction ──────

    def _build_ui(self) -> None:
        root = self._root
        root.title("Timelapse Deflicker")
        root.minsize(1100, 720)
        root.protocol("WM_DELETE_WINDOW", self._on_closing)

        try:
            ttk.Style().theme_use("vista")
        except Exception:
            try:
                ttk.Style().theme_use("clam")
            except Exception:
                pass

        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # ── Row 0: source / dest bar ─────────────────────────────────────────
        top = ttk.Frame(root, padding=(8, 6, 8, 0))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(4, weight=1)

        ttk.Button(top, text="📂  Open Source Folder",
                   command=self._open_source).grid(row=0, column=0, padx=(0, 6))
        self._src_label = ttk.Label(
            top, text="No folder selected", foreground="#666", anchor="w"
        )
        self._src_label.grid(row=0, column=1, sticky="ew", padx=(0, 16))
        self._src_count_label = ttk.Label(top, text="", foreground="#446")
        self._src_count_label.grid(row=0, column=2, padx=(0, 16))

        ttk.Button(top, text="📁  Select Output Folder",
                   command=self._open_dest).grid(row=0, column=3, padx=(0, 6))
        self._dst_label = ttk.Label(
            top, text="No folder selected", foreground="#666", anchor="w"
        )
        self._dst_label.grid(row=0, column=4, sticky="ew")

        ttk.Separator(root, orient="horizontal").grid(
            row=0, column=0, sticky="ew", pady=(32, 0)
        )

        # ── Row 1: settings + notebook ───────────────────────────────────────
        content = ttk.Frame(root)
        content.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        self._settings = SettingsPanel(content, width=270)
        self._settings.grid(row=0, column=0, sticky="ns", padx=(0, 6))

        self._notebook = ttk.Notebook(content)
        self._notebook.grid(row=0, column=1, sticky="nsew")

        # Tab 0 — Log
        log_tab = ttk.Frame(self._notebook)
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(0, weight=1)
        self._log = LogPanel(log_tab, padding=4)
        self._log.grid(row=0, column=0, sticky="nsew")
        self._notebook.add(log_tab, text="  Log  ")

        # Tab 1 — Graph
        graph_tab = ttk.Frame(self._notebook)
        graph_tab.columnconfigure(0, weight=1)
        graph_tab.rowconfigure(0, weight=1)
        self._graph = GraphPanel(graph_tab, on_recalculate=self._on_recalculate)
        self._graph.grid(row=0, column=0, sticky="nsew")
        self._notebook.add(graph_tab, text="  Luminance Graph  ")

        # ── Row 2: progress + run controls ───────────────────────────────────
        self._progress = ProgressPanel(
            root,
            on_run=self._on_run,
            on_cancel=self._on_cancel,
            padding=(8, 4),
        )
        self._progress.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))

    # ──────────────────────────────────────────────────── config ───────────────

    def _restore_config(self) -> None:
        self._settings.apply_settings(dict(self._config._data))
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
        self._config.save()

    # ──────────────────────────────────────────────── folder actions ───────────

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
        self._src_count_label.configure(
            text=f"{len(self._file_list)} files", foreground="#446"
        )
        self._invalidate_pass1()
        self._log.append(
            f"Source: {folder}  ({len(self._file_list)} files)", "info"
        )

    def _open_dest(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self._dest_folder = Path(folder)
            self._dst_label.configure(text=folder, foreground="#222")

    def _invalidate_pass1(self) -> None:
        self._luminance_values = []
        self._rolling_lum      = None
        self._rolling_hists    = None
        self._histograms       = None
        self._pass1_settings   = None
        self._progress.set_pass2_enabled(False)
        self._graph.clear()

    # ──────────────────────────────────────────── recalculate ─────────────────

    def _on_recalculate(self) -> None:
        """
        Recompute rolling average from stored luminance values using the
        current window size from Settings — no Pass 1 re-run.
        """
        if not self._luminance_values:
            return

        settings = self._settings.get_settings()
        window   = settings["window_size"]

        valid_lum    = [v if v is not None else 0.0 for v in self._luminance_values]
        rolling_lum  = rolling_average(valid_lum, window)
        self._rolling_lum = rolling_lum

        rolling_hists = None
        if (settings["correction_mode"] == "histogram"
                and self._histograms
                and any(h is not None for h in self._histograms)):
            valid_hists = [
                h if h is not None else self._histograms[0]
                for h in self._histograms
            ]
            rolling_hists = rolling_average_histograms(valid_hists, window)
        self._rolling_hists = rolling_hists

        self._graph.refresh_rolling(rolling_lum, window)
        self._log.append(
            f"Rolling average recalculated  —  window={window}", "info"
        )

    # ──────────────────────────────────────────────── run / cancel ────────────

    def _on_run(self) -> None:
        if not self._file_list:
            messagebox.showwarning("No source", "Please open a source folder first.")
            return

        settings  = self._settings.get_settings()
        run_mode  = self._progress.get_run_mode()

        if run_mode == "pass2":
            if self._rolling_lum is None:
                messagebox.showwarning(
                    "No Pass 1 data",
                    "Pass 1 has not been run yet.\n"
                    "Run Pass 1 first before using 'Pass 2 only'.",
                )
                return

        dest_action = "overwrite"
        if run_mode in ("both", "pass2"):
            if not self._dest_folder:
                messagebox.showwarning(
                    "No output folder",
                    "Please select an output folder first.",
                )
                return

            from core.image_io import output_extension
            ext = output_extension(settings["output_format"])
            existing = (
                [f for f in self._dest_folder.iterdir()
                 if f.is_file() and f.suffix.lower() == ext]
                if self._dest_folder.exists() else []
            )
            if existing:
                dest_action = _ask_dest_conflict(self._root, len(existing))
                if dest_action is None:
                    return

            self._dest_folder.mkdir(parents=True, exist_ok=True)

        self._cancel_event.clear()
        self._processing = True
        self._progress.set_running(True)
        self._progress.reset()
        self._log.clear()
        self._log.append("═" * 60, "dim")

        mode_label = {
            "both":  "Pass 1 + 2",
            "pass1": "Pass 1 only",
            "pass2": "Pass 2 only",
        }[run_mode]
        self._log.append(
            f"Starting  [{mode_label}]  —  {len(self._file_list)} frames  |  "
            f"window={settings['window_size']}  |  metric={settings['metric']}  |  "
            f"mode={settings['correction_mode']}  |  output={settings['output_format']}"
            + ("  |  resize=1080p" if settings.get("jpeg_resize_1080") else "")
            + f"  |  workers={settings['worker_count']}",
            "info",
        )

        # Switch to graph tab immediately when Pass 1 will run, so the user
        # can watch the curve grow.
        if run_mode in ("both", "pass1"):
            self._graph.start_live(
                n_frames    = len(self._file_list),
                window_size = settings["window_size"],
                metric      = settings["metric"],
                file_list   = self._file_list,
            )
            self._notebook.select(1)

        # Get (or lazily create) the warm persistent pool before
        # spawning the background thread.  Pool creation itself is fast
        # here; the actual process spawning happens asynchronously as
        # the first tasks are submitted inside the thread.
        executor = self._pool.get(settings["worker_count"])

        threading.Thread(
            target=self._processing_thread,
            args=(settings, run_mode, dest_action, executor),
            daemon=True,
        ).start()

    def _on_cancel(self) -> None:
        if self._processing:
            self._cancel_event.set()
            self._log.append("Cancel requested — finishing current batch…", "warn")

    # ──────────────────────────────────────────── processing thread ───────────

    def _processing_thread(
        self, settings: dict, run_mode: str, dest_action: str, executor
    ) -> None:
        q     = self._progress_queue
        files = self._file_list

        try:
            # ── Pass 1 ────────────────────────────────────────────────────────
            if run_mode in ("both", "pass1"):
                q.put(("phase", "Pass 1 – Analysing luminance…"))
                need_hist = settings["correction_mode"] == "histogram"

                lum_values, histograms = run_pass1(
                    file_list        = files,
                    metric           = settings["metric"],
                    percentile_value = settings["percentile_value"],
                    need_histogram   = need_hist,
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

                # Rolling average (fast, sequential)
                q.put(("phase", "Computing rolling averages…"))
                valid_lum   = [v if v is not None else 0.0 for v in lum_values]
                rolling_lum = rolling_average(valid_lum, settings["window_size"])

                rolling_hists = None
                if need_hist and any(h is not None for h in histograms):
                    valid_hists = [
                        h if h is not None else histograms[0] for h in histograms
                    ]
                    rolling_hists = rolling_average_histograms(
                        valid_hists, settings["window_size"]
                    )

                q.put(("pass1_done", lum_values, histograms,
                       rolling_lum, rolling_hists, settings))

                if run_mode == "pass1":
                    q.put(("done_pass1_only",))
                    return

            else:
                # Pass 2 only — reuse stored values
                lum_values    = self._luminance_values
                rolling_lum   = self._rolling_lum
                rolling_hists = self._rolling_hists

            # ── Pass 2 ────────────────────────────────────────────────────────
            q.put(("phase", "Pass 2 – Applying corrections…"))
            errors = run_pass2(
                file_list        = files,
                dest_folder      = self._dest_folder,
                rolling_lum      = rolling_lum,
                measured_lum     = lum_values,
                rolling_hists    = rolling_hists,
                correction_mode  = settings["correction_mode"],
                output_format    = settings["output_format"],
                jpeg_quality     = settings["jpeg_quality"],
                jpeg_resize_1080 = settings.get("jpeg_resize_1080", False),
                worker_count     = settings["worker_count"],
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
            self._progress.set_phase(msg[1])

        elif kind == "pass1_progress":
            _, done, total, result = msg
            self._progress.update_progress(done, total, "Pass 1 – Analysing luminance…")
            if result["error"]:
                self._log.append(
                    f"SKIP  {Path(result['filepath']).name}: {result['error']}", "error"
                )
            else:
                self._graph.add_point(result["index"], result["luminance"])
                self._log.append(
                    f"P1  {Path(result['filepath']).name}  "
                    f"Y={result['luminance']:.4f}", "dim"
                )

        elif kind == "pass1_done":
            _, lum_values, histograms, rolling_lum, rolling_hists, p1_settings = msg
            # Store in memory for Recalculate and Pass-2-only
            self._luminance_values = lum_values
            self._histograms       = histograms
            self._rolling_lum      = rolling_lum
            self._rolling_hists    = rolling_hists
            self._pass1_settings   = p1_settings
            self._progress.set_pass2_enabled(True)

            n_ok = sum(1 for v in lum_values if v is not None)
            self._log.append(
                f"Pass 1 complete  —  {n_ok}/{len(lum_values)} frames analysed",
                "success",
            )
            # Finalise the live graph (adds rolling avg + factor lines)
            self._graph.finish_live(lum_values, rolling_lum)

        elif kind == "pass2_progress":
            _, done, total, result = msg
            self._progress.update_progress(done, total, "Pass 2 – Applying corrections…")
            if result["error"]:
                self._log.append(
                    f"FAIL  {Path(result['src']).name}: {result['error']}", "error"
                )
            else:
                self._log.append(
                    f"P2  {Path(result['src']).name}  "
                    f"×{result['correction_factor']:.4f}", "dim"
                )

        elif kind == "done":
            self._finish_processing(cancelled=False, errors=msg[1])

        elif kind == "done_pass1_only":
            self._finish_processing(cancelled=False, errors=[], pass1_only=True)

        elif kind == "cancelled":
            self._finish_processing(cancelled=True, errors=[])

        elif kind == "error":
            self._log.append(f"FATAL ERROR: {msg[1]}", "error")
            self._finish_processing(cancelled=False, errors=[msg[1]])

    # ─────────────────────────────────────────── finish ───────────────────────

    def _finish_processing(
        self,
        cancelled: bool,
        errors: list[str],
        pass1_only: bool = False,
    ) -> None:
        self._processing = False
        self._progress.set_running(False)
        self._progress.set_pass2_enabled(self._rolling_lum is not None)

        if cancelled:
            self._progress.set_phase("Cancelled")
            self._log.append("Processing was cancelled by user.", "warn")
        elif pass1_only:
            self._progress.set_phase("Pass 1 done — adjust window and Recalculate, or run Pass 2")
            self._log.append(
                "Pass 1 complete. Adjust rolling window and click "
                "'↻ Recalculate' to update the graph without re-running "
                "the analysis.  Then run 'Pass 2 only' to export.",
                "success",
            )
        else:
            if errors:
                self._log.append(
                    f"Finished with {len(errors)} error(s). "
                    f"Output: {self._dest_folder}", "warn"
                )
                for e in errors[:20]:
                    self._log.append(f"  {e}", "error")
            else:
                self._progress.set_phase("Done")
                self._log.append(
                    f"All {len(self._file_list)} frames processed. "
                    f"Output: {self._dest_folder}", "success"
                )

        self._log.append("═" * 60, "dim")
        # Switch to log tab to show completion summary
        self._notebook.select(0)

        if not pass1_only and self._dest_folder:
            try:
                with open(self._dest_folder / "deflicker.log", "a", encoding="utf-8") as fh:
                    fh.write(self._log.get_all())
            except Exception:
                pass

    # ──────────────────────────────────────────────── exit ────────────────────

    def _on_closing(self) -> None:
        if self._processing:
            if not messagebox.askyesno("Quit", "Processing is running. Cancel and quit?"):
                return
            self._cancel_event.set()
        self._save_config()
        self._pool.shutdown()
        self._root.destroy()


# ─────────────────────────────────────────────────── dialogs ──────────────────

def _ask_mixed_format(parent, group_names: list[str], detail: str) -> str | None:
    win = tk.Toplevel(parent)
    win.title("Mixed file formats detected")
    win.resizable(False, False)
    win.grab_set()
    ttk.Label(
        win,
        text="The source folder contains multiple image formats:\n" + detail,
        padding=12, wraplength=380,
    ).pack()
    ttk.Label(win, text="Which format should be processed?", padding=(12, 0)).pack()
    choice_var = tk.StringVar(value=group_names[0])
    for g in group_names:
        ttk.Radiobutton(win, text=g.upper(), variable=choice_var, value=g).pack(
            anchor="w", padx=24
        )
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
    ttk.Label(
        win,
        text=f"The output folder already contains {count} file(s).\n"
             "What should be done with existing files?",
        padding=12, wraplength=360,
    ).pack()
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
