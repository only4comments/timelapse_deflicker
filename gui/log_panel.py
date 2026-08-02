"""
Scrollable, read-only log console shown at the bottom of the main window.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from datetime import datetime


class LogPanel(ttk.LabelFrame):
    def __init__(self, parent, **kw):
        super().__init__(parent, text="Log", **kw)
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._text = tk.Text(
            self,
            state="disabled",
            wrap="word",
            height=8,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            relief="flat",
            borderwidth=0,
        )
        self._text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=scrollbar.set)

        # Colour tags
        self._text.tag_configure("info",    foreground="#d4d4d4")
        self._text.tag_configure("warn",    foreground="#dcdcaa")
        self._text.tag_configure("error",   foreground="#f48771")
        self._text.tag_configure("success", foreground="#4ec9b0")
        self._text.tag_configure("dim",     foreground="#808080")

    # ------------------------------------------------------------------ public

    def append(self, message: str, level: str = "info") -> None:
        """
        Append a line to the log.  level: "info" | "warn" | "error" | "success" | "dim"
        """
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {message}\n"
        self._text.configure(state="normal")
        self._text.insert("end", line, level)
        self._text.configure(state="disabled")
        self._text.see("end")

    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

    def get_all(self) -> str:
        return self._text.get("1.0", "end")
