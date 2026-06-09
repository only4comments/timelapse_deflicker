"""
Settings persistence — saves/restores user preferences to config.json
in the same directory as main.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent / "config.json"

_DEFAULTS: dict[str, Any] = {
    "source_folder": "",
    "dest_folder": "",
    "window_size": 10,
    "metric": "median",          # "mean" | "median" | "percentile"
    "percentile_value": 95,
    "correction_mode": "scaling", # "scaling" | "histogram"
    "output_format": "jpeg",      # "jpeg" | "tiff8" | "tiff16"
    "jpeg_quality": 85,
    "jpeg_resize_1080": False,
    "worker_count": os.cpu_count() or 4,
}


class Config:
    def __init__(self) -> None:
        self._data: dict[str, Any] = dict(_DEFAULTS)
        self.load()

    # ------------------------------------------------------------------ load/save
    def load(self) -> None:
        if _CONFIG_PATH.exists():
            try:
                with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                for key in _DEFAULTS:
                    if key in saved:
                        self._data[key] = saved[key]
            except Exception:
                pass  # corrupt config — use defaults silently

    def save(self) -> None:
        try:
            with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
        except Exception:
            pass  # non-fatal

    # ------------------------------------------------------------------ accessors
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
