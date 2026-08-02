"""
Manages a temporary directory for preview thumbnail cache.
Cleaned up on application exit.
"""
from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path


class TempManager:
    def __init__(self) -> None:
        self._dir: Path | None = None
        atexit.register(self.cleanup)

    @property
    def dir(self) -> Path:
        if self._dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="deflicker_preview_"))
        return self._dir

    def preview_path(self, stem: str) -> Path:
        """Return the cache path for a given source filename stem."""
        return self.dir / f"{stem}.jpg"

    def cleanup(self) -> None:
        if self._dir and self._dir.exists():
            try:
                shutil.rmtree(self._dir, ignore_errors=True)
            except Exception:
                pass
            self._dir = None
