"""
Source-folder scanning: file discovery, format detection, mixed-format checks.
"""
from __future__ import annotations

from pathlib import Path

from core.image_io import SUPPORTED_EXTENSIONS

# Map extensions to a canonical group name used for mixed-format detection
_EXT_GROUP: dict[str, str] = {
    ".tif": "tiff",
    ".tiff": "tiff",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
}


class ScanResult:
    """Result of a folder scan."""

    def __init__(
        self,
        files: list[Path],
        groups: dict[str, list[Path]],
    ) -> None:
        self.files = files          # all matching files, sorted alphabetically
        self.groups = groups        # {"tiff": [...], "jpeg": [...]}

    @property
    def is_mixed(self) -> bool:
        return len(self.groups) > 1

    @property
    def group_names(self) -> list[str]:
        return sorted(self.groups.keys())

    def filter_to_group(self, group: str) -> "ScanResult":
        """Return a new ScanResult containing only files of the given group."""
        kept = self.groups.get(group, [])
        kept_set = set(kept)
        files = [f for f in self.files if f in kept_set]
        return ScanResult(files, {group: kept})


def scan_folder(folder: str | Path) -> ScanResult:
    """
    Discover all supported image files in *folder* (non-recursive).
    Files are sorted alphabetically by filename (case-insensitive on Windows).
    """
    folder = Path(folder)
    files: list[Path] = []
    groups: dict[str, list[Path]] = {}

    for p in folder.iterdir():
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        group = _EXT_GROUP.get(ext, ext)
        groups.setdefault(group, []).append(p)
        files.append(p)

    files.sort(key=lambda p: p.name.lower())
    for g in groups:
        groups[g].sort(key=lambda p: p.name.lower())

    return ScanResult(files, groups)
