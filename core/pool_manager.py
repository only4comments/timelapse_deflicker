"""
Persistent ProcessPoolExecutor — created once per session, reused across
Pass 1 and Pass 2 so that the 32-process spawn cost is paid only once.

On Windows (spawn context), importing numpy alone takes ~2–3 s per process.
Spawning 32 workers twice (old design) wasted ~5 min per run before a single
pixel was processed.  Keeping the pool warm eliminates that overhead entirely.

The pool is recreated automatically if worker_count changes between runs.
"""
from __future__ import annotations

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


class WorkerPool:
    """Singleton-style wrapper.  One instance lives in MainWindow."""

    def __init__(self) -> None:
        self._executor: ProcessPoolExecutor | None = None
        self._worker_count: int = 0

    def get(self, worker_count: int) -> ProcessPoolExecutor:
        """
        Return the warm executor.  Transparently recreates it only when
        worker_count has changed since the last call.
        """
        if self._executor is None or self._worker_count != worker_count:
            self._shutdown_quietly()
            from core.worker_init import set_low_priority
            self._executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=set_low_priority,
            )
            self._worker_count = worker_count
        return self._executor

    def shutdown(self) -> None:
        """Call this when the application exits."""
        self._shutdown_quietly()

    def _shutdown_quietly(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._executor = None
            self._worker_count = 0
