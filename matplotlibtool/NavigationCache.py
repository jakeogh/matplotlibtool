#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


class NavigationCache:
    """Decoded captures for file navigation, LRU-bounded by available memory.

    Entries are keyed by resolved path and validated against the file's
    mtime and size at parse time, so a rewritten file misses cleanly. The
    cache holds itself under BUDGET_FRACTION of the machine's currently
    available memory, measured at every insertion, evicting least recently
    used entries first; an array that alone exceeds the budget is not
    stored. Every call runs on the GUI thread, so no locking.
    """

    BUDGET_FRACTION = 0.25

    def __init__(self) -> None:
        self._entries: OrderedDict[Path, tuple[tuple[int, int], np.ndarray]] = (
            OrderedDict()
        )

    @staticmethod
    def stat_key(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    @staticmethod
    def _available_bytes() -> int:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("MemAvailable not present in /proc/meminfo")

    @property
    def bytes_cached(self) -> int:
        return sum(data.nbytes for _, data in self._entries.values())

    def get(self, path: Path) -> np.ndarray | None:
        """The cached array for the file as it is on disk now, else None."""
        key = path.resolve()
        entry = self._entries.get(key)
        if entry is None:
            return None
        stat, data = entry
        if not path.exists() or self.stat_key(path) != stat:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return data

    def put(self, path: Path, stat: tuple[int, int], data: np.ndarray) -> None:
        """Insert an array parsed from the file while it had `stat`.

        The stat captured before parsing keeps a mid-parse rewrite from
        being served as current: the next get sees the mismatch and misses.
        """
        budget = int(self._available_bytes() * self.BUDGET_FRACTION)
        if data.nbytes > budget:
            print(
                f"[nav] cache skip {path.name}: {data.nbytes / 1e6:.1f} MB "
                f"exceeds the {budget / 1e6:.0f} MB budget",
                file=sys.stderr,
            )
            return
        key = path.resolve()
        self._entries[key] = (stat, data)
        self._entries.move_to_end(key)
        while self.bytes_cached > budget:
            evicted, (_, gone) = self._entries.popitem(last=False)
            print(
                f"[nav] cache evict {evicted.name} ({gone.nbytes / 1e6:.1f} MB)",
                file=sys.stderr,
            )
        print(
            f"[nav] cache store {path.name} ({data.nbytes / 1e6:.1f} MB, "
            f"{self.bytes_cached / 1e6:.1f} MB of {budget / 1e6:.0f} MB budget)",
            file=sys.stderr,
        )
