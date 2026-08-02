#!/usr/bin/env python3

"""
Bounded back/forward stack of committed views.

Continuous gestures (wheel-zoom bursts) opt into coalescing so navigation
steps between distinct views rather than individual scroll ticks. Discrete
commits - a zoom box, a pan release, a typed range - always append, so two
deliberate views in quick succession are never merged.

Each entry carries the display space its y limits were measured in. A
y range in log-residual decades is meaningless once the plot returns to
linear codes, so navigation across a space boundary keeps the recorded x
window and refits y rather than restoring a range from another space.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from .ViewManager import ViewBounds


@dataclass(frozen=True)
class HistoryEntry:
    bounds: ViewBounds
    space: str


class ViewHistory:
    def __init__(
        self,
        limit: int = 200,
        coalesce_seconds: float = 0.5,
    ):
        self.limit = limit
        self.coalesce_seconds = coalesce_seconds
        self._states: list[HistoryEntry] = []
        self._cursor = -1
        self._last_record_time = 0.0
        self._last_was_coalescable = False

    @property
    def can_go_back(self) -> bool:
        return self._cursor > 0

    @property
    def can_go_forward(self) -> bool:
        return self._cursor < len(self._states) - 1

    def record(self, bounds: ViewBounds, space: str, coalesce: bool = False) -> None:
        now = monotonic()
        entry = HistoryEntry(bounds=bounds, space=space)
        # a no-op commit must not extend the coalesce window: doing so lets the
        # next genuine view overwrite the previous entry instead of appending
        if self._states and self._states[self._cursor] == entry:
            return

        del self._states[self._cursor + 1 :]

        merge = (
            coalesce
            and self._last_was_coalescable
            and now - self._last_record_time < self.coalesce_seconds
            and len(self._states) > 1
        )
        if merge:
            self._states[-1] = entry
        else:
            self._states.append(entry)
            if len(self._states) > self.limit:
                del self._states[0]

        self._cursor = len(self._states) - 1
        self._last_record_time = now
        self._last_was_coalescable = coalesce

    def back(self) -> HistoryEntry | None:
        if not self.can_go_back:
            return None
        self._cursor -= 1
        self._last_was_coalescable = False
        return self._states[self._cursor]

    def forward(self) -> HistoryEntry | None:
        if not self.can_go_forward:
            return None
        self._cursor += 1
        self._last_was_coalescable = False
        return self._states[self._cursor]
