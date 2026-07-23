#!/usr/bin/env python3

"""
Bounded back/forward stack of committed ViewBounds.

Rapid successive commits (wheel-zoom bursts, batched plot adds) coalesce
into a single entry so navigation steps between distinct views rather
than individual scroll ticks.
"""

from __future__ import annotations

from time import monotonic

from .ViewManager import ViewBounds


class ViewHistory:
    def __init__(
        self,
        limit: int = 200,
        coalesce_seconds: float = 0.5,
    ):
        self.limit = limit
        self.coalesce_seconds = coalesce_seconds
        self._states: list[ViewBounds] = []
        self._cursor = -1
        self._last_record_time = 0.0
        self._last_was_append = False

    @property
    def can_go_back(self) -> bool:
        return self._cursor > 0

    @property
    def can_go_forward(self) -> bool:
        return self._cursor < len(self._states) - 1

    def record(self, bounds: ViewBounds) -> None:
        now = monotonic()
        if self._states and self._states[self._cursor] == bounds:
            self._last_record_time = now
            return

        del self._states[self._cursor + 1 :]

        coalesce = (
            self._last_was_append
            and now - self._last_record_time < self.coalesce_seconds
            and len(self._states) > 1
        )
        if coalesce:
            self._states[-1] = bounds
        else:
            self._states.append(bounds)
            if len(self._states) > self.limit:
                del self._states[0]

        self._cursor = len(self._states) - 1
        self._last_record_time = now
        self._last_was_append = True

    def back(self) -> ViewBounds | None:
        if not self.can_go_back:
            return None
        self._cursor -= 1
        self._last_was_append = False
        return self._states[self._cursor]

    def forward(self) -> ViewBounds | None:
        if not self.can_go_forward:
            return None
        self._cursor += 1
        self._last_was_append = False
        return self._states[self._cursor]
