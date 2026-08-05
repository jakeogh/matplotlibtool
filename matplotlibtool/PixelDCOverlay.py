#!/usr/bin/env python3
# tab-width:4

"""
PixelDCOverlay - the value the averager would report, drawn across its dwell.

One horizontal segment per pixel dwell, spanning the dwell and sitting at the
mean of the averaging window. It answers the question the settle analysis
raises but cannot show: whether the window that was measured actually lands on
the level a human would call the pixel value.

The levels are computed once when the overlay is enabled and re-culled to the
view on every render, so a capture with more dwells than the screen has pixels
still costs only the segments that are visible.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection

from .PixelAnalysis import PixelAnalysisError
from .PixelAnalysis import analyse_pixels
from .PixelAnalysis import measure_dwells

DC_COLOR = "#ffffff"
DC_COLOR_LIGHT = "#000000"
DC_LINE_WIDTH = 1.4
MAX_VISIBLE_SEGMENTS = 20000


class PixelDCOverlay:
    """Per-dwell DC levels, culled to the current view."""

    def __init__(self, ax):
        self.ax = ax
        self._collection: LineCollection | None = None
        self._x0: np.ndarray | None = None
        self._x1: np.ndarray | None = None
        self._dc: np.ndarray | None = None
        self.start = 0
        self.length: int | None = None  # None averages the rest of each dwell
        self.dwell_length = 0
        self.measured = True
        self.settled = True

    @property
    def active(self) -> bool:
        return self._dc is not None

    def clear(self) -> None:
        if self._collection is not None:
            self._collection.remove()
            self._collection = None
        self._x0 = self._x1 = self._dc = None

    def compute(
        self,
        data: np.ndarray,
        *,
        value_field: str,
        pixel_field: str = "pixel",
        idle_pixel: int = 0,
        start: int | None = None,
        length: int | None = None,
    ) -> None:
        """
        Average each dwell the way the averager would, and draw what it writes.

        The window semantics mirror the averaging pipeline: start None measures
        the settled window from the capture, with length following it unless
        given; start given with length None averages the rest of each dwell;
        start and length given average exactly that window. The settle fit runs
        only when the window is measured, so an operator-set window produces
        levels on any capture, settled or not — the same as the averager.
        """
        geometry, starts, ends, usable = measure_dwells(
            data[pixel_field], idle_pixel=idle_pixel
        )
        self.dwell_length = geometry.modal_length
        self.measured = start is None

        if start is None:
            report = analyse_pixels(
                data, value_field=value_field, pixel_field=pixel_field
            )
            self.settled = report.settled
            start = report.recommended_start
            if length is None:
                length = report.recommended_length
        else:
            self.settled = True
            if start < 0:
                raise PixelAnalysisError(f"pixel dc: window start {start} is negative")
            if length is not None and length < 1:
                raise PixelAnalysisError(f"pixel dc: window length {length} is empty")

        required = start + (length if length is not None else 1)
        if required > self.dwell_length:
            raise PixelAnalysisError(
                f"pixel dc: window start {start} length "
                f"{length if length is not None else 'rest'} needs {required} "
                f"records of the {self.dwell_length} record dwell"
            )

        self.start = start
        self.length = length

        s = starts[usable]
        e = ends[usable]
        value = data[value_field].astype(np.float64)
        csum = np.concatenate(([0.0], np.cumsum(value)))
        stop = e if length is None else s + start + length
        self._dc = (csum[stop] - csum[s + start]) / (stop - s - start)
        self._x0 = s.astype(np.float64)
        self._x1 = (e - 1).astype(np.float64)

    def update(self, xlim: tuple[float, float], plot) -> None:
        """Redraw the segments that fall inside xlim, in the plot's display space."""
        if self._dc is None:
            return

        x0 = self._x0 + plot.offset_x
        x1 = self._x1 + plot.offset_x
        visible = np.flatnonzero((x1 >= xlim[0]) & (x0 <= xlim[1]))
        if visible.size > MAX_VISIBLE_SEGMENTS:
            visible = visible[:: -(-visible.size // MAX_VISIBLE_SEGMENTS)]

        y = self._dc * plot.y_scale + plot.offset_y
        if plot.settle_ref is not None:
            residual = np.abs(y - plot.settle_ref)
            positive = residual[residual > 0.0]
            floor = float(positive.min()) * 0.5 if positive.size else 1.0
            y = np.log10(np.maximum(residual, floor))

        segments = np.stack(
            [
                np.column_stack([x0[visible], y[visible]]),
                np.column_stack([x1[visible], y[visible]]),
            ],
            axis=1,
        )

        color = DC_COLOR if _is_dark(self.ax) else DC_COLOR_LIGHT
        if self._collection is None:
            self._collection = LineCollection(
                segments, colors=color, linewidths=DC_LINE_WIDTH, zorder=900
            )
            self.ax.add_collection(self._collection)
        else:
            self._collection.set_segments(segments)
            self._collection.set_color(color)
        self._collection.set_visible(True)


def _is_dark(ax) -> bool:
    from matplotlib.colors import to_rgba

    r, g, b, _ = to_rgba(ax.get_facecolor())
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
