#!/usr/bin/env python3
# tab-width:4

"""
PixelDCOverlay - the value the averager would report, drawn across its window.

One horizontal segment per pixel dwell, spanning exactly the averaged records
and sitting at their mean. The segment's left edge is on the first sample of
the averaging window and its right edge on the last, so which samples produced
the value is readable straight off the plot. It answers the question the
settle analysis raises but cannot show: whether the window that was measured
actually lands on the level a human would call the pixel value.

The averaged samples themselves are drawn over the plot's own in the segment's
colour, once the view is close enough that the plot draws its samples as
points, so zoomed in it is plain which samples the value came from.

Each dwell's plateau is measured beside its level: how level the window's
samples lie, 1 when they sit at one height and 0 when they climb the dwell's
whole swing, the averager's own measure (iioparser.plateau_of).

The levels are computed once when the overlay is enabled and re-culled to the
view on every render, so a capture with more dwells than the screen has pixels
still costs only the segments that are visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from iioparser import plateau_of
from matplotlib.collections import LineCollection

from .PixelAnalysis import PixelAnalysisError
from .PixelAnalysis import analyse_pixels
from .PixelAnalysis import measure_dwells

DC_COLOR = "#ffffff"
DC_COLOR_LIGHT = "#000000"
DC_LINE_WIDTH = 1.4
MAX_VISIBLE_SEGMENTS = 20000
# the averaged samples are drawn only while there are no more than this many
# in view: past it they would paint the trace over in one colour
MAX_WINDOW_POINTS = 20000


@dataclass(frozen=True)
class DwellWindows:
    """Every record's x as drawn and its value; and by dwell, its window as
    record indices, the first and one past the last, its pixel, its plateau,
    and the step into it from the dwell before when that dwell was averaged
    too, NaN when it was not."""

    x: np.ndarray
    value: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    pixel: np.ndarray
    plateau: np.ndarray
    step: np.ndarray

    def plateau_words(self) -> str:
        """The plateaus as one line: their median, the tenth of dwells at
        the bottom, the lowest and its pixel, and the medians of the dwells
        stepped into upward and downward among the larger half of the steps,
        which part a readout that slews one way slower than the other."""
        plateau = self.plateau
        good = np.isfinite(plateau)
        if not good.any():
            return "plateau: no window holds two records, so none slopes"
        values = plateau[good]
        lowest = int(np.flatnonzero(good)[np.argmin(values)])
        words = (
            f"plateau: median {np.median(values) * 100:.1f}%, a tenth of dwells below "
            f"{np.percentile(values, 10) * 100:.1f}%, lowest {plateau[lowest] * 100:.1f}% "
            f"(pixel {self.pixel[lowest]})"
        )
        stepped = good & np.isfinite(self.step)
        if stepped.any():
            size = np.abs(self.step[stepped])
            large = np.zeros_like(good)
            large[stepped] = size >= np.median(size)
            rising, falling = large & (self.step > 0), large & (self.step < 0)
            if rising.any() and falling.any():
                words += (
                    f"; the larger half of the steps, up into the dwell "
                    f"{np.median(plateau[rising]) * 100:.1f}%, down "
                    f"{np.median(plateau[falling]) * 100:.1f}%"
                )
        return words


class PixelDCOverlay:
    """Per-dwell DC levels, culled to the current view."""

    def __init__(self, ax):
        self.ax = ax
        self._collection: LineCollection | None = None
        self._points = None
        self._x0: np.ndarray | None = None
        self._x1: np.ndarray | None = None
        self._dc: np.ndarray | None = None
        self.windows: DwellWindows | None = None
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
        if self._points is not None:
            self._points.remove()
            self._points = None
        self._x0 = self._x1 = self._dc = None
        self.windows = None

    def compute(
        self,
        data: np.ndarray,
        *,
        value_field: str,
        pixel_field: str = "pixel",
        x_field: str = "sample",
        x_scale: float = 1.0,
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
        # The segment covers exactly the averaged records: left edge on the
        # first sample in the window, right edge on the last. Read off the
        # plot's own x field rather than assumed to be the record index. A
        # capture drawn against seconds, or against any axis but the raw
        # index, put these segments tens of thousands of units off screen and
        # looked like an overlay that had computed nothing.
        x = data[x_field].astype(np.float64) * x_scale
        self._x0 = x[s + start]
        self._x1 = x[stop - 1]
        # the step is from the dwell just before, read straight before this
        # one: the transition the window is settling from
        step = np.full(self._dc.size, np.nan)
        follows = np.flatnonzero(s[1:] == e[:-1]) + 1
        step[follows] = self._dc[follows] - self._dc[follows - 1]
        self.windows = DwellWindows(
            x=x,
            value=value,
            lo=s + start,
            hi=stop,
            pixel=data[pixel_field][s].astype(np.int64),
            plateau=plateau_of(value, s, e, start=start, length=length),
            step=step,
        )

    def update(self, xlim: tuple[float, float], plot) -> None:
        """Redraw the segments that fall inside xlim, in the plot's display space."""
        if self._dc is None:
            return

        x0 = self._x0 + plot.offset_x
        x1 = self._x1 + plot.offset_x
        visible = np.flatnonzero((x1 >= xlim[0]) & (x0 <= xlim[1]))
        if visible.size > MAX_VISIBLE_SEGMENTS:
            visible = visible[:: -(-visible.size // MAX_VISIBLE_SEGMENTS)]

        y = plot.display_y(self._dc)

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
        if self.windows is not None:
            self._update_points(self.windows, visible, plot, color)

    def _update_points(self, windows: DwellWindows, visible: np.ndarray, plot, color: str) -> None:
        """The averaged samples of the dwells in view, over the plot's own
        and at their size, while the plot draws its samples as points."""
        counts = windows.hi[visible] - windows.lo[visible]
        drawn_as_points = plot.scatter_artist is not None and plot.scatter_artist.get_visible()
        if not drawn_as_points or int(counts.sum()) > MAX_WINDOW_POINTS:
            if self._points is not None:
                self._points.set_visible(False)
            return
        index = np.repeat(windows.lo[visible] - np.cumsum(counts) + counts, counts) + np.arange(int(counts.sum()))
        offsets = np.column_stack((windows.x[index] + plot.offset_x, plot.display_y(windows.value[index])))
        if self._points is None:
            self._points = self.ax.scatter(offsets[:, 0], offsets[:, 1], s=plot.size, c=color, linewidths=0, zorder=850)
        else:
            self._points.set_offsets(offsets)
            self._points.set_sizes([plot.size])
            self._points.set_facecolors(color)
        self._points.set_visible(True)


def _is_dark(ax) -> bool:
    from matplotlib.colors import to_rgba

    r, g, b, _ = to_rgba(ax.get_facecolor())
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
