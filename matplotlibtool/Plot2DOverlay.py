#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np

from .PlotSource import Records
from .PlotSource import Source
from .PlotSource import settle_floor


def lane_placement(ylim: tuple[float, float], slot: int, slot_count: int, amplitude: float = 0.12) -> tuple[float, float, float]:
    """Where lane `slot` of `slot_count` sits over a view: its baseline, the
    height of a full swing, and where its label goes, in the view's y
    units. Lanes run top-down from the middle of the view, each centred in
    its band with a gap to its neighbours, and the stack compresses once it
    would fill most of the view. The one arithmetic every viewer that lays
    logic lanes over data uses, so the lanes look the same everywhere."""
    low, high = ylim
    span = high - low
    mid = low + span / 2.0
    pitch = amplitude * 1.5
    if pitch * slot_count > 0.85:
        pitch = 0.85 / slot_count
    height = pitch / 1.5 * span
    lane_top = mid + span * (pitch * slot_count / 2.0 - slot * pitch)
    baseline = lane_top - span * pitch / 2.0 - height / 2.0
    label_y = lane_top - span * pitch / 2.0
    return baseline, height, label_y


@dataclass
class Overlay:
    """Configuration and state for a single plot: how its source is drawn."""

    source: Source
    cmap: str
    draw_lines: bool = False
    size: float = 2.0
    color: str | None = None
    offset_x: float = 0.0
    offset_y: float = 0.0
    y_scale: float = 1.0
    visible: bool = True
    line_color: str | None = None  # None = use point colors
    line_width: float = 1.0
    settle_ref: float | None = None  # when set, display y = log10|y - ref|
    # the least residual a settle view draws, set with settle_ref
    settle_floor: float = 1.0
    auto_size: bool = True  # size follows how many points are actually drawn

    # A tracking overlay is positioned by the view rather than by its own
    # values: y_scale and offset_y are recomputed from the current ylim before
    # every render. Logic lines carry no useful magnitude, only timing, so at
    # their own scale they are invisible beside data measured in millions of
    # ADC codes. Tracking keeps them beside whatever is on screen through any
    # zoom, without the operator setting a scale and offset by hand.
    viewport_track: bool = False
    viewport_amplitude: float = 0.12  # high level, as a fraction of the view span

    # data-space y of this overlay's lane centre, valid after track_viewport;
    # where the lane's name label is drawn
    track_label_y: float = field(default=0.0, init=False, repr=False)

    scatter_artist: Any = field(default=None, init=False, repr=False)

    @property
    def count(self) -> int:
        return self.source.count

    def raw_y_range(self) -> tuple[float, float]:
        """Span of this overlay's own y values, before scale and offset."""
        extent = self.source.extent()
        return (0.0, 0.0) if extent is None else (extent[2], extent[3])

    def track_viewport(
        self,
        ylim: tuple[float, float],
        slot: int = 0,
        slot_count: int = 1,
    ) -> None:
        """Place this overlay in its own lane of a stack centred on the view.

        Every tracked overlay at one shared baseline is a single unreadable
        pile, so lane `slot` of `slot_count` gets its own band: lanes run
        top-down in add order, each line centred inside its lane with a gap to
        its neighbours, and the whole stack compresses once it would fill most
        of the view. A line that never changes state has no span to scale, so
        it sits at its own level rather than dividing by zero to get there:
        the lane's high position when pinned high, the baseline when pinned
        low. On the baseline regardless, a lane forced high for a whole
        capture would read as low.
        """
        baseline, height, self.track_label_y = lane_placement(ylim, slot, slot_count, self.viewport_amplitude)
        raw_low, raw_high = self.raw_y_range()
        raw_span = raw_high - raw_low
        if raw_span <= 0.0 or (ylim[1] - ylim[0]) <= 0.0:
            self.y_scale = 0.0
            self.offset_y = baseline + (height if raw_high > 0.0 else 0.0)
            return
        self.y_scale = height / raw_span
        self.offset_y = baseline - raw_low * self.y_scale

    def linear_y(self, y: np.ndarray) -> np.ndarray:
        """y multiplied by y_scale, then offset: before any settle view."""
        if self.y_scale != 1.0:
            y = y * self.y_scale
        if self.offset_y != 0.0:
            y = y + self.offset_y
        return y

    def settle_on(self, reference: float, linear: np.ndarray) -> None:
        """Draw the residual about `reference`, its floor taken from the
        linear values the reference was measured over."""
        self.settle_ref = reference
        self.settle_floor = settle_floor(np.abs(np.asarray(linear, dtype=np.float64) - reference))

    def display_y(self, y: np.ndarray) -> np.ndarray:
        """y as drawn: scaled and offset, and with settle_ref set, log10 of
        the residual about it, floored."""
        linear = self.linear_y(np.asarray(y, dtype=np.float64))
        if self.settle_ref is None:
            return linear
        return np.log10(np.maximum(np.abs(linear - self.settle_ref), self.settle_floor))

    def display_span(self, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The least and the greatest of bins as drawn. A settle view of a
        bin that straddles the reference reaches down to the floor."""
        a = self.linear_y(np.asarray(low, dtype=np.float64))
        b = self.linear_y(np.asarray(high, dtype=np.float64))
        least, greatest = np.minimum(a, b), np.maximum(a, b)
        if self.settle_ref is None:
            return least, greatest
        below, above = least - self.settle_ref, greatest - self.settle_ref
        near = np.where((below <= 0.0) & (above >= 0.0), 0.0, np.minimum(np.abs(below), np.abs(above)))
        far = np.maximum(np.abs(below), np.abs(above))
        return (
            np.log10(np.maximum(near, self.settle_floor)),
            np.log10(np.maximum(far, self.settle_floor)),
        )

    def display_extent(self) -> tuple[float, float, float, float] | None:
        """(x least, x greatest, y least, y greatest) of the whole source as drawn."""
        extent = self.source.extent()
        if extent is None:
            return None
        low, high = self.display_span(np.array([extent[2]]), np.array([extent[3]]))
        return extent[0] + self.offset_x, extent[1] + self.offset_x, float(low[0]), float(high[0])

    def records_in(self, xlim: tuple[float, float], limit: int) -> Records | None:
        """Every point with display x in xlim, x as displayed and y as the
        source has it; None when there are more than limit."""
        x0, x1 = xlim[0] - self.offset_x, xlim[1] - self.offset_x
        if self.source.count_between(x0, x1) > limit:
            return None
        taken = self.source.records(x0, x1)
        x = taken.x + self.offset_x if self.offset_x != 0.0 else taken.x
        return Records(x=np.asarray(x, dtype=np.float64), y=taken.y, color=taken.color)

    def count_in(self, xlim: tuple[float, float]) -> int:
        return self.source.count_between(xlim[0] - self.offset_x, xlim[1] - self.offset_x)

    def y_between(self, xlim: tuple[float, float]) -> tuple[float, float] | None:
        """The least and the greatest y drawn with display x in xlim."""
        moments = self.source.moments(xlim[0] - self.offset_x, xlim[1] - self.offset_x, 0)
        if moments.count == 0:
            return None
        low, high = self.display_span(np.array([moments.low]), np.array([moments.high]))
        return float(low[0]), float(high[0])
