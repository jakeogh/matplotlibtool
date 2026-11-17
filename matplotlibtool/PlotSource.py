#!/usr/bin/env python3
# tab-width:4

"""What a plot draws from.

A plot never holds its points. It asks its source for what a view needs:
every point in a window of x when there are few enough to draw, the least
and the greatest of bins of consecutive points when there are not, and the
count, sum and spread of the points in a window for the statistics. A
source held in memory answers from its arrays; one read from a file answers
from the file and whatever summary of it the file keeps, so a source of any
length costs a view what the view shows.

A table is named columns sharing one x, from which a source is made for one
column against another: a structured array, or a capture read as it is
drawn. The viewer registers tables, and builds a plot's source from one
when a field is shown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .CoordinateTransformEngine import CoordinateTransformEngine

# the most points any analysis of a view reads: an FFT, a settling fit, a
# saved window, the pixel analyses, a median. Past it the view is to be
# narrowed, since what is analysed is what is shown.
ANALYSIS_RECORDS = 1 << 24

# points to a bin of the finest level an array source keeps of its extremes
ARRAY_BIN_POINTS = 16


@dataclass(frozen=True)
class Records:
    """Every point of a source in a window of x, in the source's order."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    color: NDArray[np.float32] | None


@dataclass(frozen=True)
class Band:
    """A window of a source as bins of consecutive points, each drawn at
    the x of its first point: the least and the greatest y of each."""

    x: NDArray[np.float64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]


@dataclass(frozen=True)
class Moments:
    """The points of a window: how many, their sum and their spread about
    their own mean, their extremes, and their median when it was taken."""

    count: int
    total: float
    m2: float
    low: float
    high: float
    median: float | None


@runtime_checkable
class Source(Protocol):
    """One plot's points, read a window at a time."""

    @property
    def count(self) -> int: ...

    @property
    def ascending(self) -> bool:
        """Whether x never decreases: a source that does not cannot be
        binned along x, and is drawn from its points only."""
        ...

    @property
    def colored(self) -> bool: ...

    @property
    def color_key(self) -> object:
        """Sources with equal keys share one colour scale."""
        ...

    def extent(self) -> tuple[float, float, float, float] | None:
        """(x least, x greatest, y least, y greatest), None when empty."""
        ...

    def color_span(self) -> tuple[float, float] | None:
        """The colour values' extremes over the whole source, None when
        uncoloured or when they are not known without reading it all."""
        ...

    def count_between(self, x0: float, x1: float) -> int: ...

    def records(self, x0: float, x1: float) -> Records:
        """Every point with x in [x0, x1]; the caller counts first."""
        ...

    def band(self, x0: float, x1: float, columns: int) -> Band:
        """The window as bins of at most half a column of points each, a
        bin either side so a line runs to the edges."""
        ...

    def moments(self, x0: float, x1: float, median_limit: int) -> Moments:
        """The points with x in [x0, x1], their median when there are no
        more than median_limit of them."""
        ...


@runtime_checkable
class Table(Protocol):
    """Named columns sharing one x, from which plots are made."""

    @property
    def fields(self) -> tuple[str, ...]:
        """The columns that can be drawn against x."""
        ...

    @property
    def color_fields(self) -> tuple[str, ...]: ...

    @property
    def nbytes(self) -> int:
        """What the table holds in memory."""
        ...

    def source(
        self,
        *,
        x_field: str,
        y_field: str,
        color_field: str | None,
        x_scale: float,
        transform: dict,
    ) -> Source: ...

    def window(self, *, x_field: str, x_scale: float, x0: float, x1: float, limit: int) -> NDArray | None:
        """Every record with x in [x0, x1], all its columns, as a
        structured array in x order; None when there are more than limit."""
        ...


def per_bin_for(points: int, columns: int) -> int:
    """Points to a bin, a power of two, for no more than half a column of
    points to a bin."""
    return 1 << max(math.ceil(math.log2(max(points / (2.0 * max(columns, 1)), 1.0))), 0)


def spread_of(y: NDArray) -> tuple[float, float]:
    """The sum of y and its spread about its own mean."""
    if y.size == 0:
        return 0.0, 0.0
    total = float(y.sum(dtype=np.float64))
    deviation = y - total / y.size
    return total, float(np.dot(deviation, deviation))


class ArraySource:
    """Points held in memory: x and y, and a colour value for each when
    the plot is coloured.

    A view of more points than are drawn is drawn from bins of them: their
    extremes are kept as levels of halving resolution, built when a view
    first needs them, so a view of any width costs the bins it shows.
    """

    def __init__(self, points: NDArray[np.float64], color: NDArray[np.float32] | None = None) -> None:
        self.points = points
        self.color = color
        self._levels: list[tuple[NDArray[np.float64], NDArray[np.float64]]] | None = None

    @property
    def count(self) -> int:
        return int(self.points.shape[0])

    @property
    def x(self) -> NDArray[np.float64]:
        return self.points[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        return self.points[:, 1]

    @cached_property
    def ascending(self) -> bool:
        x = self.x
        return bool(x.size < 2 or np.all(x[1:] >= x[:-1]))

    @property
    def colored(self) -> bool:
        return self.color is not None

    @property
    def color_key(self) -> object:
        return id(self)

    @cached_property
    def _extent(self) -> tuple[float, float, float, float] | None:
        if self.count == 0:
            return None
        x, y = self.x, self.y
        if self.ascending:
            x_low, x_high = float(x[0]), float(x[-1])
        else:
            x_low, x_high = float(x.min()), float(x.max())
        return x_low, x_high, float(y.min()), float(y.max())

    def extent(self) -> tuple[float, float, float, float] | None:
        return self._extent

    @cached_property
    def _color_span(self) -> tuple[float, float] | None:
        if self.color is None or self.color.size == 0:
            return None
        return float(self.color.min()), float(self.color.max())

    def color_span(self) -> tuple[float, float] | None:
        return self._color_span

    def _span(self, x0: float, x1: float) -> tuple[int, int]:
        x = self.x
        return int(np.searchsorted(x, x0, side="left")), int(np.searchsorted(x, x1, side="right"))

    def _mask(self, x0: float, x1: float) -> NDArray[np.bool_]:
        x = self.x
        return (x >= x0) & (x <= x1)

    def count_between(self, x0: float, x1: float) -> int:
        if self.ascending:
            lo, hi = self._span(x0, x1)
            return hi - lo
        return int(self._mask(x0, x1).sum())

    def records(self, x0: float, x1: float) -> Records:
        if self.ascending:
            lo, hi = self._span(x0, x1)
            return Records(
                x=self.x[lo:hi],
                y=self.y[lo:hi],
                color=None if self.color is None else self.color[lo:hi],
            )
        keep = np.flatnonzero(self._mask(x0, x1))
        return Records(
            x=self.x[keep],
            y=self.y[keep],
            color=None if self.color is None else self.color[keep],
        )

    def _level(self, index: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._levels is None:
            y = self.y
            bins = -(-y.size // ARRAY_BIN_POINTS)
            padded = np.empty(bins * ARRAY_BIN_POINTS, dtype=np.float64)
            padded[: y.size] = y
            # the tail repeats the last point, which moves neither extreme
            padded[y.size :] = y[-1]
            shaped = padded.reshape(bins, ARRAY_BIN_POINTS)
            levels = [(shaped.min(axis=1), shaped.max(axis=1))]
            while levels[-1][0].size > 1:
                low, high = levels[-1]
                pairs = low.size // 2
                next_low = np.minimum(low[0 : pairs * 2 : 2], low[1 : pairs * 2 : 2])
                next_high = np.maximum(high[0 : pairs * 2 : 2], high[1 : pairs * 2 : 2])
                if low.size % 2:
                    next_low = np.append(next_low, low[-1])
                    next_high = np.append(next_high, high[-1])
                levels.append((next_low, next_high))
            self._levels = levels
        return self._levels[min(index, len(self._levels) - 1)]

    def band(self, x0: float, x1: float, columns: int) -> Band:
        lo, hi = self._span(x0, x1)
        per_bin = per_bin_for(hi - lo, columns)
        first = max(lo // per_bin - 1, 0)
        last = min(-(-hi // per_bin) + 1, -(-self.count // per_bin))
        starts = np.arange(first, last, dtype=np.int64) * per_bin
        if per_bin < ARRAY_BIN_POINTS:
            y = self.y
            stop = min(last * per_bin, self.count)
            return Band(
                x=self.x[starts],
                low=np.minimum.reduceat(y[starts[0] : stop], starts - starts[0]),
                high=np.maximum.reduceat(y[starts[0] : stop], starts - starts[0]),
            )
        low, high = self._level(int(math.log2(per_bin // ARRAY_BIN_POINTS)))
        return Band(x=self.x[starts], low=low[first:last], high=high[first:last])

    def moments(self, x0: float, x1: float, median_limit: int) -> Moments:
        y = self.records(x0, x1).y
        if y.size == 0:
            return Moments(0, 0.0, 0.0, math.nan, math.nan, None)
        total, m2 = spread_of(y)
        return Moments(
            count=int(y.size),
            total=total,
            m2=m2,
            low=float(y.min()),
            high=float(y.max()),
            median=float(np.median(y)) if y.size <= median_limit else None,
        )


class StructuredTable:
    """A structured array as a table: its fields drawn against one of them."""

    def __init__(self, array: NDArray) -> None:
        if not isinstance(array, np.ndarray) or array.dtype.names is None:
            raise TypeError("a table is made from a structured numpy array with named fields")
        self.array = array

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.array.dtype.names)

    @property
    def color_fields(self) -> tuple[str, ...]:
        return self.fields

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)

    def points(self, x_field: str, y_field: str, x_scale: float) -> NDArray[np.float64]:
        """x and y as an (N, 2) array, x multiplied by x_scale.

        Double precision, because x is often a sample counter. A float32
        carries twenty four bits of integer exactly, and a capture passes
        sample 16,777,216 eight seconds in; past it neighbouring samples
        round to one x and every spacing read off the plot is wrong.
        """
        for name in (x_field, y_field):
            if name not in self.fields:
                raise ValueError(f"field {name!r} not found in data. Available: {self.fields}")
        column = self.array[x_field]
        x = (column * x_scale if x_scale != 1.0 else column).astype(np.float64)
        return np.column_stack((x, self.array[y_field].astype(np.float64)))

    def colors(self, color_field: str | None) -> NDArray[np.float32] | None:
        if color_field is None:
            return None
        if color_field not in self.fields:
            raise ValueError(f"color field {color_field!r} not found in data. Available: {self.fields}")
        return self.array[color_field].astype(np.float32)

    def source(
        self,
        *,
        x_field: str,
        y_field: str,
        color_field: str | None,
        x_scale: float,
        transform: dict,
    ) -> ArraySource:
        points = CoordinateTransformEngine(dimensions=2).apply_transform(
            self.points(x_field, y_field, x_scale), transform
        )
        return ArraySource(points, self.colors(color_field))

    def window(self, *, x_field: str, x_scale: float, x0: float, x1: float, limit: int) -> NDArray | None:
        x = self.array[x_field].astype(np.float64) * x_scale
        inside = (x >= x0) & (x <= x1)
        if int(inside.sum()) > limit:
            return None
        if inside.all():
            picked, picked_x = self.array, x
        else:
            keep = np.flatnonzero(inside)
            picked, picked_x = self.array[keep], x[keep]
        if picked_x.size > 1 and np.any(picked_x[1:] < picked_x[:-1]):
            picked = picked[np.argsort(picked_x, kind="stable")]
        return picked


def as_table(data) -> Table:
    """A table for what a caller hands the viewer: a structured array is
    wrapped, a table is taken as it is."""
    if isinstance(data, np.ndarray):
        return StructuredTable(data)
    if isinstance(data, Table):
        return data
    raise TypeError(f"data must be a structured numpy array or a table, not {type(data).__name__}")


def settle_floor(residual: NDArray[np.float64]) -> float:
    """The least residual a settle view draws: half the smallest that is
    not zero, so an exact zero sits just below everything else; one code
    when there is none."""
    positive = residual[residual > 0.0]
    return float(positive.min()) * 0.5 if positive.size else 1.0
