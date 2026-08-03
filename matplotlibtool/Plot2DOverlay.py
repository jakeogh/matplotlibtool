#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np


@dataclass
class Overlay:
    """Configuration and state for a single plot."""

    points: np.ndarray
    cmap: str
    color_data: np.ndarray | None = None
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

    scatter_artist: Any = field(default=None, init=False, repr=False)

    # caches keyed on the identity of color_data
    _range_cache: tuple[int, tuple[float, float]] | None = field(
        default=None, init=False, repr=False
    )
    _norm_cache: tuple[int, float, float, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )
    _settle_cache: tuple[tuple, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )

    def display_points(self) -> np.ndarray:
        """
        Points as rendered: y multiplied by y_scale, then offsets. With
        settle_ref set, y becomes log10 of the residual about that reference,
        measured in the same scaled-and-offset space the reference was taken in.
        """
        pts = self.points
        if self.y_scale != 1.0:
            pts = pts * np.array([1.0, self.y_scale], dtype=np.float32)
        if self.offset_x != 0.0 or self.offset_y != 0.0:
            pts = pts + np.array([self.offset_x, self.offset_y], dtype=np.float32)
        if self.settle_ref is None:
            return pts

        key = (id(self.points), self.y_scale, self.offset_x, self.offset_y,
               self.settle_ref)
        cache = self._settle_cache
        if cache is not None and cache[0] == key:
            return cache[1]

        residual = np.abs(pts[:, 1].astype(np.float64) - self.settle_ref)
        positive = residual[residual > 0.0]
        # zero residuals floored at half the smallest nonzero one; a trace that
        # is exactly constant has no nonzero residual at all, and is a
        # degenerate case to render at one code, not to raise from a paint path
        floor = float(positive.min()) * 0.5 if positive.size else 1.0

        out = np.empty_like(pts, dtype=np.float32)
        out[:, 0] = pts[:, 0]
        out[:, 1] = np.log10(np.maximum(residual, floor))

        self._settle_cache = (key, out)
        return out

    def color_range(self) -> tuple[float, float]:
        """Full-array (min, max) of color_data, cached."""
        key = id(self.color_data)
        if self._range_cache is None or self._range_cache[0] != key:
            self._range_cache = (
                key,
                (float(self.color_data.min()), float(self.color_data.max())),
            )
        return self._range_cache[1]

    def normalized_colors(self, vmin: float, vmax: float) -> np.ndarray:
        """color_data mapped to [0, 1] over (vmin, vmax), cached."""
        key = id(self.color_data)
        cache = self._norm_cache
        if cache is not None and cache[0] == key and cache[1] == vmin and cache[2] == vmax:
            return cache[3]

        span = vmax - vmin
        if span > 1e-9:
            norm = (self.color_data.astype(np.float32) - vmin) / span
        else:
            norm = np.full(len(self.color_data), 0.5, dtype=np.float32)

        self._norm_cache = (key, vmin, vmax, norm)
        return norm
