#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .Plot2DOverlay import Overlay


def autocolor_overrides(
    plots: Sequence[Overlay],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    enabled: bool,
) -> list[np.ndarray | None]:
    """Per-plot normalized color overrides for single-value color views.

    When every point of a plot inside the view carries the same color value
    (one dm0049 pixel filling the window, typically after zooming onto one
    pixel's DC level), the color axis says nothing: the whole cloud renders
    one shade and outlier samples vanish into it. That plot is recolored by
    each sample's distance from the mean of the in-view samples, normalized
    over the in-view spread, so outliers land at the far end of the palette.

    Distances use display coordinates, which for such plots differ from raw
    values only by an affine transform, so the normalized result is
    identical either way. Entries are full-length [0, 1] arrays aligned with
    display_points(), or None where the plot keeps its own colors.
    """
    overrides: list[np.ndarray | None] = []
    for plot in plots:
        if (
            not enabled
            or not plot.visible
            or plot.viewport_track
            or plot.color_data is None
            or len(plot.points) == 0
        ):
            overrides.append(None)
            continue

        points = plot.display_points()
        x = points[:, 0]
        y = points[:, 1]
        visible = (
            (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        )
        if int(visible.sum()) < 2:
            overrides.append(None)
            continue
        if np.unique(plot.color_data[visible]).size != 1:
            overrides.append(None)
            continue

        y64 = y.astype(np.float64)
        distance = np.abs(y64 - float(y64[visible].mean()))
        dmax = float(distance[visible].max())
        if dmax == 0.0:
            overrides.append(np.full(len(points), 0.5, dtype=np.float32))
            continue
        overrides.append(
            np.clip(distance / dmax, 0.0, 1.0).astype(np.float32)
        )
    return overrides
