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


def _view_color_span(
    plot: Overlay,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[float, float] | None:
    """(min, max) of the plot's color values inside the view, None if empty."""
    points = plot.display_points()
    x = points[:, 0]
    y = points[:, 1]
    if plot.x_ascending:
        lo = int(np.searchsorted(x, xlim[0], side="left"))
        hi = int(np.searchsorted(x, xlim[1], side="right"))
        y = y[lo:hi]
        colors = plot.color_data[lo:hi][(y >= ylim[0]) & (y <= ylim[1])]
    else:
        colors = plot.color_data[
            (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        ]
    if colors.size == 0:
        return None
    return float(colors.min()), float(colors.max())


def view_color_ranges(
    plots: Sequence[Overlay],
    color_ranges: Sequence[tuple[float, float] | None],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[float, float] | None]:
    """Normalization ranges narrowed to the color values inside the view.

    A range taken over the whole record spends the palette on values that
    are off screen: zoomed onto a few dm0049 pixels out of thousands, the
    points in view differ by a sliver of the scale and render as near
    identical shades. Narrowing the range to the values in view spreads the
    whole palette across what is on screen.

    Plots that share a full range share a color scale (a group, or the
    fields of one array), and keep sharing one: their in-view spans are
    pooled, so a color value still maps to the same shade on each of them.
    A scale with nothing in view keeps its full range, as do tracking
    overlays, which the view positions and so are always inside it.

    The span is taken over every in-view point rather than the drawn
    subsample, so it does not move with the display budget.
    """
    pooled: dict[tuple[float, float], tuple[float, float]] = {}
    fitted: list[bool] = []
    for plot, full_range in zip(plots, color_ranges):
        fit = (
            full_range is not None
            and plot.visible
            and not plot.viewport_track
            and len(plot.points) > 0
        )
        fitted.append(fit)
        if not fit:
            continue
        span = _view_color_span(plot, xlim, ylim)
        if span is None:
            continue
        seen = pooled.get(full_range)
        pooled[full_range] = (
            span
            if seen is None
            else (min(seen[0], span[0]), max(seen[1], span[1]))
        )
    return [
        pooled.get(full_range, full_range) if fit else full_range
        for fit, full_range in zip(fitted, color_ranges)
    ]
