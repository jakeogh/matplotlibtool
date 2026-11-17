#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .Plot2DOverlay import Overlay


def autocolor(y: np.ndarray, color: np.ndarray, in_view: np.ndarray) -> np.ndarray | None:
    """[0, 1] colours by distance from the mean, for a plot whose points in
    view all carry one colour value; None where the plot keeps its own.

    When every point of a plot inside the view carries the same color value
    (one dm0049 pixel filling the window, typically after zooming onto one
    pixel's DC level), the color axis says nothing: the whole cloud renders
    one shade and outlier samples vanish into it. That plot is recolored by
    each sample's distance from the mean of the in-view samples, normalized
    over the in-view spread, so outliers land at the far end of the palette.

    Distances use display coordinates, which for such plots differ from raw
    values only by an affine transform, so the normalized result is
    identical either way.
    """
    if int(in_view.sum()) < 2:
        return None
    if np.unique(color[in_view]).size != 1:
        return None
    distance = np.abs(y - float(y[in_view].mean()))
    dmax = float(distance[in_view].max())
    if dmax == 0.0:
        return np.full(y.size, 0.5, dtype=np.float32)
    return np.clip(distance / dmax, 0.0, 1.0).astype(np.float32)


def pooled_view_spans(
    plots: Sequence[Overlay],
    shown: Sequence[object],
    color_ranges: Sequence[tuple[float, float] | None],
    palette_view: bool,
) -> list[tuple[float, float] | None]:
    """Each plot's colour range for this view.

    A range taken over the whole record spends the palette on values that
    are off screen: zoomed onto a few dm0049 pixels out of thousands, the
    points in view differ by a sliver of the scale and render as near
    identical shades. With palette_view the range is narrowed to the values
    in view, and a plot whose range over its whole source is not known is
    always coloured that way.

    Plots that share a full range share a color scale (a group, or the
    fields of one array), as do plots whose sources share a colour key, and
    they keep sharing one: their in-view spans are pooled, so a color value
    still maps to the same shade on each of them. A scale with nothing in
    view keeps its full range.
    """
    from .Plot2DRenderer import Points

    pooled: dict[object, tuple[float, float]] = {}
    keys: list[object | None] = []
    for plot, view, full_range in zip(plots, shown, color_ranges):
        if not isinstance(view, Points) or view.color is None or plot.viewport_track:
            keys.append(None)
            continue
        key = full_range if full_range is not None else plot.source.color_key
        keys.append(key)
        if full_range is not None and (not palette_view or not view.in_view.any()):
            continue
        colors = view.color[view.in_view] if view.in_view.any() else view.color
        span = (float(colors.min()), float(colors.max()))
        seen = pooled.get(key)
        pooled[key] = span if seen is None else (min(seen[0], span[0]), max(seen[1], span[1]))
    return [
        full_range if key is None else pooled.get(key, full_range)
        for key, full_range in zip(keys, color_ranges)
    ]
