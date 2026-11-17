#!/usr/bin/env python3
# tab-width:4

"""
Plot2DRenderer - Matplotlib/NumPy renderer for 2D point clouds.

Every render asks each plot's source for the view: every point in it when
there are few enough to draw, else the least and the greatest of bins of
at most half a pixel column of points, filled between from each bin's
first point to the next's. A column can show no more than its extremes, so
the fill is what every point would have drawn, at the cost of the columns
rather than the points.
Artists are reused, and every render re-asks against the current view, so
they are always consistent with the axes limits however the view changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from .AutoColor import autocolor
from .AutoColor import pooled_view_spans
from .Plot2DOverlay import Overlay


# Markers are sized against the horizontal spacing of the data in the view:
# the axes width over the largest number of points any one plot has inside
# the view window, margin excluded. The diameter follows that spacing when
# the view is sparse and blends smoothly (hypot) into a one-pixel floor when
# it is dense, so zooming never crosses a hard knee: no clipped plateau, no
# sudden growth onset, and no change with vertical zoom, with a channel
# leaving the y window, or with how many overlaid plots share the same x
# positions. The scatter size is an area in points squared, hence the square
# of the diameter.
AUTO_FILL = 0.576       # fraction of the inter-point spacing a marker occupies
AUTO_AREA_FLOOR = 0.6   # about a one-pixel dot at 100 dpi; dense views hold this
AUTO_SIZE_MAX = 144.0   # 12 point diameter

# The points a view may draw as points is shared across the plots that are
# visible, not granted to each of them: one axes has one set of pixels however
# many captures are loaded into it. A plot with more than its share in the
# view is drawn as its extremes.
MIN_DISPLAY_POINTS = 2_000

# A frame drawn while the mouse is still moving is replaced before anyone
# reads it, so it is drawn from fewer points. The figure itself costs about as
# much as tens of thousands of points, so this is worth roughly a halving of
# the frame and no more; the rest of making a drag follow the mouse is not
# letting frames queue behind it.
INTERACTIVE_DIVISOR = 8
INTERACTIVE_DISPLAY_POINTS = 5_000

# where on its colormap a coloured plot drawn as extremes takes its one colour
BAND_COLOR_AT = 0.5


def auto_point_size(ax, drawn_count: int) -> float:
    """Marker area for the given number of in-view points across the axes."""
    if drawn_count <= 0:
        return AUTO_AREA_FLOOR
    width_px = ax.get_window_extent().width
    width_pt = width_px * 72.0 / ax.get_figure().dpi
    diameter_pt = float(np.hypot(
        (width_pt / drawn_count) * AUTO_FILL,
        AUTO_AREA_FLOOR ** 0.5,
    ))
    return float(min(diameter_pt * diameter_pt, AUTO_SIZE_MAX))


@dataclass
class Points:
    """A plot's points in the view, as drawn."""

    x: np.ndarray
    y: np.ndarray
    color: np.ndarray | None
    # which of them are inside the view, margin excluded, for the colours
    in_view: np.ndarray


@dataclass
class Strokes:
    """A plot's bins in the view, as drawn."""

    x: np.ndarray
    low: np.ndarray
    high: np.ndarray


def band_color(plot: Overlay) -> tuple[float, float, float, float]:
    """The one colour a plot drawn as its extremes takes."""
    if plot.color is not None:
        return to_rgba(plot.color)
    if plot.source.colored:
        return colormaps[plot.cmap](BAND_COLOR_AT)
    return to_rgba("white")


def joined(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bins stretched to meet their neighbours: a line through the points
    crosses every gap between one bin's range and the next's."""
    low, high = low.copy(), high.copy()
    rising = low[1:] > high[:-1]
    falling = high[1:] < low[:-1]
    low[1:] = np.where(rising, high[:-1], low[1:])
    high[1:] = np.where(falling, low[:-1], high[1:])
    return low, high


class Matplotlib2DRenderer:
    def __init__(self):
        self.plot_initialized = False
        self.last_drawn_points = 0
        self._batch_solid_line_collection: LineCollection | None = None
        self._batch_colored_line_collection: LineCollection | None = None
        self._band_artists: list = []

    def render(
        self,
        ax: Axes,
        *,
        plots: Sequence[Overlay],
        view_xlim: tuple[float, float],
        view_ylim: tuple[float, float],
        color_ranges: Sequence[tuple[float, float] | None],
        palette_view: bool,
        autocolor_enabled: bool,
        cull_margin: float,
        max_display_points: int,
        max_line_segments: int,
        disable_antialiasing: bool,
    ) -> None:
        """
        Render all plots at the given view.

        color_ranges[i] is plot i's colour range over its whole source, or
        None where that is not known; a plot without one is coloured over
        the values in view, as every plot is with palette_view set.
        """
        if not self.plot_initialized:
            self._initialize_axes(ax)
            self.plot_initialized = True

        x_pad = (view_xlim[1] - view_xlim[0]) * cull_margin
        y_pad = (view_ylim[1] - view_ylim[0]) * cull_margin
        cx0, cx1 = view_xlim[0] - x_pad, view_xlim[1] + x_pad
        cy0, cy1 = view_ylim[0] - y_pad, view_ylim[1] + y_pad
        columns = max(int(round(ax.get_window_extent().width * (1.0 + 2.0 * cull_margin))), 1)

        rasterized = not disable_antialiasing

        drawn_plots = sum(1 for plot in plots if plot.visible and plot.count > 0)
        budget = (
            max(MIN_DISPLAY_POINTS, max_display_points // drawn_plots)
            if drawn_plots
            else max_display_points
        )

        # first pass: what each visible plot shows. The auto marker size
        # comes from the x extent of the view alone, counted without the
        # cull margin: margin points would inflate the count by up to half
        # again and shrink toward a data edge, making the size drift while
        # panning; y-culled counts make the diameter jump when a channel
        # leaves the y window, instead of scaling with zoom.
        shown: list[Points | Strokes | None] = []
        max_x_count = 0
        for plot in plots:
            if not plot.visible or plot.count == 0:
                shown.append(None)
                continue
            source = plot.source
            x0, x1 = cx0 - plot.offset_x, cx1 - plot.offset_x
            max_x_count = max(max_x_count, plot.count_in(view_xlim))
            n = source.count_between(x0, x1)
            if n == 0:
                shown.append(None)
            elif n <= budget or not source.ascending:
                taken = source.records(x0, x1)
                x = taken.x + plot.offset_x
                y = plot.display_y(taken.y)
                keep = np.flatnonzero((y >= cy0) & (y <= cy1))
                if keep.size > budget:
                    # a cloud that is not a function of x has no extremes
                    # along it to draw instead; it is strided, as it is
                    keep = keep[:: -(-keep.size // budget)]
                if keep.size == 0:
                    shown.append(None)
                    continue
                x, y = x[keep], y[keep]
                color = None if taken.color is None else taken.color[keep]
                in_view = (x >= view_xlim[0]) & (x <= view_xlim[1]) & (y >= view_ylim[0]) & (y <= view_ylim[1])
                shown.append(Points(x=x, y=y, color=color, in_view=in_view))
            else:
                band = source.band(x0, x1, columns)
                low, high = plot.display_span(band.low, band.high)
                if plot.draw_lines:
                    low, high = joined(low, high)
                shown.append(Strokes(x=band.x + plot.offset_x, low=low, high=high))

        ranges = pooled_view_spans(plots, shown, color_ranges, palette_view)

        self.last_drawn_points = 0
        shared_size = auto_point_size(ax, max_x_count)
        for plot in plots:
            if plot.auto_size:
                plot.size = shared_size

        solid_segments: list[np.ndarray] = []
        solid_colors: list[np.ndarray] = []
        solid_widths: list[np.ndarray] = []

        colored_segments: list[np.ndarray] = []
        colored_arrays: list[np.ndarray] = []
        colored_cmap: str | None = None

        bands: list[tuple[Overlay, Strokes]] = []

        for plot, view, color_range in zip(plots, shown, ranges):
            if not isinstance(view, Points):
                if plot.scatter_artist is not None:
                    plot.scatter_artist.set_visible(False)
            if view is None:
                continue
            if isinstance(view, Strokes):
                self.last_drawn_points += view.x.size
                bands.append((plot, view))
                continue

            self.last_drawn_points += view.x.size
            display_points = np.column_stack((view.x, view.y))
            if view.color is None or color_range is None:
                display_colors = None
            else:
                display_colors = (
                    autocolor(view.y, view.color, view.in_view) if autocolor_enabled else None
                )
                if display_colors is None:
                    span = color_range[1] - color_range[0]
                    display_colors = (
                        (view.color - color_range[0]) / span
                        if span > 0.0
                        else np.full(view.color.size, 0.5, dtype=np.float32)
                    )

            self._update_scatter(ax, plot, display_points, display_colors, rasterized)

            if plot.draw_lines and len(display_points) > 1:
                # segments denser than the screen are pure Path-creation cost
                if len(display_points) - 1 > max_line_segments:
                    step = -(-(len(display_points) - 1) // max_line_segments)
                    line_points = display_points[::step]
                    line_colors = (
                        display_colors[::step] if display_colors is not None else None
                    )
                else:
                    line_points = display_points
                    line_colors = display_colors

                segments = np.stack([line_points[:-1], line_points[1:]], axis=1)
                n = len(segments)

                if plot.line_color is not None or line_colors is None:
                    color = to_rgba(plot.line_color or "gray")
                    solid_segments.append(segments)
                    solid_colors.append(np.tile(color, (n, 1)))
                    solid_widths.append(np.full(n, plot.line_width))
                else:
                    colored_segments.append(segments)
                    colored_arrays.append((line_colors[:-1] + line_colors[1:]) * 0.5)
                    if colored_cmap is None:
                        colored_cmap = plot.cmap

        self._rebuild_line_collections(
            ax,
            solid_segments,
            solid_colors,
            solid_widths,
            colored_segments,
            colored_arrays,
            colored_cmap,
            rasterized,
        )
        self._rebuild_bands(ax, bands, rasterized)

        ax.set_xlim(*view_xlim)
        ax.set_ylim(*view_ylim)

    def _update_scatter(
        self,
        ax: Axes,
        plot: Overlay,
        display_points: np.ndarray,
        display_colors: np.ndarray | None,
        rasterized: bool,
    ) -> None:
        artist = plot.scatter_artist

        if display_colors is not None:
            if artist is None:
                plot.scatter_artist = ax.scatter(
                    display_points[:, 0],
                    display_points[:, 1],
                    c=display_colors,
                    s=plot.size,
                    cmap=plot.cmap,
                    alpha=0.8,
                    rasterized=rasterized,
                    vmin=0.0,
                    vmax=1.0,
                )
            else:
                artist.set_offsets(display_points)
                artist.set_array(display_colors)
                artist.set_sizes([plot.size])
                artist.set_cmap(plot.cmap)
                artist.set_clim(0.0, 1.0)
                artist.set_visible(True)
        else:
            if artist is None:
                plot.scatter_artist = ax.scatter(
                    display_points[:, 0],
                    display_points[:, 1],
                    c=plot.color or "white",
                    s=plot.size,
                    alpha=0.8,
                    rasterized=rasterized,
                )
            else:
                artist.set_offsets(display_points)
                artist.set_sizes([plot.size])
                artist.set_facecolors(plot.color or "white")
                artist.set_visible(True)

    def _rebuild_line_collections(
        self,
        ax: Axes,
        solid_segments: list[np.ndarray],
        solid_colors: list[np.ndarray],
        solid_widths: list[np.ndarray],
        colored_segments: list[np.ndarray],
        colored_arrays: list[np.ndarray],
        colored_cmap: str | None,
        rasterized: bool,
    ) -> None:
        if self._batch_solid_line_collection is not None:
            self._batch_solid_line_collection.remove()
            self._batch_solid_line_collection = None

        if self._batch_colored_line_collection is not None:
            self._batch_colored_line_collection.remove()
            self._batch_colored_line_collection = None

        if solid_segments:
            lc = LineCollection(
                np.concatenate(solid_segments),
                colors=np.concatenate(solid_colors),
                linewidths=np.concatenate(solid_widths),
                alpha=0.6,
                rasterized=rasterized,
            )
            ax.add_collection(lc, autolim=False)
            self._batch_solid_line_collection = lc

        if colored_segments:
            lc = LineCollection(
                np.concatenate(colored_segments),
                array=np.concatenate(colored_arrays),
                cmap=colored_cmap,
                linewidths=1.0,
                alpha=0.6,
                rasterized=rasterized,
            )
            lc.set_clim(0.0, 1.0)
            ax.add_collection(lc, autolim=False)
            self._batch_colored_line_collection = lc

    def _rebuild_bands(
        self,
        ax: Axes,
        bands: list[tuple[Overlay, Strokes]],
        rasterized: bool,
    ) -> None:
        """Every plot drawn as its extremes, each bin filled from its own x
        to the next bin's, so the bins tile the axis whatever pixels they
        land on. The fill's edge is a pixel wide: a bin whose extremes are
        one value has nothing to fill and is drawn by its edge."""
        for artist in self._band_artists:
            artist.remove()
        self._band_artists = []
        pixel_width = 72.0 / ax.get_figure().dpi
        for plot, view in bands:
            last = view.x[-1] + (view.x[-1] - view.x[-2] if view.x.size > 1 else 0.0)
            x = np.append(view.x, last)
            color = band_color(plot) if plot.line_color is None else to_rgba(plot.line_color)
            self._band_artists.append(
                ax.fill_between(
                    x,
                    np.append(view.low, view.low[-1]),
                    np.append(view.high, view.high[-1]),
                    step="post",
                    facecolor=color,
                    edgecolor=color,
                    linewidth=max(plot.line_width, pixel_width) if plot.draw_lines else pixel_width,
                    alpha=0.8,
                    rasterized=rasterized,
                )
            )

    def _initialize_axes(self, ax: Axes) -> None:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        self._batch_solid_line_collection = None
        self._batch_colored_line_collection = None
        self._band_artists = []
