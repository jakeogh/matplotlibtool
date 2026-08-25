#!/usr/bin/env python3
# tab-width:4

"""
Plot2DRenderer - Matplotlib/NumPy renderer for 2D point clouds.

Owns viewport culling, decimation, and artist reuse. Every render re-culls
against the current view, so artists are always consistent with the axes
limits regardless of how the view was changed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

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


class Matplotlib2DRenderer:
    def __init__(self):
        self.plot_initialized = False
        self._batch_solid_line_collection: LineCollection | None = None
        self._batch_colored_line_collection: LineCollection | None = None

    def render(
        self,
        ax: Axes,
        *,
        plots: Sequence[Overlay],
        view_xlim: tuple[float, float],
        view_ylim: tuple[float, float],
        color_ranges: Sequence[tuple[float, float] | None],
        color_overrides: Sequence[np.ndarray | None],
        cull_margin: float,
        max_display_points: int,
        max_line_segments: int,
        disable_antialiasing: bool,
    ) -> None:
        """
        Render all plots at the given view.

        color_ranges[i] is the (vmin, vmax) normalization range for plot i,
        or None for plots without color data. color_overrides[i] is a
        full-length [0, 1] color array that replaces the plot's own color
        data for this render, or None to use it unchanged.
        """
        if not self.plot_initialized:
            self._initialize_axes(ax)
            self.plot_initialized = True

        x_pad = (view_xlim[1] - view_xlim[0]) * cull_margin
        y_pad = (view_ylim[1] - view_ylim[0]) * cull_margin
        cx0, cx1 = view_xlim[0] - x_pad, view_xlim[1] + x_pad
        cy0, cy1 = view_ylim[0] - y_pad, view_ylim[1] + y_pad

        rasterized = not disable_antialiasing

        solid_segments: list[np.ndarray] = []
        solid_colors: list[np.ndarray] = []
        solid_widths: list[np.ndarray] = []

        colored_segments: list[np.ndarray] = []
        colored_arrays: list[np.ndarray] = []
        colored_cmap: str | None = None

        # first pass: cull and subsample every visible plot. The auto marker
        # size comes from the x extent of the view alone, counted before the
        # display subsample and without the cull margin: margin points would
        # inflate the count by up to half again and shrink toward a data
        # edge, making the size drift while panning; y-culled or subsampled
        # counts make the diameter jump when a channel leaves the y window or
        # a subsample threshold is crossed, instead of scaling with zoom.
        prepared: list[
            tuple[
                Overlay,
                tuple[float, float] | None,
                np.ndarray | None,
                np.ndarray,
                np.ndarray,
            ]
            | None
        ] = []
        max_x_count = 0
        for plot, color_range, color_override in zip(
            plots, color_ranges, color_overrides
        ):
            if not plot.visible or len(plot.points) == 0:
                if plot.scatter_artist is not None:
                    plot.scatter_artist.set_visible(False)
                prepared.append(None)
                continue

            points = plot.display_points()

            x = points[:, 0]
            y = points[:, 1]
            if plot.x_ascending:
                # two searches instead of a comparison against every point,
                # and the y cull then runs over the span the x cull kept
                # rather than over the whole record. The indices stay absolute
                # so the colour arrays below index the same way they always
                # did.
                lo = int(np.searchsorted(x, cx0, side="left"))
                hi = int(np.searchsorted(x, cx1, side="right"))
                max_x_count = max(
                    max_x_count,
                    int(np.searchsorted(x, view_xlim[1], side="right"))
                    - int(np.searchsorted(x, view_xlim[0], side="left")),
                )
                span = y[lo:hi]
                idx = lo + np.flatnonzero((span >= cy0) & (span <= cy1))
            else:
                mask_x = (x >= cx0) & (x <= cx1)
                max_x_count = max(
                    max_x_count,
                    int(((x >= view_xlim[0]) & (x <= view_xlim[1])).sum()),
                )
                mask = mask_x & (y >= cy0) & (y <= cy1)
                idx = np.flatnonzero(mask)

            if idx.size == 0:
                if plot.scatter_artist is not None:
                    plot.scatter_artist.set_visible(False)
                prepared.append(None)
                continue

            if idx.size > max_display_points:
                step = -(-idx.size // max_display_points)  # ceil div
                idx = idx[::step]

            prepared.append((plot, color_range, color_override, points, idx))

        shared_size = auto_point_size(ax, max_x_count)
        for plot in plots:
            if plot.auto_size:
                plot.size = shared_size

        for entry in prepared:
            if entry is None:
                continue
            plot, color_range, color_override, points, idx = entry

            display_points = points[idx]

            if color_override is not None:
                display_colors = color_override[idx]
            elif plot.color_data is not None and color_range is not None:
                display_colors = plot.normalized_colors(*color_range)[idx]
            else:
                display_colors = None

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

    def _initialize_axes(self, ax: Axes) -> None:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        self._batch_solid_line_collection = None
        self._batch_colored_line_collection = None
