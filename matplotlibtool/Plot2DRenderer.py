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
        cull_margin: float,
        max_display_points: int,
        max_line_segments: int,
        disable_antialiasing: bool,
    ) -> None:
        """
        Render all plots at the given view.

        color_ranges[i] is the (vmin, vmax) normalization range for plot i,
        or None for plots without color data.
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

        for plot, color_range in zip(plots, color_ranges):
            if not plot.visible or len(plot.points) == 0:
                if plot.scatter_artist is not None:
                    plot.scatter_artist.set_visible(False)
                continue

            base_points = plot.display_points()
            if plot.offset_x != 0.0 or plot.offset_y != 0.0:
                points = base_points + np.array(
                    [plot.offset_x, plot.offset_y], dtype=np.float32
                )
            else:
                points = base_points

            x = points[:, 0]
            y = points[:, 1]
            mask = (x >= cx0) & (x <= cx1) & (y >= cy0) & (y <= cy1)
            idx = np.flatnonzero(mask)

            if idx.size == 0:
                if plot.scatter_artist is not None:
                    plot.scatter_artist.set_visible(False)
                continue

            if idx.size > max_display_points:
                step = -(-idx.size // max_display_points)  # ceil div
                idx = idx[::step]

            display_points = points[idx]

            if plot.color_data is not None and color_range is not None:
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
