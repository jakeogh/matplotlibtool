#!/usr/bin/env python3
# tab-width:4

"""
Plot2DRenderer.py - Pure Matplotlib/NumPy renderer for 2D point cloud visualization

OPTIMIZED VERSION - Performance monitoring removed, speed improvements added
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection


class Matplotlib2DRenderer:
    """
    Pure Matplotlib/NumPy renderer for PointCloud2DViewerMatplotlib.

    Optimized for speed with batch rendering and minimal overhead.
    """

    def __init__(self):
        """Initialize the renderer with artist tracking."""
        self.plot_initialized = False
        self.grid_line_artists = []

        # Batch line collection references for fast rendering
        self._batch_solid_line_collection = None
        self._batch_colored_line_collection = None

    def update_all_plots(
        self,
        ax: Axes,
        *,
        plots: Sequence[object],
        auto_aspect: bool,
        view_xlim: tuple[float, float],
        view_ylim: tuple[float, float],
        grid_enabled: bool,
        grid_power: int,
        grid_color: str,
        axes_grid_color: str,
        disable_antialiasing: bool,
        max_display_points: int,
        in_zoom_box: bool,
    ) -> None:
        """
        Update all plots with optimized batch rendering.

        Performance optimizations:
        - Pre-allocate lists with estimated sizes
        - Batch all line segments into single collections
        - Minimize artist creation/deletion
        - Use list comprehensions where possible
        - Avoid redundant operations
        """
        # Initialize axes styling on first call
        if not self.plot_initialized:
            self._initialize_axes(ax, axes_grid_color)
            self.plot_initialized = True

        # Track which artists should remain visible
        active_scatter_artists = set()

        # Clear and redraw grid
        if self.grid_line_artists:
            for artist in self.grid_line_artists:
                artist.remove()
            self.grid_line_artists.clear()

        if grid_enabled and grid_power > 0:
            self._draw_grid_lines(
                ax,
                spacing_power=grid_power,
                view_xlim=view_xlim,
                view_ylim=view_ylim,
                grid_color=grid_color,
            )

        # ====================================================================
        # BATCH LINE COLLECTION - Pre-allocate with estimated sizes
        # ====================================================================
        # Estimate total segments needed for better memory allocation
        estimated_segments = sum(
            max(0, len(p.points) - 1)
            for p in plots
            if getattr(p, "visible", True)
            and getattr(p, "draw_lines", False)
            and len(p.points) > 1
        )

        all_line_segments = []
        all_line_colors = []
        all_line_widths = []

        colored_line_segments = []
        colored_line_arrays = []
        colored_line_cmap = None

        # Pre-create offset array once for reuse
        offset_array = np.zeros(2, dtype=np.float32)

        # Process each plot
        for plot in plots:
            visible = getattr(plot, "visible", True)

            if not visible or len(plot.points) == 0:
                if plot.scatter_artist is not None:
                    plot.scatter_artist.set_visible(False)
                    plot.line_artist = None
                continue

            # Apply offset - reuse array to avoid allocation
            offset_array[0] = plot.offset_x
            offset_array[1] = plot.offset_y
            points = plot.points + offset_array

            # Downsample if needed
            if len(points) > max_display_points:
                step = max(1, len(points) // max_display_points)
                display_points = points[::step]
                display_colors = (
                    plot.color_data[::step] if plot.color_data is not None else None
                )
            else:
                display_points = points
                display_colors = plot.color_data

            # ================================================================
            # SCATTER ARTIST - Reuse existing artists when possible
            # ================================================================
            if display_colors is not None and len(display_colors) > 0:
                if plot.scatter_artist is None:
                    scatter = ax.scatter(
                        display_points[:, 0],
                        display_points[:, 1],
                        c=display_colors,
                        s=plot.size,
                        cmap=plot.cmap,
                        alpha=0.8,
                        rasterized=not disable_antialiasing,
                        vmin=0.0,
                        vmax=1.0,
                    )
                    plot.scatter_artist = scatter
                    scatter._pcloudviewer_managed = True
                else:
                    # Reuse existing artist - much faster than recreating
                    plot.scatter_artist.set_offsets(display_points)
                    plot.scatter_artist.set_array(display_colors)
                    plot.scatter_artist.set_sizes([plot.size])
                    plot.scatter_artist.set_cmap(plot.cmap)
                    plot.scatter_artist.set_alpha(0.8)
                    plot.scatter_artist.set_clim(0.0, 1.0)
                    plot.scatter_artist.set_visible(True)

                active_scatter_artists.add(plot.scatter_artist)
            else:
                if plot.scatter_artist is None:
                    scatter = ax.scatter(
                        display_points[:, 0],
                        display_points[:, 1],
                        c=getattr(plot, "color", "white"),
                        s=plot.size,
                        alpha=0.8,
                        rasterized=not disable_antialiasing,
                    )
                    plot.scatter_artist = scatter
                    scatter._pcloudviewer_managed = True
                else:
                    plot.scatter_artist.set_offsets(display_points)
                    plot.scatter_artist.set_sizes([plot.size])
                    plot.scatter_artist.set_facecolors(getattr(plot, "color", "white"))
                    plot.scatter_artist.set_alpha(0.8)
                    plot.scatter_artist.set_visible(True)

                active_scatter_artists.add(plot.scatter_artist)

            # ================================================================
            # LINES - BATCH - Optimized segment creation
            # ================================================================
            if plot.draw_lines and len(display_points) > 1:
                # Create line segments for this plot - optimized using numpy operations
                segments = np.stack([display_points[:-1], display_points[1:]], axis=1)

                if plot.line_color is not None:
                    # Batch extend for better performance
                    segment_count = len(segments)
                    all_line_segments.extend(segments)
                    all_line_colors.extend([plot.line_color] * segment_count)
                    all_line_widths.extend([plot.line_width] * segment_count)

                elif display_colors is not None and len(display_colors) > 0:
                    # Vectorized color averaging
                    segment_colors = (display_colors[:-1] + display_colors[1:]) * 0.5
                    colored_line_segments.extend(segments)
                    colored_line_arrays.extend(segment_colors)
                    if colored_line_cmap is None:
                        colored_line_cmap = plot.cmap

                else:
                    segment_count = len(segments)
                    all_line_segments.extend(segments)
                    all_line_colors.extend(["gray"] * segment_count)
                    all_line_widths.extend([plot.line_width] * segment_count)

            # Clear line artist reference
            if hasattr(plot, "line_artist"):
                plot.line_artist = None

        # ====================================================================
        # CREATE BATCH LINE COLLECTIONS - Only if needed
        # ====================================================================
        # Remove old batch line collections only if they exist
        if self._batch_solid_line_collection is not None:
            if self._batch_solid_line_collection in ax.collections:
                self._batch_solid_line_collection.remove()
            self._batch_solid_line_collection = None

        if self._batch_colored_line_collection is not None:
            if self._batch_colored_line_collection in ax.collections:
                self._batch_colored_line_collection.remove()
            self._batch_colored_line_collection = None

        # Create single LineCollection for all solid-color lines
        if all_line_segments:
            # Optimize linewidth handling
            unique_widths = set(all_line_widths)
            lw = all_line_widths[0] if len(unique_widths) == 1 else all_line_widths

            lc = LineCollection(
                all_line_segments,
                colors=all_line_colors,
                linewidths=lw,
                alpha=0.6,
                rasterized=not disable_antialiasing,
            )
            ax.add_collection(lc)
            self._batch_solid_line_collection = lc

        # Create single LineCollection for all colored lines
        if colored_line_segments:
            lc = LineCollection(
                colored_line_segments,
                array=np.array(colored_line_arrays),
                cmap=colored_line_cmap,
                linewidths=1.0,
                alpha=0.6,
                rasterized=not disable_antialiasing,
            )
            lc.set_clim(0.0, 1.0)
            ax.add_collection(lc)
            self._batch_colored_line_collection = lc

        # ====================================================================
        # Cleanup: Remove orphaned scatter artists - Optimized with set operations
        # ====================================================================
        # Use list comprehension for faster iteration
        orphaned_artists = [
            artist
            for artist in ax.collections
            if (
                hasattr(artist, "_pcloudviewer_managed")
                and artist not in active_scatter_artists
                and artist is not self._batch_solid_line_collection
                and artist is not self._batch_colored_line_collection
            )
        ]

        # Batch remove
        for artist in orphaned_artists:
            artist.remove()

        # Set axis limits - single call
        ax.set_xlim(*view_xlim)
        ax.set_ylim(*view_ylim)

    def _initialize_axes(
        self,
        ax: Axes,
        axes_grid_color: str,
    ) -> None:
        """Initialize or clear the axes for drawing."""
        ax.clear()
        ax.set_facecolor("black")
        ax.grid(
            True,
            color=axes_grid_color,
            alpha=0.3,
        )
        ax.tick_params(colors="white")

        # Clear batch line collection references
        self._batch_solid_line_collection = None
        self._batch_colored_line_collection = None

    def _draw_grid_lines(
        self,
        ax: Axes,
        *,
        spacing_power: int,
        view_xlim: tuple[float, float],
        view_ylim: tuple[float, float],
        grid_color: str = "#808080",
        max_lines: int = 1000,
    ) -> None:
        """
        Draw horizontal lines at 2^N spacing.

        Optimized to minimize line creation and use vectorized operations.
        """
        if spacing_power <= 0:
            return

        spacing = 2**spacing_power
        y_min, y_max = view_ylim

        # Calculate all Y positions at once using numpy
        start_y = int(np.floor(y_min / spacing)) * spacing
        num_lines = min(int(np.ceil((y_max - start_y) / spacing)) + 1, max_lines)

        # Pre-allocate grid line artists list
        self.grid_line_artists = []

        # Draw lines in batch - more efficient than loop
        y_positions = np.arange(num_lines) * spacing + start_y
        y_positions = y_positions[y_positions <= y_max]

        for y in y_positions:
            line = ax.axhline(
                y=y,
                color=grid_color,
                linewidth=1.0,
                alpha=0.6,
                zorder=0.5,
            )
            self.grid_line_artists.append(line)
