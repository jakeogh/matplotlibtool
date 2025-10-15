#!/usr/bin/env python3
# tab-width:4

"""
Plot2DRenderer.py - Pure Matplotlib/NumPy renderer for 2D point cloud visualization

OPTIMIZED VERSION WITH PERFORMANCE DEBUG PRINTS
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


class Matplotlib2DRenderer:
    """
    Pure Matplotlib/NumPy renderer for PointCloud2DViewerMatplotlib.

    OPTIMIZED with performance debugging enabled.
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
        Update all plots - WITH PERFORMANCE DEBUG PRINTS.
        """
        t_start = time.perf_counter()
        print(f"\n[PERF] ========== update_all_plots START ==========")
        print(f"[PERF] Number of plots: {len(plots)}")

        # Count visible plots and plots with lines
        visible_count = sum(
            1 for p in plots if getattr(p, "visible", True) and len(p.points) > 0
        )
        lines_count = sum(
            1
            for p in plots
            if getattr(p, "visible", True) and len(p.points) > 0 and p.draw_lines
        )
        print(f"[PERF] Visible plots: {visible_count}, with lines: {lines_count}")

        # Initialize axes styling on first call
        if not self.plot_initialized:
            self._initialize_axes(ax, axes_grid_color)
            self.plot_initialized = True

        # Track which artists should remain visible
        active_scatter_artists = set()

        # Clear and redraw grid
        t_grid_start = time.perf_counter()
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
        t_grid = time.perf_counter() - t_grid_start
        print(f"[PERF] Grid setup: {t_grid*1000:.1f}ms")

        # ====================================================================
        # BATCH LINE COLLECTION
        # ====================================================================
        all_line_segments = []
        all_line_colors = []
        all_line_widths = []

        colored_line_segments = []
        colored_line_arrays = []
        colored_line_cmap = None

        # Process each plot
        t_process_start = time.perf_counter()
        for i, plot in enumerate(plots):
            visible = getattr(
                plot,
                "visible",
                True,
            )

            if not visible or len(plot.points) == 0:
                if plot.scatter_artist is not None:
                    plot.scatter_artist.set_visible(False)
                plot.line_artist = None
                continue

            # Apply offset
            points = plot.points + np.array(
                [plot.offset_x, plot.offset_y], dtype=np.float32
            )

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
            # SCATTER ARTIST
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
            # LINES - BATCH
            # ================================================================
            if plot.draw_lines and len(display_points) > 1:
                # Create line segments for this plot
                segments = np.array(
                    [display_points[:-1], display_points[1:]]
                ).transpose(
                    1,
                    0,
                    2,
                )

                if plot.line_color is not None:
                    all_line_segments.extend(segments)
                    all_line_colors.extend([plot.line_color] * len(segments))
                    all_line_widths.extend([plot.line_width] * len(segments))

                elif display_colors is not None and len(display_colors) > 0:
                    segment_colors = (display_colors[:-1] + display_colors[1:]) / 2.0
                    colored_line_segments.extend(segments)
                    colored_line_arrays.extend(segment_colors)
                    if colored_line_cmap is None:
                        colored_line_cmap = plot.cmap

                else:
                    all_line_segments.extend(segments)
                    all_line_colors.extend(["gray"] * len(segments))
                    all_line_widths.extend([plot.line_width] * len(segments))

            plot.line_artist = None

        t_process = time.perf_counter() - t_process_start
        print(f"[PERF] Process plots loop: {t_process*1000:.1f}ms")
        print(f"[PERF] Collected {len(all_line_segments)} solid line segments")
        print(f"[PERF] Collected {len(colored_line_segments)} colored line segments")

        # ====================================================================
        # CREATE BATCH LINE COLLECTIONS
        # ====================================================================
        t_cleanup_start = time.perf_counter()

        # Remove old batch line collections
        if self._batch_solid_line_collection is not None:
            if self._batch_solid_line_collection in ax.collections:
                self._batch_solid_line_collection.remove()
            self._batch_solid_line_collection = None

        if self._batch_colored_line_collection is not None:
            if self._batch_colored_line_collection in ax.collections:
                self._batch_colored_line_collection.remove()
            self._batch_colored_line_collection = None

        t_cleanup = time.perf_counter() - t_cleanup_start
        print(f"[PERF] Cleanup old collections: {t_cleanup*1000:.1f}ms")

        # Create single LineCollection for all solid-color lines
        t_solid_start = time.perf_counter()
        if all_line_segments:
            unique_widths = set(all_line_widths)
            if len(unique_widths) == 1:
                lw = all_line_widths[0]
            else:
                lw = all_line_widths

            print(
                f"[PERF] Creating LineCollection with {len(all_line_segments)} segments..."
            )
            lc = LineCollection(
                all_line_segments,
                colors=all_line_colors,
                linewidths=lw,
                alpha=0.6,
                rasterized=not disable_antialiasing,
            )
            print(f"[PERF] LineCollection created, adding to axes...")
            ax.add_collection(lc)
            self._batch_solid_line_collection = lc
            print(f"[PERF] LineCollection added to axes")

        t_solid = time.perf_counter() - t_solid_start
        print(f"[PERF] Create solid LineCollection: {t_solid*1000:.1f}ms")

        # Create single LineCollection for all colored lines
        t_colored_start = time.perf_counter()
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

        t_colored = time.perf_counter() - t_colored_start
        print(f"[PERF] Create colored LineCollection: {t_colored*1000:.1f}ms")

        # ====================================================================
        # Cleanup: Remove orphaned scatter artists
        # ====================================================================
        t_scatter_cleanup_start = time.perf_counter()
        for artist in list(ax.collections):
            if hasattr(artist, "_pcloudviewer_managed"):
                if artist not in active_scatter_artists:
                    if (
                        artist is not self._batch_solid_line_collection
                        and artist is not self._batch_colored_line_collection
                    ):
                        artist.remove()

        t_scatter_cleanup = time.perf_counter() - t_scatter_cleanup_start
        print(f"[PERF] Scatter cleanup: {t_scatter_cleanup*1000:.1f}ms")

        # Set axis limits
        t_limits_start = time.perf_counter()
        ax.set_xlim(*view_xlim)
        ax.set_ylim(*view_ylim)
        t_limits = time.perf_counter() - t_limits_start
        print(f"[PERF] Set axis limits: {t_limits*1000:.1f}ms")

        t_total = time.perf_counter() - t_start
        print(f"[PERF] TOTAL update_all_plots: {t_total*1000:.1f}ms")
        print(f"[PERF] ========== update_all_plots END ==========\n")

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
        """Draw horizontal lines at 2^N spacing."""
        if spacing_power <= 0:
            return

        spacing = 2**spacing_power
        y_min, y_max = view_ylim

        # Draw horizontal lines
        start_y = int(np.floor(y_min / spacing)) * spacing
        y = start_y
        count = 0

        while y <= y_max and count < max_lines:
            line = ax.axhline(
                y=y,
                color=grid_color,
                linewidth=1.0,
                alpha=0.6,
                zorder=0.5,
            )
            self.grid_line_artists.append(line)
            y += spacing
            count += 1
