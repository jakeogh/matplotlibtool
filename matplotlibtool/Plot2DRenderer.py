#!/usr/bin/env python3
# tab-width:4

"""
Plot2DRenderer.py - Pure Matplotlib/NumPy renderer for 2D point cloud visualization

Updated for efficient axis scaling approach with global color range support.
No Qt imports. Uses original point coordinates - scaling handled via axis limits.
Does NOT set aspect ratio (handled by Plot2D._update_plot()).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes


class Matplotlib2DRenderer:
    """
    Pure Matplotlib/NumPy renderer for PointCloud2DViewerMatplotlib.
    Updated for efficient axis scaling approach with global color range support.

    - No Qt imports.
    - Accepts plots as dataclass-like objects (duck-typed)
    - Uses original point coordinates - scaling is handled via axis limits.
    - Does NOT set aspect ratio (handled by Plot2D._update_plot())
    - Supports global color range for consistent coloring across plot groups
    """

    def __init__(self):
        """Initialize the renderer with artist tracking."""
        self.plot_initialized = False
        self.grid_line_artists = []

    def update_all_plots(
        self,
        ax: Axes,
        *,
        plots: Sequence[object],  # List of Overlay objects
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
        Update all plots in the viewer.

        Args:
            ax: Matplotlib axes
            plots: List of plot objects (Overlay instances)
            auto_aspect: If True, use auto aspect ratio (NOT USED - kept for API compat)
            view_xlim: X axis limits
            view_ylim: Y axis limits
            grid_enabled: Whether grid is enabled
            grid_power: Grid spacing power (2^N)
            grid_color: Grid line color
            axes_grid_color: Axes grid color
            disable_antialiasing: Whether to disable antialiasing
            max_display_points: Maximum points to display per plot
            in_zoom_box: Whether currently in zoom box mode (NOT USED - kept for API compat)
        """
        # Initialize axes if needed
        if not self.plot_initialized:
            self._initialize_axes(ax, axes_grid_color)
            self.plot_initialized = True

        # Clear old artists
        ax.clear()

        # Reapply styling after clear
        # DO NOT set aspect ratio here - it's handled by Plot2D._update_plot()
        ax.set_facecolor("black")
        ax.grid(
            True,
            color=axes_grid_color,
            alpha=0.3,
        )
        ax.tick_params(colors="white")

        # Draw grid if enabled
        if grid_enabled and grid_power > 0:
            self._draw_grid_lines(
                ax,
                spacing_power=grid_power,
                view_xlim=view_xlim,
                view_ylim=view_ylim,
                grid_color=grid_color,
            )

        # Draw all plots
        for i, plot in enumerate(plots):
            visible = getattr(
                plot,
                "visible",
                True,
            )
            if not visible or len(plot.points) == 0:
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

            # Draw scatter
            if display_colors is not None and len(display_colors) > 0:
                # CRITICAL: Color data should already be normalized to [0, 1] by Plot2D._update_plot()
                # with either global or local color range applied
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
                # Store artist reference
                plot.scatter_artist = scatter
            else:
                scatter = ax.scatter(
                    display_points[:, 0],
                    display_points[:, 1],
                    c=getattr(plot, "color", "white"),
                    s=plot.size,
                    alpha=0.8,
                    rasterized=not disable_antialiasing,
                )
                plot.scatter_artist = scatter

            # Draw lines if enabled
            if plot.draw_lines and len(display_points) > 1:
                if plot.line_color is not None:
                    # Solid color line
                    lines = ax.plot(
                        display_points[:, 0],
                        display_points[:, 1],
                        color=plot.line_color,
                        linewidth=plot.line_width,
                        alpha=0.6,
                    )
                    plot.line_artist = lines[0] if lines else None
                elif display_colors is not None and len(display_colors) > 0:
                    # Colored line segments
                    from matplotlib.collections import LineCollection

                    segments = np.array(
                        [display_points[:-1], display_points[1:]]
                    ).transpose(
                        1,
                        0,
                        2,
                    )
                    segment_colors = (display_colors[:-1] + display_colors[1:]) / 2.0

                    lc = LineCollection(
                        segments,
                        array=segment_colors,
                        cmap=plot.cmap,
                        linewidths=plot.line_width,
                        alpha=0.6,
                    )
                    lc.set_clim(0.0, 1.0)
                    ax.add_collection(lc)
                    plot.line_artist = lc
                else:
                    # Default gray line
                    lines = ax.plot(
                        display_points[:, 0],
                        display_points[:, 1],
                        color="gray",
                        linewidth=plot.line_width,
                        alpha=0.6,
                    )
                    plot.line_artist = lines[0] if lines else None

        # Set limits
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

        # DO NOT set aspect ratio here - it's handled by Plot2D._update_plot()

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
            ax.axhline(
                y=y,
                color=grid_color,
                linewidth=1.0,
                alpha=0.6,
                zorder=0.5,
            )
            y += spacing
            count += 1
