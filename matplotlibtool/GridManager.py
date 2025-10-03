#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib.axes import Axes


class GridManager:
    """
    Manages grid line rendering for matplotlib plots.

    Handles both normal axes grid and 2^N spaced horizontal grid lines
    for ADC data visualization.
    """

    def __init__(self, ax: Axes):
        """
        Initialize grid manager.

        Args:
            ax: Matplotlib axes object
        """
        self.ax = ax
        self.grid_lines: list = []
        self.grid_enabled = False
        self.grid_spacing_power = 0
        self.grid_color = "#808080"
        self.axes_grid_color = "gray"

    def set_grid_spacing(
        self,
        power: int,
        enabled: bool = True,
    ):
        """
        Set the grid spacing to 2^power.

        Args:
            power: Power of 2 for grid spacing (0 to disable)
            enabled: Whether grid is enabled
        """
        self.grid_spacing_power = power
        self.grid_enabled = enabled and power > 0

    def set_grid_colors(
        self,
        grid_color: str,
        axes_grid_color: str,
    ):
        """
        Set grid colors.

        Args:
            grid_color: Color for 2^N horizontal grid lines
            axes_grid_color: Color for normal axes grid
        """
        self.grid_color = grid_color
        self.axes_grid_color = axes_grid_color

    def clear_grid_lines(self):
        """Remove all custom grid lines."""
        for line in self.grid_lines:
            try:
                # Use the proper matplotlib method to remove lines
                if hasattr(line, "remove"):
                    line.remove()
                elif hasattr(self.ax, "lines") and line in self.ax.lines:
                    self.ax.lines.remove(line)
            except (ValueError, AttributeError, NotImplementedError):
                # Line might already be removed, invalid, or not removable
                # Try alternative removal methods
                try:
                    if hasattr(self.ax, "collections") and line in self.ax.collections:
                        self.ax.collections.remove(line)
                    elif hasattr(self.ax, "patches") and line in self.ax.patches:
                        self.ax.patches.remove(line)
                except (ValueError, AttributeError):
                    pass
        self.grid_lines.clear()

    def draw_horizontal_grid(self, max_lines: int = 1000):
        """
        Draw horizontal grid lines at 2^N spacing.

        Args:
            max_lines: Maximum number of lines to draw (safety limit)
        """
        if not self.grid_enabled or self.grid_spacing_power <= 0:
            return

        # Calculate spacing
        spacing = 2**self.grid_spacing_power

        # Get current view limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        y_min, y_max = ylim

        # Calculate line positions starting from 0 and going both directions
        line_positions = []

        # Add y=0 if it's in range
        if y_min <= 0 <= y_max:
            line_positions.append(0.0)

        # Positive lines
        y = spacing
        while y <= y_max and len(line_positions) < max_lines:
            if y >= y_min:
                line_positions.append(float(y))
            y += spacing

        # Negative lines
        y = -spacing
        while y >= y_min and len(line_positions) < max_lines:
            if y <= y_max:
                line_positions.append(float(y))
            y -= spacing

        # Debug output
        print(
            f"[GridManager] Drawing {len(line_positions)} lines with spacing 2^{self.grid_spacing_power}={spacing}"
        )
        print(f"[GridManager] Y range: {y_min:.1f} to {y_max:.1f}")
        if line_positions:
            sorted_positions = sorted(line_positions)
            print(
                f"[GridManager] Line positions: {sorted_positions[:3]} ... {sorted_positions[-3:]} (showing first/last 3)"
            )

        # Draw the lines
        for y_pos in line_positions:
            try:
                line = self.ax.axhline(
                    y=y_pos,
                    color=self.grid_color,
                    linewidth=1.0,
                    alpha=0.6,
                    zorder=0.5,  # Behind data but above background
                )
                self.grid_lines.append(line)
            except Exception as e:
                print(f"[GridManager] Failed to draw line at y={y_pos}: {e}")
                break

        print(f"[GridManager] Successfully drew {len(self.grid_lines)} grid lines")

    def setup_axes_grid(self, enabled: bool = True):
        """
        Setup standard matplotlib axes grid.

        Args:
            enabled: Whether to enable the axes grid
        """
        if enabled:
            self.ax.grid(
                True,
                color=self.axes_grid_color,
                alpha=0.3,
                linewidth=0.5,
            )
        else:
            self.ax.grid(False)

    def update_grid(
        self,
        axes_grid_enabled: bool = True,
        horizontal_grid_enabled: bool = None,
        max_lines: int = 1000,
    ):
        """
        Update all grid elements.

        Args:
            axes_grid_enabled: Whether to show standard axes grid
            horizontal_grid_enabled: Whether to show horizontal grid (None = use current setting)
            max_lines: Maximum horizontal lines to draw
        """
        # Setup axes grid
        self.setup_axes_grid(axes_grid_enabled)

        # Clear old horizontal grid lines
        self.clear_grid_lines()

        # Draw new horizontal grid if enabled
        if horizontal_grid_enabled is not None:
            self.grid_enabled = horizontal_grid_enabled and self.grid_spacing_power > 0

        if self.grid_enabled:
            self.draw_horizontal_grid(max_lines)
