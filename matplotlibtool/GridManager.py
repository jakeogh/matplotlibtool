#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


class GridManager:
    """
    Grid rendering: standard axes grid plus 2^N-spaced horizontal lines
    for ADC data visualization.
    """

    def __init__(self, ax: Axes):
        self.ax = ax
        self.grid_lines: list = []
        self.grid_enabled = False
        self.grid_spacing_power = 0
        self.grid_color = "#808080"
        self.axes_grid_color = "gray"

    def set_grid_spacing(self, power: int, enabled: bool = True) -> None:
        self.grid_spacing_power = power
        self.grid_enabled = enabled and power > 0

    def set_grid_colors(self, grid_color: str, axes_grid_color: str) -> None:
        self.grid_color = grid_color
        self.axes_grid_color = axes_grid_color

    def clear_grid_lines(self) -> None:
        for line in self.grid_lines:
            line.remove()
        self.grid_lines.clear()

    def draw_horizontal_grid(self, max_lines: int = 1000) -> None:
        if not self.grid_enabled or self.grid_spacing_power <= 0:
            return

        spacing = 2**self.grid_spacing_power
        y_min, y_max = self.ax.get_ylim()

        first = int(np.ceil(y_min / spacing))
        last = int(np.floor(y_max / spacing))
        positions = np.arange(first, last + 1, dtype=np.float64) * spacing
        positions = positions[:max_lines]

        for y_pos in positions:
            line = self.ax.axhline(
                y=y_pos,
                color=self.grid_color,
                linewidth=1.0,
                alpha=0.6,
                zorder=0.5,
            )
            self.grid_lines.append(line)

    def setup_axes_grid(self, enabled: bool = True) -> None:
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
        horizontal_grid_enabled: bool | None = None,
        max_lines: int = 1000,
    ) -> None:
        self.setup_axes_grid(axes_grid_enabled)
        self.clear_grid_lines()

        if horizontal_grid_enabled is not None:
            self.grid_enabled = horizontal_grid_enabled and self.grid_spacing_power > 0

        if self.grid_enabled:
            self.draw_horizontal_grid(max_lines)
