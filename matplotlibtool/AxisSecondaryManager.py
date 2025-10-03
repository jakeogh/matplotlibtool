#!/usr/bin/env python3

"""
Enhanced Secondary Axis Configuration with support for both X and Y axes.
Includes pint unit handling and automatic scaling.

FIXES:
1. Use MaxNLocator with more ticks for X-axis (nbins=10-12 instead of AutoLocator's ~6)
2. Ensure unit string appears in the axis label
3. Remove white spine line from top axis (make it invisible like other axes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

from .AxisType import AxisType

if TYPE_CHECKING:
    from .AxisSecondaryConfig import AxisSecondaryConfig


class AxisSecondaryManager:
    """Manages a single secondary axis (X or Y) with pint-based automatic unit scaling."""

    def __init__(
        self,
        primary_ax: Axes,
        axis_type: AxisType,
    ):
        self.primary_ax = primary_ax
        self.axis_type = axis_type
        self.secondary_ax: None | Axes = None
        self.config: None | AxisSecondaryConfig = None
        self._enabled = False
        self._current_unit_str = ""
        self._display_min = 0
        self._display_max = 1
        self._conversion_factor = 1.0

    def enable_secondary_axis(self, config: AxisSecondaryConfig) -> None:
        """Enable secondary axis with given configuration."""
        self.config = config

        if self.secondary_ax is None:
            # Create secondary axis based on type
            if self.axis_type == AxisType.Y:
                self.secondary_ax = self.primary_ax.twinx()
            else:  # AxisType.X
                self.secondary_ax = self.primary_ax.twiny()

        self._update_secondary_axis()
        self._enabled = True

    def disable_secondary_axis(self) -> None:
        """Disable and hide secondary axis."""
        if self.secondary_ax is not None:
            self.secondary_ax.set_visible(False)
        self._enabled = False
        self._current_unit_str = ""

    def _update_secondary_axis(self) -> None:
        """Update secondary axis limits and labels based on primary axis."""
        if not self._enabled or self.secondary_ax is None or self.config is None:
            return

        # Get primary axis limits based on axis type
        if self.axis_type == AxisType.Y:
            primary_min, primary_max = self.primary_ax.get_ylim()
        else:  # AxisType.X
            primary_min, primary_max = self.primary_ax.get_xlim()

        # Transform to secondary axis values
        secondary_min = self.config.scale * primary_min + self.config.offset
        secondary_max = self.config.scale * primary_max + self.config.offset

        if self.config.enable_auto_scale:
            try:
                # Get display values with appropriate scaling
                display_min, display_max, unit_str, conversion_factor = (
                    self.config.get_display_values(secondary_min, secondary_max)
                )

                # Store for tick formatter
                self._display_min = display_min
                self._display_max = display_max
                self._current_unit_str = unit_str
                self._conversion_factor = conversion_factor

                # Set the scaled limits based on axis type
                if self.axis_type == AxisType.Y:
                    self.secondary_ax.set_ylim(display_min, display_max)
                    # Use AutoLocator for Y-axis (works well for vertical)
                    self.secondary_ax.yaxis.set_major_locator(AutoLocator())
                else:  # AxisType.X
                    self.secondary_ax.set_xlim(display_min, display_max)
                    # FIX: Use MaxNLocator with more ticks for X-axis
                    # nbins=10 requests approximately 10 intervals (11 ticks)
                    # This gives roughly double the ticks compared to AutoLocator
                    self.secondary_ax.xaxis.set_major_locator(
                        MaxNLocator(nbins=10, prune=None)
                    )

                # Create formatter
                tick_range = display_max - display_min

                def format_tick(value, pos):
                    """Format tick with appropriate precision."""
                    if tick_range == 0:
                        return f"{value:.3f}"

                    # Calculate precision based on range
                    if tick_range > 0:
                        order = np.floor(np.log10(tick_range))
                    else:
                        order = 0

                    if order < -2:
                        precision = int(abs(order) + 2)
                    elif order < 0:
                        precision = 3
                    elif order < 1:
                        precision = 2
                    elif order < 2:
                        precision = 1
                    else:
                        precision = 0

                    if precision > 0:
                        return f"{value:.{precision}f}"
                    else:
                        return f"{value:.0f}"

                formatter = FuncFormatter(format_tick)

                if self.axis_type == AxisType.Y:
                    self.secondary_ax.yaxis.set_major_formatter(formatter)
                else:  # AxisType.X
                    self.secondary_ax.xaxis.set_major_formatter(formatter)

                # FIX: Update label with scaled unit - ensure unit appears in label
                full_label = f"{self.config.label} ({unit_str})"

            except Exception as e:
                print(f"[ERROR] Auto-scaling failed: {e}")
                import traceback

                traceback.print_exc()

                # Fall back to no scaling
                if self.axis_type == AxisType.Y:
                    self.secondary_ax.set_ylim(secondary_min, secondary_max)
                else:
                    self.secondary_ax.set_xlim(secondary_min, secondary_max)

                full_label = f"{self.config.label} ({self.config.unit})"
                self._current_unit_str = self.config.unit
                self._conversion_factor = 1.0

        else:
            # No auto-scaling
            self._current_unit_str = self.config.unit
            self._conversion_factor = 1.0

            if self.axis_type == AxisType.Y:
                self.secondary_ax.set_ylim(secondary_min, secondary_max)
                self.secondary_ax.yaxis.set_major_locator(AutoLocator())
            else:
                self.secondary_ax.set_xlim(secondary_min, secondary_max)
                # FIX: Use MaxNLocator for X-axis even without auto-scaling
                self.secondary_ax.xaxis.set_major_locator(
                    MaxNLocator(nbins=10, prune=None)
                )

            full_label = f"{self.config.label}"
            if self.config.unit:
                full_label += f" ({self.config.unit})"

        # Set label and styling based on axis type
        if self.axis_type == AxisType.Y:
            self.secondary_ax.set_ylabel(full_label, color="white")
            self.secondary_ax.tick_params(
                axis="y",
                colors="white",
                labelcolor="white",
            )
            # FIX: Make the right spine invisible (no white line)
            # Other axes don't show spine lines, so this one shouldn't either
            self.secondary_ax.spines["right"].set_visible(False)
            # Also make other spines invisible for consistency
            self.secondary_ax.spines["top"].set_visible(False)
            self.secondary_ax.spines["bottom"].set_visible(False)
            self.secondary_ax.spines["left"].set_visible(False)
        else:  # AxisType.X
            self.secondary_ax.set_xlabel(full_label, color="white")
            self.secondary_ax.tick_params(
                axis="x",
                colors="white",
                labelcolor="white",
            )
            # FIX: Make the top spine invisible (no white line)
            # Other axes don't show spine lines, so this one shouldn't either
            self.secondary_ax.spines["top"].set_visible(False)
            # Also make other spines invisible for consistency
            self.secondary_ax.spines["bottom"].set_visible(False)
            self.secondary_ax.spines["left"].set_visible(False)
            self.secondary_ax.spines["right"].set_visible(False)

        # Make visible
        self.secondary_ax.set_visible(True)

    def update_on_primary_change(self) -> None:
        """Call this when primary axis limits change."""
        if self._enabled:
            self._update_secondary_axis()

    def is_enabled(self) -> bool:
        """Check if secondary axis is enabled."""
        return self._enabled

    # def get_secondary_value(self, primary_value: float) -> float:
    #    """Convert primary axis value to secondary axis value."""
    #    if self.config is None:
    #        return primary_value
    #    return self.config.scale * primary_value + self.config.offset

    # def get_primary_value(self, secondary_value: float) -> float:
    #    """Convert secondary axis value to primary axis value."""
    #    if self.config is None:
    #        return secondary_value
    #    if self.config.scale == 0:
    #        return 0
    #    return (secondary_value - self.config.offset) / self.config.scale
