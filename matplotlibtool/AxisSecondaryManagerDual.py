#!/usr/bin/env python3

"""
Dual Secondary Axis Manager for managing both X and Y secondary axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.axes import Axes

from .AxisSecondaryManager import AxisSecondaryManager
from .AxisType import AxisType

if TYPE_CHECKING:
    from .AxisSecondaryConfig import AxisSecondaryConfig


class AxisSecondaryManagerDual:
    """Manages both X and Y secondary axes for a matplotlib plot."""

    def __init__(self, primary_ax: Axes):
        """Initialize dual secondary axis manager."""
        self.primary_ax = primary_ax
        self.x_axis_manager = AxisSecondaryManager(primary_ax, AxisType.X)
        self.y_axis_manager = AxisSecondaryManager(primary_ax, AxisType.Y)

    def configure_axis(self, config: AxisSecondaryConfig) -> None:
        """Configure the appropriate secondary axis based on config.axis_type."""
        if config.axis_type == AxisType.Y:
            self.y_axis_manager.enable_secondary_axis(config)
        else:
            self.x_axis_manager.enable_secondary_axis(config)

    def disable_axis(self, axis_type: AxisType) -> None:
        """Disable a specific secondary axis."""
        if axis_type == AxisType.Y:
            self.y_axis_manager.disable_secondary_axis()
        else:
            self.x_axis_manager.disable_secondary_axis()

    def update_on_primary_change(self) -> None:
        """Update both secondary axes when primary axes change."""
        self.x_axis_manager.update_on_primary_change()
        self.y_axis_manager.update_on_primary_change()

    def is_axis_enabled(self, axis_type: AxisType) -> bool:
        """Check if a specific axis is enabled."""
        if axis_type == AxisType.Y:
            return self.y_axis_manager.is_enabled()
        else:
            return self.x_axis_manager.is_enabled()

    def get_axis_config(self, axis_type: AxisType) -> AxisSecondaryConfig | None:
        """Get configuration for a specific axis."""
        if axis_type == AxisType.Y:
            return self.y_axis_manager.config
        else:
            return self.x_axis_manager.config

    def is_any_enabled(self) -> bool:
        """Check if any secondary axis is enabled."""
        return self.x_axis_manager.is_enabled() or self.y_axis_manager.is_enabled()
