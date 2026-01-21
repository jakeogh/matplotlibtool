#!/usr/bin/env python3
# tab-width:4
# pylint: disable=no-name-in-module
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal

from .AxisSecondaryConfig import AxisSecondaryConfig
from .AxisSecondaryManagerDual import AxisSecondaryManagerDual
from .AxisType import AxisType


@dataclass
class ViewBounds:
    """Container for view boundary information."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]

    def __post_init__(self):
        """Validate bounds after initialization."""
        if self.xlim[0] >= self.xlim[1]:
            raise ValueError(
                f"xlim[0] ({self.xlim[0]}) must be less than xlim[1] ({self.xlim[1]})"
            )
        if self.ylim[0] >= self.ylim[1]:
            self.ylim = (self.ylim[0], self.ylim[1] + 1)
            #raise ValueError(
            #    f"ylim[0] ({self.ylim[0]}) must be less than ylim[1] ({self.ylim[1]})"
            #)

    @property
    def x_range(self) -> float:
        """Get X range (width)."""
        return self.xlim[1] - self.xlim[0]

    @property
    def y_range(self) -> float:
        """Get Y range (height)."""
        return self.ylim[1] - self.ylim[0]

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of the view."""
        x_center = (self.xlim[0] + self.xlim[1]) / 2
        y_center = (self.ylim[0] + self.ylim[1]) / 2
        return (x_center, y_center)

    def contains_point(
        self,
        x: float,
        y: float,
    ) -> bool:
        """Check if a point is within these bounds."""
        return self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1]

    def expand_by_ratio(self, ratio: float) -> ViewBounds:
        """Return new bounds expanded by a ratio."""
        x_pad = self.x_range * ratio
        y_pad = self.y_range * ratio
        return ViewBounds(
            xlim=(self.xlim[0] - x_pad, self.xlim[1] + x_pad),
            ylim=(self.ylim[0] - y_pad, self.ylim[1] + y_pad),
        )


class ViewManagerSignals(QObject):
    """Signal hub for view manager events."""

    # View change signals
    viewChanged = pyqtSignal()  # Emitted when view bounds change
    secondaryAxisChanged = pyqtSignal()  # Emitted when secondary axis updates


class ViewManager:
    """
    Enhanced view manager with dual secondary axis support for 2D matplotlib plots.

    Handles view bounds, zoom operations, fit-to-data, view state management,
    and secondary X/Y axis coordination.
    """

    def __init__(self, ax: Axes):
        """
        Initialize view manager.

        Args:
            ax: Matplotlib axes object
        """
        self.ax = ax
        self.signals = ViewManagerSignals()
        self.current_bounds = self._get_current_bounds()
        self.zoom_box_active = False
        self.default_pad_ratio = 0.05

        # Dual secondary axis support
        self.secondary_axis_manager = AxisSecondaryManagerDual(self.ax)

    def _get_current_bounds(self) -> ViewBounds:
        """Get current view bounds from axes."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        return ViewBounds(xlim=xlim, ylim=ylim)

    def _apply_bounds(self, bounds: ViewBounds) -> None:
        """Apply bounds to the axes and update secondary axes."""
        self.ax.set_xlim(*bounds.xlim)
        self.ax.set_ylim(*bounds.ylim)
        self.current_bounds = bounds

        # Update both secondary axes when primary view changes
        self.secondary_axis_manager.update_on_primary_change()

        # Emit signals
        self.signals.viewChanged.emit()

        # Check if any secondary axis is enabled
        if self.secondary_axis_manager.is_any_enabled():
            self.signals.secondaryAxisChanged.emit()

    def set_view_bounds(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> ViewBounds:
        """
        Set view bounds, using current bounds for any None values.

        Args:
            xlim: X limits tuple (min, max) or None to keep current
            ylim: Y limits tuple (min, max) or None to keep current

        Returns:
            New ViewBounds object

        Raises:
            ValueError: If bounds are invalid
        """
        current = self._get_current_bounds()

        new_xlim = xlim if xlim is not None else current.xlim
        new_ylim = ylim if ylim is not None else current.ylim

        new_bounds = ViewBounds(xlim=new_xlim, ylim=new_ylim)
        self._apply_bounds(new_bounds)

        return new_bounds

    def fit_to_data(
        self,
        data_points: list[np.ndarray],
        pad_ratio: float = None,
    ) -> ViewBounds:
        """
        Fit view to encompass all provided data points.

        Args:
            data_points: List of numpy arrays containing points
            pad_ratio: Padding ratio (0.05 = 5% padding), None for default

        Returns:
            New ViewBounds object
        """
        if not data_points:
            return self.current_bounds

        if pad_ratio is None:
            pad_ratio = self.default_pad_ratio

        # Find overall bounds
        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")

        for points in data_points:
            if points.size == 0:
                continue

            x_min = min(x_min, np.min(points[:, 0]))
            x_max = max(x_max, np.max(points[:, 0]))
            y_min = min(y_min, np.min(points[:, 1]))
            y_max = max(y_max, np.max(points[:, 1]))

        # Handle edge cases
        if x_min == float("inf"):
            return self.current_bounds

        # Apply padding
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Handle zero range
        if x_range == 0:
            x_range = 1.0
        if y_range == 0:
            y_range = 1.0

        x_pad = x_range * pad_ratio
        y_pad = y_range * pad_ratio

        new_bounds = ViewBounds(
            xlim=(x_min - x_pad, x_max + x_pad),
            ylim=(y_min - y_pad, y_max + y_pad),
        )

        self._apply_bounds(new_bounds)
        return new_bounds

    def apply_rectangle_zoom(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> ViewBounds:
        """
        Apply zoom to rectangle defined by two corners.

        Args:
            x1, y1: First corner coordinates
            x2, y2: Second corner coordinates

        Returns:
            New ViewBounds object
        """
        # Ensure proper min/max ordering
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        # Prevent zero-size zoom
        if x_min == x_max or y_min == y_max:
            return self.current_bounds

        new_bounds = ViewBounds(xlim=(x_min, x_max), ylim=(y_min, y_max))
        self._apply_bounds(new_bounds)
        self.zoom_box_active = False

        return new_bounds

    def set_zoom_box_active(self, active: bool) -> None:
        """Set zoom box active state."""
        self.zoom_box_active = active

    def is_zoom_box_active(self) -> bool:
        """Check if zoom box is currently active."""
        return self.zoom_box_active

    def get_current_bounds(self) -> ViewBounds:
        """Get current view bounds."""
        return self._get_current_bounds()

    def validate_bounds(
        self,
        xmin: str | None = None,
        xmax: str | None = None,
        ymin: str | None = None,
        ymax: str | None = None,
    ) -> tuple[bool, str, ViewBounds]:
        """
        Validate and parse bounds from string inputs.

        Args:
            xmin, xmax, ymin, ymax: String values or None for auto

        Returns:
            Tuple of (is_valid, error_message, parsed_bounds)
        """
        try:
            current = self._get_current_bounds()

            # Parse values, using current bounds for empty/None values
            parsed_xmin = float(xmin) if xmin and xmin.strip() else current.xlim[0]
            parsed_xmax = float(xmax) if xmax and xmax.strip() else current.xlim[1]
            parsed_ymin = float(ymin) if ymin and ymin.strip() else current.ylim[0]
            parsed_ymax = float(ymax) if ymax and ymax.strip() else current.ylim[1]

            # Validate bounds
            if parsed_xmin >= parsed_xmax:
                return False, "xmin must be less than xmax", current
            if parsed_ymin >= parsed_ymax:
                return False, "ymin must be less than ymax", current

            new_bounds = ViewBounds(
                xlim=(parsed_xmin, parsed_xmax), ylim=(parsed_ymin, parsed_ymax)
            )

            return True, "", new_bounds

        except ValueError as e:
            return False, f"Invalid number format: {e}", self._get_current_bounds()
        except Exception as e:
            return False, f"Validation error: {e}", self._get_current_bounds()

    def apply_validated_bounds(self, bounds: ViewBounds) -> ViewBounds:
        """
        Apply pre-validated bounds to the view.

        Args:
            bounds: ViewBounds object to apply

        Returns:
            The applied ViewBounds object
        """
        self._apply_bounds(bounds)
        self.zoom_box_active = False  # Clear zoom box state
        return bounds

    # ===== SECONDARY AXIS METHODS =====
    # Updated to work with AxisSecondaryManagerDual

    def configure_secondary_axis(self, config: AxisSecondaryConfig) -> None:
        """
        Configure secondary axis (X or Y based on config.axis_type).

        Args:
            config: AxisSecondaryConfig object with mapping parameters
        """
        self.secondary_axis_manager.configure_axis(config)
        self.signals.secondaryAxisChanged.emit()

        axis_name = "X" if config.axis_type == AxisType.X else "Y"
        print(
            f"[INFO] Secondary {axis_name}-axis configured: {config.label} ({config.unit})"
        )

    def disable_secondary_axis(self, axis_type: AxisType = AxisType.Y) -> None:
        """
        Disable secondary axis.

        Args:
            axis_type: Which axis to disable (default Y for backward compatibility)
        """
        self.secondary_axis_manager.disable_axis(axis_type)
        self.signals.secondaryAxisChanged.emit()

        axis_name = "X" if axis_type == AxisType.X else "Y"
        print(f"[INFO] Secondary {axis_name}-axis disabled")

    def is_secondary_axis_enabled(self, axis_type: AxisType = AxisType.Y) -> bool:
        """
        Check if secondary axis is enabled.

        Args:
            axis_type: Which axis to check (default Y for backward compatibility)
        """
        return self.secondary_axis_manager.is_axis_enabled(axis_type)

    def get_secondary_axis_config(
        self, axis_type: AxisType = AxisType.Y
    ) -> AxisSecondaryConfig | None:
        """
        Get current secondary axis configuration.

        Args:
            axis_type: Which axis to get config for (default Y for backward compatibility)
        """
        return self.secondary_axis_manager.get_axis_config(axis_type)
