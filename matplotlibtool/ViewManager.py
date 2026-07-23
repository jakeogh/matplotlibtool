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


@dataclass(frozen=True)
class ViewBounds:
    """View boundary container. Invalid bounds are a hard error."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]

    def __post_init__(self):
        if not (self.xlim[0] < self.xlim[1]):
            raise ValueError(f"xlim {self.xlim}: min must be < max")
        if not (self.ylim[0] < self.ylim[1]):
            raise ValueError(f"ylim {self.ylim}: min must be < max")

    @property
    def x_range(self) -> float:
        return self.xlim[1] - self.xlim[0]

    @property
    def y_range(self) -> float:
        return self.ylim[1] - self.ylim[0]


class ViewManagerSignals(QObject):
    viewChanged = pyqtSignal()
    secondaryAxisChanged = pyqtSignal()


class ViewManager:
    """Owns view bounds and secondary X/Y axis coordination for one Axes."""

    def __init__(self, ax: Axes):
        self.ax = ax
        self.signals = ViewManagerSignals()
        self.secondary_axis_manager = AxisSecondaryManagerDual(self.ax)

    def get_current_bounds(self) -> ViewBounds:
        return ViewBounds(xlim=self.ax.get_xlim(), ylim=self.ax.get_ylim())

    def apply(self, bounds: ViewBounds) -> None:
        """Apply bounds to the axes and propagate to secondary axes."""
        self.ax.set_xlim(*bounds.xlim)
        self.ax.set_ylim(*bounds.ylim)

        self.secondary_axis_manager.update_on_primary_change()
        self.signals.viewChanged.emit()
        if self.secondary_axis_manager.is_any_enabled():
            self.signals.secondaryAxisChanged.emit()

    def set_view_bounds(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> ViewBounds:
        """Apply bounds, keeping the current value for any None axis."""
        current = self.get_current_bounds()
        bounds = ViewBounds(
            xlim=xlim if xlim is not None else current.xlim,
            ylim=ylim if ylim is not None else current.ylim,
        )
        self.apply(bounds)
        return bounds

    @staticmethod
    def compute_fit_bounds(
        data_points: list[np.ndarray],
        pad_ratio: float = 0.05,
    ) -> ViewBounds | None:
        """Bounds enclosing all points, padded. None if there are no points."""
        nonempty = [p for p in data_points if p.size]
        if not nonempty:
            return None

        x_min = min(float(np.min(p[:, 0])) for p in nonempty)
        x_max = max(float(np.max(p[:, 0])) for p in nonempty)
        y_min = min(float(np.min(p[:, 1])) for p in nonempty)
        y_max = max(float(np.max(p[:, 1])) for p in nonempty)

        x_range = (x_max - x_min) or 1.0
        y_range = (y_max - y_min) or 1.0
        x_pad = x_range * pad_ratio
        y_pad = y_range * pad_ratio
        # zero-extent axes still need a nonzero window
        if x_max == x_min:
            x_pad = max(x_pad, 0.5)
        if y_max == y_min:
            y_pad = max(y_pad, 0.5)

        return ViewBounds(
            xlim=(x_min - x_pad, x_max + x_pad),
            ylim=(y_min - y_pad, y_max + y_pad),
        )

    def validate_bounds(
        self,
        xmin: str | None = None,
        xmax: str | None = None,
        ymin: str | None = None,
        ymax: str | None = None,
    ) -> tuple[bool, str, ViewBounds]:
        """Parse user-typed bounds, falling back to current values for blanks."""
        current = self.get_current_bounds()

        try:
            parsed_xmin = float(xmin) if xmin and xmin.strip() else current.xlim[0]
            parsed_xmax = float(xmax) if xmax and xmax.strip() else current.xlim[1]
            parsed_ymin = float(ymin) if ymin and ymin.strip() else current.ylim[0]
            parsed_ymax = float(ymax) if ymax and ymax.strip() else current.ylim[1]
        except ValueError as e:
            return False, f"Invalid number format: {e}", current

        if parsed_xmin >= parsed_xmax:
            return False, "xmin must be less than xmax", current
        if parsed_ymin >= parsed_ymax:
            return False, "ymin must be less than ymax", current

        return (
            True,
            "",
            ViewBounds(xlim=(parsed_xmin, parsed_xmax), ylim=(parsed_ymin, parsed_ymax)),
        )

    # ===== secondary axis passthrough =====

    def configure_secondary_axis(self, config: AxisSecondaryConfig) -> None:
        self.secondary_axis_manager.configure_axis(config)
        self.signals.secondaryAxisChanged.emit()
        axis_name = "X" if config.axis_type == AxisType.X else "Y"
        print(f"[INFO] Secondary {axis_name}-axis configured: {config.label} ({config.unit})")

    def disable_secondary_axis(self, axis_type: AxisType = AxisType.Y) -> None:
        self.secondary_axis_manager.disable_axis(axis_type)
        self.signals.secondaryAxisChanged.emit()

    def is_secondary_axis_enabled(self, axis_type: AxisType = AxisType.Y) -> bool:
        return self.secondary_axis_manager.is_axis_enabled(axis_type)

    def get_secondary_axis_config(
        self, axis_type: AxisType = AxisType.Y
    ) -> AxisSecondaryConfig | None:
        return self.secondary_axis_manager.get_axis_config(axis_type)
