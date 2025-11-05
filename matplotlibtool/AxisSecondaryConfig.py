#!/usr/bin/env python3

"""
Enhanced Secondary Axis Configuration with pint unit handling

This module provides configuration for secondary axes with automatic
unit scaling using the pint library for proper unit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

from pint import UnitRegistry

from .AxisType import AxisType

# Initialize unit registry as module-level singleton
ureg = UnitRegistry()


@dataclass
class AxisSecondaryConfig:
    """Configuration for secondary axis with pint-based unit scaling."""

    scale: float
    offset: float
    label: str = "Secondary Axis"
    unit: str = ""
    enable_auto_scale: bool = True
    axis_type: AxisType = AxisType.Y

    def __post_init__(self):
        """Initialize pint quantity if unit is specified."""
        self._ureg = ureg
        self._base_unit = None

        # Extended unit map including time units
        unit_map = {
            "V": "volt",
            "v": "volt",
            "A": "ampere",
            "a": "ampere",
            "Hz": "hertz",
            "hz": "hertz",
            "s": "second",
            "S": "second",
            "t": "second",
            "ms": "millisecond",
            "us": "microsecond",
            "µs": "microsecond",
            "ns": "nanosecond",
        }

        # Convert unit to pint-compatible name
        pint_unit = unit_map.get(self.unit, self.unit.lower())

        if pint_unit:
            self._base_unit = getattr(self._ureg, pint_unit)

    @classmethod
    def from_range_mapping(
        cls,
        primary_min: float,
        primary_max: float,
        secondary_min: float,
        secondary_max: float,
        label: str = "Secondary Axis",
        unit: str = "",
        enable_auto_scale: bool = True,
        axis_type: AxisType = AxisType.Y,
    ) -> AxisSecondaryConfig:
        """Create configuration from range mapping."""
        primary_range = primary_max - primary_min
        secondary_range = secondary_max - secondary_min

        if primary_range == 0:
            scale = 1.0
            offset = secondary_min
        else:
            scale = secondary_range / primary_range
            offset = secondary_min - scale * primary_min

        return cls(
            scale=scale,
            offset=offset,
            label=label,
            unit=unit,
            enable_auto_scale=enable_auto_scale,
            axis_type=axis_type,
        )

    @classmethod
    def from_frequency(
        cls,
        frequency: float,
        label: str = "Time",
        unit: str = "s",
        enable_auto_scale: bool = True,
        data_min: float = 0,
        axis_type: AxisType = AxisType.X,
    ) -> AxisSecondaryConfig:
        """
        Create configuration for time axis from sampling frequency.

        Args:
            frequency: Sampling frequency in Hz
            label: Axis label (default "Time")
            unit: Time unit (default "s" for seconds)
            enable_auto_scale: Enable automatic unit scaling
            data_min: Minimum data value (default 0 for sample indices)
            axis_type: Which axis to configure (default X)
        """
        time_per_sample = 1.0 / frequency

        return cls(
            scale=time_per_sample,
            offset=data_min * time_per_sample,
            label=label,
            unit=unit,
            enable_auto_scale=enable_auto_scale,
            axis_type=axis_type,
        )

    def get_display_values(
        self,
        value_min: float,
        value_max: float,
    ) -> tuple[float, float, str, float]:
        """
        Get display values with appropriate unit scaling using pint.

        Args:
            value_min: Minimum value in base units
            value_max: Maximum value in base units

        Returns:
            Tuple of (display_min, display_max, unit_string, conversion_factor)
            where conversion_factor is the multiplier from base to display units
        """
        if not self.enable_auto_scale or not self._base_unit:
            return value_min, value_max, self.unit, 1.0

        # Create pint quantities
        q_min = value_min * self._base_unit
        q_max = value_max * self._base_unit

        # Use the maximum absolute value to determine the best scale
        max_abs = max(abs(value_min), abs(value_max))
        if max_abs == 0:
            return value_min, value_max, self.unit, 1.0

        q_ref = max_abs * self._base_unit
        q_compact = q_ref.to_compact()

        # Get the compact unit
        compact_unit = q_compact.units

        # Convert both min and max to the same compact unit
        q_min_scaled = q_min.to(compact_unit)
        q_max_scaled = q_max.to(compact_unit)

        # Calculate conversion factor
        conversion_factor = q_compact.magnitude / max_abs if max_abs != 0 else 1.0

        # Format the unit string nicely
        unit_str = format(compact_unit, "~")

        return (
            q_min_scaled.magnitude,
            q_max_scaled.magnitude,
            unit_str,
            conversion_factor,
        )
