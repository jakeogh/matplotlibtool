#!/usr/bin/env python3
# tab-width:4

"""
Array Field Manager

Manages the relationship between structured arrays and their field plots.
Each array can have multiple fields plotted from it, and this manager
tracks which fields are active and their corresponding plot indices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .PlotManager import PlotManager


class ArrayFieldManager:
    """
    Manages arrays and their associated field plots.

    An "array" is a structured numpy array with multiple fields.
    Each array can have multiple fields plotted, and this manager
    tracks the relationship between arrays, fields, and plots.
    """

    def __init__(self, plot_manager: PlotManager):
        """
        Initialize array field manager.

        Args:
            plot_manager: Reference to the PlotManager instance
        """
        self.plot_manager = plot_manager

        # Array tracking
        # array_index -> {'data': structured_array, 'x_field': str, 'name': str, 'properties': dict}
        self.arrays: dict[int, dict] = {}

        # Field tracking
        # array_index -> {field_name: plot_index or None}
        self.array_fields: dict[int, dict[str, int | None]] = {}

        # Reverse mapping: plot_index -> (array_index, field_name)
        self.plot_to_array_field: dict[int, tuple[int, str]] = {}

        self.next_array_index = 0

    def register_array(
        self,
        data: np.ndarray,
        x_field: str,
        y_field: str,
        array_name: str | None = None,
        **properties,
    ) -> int:
        """
        Register a structured array and its initial field plot.

        Args:
            data: Structured numpy array
            x_field: Name of X-axis field
            y_field: Y-axis field to initially plot (single field)
            array_name: Optional custom name for the array
            **properties: Plot properties (colormap, size, normalize, etc.)

        Returns:
            Array index
        """
        array_index = self.next_array_index
        self.next_array_index += 1

        # Store array metadata
        self.arrays[array_index] = {
            "data": data,
            "x_field": x_field,
            "name": array_name or f"Array {array_index + 1}",
            "properties": properties,
        }

        # Initialize field tracking for all fields in the array
        field_names = [f for f in data.dtype.names if f != x_field]
        self.array_fields[array_index] = {field: None for field in field_names}

        # Mark the initially plotted field (will be populated when plot is created)
        if y_field in self.array_fields[array_index]:
            # Will be populated via register_field_plot()
            pass

        return array_index

    def register_field_plot(
        self,
        array_index: int,
        field_name: str,
        plot_index: int,
    ) -> None:
        """
        Register that a plot has been created for a specific field.

        Args:
            array_index: Index of the array
            field_name: Name of the field
            plot_index: Index of the created plot
        """
        if array_index in self.array_fields:
            self.array_fields[array_index][field_name] = plot_index
            self.plot_to_array_field[plot_index] = (array_index, field_name)

    def get_array_fields(self, array_index: int) -> list[str]:
        """
        Get all field names for an array (excluding X field).

        Args:
            array_index: Index of the array

        Returns:
            List of field names
        """
        if array_index in self.array_fields:
            return list(self.array_fields[array_index].keys())
        return []

    def get_active_fields(self, array_index: int) -> list[str]:
        """
        Get currently plotted field names for an array.

        Args:
            array_index: Index of the array

        Returns:
            List of field names that are currently plotted
        """
        if array_index in self.array_fields:
            return [
                field
                for field, plot_idx in self.array_fields[array_index].items()
                if plot_idx is not None
            ]
        return []

    def is_field_active(
        self,
        array_index: int,
        field_name: str,
    ) -> bool:
        """
        Check if a field is currently plotted.

        Args:
            array_index: Index of the array
            field_name: Name of the field

        Returns:
            True if field is plotted
        """
        if array_index in self.array_fields:
            return self.array_fields[array_index].get(field_name) is not None
        return False

    def get_field_plot_index(
        self,
        array_index: int,
        field_name: str,
    ) -> int | None:
        """
        Get the plot index for a specific field.

        Args:
            array_index: Index of the array
            field_name: Name of the field

        Returns:
            Plot index or None if not plotted
        """
        if array_index in self.array_fields:
            return self.array_fields[array_index].get(field_name)
        return None

    def get_array_info(self, array_index: int) -> dict | None:
        """
        Get array metadata.

        Args:
            array_index: Index of the array

        Returns:
            Dictionary with array info or None
        """
        return self.arrays.get(array_index)

    def get_array_name(self, array_index: int) -> str | None:
        """
        Get array name.

        Args:
            array_index: Index of the array

        Returns:
            Array name or None
        """
        if array_index in self.arrays:
            return self.arrays[array_index]["name"]
        return None

    def unregister_field_plot(
        self,
        array_index: int,
        field_name: str,
    ) -> int | None:
        """
        Mark a field as no longer plotted.

        Args:
            array_index: Index of the array
            field_name: Name of the field

        Returns:
            The plot index that was removed, or None
        """
        if array_index in self.array_fields:
            plot_index = self.array_fields[array_index].get(field_name)
            if plot_index is not None:
                self.array_fields[array_index][field_name] = None
                if plot_index in self.plot_to_array_field:
                    del self.plot_to_array_field[plot_index]
                return plot_index
        return None

    def get_array_count(self) -> int:
        """Get total number of registered arrays."""
        return len(self.arrays)
