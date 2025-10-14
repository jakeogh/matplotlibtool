#!/usr/bin/env python3
# tab-width:4

"""
Array Field Integration Module

Handles the integration of array field management, visibility controls,
and scale factor controls into the PointCloud2DViewerMatplotlib viewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .ArrayFieldManager import ArrayFieldManager
from .ArrayFieldScaleRow import ArrayFieldScaleRow
from .ArrayFieldVisibilityRow import ArrayFieldVisibilityRow

if TYPE_CHECKING:
    from .PointCloud2DViewerMatplotlib import PointCloud2DViewerMatplotlib


class ArrayFieldIntegration:
    """
    Handles array field management integration for the 2D matplotlib viewer.

    This class coordinates between the ArrayFieldManager, ArrayFieldVisibilityRow,
    ArrayFieldScaleRow, and the viewer to enable dynamic field plotting with scaling.
    """

    def __init__(self, viewer: PointCloud2DViewerMatplotlib):
        """
        Initialize array field integration.

        Args:
            viewer: Reference to the main viewer instance
        """
        self.viewer = viewer
        self.array_field_manager: ArrayFieldManager | None = None
        self.visibility_row: ArrayFieldVisibilityRow | None = None
        self.scale_row: ArrayFieldScaleRow | None = None

        # Track which group each array belongs to
        self.array_to_group: dict[int, int] = {}  # array_index -> group_id

    def initialize(self) -> None:
        """
        Initialize the array field management system.

        This should be called after the viewer's PlotManager is ready.
        """
        self.array_field_manager = ArrayFieldManager(self.viewer.plot_manager)
        self.visibility_row = ArrayFieldVisibilityRow(self.array_field_manager)
        self.scale_row = ArrayFieldScaleRow(self.array_field_manager)

        # Connect signals
        self.visibility_row.signals.fieldToggled.connect(self.on_field_toggled)
        self.scale_row.signals.scaleChanged.connect(self.on_scale_changed)

        # Connect to plot manager selection changes
        self.viewer.plot_manager.signals.selectionChanged.connect(
            self.on_array_selection_changed
        )

    def create_visibility_widget(self):
        """
        Create the field visibility widget.

        Returns:
            QWidget containing the field visibility controls
        """
        if not self.visibility_row:
            self.initialize()

        return self.visibility_row.create_widget()

    def create_scale_widget(self):
        """
        Create the field scale factor widget.

        Returns:
            QWidget containing the scale factor controls
        """
        if not self.scale_row:
            self.initialize()

        return self.scale_row.create_widget()

    def register_array(
        self,
        data: np.ndarray,
        x_field: str,
        y_field: str,
        array_name: str | None = None,
        global_color_min: float | None = None,
        global_color_max: float | None = None,
        **properties,
    ) -> int:
        """
        Register a structured array for field management.

        Args:
            data: Structured numpy array
            x_field: Name of X-axis field
            y_field: Y-axis field to initially plot (single field)
            array_name: Optional custom name for the array
            global_color_min: Optional global minimum for color normalization
            global_color_max: Optional global maximum for color normalization
            **properties: Plot properties

        Returns:
            Array index
        """
        if not self.array_field_manager:
            self.initialize()

        # Store global color range in properties if provided
        if global_color_min is not None and global_color_max is not None:
            properties["global_color_min"] = global_color_min
            properties["global_color_max"] = global_color_max

        array_index = self.array_field_manager.register_array(
            data=data,
            x_field=x_field,
            y_field=y_field,
            array_name=array_name,
            **properties,
        )

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
        if self.array_field_manager:
            self.array_field_manager.register_field_plot(
                array_index,
                field_name,
                plot_index,
            )

    def register_array_group(
        self,
        array_index: int,
        group_id: int,
    ) -> None:
        """
        Register that an array belongs to a specific group.

        Args:
            array_index: Index of the array
            group_id: ID of the group it belongs to
        """
        self.array_to_group[array_index] = group_id

    def get_array_group_id(self, array_index: int) -> int | None:
        """
        Get the group ID for an array.

        Args:
            array_index: Index of the array

        Returns:
            Group ID or None if not in a group
        """
        return self.array_to_group.get(array_index)

    def on_field_toggled(
        self,
        array_index: int,
        field_name: str,
        checked: bool,
    ) -> None:
        """
        Handle field checkbox toggle.

        Args:
            array_index: Index of the array
            field_name: Name of the field
            checked: New checked state
        """
        if checked:
            self._add_field_plot(array_index, field_name)
        else:
            self._remove_field_plot(array_index, field_name)

    def on_scale_changed(
        self,
        array_index: int,
        field_name: str,
        scale_factor: float,
    ) -> None:
        """
        Handle scale factor change for a field.

        Args:
            array_index: Index of the array
            field_name: Name of the field
            scale_factor: New scale factor
        """
        # Get the plot index for this field
        plot_index = self.array_field_manager.get_field_plot_index(
            array_index, field_name
        )

        if plot_index is None:
            print(
                f"[WARNING] Field '{field_name}' is not currently plotted, cannot apply scale"
            )
            return

        # Get the original unscaled data from the array
        array_info = self.array_field_manager.get_array_info(array_index)
        if not array_info:
            print(f"[ERROR] Array {array_index} not found")
            return

        data = array_info["data"]
        x_field = array_info["x_field"]

        # Extract fresh unscaled Y data
        y_data_original = data[field_name].astype(np.float32)

        # Apply scale factor
        y_data_scaled = y_data_original * scale_factor

        # Extract X data
        x_data = data[x_field].astype(np.float32)

        # Create new points array with scaled Y
        points_xy = np.column_stack((x_data, y_data_scaled))

        # Get existing plot properties
        properties = array_info["properties"]

        # Get color data if specified
        color_field = properties.get("color_field", None)
        color_data = (
            data[color_field].astype(np.float32)
            if color_field is not None and color_field in data.dtype.names
            else None
        )

        # Apply coordinate transformation (same as parent array)
        transform_params = properties.get("transform_params")
        if transform_params:
            from .CoordinateTransformEngine import TransformParams

            transform_params_obj = TransformParams.from_dict(transform_params)
            transformed_points = self.viewer.transform_engine.apply_transform(
                points_xy, transform_params_obj
            )
        elif properties.get("normalize", False):
            transformed_points, params = self.viewer.transform_engine.normalize_points(
                points_xy
            )
            transform_params = params.to_dict()
        elif properties.get("center", False):
            transformed_points, params = self.viewer.transform_engine.center_points(
                points_xy
            )
            transform_params = params.to_dict()
        else:
            transformed_points, params = self.viewer.transform_engine.raw_points(
                points_xy
            )
            transform_params = params.to_dict()

        # Update the plot's points directly in PlotManager
        plot = self.viewer.plot_manager.plots[plot_index]
        plot.points = transformed_points

        print(
            f"[INFO] Applied scale factor {scale_factor:.3f} to field '{field_name}' (plot {plot_index})"
        )

        # Force update and redraw
        self.viewer._update_plot()
        self.viewer.canvas.draw_idle()

        # CRITICAL: Refresh the dropdown to show the new plot in the group
        self.viewer.control_bar_integration.refresh_plot_selector()

    def _add_field_plot(
        self,
        array_index: int,
        field_name: str,
    ) -> None:
        """
        Add a plot for a specific field.

        Args:
            array_index: Index of the array
            field_name: Name of the field
        """
        if not self.array_field_manager:
            return

        # Get array info
        array_info = self.array_field_manager.get_array_info(array_index)
        if not array_info:
            print(f"[ERROR] Array {array_index} not found")
            return

        data = array_info["data"]
        x_field = array_info["x_field"]
        properties = array_info["properties"]

        # Check if field exists in data
        if field_name not in data.dtype.names:
            print(f"[ERROR] Field '{field_name}' not found in array")
            return

        # Extract the data for this field
        x_data = data[x_field].astype(np.float32)
        y_data = data[field_name].astype(np.float32)

        # Apply scale factor if one exists
        if self.scale_row:
            scale_factor = self.scale_row.get_scale_factor(field_name)
            if scale_factor != 1.0:
                y_data = y_data * scale_factor
                print(
                    f"[INFO] Applying existing scale factor {scale_factor:.3f} to '{field_name}'"
                )

        # Create points array
        points_xy = np.column_stack((x_data, y_data))

        # Get color data if specified in properties
        color_field = properties.get("color_field", None)
        color_data = (
            data[color_field].astype(np.float32)
            if color_field is not None and color_field in data.dtype.names
            else None
        )

        # Get global color range if this array is in a group
        global_color_min = properties.get("global_color_min")
        global_color_max = properties.get("global_color_max")

        # Apply coordinate transformation (same as parent array)
        transform_params = properties.get("transform_params")
        if transform_params:
            from .CoordinateTransformEngine import TransformParams

            transform_params_obj = TransformParams.from_dict(transform_params)
            transformed_points = self.viewer.transform_engine.apply_transform(
                points_xy, transform_params_obj
            )
        elif properties.get("normalize", False):
            transformed_points, params = self.viewer.transform_engine.normalize_points(
                points_xy
            )
            transform_params = params.to_dict()
        elif properties.get("center", False):
            transformed_points, params = self.viewer.transform_engine.center_points(
                points_xy
            )
            transform_params = params.to_dict()
        else:
            transformed_points, params = self.viewer.transform_engine.raw_points(
                points_xy
            )
            transform_params = params.to_dict()

        # Add the plot directly to PlotManager (NOT through viewer.add_plot)
        with self.viewer.busy_manager.busy_operation(f"Adding field {field_name}"):
            # Add plot directly to plot manager
            plot_index = self.viewer.plot_manager.add_plot(
                points=transformed_points,
                color_data=color_data,
                colormap=properties.get("colormap", "turbo"),
                point_size=properties.get("point_size", 2.0),
                draw_lines=properties.get("draw_lines", False),
                line_color=properties.get("line_color", None),
                line_width=properties.get("line_width", 1.0),
                offset_x=properties.get("x_offset", 0.0),
                offset_y=properties.get("y_offset", 0.0),
                visible=True,
                transform_params=transform_params,
                plot_name=field_name,
                is_array_parent=False,  # Field plots are NOT array parents
                global_color_min=global_color_min,
                global_color_max=global_color_max,
            )

            # Register the field plot
            self.array_field_manager.register_field_plot(
                array_index,
                field_name,
                plot_index,
            )

            # Add this plot to the array's group if it has one
            group_id = self.get_array_group_id(array_index)
            if group_id is not None:
                group_info = self.viewer.plot_manager.get_group_info(group_id)
                if group_info:
                    # Add this plot to the group's plot_indices
                    if plot_index not in group_info.plot_indices:
                        group_info.plot_indices.append(plot_index)
                        # Update the reverse mapping
                        self.viewer.plot_manager.plot_to_group[plot_index] = group_id
                        print(f"[INFO] Added plot {plot_index} to group {group_id}")

            print(f"[INFO] Added field plot: {field_name} (plot index {plot_index})")

            # CRITICAL: Force plot update and redraw
            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

        # CRITICAL: Refresh the dropdown to show the new plot in the group
        self.viewer.control_bar_integration.refresh_plot_selector()

    def _remove_field_plot(
        self,
        array_index: int,
        field_name: str,
    ) -> None:
        """
        Remove a plot for a specific field.

        Args:
            array_index: Index of the array
            field_name: Name of the field
        """
        if not self.array_field_manager:
            return

        # Get the plot index for this field
        plot_index = self.array_field_manager.get_field_plot_index(
            array_index, field_name
        )

        if plot_index is None:
            print(f"[WARNING] Field '{field_name}' is not currently plotted")
            return

        try:
            with self.viewer.busy_manager.busy_operation(
                f"Removing field {field_name}"
            ):
                # Set plot visibility to False instead of removing it
                # This preserves the plot index mappings
                self.viewer.plot_manager.set_plot_visibility(plot_index, False)

                # Unregister the field plot
                self.array_field_manager.unregister_field_plot(array_index, field_name)

                print(
                    f"[INFO] Removed field plot: {field_name} (plot index {plot_index})"
                )

                # CRITICAL: Force plot update and redraw
                self.viewer._update_plot()
                self.viewer.canvas.draw_idle()

        except Exception as e:
            print(f"[ERROR] Failed to remove field plot '{field_name}': {e}")
            import traceback

            traceback.print_exc()

    def on_array_selection_changed(self, plot_index: int) -> None:
        """
        Handle when array selection changes in the dropdown.

        Args:
            plot_index: Index of the selected plot
        """
        # Find which array this plot belongs to
        if not self.array_field_manager or not self.visibility_row:
            return

        # Get the array index from plot_index
        array_index = self._get_array_index_for_plot(plot_index)

        if array_index is not None:
            self.visibility_row.set_current_array(array_index)
            if self.scale_row:
                self.scale_row.set_current_array(array_index)

    def _get_array_index_for_plot(self, plot_index: int) -> int | None:
        """
        Get the array index that owns a specific plot.

        Args:
            plot_index: Index of the plot

        Returns:
            Array index or None
        """
        if not self.array_field_manager:
            return None

        # Check reverse mapping
        if plot_index in self.array_field_manager.plot_to_array_field:
            array_index, _ = self.array_field_manager.plot_to_array_field[plot_index]
            return array_index

        # Fallback: assume first plot of each array is the "main" one
        return 0 if self.array_field_manager.get_array_count() > 0 else None

    def update_visibility_row(self) -> None:
        """
        Update the visibility row to reflect current state.
        """
        if self.visibility_row and self.visibility_row.current_array_index is not None:
            self.visibility_row.sync_all_checkboxes()

    def update_scale_row(self) -> None:
        """
        Update the scale row to reflect current state.
        """
        if self.scale_row and self.scale_row.current_array_index is not None:
            self.scale_row.sync_all_scale_inputs()
