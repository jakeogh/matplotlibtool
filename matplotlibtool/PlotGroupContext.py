#!/usr/bin/env python3
# tab-width:4

"""
Plot Group Context Manager

Manages a group of plots with shared global color mapping.
Accumulates plots during the context, then applies global color range on exit.
Registers the group with PlotManager for group-level operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .Plot2D import Plot2D


class PlotGroupContext:
    """
    Context manager for adding multiple plots with shared global color range.

    Usage:
        with viewer.plot_group(color_field='frame', group_name='My Data') as group:
            group.add_plot(data1, x_field='x', y_field='y', color_field='frame')
            group.add_plot(data2, x_field='x', y_field='y', color_field='frame')
        # On exit, all plots are rendered with consistent color mapping
        # and registered as a group for group-level operations
    """

    def __init__(
        self,
        viewer: Plot2D,
        color_field: str,
        group_name: str | None = None,
    ):
        """
        Initialize plot group context.

        Args:
            viewer: Reference to the Plot2D viewer
            color_field: Name of the field to use for global color mapping
            group_name: Optional custom name for the group (auto-generated if None)
        """
        self.viewer = viewer
        self.color_field = color_field
        self.group_name = group_name

        # Track accumulated plot data
        self.accumulated_plots: list[dict] = []

        # Track global color range
        self.global_color_min: float | None = None
        self.global_color_max: float | None = None

        # Track which plot indices belong to this group
        self.group_plot_indices: list[int] = []

        # Store original busy manager state
        self._original_busy_state = None

        # Track fields used in this group (for validation)
        self._group_fields: set[str] | None = None

    def __enter__(self) -> PlotGroupContext:
        """Enter the plot group context."""
        print(f"[INFO] Starting plot group with color_field='{self.color_field}'")

        # Disable rendering during accumulation
        self._original_busy_state = self.viewer.busy_manager.is_busy

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        """Exit the plot group context and apply global color mapping."""
        if exc_type is not None:
            # Exception occurred, don't render
            print(f"[ERROR] Plot group context exited with exception: {exc_val}")
            return False

        # Calculate global color range
        if not self.accumulated_plots:
            print("[INFO] No plots accumulated in plot group")
            return

        print(f"[INFO] Finalizing plot group: {len(self.accumulated_plots)} plots")
        print(
            f"[INFO] Global color range: [{self.global_color_min:.3f}, {self.global_color_max:.3f}]"
        )

        # Generate group name if not provided
        if self.group_name is None:
            group_count = len(self.viewer.plot_manager.plot_groups)
            self.group_name = f"Group {group_count + 1}"

        # Now add all plots with global color range
        with self.viewer.busy_manager.busy_operation("Adding plot group"):
            for plot_data in self.accumulated_plots:
                plot_index = self._add_plot_with_global_range(plot_data)
                self.group_plot_indices.append(plot_index)

        # Register the group with PlotManager
        group_id = self.viewer.plot_manager.register_plot_group(
            group_name=self.group_name,
            plot_indices=self.group_plot_indices,
            color_field=self.color_field,
            color_range=(self.global_color_min, self.global_color_max),
        )

        # Final render
        self.viewer._update_plot()
        self.viewer.canvas.draw_idle()

        # Update UI
        self.viewer.control_bar_integration.refresh_plot_selector()
        self.viewer.control_bar_integration.sync_controls_to_selection()

        print(
            f"[INFO] Plot group '{self.group_name}' finalized: {len(self.accumulated_plots)} plots added (group_id={group_id})"
        )

    def add_plot(
        self,
        data,
        *,
        x_field: str,
        y_field: str,
        normalize: bool = False,
        center: bool = False,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        colormap: str = "turbo",
        point_size: float = 2.0,
        draw_lines: bool = False,
        line_color: str | None = None,
        line_width: float = 1.0,
        visible: bool = True,
        transform_params: dict | None = None,
        plot_name: str | None = None,
        color_field: str | None = None,
    ) -> None:
        """
        Add a plot to the group (deferred rendering).

        Args:
            data: Structured array
            x_field: Name of field to use for X axis
            y_field: Name of field to use for Y axis
            normalize: If True, normalize points to unit square
            center: If True, center points at origin
            x_offset: X offset for the plot
            y_offset: Y offset for the plot
            colormap: Colormap for the plot
            point_size: Point size for the plot
            draw_lines: Whether to draw lines between points
            line_color: Color for lines (None = use point colors)
            line_width: Width of lines
            visible: Whether plot is initially visible
            transform_params: Optional transform parameters
            plot_name: Optional custom name for the plot
            color_field: Optional field to use for coloring (overrides group color_field)
        """
        # Use group's color_field if not specified
        if color_field is None:
            color_field = self.color_field

        # Validate that color_field exists in data
        if color_field not in data.dtype.names:
            raise ValueError(
                f"Color field '{color_field}' not found in data. Available: {data.dtype.names}"
            )

        # Validate fields match within group
        current_fields = set(data.dtype.names)
        if self._group_fields is None:
            # First plot in group - establish the field set
            self._group_fields = current_fields
        else:
            # Subsequent plots - must match first plot's fields
            if current_fields != self._group_fields:
                missing = self._group_fields - current_fields
                extra = current_fields - self._group_fields
                error_msg = f"All plots in a group must have the same fields.\n"
                if missing:
                    error_msg += f"  Missing fields: {missing}\n"
                if extra:
                    error_msg += f"  Extra fields: {extra}\n"
                error_msg += f"  Expected fields: {sorted(self._group_fields)}\n"
                error_msg += f"  Got fields: {sorted(current_fields)}"
                raise ValueError(error_msg)

        # Extract color data and update global range
        color_data = data[color_field].astype(np.float32)

        if len(color_data) > 0:
            local_min = float(color_data.min())
            local_max = float(color_data.max())

            if self.global_color_min is None:
                self.global_color_min = local_min
                self.global_color_max = local_max
            else:
                self.global_color_min = min(self.global_color_min, local_min)
                self.global_color_max = max(self.global_color_max, local_max)

        # Store plot data for later rendering
        plot_data = {
            "data": data,
            "x_field": x_field,
            "y_field": y_field,
            "color_field": color_field,
            "normalize": normalize,
            "center": center,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "colormap": colormap,
            "point_size": point_size,
            "draw_lines": draw_lines,
            "line_color": line_color,
            "line_width": line_width,
            "visible": visible,
            "transform_params": transform_params,
            "plot_name": plot_name,
        }

        self.accumulated_plots.append(plot_data)

        print(
            f"[DEBUG] Accumulated plot {len(self.accumulated_plots)}: {plot_name or y_field} "
            f"(color range: [{local_min:.3f}, {local_max:.3f}])"
        )

    def _add_plot_with_global_range(self, plot_data: dict) -> int:
        """
        Add a plot using the global color range.

        Args:
            plot_data: Dictionary containing plot parameters

        Returns:
            The plot index that was added
        """
        data = plot_data["data"]
        x_field = plot_data["x_field"]
        y_field = plot_data["y_field"]
        color_field = plot_data["color_field"]

        # Register array with field manager
        array_index = self.viewer.array_field_integration.register_array(
            data=data,
            x_field=x_field,
            y_field=y_field,
            array_name=plot_data["plot_name"],
            normalize=plot_data["normalize"],
            center=plot_data["center"],
            x_offset=plot_data["x_offset"],
            y_offset=plot_data["y_offset"],
            colormap=plot_data["colormap"],
            point_size=plot_data["point_size"],
            draw_lines=plot_data["draw_lines"],
            line_color=plot_data["line_color"],
            line_width=plot_data["line_width"],
            visible=plot_data["visible"],
            transform_params=plot_data["transform_params"],
            color_field=color_field,
            global_color_min=self.global_color_min,
            global_color_max=self.global_color_max,
        )

        # Extract X and Y data
        x_data = data[x_field].astype(np.float32)
        y_data = data[y_field].astype(np.float32)
        points_xy = np.column_stack((x_data, y_data))

        # Extract color data
        color_data = data[color_field].astype(np.float32)

        # Apply coordinate transformation
        transform_params = plot_data["transform_params"]
        if transform_params is not None:
            from .CoordinateTransformEngine import TransformParams

            transform_params_obj = TransformParams.from_dict(transform_params)
            transformed_points = self.viewer.transform_engine.apply_transform(
                points_xy, transform_params_obj
            )
            result_transform_params = transform_params
        elif plot_data["normalize"]:
            transformed_points, params = self.viewer.transform_engine.normalize_points(
                points_xy
            )
            result_transform_params = params.to_dict()
        elif plot_data["center"]:
            transformed_points, params = self.viewer.transform_engine.center_points(
                points_xy
            )
            result_transform_params = params.to_dict()
        else:
            transformed_points, params = self.viewer.transform_engine.raw_points(
                points_xy
            )
            result_transform_params = params.to_dict()

        # Generate plot name
        plot_name = plot_data["plot_name"] or y_field

        # Add plot to PlotManager with global color range
        plot_index = self.viewer.plot_manager.add_plot(
            points=transformed_points,
            color_data=color_data,
            colormap=plot_data["colormap"],
            point_size=plot_data["point_size"],
            draw_lines=plot_data["draw_lines"],
            line_color=plot_data["line_color"],
            line_width=plot_data["line_width"],
            offset_x=plot_data["x_offset"],
            offset_y=plot_data["y_offset"],
            visible=plot_data["visible"],
            transform_params=result_transform_params,
            plot_name=plot_name,
            is_array_parent=True,
            global_color_min=self.global_color_min,
            global_color_max=self.global_color_max,
        )

        # Register field plot
        self.viewer.array_field_integration.register_field_plot(
            array_index,
            y_field,
            plot_index,
        )

        return plot_index
