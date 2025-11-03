#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import time

import numpy as np


class PlotEventHandlers:
    """
    Handles events and user interactions for the 2D matplotlib viewer.
    Updated for efficient axis scaling instead of point coordinate transformation.
    Updated for new PlotManager (no primary/overlay distinction).
    """

    def __init__(self, viewer):
        """
        Initialize event handlers with reference to the main viewer.

        Args:
            viewer: The PointCloud2DViewerMatplotlib instance
        """
        self.viewer = viewer

        # Throttling for keyboard operations to prevent key repeat issues
        self.last_scale_update = 0.0
        self.scale_throttle_ms = 50

    def _should_throttle_scaling(self) -> bool:
        """Check if scaling should be throttled to prevent key repeat issues."""
        now = time.time() * 1000
        if now - self.last_scale_update < self.scale_throttle_ms:
            return True
        self.last_scale_update = now
        return False

    def on_timer(self):
        """Timer callback for continuous updates with axis scaling."""
        now = time.time()
        dt = now - self.viewer.last_time
        self.viewer.last_time = now

        old_scale = self.viewer.state.scale.copy()
        self.viewer.keyboard_manager.update_scaling(dt, dimensions=2)

        if not np.allclose(old_scale, self.viewer.state.scale):
            if not self._should_throttle_scaling():
                if hasattr(self.viewer, "interactions") and hasattr(
                    self.viewer.interactions, "_disable_axis_scaling"
                ):
                    if self.viewer.interactions._disable_axis_scaling:
                        return

                if not self.viewer.busy_manager.is_busy:
                    with self.viewer.busy_manager.busy_operation("Scaling"):
                        self.viewer._apply_axis_scaling()
                        self.viewer.canvas.draw_idle()

    def on_dark_mode_toggled(self, enabled: bool):
        """Handle dark mode toggle."""
        self.viewer.set_dark_mode(enabled)

    def on_add_files(self):
        """Open QFileDialog for supported file types and append resulting plots."""
        paths = self.viewer.file_loader_registry.open_file_dialog(self.viewer)
        if not paths:
            return

        with self.viewer.busy_manager.busy_operation("Loading data files"):
            all_plots = self.viewer.file_loader_registry.load_files(paths)

            added = 0
            for arr in all_plots:
                self.viewer.add_plot(
                    arr,
                    x_field="sample",
                    y_field="in0",
                )
                added += 1

            if added:
                print(f"[INFO] Successfully added {added} plot(s) total")
            else:
                print("[INFO] No plots were successfully added")

    def on_plot_selection_changed(self, plot_index: int) -> None:
        """Retarget controls to the newly selected plot."""
        self.viewer.plot_manager.select_plot(plot_index)

    def on_acceleration_changed(self, value: float) -> None:
        """Handle acceleration change."""
        self.viewer.acceleration = float(value)
        self.viewer.keyboard_manager.set_acceleration(self.viewer.acceleration)

    def on_point_size_changed(self, value: float) -> None:
        """Handle point size change for selected plot."""
        plot_index = self.viewer.plot_manager.selected_plot_index
        self.viewer.plot_manager.set_plot_property(
            plot_index,
            "size",
            value,
        )

    def on_line_width_changed(self, value: float) -> None:
        """Handle line width change for selected plot(s) or group."""
        if self.viewer.plot_manager.is_group_selected():
            group_id = self.viewer.plot_manager.selected_group_id
            self.viewer.plot_manager.set_group_property(
                group_id,
                "line_width",
                value,
            )
        else:
            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_property(
                plot_index,
                "line_width",
                value,
            )

    def on_lines_toggled(self, checked: bool) -> None:
        """Handle lines toggle for selected plot."""
        plot_index = self.viewer.plot_manager.selected_plot_index
        self.viewer.plot_manager.set_plot_property(
            plot_index,
            "draw_lines",
            checked,
        )

    def on_save_figure(self):
        """Auto-save figure to /delme with timestamp."""
        from datetime import datetime
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = Path("/delme")
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"autosave_figure_{timestamp}.jpg"

        with self.viewer.busy_manager.busy_operation("Saving figure"):
            try:
                self.viewer._render_to_file(filepath, dpi=300)
                print(f"[INFO] Figure auto-saved to: {filepath}")
            except Exception as e:
                print(f"[ERROR] Failed to save figure: {e}")
                import traceback

                traceback.print_exc()

    def on_palette_changed(self, palette_name: str):
        """Handle palette change for the selected plot."""
        if palette_name.startswith("───"):
            return

        plot_index = self.viewer.plot_manager.selected_plot_index
        self.viewer.plot_manager.set_plot_property(
            plot_index,
            "colormap",
            palette_name,
        )

    def on_visibility_toggled(self, visible: bool):
        """Handle visibility toggle for the selected plot."""
        plot_index = self.viewer.plot_manager.selected_plot_index
        self.viewer.plot_manager.set_plot_visibility(plot_index, visible)

    def on_grid_changed(self, grid_text: str):
        """Handle grid spacing changes."""
        with self.viewer.busy_manager.busy_operation("Updating grid"):
            if grid_text == "Off":
                self.viewer.grid_manager.set_grid_spacing(0, False)
            else:
                if "^" in grid_text:
                    power_str = grid_text.split("^")[1].split()[0]
                    power = int(power_str)
                    self.viewer.grid_manager.set_grid_spacing(power, True)
                else:
                    self.viewer.grid_manager.set_grid_spacing(0, False)

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def on_pick_axes_grid_color(self):
        """Handle axes grid color picker."""
        new_hex = self.viewer._pick_color(self.viewer.axes_grid_color)
        if new_hex:
            self.viewer.axes_grid_color = new_hex
            self.viewer.grid_manager.set_grid_colors(
                self.viewer.grid_color, self.viewer.axes_grid_color
            )
            self.viewer.control_bar_manager.set_axes_grid_color_swatch(new_hex)
            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def on_pick_grid2n_color(self):
        """Handle ADC grid color picker."""
        new_hex = self.viewer._pick_color(self.viewer.grid_color)
        if new_hex:
            self.viewer.grid_color = new_hex
            self.viewer.grid_manager.set_grid_colors(
                self.viewer.grid_color, self.viewer.axes_grid_color
            )
            self.viewer.control_bar_manager.set_adc_grid_color_swatch(new_hex)
            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def fit_view_to_data(self):
        """Fit the view to show all data points with proper scaling."""
        with self.viewer.busy_manager.busy_operation("Fitting view to data"):
            visible_plots = self.viewer.plot_manager.get_visible_plots()

            if not visible_plots:
                print("[INFO] No visible data to fit to")
                return

            all_points = []
            for plot in visible_plots:
                offset_points = plot.points + np.array(
                    [plot.offset_x, plot.offset_y], dtype=np.float32
                )
                all_points.append(offset_points)

            new_bounds = self.viewer.view_manager.fit_to_data(
                all_points, pad_ratio=0.05
            )

            self.viewer.state.scale[:2] = 1.0
            self.viewer.state.velocity[:2] = 0.0

            self.viewer.base_xlim = new_bounds.xlim
            self.viewer.base_ylim = new_bounds.ylim

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

            print(
                f"[INFO] Fit view to data bounds: X({new_bounds.xlim[0]:.3f}, {new_bounds.xlim[1]:.3f}), Y({new_bounds.ylim[0]:.3f}, {new_bounds.ylim[1]:.3f})"
            )

    def apply_view_bounds(self):
        """Apply custom view bounds from the text fields."""
        with self.viewer.busy_manager.busy_operation("Applying view bounds"):
            xmin, xmax, ymin, ymax = self.viewer.control_bar_manager.get_view_bounds()

            is_valid, error_msg, new_bounds = self.viewer.view_manager.validate_bounds(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )

            if not is_valid:
                print(f"[ERROR] {error_msg}")
                return

            applied_bounds = self.viewer.view_manager.apply_validated_bounds(new_bounds)

            self.viewer.state.scale[:2] = 1.0
            self.viewer.state.velocity[:2] = 0.0

            self.viewer.base_xlim = applied_bounds.xlim
            self.viewer.base_ylim = applied_bounds.ylim

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

            print(
                f"[INFO] Applied custom view bounds: X({applied_bounds.xlim[0]:.3f}, {applied_bounds.xlim[1]:.3f}), Y({applied_bounds.ylim[0]:.3f}, {applied_bounds.ylim[1]:.3f})"
            )

    def apply_offset_values(self):
        """Apply offset values from spinboxes to the selected plot."""
        with self.viewer.busy_manager.busy_operation("Applying plot offset"):
            x_offset, y_offset = self.viewer.control_bar_manager.get_offset_values()

            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_property(
                plot_index,
                "offset_x",
                x_offset,
            )
            self.viewer.plot_manager.set_plot_property(
                plot_index,
                "offset_y",
                y_offset,
            )

            plot_info = self.viewer.plot_manager.get_plot_info(plot_index)
            if plot_info:
                print(
                    f"[INFO] Applied offset to plot {plot_index + 1}: ({x_offset:.3f}, {y_offset:.3f})"
                )

    def reset_view(self) -> None:
        """Reset axes limits to data bounds and reset scaling."""
        with self.viewer.busy_manager.busy_operation("Resetting view"):
            visible_plots = self.viewer.plot_manager.get_visible_plots()

            if not visible_plots:
                new_bounds = self.viewer.view_manager.set_view_bounds(
                    xlim=(0, 1), ylim=(0, 1)
                )
            else:
                all_points = []
                for plot in visible_plots:
                    offset_points = plot.points + np.array(
                        [plot.offset_x, plot.offset_y], dtype=np.float32
                    )
                    all_points.append(offset_points)

                pad_ratio = 0.1

                new_bounds = self.viewer.view_manager.reset_to_data_bounds(
                    all_points, pad_ratio
                )

            self.viewer.state.scale[:] = 1.0
            self.viewer.state.velocity[:] = 0.0

            self.viewer.base_xlim = new_bounds.xlim
            self.viewer.base_ylim = new_bounds.ylim

            self.viewer.view_manager.set_zoom_box_active(False)

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def immediate_exit(self) -> None:
        """Close the viewer window immediately."""
        print("[INFO] Exit button pressed, closing viewer.")
        self.viewer.close()

    def on_group_selection_changed(self, group_id: int) -> None:
        """Handle when a plot group is selected."""
        self.viewer.plot_manager.select_group(group_id)
        self.viewer.control_bar_integration.sync_controls_to_selection()

    def on_plot_selection_changed(self, plot_index: int) -> None:
        """Retarget controls to the newly selected plot."""
        self.viewer.plot_manager.select_plot(plot_index)
        self.viewer.control_bar_integration.sync_controls_to_selection()

    def on_point_size_changed(self, value: float) -> None:
        """Handle point size change for selected plot(s) or group."""
        if self.viewer.plot_manager.is_group_selected():
            group_id = self.viewer.plot_manager.selected_group_id
            self.viewer.plot_manager.set_group_property(
                group_id,
                "size",
                value,
            )
        else:
            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_property(
                plot_index,
                "size",
                value,
            )

    def on_lines_toggled(self, checked: bool) -> None:
        """Handle lines toggle for selected plot(s) or group."""
        if self.viewer.plot_manager.is_group_selected():
            group_id = self.viewer.plot_manager.selected_group_id
            self.viewer.plot_manager.set_group_property(
                group_id,
                "draw_lines",
                checked,
            )
        else:
            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_property(
                plot_index,
                "draw_lines",
                checked,
            )

    def on_palette_changed(self, palette_name: str):
        """Handle palette change for the selected plot(s) or group."""
        if palette_name.startswith("───"):
            return

        if palette_name == "(Mixed)":
            return

        if self.viewer.plot_manager.is_group_selected():
            group_id = self.viewer.plot_manager.selected_group_id
            self.viewer.plot_manager.set_group_property(
                group_id,
                "colormap",
                palette_name,
            )
        else:
            # Apply to single plot
            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_property(
                plot_index,
                "colormap",
                palette_name,
            )

    def on_visibility_toggled(self, visible: bool):
        """Handle visibility toggle for the selected plot(s) or group."""
        if self.viewer.plot_manager.is_group_selected():
            # Apply to all plots in group
            group_id = self.viewer.plot_manager.selected_group_id
            group_info = self.viewer.plot_manager.get_group_info(group_id)
            if group_info:
                for plot_index in group_info.plot_indices:
                    self.viewer.plot_manager.set_plot_visibility(plot_index, visible)
        else:
            # Apply to single plot
            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_visibility(plot_index, visible)

    def apply_offset_values(self):
        """Apply offset values from spinboxes to the selected plot(s) or group."""
        with self.viewer.busy_manager.busy_operation("Applying plot offset"):
            x_offset, y_offset = self.viewer.control_bar_manager.get_offset_values()

            if self.viewer.plot_manager.is_group_selected():
                # Apply to all plots in group
                group_id = self.viewer.plot_manager.selected_group_id
                self.viewer.plot_manager.set_group_property(
                    group_id,
                    "offset_x",
                    x_offset,
                )
                self.viewer.plot_manager.set_group_property(
                    group_id,
                    "offset_y",
                    y_offset,
                )

                group_info = self.viewer.plot_manager.get_group_info(group_id)
                if group_info:
                    print(
                        f"[INFO] Applied offset to group '{group_info.group_name}': ({x_offset:.3f}, {y_offset:.3f})"
                    )
            else:
                # Apply to single plot
                plot_index = self.viewer.plot_manager.selected_plot_index
                self.viewer.plot_manager.set_plot_property(
                    plot_index,
                    "offset_x",
                    x_offset,
                )
                self.viewer.plot_manager.set_plot_property(
                    plot_index,
                    "offset_y",
                    y_offset,
                )

                plot_info = self.viewer.plot_manager.get_plot_info(plot_index)
                if plot_info:
                    print(
                        f"[INFO] Applied offset to plot {plot_index + 1}: ({x_offset:.3f}, {y_offset:.3f})"
                    )

    def on_color_field_changed(self, field_name: str) -> None:
        """
        Handle color field change for selected plot(s) or group.

        Args:
            field_name: Name of the field to use for coloring
        """
        with self.viewer.busy_manager.busy_operation(
            f"Changing color field to {field_name}"
        ):
            # Get selected plot indices (single plot or all plots in group)
            if self.viewer.plot_manager.is_group_selected():
                group_id = self.viewer.plot_manager.selected_group_id
                group_info = self.viewer.plot_manager.get_group_info(group_id)
                if not group_info:
                    return
                plot_indices = group_info.plot_indices
                print(
                    f"[INFO] Changing color field to '{field_name}' for group {group_id} ({len(plot_indices)} plots)"
                )
            else:
                plot_indices = [self.viewer.plot_manager.selected_plot_index]
                print(
                    f"[INFO] Changing color field to '{field_name}' for plot {plot_indices[0]}"
                )

            # First pass: Calculate global color range across all plots if in a group
            global_color_min = None
            global_color_max = None

            if self.viewer.plot_manager.is_group_selected():
                # For groups, calculate global range across all plots
                for plot_index in plot_indices:
                    array_index = (
                        self.viewer.array_field_integration._get_array_index_for_plot(
                            plot_index
                        )
                    )
                    if array_index is None:
                        continue

                    array_info = self.viewer.array_field_integration.array_field_manager.get_array_info(
                        array_index
                    )
                    if not array_info:
                        continue

                    data = array_info["data"]
                    if field_name not in data.dtype.names:
                        continue

                    field_data = data[field_name].astype(np.float32)
                    if len(field_data) > 0:
                        local_min = float(field_data.min())
                        local_max = float(field_data.max())

                        if global_color_min is None:
                            global_color_min = local_min
                            global_color_max = local_max
                        else:
                            global_color_min = min(global_color_min, local_min)
                            global_color_max = max(global_color_max, local_max)

                print(
                    f"[INFO] Calculated global color range for group: [{global_color_min:.3f}, {global_color_max:.3f}]"
                )

            # Second pass: Update each plot
            for plot_index in plot_indices:
                array_index = (
                    self.viewer.array_field_integration._get_array_index_for_plot(
                        plot_index
                    )
                )

                if array_index is None:
                    print(f"[WARNING] Could not find array for plot {plot_index}")
                    continue

                # Get array info
                array_info = self.viewer.array_field_integration.array_field_manager.get_array_info(
                    array_index
                )
                if not array_info:
                    print(f"[WARNING] Could not get array info for array {array_index}")
                    continue

                data = array_info["data"]

                # Validate field exists
                if field_name not in data.dtype.names:
                    print(
                        f"[ERROR] Field '{field_name}' not found in array {array_index}"
                    )
                    continue

                # Extract new color data
                new_color_data = data[field_name].astype(np.float32)

                # Update the plot's color data
                plot = self.viewer.plot_manager.plots[plot_index]
                plot.color_data = new_color_data

                # Update the array properties to remember this color field
                array_info["properties"]["color_field"] = field_name

                # Update global color range for this plot
                if global_color_min is not None and global_color_max is not None:
                    # Store in plot manager's global color range mapping
                    self.viewer.plot_manager.plot_global_color_ranges[plot_index] = (
                        global_color_min,
                        global_color_max,
                    )
                    # Also store in array properties
                    array_info["properties"]["global_color_min"] = global_color_min
                    array_info["properties"]["global_color_max"] = global_color_max
                    print(
                        f"[INFO] Updated plot {plot_index} with global color range: [{global_color_min:.3f}, {global_color_max:.3f}]"
                    )
                else:
                    # For individual plots, use local range (will be calculated in _update_plot)
                    if plot_index in self.viewer.plot_manager.plot_global_color_ranges:
                        del self.viewer.plot_manager.plot_global_color_ranges[
                            plot_index
                        ]
                    array_info["properties"].pop("global_color_min", None)
                    array_info["properties"].pop("global_color_max", None)

                print(
                    f"[INFO] Updated plot {plot_index} to use color field '{field_name}'"
                )

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

            print(
                f"[INFO] Color field changed to '{field_name}' for {len(plot_indices)} plot(s)"
            )
