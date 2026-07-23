#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np


class PlotEventHandlers:
    """Event and user-interaction handlers for the 2D matplotlib viewer."""

    def __init__(self, viewer):
        self.viewer = viewer

        # throttle keyboard scaling to avoid key-repeat render storms
        self.last_scale_update = 0.0
        self.scale_throttle_ms = 50

    def _should_throttle_scaling(self) -> bool:
        now = time.time() * 1000
        if now - self.last_scale_update < self.scale_throttle_ms:
            return True
        self.last_scale_update = now
        return False

    def on_timer(self):
        """Timer callback driving keyboard axis scaling."""
        now = time.time()
        dt = now - self.viewer.last_time
        self.viewer.last_time = now

        old_scale = self.viewer.state.scale.copy()
        self.viewer.keyboard_manager.update_scaling(dt, dimensions=2)

        if np.allclose(old_scale, self.viewer.state.scale):
            return
        if self._should_throttle_scaling():
            return
        if self.viewer.busy_manager.is_busy:
            return

        with self.viewer.busy_manager.busy_operation("Scaling"):
            self.viewer.apply_keyboard_scale()

    def on_dark_mode_toggled(self, enabled: bool):
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

    def on_acceleration_changed(self, value: float) -> None:
        self.viewer.acceleration = float(value)
        self.viewer.keyboard_manager.set_acceleration(self.viewer.acceleration)

    def on_plot_selection_changed(self, plot_index: int) -> None:
        self.viewer.plot_manager.select_plot(plot_index)
        self.viewer.control_bar_integration.sync_controls_to_selection()

    def on_group_selection_changed(self, group_id: int) -> None:
        self.viewer.plot_manager.select_group(group_id)
        self.viewer.control_bar_integration.sync_controls_to_selection()

    def _set_selected_property(self, property_name: str, value) -> None:
        if self.viewer.plot_manager.is_group_selected():
            self.viewer.plot_manager.set_group_property(
                self.viewer.plot_manager.selected_group_id,
                property_name,
                value,
            )
        else:
            self.viewer.plot_manager.set_plot_property(
                self.viewer.plot_manager.selected_plot_index,
                property_name,
                value,
            )

    def on_point_size_changed(self, value: float) -> None:
        self._set_selected_property("size", value)

    def on_line_width_changed(self, value: float) -> None:
        self._set_selected_property("line_width", value)

    def on_lines_toggled(self, checked: bool) -> None:
        self._set_selected_property("draw_lines", checked)

    def on_palette_changed(self, palette_name: str):
        if palette_name.startswith("───") or palette_name == "(Mixed)":
            return
        self._set_selected_property("colormap", palette_name)

    def on_visibility_toggled(self, visible: bool):
        if self.viewer.plot_manager.is_group_selected():
            group_id = self.viewer.plot_manager.selected_group_id
            group_info = self.viewer.plot_manager.get_group_info(group_id)
            if group_info:
                for plot_index in group_info.plot_indices:
                    self.viewer.plot_manager.set_plot_visibility(plot_index, visible)
        else:
            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_visibility(plot_index, visible)

    def on_save_figure(self):
        """Auto-save figure to /delme with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("/delme")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"autosave_figure_{timestamp}.jpg"

        with self.viewer.busy_manager.busy_operation("Saving figure"):
            self.viewer._render_to_file(filepath, dpi=300)
            print(f"[INFO] Figure auto-saved to: {filepath}")

    def on_grid_changed(self, grid_text: str):
        with self.viewer.busy_manager.busy_operation("Updating grid"):
            if "^" in grid_text:
                power = int(grid_text.split("^")[1].split()[0])
                self.viewer.grid_manager.set_grid_spacing(power, True)
            else:
                self.viewer.grid_manager.set_grid_spacing(0, False)

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def on_pick_axes_grid_color(self):
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
        with self.viewer.busy_manager.busy_operation("Fitting view to data"):
            bounds = self.viewer.fit_view(pad_ratio=0.05)
            print(
                f"[INFO] Fit view to data bounds: X({bounds.xlim[0]:.3f}, {bounds.xlim[1]:.3f}), Y({bounds.ylim[0]:.3f}, {bounds.ylim[1]:.3f})"
            )

    def reset_view(self) -> None:
        with self.viewer.busy_manager.busy_operation("Resetting view"):
            self.viewer.fit_view(pad_ratio=0.1)

    def apply_view_bounds(self):
        """Apply custom view bounds from the text fields."""
        with self.viewer.busy_manager.busy_operation("Applying view bounds"):
            xmin, xmax, ymin, ymax = self.viewer.control_bar_manager.get_view_bounds()

            is_valid, error_msg, bounds = self.viewer.view_manager.validate_bounds(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )

            if not is_valid:
                print(f"[ERROR] {error_msg}")
                return

            self.viewer.set_view(bounds.xlim, bounds.ylim)
            print(
                f"[INFO] Applied custom view bounds: X({bounds.xlim[0]:.3f}, {bounds.xlim[1]:.3f}), Y({bounds.ylim[0]:.3f}, {bounds.ylim[1]:.3f})"
            )

    def immediate_exit(self) -> None:
        print("[INFO] Exit button pressed, closing viewer.")
        self.viewer.close()

    def apply_offset_values(self):
        """Apply offset values from spinboxes to the selected plot(s) or group."""
        with self.viewer.busy_manager.busy_operation("Applying plot offset"):
            x_offset, y_offset = self.viewer.control_bar_manager.get_offset_values()

            if self.viewer.plot_manager.is_group_selected():
                group_id = self.viewer.plot_manager.selected_group_id
                self.viewer.plot_manager.set_group_property(group_id, "offset_x", x_offset)
                self.viewer.plot_manager.set_group_property(group_id, "offset_y", y_offset)

                group_info = self.viewer.plot_manager.get_group_info(group_id)
                if group_info:
                    print(
                        f"[INFO] Applied offset to group '{group_info.group_name}': ({x_offset:.3f}, {y_offset:.3f})"
                    )
            else:
                plot_index = self.viewer.plot_manager.selected_plot_index
                self.viewer.plot_manager.set_plot_property(plot_index, "offset_x", x_offset)
                self.viewer.plot_manager.set_plot_property(plot_index, "offset_y", y_offset)

                plot_info = self.viewer.plot_manager.get_plot_info(plot_index)
                if plot_info:
                    print(
                        f"[INFO] Applied offset to plot {plot_index + 1}: ({x_offset:.3f}, {y_offset:.3f})"
                    )

    def on_color_field_changed(self, field_name: str) -> None:
        """Switch the color field for the selected plot(s) or group."""
        with self.viewer.busy_manager.busy_operation(
            f"Changing color field to {field_name}"
        ):
            if self.viewer.plot_manager.is_group_selected():
                group_id = self.viewer.plot_manager.selected_group_id
                group_info = self.viewer.plot_manager.get_group_info(group_id)
                if not group_info:
                    return
                plot_indices = group_info.plot_indices
            else:
                plot_indices = [self.viewer.plot_manager.selected_plot_index]

            # group selection: shared color range across all member plots
            global_color_min = None
            global_color_max = None

            if self.viewer.plot_manager.is_group_selected():
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

            for plot_index in plot_indices:
                array_index = (
                    self.viewer.array_field_integration._get_array_index_for_plot(
                        plot_index
                    )
                )
                if array_index is None:
                    print(f"[WARNING] Could not find array for plot {plot_index}")
                    continue

                array_info = self.viewer.array_field_integration.array_field_manager.get_array_info(
                    array_index
                )
                if not array_info:
                    print(f"[WARNING] Could not get array info for array {array_index}")
                    continue

                data = array_info["data"]
                if field_name not in data.dtype.names:
                    print(
                        f"[ERROR] Field '{field_name}' not found in array {array_index}"
                    )
                    continue

                plot = self.viewer.plot_manager.plots[plot_index]
                plot.color_data = data[field_name].astype(np.float32)
                array_info["properties"]["color_field"] = field_name

                if global_color_min is not None and global_color_max is not None:
                    self.viewer.plot_manager.plot_global_color_ranges[plot_index] = (
                        global_color_min,
                        global_color_max,
                    )
                    array_info["properties"]["global_color_min"] = global_color_min
                    array_info["properties"]["global_color_max"] = global_color_max
                else:
                    self.viewer.plot_manager.plot_global_color_ranges.pop(
                        plot_index, None
                    )
                    array_info["properties"].pop("global_color_min", None)
                    array_info["properties"].pop("global_color_max", None)

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

            print(
                f"[INFO] Color field changed to '{field_name}' for {len(plot_indices)} plot(s)"
            )
