#!/usr/bin/env python3
# tab-width:4

"""
Control Bar Integration Module for PointCloud2DViewerMatplotlib

This module handles the integration between the control bar UI and the viewer,
including signal connections, control synchronization, and state updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict

if TYPE_CHECKING:
    from .PointCloud2DViewerMatplotlib import PointCloud2DViewerMatplotlib


class ControlBarIntegration:
    """
    Manages control bar integration for the 2D matplotlib viewer.

    This class handles:
    - Signal connections between UI and viewer
    - Control state synchronization
    - Plot selector updates
    - View bounds display updates
    - Color field dropdown synchronization
    """

    def __init__(self, viewer: PointCloud2DViewerMatplotlib):
        """
        Initialize control bar integration.

        Args:
            viewer: Reference to the main viewer instance
        """
        self.viewer = viewer

    def connect_signals(self) -> None:
        """Connect all control bar signals to their handlers."""
        signal_map = {
            # File operations
            "addRequested": self.viewer.event_handlers.on_add_files,
            "saveFigureRequested": self.viewer.event_handlers.on_save_figure,
            # Plot selection and properties
            "plotChanged": self.viewer.event_handlers.on_plot_selection_changed,
            "groupSelectionChanged": self.viewer.event_handlers.on_group_selection_changed,
            "visibilityToggled": self.viewer.event_handlers.on_visibility_toggled,
            # Rendering controls
            "accelChanged": self.viewer.event_handlers.on_acceleration_changed,
            "sizeChanged": self.viewer.event_handlers.on_point_size_changed,
            "lineWidthChanged": self.viewer.event_handlers.on_line_width_changed,
            "linesToggled": self.viewer.event_handlers.on_lines_toggled,
            "paletteChanged": self.viewer.event_handlers.on_palette_changed,
            "colorFieldChanged": self.viewer.event_handlers.on_color_field_changed,
            "darkModeToggled": self.viewer.event_handlers.on_dark_mode_toggled,
            # Grid controls
            "gridSpacingChanged": self.viewer.event_handlers.on_grid_changed,
            "axesGridColorPickRequested": self.viewer.event_handlers.on_pick_axes_grid_color,
            "adcGridColorPickRequested": self.viewer.event_handlers.on_pick_grid2n_color,
            # View controls
            "resetRequested": self.viewer.event_handlers.reset_view,
            "exitRequested": self.viewer.event_handlers.immediate_exit,
            "fitViewRequested": self.viewer.event_handlers.fit_view_to_data,
            "applyViewRequested": self.viewer.event_handlers.apply_view_bounds,
            "applyOffsetRequested": self.viewer.event_handlers.apply_offset_values,
        }

        # Add secondary axis signals from the integration module
        if hasattr(self.viewer, "secondary_axis"):
            signal_map.update(self.viewer.secondary_axis.connect_signals())

        self.viewer.control_bar_manager.connect_signals(signal_map)

    def sync_controls_to_selection(self) -> None:
        """Synchronize control values to currently selected plot(s) or group."""
        if not self._has_control_bar():
            return

        props = self.viewer.plot_manager.get_selected_plot_properties()
        if not props:
            return

        # Sync basic plot properties (handles "mixed" values)
        self._sync_plot_properties(props)

        # Sync grid colors
        self._sync_grid_colors()

        # Sync color field dropdown
        self._sync_color_field_dropdown()

        # Sync secondary axis state
        if hasattr(self.viewer, "secondary_axis"):
            self.viewer.secondary_axis.sync_ui_state()

        # Update view bounds display
        self.update_view_bounds_display()

    def _sync_plot_properties(self, props: dict[str, Any]) -> None:
        """Sync plot-specific properties to controls (handles mixed values)."""
        manager = self.viewer.control_bar_manager

        if props["size"] == "mixed":
            manager.set_point_size_mixed()
        else:
            manager.set_point_size(props["size"])

        if props.get("line_width") == "mixed":
            manager.set_line_width_mixed()
        elif "line_width" in props:
            manager.set_line_width(props["line_width"])

        if props["draw_lines"] == "mixed":
            manager.set_lines_tristate()
        else:
            manager.set_lines_checked(props["draw_lines"])

        if props["colormap"] == "mixed":
            manager.set_selected_palette_mixed()
        else:
            manager.set_selected_palette(props["colormap"])

        manager.set_palette_enabled(props["has_color_data"] not in [False, "mixed"])

        if props["offset_x"] == "mixed" or props["offset_y"] == "mixed":
            manager.set_offset_mixed()
        else:
            manager.set_offset(props["offset_x"], props["offset_y"])

        if props["visible"] == "mixed":
            manager.set_visibility_tristate()
        else:
            manager.set_visibility_checked(props["visible"])

    def _sync_color_field_dropdown(self) -> None:
        """Synchronize color field dropdown with current selection."""
        if not self._has_control_bar():
            return

        # Check if we have a single plot selected or group
        if self.viewer.plot_manager.is_group_selected():
            group_id = self.viewer.plot_manager.selected_group_id
            group_info = self.viewer.plot_manager.get_group_info(group_id)
            if not group_info or not group_info.plot_indices:
                return

            plot_index = group_info.plot_indices[0]
        else:
            plot_index = self.viewer.plot_manager.selected_plot_index

        # Get the array index for this plot to find available fields
        array_index = self.viewer.array_field_integration._get_array_index_for_plot(
            plot_index
        )

        if (
            array_index is not None
            and self.viewer.array_field_integration.array_field_manager
        ):
            array_info = (
                self.viewer.array_field_integration.array_field_manager.get_array_info(
                    array_index
                )
            )
            if array_info:
                data = array_info["data"]
                field_names = list(data.dtype.names)

                current_color_field = array_info["properties"].get("color_field")

                self.viewer.control_bar_manager.populate_color_field_combo(
                    field_names, current_color_field
                )
                return

        # No valid selection - clear dropdown
        self.viewer.control_bar_manager.populate_color_field_combo([], None)

    def refresh_plot_selector(self) -> None:
        """Update plot selector combobox with current arrays and groups."""
        self.viewer.control_bar_manager.populate_hierarchical_dropdown(
            self.viewer.plot_manager
        )

        combo = self.viewer.control_bar_manager.get_widget("plot_combo")
        combo.update()
        combo.repaint()

    def update_view_bounds_display(self) -> None:
        """Update the view bounds text fields with current values."""
        if not self._has_control_bar():
            return

        current_bounds = self.viewer.view_manager.get_current_bounds()
        self.viewer.control_bar_manager.set_view_bounds(
            current_bounds.xlim[0],
            current_bounds.xlim[1],
            current_bounds.ylim[0],
            current_bounds.ylim[1],
        )

    def set_initial_state(self) -> None:
        """Set initial control states after UI is created."""
        self.refresh_plot_selector()

        # Sync controls to current selection (if any plots exist)
        if self.viewer.plot_manager.get_plot_count() > 0:
            self.sync_controls_to_selection()

        self.viewer.control_bar_manager.set_accel(self.viewer.acceleration)

        self.viewer.control_bar_manager.set_dark_mode_checked(self.viewer.dark_mode)

    def update_info_text(self, text: str) -> None:
        """
        Update the info label text.

        Args:
            text: Text to display in info label
        """
        if self._has_control_bar():
            self.viewer.control_bar_manager.set_info_text(text)

    def update_point_count(self, count: int) -> None:
        """
        Update point count display.

        Args:
            count: Number of points to display
        """
        self.update_info_text(f"{count:,} pts")

    def _has_control_bar(self) -> bool:
        """
        Check if control bar manager is initialized.

        Returns:
            True if control bar manager exists
        """
        return (
            hasattr(self.viewer, "control_bar_manager")
            and self.viewer.control_bar_manager is not None
        )

    def _sync_grid_colors(self) -> None:
        """Synchronize grid color button swatches with current values."""
        if not self._has_control_bar():
            return

        self.viewer.control_bar_manager.set_axes_grid_color_swatch(
            self.viewer.axes_grid_color
        )

        self.viewer.control_bar_manager.set_adc_grid_color_swatch(
            self.viewer.grid_color
        )
