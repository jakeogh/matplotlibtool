#!/usr/bin/env python3
# tab-width:4

"""
Secondary Axis Integration Module for PointCloud2DViewerMatplotlib

This module handles all secondary Y-axis functionality including:
- Configuration and setup
- Event handling
- UI synchronization
- Data range mapping
- Automatic unit scaling (mV, µV, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Optional

from .AxisSecondaryConfig import AxisSecondaryConfig

if TYPE_CHECKING:
    from .PointCloud2DViewerMatplotlib import PointCloud2DViewerMatplotlib


class AxisSecondaryIntegration:
    """
    Handles secondary Y-axis integration for the 2D matplotlib viewer.

    This class manages all secondary axis functionality including configuration,
    event handling, UI synchronization, and automatic unit scaling.
    """

    def __init__(self, viewer: PointCloud2DViewerMatplotlib):
        """
        Initialize secondary axis integration.

        Args:
            viewer: Reference to the main viewer instance
        """
        self.viewer = viewer

    def configure_from_data_range(
        self,
        data_min: float,
        data_max: float,
        target_min: float,
        target_max: float,
        label: str,
        unit: str = "",
        enable_auto_scale: bool = True,
    ) -> None:
        """
        Configure secondary axis from data range mapping with automatic unit scaling.

        This is a convenience method that creates a configuration from a range mapping
        and applies it to the viewer.

        Args:
            data_min: Minimum value in primary axis
            data_max: Maximum value in primary axis
            target_min: Minimum value for secondary axis
            target_max: Maximum value for secondary axis
            label: Axis label (e.g., "Voltage")
            unit: Unit string (e.g., "volt" for pint)
            enable_auto_scale: Enable automatic unit scaling
        """
        config = AxisSecondaryConfig.from_range_mapping(
            primary_min=data_min,
            primary_max=data_max,
            secondary_min=target_min,
            secondary_max=target_max,
            label=label,
            unit=unit,
            enable_auto_scale=enable_auto_scale,
        )

        self.viewer.view_manager.configure_secondary_axis(config)

        # Auto-enable and populate the UI controls
        if (
            hasattr(self.viewer, "control_bar_manager")
            and self.viewer.control_bar_manager
        ):
            self.viewer.control_bar_manager.set_secondary_axis_enabled(True)

            # Set the actual values used for the configuration
            widgets = self.viewer.control_bar_manager.secondary_axis_widgets
            widgets["primary_min"].setText(str(data_min))
            widgets["primary_max"].setText(str(data_max))
            widgets["secondary_min"].setText(str(target_min))
            widgets["secondary_max"].setText(str(target_max))
            widgets["label"].setText(label)
            widgets["unit"].setText(unit)

        self.viewer._update_plot()
        self.viewer.canvas.draw_idle()

        if enable_auto_scale:
            print(
                f"[INFO] Secondary axis configured with automatic unit scaling: {label} ({unit})"
            )
        else:
            print(
                f"[INFO] Secondary axis configured with fixed units: {label} ({unit})"
            )

    def on_secondary_axis_toggled(self, enabled: bool) -> None:
        """
        Handle secondary axis enable/disable toggle.

        Args:
            enabled: Whether to enable or disable the secondary axis
        """
        if not enabled:
            self.viewer.view_manager.disable_secondary_axis()
            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()
            print("[INFO] Secondary Y-axis disabled")
        else:
            # When enabled, check if we have configuration values to apply automatically
            if (
                hasattr(self.viewer, "control_bar_manager")
                and self.viewer.control_bar_manager
            ):
                widgets = self.viewer.control_bar_manager.secondary_axis_widgets

                try:
                    primary_min = widgets["primary_min"].text().strip()
                    primary_max = widgets["primary_max"].text().strip()
                    secondary_min = widgets["secondary_min"].text().strip()
                    secondary_max = widgets["secondary_max"].text().strip()
                    label = widgets["label"].text().strip()
                    unit = widgets["unit"].text().strip()

                    if all(
                        [primary_min, primary_max, secondary_min, secondary_max, label]
                    ):
                        config = AxisSecondaryConfig.from_range_mapping(
                            primary_min=float(primary_min),
                            primary_max=float(primary_max),
                            secondary_min=float(secondary_min),
                            secondary_max=float(secondary_max),
                            label=label,
                            unit=unit,
                            enable_auto_scale=True,
                        )

                        self.viewer.view_manager.configure_secondary_axis(config)
                        self.viewer._update_plot()
                        self.viewer.canvas.draw_idle()
                        print(
                            f"[INFO] Secondary Y-axis auto-configured with auto-scaling: {label} ({unit})"
                        )
                    else:
                        print(
                            "[INFO] Secondary Y-axis enabled - fill in the mapping values and click Apply"
                        )

                except (ValueError, KeyError) as e:
                    print(
                        f"[INFO] Secondary Y-axis enabled - configure range mapping below (error: {e})"
                    )
            else:
                print("[INFO] Secondary Y-axis enabled - configure range mapping below")

    def on_secondary_axis_config_requested(self, config: AxisSecondaryConfig) -> None:
        """
        Handle secondary axis configuration request.

        Args:
            config: AxisSecondaryConfig object with mapping parameters
        """
        try:
            with self.viewer.busy_manager.busy_operation("Configuring secondary axis"):
                self.viewer.view_manager.configure_secondary_axis(config)

                # Auto-enable the checkbox when configuration is applied
                self.viewer.control_bar_manager.set_secondary_axis_enabled(True)

                # Populate the text fields with the applied configuration
                self.viewer.control_bar_manager.set_secondary_axis_config(config)

                # Force complete redraw with secondary axis
                self.viewer._update_plot()
                self.viewer.canvas.draw_idle()

                print(
                    f"[INFO] Secondary Y-axis configured: {config.label} ({config.unit})"
                )

                if config.enable_auto_scale:
                    print(
                        "[INFO] Automatic unit scaling enabled (will show mV, µV, etc. when zoomed)"
                    )

                print(
                    f"[INFO] Range mapping: Primary [{config.scale:.3e}x + {config.offset:.3f}] → Secondary"
                )

        except Exception as e:
            print(f"[ERROR] Failed to configure secondary axis: {e}")

    def sync_ui_state(self) -> None:
        """
        Synchronize UI controls with current secondary axis state.

        This method updates the control bar widgets to reflect the current
        secondary axis configuration.
        """
        if (
            not hasattr(self.viewer, "control_bar_manager")
            or self.viewer.control_bar_manager is None
        ):
            return

        # Sync secondary axis state
        self.viewer.control_bar_manager.set_secondary_axis_enabled(
            self.viewer.view_manager.is_secondary_axis_enabled()
        )
        self.viewer.control_bar_manager.set_secondary_axis_config(
            self.viewer.view_manager.get_secondary_axis_config()
        )

    def connect_signals(self) -> dict:
        """
        Get signal connections for secondary axis functionality.

        Returns:
            Dictionary mapping signal names to handler methods
        """
        return {
            "secondaryAxisToggled": self.on_secondary_axis_toggled,
            "secondaryAxisConfigRequested": self.on_secondary_axis_config_requested,
        }

    def update_after_plot(self) -> None:
        """
        Update secondary axis after main plot is rendered.

        This ensures the secondary axis is visible and properly configured
        after the main axes are drawn.
        """
        if self.viewer.view_manager.is_secondary_axis_enabled():
            self.viewer.view_manager.secondary_axis_manager.update_on_primary_change()

    def is_enabled(self) -> bool:
        """
        Check if secondary axis is currently enabled.

        Returns:
            True if secondary axis is enabled, False otherwise
        """
        return self.viewer.view_manager.is_secondary_axis_enabled()
