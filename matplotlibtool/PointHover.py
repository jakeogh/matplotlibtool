#!/usr/bin/env python3
# tab-width:4

"""
PointHover: Mouse hover point identification with snap-to functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class PointHoverManager:
    """
    Manages mouse hover point identification with snap-to functionality.
    Shows coordinates of nearest point when hovering.
    """

    def __init__(
        self,
        viewer,
        ax,
        canvas,
    ):
        """
        Initialize point hover manager.

        Args:
            viewer: The Plot2D viewer instance
            ax: Matplotlib axes
            canvas: Matplotlib canvas
        """
        self.viewer = viewer
        self.ax = ax
        self.canvas = canvas

        # Hover state
        self.enabled = False
        self.snap_distance = 10  # pixels
        self.current_point = None
        self.hover_marker = None
        self.hover_annotation = None

        # Connect right-click handler for copying coordinates
        self._click_connection = None

    def enable(self):
        """Enable point hover identification."""
        self.enabled = True

        # CRITICAL: Disable RectangleSelector to allow motion events through
        if hasattr(self.viewer, "rect_selector") and self.viewer.rect_selector:
            self.viewer.rect_selector.set_active(False)
            print("[DEBUG] RectangleSelector disabled for hover mode")

        # Connect right-click handler
        if self._click_connection is None:
            self._click_connection = self.canvas.mpl_connect(
                "button_press_event", self._on_click
            )

        print("[INFO] Point hover enabled - hover over points to see coordinates")
        print("[INFO] Right-click to copy coordinates to clipboard")

    def disable(self):
        """Disable point hover identification and clear markers."""
        self.enabled = False
        self.clear_hover_display()
        self.canvas.draw_idle()

        # Disconnect right-click handler
        if self._click_connection is not None:
            self.canvas.mpl_disconnect(self._click_connection)
            self._click_connection = None

        # Re-enable RectangleSelector
        if hasattr(self.viewer, "rect_selector") and self.viewer.rect_selector:
            self.viewer.rect_selector.set_active(True)
            print("[DEBUG] RectangleSelector re-enabled")

        print("[INFO] Point hover disabled")

    def toggle(self):
        """Toggle point hover on/off."""
        if self.enabled:
            self.disable()
        else:
            self.enable()

    def clear_hover_display(self):
        """Remove hover marker and annotation from display."""
        if self.hover_marker is not None:
            self.hover_marker.remove()
            self.hover_marker = None

        if self.hover_annotation is not None:
            self.hover_annotation.remove()
            self.hover_annotation = None

        self.current_point = None

    def on_hover_motion(self, event):
        """
        Handle mouse motion for point identification.

        Args:
            event: Matplotlib mouse event
        """
        print(f"[HOVER-START] on_hover_motion called! enabled={self.enabled}")

        # Don't interfere with other operations
        if not self.enabled:
            print(f"[HOVER-START] Exiting - not enabled")
            return

        print(f"[HOVER-AXES] event.inaxes={event.inaxes}, self.ax={self.ax}")

        # Get all axes including secondary
        all_axes = [self.ax]
        if hasattr(self.viewer, "view_manager") and self.viewer.view_manager:
            sec_mgr = self.viewer.view_manager.secondary_axis_manager
            if sec_mgr.y_axis_manager.secondary_ax:
                all_axes.append(sec_mgr.y_axis_manager.secondary_ax)
            if sec_mgr.x_axis_manager.secondary_ax:
                all_axes.append(sec_mgr.x_axis_manager.secondary_ax)

        print(f"[HOVER-AXES] All axes: {all_axes}")

        # Check if in any axes
        if event.inaxes not in all_axes:
            print(f"[HOVER-AXES] Not in any axes, clearing display")
            if self.hover_marker or self.hover_annotation:
                self.clear_hover_display()
                self.canvas.draw_idle()
            return

        print(f"[HOVER] Mouse motion event received, enabled={self.enabled}")

        # Get mouse position in PRIMARY AXIS data coordinates
        # CRITICAL: event.xdata/ydata might be in secondary axis coordinates!
        if event.inaxes == self.ax:
            # Already in primary axis
            mouse_x, mouse_y = event.xdata, event.ydata
            print(f"[HOVER] Event in primary axis")
        else:
            # In secondary axis - transform to primary axis coordinates
            print(f"[HOVER] Event in secondary axis, transforming...")
            # Get the mouse position in pixels
            pixel_x, pixel_y = event.x, event.y
            # Transform to primary axis data coordinates
            inv = self.ax.transData.inverted()
            mouse_x, mouse_y = inv.transform((pixel_x, pixel_y))
            print(
                f"[HOVER] Transformed from pixels ({pixel_x}, {pixel_y}) to data ({mouse_x:.2f}, {mouse_y:.2f})"
            )

        if mouse_x is None or mouse_y is None:
            print(f"[HOVER] Mouse position is None")
            return

        print(f"[HOVER] Mouse at PRIMARY coordinates ({mouse_x:.2f}, {mouse_y:.2f})")

        # Find nearest point
        nearest_point, nearest_dist_pixels, plot_index = self._find_nearest_point(
            mouse_x, mouse_y, event
        )

        print(
            f"[HOVER] Nearest point: {nearest_point}, dist={nearest_dist_pixels:.1f}px, threshold={self.snap_distance}px"
        )

        # Check if point is within snap distance
        if nearest_point is not None and nearest_dist_pixels <= self.snap_distance:
            print(f"[HOVER] Point within threshold, showing display")
            # Update or create hover display
            self._show_hover_display(nearest_point, plot_index)
            self.canvas.draw_idle()
        else:
            print(f"[HOVER] No point within threshold")
            # Clear hover display if no point nearby
            if self.hover_marker or self.hover_annotation:
                self.clear_hover_display()
                self.canvas.draw_idle()

    def _find_nearest_point(
        self,
        mouse_x,
        mouse_y,
        event,
    ):
        """
        Find the nearest visible point to mouse position.
        Only searches points that are currently in view (culled).

        Args:
            mouse_x: Mouse X in data coordinates
            mouse_y: Mouse Y in data coordinates
            event: Matplotlib event for pixel conversion

        Returns:
            tuple: (nearest_point, distance_in_pixels, plot_index) or (None, inf, None)
        """
        # Get visible plots
        visible_plots = self.viewer.plot_manager.get_visible_plots()
        if not visible_plots:
            print(f"[HOVER] No visible plots")
            return None, float("inf"), None

        print(f"[HOVER] Found {len(visible_plots)} visible plots")

        # Get axes dimensions for pixel conversion
        bbox = self.ax.get_window_extent()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_scale = bbox.width / (xlim[1] - xlim[0])
        y_scale = bbox.height / (ylim[1] - ylim[0])

        # Create a culling rectangle around the current view
        # Use same margin as Plot2D._update_plot()
        margin = 0.1
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        cull_xlim = (
            xlim[0] - x_range * margin,
            xlim[1] + x_range * margin,
        )
        cull_ylim = (
            ylim[0] - y_range * margin,
            ylim[1] + y_range * margin,
        )

        print(
            f"[HOVER] View: X({xlim[0]:.1f}, {xlim[1]:.1f}), Y({ylim[0]:.1f}, {ylim[1]:.1f})"
        )
        print(
            f"[HOVER] Cull: X({cull_xlim[0]:.1f}, {cull_xlim[1]:.1f}), Y({cull_ylim[0]:.1f}, {cull_ylim[1]:.1f})"
        )

        nearest_point = None
        nearest_dist = float("inf")
        nearest_plot_idx = None

        # Search through all visible plots
        for plot_idx, plot in enumerate(visible_plots):
            if len(plot.points) == 0:
                continue

            print(f"[HOVER] Plot {plot_idx}: {len(plot.points)} total points")

            # Apply offset to points
            offset_points = plot.points + np.array([plot.offset_x, plot.offset_y])

            # CRITICAL: Cull points to only those in view (like the renderer does)
            mask = (
                (offset_points[:, 0] >= cull_xlim[0])
                & (offset_points[:, 0] <= cull_xlim[1])
                & (offset_points[:, 1] >= cull_ylim[0])
                & (offset_points[:, 1] <= cull_ylim[1])
            )

            culled_count = mask.sum()
            print(f"[HOVER] Plot {plot_idx}: {culled_count} culled points in view")

            if not mask.any():
                continue  # No points in view for this plot

            culled_points = offset_points[mask]

            # Calculate distances in data coordinates
            dx = culled_points[:, 0] - mouse_x
            dy = culled_points[:, 1] - mouse_y

            # Convert to pixel distances (accounting for axis scaling)
            dx_pixels = dx * x_scale
            dy_pixels = dy * y_scale
            distances = np.sqrt(dx_pixels**2 + dy_pixels**2)

            # Find minimum distance
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            print(
                f"[HOVER] Plot {plot_idx}: nearest at {min_dist:.1f}px, point=({culled_points[min_idx][0]:.2f}, {culled_points[min_idx][1]:.2f})"
            )

            if min_dist < nearest_dist:
                nearest_dist = min_dist
                nearest_point = culled_points[min_idx]
                nearest_plot_idx = plot_idx

        print(f"[HOVER] Overall nearest: {nearest_dist:.1f}px")
        return nearest_point, nearest_dist, nearest_plot_idx

    def _show_hover_display(
        self,
        point,
        plot_index,
    ):
        """
        Display hover marker and annotation for the given point.

        Args:
            point: (x, y) coordinates of the point
            plot_index: Index of the plot containing the point
        """
        x, y = point

        print(f"[HOVER] Showing display at ({x:.6g}, {y:.6g})")

        # Remove old marker and annotation
        self.clear_hover_display()

        # Create new marker (highlight circle)
        self.hover_marker = self.ax.plot(
            x,
            y,
            "o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor="yellow",
            markeredgewidth=2,
            zorder=1000,  # Draw on top
        )[0]

        # Create annotation with coordinates
        coord_text = f"({x:.6g}, {y:.6g})"

        self.hover_annotation = self.ax.annotate(
            coord_text,
            xy=(x, y),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0", color="yellow"
            ),
            fontsize=10,
            zorder=1001,
            color="black",
        )

        self.current_point = point
        print(
            f"[HOVER] Display created: marker={self.hover_marker}, annotation={self.hover_annotation}"
        )

    def _on_click(self, event):
        """
        Handle mouse click events for copying coordinates.

        Args:
            event: Matplotlib button press event
        """
        # Only handle right-click (button 3)
        if event.button != 3:
            return

        # Only copy if we have a current hovered point
        if self.current_point is None:
            print("[HOVER] No point currently hovered - nothing to copy")
            return

        # Format coordinates as "x, y"
        x, y = self.current_point
        coord_text = f"{x:.6g}, {y:.6g}"

        # Copy to clipboard (middle-click clipboard on Linux, standard clipboard on others)
        try:
            from PyQt6.QtGui import QClipboard
            from PyQt6.QtWidgets import QApplication

            clipboard = QApplication.clipboard()

            # Set both selection (middle-click) and clipboard (Ctrl+V) on Linux
            # On other platforms, selection mode is ignored
            clipboard.setText(coord_text, QClipboard.Mode.Selection)
            clipboard.setText(coord_text, QClipboard.Mode.Clipboard)

            print(f"[HOVER] Copied to clipboard: {coord_text}")

        except Exception as e:
            print(f"[HOVER] Failed to copy to clipboard: {e}")
