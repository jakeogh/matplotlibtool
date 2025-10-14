from __future__ import annotations

# pylint: disable=no-name-in-module
import sys
from typing import Tuple

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent


class Plot2DInteractions:
    """Encapsulates Matplotlib + Qt interaction handlers.

    The viewer delegates UI events to this object; state changes and rendering
    are still performed via the viewer's public/internal methods.

    UPDATED: Added middle-mouse button and Shift+Left panning support with
    proper pixel-space calculation for uniform panning regardless of axis scale.
    FIXED: Mouse wheel zoom now correctly zooms to cursor position.
    """

    def __init__(
        self,
        viewer,
        ax: Axes,
        canvas: FigureCanvasQTAgg,
        state,
    ):
        self.viewer = viewer
        self.ax = ax
        self.canvas = canvas
        self.state = state

        # Panning state
        self._panning = False
        self._pan_start_data = None
        self._pan_start_pixel = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

        # Flag to disable axis scaling temporarily during mouse operations
        self._disable_axis_scaling = False

    # ---------- Matplotlib callbacks ----------
    def on_zoom_box(
        self,
        eclick,
        erelease,
    ):
        # Check if shift was held during EITHER the click or release
        # This prevents zoom box when the user intended to pan
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        qt_modifiers = QApplication.keyboardModifiers()
        shift_held_now = bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)

        # Also check the event keys
        shift_at_click = eclick.key == "shift"
        shift_at_release = erelease.key == "shift"

        if shift_at_click or shift_at_release or shift_held_now:
            print("[INFO] Ignoring zoom box - shift was held for panning")
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        self.viewer._in_zoom_box = True
        self.ax.set_aspect("auto")
        self.viewer.current_xlim = (x_min, x_max)
        self.viewer.current_ylim = (y_min, y_max)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.viewer._update_plot()
        self.canvas.draw_idle()

        print(
            f"[INFO] Zoomed to box: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.1f}, {y_max:.1f})"
        )

        # CRITICAL FIX: Recreate the RectangleSelector after each use
        # This is necessary because RectangleSelector with useblit=True only works once
        self._recreate_rectangle_selector()

    def on_mouse_scroll(self, event):
        """Handle mouse wheel scroll for zooming to cursor position."""
        # Check if mouse is over any axes
        if event.inaxes is None:
            return

        # CRITICAL: Disable axis scaling during this operation
        self._disable_axis_scaling = True

        # Store pixel coordinates - these don't change with layout shifts
        pixel_x, pixel_y = event.x, event.y

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Calculate mouse position as FRACTION of axis range (0.0 to 1.0)
        # This is more stable than absolute coordinates
        bbox = self.ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            # Fallback to center zoom if we can't calculate position
            self._zoom_to_center(event)
            return

        # Mouse position as fraction (0.0 = left/bottom, 1.0 = right/top)
        mouse_frac_x = (pixel_x - bbox.x0) / bbox.width
        mouse_frac_y = (pixel_y - bbox.y0) / bbox.height

        # Clamp to valid range
        mouse_frac_x = max(0.0, min(1.0, mouse_frac_x))
        mouse_frac_y = max(0.0, min(1.0, mouse_frac_y))

        # Current mouse position in data coordinates
        x_mouse = xlim[0] + mouse_frac_x * (xlim[1] - xlim[0])
        y_mouse = ylim[0] + mouse_frac_y * (ylim[1] - ylim[0])

        # Zoom factor
        zoom_factor = 1.1 if event.step > 0 else 1.0 / 1.1

        # Calculate new limits keeping the mouse position fixed
        # The mouse data coordinate should map to the same fraction after zoom
        new_x_range = (xlim[1] - xlim[0]) / zoom_factor
        new_y_range = (ylim[1] - ylim[0]) / zoom_factor

        new_xlim = (
            x_mouse - mouse_frac_x * new_x_range,
            x_mouse + (1.0 - mouse_frac_x) * new_x_range,
        )
        new_ylim = (
            y_mouse - mouse_frac_y * new_y_range,
            y_mouse + (1.0 - mouse_frac_y) * new_y_range,
        )

        # Clear any keyboard input
        self.viewer.state.base_keys.clear()
        self.viewer.state.shift_keys.clear()
        self.viewer.keyboard_manager.currently_pressed_keys.clear()

        # Reset velocity and scale
        self.viewer.state.velocity[:] = 0.0
        self.viewer.state.scale[:] = 1.0

        # Update base limits
        self.viewer.base_xlim = new_xlim
        self.viewer.base_ylim = new_ylim

        # Apply limits to primary axis
        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)

        # Update viewer state
        self.viewer.current_xlim = new_xlim
        self.viewer.current_ylim = new_ylim

        # Update secondary axes
        if hasattr(self.viewer, "view_manager"):
            self.viewer.view_manager.secondary_axis_manager.update_on_primary_change()

        # Re-enable axis scaling after a delay
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(200, lambda: setattr(self, "_disable_axis_scaling", False))

        # Redraw
        self.canvas.draw_idle()

    def _zoom_to_center(self, event):
        """Fallback: zoom centered on view."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        zoom_factor = 0.9 if event.step > 0 else 1.1

        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor

        new_xlim = (x_center - new_x_range / 2, x_center + new_x_range / 2)
        new_ylim = (y_center - new_y_range / 2, y_center + new_y_range / 2)

        self.viewer.base_xlim = new_xlim
        self.viewer.base_ylim = new_ylim
        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        self.viewer.current_xlim = new_xlim
        self.viewer.current_ylim = new_ylim

        if hasattr(self.viewer, "view_manager"):
            self.viewer.view_manager.secondary_axis_manager.update_on_primary_change()

        from PyQt6.QtCore import QTimer

        QTimer.singleShot(200, lambda: setattr(self, "_disable_axis_scaling", False))

        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        """Handle mouse button press for panning."""
        # Middle button (button 2) or Shift+Left button (button 1) initiates panning
        if event.inaxes is None:
            return

        # Check shift key state from Qt since matplotlib might not report it
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        qt_modifiers = QApplication.keyboardModifiers()
        shift_held = bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)

        # Check for panning conditions
        is_shift_left = event.button == 1 and shift_held
        is_middle = event.button == 2

        # ONLY handle panning events - don't touch normal left-clicks at all
        if is_middle or is_shift_left:
            # Disable rectangle selector during panning
            if hasattr(self.viewer, "rect_selector"):
                self.viewer.rect_selector.set_active(False)

            # Start panning - store BOTH data and pixel coordinates
            self._panning = True
            self._pan_start_data = (event.xdata, event.ydata)
            self._pan_start_pixel = (event.x, event.y)  # Pixel coordinates
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()

            print(f"[INFO] Panning started")
        # Don't do anything for normal left-clicks - let RectangleSelector handle them

    def on_mouse_release(self, event):
        """Handle mouse button release for panning."""
        if self._panning:
            self._panning = False

            # CRITICAL FIX: Update base limits after panning
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            self.viewer.base_xlim = current_xlim
            self.viewer.base_ylim = current_ylim

            # Reset pan state
            self._pan_start_data = None
            self._pan_start_pixel = None
            self._pan_start_xlim = None
            self._pan_start_ylim = None

            # Re-enable rectangle selector for next zoom box
            if hasattr(self.viewer, "rect_selector"):
                self.viewer.rect_selector.set_active(True)

            print("[INFO] Panning ended")

    def on_mouse_move(self, event):
        """Handle mouse motion for panning."""
        # REMOVED: hover handling (now done via direct canvas connection in Plot2D)

        if not self._panning:
            return

        # Calculate pixel delta (same units for both axes)
        pixel_dx = event.x - self._pan_start_pixel[0]
        pixel_dy = event.y - self._pan_start_pixel[1]

        # Get current axis ranges to calculate data-per-pixel ratios
        x_range = self._pan_start_xlim[1] - self._pan_start_xlim[0]
        y_range = self._pan_start_ylim[1] - self._pan_start_ylim[0]

        # Get axes dimensions in pixels
        bbox = self.ax.get_window_extent()
        width_pixels = bbox.width
        height_pixels = bbox.height

        # Convert pixel movement to data movement (proportional to current view)
        data_dx = -pixel_dx * (x_range / width_pixels)
        data_dy = -pixel_dy * (y_range / height_pixels)

        # Apply pan by shifting limits
        new_xlim = (
            self._pan_start_xlim[0] + data_dx,
            self._pan_start_xlim[1] + data_dx,
        )
        new_ylim = (
            self._pan_start_ylim[0] + data_dy,
            self._pan_start_ylim[1] + data_dy,
        )

        # Update the view using ViewManager to ensure proper secondary axis updates
        if hasattr(self.viewer, "view_manager"):
            self.viewer.view_manager.set_view_bounds(xlim=new_xlim, ylim=new_ylim)
        else:
            # Fallback if no view manager
            self.ax.set_xlim(*new_xlim)
            self.ax.set_ylim(*new_ylim)

        # Update viewer state
        self.viewer.current_xlim = new_xlim
        self.viewer.current_ylim = new_ylim

        # Redraw
        self.canvas.draw_idle()

    def _recreate_rectangle_selector(self):
        """Recreate the RectangleSelector to work around matplotlib limitation."""
        from matplotlib.widgets import RectangleSelector

        # Disconnect old selector
        if hasattr(self.viewer, "rect_selector") and self.viewer.rect_selector:
            try:
                self.viewer.rect_selector.disconnect_events()
            except:
                pass

        # Create new selector
        self.viewer.rect_selector = RectangleSelector(
            self.ax,
            self.on_zoom_box,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            ignore_event_outside=True,
            state_modifier_keys={
                "move": "",
                "clear": "",
                "square": "",
                "center": "ctrl",
            },
        )
        self.viewer.rect_selector.set_active(True)

    def on_matplotlib_key_press(self, event):
        """Handle matplotlib key press events."""
        # Toggle point hover with 'H' key
        if event.key == "h" or event.key == "H":
            if hasattr(self.viewer, "point_hover"):
                self.viewer.point_hover.toggle()
            return

        if event.key == "q" or event.key == "escape":
            print(f"[INFO] '{event.key}' pressed, closing viewer.")
            from PyQt6.QtWidgets import QApplication

            self.viewer.close()
        elif event.key:
            key_name = event.key.upper()
            if key_name in ["X", "Y", "Z"]:
                self.state.add_key(key_name, has_shift=False)

    # ---------- Qt key events (delegated) ----------
    def keyPressEvent(self, event: QKeyEvent):
        key_name = event.text().upper() if event.text() else None
        has_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key_name:
            self.state.add_key(key_name, has_shift)
        if event.key() == Qt.Key.Key_Escape:
            print("[INFO] 'ESC' pressed, closing viewer.")
            from PyQt6.QtWidgets import QApplication

            self.viewer.close()
        elif event.key() == Qt.Key.Key_Q:
            print("[INFO] 'q' pressed, closing viewer.")
            from PyQt6.QtWidgets import QApplication

            self.viewer.close()

    def keyReleaseEvent(self, event: QKeyEvent):
        key_name = event.text().upper() if event.text() else None
        if key_name:
            self.state.remove_key(key_name)
