from __future__ import annotations

# pylint: disable=no-name-in-module
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication


class Plot2DInteractions:
    """
    Matplotlib + Qt interaction handlers.

    Every view change (zoom box, wheel zoom, pan) goes through
    viewer.set_view(), which re-renders so culled artist data always
    matches the visible region.
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

        self.panning = False
        self._pan_start_pixel: tuple[float, float] | None = None
        self._pan_start_xlim: tuple[float, float] | None = None
        self._pan_start_ylim: tuple[float, float] | None = None

    # ---------- zoom box ----------

    def on_zoom_box(self, eclick, erelease):
        # shift indicates the drag was intended as a pan
        qt_modifiers = QApplication.keyboardModifiers()
        shift_held_now = bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)
        if eclick.key == "shift" or erelease.key == "shift" or shift_held_now:
            print("[INFO] Ignoring zoom box - shift was held for panning")
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return

        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        if x_min == x_max or y_min == y_max:
            return

        self.ax.set_aspect("auto")
        self.viewer.set_view((x_min, x_max), (y_min, y_max))

        print(
            f"[INFO] Zoomed to box: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.1f}, {y_max:.1f})"
        )

        # RectangleSelector with useblit caches a stale background after use
        self._recreate_rectangle_selector()

    # ---------- wheel zoom ----------

    def on_mouse_scroll(self, event):
        """Zoom anchored at the cursor position."""
        if event.inaxes is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        bbox = self.ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            mouse_frac_x = 0.5
            mouse_frac_y = 0.5
        else:
            mouse_frac_x = min(1.0, max(0.0, (event.x - bbox.x0) / bbox.width))
            mouse_frac_y = min(1.0, max(0.0, (event.y - bbox.y0) / bbox.height))

        x_mouse = xlim[0] + mouse_frac_x * (xlim[1] - xlim[0])
        y_mouse = ylim[0] + mouse_frac_y * (ylim[1] - ylim[0])

        zoom_factor = 1.1 if event.step > 0 else 1.0 / 1.1
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

        self.viewer.keyboard_manager.clear_input_state()
        self.viewer.set_view(new_xlim, new_ylim)

    # ---------- panning ----------

    def on_mouse_press(self, event):
        if event.inaxes is None:
            return

        qt_modifiers = QApplication.keyboardModifiers()
        shift_held = bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)

        is_shift_left = event.button == 1 and shift_held
        is_middle = event.button == 2

        if is_middle or is_shift_left:
            self.viewer.rect_selector.set_active(False)
            self.panning = True
            self._pan_start_pixel = (event.x, event.y)
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()

    def on_mouse_move(self, event):
        if not self.panning:
            return

        pixel_dx = event.x - self._pan_start_pixel[0]
        pixel_dy = event.y - self._pan_start_pixel[1]

        x_range = self._pan_start_xlim[1] - self._pan_start_xlim[0]
        y_range = self._pan_start_ylim[1] - self._pan_start_ylim[0]

        bbox = self.ax.get_window_extent()
        data_dx = -pixel_dx * (x_range / bbox.width)
        data_dy = -pixel_dy * (y_range / bbox.height)

        self.viewer.set_view(
            (self._pan_start_xlim[0] + data_dx, self._pan_start_xlim[1] + data_dx),
            (self._pan_start_ylim[0] + data_dy, self._pan_start_ylim[1] + data_dy),
        )

    def on_mouse_release(self, event):
        if self.panning:
            self.panning = False
            self._pan_start_pixel = None
            self._pan_start_xlim = None
            self._pan_start_ylim = None
            self.viewer.rect_selector.set_active(True)

    # ---------- selector lifecycle ----------

    def make_rectangle_selector(self) -> RectangleSelector:
        selector = RectangleSelector(
            self.ax,
            self.on_zoom_box,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            ignore_event_outside=False,
            state_modifier_keys={
                "move": "",
                "clear": "",
                "square": "",
                "center": "ctrl",
            },
        )
        selector.set_active(True)
        return selector

    def _recreate_rectangle_selector(self) -> None:
        self.viewer.rect_selector.disconnect_events()
        self.viewer.rect_selector = self.make_rectangle_selector()

    # ---------- keyboard ----------

    def on_matplotlib_key_press(self, event):
        if event.key in ("h", "H"):
            self.viewer.point_hover.toggle()
            return

        if event.key in ("q", "escape"):
            print(f"[INFO] '{event.key}' pressed, closing viewer.")
            self.viewer.close()
        elif event.key:
            key_name = event.key.upper()
            if key_name in ("X", "Y", "Z"):
                self.state.add_key(key_name, has_shift=False)

    def keyPressEvent(self, event: QKeyEvent):
        key_name = event.text().upper() if event.text() else None
        has_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key_name:
            self.state.add_key(key_name, has_shift)
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            print("[INFO] Close key pressed, closing viewer.")
            self.viewer.close()

    def keyReleaseEvent(self, event: QKeyEvent):
        key_name = event.text().upper() if event.text() else None
        if key_name:
            self.state.remove_key(key_name)
