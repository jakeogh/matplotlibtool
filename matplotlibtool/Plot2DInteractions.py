from __future__ import annotations

# pylint: disable=no-name-in-module
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
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

        self.zoom_box_enabled = True
        self.drawing_zoom_box = False
        self.min_span_pixels = 5.0
        self._box_start_pixel: tuple[float, float] | None = None
        self._box_background = None
        self._box_rect = Rectangle(
            (0.0, 0.0),
            0.0,
            0.0,
            linewidth=1.0,
            linestyle="--",
            edgecolor="#cccccc",
            facecolor="#ffffff",
            alpha=0.25,
            animated=True,
            visible=False,
        )
        ax.add_patch(self._box_rect)

    # ---------- zoom box ----------
    #
    # The box may start and end anywhere on the canvas, including outside
    # the axes. Each axis of the box clamps to the intersection with the
    # current view; an axis with no intersection or with a drag span under
    # min_span_pixels keeps its full current range. Dragging above the
    # plot therefore zooms X only, dragging beside it zooms Y only.

    def _begin_zoom_box(self, event) -> None:
        self.drawing_zoom_box = True
        self._box_start_pixel = (event.x, event.y)
        self.canvas.draw()
        self._box_background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)

    def _clamped_box(
        self,
        event,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        inv = self.ax.transData.inverted()
        x1, y1 = inv.transform(self._box_start_pixel)
        x2, y2 = inv.transform((event.x, event.y))

        span_px_x = abs(event.x - self._box_start_pixel[0])
        span_px_y = abs(event.y - self._box_start_pixel[1])
        if span_px_x < self.min_span_pixels and span_px_y < self.min_span_pixels:
            return None

        view_xlim = self.ax.get_xlim()
        view_ylim = self.ax.get_ylim()

        if span_px_x < self.min_span_pixels:
            xlim = view_xlim
        else:
            x_lo, x_hi = sorted((x1, x2))
            x_lo = max(x_lo, view_xlim[0])
            x_hi = min(x_hi, view_xlim[1])
            xlim = (x_lo, x_hi) if x_lo < x_hi else view_xlim

        if span_px_y < self.min_span_pixels:
            ylim = view_ylim
        else:
            y_lo, y_hi = sorted((y1, y2))
            y_lo = max(y_lo, view_ylim[0])
            y_hi = min(y_hi, view_ylim[1])
            ylim = (y_lo, y_hi) if y_lo < y_hi else view_ylim

        return xlim, ylim

    def _update_zoom_box(self, event) -> None:
        self.canvas.restore_region(self._box_background)
        box = self._clamped_box(event)
        if box is not None:
            xlim, ylim = box
            self._box_rect.set_bounds(
                xlim[0],
                ylim[0],
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
            )
            self._box_rect.set_visible(True)
            self.ax.draw_artist(self._box_rect)
        self.canvas.blit(self.canvas.figure.bbox)

    def _finish_zoom_box(self, event) -> None:
        box = self._clamped_box(event)
        self._end_zoom_box_drawing()
        if box is None:
            self.canvas.draw_idle()
            return

        xlim, ylim = box
        self.ax.set_aspect("auto")
        self.viewer.set_view(xlim, ylim)
        print(
            f"[INFO] Zoomed to box: X({xlim[0]:.1f}, {xlim[1]:.1f}), Y({ylim[0]:.1f}, {ylim[1]:.1f})"
        )

    def cancel_zoom_box(self) -> None:
        self._end_zoom_box_drawing()
        self.canvas.draw_idle()

    def _end_zoom_box_drawing(self) -> None:
        self.drawing_zoom_box = False
        self._box_start_pixel = None
        self._box_background = None
        self._box_rect.set_visible(False)

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
        if self.panning or self.drawing_zoom_box:
            return

        qt_modifiers = QApplication.keyboardModifiers()
        shift_held = bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)

        is_shift_left = event.button == 1 and shift_held
        is_middle = event.button == 2

        if (is_middle or is_shift_left) and event.inaxes is not None:
            self.panning = True
            self._pan_start_pixel = (event.x, event.y)
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()
        elif event.button == 1 and not shift_held and self.zoom_box_enabled:
            self._begin_zoom_box(event)

    def on_mouse_move(self, event):
        if self.drawing_zoom_box:
            self._update_zoom_box(event)
            return
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
            record=False,
        )

    def on_mouse_release(self, event):
        if self.drawing_zoom_box:
            self._finish_zoom_box(event)
            return
        if self.panning:
            self.viewer.record_view_history()
            self.panning = False
            self._pan_start_pixel = None
            self._pan_start_xlim = None
            self._pan_start_ylim = None

    # ---------- keyboard ----------

    def on_matplotlib_key_press(self, event):
        if event.key == "escape" and self.drawing_zoom_box:
            self.cancel_zoom_box()
            return
        if event.key == "escape" and self.viewer.point_hover.clear_measurement_if_active():
            return

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
