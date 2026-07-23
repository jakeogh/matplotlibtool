#!/usr/bin/env python3
# tab-width:4

"""
PointHover: hover point identification, coordinate copy, and two-point
measurement.

Press 'H' to toggle. While enabled:
- hovering snaps to the nearest point and shows its coordinates
- right-click copies the snapped point's coordinates to the clipboard
- left-click on a point anchors a measurement; a second left-click on
  another point shows and copies dx, dy (and distance); left-click on
  empty space or escape clears the measurement
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtGui import QClipboard
from PyQt6.QtWidgets import QApplication

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


def _copy_to_clipboard(text: str) -> None:
    clipboard = QApplication.clipboard()
    clipboard.setText(text, QClipboard.Mode.Selection)
    clipboard.setText(text, QClipboard.Mode.Clipboard)


def _full_precision(value: float) -> str:
    return repr(float(value))


class PointHoverManager:
    def __init__(
        self,
        viewer,
        ax,
        canvas,
    ):
        self.viewer = viewer
        self.ax = ax
        self.canvas = canvas

        self.enabled = False
        self.snap_distance = 10  # pixels
        self.current_point = None
        self.hover_marker = None
        self.hover_annotation = None

        self.measure_anchor: tuple[float, float] | None = None
        self._measure_artists: list = []

        self._click_connection = None

    # ---------- lifecycle ----------

    def enable(self):
        self.enabled = True
        self.viewer.interactions.zoom_box_enabled = False
        if self._click_connection is None:
            self._click_connection = self.canvas.mpl_connect(
                "button_press_event", self._on_click
            )
        print(
            "[INFO] Point hover enabled - right-click copies coordinates, "
            "left-click two points measures dx/dy"
        )

    def disable(self):
        self.enabled = False
        self.clear_hover_display()
        self.clear_measurement()
        self.canvas.draw_idle()
        if self._click_connection is not None:
            self.canvas.mpl_disconnect(self._click_connection)
            self._click_connection = None
        self.viewer.interactions.zoom_box_enabled = True

    def toggle(self):
        if self.enabled:
            self.disable()
        else:
            self.enable()

    # ---------- display state ----------

    def clear_hover_display(self):
        if self.hover_marker is not None:
            self.hover_marker.remove()
            self.hover_marker = None
        if self.hover_annotation is not None:
            self.hover_annotation.remove()
            self.hover_annotation = None
        self.current_point = None

    def clear_measurement(self) -> bool:
        """Clear any measurement display. Returns True if there was one."""
        had_any = self.measure_anchor is not None or bool(self._measure_artists)
        for artist in self._measure_artists:
            artist.remove()
        self._measure_artists = []
        self.measure_anchor = None
        return had_any

    def clear_measurement_if_active(self) -> bool:
        if self.clear_measurement():
            self.canvas.draw_idle()
            return True
        return False

    # ---------- events ----------

    def _mouse_primary_coords(self, event) -> tuple[float, float] | None:
        """Mouse position in primary-axis data coordinates, or None if the
        event is outside every axes (primary and secondary)."""
        all_axes = [self.ax]
        sec_mgr = self.viewer.view_manager.secondary_axis_manager
        if sec_mgr.y_axis_manager.secondary_ax:
            all_axes.append(sec_mgr.y_axis_manager.secondary_ax)
        if sec_mgr.x_axis_manager.secondary_ax:
            all_axes.append(sec_mgr.x_axis_manager.secondary_ax)

        if event.inaxes not in all_axes:
            return None

        if event.inaxes == self.ax:
            return event.xdata, event.ydata
        # secondary axes report their own data coordinates
        inv = self.ax.transData.inverted()
        mouse_x, mouse_y = inv.transform((event.x, event.y))
        return mouse_x, mouse_y

    def on_hover_motion(self, event):
        if not self.enabled:
            return

        coords = self._mouse_primary_coords(event)
        if coords is None:
            if self.hover_marker or self.hover_annotation:
                self.clear_hover_display()
                self.canvas.draw_idle()
            return

        nearest_point, nearest_dist_pixels, _ = self._find_nearest_point(*coords)

        if nearest_point is not None and nearest_dist_pixels <= self.snap_distance:
            self._show_hover_display(nearest_point)
            self.canvas.draw_idle()
        elif self.hover_marker or self.hover_annotation:
            self.clear_hover_display()
            self.canvas.draw_idle()

    def _on_click(self, event):
        if event.button == 3:
            self._copy_hovered_point()
            return
        if event.button == 1:
            self._measure_click(event)

    # ---------- coordinate copy ----------

    def _copy_hovered_point(self) -> None:
        if self.current_point is None:
            return
        x, y = self.current_point
        coord_text = f"{_full_precision(x)}, {_full_precision(y)}"
        _copy_to_clipboard(coord_text)
        print(f"[INFO] Copied to clipboard: {coord_text}")

    # ---------- measurement ----------

    def _measure_click(self, event) -> None:
        coords = self._mouse_primary_coords(event)
        if coords is None:
            return

        nearest_point, nearest_dist_pixels, _ = self._find_nearest_point(*coords)
        snapped = (
            nearest_point is not None and nearest_dist_pixels <= self.snap_distance
        )

        if not snapped:
            self.clear_measurement_if_active()
            return

        point = (float(nearest_point[0]), float(nearest_point[1]))

        if self.measure_anchor is None:
            self.clear_measurement()
            self._set_measure_anchor(point)
        else:
            anchor = self.measure_anchor
            self._complete_measurement(anchor, point)
        self.canvas.draw_idle()

    def _set_measure_anchor(self, point: tuple[float, float]) -> None:
        self.measure_anchor = point
        marker = self.ax.plot(
            point[0],
            point[1],
            "o",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor="cyan",
            markeredgewidth=2,
            zorder=1000,
        )[0]
        self._measure_artists.append(marker)
        print(
            f"[INFO] Measurement anchor: "
            f"({_full_precision(point[0])}, {_full_precision(point[1])})"
        )

    def _complete_measurement(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = float(np.hypot(dx, dy))

        marker = self.ax.plot(
            p2[0],
            p2[1],
            "o",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor="cyan",
            markeredgewidth=2,
            zorder=1000,
        )[0]
        line = self.ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "--",
            color="cyan",
            linewidth=1.5,
            zorder=999,
        )[0]
        label = self.ax.annotate(
            f"Δx={dx:.6g}\nΔy={dy:.6g}\nd={dist:.6g}",
            xy=((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="cyan", alpha=0.85),
            fontsize=10,
            zorder=1001,
            color="black",
        )
        self._measure_artists.extend((marker, line, label))
        self.measure_anchor = None

        delta_text = f"{_full_precision(dx)}, {_full_precision(dy)}"
        _copy_to_clipboard(delta_text)
        print(
            f"[INFO] Measurement: dx={_full_precision(dx)} dy={_full_precision(dy)} "
            f"distance={_full_precision(dist)} (dx, dy copied to clipboard)"
        )

    # ---------- nearest point ----------

    def _find_nearest_point(
        self,
        mouse_x,
        mouse_y,
    ):
        """
        Nearest visible point to the mouse, searching only points within
        the renderer's cull window.

        Returns:
            (point, distance_in_pixels, plot) or (None, inf, None)
        """
        visible_plots = self.viewer.plot_manager.get_visible_plots()
        if not visible_plots:
            return None, float("inf"), None

        bbox = self.ax.get_window_extent()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_scale = bbox.width / (xlim[1] - xlim[0])
        y_scale = bbox.height / (ylim[1] - ylim[0])

        margin = self.viewer.cull_margin
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        cull_xlim = (xlim[0] - x_range * margin, xlim[1] + x_range * margin)
        cull_ylim = (ylim[0] - y_range * margin, ylim[1] + y_range * margin)

        nearest_point = None
        nearest_dist = float("inf")
        nearest_plot = None

        for plot in visible_plots:
            if len(plot.points) == 0:
                continue

            offset_points = plot.points + np.array([plot.offset_x, plot.offset_y])

            mask = (
                (offset_points[:, 0] >= cull_xlim[0])
                & (offset_points[:, 0] <= cull_xlim[1])
                & (offset_points[:, 1] >= cull_ylim[0])
                & (offset_points[:, 1] <= cull_ylim[1])
            )
            if not mask.any():
                continue

            culled_points = offset_points[mask]

            dx_pixels = (culled_points[:, 0] - mouse_x) * x_scale
            dy_pixels = (culled_points[:, 1] - mouse_y) * y_scale
            distances = np.sqrt(dx_pixels**2 + dy_pixels**2)

            min_idx = int(np.argmin(distances))
            if distances[min_idx] < nearest_dist:
                nearest_dist = float(distances[min_idx])
                nearest_point = culled_points[min_idx]
                nearest_plot = plot

        return nearest_point, nearest_dist, nearest_plot

    # ---------- hover display ----------

    def _show_hover_display(self, point):
        x, y = point
        self.clear_hover_display()

        self.hover_marker = self.ax.plot(
            x,
            y,
            "o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor="yellow",
            markeredgewidth=2,
            zorder=1000,
        )[0]

        self.hover_annotation = self.ax.annotate(
            f"({x:.6g}, {y:.6g})",
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
