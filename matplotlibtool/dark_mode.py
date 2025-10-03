#!/usr/bin/env python3
"""
Point Cloud 3D Viewer Library

A library for visualizing 3D point clouds with unlimited rotation and scaling controls.
"""
# -*- coding: utf8 -*-
# tab-width:4

# pylint: disable=invalid-name                    # [C0103] single letter var names, name too descriptive(!)
# pylint: disable=no-member                       # [E1101] no member for base
# pylint: disable=no-name-in-module

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt  # type: ignore
from PyQt6.QtGui import QColor  # type: ignore
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication  # type: ignore
from vispy import scene  # type: ignore
from vispy.app import use_app  # type: ignore
from vispy.scene import visuals  # type: ignore
from vispy.scene.cameras import TurntableCamera  # type: ignore

use_app("pyqt6")


def enable_dark_mode(app_qt: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app_qt.setPalette(palette)


## =============================================================================
## Camera System
## =============================================================================
#
#
# class UnlimitedTurntableCamera(TurntableCamera):
#    """TurntableCamera with unlimited elevation and shift+drag translation."""
#
#    @property
#    def elevation(self):
#        return self._elevation  # noqa: SLF001
#
#    @elevation.setter
#    def elevation(self, elev):
#        elev = float(elev)
#        self._elevation = elev  # noqa: SLF001
#        self.view_changed()
#
#    def _get_rotation_tr(self):
#        from vispy.util import transforms  # type: ignore
#
#        up, forward, right = self._get_dim_vectors()
#        matrix = (
#            transforms.rotate(self.elevation, -right)
#            .dot(transforms.rotate(self.azimuth, up))
#            .dot(transforms.rotate(self.roll, forward))
#        )
#        return matrix
#
#    def _get_shift_state(self) -> bool:
#        try:
#            qt_modifiers = QGuiApplication.keyboardModifiers()
#            return bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)
#        except Exception:
#            return False
#
#    def viewbox_mouse_event(self, event):
#        # Always get fresh shift state from Qt
#        shift_held = self._get_shift_state()
#
#        # Override interaction mode based on shift+button combination
#        if event.type == "mouse_press":
#            if event.button == 1 and shift_held:
#                # Force translation mode when shift+left mouse is pressed
#                self._event_value = None
#                self._interaction = "translate"  # noqa: SLF001
#                self._event_value = event.pos
#                # Still call parent to set up mouse tracking, but override the interaction
#                super().viewbox_mouse_event(event)
#                # Re-override the interaction after parent call
#                self._interaction = "translate"  # noqa: SLF001
#                return
#            elif event.button == 1:
#                # Normal rotation mode - let parent handle it
#                self._interaction = "rotate"  # noqa: SLF001
#
#        elif event.type == "mouse_move":
#            # Re-check shift state on every mouse move
#            shift_held = self._get_shift_state()
#
#            if (
#                hasattr(self, "_interaction")
#                and self._interaction == "translate"  # noqa: SLF001
#                and event.button == 1
#                and self._event_value is not None
#            ):
#                # Handle translation
#                p1 = np.array(self._event_value)
#                p2 = np.array(event.pos)
#                delta = p2 - p1
#
#                # Scale the translation based on distance and viewport size
#                scale_factor = self.distance * 0.001
#
#                # Apply translation relative to current camera orientation
#                self.center = self.center + np.array(
#                    [
#                        -delta[0] * scale_factor,  # Horizontal movement
#                        delta[1] * scale_factor,  # Vertical movement
#                        0.0,
#                    ]
#                )
#
#                self._event_value = event.pos
#                self.view_changed()
#                return  # Don't call parent - we handled this
#
#        elif event.type == "mouse_release":
#            if hasattr(self, "_interaction"):
#                self._interaction = None  # noqa: SLF001
#            self._event_value = None
#
#        # For all other cases, use the default behavior
#        super().viewbox_mouse_event(event)
#
#
## =============================================================================
## Scene Management
## =============================================================================
#
#
# class PointCloudScene:
#    def __init__(
#        self,
#        points: np.ndarray,
#        color_data: np.ndarray | None = None,
#        title: str = "3D Viewer",
#    ):
#        from vispy.color import Colormap
#
#        self.points = points
#        self.color_data = color_data
#        self.canvas = scene.SceneCanvas(
#            keys="interactive",
#            size=(1024, 768),
#            show=True,
#            title=title,
#        )
#        self.canvas.unfreeze()
#
#        self.view = self.canvas.central_widget.add_view()
#
#        if color_data is not None:
#            cmap = Colormap(["blue", "cyan", "green", "yellow", "red"])
#            norm = (color_data - color_data.min()) / max(np.ptp(color_data), 1e-6)
#            self.face_colors = cmap.map(norm)
#        else:
#            self.face_colors = "white"
#
#        self.scatter = visuals.Markers()
#        self.scatter.set_data(
#            self.points,
#            face_color=self.face_colors,
#            size=3,
#        )
#        self.view.add(self.scatter)
#
#    def update_points(self, scaled_points: np.ndarray):
#        self.scatter.set_data(
#            scaled_points,
#            face_color=self.face_colors,
#            size=3,
#        )
