#!/usr/bin/env python3
# tab-width:4

"""
Fields popup panel.

One control-bar button opening a popup listing every array with all of
its fields: a checkbox enables or disables the field's plot and a line
edit sets its Y multiplier. Replaces the per-selected-array visibility
and scale rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QMenu
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QToolButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from PyQt6.QtWidgets import QWidgetAction

if TYPE_CHECKING:
    from .ArrayFieldIntegration import ArrayFieldIntegration


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())


class ArrayFieldPanel:
    """Popup listing every field of every array with enable and multiplier."""

    def __init__(self, integration: ArrayFieldIntegration):
        self.integration = integration
        self.button: QToolButton | None = None
        self.max_popup_height = 700
        self._content: QWidget | None = None
        self._content_layout: QVBoxLayout | None = None
        self._scroll: QScrollArea | None = None

    def create_button(self, parent=None) -> QToolButton:
        button = QToolButton(parent)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setToolTip(
            "Enable or disable any field from any array and set Y multipliers"
        )

        menu = QMenu(button)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 8, 10, 8)
        self._content_layout.setSpacing(4)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._content)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        action = QWidgetAction(menu)
        action.setDefaultWidget(self._scroll)
        menu.addAction(action)
        menu.aboutToShow.connect(self.rebuild)

        button.setMenu(menu)
        self.button = button
        self.update_button_label()
        return button

    def update_button_label(self) -> None:
        manager = self.integration.array_field_manager
        total = 0
        enabled = 0
        for array_index in manager.arrays:
            for field_name in manager.get_array_fields(array_index):
                total += 1
                if self.integration.is_field_visible(array_index, field_name):
                    enabled += 1
        self.button.setText(f"Fields {enabled}/{total} ▾" if total else "Fields ▾")

    def rebuild(self) -> None:
        _clear_layout(self._content_layout)
        manager = self.integration.array_field_manager

        if not manager.arrays:
            self._content_layout.addWidget(QLabel("no arrays loaded"))
        else:
            for array_index in sorted(manager.arrays):
                info = manager.get_array_info(array_index)
                header = QLabel(
                    f"<b>{info['name']}</b>&nbsp;&nbsp;<i>x: {info['x_field']}</i>"
                )
                self._content_layout.addWidget(header)

                for field_name in manager.get_array_fields(array_index):
                    self._content_layout.addLayout(
                        self._build_field_row(array_index, field_name)
                    )

        self._size_popup_to_content()

    def _size_popup_to_content(self) -> None:
        # QScrollArea's own sizeHint ignores its content, which leaves the
        # menu at a stub height; force min sizes from the content instead
        self._content_layout.activate()
        self._content.adjustSize()
        hint = self._content.sizeHint()
        scrollbar_width = self._scroll.verticalScrollBar().sizeHint().width()
        frame = 2 * self._scroll.frameWidth()

        height = min(hint.height() + frame, self.max_popup_height)
        self._scroll.setMinimumHeight(height)
        self._scroll.setMaximumHeight(self.max_popup_height)
        self._scroll.setMinimumWidth(hint.width() + scrollbar_width + frame)

    def _build_field_row(self, array_index: int, field_name: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(16, 0, 0, 0)
        row.setSpacing(8)

        checkbox = QCheckBox(field_name)
        checkbox.setChecked(self.integration.is_field_visible(array_index, field_name))
        checkbox.toggled.connect(
            lambda checked, ai=array_index, fn=field_name: self._on_toggled(
                ai, fn, checked
            )
        )
        row.addWidget(checkbox, 1)

        multiplier = self.integration.get_multiplier(array_index, field_name)
        editor = QLineEdit(f"{multiplier:g}")
        editor.setMaximumWidth(80)
        editor.setToolTip("Y multiplier (scientific notation accepted)")
        editor.editingFinished.connect(
            lambda ed=editor, ai=array_index, fn=field_name: self._on_multiplier_edited(
                ed, ai, fn
            )
        )
        row.addWidget(editor)
        return row

    def _on_toggled(self, array_index: int, field_name: str, checked: bool) -> None:
        self.integration.set_field_enabled(array_index, field_name, checked)

    def _on_multiplier_edited(
        self,
        editor: QLineEdit,
        array_index: int,
        field_name: str,
    ) -> None:
        try:
            value = float(editor.text())
        except ValueError:
            current = self.integration.get_multiplier(array_index, field_name)
            editor.setText(f"{current:g}")
            return
        editor.setText(f"{value:g}")
        self.integration.set_multiplier(array_index, field_name, value)
