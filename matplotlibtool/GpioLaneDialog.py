#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget


class GpioLaneDialog(QDialog):
    """One checkbox per GPIO line.

    Unchecking a line removes it from the stack and the remaining lanes close
    ranks; the window stays open so the reflow is watched live. The GPIO
    checkbox on the control bar still gates the whole stack: choices made here
    wait behind it.
    """

    def __init__(
        self,
        parent: QWidget,
        *,
        lanes: list[tuple[str, bool]],
        on_toggle: Callable[[str, bool], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("GPIO Lines")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(self)

        hint = QLabel("A hidden line gives up its lane; the stack closes the gap.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        for name, shown in lanes:
            box = QCheckBox(name)
            box.setChecked(shown)
            box.toggled.connect(
                lambda checked, _name=name: on_toggle(_name, checked)
            )
            layout.addWidget(box)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
