#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget


class GpioLaneDialog(QDialog):
    """One checkbox per GPIO line, and the stroke width of the lanes.

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
        line_width: float,
        on_line_width: Callable[[float], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("GPIO Lines")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(self)

        hint = QLabel("A hidden line gives up its lane; the stack closes the gap.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        width_row = QWidget()
        width_layout = QHBoxLayout(width_row)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.addWidget(QLabel("Line width:"))
        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.1, 10.0)
        width_spin.setSingleStep(0.25)
        width_spin.setDecimals(2)
        width_spin.setValue(line_width)
        width_spin.valueChanged.connect(on_line_width)
        width_layout.addWidget(width_spin)
        width_layout.addStretch()
        layout.addWidget(width_row)

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
