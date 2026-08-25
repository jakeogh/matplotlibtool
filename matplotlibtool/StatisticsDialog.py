#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QPlainTextEdit
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget


class StatisticsDialog(QDialog):
    """Every plot's statistics for the current window, as selectable text.

    The rows in the figure margin are capped and elided to fit under the axes;
    this is the whole set, in a box that can be selected, scrolled and copied
    into a notebook. It does not follow the view: the numbers belong to the
    window they were taken from, and a panel that rewrote itself under the
    cursor while a number was being read would be worse than one that does not.
    Refresh takes them again for wherever the view is now.
    """

    def __init__(
        self,
        parent: QWidget,
        *,
        text: str,
        on_refresh: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Statistics")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(900, 400)
        self._on_refresh = on_refresh

        layout = QVBoxLayout(self)
        hint = QLabel(
            "The window these were taken from, not the one on screen now. "
            "Refresh to take them again."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._body = QPlainTextEdit()
        self._body.setReadOnly(True)
        self._body.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._body.setPlainText(text)
        font = self._body.font()
        font.setFamily("monospace")
        self._body.setFont(font)
        layout.addWidget(self._body)

        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)

        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy every row to the clipboard.")
        copy_btn.clicked.connect(self._copy)
        row.addWidget(copy_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Take the statistics again for the current view.")
        refresh_btn.clicked.connect(self._refresh)
        row.addWidget(refresh_btn)

        row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row.addWidget(close_btn)

        layout.addWidget(buttons)

    def _copy(self) -> None:
        self._body.selectAll()
        self._body.copy()
        cursor = self._body.textCursor()
        cursor.clearSelection()
        self._body.setTextCursor(cursor)

    def _refresh(self) -> None:
        self._body.setPlainText(self._on_refresh())
