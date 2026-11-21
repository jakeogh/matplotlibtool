#!/usr/bin/env python3
# tab-width:4

"""
FlowLayout - a row of controls that wraps instead of squeezing.

A box layout given less width than its controls ask for shrinks each of them
below what it needs, so a checkbox reads "Secon" and a label "Autocc", and
past their minimums the rest of the row is pushed off the right edge where
nothing can reach it. Laid out here, every control keeps the width it asks
for, and those that do not fit carry on along a new line under the first.
A label stays on the line of the control after it, which it names, so a
wrap never leaves "xmax:" at the end of one line and its field at the start
of the next. The height a width needs is reported to the parent layout, so
the lines under it move down rather than the controls overlapping them.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint
from PyQt6.QtCore import QRect
from PyQt6.QtCore import QSize
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLayout
from PyQt6.QtWidgets import QLayoutItem
from PyQt6.QtWidgets import QWidget
from PyQt6.QtWidgets import QWidgetItem


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, *, spacing: int = 8, line_spacing: int = 4):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self._line_spacing = line_spacing

    def addItem(self, item: QLayoutItem | None) -> None:
        if item is None:
            raise TypeError("FlowLayout.addItem: no item to add")
        self._items.append(item)

    def insertWidget(self, index: int, widget: QWidget) -> None:
        """`widget` placed at `index` among the items, as a box layout's."""
        self.addChildWidget(widget)
        self._items.insert(index, QWidgetItem(widget))
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), place=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, place=True)

    def sizeHint(self) -> QSize:
        """Every control on one line: what the row would like."""
        left, top, right, bottom = self.getContentsMargins()
        shown = [item.sizeHint() for item in self._items if not item.isEmpty()]
        width = sum(size.width() for size in shown) + self._spacing * max(len(shown) - 1, 0)
        height = max((size.height() for size in shown), default=0)
        return QSize(width + left + right, height + top + bottom)

    def minimumSize(self) -> QSize:
        """The widest run that wraps whole: as narrow as the row can be laid out."""
        left, top, right, bottom = self.getContentsMargins()
        size = QSize()
        for unit in self._units():
            width = sum(item.minimumSize().width() for item, _ in unit) + self._spacing * (len(unit) - 1)
            size = size.expandedTo(QSize(width, max(item.minimumSize().height() for item, _ in unit)))
        return QSize(size.width() + left + right, size.height() + top + bottom)

    def _units(self) -> list[list[tuple[QLayoutItem, QSize]]]:
        """The shown controls in the runs that wrap whole: a label and the
        control it names, else a control alone."""
        units: list[list[tuple[QLayoutItem, QSize]]] = []
        joined = False
        for item in self._items:
            if item.isEmpty():
                continue
            if joined:
                units[-1].append((item, item.sizeHint()))
            else:
                units.append([(item, item.sizeHint())])
            joined = isinstance(item.widget(), QLabel)
        return units

    def _arrange(self, rect: QRect, *, place: bool) -> int:
        """Lay the controls out in lines across `rect`, each centred on its
        line's height, placing them when `place` is set; the height the
        lines take either way."""
        left, top, right, bottom = self.getContentsMargins()
        x0, y = rect.x() + left, rect.y() + top
        x1 = rect.x() + rect.width() - right
        lines: list[list[tuple[QLayoutItem, QSize]]] = [[]]
        x = x0
        for unit in self._units():
            width = sum(size.width() for _, size in unit) + self._spacing * (len(unit) - 1)
            if lines[-1] and x + width > x1:
                lines.append([])
                x = x0
            lines[-1].extend(unit)
            x += width + self._spacing
        for line in lines:
            if not line:
                continue
            height = max(size.height() for _, size in line)
            if place:
                x = x0
                for item, size in line:
                    item.setGeometry(QRect(QPoint(x, y + (height - size.height()) // 2), size))
                    x += size.width() + self._spacing
            y += height + self._line_spacing
        return y - self._line_spacing - rect.y() + bottom if any(lines) else top + bottom
