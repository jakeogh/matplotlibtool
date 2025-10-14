#!/usr/bin/env python3
# tab-width:4

"""
Array Field Visibility Row

Provides a dynamic row of checkboxes to control which fields from the
currently selected structured array are plotted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from .ArrayFieldManager import ArrayFieldManager


class ArrayFieldVisibilitySignals(QObject):
    """Signal hub for array field visibility events."""

    fieldToggled = pyqtSignal(
        int,
        str,
        bool,
    )  # array_index, field_name, checked


class ArrayFieldVisibilityRow:
    """
    Manages a row of checkboxes for controlling field visibility.

    This creates a dynamic row that updates based on the currently selected
    array. Each checkbox corresponds to one field in the structured array.
    """

    def __init__(self, array_field_manager: ArrayFieldManager):
        """
        Initialize array field visibility row.

        Args:
            array_field_manager: Reference to the ArrayFieldManager instance
        """
        self.array_field_manager = array_field_manager
        self.signals = ArrayFieldVisibilitySignals()

        # Track checkbox widgets by field name
        self.checkboxes: dict[str, QCheckBox] = {}

        # Currently displayed array
        self.current_array_index: int | None = None

        # Main widget components
        self.container_widget: QWidget | None = None
        self.scroll_area: QScrollArea | None = None
        self.checkbox_container: QWidget | None = None
        self.checkbox_layout: QHBoxLayout | None = None
        self.info_label: QLabel | None = None

    def create_widget(self) -> QWidget:
        """
        Create the array field visibility row widget.

        Returns:
            QWidget containing the field visibility controls
        """
        # Main container
        self.container_widget = QWidget()
        container_layout = QHBoxLayout(self.container_widget)
        container_layout.setContentsMargins(
            8,
            4,
            8,
            4,
        )
        container_layout.setSpacing(8)

        # Label
        self.info_label = QLabel("Show Fields:")
        container_layout.addWidget(self.info_label)

        # Scrollable area for checkboxes
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(40)

        # Container for checkboxes inside scroll area
        self.checkbox_container = QWidget()
        self.checkbox_layout = QHBoxLayout(self.checkbox_container)
        self.checkbox_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        self.checkbox_layout.setSpacing(12)
        self.checkbox_layout.addStretch()

        self.scroll_area.setWidget(self.checkbox_container)
        container_layout.addWidget(self.scroll_area, 1)

        return self.container_widget

    def set_current_array(self, array_index: int) -> None:
        """
        Set the currently displayed array and rebuild checkboxes.

        Args:
            array_index: Index of the array to display
        """
        self.current_array_index = array_index
        self._rebuild_checkboxes()

    def _rebuild_checkboxes(self) -> None:
        """
        Rebuild all checkboxes based on current array's fields.
        """
        self._clear_checkboxes()

        if self.current_array_index is None:
            self.info_label.setText("Show Fields: (no array selected)")
            return

        # Get array info
        array_info = self.array_field_manager.get_array_info(self.current_array_index)
        if not array_info:
            self.info_label.setText("Show Fields: (invalid array)")
            return

        array_name = array_info["name"]
        self.info_label.setText(f"Show Fields ({array_name}):")

        # Get all fields for this array
        fields = self.array_field_manager.get_array_fields(self.current_array_index)

        if not fields:
            return

        # Create checkbox for each field
        for field_name in fields:
            is_plotted = self.array_field_manager.is_field_active(
                self.current_array_index, field_name
            )

            # Create checkbox
            checkbox = QCheckBox(field_name)
            checkbox.setChecked(is_plotted)

            # Connect to handler with field_name captured
            checkbox.toggled.connect(
                lambda checked, fname=field_name: self._on_checkbox_toggled(
                    fname, checked
                )
            )

            # Store reference
            self.checkboxes[field_name] = checkbox

            # Add to layout (before the stretch)
            self.checkbox_layout.insertWidget(
                self.checkbox_layout.count() - 1, checkbox
            )

    def _clear_checkboxes(self) -> None:
        """Remove all existing checkboxes from the layout."""
        for checkbox in self.checkboxes.values():
            self.checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()

        self.checkboxes.clear()

    def _on_checkbox_toggled(
        self,
        field_name: str,
        checked: bool,
    ) -> None:
        """
        Handle checkbox toggle event.

        Args:
            field_name: Name of the field
            checked: New checked state
        """
        if self.current_array_index is None:
            return

        # Emit signal for viewer to handle plot creation/removal
        self.signals.fieldToggled.emit(
            self.current_array_index,
            field_name,
            checked,
        )

        action = "enabled" if checked else "disabled"
        print(
            f"[INFO] Field '{field_name}' {action} for array {self.current_array_index}"
        )

    def sync_checkbox_state(
        self,
        field_name: str,
        checked: bool,
    ) -> None:
        """
        Synchronize a specific checkbox state.

        Args:
            field_name: Name of the field
            checked: Desired checked state
        """
        if field_name in self.checkboxes:
            checkbox = self.checkboxes[field_name]
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def sync_all_checkboxes(self) -> None:
        """
        Synchronize all checkbox states with ArrayFieldManager.
        """
        if self.current_array_index is None:
            return

        for field_name, checkbox in self.checkboxes.items():
            is_plotted = self.array_field_manager.is_field_active(
                self.current_array_index, field_name
            )
            checkbox.blockSignals(True)
            checkbox.setChecked(is_plotted)
            checkbox.blockSignals(False)
