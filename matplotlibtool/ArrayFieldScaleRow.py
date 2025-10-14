#!/usr/bin/env python3
# tab-width:4

"""
Array Field Scale Row

Provides a dynamic row of scale factor inputs to control Y-axis scaling
for each field from the currently selected structured array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from .ArrayFieldManager import ArrayFieldManager


class ArrayFieldScaleSignals(QObject):
    """Signal hub for array field scale events."""

    scaleChanged = pyqtSignal(
        int,
        str,
        float,
    )  # array_index, field_name, scale_factor


class ArrayFieldScaleRow:
    """
    Manages a row of scale factor inputs for controlling field Y-axis scaling.

    This creates a dynamic row that updates based on the currently selected
    array. Each input corresponds to one field in the structured array.
    """

    def __init__(self, array_field_manager: ArrayFieldManager):
        """
        Initialize array field scale row.

        Args:
            array_field_manager: Reference to the ArrayFieldManager instance
        """
        self.array_field_manager = array_field_manager
        self.signals = ArrayFieldScaleSignals()

        # Track scale input widgets by field name
        self.scale_inputs: dict[str, QDoubleSpinBox] = {}

        # Track current scale factors: field_name -> scale_factor
        self.current_scales: dict[str, float] = {}

        # Currently displayed array
        self.current_array_index: int | None = None

        # Main widget components
        self.container_widget: QWidget | None = None
        self.scroll_area: QScrollArea | None = None
        self.input_container: QWidget | None = None
        self.input_layout: QHBoxLayout | None = None
        self.info_label: QLabel | None = None

    def create_widget(self) -> QWidget:
        """
        Create the array field scale row widget.

        Returns:
            QWidget containing the scale factor controls
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
        self.info_label = QLabel("Scale Factors:")
        container_layout.addWidget(self.info_label)

        # Scrollable area for scale inputs
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(40)

        # Container for inputs inside scroll area
        self.input_container = QWidget()
        self.input_layout = QHBoxLayout(self.input_container)
        self.input_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        self.input_layout.setSpacing(6)
        self.input_layout.addStretch()

        self.scroll_area.setWidget(self.input_container)
        container_layout.addWidget(self.scroll_area, 1)

        return self.container_widget

    def set_current_array(self, array_index: int) -> None:
        """
        Set the currently displayed array and rebuild scale inputs.

        Args:
            array_index: Index of the array to display
        """
        # Prevent redundant rebuilds
        if self.current_array_index == array_index:
            return

        self.current_array_index = array_index
        self._rebuild_scale_inputs()

    def _rebuild_scale_inputs(self) -> None:
        """
        Rebuild all scale inputs based on current array's fields.
        """
        # Clear existing inputs
        self._clear_scale_inputs()

        if self.current_array_index is None:
            self.info_label.setText("Scale Factors: (no array selected)")
            return

        # Get array info
        array_info = self.array_field_manager.get_array_info(self.current_array_index)
        if not array_info:
            self.info_label.setText("Scale Factors: (invalid array)")
            return

        array_name = array_info["name"]
        self.info_label.setText(f"Scale Factors ({array_name}):")

        # Get all fields for this array
        fields = self.array_field_manager.get_array_fields(self.current_array_index)

        if not fields:
            return

        # Create scale input for each field
        for field_name in fields:
            current_scale = self.current_scales.get(field_name, 1.0)

            # Create container for label + spinbox
            field_container = QWidget()
            field_layout = QHBoxLayout(field_container)
            field_layout.setContentsMargins(
                0,
                0,
                0,
                0,
            )
            field_layout.setSpacing(2)

            # Create label
            label = QLabel(f"{field_name}:")
            field_layout.addWidget(label)

            # Create spin box
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(current_scale)
            spin.setFixedWidth(60)

            # Disable keyboard tracking - only apply on Enter
            spin.setKeyboardTracking(False)

            # Connect to handler with field_name captured
            spin.editingFinished.connect(
                lambda fname=field_name, s=spin: self._on_scale_changed(fname, s)
            )

            # Connect to value changed to highlight pending changes
            spin.valueChanged.connect(
                lambda value, fname=field_name, s=spin: self._on_scale_value_changed(
                    fname, s
                )
            )

            field_layout.addWidget(spin)

            # Store reference
            self.scale_inputs[field_name] = spin

            # Add container to main layout (before the stretch)
            self.input_layout.insertWidget(
                self.input_layout.count() - 1, field_container
            )

    def _clear_scale_inputs(self) -> None:
        """Remove all existing scale input widgets from the layout."""
        # Remove and delete all input widgets
        while self.input_layout.count() > 1:  # Keep the stretch at the end
            item = self.input_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()

        self.scale_inputs.clear()

    def _on_scale_value_changed(
        self,
        field_name: str,
        spin: QDoubleSpinBox,
    ) -> None:
        """
        Handle scale value changed (while typing) - highlight as pending.

        Args:
            field_name: Name of the field
            spin: The spinbox widget
        """
        current_stored = self.current_scales.get(field_name, 1.0)

        if abs(spin.value() - current_stored) > 1e-6:
            # Value changed but not applied - highlight with yellow background
            spin.setStyleSheet("QDoubleSpinBox { background-color: #FFFF99; }")
        else:
            # Value matches stored - clear highlight
            spin.setStyleSheet("")

    def _on_scale_changed(
        self,
        field_name: str,
        spin: QDoubleSpinBox,
    ) -> None:
        """
        Handle scale factor change (Enter pressed).

        Args:
            field_name: Name of the field
            spin: The spinbox widget
        """
        if self.current_array_index is None:
            return

        new_scale = spin.value()

        # Store the new scale factor
        self.current_scales[field_name] = new_scale

        # Clear highlight - change is now applied
        spin.setStyleSheet("")

        # Emit signal for viewer to handle scaling
        self.signals.scaleChanged.emit(
            self.current_array_index,
            field_name,
            new_scale,
        )

        print(f"[INFO] Scale factor for '{field_name}' changed to {new_scale:.3f}")

    def get_scale_factor(self, field_name: str) -> float:
        """
        Get the current scale factor for a field.

        Args:
            field_name: Name of the field

        Returns:
            Scale factor (default 1.0)
        """
        return self.current_scales.get(field_name, 1.0)

    def set_scale_factor(
        self,
        field_name: str,
        scale: float,
    ) -> None:
        """
        Set the scale factor for a field programmatically.

        Args:
            field_name: Name of the field
            scale: Scale factor to set
        """
        self.current_scales[field_name] = scale

        if field_name in self.scale_inputs:
            spin = self.scale_inputs[field_name]
            spin.blockSignals(True)
            spin.setValue(scale)
            spin.setStyleSheet("")  # Clear any highlight
            spin.blockSignals(False)

    def sync_all_scale_inputs(self) -> None:
        """
        Synchronize all scale input values with stored scale factors.
        """
        if self.current_array_index is None:
            return

        for field_name, spin in self.scale_inputs.items():
            stored_scale = self.current_scales.get(field_name, 1.0)
            spin.blockSignals(True)
            spin.setValue(stored_scale)
            spin.setStyleSheet("")  # Clear any highlight
            spin.blockSignals(False)
