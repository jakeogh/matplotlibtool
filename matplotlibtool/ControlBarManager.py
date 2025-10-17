#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from .AxisSecondaryConfig import AxisSecondaryConfig


class ControlBarSignals(QObject):
    """Signal hub for control bar events."""

    # File operations
    addRequested = pyqtSignal()
    resetRequested = pyqtSignal()
    exitRequested = pyqtSignal()
    saveFigureRequested = pyqtSignal()

    # Plot operations
    plotChanged = pyqtSignal(int)
    groupSelectionChanged = pyqtSignal(int)
    visibilityToggled = pyqtSignal(bool)

    # Rendering controls
    accelChanged = pyqtSignal(float)
    sizeChanged = pyqtSignal(float)
    lineWidthChanged = pyqtSignal(float)
    linesToggled = pyqtSignal(bool)
    paletteChanged = pyqtSignal(str)
    darkModeToggled = pyqtSignal(bool)
    colorFieldChanged = pyqtSignal(str)

    # Grid controls
    gridSpacingChanged = pyqtSignal(str)
    axesGridColorPickRequested = pyqtSignal()
    adcGridColorPickRequested = pyqtSignal()

    # View controls
    fitViewRequested = pyqtSignal()
    applyViewRequested = pyqtSignal()
    applyOffsetRequested = pyqtSignal()

    # View bounds changed
    viewBoundsChanged = pyqtSignal(
        str,
        str,
        str,
        str,
    )

    # Secondary axis signals
    secondaryAxisToggled = pyqtSignal(bool)
    secondaryAxisConfigRequested = pyqtSignal(object)


class ControlBarManager:
    """
    Enhanced control bar manager with secondary Y-axis support.

    Handles:
    - Dynamic control bar creation
    - Signal routing and management
    - Control state synchronization
    - Widget factory methods
    - Secondary Y-axis configuration
    """

    def __init__(
        self,
        parent_widget: QWidget,
        palette_groups: dict,
    ):
        """
        Initialize control bar manager.

        Args:
            parent_widget: Parent widget for the control bars
            palette_groups: Dictionary of color palette groups
        """
        self.parent = parent_widget
        self.palette_groups = palette_groups
        self.signals = ControlBarSignals()

        # Store references to created widgets
        self.widgets = {}
        self.layouts = {}

        # Control state tracking
        self._block_signals = False

        # Secondary axis widgets
        self.secondary_axis_widgets = {}

    def create_four_row_controls(self) -> QWidget:
        """
        Create the complete four-row control layout including secondary axis.

        Returns:
            Widget containing all control rows
        """
        controls_widget = QWidget(self.parent)
        main_layout = QVBoxLayout(controls_widget)
        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        main_layout.setSpacing(0)

        row1 = self._create_row1()
        row2 = self._create_row2()
        row3 = self._create_row3()
        row4 = self._create_secondary_axis_row()

        main_layout.addWidget(row1)
        main_layout.addWidget(row2)
        main_layout.addWidget(row3)
        main_layout.addWidget(row4)

        self.layouts["main"] = main_layout
        self.layouts["row1"] = row1.layout()
        self.layouts["row2"] = row2.layout()
        self.layouts["row3"] = row3.layout()
        self.layouts["row4"] = row4.layout()

        return controls_widget

    def create_three_row_controls(self) -> QWidget:
        """
        Create the original three-row control layout (backwards compatibility).

        Returns:
            Widget containing all control rows
        """
        controls_widget = QWidget(self.parent)
        main_layout = QVBoxLayout(controls_widget)
        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        main_layout.setSpacing(0)

        row1 = self._create_row1()
        row2 = self._create_row2()
        row3 = self._create_row3()

        main_layout.addWidget(row1)
        main_layout.addWidget(row2)
        main_layout.addWidget(row3)

        self.layouts["main"] = main_layout
        self.layouts["row1"] = row1.layout()
        self.layouts["row2"] = row2.layout()
        self.layouts["row3"] = row3.layout()

        return controls_widget

    def populate_hierarchical_dropdown(self, plot_manager) -> None:
        """
        Populate the plot/group dropdown with hierarchical structure.

        Args:
            plot_manager: PlotManager instance with group and plot info
        """
        combo = self.widgets["plot_combo"]
        if combo is None:
            return

        combo.blockSignals(True)

        old_selection_data = combo.currentData()

        combo.clear()

        current_selection_type = None
        current_selection_id = None

        if old_selection_data is not None and len(old_selection_data) == 2:
            current_selection_type, current_selection_id = old_selection_data
        elif plot_manager.is_group_selected():
            current_selection_type = "group"
            current_selection_id = plot_manager.selected_group_id
        else:
            current_selection_type = "plot"
            current_selection_id = plot_manager.selected_plot_index

        groups = plot_manager.get_all_groups()

        grouped_plot_indices = set()
        for group_info in groups:
            grouped_plot_indices.update(group_info.plot_indices)

        selection_index = -1  # Will be set to last group if no specific selection
        current_index = 0
        last_group_index = -1  # Track the last group added

        # Add groups with their plots
        for group_info in groups:
            # Add group header
            group_label = (
                f"📦 {group_info.group_name} ({len(group_info.plot_indices)} plots)"
            )
            combo.addItem(group_label)
            combo.setItemData(current_index, ("group", group_info.group_id))

            if (
                current_selection_type == "group"
                and current_selection_id == group_info.group_id
            ):
                selection_index = current_index

            last_group_index = current_index  # Track last group
            current_index += 1

            # Add individual plots in this group (indented)
            for plot_index in group_info.plot_indices:
                if plot_index < len(plot_manager.plots):
                    plot = plot_manager.plots[plot_index]
                    custom_name = plot_manager.plot_names.get(plot_index, None)

                    if custom_name:
                        plot_label = f"  └─ {custom_name} ({len(plot.points):,} pts)"
                    else:
                        plot_label = (
                            f"  └─ Plot {plot_index + 1} ({len(plot.points):,} pts)"
                        )

                    combo.addItem(plot_label)
                    combo.setItemData(current_index, ("plot", plot_index))

                    if (
                        current_selection_type == "plot"
                        and current_selection_id == plot_index
                    ):
                        selection_index = current_index

                    current_index += 1

        # Add ungrouped plots (if any)
        for plot_index in range(len(plot_manager.plots)):
            if plot_index not in grouped_plot_indices:
                plot = plot_manager.plots[plot_index]
                custom_name = plot_manager.plot_names.get(plot_index, None)

                if custom_name:
                    plot_label = f"{custom_name} ({len(plot.points):,} pts)"
                else:
                    plot_label = f"Plot {plot_index + 1} ({len(plot.points):,} pts)"

                combo.addItem(plot_label)
                combo.setItemData(current_index, ("plot", plot_index))

                if (
                    current_selection_type == "plot"
                    and current_selection_id == plot_index
                ):
                    selection_index = current_index

                current_index += 1

        # If no specific selection was found, default to last group
        if selection_index == -1 and last_group_index != -1:
            selection_index = last_group_index
        elif selection_index == -1:
            selection_index = 0  # Fallback to first item if no groups

        # Set the current selection
        combo.setCurrentIndex(selection_index)

        combo.blockSignals(False)

    def _create_row1(self) -> QWidget:
        """Create first control row: Add, Plot/Group selector, Visible, Accel, Size, Lines, Palette, Color Field."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(
            8,
            4,
            8,
            4,
        )
        layout.setSpacing(8)

        add_btn = QPushButton("Add…")
        add_btn.setMaximumWidth(60)
        add_btn.clicked.connect(self.signals.addRequested.emit)
        layout.addWidget(add_btn)
        self.widgets["add_btn"] = add_btn

        layout.addWidget(QLabel("Plot/Group:"))
        plot_combo = QComboBox()
        plot_combo.setMaximumWidth(300)
        plot_combo.currentIndexChanged.connect(self._on_plot_group_selection_changed)
        layout.addWidget(plot_combo)
        self.widgets["plot_combo"] = plot_combo

        visible_chk = QCheckBox("Visible")
        visible_chk.setChecked(True)
        visible_chk.toggled.connect(self.signals.visibilityToggled.emit)
        layout.addWidget(visible_chk)
        self.widgets["visible_chk"] = visible_chk

        layout.addWidget(QLabel("Accel:"))
        accel_spin = QDoubleSpinBox()
        accel_spin.setRange(1.001, 5.0)
        accel_spin.setSingleStep(0.01)
        accel_spin.setDecimals(3)
        accel_spin.setMaximumWidth(80)
        accel_spin.valueChanged.connect(self.signals.accelChanged.emit)
        layout.addWidget(accel_spin)
        self.widgets["accel_spin"] = accel_spin

        layout.addWidget(QLabel("Size:"))
        size_spin = QDoubleSpinBox()
        size_spin.setRange(0.1, 1000.0)
        size_spin.setSingleStep(0.1)
        size_spin.setDecimals(3)
        size_spin.setMaximumWidth(80)
        size_spin.setKeyboardTracking(False)
        size_spin.editingFinished.connect(
            lambda: self.signals.sizeChanged.emit(size_spin.value())
        )
        layout.addWidget(size_spin)
        self.widgets["size_spin"] = size_spin

        layout.addWidget(QLabel("Line Width:"))
        line_width_spin = QDoubleSpinBox()
        line_width_spin.setRange(0.1, 10.0)
        line_width_spin.setSingleStep(0.1)
        line_width_spin.setDecimals(2)
        line_width_spin.setMaximumWidth(80)
        line_width_spin.setKeyboardTracking(False)
        line_width_spin.editingFinished.connect(
            lambda: self.signals.lineWidthChanged.emit(line_width_spin.value())
        )
        layout.addWidget(line_width_spin)
        self.widgets["line_width_spin"] = line_width_spin

        lines_chk = QCheckBox("Lines")
        lines_chk.toggled.connect(self.signals.linesToggled.emit)
        layout.addWidget(lines_chk)
        self.widgets["lines_chk"] = lines_chk

        dark_mode_chk = QCheckBox("Dark")
        dark_mode_chk.setChecked(True)
        dark_mode_chk.toggled.connect(self.signals.darkModeToggled.emit)
        layout.addWidget(dark_mode_chk)
        self.widgets["dark_mode_chk"] = dark_mode_chk

        layout.addWidget(QLabel("Palette:"))
        palette_combo = QComboBox()
        palette_combo.setMaximumWidth(160)
        self._populate_palette_combo(palette_combo)
        palette_combo.currentTextChanged.connect(self._on_palette_changed)
        palette_combo.currentIndexChanged.connect(self._on_palette_index_changed)
        layout.addWidget(palette_combo)
        self.widgets["palette_combo"] = palette_combo

        layout.addWidget(QLabel("Color:"))
        color_field_combo = QComboBox()
        color_field_combo.setMaximumWidth(120)
        color_field_combo.currentTextChanged.connect(self._on_color_field_changed)
        layout.addWidget(color_field_combo)
        self.widgets["color_field_combo"] = color_field_combo

        layout.addStretch()
        return row

    def _on_color_field_changed(self, field_name: str):
        """Handle color field selection change."""
        if field_name and not field_name.startswith("───"):
            self.signals.colorFieldChanged.emit(field_name)

    def populate_color_field_combo(
        self,
        field_names: list[str],
        current_field: str | None = None,
    ):
        """
        Populate the color field dropdown with available fields.

        Args:
            field_names: List of field names available for coloring
            current_field: Currently selected color field (or None)
        """
        combo = self.widgets.get("color_field_combo")

        if combo is None:
            return

        combo.blockSignals(True)
        combo.clear()

        if not field_names:
            combo.addItem("(No fields)")
            combo.setEnabled(False)
        else:
            for field in field_names:
                combo.addItem(field)

            if current_field and current_field in field_names:
                idx = combo.findText(current_field)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

            combo.setEnabled(True)

        combo.blockSignals(False)

    def set_color_field(self, field_name: str | None):
        """
        Set the selected color field in the dropdown.

        Args:
            field_name: Name of the field to select, or None
        """
        combo = self.widgets.get("color_field_combo")
        if not combo:
            return

        combo.blockSignals(True)

        if field_name:
            idx = combo.findText(field_name)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        combo.blockSignals(False)

    def _on_plot_group_selection_changed(self, index: int):
        """Handle hierarchical plot/group selection changes."""
        combo = self.widgets.get("plot_combo")
        if not combo or index < 0:
            return

        item_data = combo.itemData(index)

        if item_data is None:
            return

        item_type, item_id = item_data

        if item_type == "group":
            self.signals.groupSelectionChanged.emit(item_id)
        elif item_type == "plot":
            self.signals.plotChanged.emit(item_id)

    def _create_row2(self) -> QWidget:
        """Create second control row: Grid controls."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(
            8,
            4,
            8,
            4,
        )
        layout.setSpacing(8)

        layout.addWidget(QLabel("Axes Grid:"))
        axes_grid_btn = QPushButton("Pick")
        axes_grid_btn.clicked.connect(self.signals.axesGridColorPickRequested.emit)
        layout.addWidget(axes_grid_btn)
        self.widgets["axes_grid_btn"] = axes_grid_btn

        layout.addWidget(QLabel("Grid 2^N:"))
        adc_grid_btn = QPushButton("Pick")
        adc_grid_btn.clicked.connect(self.signals.adcGridColorPickRequested.emit)
        layout.addWidget(adc_grid_btn)
        self.widgets["adc_grid_btn"] = adc_grid_btn

        layout.addWidget(QLabel("Spacing:"))
        grid_combo = QComboBox()
        grid_combo.setMaximumWidth(110)
        self._populate_grid_combo(grid_combo)
        grid_combo.currentTextChanged.connect(self.signals.gridSpacingChanged.emit)
        layout.addWidget(grid_combo)
        self.widgets["grid_combo"] = grid_combo

        save_fig_btn = QPushButton("Save Figure")
        save_fig_btn.setMaximumWidth(90)
        save_fig_btn.clicked.connect(self.signals.saveFigureRequested.emit)
        layout.addWidget(save_fig_btn)
        self.widgets["save_fig_btn"] = save_fig_btn

        reset_btn = QPushButton("Reset View")
        reset_btn.setMaximumWidth(90)
        reset_btn.clicked.connect(self.signals.resetRequested.emit)
        layout.addWidget(reset_btn)
        self.widgets["reset_btn"] = reset_btn

        exit_btn = QPushButton("Exit")
        exit_btn.setMaximumWidth(60)
        exit_btn.clicked.connect(self.signals.exitRequested.emit)
        layout.addWidget(exit_btn)
        self.widgets["exit_btn"] = exit_btn

        info_label = QLabel("")
        layout.addWidget(info_label)
        self.widgets["info_label"] = info_label

        status_label = QLabel("")
        status_label.setMinimumWidth(60)
        status_label.setMaximumWidth(80)
        layout.addWidget(status_label)
        self.widgets["status_label"] = status_label

        layout.addStretch()
        return row

    def _create_row3(self) -> QWidget:
        """Create third control row: View bounds and offset controls."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(
            8,
            4,
            8,
            4,
        )
        layout.setSpacing(8)

        fit_view_btn = QPushButton("Fit View")
        fit_view_btn.setMaximumWidth(80)
        fit_view_btn.clicked.connect(self.signals.fitViewRequested.emit)
        fit_view_btn.setToolTip("Fit view to show all data with original aspect ratio")
        layout.addWidget(fit_view_btn)
        self.widgets["fit_view_btn"] = fit_view_btn

        bounds_widgets = self._create_view_bounds_controls()
        for widget in bounds_widgets:
            layout.addWidget(widget)

        layout.addWidget(QLabel("Offset"))

        offset_widgets = self._create_offset_controls()
        for widget in offset_widgets:
            layout.addWidget(widget)

        layout.addStretch()
        return row

    def _create_secondary_axis_row(self) -> QWidget:
        """Create fourth control row: Secondary Y-axis configuration."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(
            8,
            4,
            8,
            4,
        )
        layout.setSpacing(8)

        secondary_enable_chk = QCheckBox("Secondary Y-Axis")
        secondary_enable_chk.toggled.connect(self._on_secondary_axis_toggled)
        layout.addWidget(secondary_enable_chk)
        self.secondary_axis_widgets["enable"] = secondary_enable_chk

        layout.addWidget(QLabel("Primary:"))

        primary_min_edit = QLineEdit()
        primary_min_edit.setPlaceholderText("min (e.g., -8388608)")
        primary_min_edit.setMaximumWidth(100)
        primary_min_edit.setEnabled(False)
        layout.addWidget(primary_min_edit)
        self.secondary_axis_widgets["primary_min"] = primary_min_edit

        layout.addWidget(QLabel("to"))

        primary_max_edit = QLineEdit()
        primary_max_edit.setPlaceholderText("max (e.g., 8388607)")
        primary_max_edit.setMaximumWidth(100)
        primary_max_edit.setEnabled(False)
        layout.addWidget(primary_max_edit)
        self.secondary_axis_widgets["primary_max"] = primary_max_edit

        layout.addWidget(QLabel("→"))

        layout.addWidget(QLabel("Secondary:"))

        secondary_min_edit = QLineEdit()
        secondary_min_edit.setPlaceholderText("min (e.g., -5.0)")
        secondary_min_edit.setMaximumWidth(80)
        secondary_min_edit.setEnabled(False)
        layout.addWidget(secondary_min_edit)
        self.secondary_axis_widgets["secondary_min"] = secondary_min_edit

        layout.addWidget(QLabel("to"))

        secondary_max_edit = QLineEdit()
        secondary_max_edit.setPlaceholderText("max (e.g., 5.0)")
        secondary_max_edit.setMaximumWidth(80)
        secondary_max_edit.setEnabled(False)
        layout.addWidget(secondary_max_edit)
        self.secondary_axis_widgets["secondary_max"] = secondary_max_edit

        label_edit = QLineEdit()
        label_edit.setPlaceholderText("Label (e.g., Voltage)")
        label_edit.setMaximumWidth(100)
        label_edit.setEnabled(False)
        layout.addWidget(label_edit)
        self.secondary_axis_widgets["label"] = label_edit

        unit_edit = QLineEdit()
        unit_edit.setPlaceholderText("Unit (e.g., V)")
        unit_edit.setMaximumWidth(60)
        unit_edit.setEnabled(False)
        layout.addWidget(unit_edit)
        self.secondary_axis_widgets["unit"] = unit_edit

        apply_btn = QPushButton("Apply")
        apply_btn.setMaximumWidth(60)
        apply_btn.clicked.connect(self._on_apply_secondary_axis)
        apply_btn.setEnabled(False)
        layout.addWidget(apply_btn)
        self.secondary_axis_widgets["apply"] = apply_btn

        layout.addStretch()
        return row

    def _create_view_bounds_controls(self) -> list[QWidget]:
        """Create view bounds input controls."""
        widgets = []

        widgets.append(QLabel("xmin:"))
        xmin_edit = QLineEdit()
        xmin_edit.setMaximumWidth(80)
        xmin_edit.setPlaceholderText("auto")
        xmin_edit.returnPressed.connect(self.signals.applyViewRequested.emit)
        widgets.append(xmin_edit)
        self.widgets["xmin_edit"] = xmin_edit

        widgets.append(QLabel("xmax:"))
        xmax_edit = QLineEdit()
        xmax_edit.setMaximumWidth(80)
        xmax_edit.setPlaceholderText("auto")
        xmax_edit.returnPressed.connect(self.signals.applyViewRequested.emit)
        widgets.append(xmax_edit)
        self.widgets["xmax_edit"] = xmax_edit

        widgets.append(QLabel("ymin:"))
        ymin_edit = QLineEdit()
        ymin_edit.setMaximumWidth(80)
        ymin_edit.setPlaceholderText("auto")
        ymin_edit.returnPressed.connect(self.signals.applyViewRequested.emit)
        widgets.append(ymin_edit)
        self.widgets["ymin_edit"] = ymin_edit

        widgets.append(QLabel("ymax:"))
        ymax_edit = QLineEdit()
        ymax_edit.setMaximumWidth(80)
        ymax_edit.setPlaceholderText("auto")
        ymax_edit.returnPressed.connect(self.signals.applyViewRequested.emit)
        widgets.append(ymax_edit)
        self.widgets["ymax_edit"] = ymax_edit

        return widgets

    def _create_offset_controls(self) -> list[QWidget]:
        """Create offset input controls."""
        widgets = []

        widgets.append(QLabel("X:"))
        offset_x_spin = QDoubleSpinBox()
        offset_x_spin.setRange(-1e12, 1e12)
        offset_x_spin.setDecimals(6)
        offset_x_spin.setSingleStep(0.1)
        offset_x_spin.setMaximumWidth(100)
        offset_x_spin.setKeyboardTracking(False)
        offset_x_spin.editingFinished.connect(self.signals.applyOffsetRequested.emit)
        widgets.append(offset_x_spin)
        self.widgets["offset_x_spin"] = offset_x_spin

        widgets.append(QLabel("Y:"))
        offset_y_spin = QDoubleSpinBox()
        offset_y_spin.setRange(-1e12, 1e12)
        offset_y_spin.setDecimals(6)
        offset_y_spin.setSingleStep(0.1)
        offset_y_spin.setMaximumWidth(100)
        offset_y_spin.setKeyboardTracking(False)
        offset_y_spin.editingFinished.connect(self.signals.applyOffsetRequested.emit)
        widgets.append(offset_y_spin)
        self.widgets["offset_y_spin"] = offset_y_spin

        return widgets

    def _populate_palette_combo(self, combo: QComboBox):
        """Populate palette combobox with grouped palettes."""
        combo.clear()
        for category, palettes in self.palette_groups.items():
            combo.addItem(f"───〖{category}〗───")
            idx = combo.count() - 1
            item = combo.model().item(idx)
            item.setEnabled(False)
            for p in palettes:
                combo.addItem(p)

    def _populate_grid_combo(self, combo: QComboBox):
        """Populate grid spacing combobox."""
        combo.clear()
        combo.addItem("Off")
        for n in range(1, 25):
            spacing = 2**n
            combo.addItem(f"2^{n} ({spacing})" if n <= 10 else f"2^{n}")
        combo.setCurrentIndex(0)

    def _on_palette_changed(self, name: str):
        """Handle text-based palette changes."""
        if name.startswith("───"):
            return
        self.signals.paletteChanged.emit(name)

    def _on_palette_index_changed(self, index: int):
        """Handle index-based palette changes."""
        if index < 0:
            return
        combo = self.widgets.get("palette_combo")
        if combo:
            name = combo.itemText(index)
            if name.startswith("───"):
                return
            self.signals.paletteChanged.emit(name)

    def _on_secondary_axis_toggled(self, enabled: bool):
        """Handle secondary axis enable/disable."""
        for key, widget in self.secondary_axis_widgets.items():
            if key != "enable":
                widget.setEnabled(enabled)

        if enabled:
            if not self.secondary_axis_widgets["primary_min"].text().strip():
                self.secondary_axis_widgets["primary_min"].setText("-8388608")
            if not self.secondary_axis_widgets["primary_max"].text().strip():
                self.secondary_axis_widgets["primary_max"].setText("8388607")
            if not self.secondary_axis_widgets["secondary_min"].text().strip():
                self.secondary_axis_widgets["secondary_min"].setText("-5.0")
            if not self.secondary_axis_widgets["secondary_max"].text().strip():
                self.secondary_axis_widgets["secondary_max"].setText("5.0")
            if not self.secondary_axis_widgets["label"].text().strip():
                self.secondary_axis_widgets["label"].setText("Voltage")
            if not self.secondary_axis_widgets["unit"].text().strip():
                self.secondary_axis_widgets["unit"].setText("V")

        self.signals.secondaryAxisToggled.emit(enabled)

        if not enabled:
            print("[INFO] Secondary axis disabled")
        else:
            print("[INFO] Secondary axis enabled with default ADC→Voltage mapping")

    def _on_apply_secondary_axis(self):
        """Apply secondary axis configuration."""
        try:
            primary_min_text = self.secondary_axis_widgets["primary_min"].text().strip()
            primary_max_text = self.secondary_axis_widgets["primary_max"].text().strip()
            secondary_min_text = (
                self.secondary_axis_widgets["secondary_min"].text().strip()
            )
            secondary_max_text = (
                self.secondary_axis_widgets["secondary_max"].text().strip()
            )
            label_text = self.secondary_axis_widgets["label"].text().strip()
            unit_text = self.secondary_axis_widgets["unit"].text().strip()

            if not all(
                [
                    primary_min_text,
                    primary_max_text,
                    secondary_min_text,
                    secondary_max_text,
                    label_text,
                ]
            ):
                print(
                    "[ERROR] All fields except unit are required for secondary axis configuration"
                )
                return

            config = AxisSecondaryConfig.from_range_mapping(
                primary_min=float(primary_min_text),
                primary_max=float(primary_max_text),
                secondary_min=float(secondary_min_text),
                secondary_max=float(secondary_max_text),
                label=label_text,
                unit=unit_text,
            )

            self.signals.secondaryAxisConfigRequested.emit(config)
            print(
                f"[INFO] Secondary axis configuration applied: {config.label} ({config.unit})"
            )

        except ValueError as e:
            print(f"[ERROR] Invalid secondary axis configuration: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to apply secondary axis configuration: {e}")

    def set_dark_mode_checked(self, checked: bool):
        """Set dark mode checkbox state."""
        chk = self.widgets.get("dark_mode_chk")
        if chk:
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)

    def set_plots(
        self,
        labels: list[str],
        current_index: int = 0,
    ):
        """Set plot selector options."""
        combo = self.widgets.get("plot_combo")
        if combo is None:
            return

        combo.blockSignals(True)
        combo.clear()

        if not labels:
            combo.addItem("No plots available")
            combo.setCurrentIndex(0)
        else:
            for i, label in enumerate(labels):
                combo.addItem(label)

            if labels:
                idx = min(max(current_index, 0), len(labels) - 1)
                combo.setCurrentIndex(idx)
            else:
                combo.setCurrentIndex(0)

        combo.blockSignals(False)
        combo.update()
        combo.repaint()

    def set_accel(self, value: float):
        """Set acceleration value."""
        spin = self.widgets.get("accel_spin")
        if spin:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def set_point_size(self, value: float):
        """Set point size value."""
        spin = self.widgets.get("size_spin")
        if spin:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def set_line_width(self, width: float):
        """Set line width spinbox value."""
        spin = self.widgets.get("line_width_spin")
        if spin:
            spin.blockSignals(True)
            spin.setValue(width)
            spin.blockSignals(False)

    def set_line_width_mixed(self):
        """Set line width to mixed state."""
        spin = self.widgets.get("line_width_spin")
        if spin:
            spin.blockSignals(True)
            spin.setSpecialValueText("Mixed")
            spin.setValue(spin.minimum())
            spin.blockSignals(False)

    def set_lines_checked(self, checked: bool):
        """Set lines checkbox state."""
        chk = self.widgets.get("lines_chk")
        if chk:
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)

    def set_visibility_checked(self, checked: bool):
        """Set visibility checkbox state."""
        chk = self.widgets.get("visible_chk")
        if chk:
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)

    def set_selected_palette(self, name: str):
        """Set selected palette."""
        combo = self.widgets.get("palette_combo")
        if combo:
            idx = combo.findText(name)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

    def set_palette_enabled(self, enabled: bool):
        """Enable/disable palette control."""
        combo = self.widgets.get("palette_combo")
        if combo:
            combo.setEnabled(enabled)

    def set_axes_grid_color_swatch(self, hex_color: str):
        """Set axes grid color button appearance."""
        btn = self.widgets.get("axes_grid_btn")
        if btn:
            self._apply_color_swatch(btn, hex_color)

    def set_adc_grid_color_swatch(self, hex_color: str):
        """Set ADC grid color button appearance."""
        btn = self.widgets.get("adc_grid_btn")
        if btn:
            self._apply_color_swatch(btn, hex_color)

    def set_info_text(self, text: str):
        """Set info label text."""
        label = self.widgets.get("info_label")
        if label:
            label.setText(text)

    def set_offset(
        self,
        x: float,
        y: float,
    ):
        """Set offset spinbox values."""
        x_spin = self.widgets.get("offset_x_spin")
        y_spin = self.widgets.get("offset_y_spin")

        if x_spin:
            x_spin.blockSignals(True)
            x_spin.setValue(x)
            x_spin.blockSignals(False)

        if y_spin:
            y_spin.blockSignals(True)
            y_spin.setValue(y)
            y_spin.blockSignals(False)

    def set_view_bounds(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ):
        """Set view bounds text field values."""
        self._set_text_field("xmin_edit", f"{xmin:.6g}")
        self._set_text_field("xmax_edit", f"{xmax:.6g}")
        self._set_text_field("ymin_edit", f"{ymin:.6g}")
        self._set_text_field("ymax_edit", f"{ymax:.6g}")

    def get_view_bounds(self) -> tuple[str, str, str, str]:
        """Get current view bounds from text fields."""
        xmin = self._get_text_field("xmin_edit")
        xmax = self._get_text_field("xmax_edit")
        ymin = self._get_text_field("ymin_edit")
        ymax = self._get_text_field("ymax_edit")
        return xmin, xmax, ymin, ymax

    def get_offset_values(self) -> tuple[float, float]:
        """Get current offset values."""
        x_spin = self.widgets.get("offset_x_spin")
        y_spin = self.widgets.get("offset_y_spin")

        x_val = x_spin.value() if x_spin else 0.0
        y_val = y_spin.value() if y_spin else 0.0

        return x_val, y_val

    def set_secondary_axis_enabled(self, enabled: bool):
        """Set secondary axis checkbox state."""
        chk = self.secondary_axis_widgets.get("enable")
        if chk:
            chk.blockSignals(True)
            chk.setChecked(enabled)
            chk.blockSignals(False)

            for key, widget in self.secondary_axis_widgets.items():
                if key != "enable":
                    widget.setEnabled(enabled)

    def set_secondary_axis_config(self, config: AxisSecondaryConfig | None):
        """Set secondary axis configuration values."""
        if config is None:
            for key in [
                "primary_min",
                "primary_max",
                "secondary_min",
                "secondary_max",
                "label",
                "unit",
            ]:
                widget = self.secondary_axis_widgets.get(key)
                if widget and hasattr(widget, "setText"):
                    widget.setText("")
            return

        widgets = self.secondary_axis_widgets

        if widgets.get("label"):
            widgets["label"].setText(config.label)

        if widgets.get("unit"):
            widgets["unit"].setText(config.unit)

        primary_min_text = widgets.get("primary_min", None)
        primary_max_text = widgets.get("primary_max", None)

        if primary_min_text and primary_min_text.text().strip():
            primary_min = float(primary_min_text.text())
        else:
            primary_min = -8388608
            if widgets.get("primary_min"):
                widgets["primary_min"].setText(str(primary_min))

        if primary_max_text and primary_max_text.text().strip():
            primary_max = float(primary_max_text.text())
        else:
            primary_max = 8388607
            if widgets.get("primary_max"):
                widgets["primary_max"].setText(str(primary_max))

        secondary_min = config.scale * primary_min + config.offset
        secondary_max = config.scale * primary_max + config.offset

        if widgets.get("secondary_min"):
            widgets["secondary_min"].setText(f"{secondary_min:.3f}")

        if widgets.get("secondary_max"):
            widgets["secondary_max"].setText(f"{secondary_max:.3f}")

    def _apply_color_swatch(
        self,
        button: QPushButton,
        hex_color: str,
    ):
        """Apply color swatch styling to button."""
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {hex_color};
                border: 1px solid #656565;
                padding: 4px 10px;
                min-width: 48px;
                color: black;
            }}
            QPushButton:hover {{ border: 1px solid #9a9a9a; }}
            """
        )

    def _set_text_field(
        self,
        widget_name: str,
        text: str,
    ):
        """Set text field value with signal blocking."""
        widget = self.widgets.get(widget_name)
        if widget and hasattr(widget, "setText"):
            widget.blockSignals(True)
            widget.setText(text)
            widget.blockSignals(False)

    def _get_text_field(self, widget_name: str) -> str:
        """Get text field value."""
        widget = self.widgets.get(widget_name)
        if widget and hasattr(widget, "text"):
            return widget.text().strip()
        return ""

    def get_widget(self, name: str) -> QWidget | None:
        """Get widget by name."""
        widget = self.widgets.get(name)
        return widget

    def connect_signals(self, handler_map: dict[str, Callable]):
        """Connect signals to handlers using a mapping."""
        for signal_name, handler in handler_map.items():
            if hasattr(self.signals, signal_name):
                signal = getattr(self.signals, signal_name)
                signal.connect(handler)

    def create_five_row_controls(self, field_visibility_widget=None) -> QWidget:
        """
        Create the complete five-row control layout including array field visibility.

        Args:
            field_visibility_widget: Optional pre-created field visibility widget

        Returns:
            Widget containing all control rows
        """
        controls_widget = QWidget(self.parent)
        main_layout = QVBoxLayout(controls_widget)
        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        main_layout.setSpacing(0)

        row1 = self._create_row1()
        row2 = self._create_row2()
        row3 = self._create_row3()
        row4 = self._create_secondary_axis_row()

        if field_visibility_widget:
            row5 = field_visibility_widget
        else:
            row5 = QWidget()
            row5_layout = QHBoxLayout(row5)
            row5_layout.setContentsMargins(
                8,
                4,
                8,
                4,
            )
            row5_layout.addWidget(QLabel("Show Fields: (no arrays loaded)"))
            row5_layout.addStretch()

        main_layout.addWidget(row1)
        main_layout.addWidget(row2)
        main_layout.addWidget(row3)
        main_layout.addWidget(row4)
        main_layout.addWidget(row5)

        self.layouts["main"] = main_layout
        self.layouts["row1"] = row1.layout()
        self.layouts["row2"] = row2.layout()
        self.layouts["row3"] = row3.layout()
        self.layouts["row4"] = row4.layout()
        if hasattr(row5, "layout") and row5.layout():
            self.layouts["row5"] = row5.layout()

        self.widgets["field_visibility_row"] = row5

        return controls_widget

    def create_six_row_controls(
        self,
        field_visibility_widget=None,
        field_scale_widget=None,
    ) -> QWidget:
        """
        Create the complete six-row control layout including array field visibility and scale factors.

        Args:
            field_visibility_widget: Optional pre-created field visibility widget
            field_scale_widget: Optional pre-created field scale widget

        Returns:
            Widget containing all control rows
        """
        controls_widget = QWidget(self.parent)
        main_layout = QVBoxLayout(controls_widget)
        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        main_layout.setSpacing(0)

        row1 = self._create_row1()
        row2 = self._create_row2()
        row3 = self._create_row3()
        row4 = self._create_secondary_axis_row()

        if field_visibility_widget:
            row5 = field_visibility_widget
        else:
            row5 = QWidget()
            row5_layout = QHBoxLayout(row5)
            row5_layout.setContentsMargins(
                8,
                4,
                8,
                4,
            )
            row5_layout.addWidget(QLabel("Show Fields: (no arrays loaded)"))
            row5_layout.addStretch()

        if field_scale_widget:
            row6 = field_scale_widget
        else:
            row6 = QWidget()
            row6_layout = QHBoxLayout(row6)
            row6_layout.setContentsMargins(
                8,
                4,
                8,
                4,
            )
            row6_layout.addWidget(QLabel("Scale Factors: (no arrays loaded)"))
            row6_layout.addStretch()

        main_layout.addWidget(row1)
        main_layout.addWidget(row2)
        main_layout.addWidget(row3)
        main_layout.addWidget(row4)
        main_layout.addWidget(row5)
        main_layout.addWidget(row6)

        self.layouts["main"] = main_layout
        self.layouts["row1"] = row1.layout()
        self.layouts["row2"] = row2.layout()
        self.layouts["row3"] = row3.layout()
        self.layouts["row4"] = row4.layout()
        if hasattr(row5, "layout") and row5.layout():
            self.layouts["row5"] = row5.layout()
        if hasattr(row6, "layout") and row6.layout():
            self.layouts["row6"] = row6.layout()

        self.widgets["field_visibility_row"] = row5
        self.widgets["field_scale_row"] = row6

        return controls_widget

    def update_field_visibility_row(self, field_visibility_widget) -> None:
        """
        Update the field visibility row widget after initial creation.

        Args:
            field_visibility_widget: The field visibility widget to use
        """
        if "field_visibility_row" in self.widgets:
            old_widget = self.widgets["field_visibility_row"]

            main_layout = self.layouts.get("main")
            if main_layout:
                for i in range(main_layout.count()):
                    if main_layout.itemAt(i).widget() == old_widget:
                        main_layout.takeAt(i)
                        old_widget.deleteLater()

                        main_layout.insertWidget(i, field_visibility_widget)

                        self.widgets["field_visibility_row"] = field_visibility_widget

                        if (
                            hasattr(field_visibility_widget, "layout")
                            and field_visibility_widget.layout()
                        ):
                            self.layouts["row5"] = field_visibility_widget.layout()

                        print("[INFO] Field visibility row widget updated")
                        break

    def update_field_scale_row(self, field_scale_widget) -> None:
        """
        Update the field scale row widget after initial creation.

        Args:
            field_scale_widget: The field scale widget to use
        """
        if "field_scale_row" in self.widgets:
            old_widget = self.widgets["field_scale_row"]

            main_layout = self.layouts.get("main")
            if main_layout:
                for i in range(main_layout.count()):
                    if main_layout.itemAt(i).widget() == old_widget:
                        main_layout.takeAt(i)
                        old_widget.deleteLater()

                        main_layout.insertWidget(i, field_scale_widget)

                        self.widgets["field_scale_row"] = field_scale_widget

                        if (
                            hasattr(field_scale_widget, "layout")
                            and field_scale_widget.layout()
                        ):
                            self.layouts["row6"] = field_scale_widget.layout()

                        print("[INFO] Field scale row widget updated")
                        break

    def set_point_size_mixed(self):
        """Set point size spinbox to show mixed state."""
        spin = self.widgets.get("size_spin")
        if spin:
            spin.blockSignals(True)
            spin.setSpecialValueText("Mixed")
            spin.setValue(spin.minimum())
            spin.blockSignals(False)

    def set_lines_tristate(self):
        """Set lines checkbox to tristate (partially checked)."""
        chk = self.widgets.get("lines_chk")
        if chk:
            chk.blockSignals(True)
            chk.setTristate(True)
            chk.setCheckState(Qt.CheckState.PartiallyChecked)
            chk.blockSignals(False)

    def set_visibility_tristate(self):
        """Set visibility checkbox to tristate (partially checked)."""
        chk = self.widgets.get("visible_chk")
        if chk:
            chk.blockSignals(True)
            chk.setTristate(True)
            chk.setCheckState(Qt.CheckState.PartiallyChecked)
            chk.blockSignals(False)

    def set_selected_palette_mixed(self):
        """Set palette combobox to show mixed state."""
        combo = self.widgets.get("palette_combo")
        if combo:
            combo.blockSignals(True)
            mixed_index = combo.findText("(Mixed)")
            if mixed_index < 0:
                combo.insertItem(0, "(Mixed)")
                mixed_index = 0
            combo.setCurrentIndex(mixed_index)
            combo.blockSignals(False)

    def set_offset_mixed(self):
        """Set offset spinboxes to show mixed state."""
        x_spin = self.widgets.get("offset_x_spin")
        y_spin = self.widgets.get("offset_y_spin")

        if x_spin:
            x_spin.blockSignals(True)
            x_spin.setSpecialValueText("Mixed")
            x_spin.setValue(x_spin.minimum())
            x_spin.blockSignals(False)

        if y_spin:
            y_spin.blockSignals(True)
            y_spin.setSpecialValueText("Mixed")
            y_spin.setValue(y_spin.minimum())
            y_spin.blockSignals(False)
