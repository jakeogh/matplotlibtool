#!/usr/bin/env python3
# tab-width:4

"""
Plot2D

- User must call add_plot() to add any data
- Plot-specific settings (size, colormap, normalize, etc.) only in add_plot()

View model: every view change (zoom box, wheel zoom, pan, fit, manual bounds)
goes through set_view(), which commits new base bounds and re-renders so
viewport-culled artists always match the visible region. Keyboard scaling is
the one transient path: it renders base bounds divided by the cumulative
scale factor without moving the base.
"""

from __future__ import annotations

# pylint: disable=no-name-in-module
import sys
from pathlib import Path
from time import time

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt  # type: ignore
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor  # type: ignore
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtGui import QKeySequence
from PyQt6.QtGui import QShortcut
from PyQt6.QtWidgets import QApplication  # type: ignore
from PyQt6.QtWidgets import QColorDialog
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from .ArrayFieldIntegration import ArrayFieldIntegration
from .AxisSecondaryIntegration import AxisSecondaryIntegration
from .BusyIndicatorManager import BusyIndicatorManager
from .color_paletts import COLOR_PALETTES
from .ControlBarIntegration import ControlBarIntegration
from .ControlBarManager import ControlBarManager
from .CoordinateTransformEngine import CoordinateTransformEngine
from .FileLoaderRegistry import FileLoaderRegistry
from .GridManager import GridManager
from .InputState import InputState
from .KeyboardInputManager import KeyboardInputManager
from .MouseMode import MouseMode
from .Plot2DInteractions import Plot2DInteractions
from .Plot2DRenderer import Matplotlib2DRenderer
from .PlotDataProcessor import PlotDataProcessor
from .PlotEventHandlers import PlotEventHandlers
from .PlotGroupContext import PlotGroupContext
from .PlotManager import PlotManager
from .PointHover import PointHoverManager
from .ViewHistory import ViewHistory
from .ViewManager import ViewBounds
from .ViewManager import ViewManager


class Plot2D(QMainWindow):
    """
    2D point-cloud viewer - all plots are equal, no primary/overlay distinction.

    Features:
    - No default data - start with empty viewer
    - Add plots dynamically via add_plot()
    - Zoom box, mouse zoom/pan, keyboard scaling (X/Y)
    - Context-manager support
    - Per-plot controls via control bar
    - Secondary X/Y axes for unit conversions
    - Pluggable file loader system
    """

    def __init__(
        self,
        *,
        auto_aspect: bool = False,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        dark_mode: bool = True,
        disable_antialiasing: bool = False,
        render_image: Path | None = None,
        figsize: tuple[float, float] | None = None,
        colormap: str = "turbo",
        draw_lines: bool = False,
    ):
        self._owns_qapp = False
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)
            self._owns_qapp = True

        super().__init__()

        self.auto_aspect = auto_aspect
        self.disable_antialiasing = disable_antialiasing
        self.dark_mode = dark_mode
        self.default_colormap = colormap
        self.default_draw_lines = draw_lines

        self.state = InputState()
        self.last_time = time()
        self.acceleration = 0.5

        self._shutdown_done = False

        self.busy_manager = BusyIndicatorManager()
        self.keyboard_manager = KeyboardInputManager(self.state, self.acceleration)
        self.transform_engine = CoordinateTransformEngine(dimensions=2)
        self.plot_processor = PlotDataProcessor(self)
        self.plot_manager = PlotManager(transform_engine=self.transform_engine)

        self.array_field_integration = ArrayFieldIntegration(self)
        self.array_field_integration.initialize()

        if figsize is not None:
            self.fig = Figure(figsize=figsize, facecolor="black")
            print(
                f'[INFO] Using custom figure size: {figsize[0]:.1f}" × {figsize[1]:.1f}"'
            )
        else:
            self.fig = Figure(facecolor="black")

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMouseTracking(True)

        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(
            left=0.06,
            right=0.92,
            top=0.94,
            bottom=0.12,
        )

        self.grid_manager = GridManager(self.ax)
        self.view_manager = ViewManager(self.ax)
        self.secondary_axis = AxisSecondaryIntegration(self)
        self.file_loader_registry = FileLoaderRegistry(self)
        self.control_bar_integration = ControlBarIntegration(self)

        self.axes_grid_color = "gray"
        self.grid_color = "#808080"
        self.grid_manager.set_grid_colors(self.grid_color, self.axes_grid_color)

        self.event_handlers = PlotEventHandlers(self)

        # Performance
        self.max_display_points = 100_000
        self.max_line_segments = 10_000
        self.cull_margin = 0.25

        self.renderer = Matplotlib2DRenderer()
        self.interactions = Plot2DInteractions(
            self,
            self.ax,
            self.canvas,
            self.state,
        )

        self.point_hover = PointHoverManager(
            self,
            self.ax,
            self.canvas,
        )
        print(
            "[INFO] Mouse modes: Zoom (left-drag box), Pan (left-drag pans), "
            "Hover (snap/copy/measure) - 'H' toggles hover"
        )

        # Initial view bounds
        if (
            xmin is not None
            and xmax is not None
            and ymin is not None
            and ymax is not None
        ):
            xlim = (float(xmin), float(xmax))
            ylim = (float(ymin), float(ymax))
            print(
                f"[INFO] Using custom view bounds: X({xlim[0]:.3f}, {xlim[1]:.3f}), Y({ylim[0]:.3f}, {ylim[1]:.3f})"
            )
            self._custom_bounds_provided = True
        else:
            xlim, ylim = (0.0, 1.0), (0.0, 1.0)
            self._custom_bounds_provided = False

        self.base_xlim: tuple[float, float] = xlim
        self.base_ylim: tuple[float, float] = ylim
        self.view_manager.set_view_bounds(xlim=xlim, ylim=ylim)
        self.view_history = ViewHistory()

        # Panning
        self.canvas.mpl_connect("button_press_event", self.interactions.on_mouse_press)
        self.canvas.mpl_connect(
            "button_release_event", self.interactions.on_mouse_release
        )
        self.canvas.mpl_connect("motion_notify_event", self.interactions.on_mouse_move)

        self._hover_connection = self.canvas.mpl_connect(
            "motion_notify_event", self._on_hover_motion_wrapper
        )

        self.canvas.mpl_connect(
            "key_press_event", self.interactions.on_matplotlib_key_press
        )
        self.canvas.mpl_connect("scroll_event", self.interactions.on_mouse_scroll)

        self._setup_ui()
        self.control_bar_integration.connect_signals()

        self.plot_manager.signals.plotAdded.connect(self._on_plot_added)
        self.plot_manager.signals.plotsChanged.connect(self._on_plots_changed)
        self.plot_manager.signals.selectionChanged.connect(
            self._on_plot_selection_changed
        )
        self.plot_manager.signals.plotVisibilityChanged.connect(
            self._on_plot_visibility_changed
        )
        self.plot_manager.signals.plotPropertiesChanged.connect(
            self._on_plot_properties_changed
        )

        self.view_manager.signals.viewChanged.connect(self._on_view_changed)
        self.view_manager.signals.secondaryAxisChanged.connect(
            self._on_secondary_axis_changed
        )

        self.control_bar_integration.set_initial_state()
        if self._custom_bounds_provided:
            self.control_bar_integration.update_view_bounds_display()

        status_label = self.control_bar_manager.get_widget("status_label")
        if status_label is None:
            raise RuntimeError(
                "CRITICAL: status_label widget not found in ControlBarManager!"
            )
        self.busy_manager.set_status_label(status_label)

        self.set_dark_mode(self.dark_mode)
        self._update_plot()
        self.view_history.record(ViewBounds(xlim=xlim, ylim=ylim))

        self._hover_shortcut = QShortcut(QKeySequence("H"), self)
        self._hover_shortcut.activated.connect(self.toggle_hover_mode)
        self._apply_mouse_mode_cursor(MouseMode.ZOOM)

        # Timer ~60FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.event_handlers.on_timer)
        self.timer.start(16)

        print("[INFO] Initialized empty 2D viewer (Matplotlib with Axis Scaling)")
        print(
            f"[INFO] Antialiasing: {'enabled' if not disable_antialiasing else 'disabled'}"
        )
        print("[INFO] Call add_plot(data, ...) to add point cloud data")

        self._render_image_path = render_image
        if render_image is not None:
            print(
                f"[INFO] Image will be rendered to: {render_image} after setup completes"
            )

    # ===== mouse modes =====

    def set_mouse_mode(self, mode: MouseMode) -> None:
        previous = self.interactions.mouse_mode
        if mode == previous:
            return

        if previous == MouseMode.HOVER:
            self.point_hover.disable()
        self.interactions.mouse_mode = mode
        if mode == MouseMode.HOVER:
            self.point_hover.enable()

        self._apply_mouse_mode_cursor(mode)
        self.control_bar_manager.set_mouse_mode_indicator(mode.name)
        print(f"[INFO] Mouse mode: {mode.name}")

    def toggle_hover_mode(self) -> None:
        if self.interactions.mouse_mode == MouseMode.HOVER:
            self.set_mouse_mode(MouseMode.ZOOM)
        else:
            self.set_mouse_mode(MouseMode.HOVER)

    def _apply_mouse_mode_cursor(self, mode: MouseMode) -> None:
        cursor = {
            MouseMode.ZOOM: Qt.CursorShape.CrossCursor,
            MouseMode.PAN: Qt.CursorShape.OpenHandCursor,
            MouseMode.HOVER: Qt.CursorShape.ArrowCursor,
        }[mode]
        self.canvas.setCursor(cursor)

    # ===== view API =====

    def set_view(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        record: bool = True,
    ) -> ViewBounds:
        """
        Commit new view bounds: apply, reset keyboard scaling, re-render.

        The single entry point for every discrete view change. Re-rendering
        here is what keeps culled artist data consistent with the axes.
        record=False applies without adding a history entry; continuous
        gestures use it for intermediate states and record once on release.
        """
        bounds = ViewBounds(xlim=xlim, ylim=ylim)

        self.state.scale[:] = 1.0
        self.state.velocity[:] = 0.0
        self.base_xlim = bounds.xlim
        self.base_ylim = bounds.ylim

        self.view_manager.apply(bounds)
        self._update_plot()
        self.canvas.draw_idle()
        if record:
            self.view_history.record(bounds)
            self._sync_history_buttons()
        return bounds

    def record_view_history(self) -> None:
        """Commit the current view as a history entry (end of a gesture)."""
        self.view_history.record(
            ViewBounds(xlim=self.base_xlim, ylim=self.base_ylim)
        )
        self._sync_history_buttons()

    def view_back(self) -> None:
        bounds = self.view_history.back()
        if bounds is not None:
            self.set_view(bounds.xlim, bounds.ylim, record=False)
        self._sync_history_buttons()

    def view_forward(self) -> None:
        bounds = self.view_history.forward()
        if bounds is not None:
            self.set_view(bounds.xlim, bounds.ylim, record=False)
        self._sync_history_buttons()

    def _sync_history_buttons(self) -> None:
        back_btn = self.control_bar_manager.get_widget("view_back_btn")
        forward_btn = self.control_bar_manager.get_widget("view_forward_btn")
        back_btn.setEnabled(self.view_history.can_go_back)
        forward_btn.setEnabled(self.view_history.can_go_forward)

    def apply_keyboard_scale(self) -> None:
        """Render base bounds divided by the cumulative keyboard scale."""
        scale_x = float(self.state.scale[0])
        scale_y = float(self.state.scale[1])

        x_center = (self.base_xlim[0] + self.base_xlim[1]) / 2
        y_center = (self.base_ylim[0] + self.base_ylim[1]) / 2
        x_half = (self.base_xlim[1] - self.base_xlim[0]) / (2 * scale_x)
        y_half = (self.base_ylim[1] - self.base_ylim[0]) / (2 * scale_y)

        bounds = ViewBounds(
            xlim=(x_center - x_half, x_center + x_half),
            ylim=(y_center - y_half, y_center + y_half),
        )
        self.view_manager.apply(bounds)
        self._update_plot()
        self.canvas.draw_idle()

    def fit_view(self, pad_ratio: float = 0.05) -> ViewBounds:
        """Fit view to all visible data; unit square when there is none."""
        all_points = []
        for plot in self.plot_manager.get_visible_plots():
            if plot.offset_x != 0.0 or plot.offset_y != 0.0:
                all_points.append(
                    plot.points
                    + np.array([plot.offset_x, plot.offset_y], dtype=np.float32)
                )
            else:
                all_points.append(plot.points)

        bounds = ViewManager.compute_fit_bounds(all_points, pad_ratio)
        if bounds is None:
            return self.set_view((0.0, 1.0), (0.0, 1.0))
        return self.set_view(bounds.xlim, bounds.ylim)

    # ===== plotting =====

    def plot_group(
        self,
        color_field: str,
        group_name: str | None = None,
    ) -> PlotGroupContext:
        """
        Plot group context for adding multiple plots with shared color mapping.

        Usage:
            with viewer.plot_group(color_field='frame', group_name='My Data') as group:
                group.add_plot(data1, x_field='x', y_field='y', color_field='frame')
                group.add_plot(data2, x_field='x', y_field='y', color_field='frame')
        """
        return PlotGroupContext(
            self,
            color_field,
            group_name,
        )

    def add_plot(
        self,
        data,
        *,
        x_field: str,
        y_field: str,
        normalize: bool = False,
        center: bool = False,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        colormap: str | None = None,
        point_size: float = 0.2,
        draw_lines: bool | None = None,
        line_color: str | None = None,
        line_width: float = 1.0,
        visible: bool = True,
        transform_params: dict | None = None,
        plot_name: str | None = None,
        color_field: str | None = None,
    ) -> dict:
        """
        Add a single plot from a structured array field.

        Registers the array with the field management system and creates an
        auto-group so additional fields can be added to the same group.

        Returns:
            Transform parameters for the added plot
        """
        processed = self.plot_processor.process_structured_array(
            data,
            x_field=x_field,
            y_field=y_field,
            color_field=color_field,
            normalize=normalize,
            center=center,
            x_offset=x_offset,
            y_offset=y_offset,
            colormap=colormap,
            point_size=point_size,
            draw_lines=draw_lines,
            line_color=line_color,
            line_width=line_width,
            visible=visible,
            transform_params=transform_params,
            plot_name=plot_name,
        )

        global_color_min = None
        global_color_max = None
        if processed.color_data is not None and len(processed.color_data) > 0:
            global_color_min = float(processed.color_data.min())
            global_color_max = float(processed.color_data.max())

        array_index = self.array_field_integration.register_array(
            data=data,
            x_field=x_field,
            y_field=y_field,
            array_name=processed.plot_name,
            normalize=normalize,
            center=center,
            x_offset=x_offset,
            y_offset=y_offset,
            colormap=processed.colormap,
            point_size=processed.point_size,
            draw_lines=processed.draw_lines,
            line_color=processed.line_color,
            line_width=processed.line_width,
            visible=processed.visible,
            transform_params=processed.transform_params,
            color_field=color_field,
            global_color_min=global_color_min,
            global_color_max=global_color_max,
        )

        generated_name = y_field if plot_name is None else plot_name

        added_index = self.plot_manager.add_plot(
            points=processed.points,
            color_data=processed.color_data,
            colormap=processed.colormap,
            point_size=processed.point_size,
            draw_lines=processed.draw_lines,
            line_color=processed.line_color,
            line_width=processed.line_width,
            offset_x=x_offset,
            offset_y=y_offset,
            visible=visible,
            transform_params=processed.transform_params,
            plot_name=generated_name,
            is_array_parent=True,
            global_color_min=global_color_min,
            global_color_max=global_color_max,
        )

        self.array_field_integration.register_field_plot(
            array_index,
            y_field,
            added_index,
        )

        group_id = PlotGroupContext.create_auto_group_for_array(
            viewer=self,
            array_index=array_index,
            data=data,
            x_field=x_field,
            y_field=y_field,
            color_field=color_field,
            properties={
                "plot_name": plot_name,
                "normalize": normalize,
                "center": center,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "colormap": processed.colormap,
                "point_size": processed.point_size,
                "draw_lines": processed.draw_lines,
                "line_color": line_color,
                "line_width": line_width,
                "color_field": color_field,
                "transform_params": processed.transform_params,
            },
        )

        self.array_field_integration.register_array_group(array_index, group_id)

        if len(self.plot_manager.plots) == 1 and not self._custom_bounds_provided:
            self.fit_view(pad_ratio=0.05 if (normalize or center) else 0.1)

        self._update_plot()
        self.canvas.draw_idle()

        self.control_bar_integration.refresh_plot_selector()
        self.control_bar_integration.sync_controls_to_selection()

        self.array_field_integration.visibility_row.set_current_array(array_index)
        if self.array_field_integration.scale_row:
            self.array_field_integration.scale_row.set_current_array(array_index)

        return processed.transform_params

    def _update_plot(self):
        """Re-render all plots at the current view bounds."""
        current_bounds = self.view_manager.get_current_bounds()
        all_plots = self.plot_manager.get_all_plots()

        # With secondary (twin) axes matplotlib requires adjustable='datalim'
        if self.view_manager.secondary_axis_manager.is_any_enabled() or self.auto_aspect:
            self.ax.set_aspect("auto", adjustable="datalim")
        else:
            self.ax.set_aspect("equal", adjustable="datalim")

        color_ranges: list[tuple[float, float] | None] = []
        for i, plot in enumerate(all_plots):
            if plot.color_data is None or len(plot.color_data) == 0:
                color_ranges.append(None)
            else:
                global_range = self.plot_manager.get_plot_global_color_range(i)
                color_ranges.append(
                    global_range if global_range is not None else plot.color_range()
                )

        self.renderer.render(
            self.ax,
            plots=all_plots,
            view_xlim=current_bounds.xlim,
            view_ylim=current_bounds.ylim,
            color_ranges=color_ranges,
            cull_margin=self.cull_margin,
            max_display_points=self.max_display_points,
            max_line_segments=self.max_line_segments,
            disable_antialiasing=self.disable_antialiasing,
        )

        self.grid_manager.update_grid(
            axes_grid_enabled=True,
            horizontal_grid_enabled=None,
            max_lines=4000,
        )

        self.secondary_axis.update_after_plot()

    # ===== appearance =====

    def set_dark_mode(self, enabled: bool):
        self.dark_mode = enabled

        if enabled:
            fig_bg = "black"
            ax_bg = "black"
            text_color = "white"
            spine_color = "white"
            grid_color = "gray"
        else:
            fig_bg = "white"
            ax_bg = "white"
            text_color = "black"
            spine_color = "black"
            grid_color = "#CCCCCC"

        self.fig.set_facecolor(fig_bg)
        self.ax.set_facecolor(ax_bg)
        self.ax.tick_params(colors=text_color, labelcolor=text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)

        for spine in self.ax.spines.values():
            spine.set_color(spine_color)

        self.axes_grid_color = grid_color
        self.grid_manager.set_grid_colors(self.grid_color, self.axes_grid_color)

        secondary = self.view_manager.secondary_axis_manager
        if secondary.y_axis_manager.secondary_ax:
            y_ax = secondary.y_axis_manager.secondary_ax
            y_ax.yaxis.label.set_color(text_color)
            y_ax.tick_params(axis="y", colors=text_color, labelcolor=text_color)

        if secondary.x_axis_manager.secondary_ax:
            x_ax = secondary.x_axis_manager.secondary_ax
            x_ax.xaxis.label.set_color(text_color)
            x_ax.tick_params(axis="x", colors=text_color, labelcolor=text_color)

        self.canvas.draw_idle()

    # ===== signal handlers =====

    def _on_hover_motion_wrapper(self, event):
        if not self.interactions.panning and not self.interactions.drawing_zoom_box:
            self.point_hover.on_hover_motion(event)

    def _on_plot_added(self, plot_index: int):
        self._update_plot()
        self.canvas.draw_idle()

    def _on_plot_visibility_changed(self, plot_index: int, visible: bool):
        self._update_plot()
        self.canvas.draw_idle()

    def _on_plots_changed(self):
        self.control_bar_integration.refresh_plot_selector()
        self._update_plot()
        self.canvas.draw_idle()

    def _on_plot_selection_changed(self, plot_index: int):
        self.control_bar_integration.sync_controls_to_selection()
        self.array_field_integration.on_array_selection_changed(plot_index)

    def _on_plot_properties_changed(self, plot_index: int):
        self._update_plot()
        self.canvas.draw_idle()

    def _on_view_changed(self):
        self.control_bar_integration.update_view_bounds_display()

    def _on_secondary_axis_changed(self):
        self.canvas.draw_idle()

    def showEvent(self, event):
        """Fit view to all data when first shown."""
        super().showEvent(event)

        if not hasattr(self, "_initial_show_done"):
            self._initial_show_done = True

            if self._custom_bounds_provided:
                print("[INFO] Skipping auto-fit due to custom view bounds")
                return

            if self.plot_manager.get_visible_plots():
                bounds = self.fit_view(pad_ratio=0.05)
                print(
                    f"[INFO] Initial view fitted to all data: X({bounds.xlim[0]:.3f}, {bounds.xlim[1]:.3f}), Y({bounds.ylim[0]:.3f}, {bounds.ylim[1]:.3f})"
                )

    # ===== secondary axis =====

    def configure_secondary_axis_from_data_range(
        self,
        *,
        axis: str,
        label: str,
        data_min: float | None = None,
        data_max: float | None = None,
        target_min: float | None = None,
        target_max: float | None = None,
        frequency: float | None = None,
        unit: str = "",
    ):
        """Configure secondary axis from data range or frequency."""
        from .AxisSecondaryConfig import AxisSecondaryConfig
        from .AxisType import AxisType

        axis_type = AxisType.X if axis.lower() == "x" else AxisType.Y

        if frequency is not None:
            config = AxisSecondaryConfig.from_frequency(
                frequency=frequency,
                label=label,
                unit=unit or "s",
                enable_auto_scale=True,
                data_min=0,
                axis_type=axis_type,
            )
        else:
            if (
                data_min is None
                or data_max is None
                or target_min is None
                or target_max is None
            ):
                raise ValueError(
                    "For range mapping, all of data_min, data_max, target_min, and target_max must be provided"
                )

            config = AxisSecondaryConfig.from_range_mapping(
                primary_min=data_min,
                primary_max=data_max,
                secondary_min=target_min,
                secondary_max=target_max,
                label=label,
                unit=unit,
                enable_auto_scale=True,
                axis_type=axis_type,
            )

        self.view_manager.secondary_axis_manager.configure_axis(config)
        self._update_plot()
        self.canvas.draw_idle()

    # ===== file loaders =====

    def register_file_loader(self, *, extensions, loader_func):
        self.file_loader_registry.register_loader(extensions, loader_func)

    def unregister_file_loader(self, extensions):
        self.file_loader_registry.unregister_loader(extensions)

    def get_registered_extensions(self):
        return self.file_loader_registry.get_registered_extensions()

    # ===== UI =====

    def _setup_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.canvas, 1)

        self.control_bar_manager = ControlBarManager(self, COLOR_PALETTES)

        field_visibility_widget = (
            self.array_field_integration.create_visibility_widget()
        )
        field_scale_widget = self.array_field_integration.create_scale_widget()

        controls_widget = self.control_bar_manager.create_six_row_controls(
            field_visibility_widget=field_visibility_widget,
            field_scale_widget=field_scale_widget,
        )

        main_layout.addWidget(controls_widget)

        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setWindowTitle("2D Point Viewer (Matplotlib + Array Fields)")

    def _qcolor_to_hex(self, qc: QColor) -> str:
        return (
            qc.name(QColor.NameFormat.HexRgb)
            if isinstance(qc, QColor) and qc.isValid()
            else ""
        )

    def _pick_color(self, initial_hex: str) -> str | None:
        qc = QColorDialog.getColor(
            QColor(initial_hex),
            self,
            "Pick Color",
        )
        return self._qcolor_to_hex(qc) if qc.isValid() else None

    # ===== keyboard =====

    def keyPressEvent(self, event: QKeyEvent):
        key_name = event.text().upper() if event.text() else None
        has_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if key_name and key_name in ("X", "Y", "Z"):
            self.keyboard_manager.add_key_with_repeat_check(key_name, has_shift)
        elif key_name:
            self.state.add_key(key_name, has_shift)

        self.interactions.keyPressEvent(event)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        key_name = event.text().upper() if event.text() else None

        if key_name and key_name in ("X", "Y", "Z"):
            self.keyboard_manager.remove_key_with_repeat_check(key_name)
        elif key_name:
            self.state.remove_key(key_name)

        self.interactions.keyReleaseEvent(event)
        super().keyReleaseEvent(event)

    # ===== lifecycle =====

    def __enter__(self) -> Plot2D:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
        return None

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True

        self.timer.stop()
        super().close()

        if self._owns_qapp and self._app is not None:
            self._app.quit()

    def close(self) -> None:
        self.shutdown()

    def closeEvent(self, event):
        print("[INFO] Viewer window closed.")
        self.shutdown()
        event.accept()

    def show_gui(self, skip_if_rendered: bool = True):
        """Show the GUI and start the application."""
        if self._render_image_path is not None:
            print("[INFO] Rendering plot to file...")
            self._render_to_file(self._render_image_path)

            if skip_if_rendered:
                print("[INFO] Render complete. Skipping GUI display.")
                self.shutdown()
                return

            print("[INFO] Render complete. GUI will close automatically.")
            QTimer.singleShot(200, self.close)

        self.show()
        self.raise_()
        self.activateWindow()

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            self._owns_qapp = True

        try:
            app.exec()
        except KeyboardInterrupt:
            pass

    def _render_to_file(
        self,
        filepath: Path,
        dpi: int = 300,
    ) -> None:
        """Render the current plot to an image file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fmt = filepath.suffix.lstrip(".").lower()
        if not fmt:
            fmt = "png"
            filepath = filepath.with_suffix(".png")

        self.set_dark_mode(self.dark_mode)
        self.canvas.draw()

        self.fig.savefig(
            filepath,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
            edgecolor="none",
        )

        if fmt in ("png", "jpg", "jpeg", "tiff", "tif"):
            width_px = int(self.fig.get_figwidth() * dpi)
            height_px = int(self.fig.get_figheight() * dpi)
            print(
                f"[INFO] Plot saved: {filepath} ({fmt.upper()}, {dpi} DPI, {width_px}×{height_px}px)"
            )
        else:
            print(f"[INFO] Plot saved: {filepath} ({fmt.upper()}, vector format)")
