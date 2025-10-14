#!/usr/bin/env python3
# tab-width:4

"""
Plot2D

- User must call add_plot() to add any data
- Plot-specific settings (size, colormap, normalize, etc.) only in add_plot()
"""

from __future__ import annotations

# pylint: disable=no-name-in-module
import sys
from pathlib import Path
from time import time

import numpy as np
from asserttool import icp
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import Qt  # type: ignore
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor  # type: ignore
from PyQt6.QtGui import QKeyEvent
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
from .Plot2DInteractions import Plot2DInteractions
from .Plot2DRenderer import Matplotlib2DRenderer
from .PlotEventHandlers import PlotEventHandlers
from .PlotGroupContext import PlotGroupContext
from .PlotManager import PlotManager
from .utils import KeyboardInputManager
from .utils import get_bounds_2d
from .ViewManager import ViewManager


class Plot2D(QMainWindow):
    """
    Enhanced 2D point-cloud viewer - All plots are equal, no primary/overlay distinction.

    Features:
    - No default data - start with empty viewer
    - Add plots dynamically via add_plot()
    - All plots have identical capabilities
    - Zoom box, mouse zoom/pan, keyboard scaling (X/Y)
    - Context-manager support
    - Per-plot controls via control bar
    - Four-row control bar with secondary Y-axis configuration
    - Efficient axis scaling
    - Pluggable file loader system
    - Secondary Y-axis for unit conversions
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
    ):
        """
        Initialize empty viewer with global configuration.

        Args:
            auto_aspect: If True, use automatic aspect ratio
            xmin: Optional minimum X axis limit
            xmax: Optional maximum X axis limit
            ymin: Optional minimum Y axis limit
            ymax: Optional maximum Y axis limit
            dark_mode: If True, use dark theme
            disable_antialiasing: If True, disable antialiasing globally
            render_image: Optional path to render image on startup
            figsize: Optional figure size (width, height) in inches
        """
        # Ensure a QApplication exists
        self._owns_qapp = False
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)
            self._owns_qapp = True

        super().__init__()

        # Store global configuration (not plot-specific)
        self.auto_aspect = auto_aspect
        self.disable_antialiasing = disable_antialiasing
        self.dark_mode = dark_mode

        # State
        self.state = InputState()
        self.last_time = time()
        self.acceleration = 0.5

        # Store the base view limits
        self.base_xlim = None
        self.base_ylim = None

        # Initialize busy manager WITHOUT status label initially
        self.busy_manager = BusyIndicatorManager()

        # Managers
        self.keyboard_manager = KeyboardInputManager(self.state, self.acceleration)

        # Initialize coordinate transformation engine
        self.transform_engine = CoordinateTransformEngine(dimensions=2)

        # Initialize plot manager WITH NO PLOTS
        self.plot_manager = PlotManager(transform_engine=self.transform_engine)

        # Initialize array field integration
        self.array_field_integration = ArrayFieldIntegration(self)
        self.array_field_integration.initialize()

        # Figure/axes
        if figsize is not None:
            self.fig = Figure(figsize=figsize, facecolor="black")
            print(
                f'[INFO] Using custom figure size: {figsize[0]:.1f}" × {figsize[1]:.1f}"'
            )
        else:
            self.fig = Figure(facecolor="black")

        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # Adjusted margins
        self.fig.subplots_adjust(
            left=0.06,
            right=0.92,
            top=0.94,
            bottom=0.12,
        )

        # Initialize managers that depend on axes
        self.grid_manager = GridManager(self.ax)
        self.view_manager = ViewManager(self.ax)

        # Initialize secondary axis integration
        self.secondary_axis = AxisSecondaryIntegration(self)

        # Initialize file loader registry
        self.file_loader_registry = FileLoaderRegistry(self)

        # Initialize control bar integration
        self.control_bar_integration = ControlBarIntegration(self)

        # Global grid state + colors
        self.axes_grid_color = "gray"
        self.grid_color = "#808080"
        self.grid_manager.set_grid_colors(self.grid_color, self.axes_grid_color)

        # Event handlers
        self.event_handlers = PlotEventHandlers(self)

        # Performance
        self.max_display_points = 100_000

        # Renderer & Interactions
        self.renderer = Matplotlib2DRenderer()
        self.interactions = Plot2DInteractions(
            self,
            self.ax,
            self.canvas,
            self.state,
        )

        # Set initial view bounds
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
            # Default bounds for empty viewer
            xlim, ylim = (0.0, 1.0), (0.0, 1.0)
            self._custom_bounds_provided = False

        self.view_manager.set_view_bounds(xlim=xlim, ylim=ylim)
        self.base_xlim = xlim
        self.base_ylim = ylim

        # Panning
        self.canvas.mpl_connect("button_press_event", self.interactions.on_mouse_press)
        self.canvas.mpl_connect(
            "button_release_event", self.interactions.on_mouse_release
        )
        self.canvas.mpl_connect("motion_notify_event", self.interactions.on_mouse_move)

        # Create RectangleSelector without monkey-patching
        self.rect_selector = RectangleSelector(
            self.ax,
            self.interactions.on_zoom_box,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            ignore_event_outside=True,
            state_modifier_keys={
                "move": "",
                "clear": "",
                "square": "",
                "center": "ctrl",
            },
        )

        # Start with selector enabled (it will be disabled during panning)
        self.rect_selector.set_active(True)

        # FINALLY: Connect keyboard and scroll events
        self.canvas.mpl_connect(
            "key_press_event", self.interactions.on_matplotlib_key_press
        )
        self.canvas.mpl_connect("scroll_event", self.interactions.on_mouse_scroll)

        # Setup UI
        self._setup_ui()

        # Wire signals
        self.control_bar_integration.connect_signals()

        # Connect plot manager signals
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

        # Connect view manager signals
        self.view_manager.signals.viewChanged.connect(self._on_view_changed)
        self.view_manager.signals.secondaryAxisChanged.connect(
            self._on_secondary_axis_changed
        )

        # Initialize control states
        self.control_bar_integration.set_initial_state()
        if self._custom_bounds_provided:
            self.control_bar_integration.update_view_bounds_display()

        # Connect busy indicator
        status_label = self.control_bar_manager.get_widget("status_label")
        if status_label is None:
            raise RuntimeError(
                "CRITICAL: status_label widget not found in ControlBarManager!"
            )
        self.busy_manager.set_status_label(status_label)

        # Apply initial dark mode setting
        self.set_dark_mode(self.dark_mode)

        # Initial plot (empty)
        self._update_plot()

        # Timer ~60FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.event_handlers.on_timer)
        self.timer.start(16)

        print("[INFO] Initialized empty 2D viewer (Matplotlib with Axis Scaling)")
        print(
            f"[INFO] Antialiasing: {'enabled' if not disable_antialiasing else 'disabled'}"
        )
        print("[INFO] Call add_plot(data, ...) to add point cloud data")

        # Store render path
        self._render_image_path = render_image
        if render_image is not None:
            print(
                f"[INFO] Image will be rendered to: {render_image} after setup completes"
            )

    def plot_group(
        self,
        color_field: str,
        group_name: str | None = None,
    ) -> PlotGroupContext:
        """
        Create a plot group context for adding multiple plots with shared color mapping.

        Usage:
            with viewer.plot_group(color_field='frame', group_name='My Data') as group:
                group.add_plot(data1, x_field='x', y_field='y', color_field='frame')
                group.add_plot(data2, x_field='x', y_field='y', color_field='frame')
            # On exit, all plots are rendered with consistent global color mapping
            # and registered as a group for group-level operations

        Args:
            color_field: Name of the field to use for global color mapping across all plots
            group_name: Optional custom name for the group (auto-generated if None)

        Returns:
            PlotGroupContext manager
        """
        from .PlotGroupContext import PlotGroupContext

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
        colormap: str = "turbo",
        point_size: float = 2.0,
        draw_lines: bool = False,
        line_color: str | None = None,
        line_width: float = 1.0,
        visible: bool = True,
        transform_params: dict | None = None,
        plot_name: str | None = None,
        color_field: None | str = None,
    ) -> dict:
        """
        Add a single plot to the viewer from a structured array field.

        This integrates with the array field management system, allowing
        dynamic enabling/disabling of fields via checkboxes.

        Automatically creates a plot group for this array so additional fields
        can be added to the same group.

        Args:
            data: Structured array (fields used by order: 1st=X, 2nd=Y, 3rd=color)
            x_field: Name of field to use for X axis
            y_field: Name of field to use for Y axis (single field)
            normalize: If True, normalize points to unit square
            center: If True, center points at origin
            x_offset: X offset for the plot
            y_offset: Y offset for the plot
            colormap: Colormap for the plot
            point_size: Point size for the plot
            draw_lines: Whether to draw lines between points
            line_color: Color for lines (None = use point colors)
            line_width: Width of lines
            visible: Whether plot is initially visible
            transform_params: Optional transform parameters (overrides normalize/center)
            plot_name: Optional custom name for the plot
            color_field: Optional field to use for coloring

        Returns:
            dict: Transform parameters for the added plot
        """
        try:
            # Validate input - must be structured array
            if not isinstance(data, np.ndarray):
                raise TypeError("data must be a numpy array")

            if data.dtype.names is None:
                raise TypeError("data must be a structured array with named fields")

            # Validate normalize/center combination
            if normalize and center:
                raise ValueError("Cannot specify both normalize=True and center=True")

            field_names = data.dtype.names
            if len(field_names) < 2:
                raise ValueError(
                    f"Structured array must have at least 2 fields. "
                    f"Found {len(field_names)}: {field_names}"
                )

            if x_field not in field_names:
                raise ValueError(
                    f"X field '{x_field}' not found in data. Available: {field_names}"
                )

            # Validate Y field
            if y_field not in field_names:
                raise ValueError(
                    f"Y field '{y_field}' not found in data. Available: {field_names}"
                )

            # Calculate global color range for auto-group creation
            global_color_min = None
            global_color_max = None
            if color_field is not None and color_field in field_names:
                color_data_for_range = data[color_field].astype(np.float32)
                if len(color_data_for_range) > 0:
                    global_color_min = float(color_data_for_range.min())
                    global_color_max = float(color_data_for_range.max())

            # Register this array with the field manager
            array_index = self.array_field_integration.register_array(
                data=data,
                x_field=x_field,
                y_field=y_field,
                array_name=plot_name,
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
                color_field=color_field,
                global_color_min=global_color_min,
                global_color_max=global_color_max,
            )

            # Extract X and Y data
            x_data = data[x_field].astype(np.float32)
            y_data = data[y_field].astype(np.float32)

            # Create points array
            points_xy = np.column_stack((x_data, y_data))

            # Extract color data if available
            color_data = (
                data[color_field].astype(np.float32)
                if color_field is not None
                else None
            )

            # Generate plot name
            generated_name = y_field if plot_name is None else plot_name

            plot_index = len(self.plot_manager.plots)
            print(
                f"[INFO] Adding plot {plot_index}: {generated_name} ({len(points_xy):,} points)"
            )
            print(
                f"[INFO] Using fields: x='{x_field}', y='{y_field}', color='{color_field if color_field else 'None'}'"
            )

            # Apply coordinate transformation
            if transform_params is not None:
                from .CoordinateTransformEngine import TransformParams

                transform_params_obj = TransformParams.from_dict(transform_params)
                transformed_points = self.transform_engine.apply_transform(
                    points_xy, transform_params_obj
                )
                result_transform_params = transform_params.copy()
            elif normalize:
                transformed_points, params = self.transform_engine.normalize_points(
                    points_xy
                )
                result_transform_params = params.to_dict()
            elif center:
                transformed_points, params = self.transform_engine.center_points(
                    points_xy
                )
                result_transform_params = params.to_dict()
            else:
                transformed_points, params = self.transform_engine.raw_points(points_xy)
                result_transform_params = params.to_dict()

            # Add plot via manager (always mark as array parent since each plot is its own array)
            added_index = self.plot_manager.add_plot(
                points=transformed_points,
                color_data=color_data,
                colormap=colormap,
                point_size=point_size,
                draw_lines=draw_lines,
                line_color=line_color,
                line_width=line_width,
                offset_x=x_offset,
                offset_y=y_offset,
                visible=visible,
                transform_params=result_transform_params,
                plot_name=generated_name,
                is_array_parent=True,
                global_color_min=global_color_min,
                global_color_max=global_color_max,
            )

            # Register the field plot with array field manager
            self.array_field_integration.register_field_plot(
                array_index,
                y_field,
                added_index,
            )

            # AUTO-CREATE GROUP: Create a group for this array
            from .PlotGroupContext import PlotGroupContext

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
                    "colormap": colormap,
                    "point_size": point_size,
                    "draw_lines": draw_lines,
                    "line_color": line_color,
                    "line_width": line_width,
                    "color_field": color_field,
                    "transform_params": result_transform_params,
                },
            )

            # Register the array-to-group mapping
            self.array_field_integration.register_array_group(array_index, group_id)

            print(
                f"[INFO] Plot {added_index} configured: {generated_name} ({len(transformed_points):,} points)"
            )

            # If this is the first plot and no custom bounds, fit view
            if len(self.plot_manager.plots) == 1 and not self._custom_bounds_provided:
                transformed_points = self.plot_manager.plots[0].points

                xlim, ylim = (
                    get_bounds_2d(transformed_points)
                    if normalize or center
                    else get_bounds_2d(transformed_points, pad_ratio=0.1)
                )

                self.view_manager.set_view_bounds(xlim=xlim, ylim=ylim)
                self.base_xlim = xlim
                self.base_ylim = ylim

            # Update plot
            self._update_plot()
            self.canvas.draw_idle()

            # Refresh UI
            self.control_bar_integration.refresh_plot_selector()
            self.control_bar_integration.sync_controls_to_selection()

            # Update field visibility checkboxes for the new array
            self.array_field_integration.visibility_row.set_current_array(array_index)

            # Update field scale inputs for the new array
            if self.array_field_integration.scale_row:
                self.array_field_integration.scale_row.set_current_array(array_index)
                print(f"[DEBUG] Scale row updated for array {array_index}")

            return result_transform_params

        except Exception as e:
            print(f"[ERROR] Failed to add plot: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _apply_axis_scaling(self):
        """Apply current scale to axis limits."""
        if self.base_xlim is None or self.base_ylim is None:
            return

        scale_x = self.state.scale[0]
        scale_y = self.state.scale[1]

        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        x_center = (current_xlim[0] + current_xlim[1]) / 2
        y_center = (current_ylim[0] + current_ylim[1]) / 2

        x_range = current_xlim[1] - current_xlim[0]
        y_range = current_ylim[1] - current_ylim[0]

        new_x_range = x_range / scale_x
        new_y_range = y_range / scale_y

        new_xlim = (x_center - new_x_range / 2, x_center + new_x_range / 2)
        new_ylim = (y_center - new_y_range / 2, y_center + new_y_range / 2)

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)

        self.view_manager.secondary_axis_manager.update_on_primary_change()

    def set_dark_mode(self, enabled: bool):
        """Toggle between dark and light mode."""
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

        if hasattr(self, "grid_manager"):
            self.axes_grid_color = grid_color
            self.grid_manager.set_grid_colors(self.grid_color, self.axes_grid_color)

        if hasattr(self, "view_manager") and self.view_manager:
            if self.view_manager.secondary_axis_manager.y_axis_manager.secondary_ax:
                y_ax = (
                    self.view_manager.secondary_axis_manager.y_axis_manager.secondary_ax
                )
                y_ax.yaxis.label.set_color(text_color)
                y_ax.tick_params(
                    axis="y",
                    colors=text_color,
                    labelcolor=text_color,
                )

            if self.view_manager.secondary_axis_manager.x_axis_manager.secondary_ax:
                x_ax = (
                    self.view_manager.secondary_axis_manager.x_axis_manager.secondary_ax
                )
                x_ax.xaxis.label.set_color(text_color)
                x_ax.tick_params(
                    axis="x",
                    colors=text_color,
                    labelcolor=text_color,
                )

        self.canvas.draw_idle()

    def _on_plot_added(self, plot_index: int):
        """Handle when a plot is added."""
        print(f"[DEBUG] _on_plot_added called with plot_index={plot_index}")
        self._update_plot()
        self.ax.stale = True
        self.fig.stale = True
        self.canvas.draw_idle()

    def _on_plot_visibility_changed(
        self,
        plot_index: int,
        visible: bool,
    ):
        """Handle when plot visibility changes."""
        print(
            f"[DEBUG] _on_plot_visibility_changed: plot_index={plot_index}, visible={visible}"
        )
        self._update_plot()
        self.canvas.draw_idle()

    def configure_secondary_axis_from_data_range(
        self,
        *,
        axis: str,
        label: str,
        data_min: float = None,
        data_max: float = None,
        target_min: float = None,
        target_max: float = None,
        frequency: None | float = None,
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

        axis_name = "X" if axis_type == AxisType.X else "Y"
        if frequency is not None:
            print(
                f"[INFO] Secondary {axis_name}-axis configured for time: {label} (sampling rate: {frequency} Hz)"
            )
        else:
            print(f"[INFO] Secondary {axis_name}-axis configured: {label} ({unit})")

    def register_file_loader(
        self,
        *,
        extensions,
        loader_func,
    ):
        """Register a loader function for specific file extensions."""
        self.file_loader_registry.register_loader(extensions, loader_func)

    def unregister_file_loader(self, extensions):
        """Unregister file loader(s) for specific extensions."""
        self.file_loader_registry.unregister_loader(extensions)

    def get_registered_extensions(self):
        """Get list of currently registered file extensions."""
        return self.file_loader_registry.get_registered_extensions()

    def _on_plots_changed(self):
        """Handle when plots are added/removed."""
        self.control_bar_integration.refresh_plot_selector()
        self._update_plot()
        self.canvas.draw_idle()

    def _on_plot_selection_changed(self, plot_index: int):
        """Handle when plot selection changes."""
        self.control_bar_integration.sync_controls_to_selection()

        # Update field visibility checkboxes to show fields for this array
        if hasattr(self, "array_field_integration") and self.array_field_integration:
            self.array_field_integration.on_array_selection_changed(plot_index)

    def _on_plot_properties_changed(self, plot_index: int):
        """Handle when plot properties change."""
        self._update_plot()
        self.canvas.draw_idle()

    def _on_view_changed(self):
        """Handle when view bounds change."""
        self.control_bar_integration.update_view_bounds_display()

    def _on_secondary_axis_changed(self):
        """Handle when secondary axis updates."""
        self.canvas.draw_idle()

    def showEvent(self, event):
        """Override showEvent to fit view to all data when first shown."""
        super().showEvent(event)

        if not hasattr(self, "_initial_show_done"):
            self._initial_show_done = True

            if self._custom_bounds_provided:
                print("[INFO] Skipping auto-fit due to custom view bounds")
                return

            visible_plots = self.plot_manager.get_visible_plots()
            if visible_plots:
                all_points = []
                for plot in visible_plots:
                    offset_points = plot.points + np.array(
                        [plot.offset_x, plot.offset_y], dtype=np.float32
                    )
                    all_points.append(offset_points)

                new_bounds = self.view_manager.fit_to_data(all_points, pad_ratio=0.05)
                self.base_xlim = new_bounds.xlim
                self.base_ylim = new_bounds.ylim
                self._update_plot()
                self.canvas.draw_idle()
                print(
                    f"[INFO] Initial view fitted to all data: X({new_bounds.xlim[0]:.3f}, {new_bounds.xlim[1]:.3f}), Y({new_bounds.ylim[0]:.3f}, {new_bounds.ylim[1]:.3f})"
                )

    def _setup_ui(self):
        """Setup the UI layout with 6 rows including array field visibility and scale factors."""
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        main_layout.setSpacing(0)
        main_layout.addWidget(self.canvas, 1)

        # Create control bar manager
        self.control_bar_manager = ControlBarManager(self, COLOR_PALETTES)

        # Create field visibility widget
        field_visibility_widget = None
        if hasattr(self, "array_field_integration") and self.array_field_integration:
            field_visibility_widget = (
                self.array_field_integration.create_visibility_widget()
            )

        # Create field scale widget
        field_scale_widget = None
        if hasattr(self, "array_field_integration") and self.array_field_integration:
            field_scale_widget = self.array_field_integration.create_scale_widget()

        # Create 6-row controls with field visibility and scale factors
        controls_widget = self.control_bar_manager.create_six_row_controls(
            field_visibility_widget=field_visibility_widget,
            field_scale_widget=field_scale_widget,
        )

        main_layout.addWidget(controls_widget)

        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setWindowTitle("2D Point Viewer (Matplotlib + Array Fields)")

    def _update_array_field_checkboxes(self):
        """
        Update array field visibility checkboxes.

        This ensures checkboxes reflect the current state.
        """
        if hasattr(self, "array_field_integration") and self.array_field_integration:
            self.array_field_integration.update_visibility_row()

    def _update_plot(self):
        """Update the complete plot."""
        print(f"[DEBUG] === _update_plot() called ===")

        with self.busy_manager.busy_operation("Updating plot"):
            current_bounds = self.view_manager.get_current_bounds()
            all_plots = self.plot_manager.get_all_plots()

            print(f"[DEBUG] Total plots: {len(all_plots)}")

            # CRITICAL: Dynamically set aspect ratio based on secondary axes state
            # Matplotlib has conflicting requirements:
            # - WITH secondary axes (twinx/twiny): MUST use adjustable='datalim', cannot use equal aspect
            # - WITHOUT secondary axes: Can use any combination
            has_secondary_axes = (
                self.view_manager.secondary_axis_manager.is_any_enabled()
            )

            if has_secondary_axes:
                # With secondary axes, force auto aspect with datalim
                self.ax.set_aspect("auto", adjustable="datalim")
            else:
                # Without secondary axes, use user preference
                if self.auto_aspect:
                    self.ax.set_aspect("auto", adjustable="datalim")
                else:
                    self.ax.set_aspect("equal", adjustable="datalim")

            # VIEWPORT CULLING
            margin = 0.1
            x_range = current_bounds.xlim[1] - current_bounds.xlim[0]
            y_range = current_bounds.ylim[1] - current_bounds.ylim[0]

            cull_xlim = (
                current_bounds.xlim[0] - x_range * margin,
                current_bounds.xlim[1] + x_range * margin,
            )
            cull_ylim = (
                current_bounds.ylim[0] - y_range * margin,
                current_bounds.ylim[1] + y_range * margin,
            )

            # Cull all plots with global color range support
            culled_plots = []
            for i, plot in enumerate(all_plots):
                if not plot.visible:
                    culled_plots.append(plot)
                    continue

                # Apply offset
                plot_with_offset = plot.points + np.array(
                    [plot.offset_x, plot.offset_y], dtype=np.float32
                )

                # Create mask
                mask = (
                    (plot_with_offset[:, 0] >= cull_xlim[0])
                    & (plot_with_offset[:, 0] <= cull_xlim[1])
                    & (plot_with_offset[:, 1] >= cull_ylim[0])
                    & (plot_with_offset[:, 1] <= cull_ylim[1])
                )

                if mask.any():
                    from copy import copy

                    culled_plot = copy(plot)
                    culled_plot.points = plot.points[mask]

                    if plot.color_data is not None:
                        culled_plot.color_data = plot.color_data[mask]

                        # Check if this plot has a global color range
                        global_range = self.plot_manager.get_plot_global_color_range(i)

                        if global_range is not None:
                            # Use global color range for normalization
                            global_min, global_max = global_range
                            global_range_value = global_max - global_min

                            if global_range_value > 1e-9:
                                culled_plot.color_data = (
                                    culled_plot.color_data - global_min
                                ) / global_range_value
                            else:
                                culled_plot.color_data = np.full_like(
                                    culled_plot.color_data,
                                    0.5,
                                    dtype=np.float32,
                                )

                            print(
                                f"[DEBUG] Plot {i} using global color range: [{global_min:.3f}, {global_max:.3f}]"
                            )
                        else:
                            # Use local (per-plot) color range
                            if len(culled_plot.color_data) > 0:
                                local_min = float(plot.color_data.min())
                                local_max = float(plot.color_data.max())
                                local_range_value = local_max - local_min
                                if local_range_value > 1e-9:
                                    culled_plot.color_data = (
                                        culled_plot.color_data - local_min
                                    ) / local_range_value
                                else:
                                    culled_plot.color_data = np.full_like(
                                        culled_plot.color_data,
                                        0.5,
                                        dtype=np.float32,
                                    )

                    culled_plot.scatter_artist = plot.scatter_artist
                    culled_plot.line_artist = plot.line_artist
                    culled_plots.append(culled_plot)
                else:
                    from copy import copy

                    culled_plot = copy(plot)
                    culled_plot.points = plot.points[:0]
                    if plot.color_data is not None:
                        culled_plot.color_data = plot.color_data[:0]
                    culled_plot.scatter_artist = plot.scatter_artist
                    culled_plot.line_artist = plot.line_artist
                    culled_plots.append(culled_plot)

            # Render all plots
            self.renderer.update_all_plots(
                self.ax,
                plots=culled_plots,
                auto_aspect=self.auto_aspect,
                view_xlim=current_bounds.xlim,
                view_ylim=current_bounds.ylim,
                grid_enabled=False,
                grid_power=0,
                grid_color=self.grid_color,
                axes_grid_color=self.axes_grid_color,
                disable_antialiasing=self.disable_antialiasing,
                max_display_points=self.max_display_points,
                in_zoom_box=self.view_manager.is_zoom_box_active(),
            )

            # Sync artist references
            for i, culled_plot in enumerate(culled_plots):
                if i < len(all_plots):
                    all_plots[i].scatter_artist = culled_plot.scatter_artist
                    all_plots[i].line_artist = culled_plot.line_artist

            self.grid_manager.update_grid(
                axes_grid_enabled=True,
                horizontal_grid_enabled=None,
                max_lines=4000,
            )

            self._apply_axis_scaling()
            self.secondary_axis.update_after_plot()

        print(f"[DEBUG] === _update_plot() completed ===")

    def _qcolor_to_hex(self, qc: QColor) -> str:
        return (
            qc.name(QColor.NameFormat.HexRgb)
            if isinstance(qc, QColor) and qc.isValid()
            else ""
        )

    def _pick_color(self, initial_hex: str) -> None | str:
        qc = QColorDialog.getColor(
            QColor(initial_hex),
            self,
            "Pick Color",
        )
        return self._qcolor_to_hex(qc) if qc.isValid() else None

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        key_name = event.text().upper() if event.text() else None
        has_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if key_name and key_name in ["X", "Y", "Z"]:
            self.keyboard_manager.add_key_with_repeat_check(key_name, has_shift)
        else:
            if key_name:
                self.state.add_key(key_name, has_shift)

        self.interactions.keyPressEvent(event)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events."""
        key_name = event.text().upper() if event.text() else None

        if key_name and key_name in ["X", "Y", "Z"]:
            self.keyboard_manager.remove_key_with_repeat_check(key_name)
        else:
            if key_name:
                self.state.remove_key(key_name)

        self.interactions.keyReleaseEvent(event)
        super().keyReleaseEvent(event)

    def __enter__(self) -> Plot2D:
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        self.shutdown()
        return None

    def shutdown(self) -> None:
        try:
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "rect_selector") and self.rect_selector is not None:
                self.rect_selector.disconnect_events()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as _plt

            _plt.close(self.fig)
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass
        try:
            if (
                getattr(
                    self,
                    "_owns_qapp",
                    False,
                )
                and self._app is not None
            ):
                self._app.quit()
        except Exception:
            pass

    def close(self) -> None:
        self.shutdown()

    def closeEvent(self, event):
        """Handle window close event."""
        print("[INFO] Viewer window closed.")
        self.shutdown()
        event.accept()

    def show_gui(self, skip_if_rendered: bool = True):
        """Show the GUI and start the application."""
        if hasattr(self, "_render_image_path") and self._render_image_path is not None:
            print("[INFO] Rendering plot to file...")
            self._render_to_file(self._render_image_path)

            if skip_if_rendered:
                print("[INFO] Render complete. Skipping GUI display.")
                self.shutdown()
                return
            else:
                print("[INFO] Render complete. GUI will close automatically.")
                QTimer.singleShot(200, self.close)
                self.show()
                try:
                    self.raise_()
                    self.activateWindow()
                except Exception:
                    pass
        else:
            self.show()
            try:
                self.raise_()
                self.activateWindow()
            except Exception:
                pass

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

        try:
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

            if fmt in ["png", "jpg", "jpeg", "tiff", "tif"]:
                width_px = int(self.fig.get_figwidth() * dpi)
                height_px = int(self.fig.get_figheight() * dpi)
                print(
                    f"[INFO] Plot saved: {filepath} ({fmt.upper()}, {dpi} DPI, {width_px}×{height_px}px)"
                )
            else:
                print(f"[INFO] Plot saved: {filepath} ({fmt.upper()}, vector format)")

        except Exception as e:
            print(f"[ERROR] Failed to render plot to {filepath}: {e}")
            import traceback

            traceback.print_exc()
