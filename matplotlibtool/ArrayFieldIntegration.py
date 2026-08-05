#!/usr/bin/env python3
# tab-width:4

"""
Array field integration.

Coordinates the ArrayFieldManager, the Fields popup panel, and the
viewer. Field plots are created lazily on first enable and toggled by
visibility afterwards; Y multipliers apply at render time through
Overlay.y_scale, so neither operation rebuilds plot data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from .ArrayFieldManager import ArrayFieldManager
from .ArrayFieldPanel import ArrayFieldPanel
from .CoordinateTransformEngine import TransformParams

if TYPE_CHECKING:
    from .Plot2D import Plot2D


class ArrayFieldIntegration:
    def __init__(self, viewer: Plot2D):
        self.viewer = viewer
        self.array_field_manager = ArrayFieldManager(viewer.plot_manager)
        self.panel = ArrayFieldPanel(self)
        self.array_to_group: dict[int, int] = {}
        self.multipliers: dict[tuple[int, str], float] = {}

    def create_panel_button(self, parent=None):
        return self.panel.create_button(parent)

    # ---------- registration ----------

    def register_array(
        self,
        data: np.ndarray,
        x_field: str,
        y_field: str,
        array_name: str | None = None,
        global_color_min: float | None = None,
        global_color_max: float | None = None,
        **properties,
    ) -> int:
        if global_color_min is not None and global_color_max is not None:
            properties["global_color_min"] = global_color_min
            properties["global_color_max"] = global_color_max

        return self.array_field_manager.register_array(
            data=data,
            x_field=x_field,
            y_field=y_field,
            array_name=array_name,
            **properties,
        )

    def register_field_plot(
        self,
        array_index: int,
        field_name: str,
        plot_index: int,
    ) -> None:
        self.array_field_manager.register_field_plot(
            array_index,
            field_name,
            plot_index,
        )
        if self.panel.button is not None:
            self.panel.update_button_label()

    def register_array_group(
        self,
        array_index: int,
        group_id: int,
    ) -> None:
        self.array_to_group[array_index] = group_id

    def array_index_for_plot(self, plot_index: int) -> int | None:
        mapping = self.array_field_manager.plot_to_array_field.get(plot_index)
        return mapping[0] if mapping is not None else None

    # ---------- panel state ----------

    def is_field_visible(self, array_index: int, field_name: str) -> bool:
        plot_index = self.array_field_manager.get_field_plot_index(
            array_index, field_name
        )
        if plot_index is None:
            return False
        return self.viewer.plot_manager.plots[plot_index].visible

    def get_multiplier(self, array_index: int, field_name: str) -> float:
        return self.multipliers.get((array_index, field_name), 1.0)

    # ---------- panel actions ----------

    def set_field_enabled(
        self,
        array_index: int,
        field_name: str,
        enabled: bool,
    ) -> None:
        plot_index = self.array_field_manager.get_field_plot_index(
            array_index, field_name
        )

        if plot_index is None:
            if not enabled:
                return
            self._create_field_plot(array_index, field_name)
        else:
            self.viewer.plot_manager.set_plot_visibility(plot_index, enabled)
            state = "enabled" if enabled else "disabled"
            print(f"[INFO] Field '{field_name}' {state} (plot {plot_index})")

        self.viewer._update_plot()
        self.viewer.canvas.draw_idle()
        self.viewer.control_bar_integration.refresh_plot_selector()
        self.panel.update_button_label()

    def set_visible_fields(
        self,
        array_index: int,
        fields: Iterable[str],
    ) -> None:
        """
        Show exactly these fields of the array, hiding every other mapped field.

        Fields not yet plotted are created lazily, the same as enabling them
        one at a time in the Fields panel.
        """
        wanted = tuple(fields)
        available = self.array_field_manager.get_array_fields(array_index)
        unknown = [f for f in wanted if f not in available]
        if unknown:
            raise KeyError(
                f"array {array_index} has no fields {unknown}, has {available}"
            )
        for field in available:
            visible = self.is_field_visible(array_index, field)
            if field in wanted and not visible:
                self.set_field_enabled(array_index, field, True)
            elif field not in wanted and visible:
                self.set_field_enabled(array_index, field, False)

    def set_multiplier(
        self,
        array_index: int,
        field_name: str,
        value: float,
    ) -> None:
        if value == self.get_multiplier(array_index, field_name):
            return
        self.multipliers[(array_index, field_name)] = value

        plot_index = self.array_field_manager.get_field_plot_index(
            array_index, field_name
        )
        print(f"[INFO] Multiplier for '{field_name}': {value:g}")
        if plot_index is None:
            return

        self.viewer.plot_manager.plots[plot_index].y_scale = value
        self.viewer._update_plot()
        self.viewer.canvas.draw_idle()

    # ---------- plot creation ----------

    def _create_field_plot(self, array_index: int, field_name: str) -> None:
        info = self.array_field_manager.get_array_info(array_index)
        data = info["data"]
        x_field = info["x_field"]
        properties = info["properties"]

        points_xy = np.column_stack(
            (
                data[x_field].astype(np.float32),
                data[field_name].astype(np.float32),
            )
        )

        transform_params = properties.get("transform_params")
        if transform_params:
            transformed_points = self.viewer.transform_engine.apply_transform(
                points_xy, TransformParams.from_dict(transform_params)
            )
        elif properties.get("normalize", False):
            transformed_points, params = self.viewer.transform_engine.normalize_points(
                points_xy
            )
            transform_params = params.to_dict()
        elif properties.get("center", False):
            transformed_points, params = self.viewer.transform_engine.center_points(
                points_xy
            )
            transform_params = params.to_dict()
        else:
            transformed_points, params = self.viewer.transform_engine.raw_points(
                points_xy
            )
            transform_params = params.to_dict()

        color_field = properties.get("color_field")
        color_data = (
            data[color_field].astype(np.float32)
            if color_field is not None and color_field in data.dtype.names
            else None
        )

        with self.viewer.busy_manager.busy_operation(f"Adding field {field_name}"):
            plot_index = self.viewer.plot_manager.add_plot(
                points=transformed_points,
                color_data=color_data,
                colormap=properties.get("colormap", self.viewer.default_colormap),
                point_size=properties.get("point_size", 2.0),
                draw_lines=properties.get(
                    "draw_lines", self.viewer.default_draw_lines
                ),
                line_color=properties.get("line_color", None),
                line_width=properties.get("line_width", 1.0),
                offset_x=properties.get("x_offset", 0.0),
                offset_y=properties.get("y_offset", 0.0),
                visible=True,
                transform_params=transform_params,
                plot_name=field_name,
                is_array_parent=False,
                global_color_min=properties.get("global_color_min"),
                global_color_max=properties.get("global_color_max"),
            )

            # a field enabled later must look like the ones already drawn, so
            # marker sizing follows a sibling field of the same array rather
            # than the constructor default
            plots = self.viewer.plot_manager.plots
            siblings = [
                i
                for i in self.array_field_manager.array_fields[array_index].values()
                if i is not None and i != plot_index and 0 <= i < len(plots)
            ]
            if siblings:
                sibling = plots[siblings[0]]
                plots[plot_index].auto_size = sibling.auto_size
                plots[plot_index].size = sibling.size

            multiplier = self.get_multiplier(array_index, field_name)
            if multiplier != 1.0:
                plots[plot_index].y_scale = multiplier

            self.array_field_manager.register_field_plot(
                array_index,
                field_name,
                plot_index,
            )

            group_id = self.array_to_group.get(array_index)
            if group_id is not None:
                group_info = self.viewer.plot_manager.get_group_info(group_id)
                if group_info and plot_index not in group_info.plot_indices:
                    group_info.plot_indices.append(plot_index)
                    self.viewer.plot_manager.plot_to_group[plot_index] = group_id

            print(f"[INFO] Added field plot: {field_name} (plot index {plot_index})")

        # creation emits no visibility signal, so the DC overlays reconcile here
        self.viewer.event_handlers.refresh_pixel_dc()
