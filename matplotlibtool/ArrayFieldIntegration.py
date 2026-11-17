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

from .ArrayFieldManager import ArrayFieldManager
from .ArrayFieldPanel import ArrayFieldPanel
from .PlotSource import Table
from .PlotSource import as_table

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
        data: Table,
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

        self.viewer.request_render()
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

    def replace_array_data(
        self,
        array_index: int,
        data,
        *,
        recompute_pixel_dc: bool = True,
    ) -> None:
        """
        Swap the table behind every plot of this array for a new capture.

        Every mapped field plot is rebuilt from the new table through the
        transform the array was registered with; nothing else changes, so the
        view, selection, visibility, and styling all survive the swap. The DC
        overlays recompute against the new data when showing.
        """
        table = as_table(data)
        manager = self.array_field_manager
        info = manager.get_array_info(array_index)
        old_names = set(info["data"].fields)
        new_names = set(table.fields)
        if new_names != old_names:
            raise ValueError(
                f"array {array_index}: replacement fields {sorted(new_names)} "
                f"do not match {sorted(old_names)}"
            )

        info["data"] = table
        properties = info["properties"]
        color_field = properties.get("color_field")
        for field, plot_index in manager.array_fields[array_index].items():
            if plot_index is None:
                continue
            source = table.source(
                x_field=info["x_field"],
                y_field=field,
                color_field=color_field,
                x_scale=info["x_scale"],
                transform=properties["transform_params"],
            )
            color_range = source.color_span()
            if color_range is not None:
                properties["global_color_min"], properties["global_color_max"] = color_range
            else:
                properties.pop("global_color_min", None)
                properties.pop("global_color_max", None)
            self.viewer.plot_manager.replace_plot_source(plot_index, source, color_range)

        self.viewer.control_bar_integration.refresh_plot_selector()
        if not (recompute_pixel_dc and self.viewer.event_handlers.recompute_pixel_dc()):
            self.viewer.request_render()

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
        self.viewer.request_render()

    # ---------- plot creation ----------

    def _create_field_plot(self, array_index: int, field_name: str) -> None:
        info = self.array_field_manager.get_array_info(array_index)
        table = info["data"]
        properties = info["properties"]

        transform_params = self.viewer.plot_processor.transform_for(
            table,
            x_field=info["x_field"],
            y_field=field_name,
            x_scale=info["x_scale"],
            normalize=properties.get("normalize", False),
            center=properties.get("center", False),
            transform_params=properties.get("transform_params"),
        )

        color_field = properties.get("color_field")
        if color_field is not None and color_field not in table.color_fields:
            color_field = None

        with self.viewer.busy_manager.busy_operation(f"Adding field {field_name}"):
            plot_index = self.viewer.plot_manager.add_plot(
                source=table.source(
                    x_field=info["x_field"],
                    y_field=field_name,
                    color_field=color_field,
                    x_scale=info["x_scale"],
                    transform=transform_params,
                ),
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
