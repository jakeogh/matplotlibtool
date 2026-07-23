#!/usr/bin/env python3
# tab-width:4

"""
Plot Manager - all plots are equal, no primary/overlay distinction.

Group property updates are batched: one signal per operation, not one per plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal

from .CoordinateTransformEngine import CoordinateTransformEngine
from .Plot2DOverlay import Overlay

# UI property name -> (Overlay attribute, coercion)
_PROPERTY_MAP: dict[str, tuple[str, Any]] = {
    "size": ("size", float),
    "colormap": ("cmap", str),
    "draw_lines": ("draw_lines", bool),
    "offset_x": ("offset_x", float),
    "offset_y": ("offset_y", float),
    "line_color": ("line_color", lambda v: v),
    "line_width": ("line_width", float),
}


@dataclass
class PlotInfo:
    index: int
    name: str
    point_count: int
    visible: bool
    size: float
    colormap: str
    has_color_data: bool
    offset_x: float
    offset_y: float
    draw_lines: bool


@dataclass
class PlotGroupInfo:
    group_id: int
    group_name: str
    plot_indices: list[int]
    color_field: str
    color_range: tuple[float, float]


class PlotManagerSignals(QObject):
    plotAdded = pyqtSignal(int)
    plotsChanged = pyqtSignal()
    selectionChanged = pyqtSignal(int)
    plotVisibilityChanged = pyqtSignal(int, bool)
    plotPropertiesChanged = pyqtSignal(int)
    groupSelectionChanged = pyqtSignal(int)


class PlotManager:
    def __init__(self, transform_engine: CoordinateTransformEngine):
        self.signals = PlotManagerSignals()
        self.transform_engine = transform_engine

        self.plots: list[Overlay] = []
        self.plot_names: dict[int, str] = {}
        self.array_parent_indices: set[int] = set()
        self.plot_global_color_ranges: dict[int, tuple[float, float]] = {}

        self.plot_groups: dict[int, PlotGroupInfo] = {}
        self.plot_to_group: dict[int, int] = {}
        self.next_group_id = 0

        self.selected_plot_index = 0
        self.selected_group_id: int | None = None  # None = individual plot selected

        self._cached_labels: list[str] | None = None

    def _invalidate_label_cache(self):
        self._cached_labels = None

    def _rebuild_label_cache(self):
        self._cached_labels = []
        for i, plot in enumerate(self.plots):
            custom_name = self.plot_names.get(i)
            base = custom_name if custom_name else f"Plot {i + 1}"
            self._cached_labels.append(f"{base} ({len(plot.points):,} pts)")

    def get_plot_count(self) -> int:
        return len(self.plots)

    # ===== groups =====

    def register_plot_group(
        self,
        group_name: str,
        plot_indices: list[int],
        color_field: str,
        color_range: tuple[float, float],
    ) -> int:
        group_id = self.next_group_id
        self.next_group_id += 1

        self.plot_groups[group_id] = PlotGroupInfo(
            group_id=group_id,
            group_name=group_name,
            plot_indices=plot_indices.copy(),
            color_field=color_field,
            color_range=color_range,
        )

        for plot_index in plot_indices:
            self.plot_to_group[plot_index] = group_id

        return group_id

    def get_group_info(self, group_id: int) -> PlotGroupInfo | None:
        return self.plot_groups.get(group_id)

    def get_plot_group_id(self, plot_index: int) -> int | None:
        return self.plot_to_group.get(plot_index)

    def get_all_groups(self) -> list[PlotGroupInfo]:
        return list(self.plot_groups.values())

    def select_group(self, group_id: int) -> bool:
        if group_id not in self.plot_groups:
            return False
        if group_id != self.selected_group_id:
            self.selected_group_id = group_id
            self.selected_plot_index = 0
            self.signals.groupSelectionChanged.emit(group_id)
            return True
        return False

    def select_plot(self, plot_index: int) -> bool:
        if plot_index < 0 or plot_index >= len(self.plots):
            return False
        if plot_index != self.selected_plot_index or self.selected_group_id is not None:
            self.selected_plot_index = plot_index
            self.selected_group_id = None
            self.signals.selectionChanged.emit(plot_index)
            return True
        return False

    def is_group_selected(self) -> bool:
        return self.selected_group_id is not None

    def get_selected_plots(self) -> list[int]:
        """Selected plot indices: group members if a group is selected."""
        if self.selected_group_id is not None:
            group_info = self.plot_groups.get(self.selected_group_id)
            return group_info.plot_indices if group_info else []
        return [self.selected_plot_index]

    # ===== plots =====

    def add_plot(
        self,
        points: np.ndarray,
        color_data: np.ndarray | None,
        colormap: str,
        point_size: float,
        draw_lines: bool,
        line_color: str | None,
        line_width: float,
        offset_x: float,
        offset_y: float,
        visible: bool,
        transform_params: dict,
        plot_name: str | None = None,
        is_array_parent: bool = False,
        global_color_min: float | None = None,
        global_color_max: float | None = None,
    ) -> int:
        plot = Overlay(
            points=points,
            cmap=colormap,
            color_data=color_data,
            draw_lines=draw_lines,
            size=point_size,
            line_color=line_color,
            line_width=line_width,
            offset_x=offset_x,
            offset_y=offset_y,
            visible=visible,
        )

        plot_index = len(self.plots)
        self.plots.append(plot)

        if plot_name:
            self.plot_names[plot_index] = plot_name

        if is_array_parent:
            self.array_parent_indices.add(plot_index)

        if global_color_min is not None and global_color_max is not None:
            self.plot_global_color_ranges[plot_index] = (
                global_color_min,
                global_color_max,
            )

        self._invalidate_label_cache()
        self.signals.plotAdded.emit(plot_index)

        return plot_index

    def get_plot_global_color_range(
        self, plot_index: int
    ) -> tuple[float, float] | None:
        return self.plot_global_color_ranges.get(plot_index)

    def set_plot_name(self, plot_index: int, name: str) -> bool:
        if 0 <= plot_index < len(self.plots):
            self.plot_names[plot_index] = name
            self._invalidate_label_cache()
            self.signals.plotPropertiesChanged.emit(plot_index)
            return True
        return False

    def get_plot_name(self, plot_index: int) -> str | None:
        return self.plot_names.get(plot_index)

    def get_selected_plot(self) -> Overlay | None:
        if 0 <= self.selected_plot_index < len(self.plots):
            return self.plots[self.selected_plot_index]
        return None

    def get_plot_info(self, plot_index: int) -> PlotInfo | None:
        if not (0 <= plot_index < len(self.plots)):
            return None

        plot = self.plots[plot_index]
        custom_name = self.plot_names.get(plot_index)
        base = custom_name if custom_name else f"Plot {plot_index + 1}"

        return PlotInfo(
            index=plot_index,
            name=f"{base} ({len(plot.points):,} pts)",
            point_count=len(plot.points),
            visible=plot.visible,
            size=plot.size,
            colormap=plot.cmap,
            has_color_data=plot.color_data is not None,
            offset_x=plot.offset_x,
            offset_y=plot.offset_y,
            draw_lines=plot.draw_lines,
        )

    def get_plot_labels(self) -> list[str]:
        if self._cached_labels is None:
            self._rebuild_label_cache()
        return self._cached_labels.copy()

    def get_array_plot_labels(self) -> list[str]:
        """Labels for array-parent plots only (dropdown display)."""
        labels = []
        for plot_index in sorted(self.array_parent_indices):
            if plot_index < len(self.plots):
                plot = self.plots[plot_index]
                custom_name = self.plot_names.get(plot_index)
                base = custom_name if custom_name else f"Array {plot_index + 1}"
                labels.append(f"{base} ({len(plot.points):,} pts)")
        return labels

    def set_plot_visibility(self, plot_index: int, visible: bool) -> bool:
        if 0 <= plot_index < len(self.plots):
            plot = self.plots[plot_index]
            if plot.visible != visible:
                plot.visible = visible
                self.signals.plotVisibilityChanged.emit(plot_index, visible)
                return True
        return False

    def _apply_property(self, plot: Overlay, property_name: str, value: Any) -> bool:
        attr, coerce = _PROPERTY_MAP[property_name]
        if getattr(plot, attr) != value:
            setattr(plot, attr, coerce(value))
            return True
        return False

    def set_group_property(
        self,
        group_id: int,
        property_name: str,
        value: Any,
    ) -> bool:
        """Set a property on every plot in a group, emitting one signal total."""
        group_info = self.plot_groups.get(group_id)
        if not group_info:
            return False

        changed = False
        for plot_index in group_info.plot_indices:
            if 0 <= plot_index < len(self.plots):
                changed |= self._apply_property(
                    self.plots[plot_index], property_name, value
                )

        if changed:
            self.signals.plotsChanged.emit()
        return changed

    def set_plot_property(
        self,
        plot_index: int,
        property_name: str,
        value: Any,
    ) -> bool:
        if not (0 <= plot_index < len(self.plots)):
            return False

        changed = self._apply_property(self.plots[plot_index], property_name, value)
        if changed:
            self.signals.plotPropertiesChanged.emit(plot_index)
        return changed

    def get_visible_plots(self) -> list[Overlay]:
        return [plot for plot in self.plots if plot.visible]

    def get_all_plots(self) -> list[Overlay]:
        return self.plots

    def get_selected_plot_properties(self) -> dict[str, Any] | None:
        """Properties of the selection; 'mixed' where group members differ."""
        plot_indices = self.get_selected_plots()
        if not plot_indices:
            return None

        def props_of(plot: Overlay) -> dict[str, Any]:
            return {
                "size": plot.size,
                "colormap": plot.cmap,
                "draw_lines": plot.draw_lines,
                "offset_x": plot.offset_x,
                "offset_y": plot.offset_y,
                "visible": plot.visible,
                "has_color_data": plot.color_data is not None,
                "line_color": plot.line_color,
                "line_width": plot.line_width,
            }

        props = props_of(self.plots[plot_indices[0]])
        for plot_index in plot_indices[1:]:
            for key, value in props_of(self.plots[plot_index]).items():
                if props[key] != value:
                    props[key] = "mixed"

        return props
