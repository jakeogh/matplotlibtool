#!/usr/bin/env python3
# tab-width:4

"""
Plot Data Processor - one plot's source made from a table, the logic shared
by Plot2D.add_plot() and PlotGroupContext.add_plot().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .PlotSource import Source
from .PlotSource import StructuredTable
from .PlotSource import Table

if TYPE_CHECKING:
    from .Plot2D import Plot2D


@dataclass
class ProcessedPlotData:
    """Container for processed plot data ready for PlotManager."""

    source: Source
    x_field: str
    y_field: str
    color_field: str | None
    colormap: str
    point_size: float
    draw_lines: bool
    line_color: str | None
    line_width: float
    x_scale: float
    x_offset: float
    y_offset: float
    visible: bool
    plot_name: str | None
    transform_params: dict
    table: Table


class PlotDataProcessor:
    """Makes a plot's source from a table."""

    def __init__(self, viewer: Plot2D):
        self.viewer = viewer

    def transform_for(
        self,
        table: Table,
        *,
        x_field: str,
        y_field: str,
        x_scale: float,
        normalize: bool,
        center: bool,
        transform_params: dict | None,
    ) -> dict:
        """The coordinate transform a plot is drawn through: the one given,
        else one fitted to the points for normalize or center, else none.
        Fitting one reads every point, so only a table held in memory can."""
        if normalize and center:
            raise ValueError("Cannot specify both normalize=True and center=True")
        if transform_params is not None:
            return transform_params.copy()
        if not (normalize or center):
            return {"type": "raw"}
        if not isinstance(table, StructuredTable):
            raise ValueError(
                "normalize and center fit a transform to every point, and this table is read as it is drawn"
            )
        engine = self.viewer.transform_engine
        points = table.points(x_field, y_field, x_scale)
        fitted = engine.normalize_points(points) if normalize else engine.center_points(points)
        return fitted[1].to_dict()

    def process_table(
        self,
        table: Table,
        *,
        x_field: str,
        y_field: str,
        color_field: str | None = None,
        normalize: bool = False,
        center: bool = False,
        x_scale: float = 1.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        colormap: str | None = None,
        point_size: float = 2.0,
        draw_lines: bool | None = None,
        line_color: str | None = None,
        line_width: float = 1.0,
        visible: bool = True,
        transform_params: dict | None = None,
        plot_name: str | None = None,
    ) -> ProcessedPlotData:
        """One field of a table as a plot's source, with the viewer's
        defaults for what was not given.

        The x scale is folded into the source rather than carried as a
        column, and the x offset is left to the overlay, which applies it
        when it asks the source for a view.
        """
        if colormap is None:
            colormap = self.viewer.default_colormap
        if draw_lines is None:
            draw_lines = self.viewer.default_draw_lines

        # x is the table's to check: a table read as it is drawn has the one
        # x, which is not among the fields drawn against it
        if y_field not in table.fields:
            raise ValueError(f"Y field '{y_field}' not found in data. Available: {table.fields}")
        if color_field is not None and color_field not in table.color_fields:
            raise ValueError(
                f"Color field '{color_field}' not found in data. Available: {table.color_fields}"
            )

        transform = self.transform_for(
            table,
            x_field=x_field,
            y_field=y_field,
            x_scale=x_scale,
            normalize=normalize,
            center=center,
            transform_params=transform_params,
        )
        source = table.source(
            x_field=x_field,
            y_field=y_field,
            color_field=color_field,
            x_scale=x_scale,
            transform=transform,
        )

        return ProcessedPlotData(
            source=source,
            x_field=x_field,
            y_field=y_field,
            color_field=color_field,
            colormap=colormap,
            point_size=point_size,
            draw_lines=draw_lines,
            line_color=line_color,
            line_width=line_width,
            x_scale=x_scale,
            x_offset=x_offset,
            y_offset=y_offset,
            visible=visible,
            plot_name=plot_name,
            transform_params=transform,
            table=table,
        )
