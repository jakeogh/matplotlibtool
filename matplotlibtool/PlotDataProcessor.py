#!/usr/bin/env python3
# tab-width:4

"""
Plot Data Processor - Shared logic for preparing plot data from structured arrays.

This module extracts the common data processing logic previously duplicated between
Plot2D.add_plot() and PlotGroupContext.add_plot().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .Plot2D import Plot2D


@dataclass
class ProcessedPlotData:
    """Container for processed plot data ready for PlotManager."""

    points: np.ndarray  # Transformed (N, 2) points
    color_data: np.ndarray | None  # Extracted color values
    x_field: str
    y_field: str
    color_field: str | None
    colormap: str
    point_size: float
    draw_lines: bool
    line_color: str | None
    line_width: float
    x_offset: float
    y_offset: float
    visible: bool
    plot_name: str | None
    transform_params: dict
    original_data: np.ndarray  # Keep reference to original structured array


class PlotDataProcessor:
    """Processes structured array data into format suitable for PlotManager."""

    def __init__(self, viewer: Plot2D):
        """
        Initialize processor with reference to viewer.

        Args:
            viewer: The Plot2D viewer instance
        """
        self.viewer = viewer

    def process_structured_array(
        self,
        data: np.ndarray,
        *,
        x_field: str,
        y_field: str,
        color_field: str | None = None,
        normalize: bool = False,
        center: bool = False,
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
        """
        Process structured array into plot data.

        Handles:
        - Default resolution (colormap, draw_lines)
        - Field validation
        - Data extraction (x, y, color)
        - Coordinate transformation (normalize/center/custom)
        - Color data preparation

        Args:
            data: Structured numpy array with named fields
            x_field: Name of field to use for X axis
            y_field: Name of field to use for Y axis
            color_field: Optional field to use for coloring
            normalize: If True, normalize points to unit square
            center: If True, center points at origin
            x_offset: X offset for the plot
            y_offset: Y offset for the plot
            colormap: Colormap name (None = use viewer default)
            point_size: Point size
            draw_lines: Whether to draw lines (None = use viewer default)
            line_color: Line color (None = use point colors)
            line_width: Line width
            visible: Whether plot is initially visible
            transform_params: Optional custom transform parameters
            plot_name: Optional custom name for the plot

        Returns:
            ProcessedPlotData ready for PlotManager.add_plot()

        Raises:
            TypeError: If data is not a structured numpy array
            ValueError: If fields are invalid or conflicting options specified
        """
        # Apply viewer defaults if not specified
        if colormap is None:
            colormap = self.viewer.default_colormap
        if draw_lines is None:
            draw_lines = self.viewer.default_draw_lines

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

        # Validate x_field
        if x_field not in field_names:
            raise ValueError(
                f"X field '{x_field}' not found in data. Available: {field_names}"
            )

        # Validate y_field
        if y_field not in field_names:
            raise ValueError(
                f"Y field '{y_field}' not found in data. Available: {field_names}"
            )

        # Extract X and Y data
        x_data = data[x_field].astype(np.float32)
        y_data = data[y_field].astype(np.float32)
        points_xy = np.column_stack((x_data, y_data))

        # Extract color data if specified
        extracted_color_data = None
        if color_field is not None:
            if color_field not in field_names:
                raise ValueError(
                    f"Color field '{color_field}' not found in data. "
                    f"Available: {field_names}"
                )
            extracted_color_data = data[color_field].astype(np.float32)

        # Apply coordinate transformation
        if transform_params is not None:
            # Use provided transform parameters
            from .CoordinateTransformEngine import TransformParams

            transform_params_obj = TransformParams.from_dict(transform_params)
            transformed_points = self.viewer.transform_engine.apply_transform(
                points_xy, transform_params_obj
            )
            result_transform_params = transform_params.copy()
        elif normalize:
            # Normalize to unit square
            transformed_points, params = self.viewer.transform_engine.normalize_points(
                points_xy
            )
            result_transform_params = params.to_dict()
        elif center:
            # Center at origin
            transformed_points, params = self.viewer.transform_engine.center_points(
                points_xy
            )
            result_transform_params = params.to_dict()
        else:
            # Raw points (identity transform)
            transformed_points, params = self.viewer.transform_engine.raw_points(
                points_xy
            )
            result_transform_params = params.to_dict()

        # Return processed data
        return ProcessedPlotData(
            points=transformed_points,
            color_data=extracted_color_data,
            x_field=x_field,
            y_field=y_field,
            color_field=color_field,
            colormap=colormap,
            point_size=point_size,
            draw_lines=draw_lines,
            line_color=line_color,
            line_width=line_width,
            x_offset=x_offset,
            y_offset=y_offset,
            visible=visible,
            plot_name=plot_name,
            transform_params=result_transform_params,
            original_data=data,
        )
