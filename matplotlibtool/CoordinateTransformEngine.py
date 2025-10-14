#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Tuple

import numpy as np


@dataclass
class TransformParams:
    """Container for coordinate transformation parameters."""

    transform_type: str
    center: np.ndarray | None = None
    scale_factor: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for external API compatibility."""
        result = {"type": self.transform_type}
        if self.center is not None:
            result["center"] = self.center
        if self.scale_factor is not None:
            result["scale_factor"] = self.scale_factor
        return result

    @classmethod
    def from_dict(cls, data: dict) -> TransformParams:
        """Create from dictionary for external API compatibility."""
        return cls(
            transform_type=data["type"],
            center=data.get("center"),
            scale_factor=data.get("scale_factor"),
        )


class CoordinateTransformEngine:
    """
    Handles coordinate transformations for point cloud data.

    Supports three transformation modes:
    - normalize: Scale to unit square/cube centered at origin
    - center: Center at origin without scaling
    - raw: No transformation applied

    Provides parameter tracking for consistent transformations across multiple datasets.
    """

    def __init__(self, dimensions: int = 2):
        """
        Initialize the transform engine.

        Args:
            dimensions: Number of spatial dimensions (2 for 2D, 3 for 3D)
        """
        self.dimensions = dimensions

    def normalize_points(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, TransformParams]:
        """
        Normalize points to unit square/cube centered at origin.

        Args:
            points: (N, D) array of points

        Returns:
            Tuple of (transformed_points, transform_params)
        """
        if points.shape[0] == 0:
            return points, TransformParams("normalize")

        # Use only the required dimensions
        working_points = points[:, : self.dimensions]

        min_vals = working_points.min(axis=0)
        max_vals = working_points.max(axis=0)
        center = (min_vals + max_vals) / 2
        size = max_vals - min_vals
        max_extent = size.max()

        if max_extent == 0:
            transformed = working_points - center
            scale_factor = 1.0
        else:
            scale_factor = 1.0 / max_extent
            transformed = (working_points - center) * scale_factor

        params = TransformParams(
            transform_type="normalize",
            center=center.copy(),
            scale_factor=scale_factor,
        )

        return transformed.astype(np.float32), params

    def center_points(self, points: np.ndarray) -> tuple[np.ndarray, TransformParams]:
        """
        Center points at origin without scaling.

        Args:
            points: (N, D) array of points

        Returns:
            Tuple of (transformed_points, transform_params)
        """
        if points.shape[0] == 0:
            return points, TransformParams("center")

        working_points = points[:, : self.dimensions]

        # Calculate center
        min_vals = working_points.min(axis=0)
        max_vals = working_points.max(axis=0)
        center = (min_vals + max_vals) / 2

        # Apply centering
        transformed = working_points - center

        params = TransformParams(transform_type="center", center=center.copy())

        return transformed.astype(np.float32), params

    def raw_points(self, points: np.ndarray) -> tuple[np.ndarray, TransformParams]:
        """
        Return points unchanged (raw coordinates).

        Args:
            points: (N, D) array of points

        Returns:
            Tuple of (points, transform_params)
        """
        working_points = points[:, : self.dimensions]
        params = TransformParams("raw")
        return working_points.astype(np.float32), params

    def apply_transform(
        self,
        points: np.ndarray,
        params: TransformParams | dict,
    ) -> np.ndarray:
        """
        Apply existing transformation parameters to new points.

        Args:
            points: (N, D) array of points to transform
            params: TransformParams object or dict with transformation parameters

        Returns:
            Transformed points array
        """
        if isinstance(params, dict):
            params = TransformParams.from_dict(params)

        if points.shape[0] == 0:
            return points

        working_points = points[:, : self.dimensions]

        if params.transform_type == "normalize":
            if params.center is None or params.scale_factor is None:
                raise ValueError("Normalize transform requires center and scale_factor")
            transformed = (working_points - params.center) * params.scale_factor

        elif params.transform_type == "center":
            if params.center is None:
                raise ValueError("Center transform requires center")
            transformed = working_points - params.center

        elif params.transform_type == "raw":
            transformed = working_points

        else:
            raise ValueError(f"Unknown transform type: {params.transform_type}")

        return transformed.astype(np.float32)
