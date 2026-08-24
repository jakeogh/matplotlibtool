#!/usr/bin/env python3
"""
Consolidated utility functions for point cloud processing and visualization.
"""

from __future__ import annotations


import numpy as np


def normalize_points(points: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    """
    Normalize point cloud to fit in a unit cube/square centered at origin.

    Args:
        points: (N, D) array of points
        dimensions: Number of dimensions to consider (None = all)

    Returns:
        Normalized points array
    """
    if points.shape[0] == 0:
        return points

    if dimensions is not None:
        points = points[:, :dimensions]

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    center = (min_vals + max_vals) / 2
    size = max_vals - min_vals
    max_extent = size.max()
    scale_factors = np.where(
        size != 0,
        max_extent / size,
        1,
    )
    return (points - center) * scale_factors


def center_points(points: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    """
    Center point cloud at origin without scaling.

    Args:
        points: (N, D) array of points
        dimensions: Number of dimensions to consider (None = all)

    Returns:
        Centered points array
    """
    if points.shape[0] == 0:
        return points

    if dimensions is not None:
        points = points[:, :dimensions]

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    center = (min_vals + max_vals) / 2
    return points - center


def compute_bounds(
    points: np.ndarray,
    pad_ratio: float = 0.05,
    return_format: str = "tuple",
) -> (
    tuple[tuple[float, float] | tuple[float, float]] | tuple[float, float, float, float]
):
    """
    Calculate bounds with padding for initial view.

    Args:
        points: (N, 2) or (N, 3) array of points
        pad_ratio: Padding ratio (fraction of size)
        return_format: "tuple" for ((xmin, xmax), (ymin, ymax)) or "rect" for (x, y, width, height)

    Returns:
        Bounds in requested format
    """
    if points.shape[0] == 0:
        if return_format == "rect":
            return (0.0, 0.0, 1.0, 1.0)
        return (0.0, 1.0), (0.0, 1.0)

    # Use only first 2 dimensions for bounds calculation
    points_2d = points[:, :2] if points.shape[1] >= 2 else points

    min_vals = points_2d.min(axis=0)
    max_vals = points_2d.max(axis=0)
    size = np.maximum(max_vals - min_vals, 1e-12)
    pad = size * pad_ratio
    lo = min_vals - pad
    hi = max_vals + pad

    if return_format == "rect":
        # Return (x, y, width, height) format
        width, height = (hi - lo).tolist()
        return (
            float(lo[0]),
            float(lo[1]),
            float(max(width, 1e-6)),
            float(max(height, 1e-6)),
        )
    else:
        # Return ((xmin, xmax), (ymin, ymax)) format
        return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def center_points_2d(points_xy: np.ndarray) -> np.ndarray:
    """Center 2D points at origin without scaling."""
    return center_points(points_xy, dimensions=2)

