#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def make_colors_from_scalar(
    color_data: np.ndarray,
    colormap: str,
    backend: str,
    global_min: float | None = None,
    global_max: float | None = None,
) -> np.ndarray:
    """
    Map scalar data to colors using specified colormap.

    Args:
        color_data: (N,) array of scalar values
        colormap: Name of colormap to use
        backend: "vispy", "matplotlib", or "datoviz"
        global_min: Global minimum for normalization (if None, uses data min)
        global_max: Global maximum for normalization (if None, uses data max)

    Returns:
        Color array - format depends on backend
    """
    if color_data is None:
        return None

    # Use global range if provided, otherwise fall back to local range
    if global_min is not None and global_max is not None:
        data_min = global_min
        data_max = global_max
    else:
        data_min = color_data.min()
        data_max = color_data.max()

    # Normalize to [0, 1] using the determined range
    color_range = data_max - data_min
    if color_range > 1e-9:
        color_norm = (color_data - data_min) / color_range
    else:
        # Map uniform value to its position in global range
        if global_min is not None and global_max is not None:
            global_range = global_max - global_min
            if global_range > 1e-9:
                color_norm = np.full_like(
                    color_data,
                    (data_min - global_min) / global_range,
                    dtype=np.float32,
                )
            else:
                color_norm = np.full_like(
                    color_data,
                    0.5,
                    dtype=np.float32,
                )
        else:
            color_norm = np.full_like(
                color_data,
                0.5,
                dtype=np.float32,
            )

    # Clamp to [0, 1] to handle any edge cases
    color_norm = np.clip(
        color_norm,
        0.0,
        1.0,
    )

    if backend == "vispy":
        from vispy.color import Colormap

        if colormap == "viridis":
            cmap = Colormap(["blue", "cyan", "green", "yellow", "red"])
        else:
            cmap = Colormap(colormap)
        return cmap.map(color_norm)

    elif backend == "matplotlib":
        from matplotlib.colors import Colormap

        if colormap == "viridis":
            cmap = Colormap.from_list(
                "custom", ["blue", "cyan", "green", "yellow", "red"]
            )
        else:
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap(colormap)
        return cmap(color_norm)

    elif backend == "datoviz":
        import datoviz as dvz

        s = color_norm.astype(np.float32)
        return dvz.cmap(colormap, s)

    else:
        raise ValueError(f"Unknown backend: {backend}")


class ColorManager:
    """
    Consolidated color generation for all viewers.
    Handles the common pattern of converting scalar data to colors.

    Now supports global min/max for consistent color mapping across zoom levels.
    """

    def __init__(
        self,
        backend: str,
        colormap: str,
        default_color: str = "white",
        global_color_min: float | None = None,
        global_color_max: float | None = None,
    ):
        """
        Initialize color manager with backend-specific settings.

        Args:
            backend: "vispy", "matplotlib", or "datoviz"
            colormap: Name of colormap to use
            default_color: Default color when no color data provided
            global_color_min: Global minimum value for color normalization
            global_color_max: Global maximum value for color normalization
        """
        self.backend = backend
        self.colormap = colormap
        self.default_color = default_color
        self.global_color_min = global_color_min
        self.global_color_max = global_color_max

    def make_colors(self, color_data: np.ndarray | None) -> np.ndarray | str:
        """
        Generate colors from color data or return default.

        Args:
            color_data: Optional array of scalar values for coloring

        Returns:
            Color array (backend-specific format) or default color string
        """
        if color_data is not None:
            return make_colors_from_scalar(
                color_data,
                colormap=self.colormap,
                backend=self.backend,
                global_min=self.global_color_min,
                global_max=self.global_color_max,
            )
        return self.default_color
