#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np


@dataclass
class Overlay:
    """Dataclass to store overlay plot configuration/state."""

    points: np.ndarray
    cmap: str
    color_data: None | np.ndarray = None
    draw_lines: bool = False
    size: float = 2.0
    color: None | str = None
    offset_x: float = 0.0
    offset_y: float = 0.0
    visible: bool = True
    line_color: None | str = (
        None  # None = use point colors, or specify like "gray", "red", etc.
    )
    line_width: float = 1.0

    # NEW: Artist tracking for persistent rendering
    # These will store matplotlib artist references when created
    scatter_artist: Any = field(
        default=None,
        init=False,
        repr=False,
    )
    line_artist: Any = field(
        default=None,
        init=False,
        repr=False,
    )
