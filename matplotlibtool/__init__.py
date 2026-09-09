"""
isort:skip_file

The viewer and its group context are imported when first asked for, not
when the package is: a module that wants only the lane arithmetic in
Plot2DOverlay does not pull in matplotlib's Qt backend to get it.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "Plot2D":
        from .Plot2D import Plot2D

        return Plot2D
    if name == "PlotGroupContext":
        from .PlotGroupContext import PlotGroupContext

        return PlotGroupContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Plot2D", "PlotGroupContext"]
