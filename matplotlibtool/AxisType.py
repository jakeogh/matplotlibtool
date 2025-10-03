#!/usr/bin/env python3


from __future__ import annotations

from enum import Enum


class AxisType(Enum):
    """Enum to specify which axis (X or Y) for secondary configuration."""

    X = "x"
    Y = "y"
