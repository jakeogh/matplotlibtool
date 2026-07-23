#!/usr/bin/env python3

from __future__ import annotations

from enum import Enum


class MouseMode(Enum):
    ZOOM = "zoom"
    PAN = "pan"
    HOVER = "hover"
