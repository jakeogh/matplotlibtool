#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import numpy as np


class InputState:
    def __init__(self):
        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.base_keys = set()
        self.shift_keys = set()

    def add_key(
        self,
        key_name: str,
        has_shift: bool,
    ):
        if key_name != "None":
            if has_shift:
                self.shift_keys.add(key_name)
            else:
                self.base_keys.add(key_name)

    def remove_key(self, key_name: str):
        if key_name != "None":
            self.shift_keys.discard(key_name)
            self.base_keys.discard(key_name)
