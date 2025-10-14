#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import time
from typing import Set


class KeyboardInputManager:
    """
    Consolidated keyboard input handling and scaling logic for all viewers.
    Handles the common pattern of axis scaling via keyboard input with key repeat prevention.
    """

    def __init__(
        self,
        input_state,
        acceleration: float = 0.5,
    ):
        """
        Initialize with a reference to the viewer's InputState.

        Args:
            input_state: The InputState instance from the viewer
            acceleration: Base acceleration value (0.1-2.0 recommended)
        """
        self.state = input_state
        self.acceleration = acceleration

        self.max_velocity = 10.0
        self.velocity_decay = 0.85

        # Key repeat prevention - track when keys were last processed
        self.last_key_process_time = {}
        self.key_repeat_threshold = 0.03

        # Active key tracking to prevent OS key repeat issues
        self.currently_pressed_keys: set[str] = set()

    def add_key_with_repeat_check(
        self,
        key_name: str,
        has_shift: bool,
    ) -> bool:
        """
        Add a key to the input state with key repeat checking.

        Args:
            key_name: Name of the key
            has_shift: Whether shift is held

        Returns:
            True if key was added (not a repeat), False if ignored
        """
        if key_name == "None":
            return False

        key_id = f"{key_name}{'_SHIFT' if has_shift else ''}"

        if key_id in self.currently_pressed_keys:
            return False

        self.currently_pressed_keys.add(key_id)

        if has_shift:
            self.state.shift_keys.add(key_name)
        else:
            self.state.base_keys.add(key_name)

        print(f"[KEYBOARD] Added key: {key_id}")
        return True

    def remove_key_with_repeat_check(self, key_name: str) -> bool:
        """
        Remove a key from the input state and tracking.

        Args:
            key_name: Name of the key

        Returns:
            True if key was removed, False if wasn't being tracked
        """
        if key_name == "None":
            return False

        key_id_base = key_name
        key_id_shift = f"{key_name}_SHIFT"

        removed = False
        if key_id_base in self.currently_pressed_keys:
            self.currently_pressed_keys.remove(key_id_base)
            removed = True

        if key_id_shift in self.currently_pressed_keys:
            self.currently_pressed_keys.remove(key_id_shift)
            removed = True

        self.state.shift_keys.discard(key_name)
        self.state.base_keys.discard(key_name)

        if removed:
            print(f"[KEYBOARD] Removed key: {key_name}")

        return removed

    def should_process_key(self, key_name: str) -> bool:
        """
        Check if a key should be processed based on timing to prevent key repeat spam.

        Args:
            key_name: Name of the key (e.g., "X", "Y", "Z")

        Returns:
            True if key should be processed, False if too soon
        """
        now = time.time()
        last_time = self.last_key_process_time.get(key_name, 0)

        if now - last_time >= self.key_repeat_threshold:
            self.last_key_process_time[key_name] = now
            return True
        return False

    def update_scaling(
        self,
        dt: float,
        dimensions: int = 3,
    ) -> None:
        """
        Update scaling based on keyboard input with improved acceleration.

        Args:
            dt: Delta time since last update
            dimensions: Number of dimensions (2 for 2D viewers, 3 for 3D viewers)
        """
        accel = self.acceleration * dt * 5.0

        axis_names = ["X", "Y", "Z"][:dimensions]

        for i, axis in enumerate(axis_names):
            old_velocity = self.state.velocity[i]

            if axis in self.state.shift_keys:
                self.state.velocity[i] = max(
                    self.state.velocity[i] - accel, -self.max_velocity
                )
            elif axis in self.state.base_keys:
                self.state.velocity[i] = min(
                    self.state.velocity[i] + accel, self.max_velocity
                )
            else:
                self.state.velocity[i] *= self.velocity_decay
                if abs(self.state.velocity[i]) < 0.001:
                    self.state.velocity[i] = 0.0

            velocity = self.state.velocity[i]

            scale_factor = 1.0 + velocity * dt * 0.5

            scale_factor = max(0.01, min(scale_factor, 2.0))

            self.state.scale[i] *= scale_factor

            self.state.scale[i] = max(1e-6, min(self.state.scale[i], 1e6))

        self.clear_input_state()

    def update_scaling_2d_lowercase(self, dt: float) -> None:
        """
        Update scaling for 2D viewers that use lowercase axis names.
        Matches the exact behavior of PointCloud2DViewerVispy but with better acceleration.

        Args:
            dt: Delta time since last update
        """
        accel = self.acceleration * dt * 5.0

        for i, axis in enumerate(["x", "y"]):
            if axis.upper() in self.state.shift_keys:
                self.state.velocity[i] = max(
                    self.state.velocity[i] - accel, -self.max_velocity
                )
            elif axis.upper() in self.state.base_keys:
                self.state.velocity[i] = min(
                    self.state.velocity[i] + accel, self.max_velocity
                )
            else:
                self.state.velocity[i] *= self.velocity_decay
                if abs(self.state.velocity[i]) < 0.001:
                    self.state.velocity[i] = 0.0

            velocity = self.state.velocity[i]
            scale_factor = 1.0 + velocity * dt * 0.5
            scale_factor = max(0.01, min(scale_factor, 2.0))

            self.state.scale[i] *= scale_factor
            self.state.scale[i] = max(1e-6, min(self.state.scale[i], 1e6))

        self.clear_input_state()

    def clear_input_state(self) -> None:
        """Clear all input state to prevent sticky keys."""
        self.state.base_keys.clear()
        self.state.shift_keys.clear()
        self.currently_pressed_keys.clear()

    def set_acceleration(self, acceleration: float) -> None:
        """Update the acceleration value with validation."""
        self.acceleration = max(0.01, min(acceleration, 5.0))
        print(f"[INFO] Acceleration set to: {self.acceleration:.3f}")
