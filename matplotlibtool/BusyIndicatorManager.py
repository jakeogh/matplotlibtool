#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import time
from contextlib import contextmanager

from PyQt6.QtCore import Qt
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QLabel


def timestamp():
    """High resolution timestamp for logging."""
    return f"{time.time():.6f}"


class BusyIndicatorManager:
    """
    Manages busy state visual indicators for the viewer.

    STRICT VERSION: Fails loudly when status label is not properly connected.
    No more silent failures that hide real problems!
    """

    def __init__(self, status_label: None | QLabel = None):
        """
        Initialize busy indicator manager.

        Args:
            status_label: Optional QLabel widget to use as status indicator
        """
        self.status_label = status_label
        self.is_busy = False
        self.busy_count = 0

        # Timer to delay busy indicator to avoid flicker on very fast operations
        self.busy_timer = QTimer()
        self.busy_timer.setSingleShot(True)
        self.busy_timer.timeout.connect(self._show_busy_immediate)
        self.busy_delay_ms = 0

        # Timer to ensure minimum busy display time for visibility
        self.min_busy_timer = QTimer()
        self.min_busy_timer.setSingleShot(True)
        self.min_busy_timer.timeout.connect(self._hide_busy_immediate)
        self.min_busy_time_ms = 1000

        self._pending_hide = False

        self.original_palette = None

    def set_status_label(self, label: QLabel) -> None:
        """Set the status label widget - REQUIRED for operations."""
        if label is None:
            raise ValueError("status_label cannot be None!")
        if not isinstance(label, QLabel):
            raise TypeError(f"status_label must be QLabel, got {type(label).__name__}")

        self.status_label = label
        self.original_palette = label.palette()
        self._apply_idle_style()

    def _require_status_label(self) -> None:
        """Internal method to ensure status label is connected before operations."""
        if self.status_label is None:
            raise RuntimeError(
                "BusyIndicatorManager: No status label connected! "
                "Call set_status_label() before using busy operations. "
                "This is a programming error that must be fixed."
            )

    @contextmanager
    def busy_operation(self, operation_name: str = "Processing"):
        """
        Context manager for operations that should show busy state.

        Args:
            operation_name: Name of the operation for debugging

        Usage:
            with busy_manager.busy_operation("Updating plot"):
                # Do expensive operation
                update_plot()
        """
        self.start_busy(operation_name)
        try:
            yield
        finally:
            self.end_busy(operation_name)

    def start_busy(self, operation_name: str = "Processing") -> None:
        """Start a busy operation (non-context manager API)."""
        self._require_status_label()
        self._start_busy(operation_name)

    def end_busy(self, operation_name: str = "Processing") -> None:
        """End a busy operation (non-context manager API)."""
        self._require_status_label()
        self._end_busy(operation_name)

    def _start_busy(self, operation_name: str) -> None:
        """Start a busy operation."""
        self.busy_count += 1

        if self.busy_count == 1:
            self._pending_hide = False

            if not self.is_busy:
                if self.busy_delay_ms == 0:
                    self._show_busy_immediate()
                else:
                    self.busy_timer.start(self.busy_delay_ms)

    def _end_busy(self, operation_name: str) -> None:
        """End a busy operation."""
        self.busy_count = max(0, self.busy_count - 1)

        if self.busy_count == 0:
            self.busy_timer.stop()

            if self.is_busy:
                self._pending_hide = True
                if not self.min_busy_timer.isActive():
                    self.min_busy_timer.start(self.min_busy_time_ms)

    def _show_busy_immediate(self) -> None:
        """Show busy indicator immediately."""
        if self.busy_count > 0 and not self.is_busy:
            self.is_busy = True
            self._apply_busy_style()

    def _hide_busy_immediate(self) -> None:
        """Hide busy indicator immediately."""
        if self._pending_hide:
            self.is_busy = False
            self._pending_hide = False
            self._apply_idle_style()

    def _apply_busy_style(self) -> None:
        """Apply busy visual style using QPalette (bypasses CSS)."""
        self._require_status_label()

        # Use QPalette to bypass CSS completely
        busy_palette = QPalette()
        busy_palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        busy_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        busy_palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)

        self.status_label.setText("BUSY")
        self.status_label.setPalette(busy_palette)
        self.status_label.setAutoFillBackground(True)

        # Also try stylesheet with maximum specificity
        self.status_label.setStyleSheet(
            """
            QLabel {
                background-color: #000000 !important;
                color: #ffffff !important;
                border: 2px solid #333333 !important;
                border-radius: 4px !important;
                padding: 4px 8px !important;
                font-weight: bold !important;
                font-size: 10px !important;
            }
        """
        )

        self.status_label.update()
        self.status_label.repaint()

    def _apply_idle_style(self) -> None:
        """Apply idle visual style."""
        self._require_status_label()

        # Restore original palette
        if self.original_palette:
            self.status_label.setPalette(self.original_palette)

        self.status_label.setAutoFillBackground(False)
        self.status_label.setText("")

        # Clear any custom stylesheet
        self.status_label.setStyleSheet("")

        self.status_label.update()
        self.status_label.repaint()
