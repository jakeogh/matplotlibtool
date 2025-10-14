#!/usr/bin/env python3
# tab-width:4

"""
File Loader Registry Module for PointCloud2DViewerMatplotlib

This module provides a pluggable file loader system that allows users to register
custom file loaders for different file extensions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QFileDialog


class FileLoaderRegistry:
    """
    Registry for file loaders that can handle different file formats.

    This class manages the registration of file loaders for specific extensions
    and provides methods to load files using the registered loaders.
    """

    def __init__(self, viewer=None):
        """
        Initialize the file loader registry.

        Args:
            viewer: Optional reference to the viewer for status messages
        """
        self.viewer = viewer
        self.file_loaders: dict[str, Callable] = {}
        self._registered_count = 0

    def register_loader(
        self,
        extensions: str | list[str],
        loader_func: Callable[[list[str]], list[np.ndarray]],
    ) -> None:
        """
        Register a loader function for specific file extensions.

        Args:
            extensions: String extension (e.g. '.iio') or list of extensions
            loader_func: Function that takes list of file paths and returns list of np.ndarray
                        Each array should have shape (N, 2+) for (x, y[, color, ...])

        Example:
            def my_iio_loader(paths):
                return [load_iio_file(path) for path in paths]

            registry.register_loader('.iio', my_iio_loader)
            registry.register_loader(['.csv', '.txt'], my_text_loader)
        """
        if isinstance(extensions, str):
            extensions = [extensions]

        for ext in extensions:
            ext_clean = self._clean_extension(ext)
            self.file_loaders[ext_clean] = loader_func
            self._registered_count += 1
            print(f"[INFO] Registered file loader for {ext_clean} files")

    def unregister_loader(self, extensions: str | list[str]) -> None:
        """
        Unregister file loader(s) for specific extensions.

        Args:
            extensions: String extension or list of extensions to unregister
        """
        if isinstance(extensions, str):
            extensions = [extensions]

        for ext in extensions:
            ext_clean = self._clean_extension(ext)
            if ext_clean in self.file_loaders:
                del self.file_loaders[ext_clean]
                self._registered_count -= 1
                print(f"[INFO] Unregistered file loader for {ext_clean} files")

    def get_registered_extensions(self) -> list[str]:
        """
        Get list of currently registered file extensions.

        Returns:
            List of registered extensions
        """
        return list(self.file_loaders.keys())

    def has_loaders(self) -> bool:
        """
        Check if any file loaders are registered.

        Returns:
            True if at least one loader is registered
        """
        return len(self.file_loaders) > 0

    def load_files(self, file_paths: list[str]) -> list[np.ndarray]:
        """
        Load files using registered loaders.

        Args:
            file_paths: List of file paths to load

        Returns:
            List of numpy arrays with loaded data

        Raises:
            ValueError: If no loader is registered for a file extension
        """
        extension_groups = self._group_files_by_extension(file_paths)

        all_plots = []

        for ext, ext_paths in extension_groups.items():
            if ext in self.file_loaders:
                loader_func = self.file_loaders[ext]
                try:
                    plots = loader_func(ext_paths)
                    if plots:
                        all_plots.extend(plots)
                        print(
                            f"[INFO] Loaded {len(plots)} plots from {len(ext_paths)} {ext} file(s)"
                        )
                except Exception as e:
                    print(f"[ERROR] Failed loading {ext} files: {e}")
            else:
                print(
                    f"[WARNING] No loader registered for {ext} files (skipping {len(ext_paths)} files)"
                )

        return all_plots

    def open_file_dialog(self, parent=None) -> list[str] | None:
        """
        Open a file dialog for selecting files to load.

        Args:
            parent: Parent widget for the dialog

        Returns:
            List of selected file paths or None if cancelled
        """
        if not self.has_loaders():
            print(
                "[INFO] No file loaders registered. Use register_loader() to add support for file types."
            )
            return None

        filter_string = self._build_file_filter()

        paths, _ = QFileDialog.getOpenFileNames(
            parent,
            "Add Data Files",
            "",
            filter_string,
        )

        return paths if paths else None

    def _clean_extension(self, ext: str) -> str:
        """
        Clean and standardize a file extension.

        Args:
            ext: Extension string to clean

        Returns:
            Cleaned extension (lowercase, with leading dot)
        """
        ext_clean = ext.lower().strip()
        if not ext_clean.startswith("."):
            ext_clean = "." + ext_clean
        return ext_clean

    def _group_files_by_extension(self, file_paths: list[str]) -> dict[str, list[str]]:
        """
        Group file paths by their extensions.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping extensions to lists of file paths
        """
        extension_groups = {}
        for path in file_paths:
            ext = Path(path).suffix.lower()
            if ext not in extension_groups:
                extension_groups[ext] = []
            extension_groups[ext].append(path)
        return extension_groups

    def _build_file_filter(self) -> str:
        """
        Build a file filter string for QFileDialog.

        Returns:
            Filter string for file dialog
        """
        registered_extensions = self.get_registered_extensions()

        if not registered_extensions:
            return "All files (*)"

        filter_parts = []
        for ext in registered_extensions:
            filter_parts.append(f"*{ext}")

        all_supported = "Supported files (" + " ".join(filter_parts) + ")"
        individual_filters = [
            f"{ext.upper()[1:]} files (*{ext})" for ext in registered_extensions
        ]

        file_filter = (
            all_supported + ";;" + ";;".join(individual_filters) + ";;All files (*)"
        )

        return file_filter
