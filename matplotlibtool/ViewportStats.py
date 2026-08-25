#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .AxisType import AxisType

if TYPE_CHECKING:
    from .AxisSecondaryConfig import AxisSecondaryConfig
    from .Plot2D import Plot2D


class ViewportStatsManager:
    """Signal statistics for the data inside the current x window.

    One row per visible plot, drawn in the figure margin below the x axis and
    recomputed on every view change. Rows window by the x limits only: a
    sample above or below the visible y range is still part of the record the
    operator is looking at, and scope-style measurements run over the visible
    time window, not the visible amplitude window. Statistics are computed on
    each plot's raw values, never its display transform: a presentation
    offset (a stddev lane parked above the data) must not shift the numbers.
    When a secondary y axis is configured its transform is applied first, so
    every value is reported in physical units, compacted with the same pint
    machinery the axis ticks use; otherwise values are raw codes.

    Tracking lanes (logic lines) carry timing rather than magnitude and are
    excluded.

    Nothing is computed unless it is going to be shown. The rows cost a pass
    over every visible point of every visible plot, so with the panel off the
    update returns before touching an array, and with it on a view change that
    left the x window where it was reuses what it already has. The window
    itself is taken with a binary search rather than a boolean mask wherever a
    plot's x is sorted, which is every plot whose x is a sample index or a
    time: a mask allocates and scans the whole array to find a span that two
    lookups locate.
    """

    BASE_BOTTOM = 0.12  # matches the Plot2D subplots_adjust default
    ROW_HEIGHT = 0.039
    MAX_ROWS = 6
    FONT_SIZE = 11.0

    def __init__(self, viewer: Plot2D, *, enabled: bool = False) -> None:
        self.viewer = viewer
        self.enabled = enabled
        # False marks every statistics row (uncalibrated): the code-to-unit
        # mapping is nominal rather than referenced to a known input voltage
        self.calibrated = True
        # the known reference on the ADC input connectors, when one was given;
        # listed at the end of every statistics row
        self.voltage_offset: float | None = None
        # when set, each row also carries the peak-to-peak span in raw ADC
        # codes, before the secondary axis transform
        self.show_adc_pp = False
        # converter word length; with it the code span also reads as the
        # bits it exercises, log2 of the span out of the full word. That is
        # range-spanned bits, not ENOB.
        self.adc_bits: int | None = None
        # one line above the statistics rows for capture context, e.g. the
        # acquisition description and any pin overrides
        self.header: str | None = None
        self._text = None
        self._bottom = self.BASE_BOTTOM
        self._cache_key: tuple | None = None
        self._cache_rows: list[str] | None = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled and self._text is not None:
            self._text.remove()
            self._text = None
            self._set_bottom(self.BASE_BOTTOM)

    def invalidate(self) -> None:
        """Drop what was computed, when what it was computed from has changed.

        Called when plots are added, hidden or restyled. The x window alone
        does not identify a set of rows: the same window over a different set
        of visible plots is a different answer.
        """
        self._cache_key = None
        self._cache_rows = None

    def _window(self, plot, xlim: tuple[float, float]) -> np.ndarray:
        """The plot's y values inside the x window.

        Sorted x is the common case and the cheap one: two searches instead of
        a full comparison and a full gather.
        """
        points = plot.points
        x = points[:, 0]
        lo = xlim[0] - plot.offset_x
        hi = xlim[1] - plot.offset_x
        if plot.x_ascending:
            start = int(np.searchsorted(x, lo, side="left"))
            stop = int(np.searchsorted(x, hi, side="right"))
            return points[start:stop, 1].astype(np.float64, copy=False)
        keep = (x >= lo) & (x <= hi)
        return points[keep, 1].astype(np.float64, copy=False)

    def restyle(self) -> None:
        if self._text is not None:
            self._text.set_color("white" if self.viewer.dark_mode else "black")

    def _set_bottom(self, bottom: float) -> None:
        if bottom != self._bottom:
            self.viewer.fig.subplots_adjust(bottom=bottom)
            self._bottom = bottom

    def _y_transform(self) -> tuple[float, float, AxisSecondaryConfig | None]:
        """(scale, offset, config) mapping raw y into secondary y units."""
        manager = self.viewer.view_manager.secondary_axis_manager
        if not manager.is_axis_enabled(AxisType.Y):
            return 1.0, 0.0, None
        config = manager.get_axis_config(AxisType.Y)
        if config is None:
            return 1.0, 0.0, None
        return config.scale, config.offset, config

    def _row(
        self,
        name: str,
        y: np.ndarray,
        config: AxisSecondaryConfig | None,
        adc_pp: float | None,
    ) -> str:
        n = int(y.size)
        if n == 0:
            return f"{name}: N=0"

        y_min = float(y.min())
        y_max = float(y.max())
        pp = y_max - y_min
        mean = float(y.mean())
        median = float(np.median(y))
        sd = float(y.std())
        rms = float(math.sqrt(np.mean(np.square(y))))
        peak = max(abs(y_min), abs(y_max))

        # DC signal-to-noise about the window mean; a flat trace is noiseless
        if sd == 0.0:
            snr = math.inf
        elif mean == 0.0:
            snr = -math.inf
        else:
            snr = 20.0 * math.log10(abs(mean) / sd)
        cf = math.inf if rms == 0.0 else peak / rms

        unit = ""
        factor = 1.0
        if config is not None:
            _, _, unit, factor = config.get_display_values(y_min, y_max)
        u = unit

        return (
            f"{name}: N={n}"
            f"  min={y_min * factor:.5g}{u}"
            f"  max={y_max * factor:.5g}{u}"
            f"  pp={pp * factor:.5g}{u}"
            f"{self._adc_pp_piece(adc_pp)}"
            f"  mean={mean * factor:.5g}{u}"
            f"  med={median * factor:.5g}{u}"
            f"  rms={rms * factor:.5g}{u}"
            f"  sd={sd * factor:.5g}{u}"
            f"  snr={snr:.4g}dB"
            f"  cf={cf:.4g}"
            f"{self._suffix()}"
        )

    def _suffix(self) -> str:
        if not self.calibrated:
            return "  (uncalibrated)"
        if self.voltage_offset is not None:
            return f"  offset={self.voltage_offset:g}V"
        return ""

    def _adc_pp_piece(self, adc_pp: float | None) -> str:
        if adc_pp is None:
            return ""
        piece = f"  adcpp={adc_pp:.6g}"
        if self.adc_bits is not None and adc_pp > 0.0:
            piece += f" ({math.log2(adc_pp):.4g}/{self.adc_bits} bits)"
        return piece

    def compute(
        self,
        xlim: tuple[float, float],
        *,
        limit: int | None = None,
    ) -> list[str]:
        """One row per visible plot, for the given x window.

        limit caps the rows and adds a count of what was left out, which is
        what the figure margin needs; without it every plot gets a row, which
        is what the dialog needs.
        """
        scale, offset, config = self._y_transform()
        rows: list[str] = []
        overflow = 0
        for index, plot in enumerate(self.viewer.plot_manager.get_all_plots()):
            if not plot.visible or plot.viewport_track or len(plot.points) == 0:
                continue
            if limit is not None and len(rows) == limit:
                overflow += 1
                continue
            y = self._window(plot, xlim)
            adc_pp = None
            if self.show_adc_pp and y.size:
                adc_pp = float(y.max() - y.min())
            if scale != 1.0 or offset != 0.0:
                y = y * scale + offset
            name = self.viewer.plot_manager.get_plot_name(index) or f"p{index}"
            rows.append(self._row(name, y, config, adc_pp))
        if overflow:
            rows.append(f"... +{overflow} more")
        return rows

    def report(self, xlim: tuple[float, float]) -> str:
        """Every plot's statistics, uncapped, for copying out."""
        rows = self.compute(xlim, limit=None)
        if self.header:
            rows[0:0] = self.header.splitlines()
        rows[0:0] = [f"x window: {xlim[0]:.10g} .. {xlim[1]:.10g}", ""]
        return "\n".join(rows)

    def update(self, xlim: tuple[float, float]) -> None:
        if not self.enabled:
            return

        # the same window over the same plots is the same answer, and a view
        # change that only moved y leaves both alone
        key = (
            xlim,
            self.show_adc_pp,
            self.adc_bits,
            self.header,
            tuple(
                (id(plot), plot.visible, plot.offset_x)
                for plot in self.viewer.plot_manager.get_all_plots()
            ),
        )
        if key == self._cache_key and self._cache_rows is not None:
            rows = self._cache_rows
        else:
            rows = self.compute(xlim, limit=self.MAX_ROWS)
            if self.header:
                rows[0:0] = self.header.splitlines()
            self._cache_key = key
            self._cache_rows = rows

        self._set_bottom(self.BASE_BOTTOM + self.ROW_HEIGHT * len(rows))

        body = "\n".join(rows)
        if self._text is None:
            self._text = self.viewer.fig.text(
                0.01,
                0.01,
                body,
                ha="left",
                va="bottom",
                family="monospace",
                fontsize=self.FONT_SIZE,
                color="white" if self.viewer.dark_mode else "black",
                linespacing=1.4,
            )
        else:
            self._text.set_text(body)
