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

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled and self._text is not None:
            self._text.remove()
            self._text = None
            self._set_bottom(self.BASE_BOTTOM)

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

    def update(self, xlim: tuple[float, float]) -> None:
        if not self.enabled:
            return

        scale, offset, config = self._y_transform()
        rows: list[str] = []
        overflow = 0
        for index, plot in enumerate(self.viewer.plot_manager.get_all_plots()):
            if not plot.visible or plot.viewport_track or len(plot.points) == 0:
                continue
            if len(rows) == self.MAX_ROWS:
                overflow += 1
                continue
            x = plot.points[:, 0].astype(np.float64) + plot.offset_x
            y = plot.points[:, 1].astype(np.float64)
            y = y[(x >= xlim[0]) & (x <= xlim[1])]
            adc_pp = None
            if self.show_adc_pp and y.size:
                adc_pp = float(y.max() - y.min())
            y = y * scale + offset
            name = self.viewer.plot_manager.get_plot_name(index) or f"p{index}"
            rows.append(self._row(name, y, config, adc_pp))
        if overflow:
            rows.append(f"... +{overflow} more")
        if self.header:
            rows[0:0] = self.header.splitlines()

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
