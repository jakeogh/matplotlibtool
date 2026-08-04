#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.ticker import EngFormatter

from .FFTAnalysis import FFTAnalysisError
from .PixelAnalysis import PixelAnalysisError
from .PixelAnalysis import analyse_pixels
from .PixelAnalysis import format_report
from .FFTAnalysis import FFTPeakArtists
from .FFTAnalysis import FFTResult
from .FFTAnalysis import OVERLAP
from .FFTAnalysis import WINDOW
from .FFTAnalysis import analyze_fft
from .MouseMode import MouseMode
from .SettleAnalysis import FLOOR_C
from .SettleAnalysis import MIN_FIT_POINTS
from .SettleAnalysis import SettleAnalysisArtists
from .SettleAnalysis import SettleAnalysisError
from .SettleAnalysis import x_formatter
from .SettleAnalysis import analyze_settle
from .SettleAnalysis import text_color


class PlotEventHandlers:
    """Event and user-interaction handlers for the 2D matplotlib viewer."""

    def __init__(self, viewer):
        self.viewer = viewer

        # throttle keyboard scaling to avoid key-repeat render storms
        self.last_scale_update = 0.0
        self.scale_throttle_ms = 50

        self._settle_artists = None
        self._ref_annotation = None
        self._last_analysis = None   # (plot_index, SettleSegments)
        self._fft_windows: list = []
        self._peak_artists = None
        self._fft_source = None      # FFTResult carried by a spectrum window
        self._fft_plot_index = None  # index of the spectrum plot in this viewer

    def _should_throttle_scaling(self) -> bool:
        now = time.time() * 1000
        if now - self.last_scale_update < self.scale_throttle_ms:
            return True
        self.last_scale_update = now
        return False

    def on_timer(self):
        """Timer callback driving keyboard axis scaling."""
        now = time.time()
        dt = now - self.viewer.last_time
        self.viewer.last_time = now

        old_scale = self.viewer.state.scale.copy()
        self.viewer.keyboard_manager.update_scaling(dt, dimensions=2)

        if np.allclose(old_scale, self.viewer.state.scale):
            return
        if self._should_throttle_scaling():
            return
        if self.viewer.busy_manager.is_busy:
            return

        with self.viewer.busy_manager.busy_operation("Scaling"):
            self.viewer.apply_keyboard_scale()

    def on_dark_mode_toggled(self, enabled: bool):
        self.viewer.set_dark_mode(enabled)

    def on_add_files(self):
        """Open QFileDialog for supported file types and append resulting plots."""
        paths = self.viewer.file_loader_registry.open_file_dialog(self.viewer)
        if not paths:
            return

        with self.viewer.busy_manager.busy_operation("Loading data files"):
            all_plots = self.viewer.file_loader_registry.load_files(paths)

            added = 0
            for arr in all_plots:
                self.viewer.add_plot(
                    arr,
                    x_field="sample",
                    y_field="in0",
                )
                added += 1

            if added:
                print(f"[INFO] Successfully added {added} plot(s) total")
            else:
                print("[INFO] No plots were successfully added")

    def on_acceleration_changed(self, value: float) -> None:
        self.viewer.acceleration = float(value)
        self.viewer.keyboard_manager.set_acceleration(self.viewer.acceleration)

    def on_plot_selection_changed(self, plot_index: int) -> None:
        self.viewer.plot_manager.select_plot(plot_index)
        self.viewer.control_bar_integration.sync_controls_to_selection()

    def on_group_selection_changed(self, group_id: int) -> None:
        self.viewer.plot_manager.select_group(group_id)
        self.viewer.control_bar_integration.sync_controls_to_selection()

    def _set_selected_property(self, property_name: str, value) -> None:
        if self.viewer.plot_manager.is_group_selected():
            self.viewer.plot_manager.set_group_property(
                self.viewer.plot_manager.selected_group_id,
                property_name,
                value,
            )
        else:
            self.viewer.plot_manager.set_plot_property(
                self.viewer.plot_manager.selected_plot_index,
                property_name,
                value,
            )

    def on_point_size_changed(self, value: float) -> None:
        self._set_selected_property("size", value)

    def on_line_width_changed(self, value: float) -> None:
        self._set_selected_property("line_width", value)

    def on_lines_toggled(self, checked: bool) -> None:
        self._set_selected_property("draw_lines", checked)

    def on_palette_changed(self, palette_name: str):
        if palette_name.startswith("───") or palette_name == "(Mixed)":
            return
        self._set_selected_property("colormap", palette_name)

    def on_visibility_toggled(self, visible: bool):
        if self.viewer.plot_manager.is_group_selected():
            group_id = self.viewer.plot_manager.selected_group_id
            group_info = self.viewer.plot_manager.get_group_info(group_id)
            if group_info:
                for plot_index in group_info.plot_indices:
                    self.viewer.plot_manager.set_plot_visibility(plot_index, visible)
        else:
            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_visibility(plot_index, visible)

    def on_save_figure(self):
        """Auto-save figure to /delme with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("/delme")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"autosave_figure_{timestamp}.jpg"

        with self.viewer.busy_manager.busy_operation("Saving figure"):
            self.viewer._render_to_file(filepath, dpi=300)
            print(f"[INFO] Figure auto-saved to: {filepath}")

    def on_save_data(self):
        """Auto-save the visible plots' in-window samples to /delme as CSV."""
        bounds = self.viewer.view_manager.get_current_bounds()
        xlim = bounds.xlim
        pm = self.viewer.plot_manager
        y_mgr = self.viewer.view_manager.secondary_axis_manager.y_axis_manager

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("/delme")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"autosave_data_{timestamp}.csv"

        with self.viewer.busy_manager.busy_operation("Saving data"):
            rows = 0
            with filepath.open("w") as handle:
                handle.write(f"# saved: {datetime.now().isoformat()}\n")
                handle.write(f"# x_window: {float(xlim[0]):.10g} {float(xlim[1]):.10g}\n")
                handle.write(
                    f"# y_window: {float(bounds.ylim[0]):.10g} "
                    f"{float(bounds.ylim[1]):.10g}\n"
                )
                rate = self.viewer.sample_rate_hz
                handle.write(
                    f"# sample_rate_sps: {rate:.10g}\n" if rate
                    else "# sample_rate_sps: unset\n"
                )
                handle.write(f"# settle_mode: {self.viewer.display_space}\n")
                if y_mgr.is_enabled() and y_mgr.config is not None:
                    cfg = y_mgr.config
                    handle.write(
                        f"# y_secondary: {cfg.label} [{cfg.unit}] = "
                        f"{cfg.scale!r} * y + {cfg.offset!r}\n"
                    )
                handle.write("# columns: plot,x,y,color\n")
                handle.write("# y is raw sample value before y_scale and offsets\n")

                for i, plot in enumerate(pm.plots):
                    if not plot.visible or len(plot.points) == 0:
                        continue
                    name = pm.get_plot_name(i) or f"plot{i}"
                    x = plot.points[:, 0] + plot.offset_x
                    mask = (x >= xlim[0]) & (x <= xlim[1])
                    n = int(mask.sum())
                    if n == 0:
                        continue
                    handle.write(
                        f"# plot: {name} n={n} y_scale={plot.y_scale!r} "
                        f"offset_x={plot.offset_x!r} offset_y={plot.offset_y!r} "
                        f"settle_ref={plot.settle_ref!r}\n"
                    )
                    sel = plot.points[mask]
                    if plot.color_data is None:
                        color = np.full(n, np.nan)
                    else:
                        color = np.asarray(plot.color_data)[mask]
                    for (px, py), pc in zip(sel, color):
                        handle.write(f"{name},{px:.10g},{py:.10g},{pc:.10g}\n")
                    rows += n

            print(f"[INFO] Data auto-saved to: {filepath} ({rows} samples)")

    def on_sample_rate_changed(self, rate_hz: float) -> None:
        """Set the x-axis sample rate used to report analysis results in time."""
        self.viewer.sample_rate_hz = rate_hz if rate_hz > 0.0 else None
        if self.viewer.sample_rate_hz is None:
            print("[INFO] Sample rate cleared; analysis reports x-units only")
        else:
            print(f"[INFO] Sample rate: {self.viewer.sample_rate_hz:,.0f} SPS")

        self._redraw_analysis_overlay()
        self._rescale_attached_spectrum()

        # spectra derive from this record: keep open spectrum windows on the
        # source rate, mirroring it into their Rate widgets
        for window in self._fft_windows:
            window.control_bar_manager.set_sample_rate_display(
                self.viewer.sample_rate_hz
            )
            window.event_handlers.on_sample_rate_changed(
                self.viewer.sample_rate_hz or 0.0
            )

    def on_settle_toggled(self, enabled: bool) -> None:
        """Toggle log10|y - ref| display; ref from the in-view tail per plot."""
        try:
            self._apply_settle_mode(enabled)
        except SettleAnalysisError as exc:
            print(f"[INFO] {exc}")
            self._apply_settle_mode(False)
            self.viewer.control_bar_manager.set_settle_checked(False)

    def _apply_settle_mode(self, enabled: bool) -> None:
        self.viewer.view_manager.secondary_axis_manager.set_residual_mode(enabled)
        self._set_settle_axis_label(enabled)
        if enabled:
            xlim = self.viewer.view_manager.get_current_bounds().xlim
            for i, plot in enumerate(self.viewer.plot_manager.plots):
                if not plot.visible or len(plot.points) == 0:
                    plot.settle_ref = None
                    continue

                lin = plot.points[:, 1] * plot.y_scale + plot.offset_y
                x = plot.points[:, 0] + plot.offset_x
                idx = np.flatnonzero((x >= xlim[0]) & (x <= xlim[1]))
                name = self.viewer.plot_manager.get_plot_name(i) or f"Plot {i + 1}"
                if idx.size < 16:
                    raise SettleAnalysisError(
                        f"settle mode: {name} has {idx.size} samples in view, "
                        f"need >= 16 to estimate a settled reference"
                    )

                # reuse the converged reference so the markers stay aligned
                # when flipping between the linear and settle views
                if self._last_analysis is not None and self._last_analysis[0] == i:
                    plot.settle_ref = self._last_analysis[1].y_final
                    print(f"[INFO] Settle ref {name}: {plot.settle_ref:.6g} (analysis)")
                    continue

                idx = idx[np.argsort(plot.points[idx, 0], kind="stable")]
                tail = idx[-max(16, idx.size // 10) :]
                plot.settle_ref = float(lin[tail].mean())
                print(
                    f"[INFO] Settle ref {name}: {plot.settle_ref:.6g} "
                    f"({tail.size} tail samples)"
                )
        else:
            for plot in self.viewer.plot_manager.plots:
                plot.settle_ref = None
            print("[INFO] Settle mode disabled")

        self._update_ref_annotation()
        self.viewer.fit_y_to_view()
        self._redraw_analysis_overlay()

    def _redraw_analysis_overlay(self) -> None:
        """Re-render the segmentation for whichever space the plot is now in."""
        if self._last_analysis is None:
            return
        if self._settle_artists is None:
            self._settle_artists = SettleAnalysisArtists(self.viewer.ax)
        self._settle_artists.draw(
            self._last_analysis[1],
            self.viewer.sample_rate_hz,
            settle=self.viewer.display_space == "settle",
        )
        self.viewer.canvas.draw_idle()

    def _set_settle_axis_label(self, enabled: bool) -> None:
        color = "white" if self.viewer.dark_mode else "black"
        label = "log10 |y \u2212 ref|  (decades of ADC codes)" if enabled else ""
        self.viewer.ax.set_ylabel(label, color=color)

    def _update_ref_annotation(self) -> None:
        """Annotate each visible plot's settle reference (codes and physical)."""
        if self._ref_annotation is not None:
            self._ref_annotation.remove()
            self._ref_annotation = None

        pm = self.viewer.plot_manager
        y_mgr = self.viewer.view_manager.secondary_axis_manager.y_axis_manager

        lines = []
        for i, plot in enumerate(pm.plots):
            if plot.settle_ref is None or not plot.visible:
                continue
            name = pm.get_plot_name(i) or f"Plot {i + 1}"
            line = f"{name}: ref {plot.settle_ref:,.1f}"
            if y_mgr.is_enabled() and y_mgr.config is not None:
                cfg = y_mgr.config
                physical = cfg.scale * plot.settle_ref + cfg.offset
                line += f" = {physical:.8g} {cfg.unit}"
            lines.append(line)

        if not lines:
            return

        ax = self.viewer.ax
        color = text_color(ax)
        self._ref_annotation = ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            color=color,
            fontsize=9,
            verticalalignment="top",
            zorder=1000,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=ax.get_facecolor(),
                edgecolor=color,
                alpha=0.8,
            ),
        )

    def on_analyze_toggled(self, enabled: bool) -> None:
        """Segment and fit the largest step in view for the selected plot."""
        try:
            self._run_analysis(enabled)
        except SettleAnalysisError as exc:
            print(f"[INFO] {exc}")
            self.viewer.control_bar_manager.set_analyze_checked(False)

    def _report_pole_fit(self, seg, fmt) -> None:
        """Report the exponential fit, or why there isn't one."""
        fit = seg.fit
        if fit is None:
            print(
                f"[INFO]   tau(fit): none. The residual clears the "
                f"{int(FLOOR_C)}-sigma floor in fewer than {MIN_FIT_POINTS} "
                f"samples, so the decay is unresolved at this sample rate and "
                f"only tau(rise) is available. Raise the sample rate to fit it"
            )
            return

        print(
            f"[INFO]   linear:   x {fit.start_x:.6g} .. {fit.end_x:.6g} "
            f"({fit.n_points} pts), slope {fit.slope:.4g} dec/x, "
            f"rms {fit.rms:.3g} dec"
        )
        print(
            f"[INFO]   tau(fit): {fmt(fit.tau)} from the log-residual slope "
            f"(1 decade per {fmt(-1.0 / fit.slope)})"
        )

        # the two tau estimates agree only for a single pole; which way they
        # disagree says what the trace is actually doing
        ratio = fit.tau / seg.tau_from_rise
        if ratio > 1.3:
            print(
                f"[INFO]   WARNING: the fitted tau is {ratio:.3g}x the one implied "
                f"by the rise time; the fit is following a slow tail, not the "
                f"edge. The dominant pole is tau(rise) = {fmt(seg.tau_from_rise)}; "
                f"tau(fit) = {fmt(fit.tau)} describes a separate slow component"
            )
        elif ratio < 0.77:
            print(
                f"[INFO]   WARNING: the measured rise is {1.0 / ratio:.3g}x slower "
                f"than the fitted tau implies; the edge is slew limited, not "
                f"bandwidth limited"
            )
        halves = (fit.slope_first_half, fit.slope_second_half)
        if abs(halves[0] - halves[1]) > 0.15 * abs(fit.slope):
            print(
                f"[INFO]   WARNING: slope changes {halves[0]:.4g} -> "
                f"{halves[1]:.4g} dec/x across the region; possible secondary "
                f"pole or thermal tail"
            )
        if fit.lead_trim_decades > 0.5:
            print(
                f"[INFO]   WARNING: fit rejected the top "
                f"{fit.lead_trim_decades:.2g} decades of the settle; the early "
                f"response is not on this pole (secondary pole or slew), tau "
                f"describes the late tail only"
            )
        if fit.tail_trim_decades > 0.5:
            print(
                f"[INFO]   WARNING: fit rejected the bottom "
                f"{fit.tail_trim_decades:.2g} decades above the noise floor; "
                f"the late response departs from this pole (thermal tail or "
                f"dielectric absorption)"
            )

    def _run_analysis(self, enabled: bool) -> None:
        if not enabled:
            if self._settle_artists is not None:
                self._settle_artists.clear()
            self._last_analysis = None
            self.viewer.canvas.draw_idle()
            print("[INFO] Analysis markers cleared")
            return

        pm = self.viewer.plot_manager
        selected = pm.get_selected_plots()
        if len(selected) != 1:
            raise SettleAnalysisError(
                "settle analysis: select a single plot in the Plot/Group dropdown"
            )
        plot_index = selected[0]
        plot = pm.plots[plot_index]
        name = pm.get_plot_name(plot_index) or f"Plot {plot_index + 1}"

        xlim = self.viewer.view_manager.get_current_bounds().xlim
        seg = analyze_settle(
            plot.points[:, 0] + plot.offset_x,
            plot.points[:, 1] * plot.y_scale + plot.offset_y,
            xlim,
        )
        self._last_analysis = (plot_index, seg)

        fmt = x_formatter(self.viewer.sample_rate_hz)
        print(f"[INFO] Settle analysis: {name}")
        print(
            f"[INFO]   step:     {seg.step_height:+.6g} "
            f"(pre {seg.y_pre:.6g} -> final {seg.y_final:.6g})"
        )
        print(
            f"[INFO]   noise:    sigma {seg.noise_sigma:.4g} "
            f"(baseline {seg.baseline_sigma:.4g}, {seg.baseline_n} samples)"
        )
        print(f"[INFO]   edge:     x {seg.edge_start_x:.6g} .. {seg.edge_end_x:.6g}")
        print(
            f"[INFO]   rise:     10-90% {fmt(seg.rise_10_90)}, "
            f"20-80% {fmt(seg.rise_20_80)}, x {seg.rise_x10:.6g} .. "
            f"{seg.rise_x90:.6g}"
        )
        print(
            f"[INFO]   tau(rise): {fmt(seg.tau_from_rise)} from the 10-90% transition"
        )
        print(
            f"[INFO]   settled:  x {seg.settled_x:.6g}, settling time "
            f"{fmt(seg.settling_time)} to the 4-sigma band"
        )

        self._report_pole_fit(seg, fmt)

        plot.settle_ref = seg.y_final
        self.viewer.control_bar_manager.set_settle_checked(True)
        self.viewer.view_manager.secondary_axis_manager.set_residual_mode(True)
        self._set_settle_axis_label(True)
        self._update_ref_annotation()

        if seg.baseline_x0 < xlim[0] or seg.span_x1 > xlim[1]:
            print(
                f"[INFO]   note:     the analyzed event spans x "
                f"{seg.baseline_x0:.6g} .. {seg.span_x1:.6g}, wider than the "
                f"current view; zoom out to see the full segmentation"
            )

        self.viewer.fit_y_to_view()

        if self._settle_artists is None:
            self._settle_artists = SettleAnalysisArtists(self.viewer.ax)
        self._settle_artists.draw(seg, self.viewer.sample_rate_hz, settle=True)
        self.viewer.canvas.draw_idle()

    def on_fft(self) -> None:
        """Open the spectrum of the selected plot's in-view samples."""
        try:
            self._run_fft()
        except FFTAnalysisError as exc:
            print(f"[INFO] {exc}")

    def _run_fft(self) -> None:
        pm = self.viewer.plot_manager
        if not pm.plots:
            raise FFTAnalysisError("fft: no plots loaded")
        selected = pm.get_selected_plots()
        if len(selected) != 1:
            raise FFTAnalysisError(
                "fft: select a single plot in the Plot/Group dropdown"
            )
        plot_index = selected[0]
        plot = pm.plots[plot_index]
        name = pm.get_plot_name(plot_index) or f"Plot {plot_index + 1}"

        xlim = self.viewer.view_manager.get_current_bounds().xlim
        with self.viewer.busy_manager.busy_operation("FFT"):
            res = analyze_fft(
                plot.points[:, 0] + plot.offset_x,
                plot.points[:, 1] * plot.y_scale + plot.offset_y,
                xlim,
                self.viewer.sample_rate_hz,
            )
        self._report_fft(name, res)
        self._open_spectrum_window(name, res)

    def _freq_formatter(self, res: FFTResult):
        if res.frequency_unit == "Hz":
            eng = EngFormatter(unit="Hz", places=3)
            return lambda value: eng(value)
        return lambda value: f"{value:.6g} cyc/x"

    def _report_fft(self, name: str, res: FFTResult) -> None:
        spec = res.spectrum
        fmt = self._freq_formatter(res)
        print(f"[INFO] FFT: {name}")
        print(
            f"[INFO]   record:  {res.n_samples:,} samples, "
            f"x {res.x0:.6g} .. {res.x1:.6g}, spacing {res.dx:.6g}"
        )
        if spec.averages > 1:
            print(
                f"[INFO]   fft:     nfft {spec.nfft}, {spec.averages} x {WINDOW} "
                f"power average @ {OVERLAP:.0%} overlap"
            )
        else:
            print(f"[INFO]   fft:     nfft {spec.nfft}, single record, {WINDOW}")
        print(
            f"[INFO]   rbw:     {fmt(spec.binwidth)}"
            f"  (enbw {fmt(spec.enbw_hz)})"
        )
        print(
            f"[INFO]   floor:   median {res.floor.median_db:+.1f} dB, "
            f"asd {res.floor.asd_median:.3g} y/\u221a{res.frequency_unit}, "
            f"band rms {res.floor.rms:.6g}"
        )
        if not res.peaks:
            print("[INFO]   peaks:   none above the floor + 10 dB threshold")
        for rank, peak in enumerate(res.peaks, start=1):
            print(
                f"[INFO]   peak {rank}:  {fmt(peak.frequency)}, "
                f"{peak.db:+.1f} dB (amp {peak.amplitude:.6g}, "
                f"snr {peak.snr_db:.1f} dB)"
            )

    def on_peaks_toggled(self, enabled: bool) -> None:
        """Label the analyzed peaks of this spectrum with their frequency."""
        if self._fft_source is None or not self._fft_source.peaks:
            print(
                "[INFO] peaks: no spectrum peak data in this window; "
                "the FFT button opens a spectrum window that carries it"
            )
            self.viewer.control_bar_manager.set_peaks_checked(False)
            return
        if self._peak_artists is None:
            self._peak_artists = FFTPeakArtists(self.viewer.ax)
        if enabled:
            self._peak_artists.draw(
                self._fft_source.peaks, self._freq_formatter(self._fft_source)
            )
        else:
            self._peak_artists.clear()
        self.viewer.canvas.draw_idle()

    def attach_spectrum(self, res: FFTResult, plot_index: int) -> None:
        """Carry the analysis in this viewer; label peaks with the box checked."""
        self._fft_source = res
        self._fft_plot_index = plot_index
        if res.peaks:
            self.viewer.control_bar_manager.set_peaks_checked(True)
            self.on_peaks_toggled(True)

    def _rescale_attached_spectrum(self) -> None:
        """
        Re-express the carried spectrum at the current sample rate.

        Amplitudes are rate-independent; only the frequency axis and the
        per-root-hertz density scale, so the displayed spectrum, the field
        arrays, the peaks, the view, and the labels are rebuilt in place
        rather than recomputed from a record this window does not hold.
        """
        res = self._fft_source
        if res is None:
            return
        rate = self.viewer.sample_rate_hz
        new_fs = rate / res.dx if rate else 1.0 / res.dx
        factor = new_fs / res.spectrum.samplerate
        if factor == 1.0:
            return

        unit = "Hz" if rate else "cyc/x"
        res = replace(
            res,
            spectrum=replace(res.spectrum, samplerate=new_fs),
            peaks=tuple(
                replace(peak, frequency=peak.frequency * factor)
                for peak in res.peaks
            ),
            floor=replace(
                res.floor, asd_median=res.floor.asd_median / math.sqrt(factor)
            ),
            samplerate=new_fs,
            frequency_unit=unit,
        )
        self._fft_source = res
        spec = res.spectrum

        plot = self.viewer.plot_manager.plots[self._fft_plot_index]
        plot.points = np.column_stack((spec.frequencies, spec.db)).astype(
            np.float32
        )
        array_index = self.viewer.array_field_integration.array_index_for_plot(
            self._fft_plot_index
        )
        data = self.viewer.array_field_integration.array_field_manager.get_array_info(
            array_index
        )["data"]
        data["frequency"] = spec.frequencies
        data["asd"] = spec.asd

        color = "white" if self.viewer.dark_mode else "black"
        self.viewer.ax.set_xlabel(f"frequency [{unit}]", color=color)

        bounds = self.viewer.view_manager.get_current_bounds()
        self.viewer.set_view(
            (bounds.xlim[0] * factor, bounds.xlim[1] * factor), bounds.ylim
        )

        if self._peak_artists is not None:
            self._peak_artists.clear()
        peaks_chk = self.viewer.control_bar_manager.get_widget("peaks_chk")
        if res.peaks and peaks_chk.isChecked():
            self.on_peaks_toggled(True)

        fmt = self._freq_formatter(res)
        print(f"[INFO] frequency axis rescaled to {unit}")
        for rank, peak in enumerate(res.peaks, start=1):
            print(f"[INFO]   peak {rank}:  {fmt(peak.frequency)}, {peak.db:+.1f} dB")
        self.viewer.canvas.draw_idle()

    def on_pixels(self) -> None:
        """Dwell-domain analysis of the selected plot's source array."""
        try:
            self._run_pixels()
        except PixelAnalysisError as exc:
            print(f"[INFO] {exc}")

    def _run_pixels(self) -> None:
        pm = self.viewer.plot_manager
        if not pm.plots:
            raise PixelAnalysisError("pixel analysis: no plots loaded")
        selected = pm.get_selected_plots()
        if len(selected) != 1:
            raise PixelAnalysisError(
                "pixel analysis: select a single plot in the Plot/Group dropdown"
            )
        plot_index = selected[0]

        integration = self.viewer.array_field_integration
        mapping = integration.array_field_manager.plot_to_array_field.get(plot_index)
        if mapping is None:
            raise PixelAnalysisError(
                "pixel analysis: this plot has no source array, so it carries no "
                "pixel or frame fields"
            )
        array_index, value_field = mapping
        info = integration.array_field_manager.get_array_info(array_index)
        data = info["data"]

        with self.viewer.busy_manager.busy_operation("Pixel analysis"):
            report = analyse_pixels(data, value_field=value_field)

        volts = None
        y_mgr = self.viewer.view_manager.secondary_axis_manager.y_axis_manager
        if y_mgr.is_enabled() and y_mgr.config is not None:
            volts = abs(y_mgr.config.scale)
        for line in format_report(
            report,
            sample_rate_hz=self.viewer.sample_rate_hz,
            volts_per_code=volts,
        ).splitlines():
            print(f"[INFO] {line}")

        self._open_profile_window(report, volts)

    def _open_profile_window(self, report, volts_per_code: float | None) -> None:
        from .Plot2D import Plot2D   # deferred: Plot2D imports this module

        p = report.profile
        scale = volts_per_code if volts_per_code else 1.0
        n = len(p.residual)
        arr = np.zeros(
            n,
            dtype=[
                ("index", np.float64),
                ("residual", np.float64),
                ("transient", np.float64),
                ("lag_gain_pct", np.float64),
            ],
        )
        arr["index"] = np.arange(n)
        arr["residual"] = p.residual * scale
        arr["transient"] = p.transient * scale
        gains = np.full(n, np.nan)
        gains[: len(report.crosstalk.start)] = report.crosstalk.lag_gain * 100.0
        arr["lag_gain_pct"] = gains

        window = Plot2D(
            auto_aspect=True,
            dark_mode=self.viewer.dark_mode,
            embedded=True,
        )
        window.add_plot(
            arr,
            x_field="index",
            y_field="residual",
            draw_lines=True,
            plot_name=f"{report.value_field} settling profile",
        )
        window.setWindowTitle(
            f"Pixels: {report.value_field}  "
            f"start {report.recommended_start} length {report.recommended_length}"
        )
        color = "white" if window.dark_mode else "black"
        window.ax.set_xlabel("index within dwell", color=color)
        unit = "V" if volts_per_code else "codes"
        window.ax.set_ylabel(f"median |residual| [{unit}]", color=color)
        window.control_bar_manager.set_sample_rate_display(self.viewer.sample_rate_hz)
        window.sample_rate_hz = self.viewer.sample_rate_hz
        window.show()
        window.raise_()
        window.activateWindow()
        self._fft_windows.append(window)
        print(
            f"[INFO] Profile window opened: Pixels: {report.value_field} "
            f"({n} indices)"
        )

    def _open_spectrum_window(self, name: str, res: FFTResult) -> None:
        from .Plot2D import Plot2D   # deferred: Plot2D imports this module

        spec = res.spectrum
        arr = np.zeros(
            len(spec),
            dtype=[
                ("frequency", np.float64),
                ("db", np.float64),
                ("amplitude", np.float64),
                ("asd", np.float64),
            ],
        )
        arr["frequency"] = spec.frequencies
        arr["db"] = spec.db
        arr["amplitude"] = spec.amplitude
        arr["asd"] = spec.asd

        window = Plot2D(
            auto_aspect=True,
            dark_mode=self.viewer.dark_mode,
            embedded=True,
        )
        window.add_plot(
            arr,
            x_field="frequency",
            y_field="db",
            draw_lines=True,
            point_size=0.5,
            line_width=1.0,
            plot_name=f"{name} fft",
        )
        window.setWindowTitle(f"FFT: {name}")
        color = "white" if window.dark_mode else "black"
        window.ax.set_xlabel(f"frequency [{res.frequency_unit}]", color=color)
        window.ax.set_ylabel("amplitude [dB]", color=color)
        window.event_handlers.attach_spectrum(
            res, window.plot_manager.get_plot_count() - 1
        )
        window.control_bar_manager.set_sample_rate_display(
            self.viewer.sample_rate_hz
        )
        window.sample_rate_hz = self.viewer.sample_rate_hz
        window.show()
        window.raise_()
        window.activateWindow()
        self._fft_windows.append(window)
        print(f"[INFO] Spectrum window opened: FFT: {name} ({len(spec):,} bins)")

    def on_grid_changed(self, grid_text: str):
        with self.viewer.busy_manager.busy_operation("Updating grid"):
            if "^" in grid_text:
                power = int(grid_text.split("^")[1].split()[0])
                self.viewer.grid_manager.set_grid_spacing(power, True)
            else:
                self.viewer.grid_manager.set_grid_spacing(0, False)

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def on_pick_axes_grid_color(self):
        new_hex = self.viewer._pick_color(self.viewer.axes_grid_color)
        if new_hex:
            self.viewer.axes_grid_color = new_hex
            self.viewer.grid_manager.set_grid_colors(
                self.viewer.grid_color, self.viewer.axes_grid_color
            )
            self.viewer.control_bar_manager.set_axes_grid_color_swatch(new_hex)
            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def on_pick_grid2n_color(self):
        new_hex = self.viewer._pick_color(self.viewer.grid_color)
        if new_hex:
            self.viewer.grid_color = new_hex
            self.viewer.grid_manager.set_grid_colors(
                self.viewer.grid_color, self.viewer.axes_grid_color
            )
            self.viewer.control_bar_manager.set_adc_grid_color_swatch(new_hex)
            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def fit_view_to_data(self):
        with self.viewer.busy_manager.busy_operation("Fitting view to data"):
            bounds = self.viewer.fit_view()
            print(
                f"[INFO] Fit view to data bounds: X({bounds.xlim[0]:.3f}, {bounds.xlim[1]:.3f}), Y({bounds.ylim[0]:.3f}, {bounds.ylim[1]:.3f})"
            )

    def on_mouse_mode_changed(self, mode_name: str) -> None:
        self.viewer.set_mouse_mode(MouseMode[mode_name])

    def view_back(self) -> None:
        self.viewer.view_back()

    def view_forward(self) -> None:
        self.viewer.view_forward()

    def reset_view(self) -> None:
        with self.viewer.busy_manager.busy_operation("Resetting view"):
            self.viewer.fit_view()

    def apply_view_bounds(self):
        """Apply custom view bounds from the text fields."""
        with self.viewer.busy_manager.busy_operation("Applying view bounds"):
            xmin, xmax, ymin, ymax = self.viewer.control_bar_manager.get_view_bounds()

            is_valid, error_msg, bounds = self.viewer.view_manager.validate_bounds(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )

            if not is_valid:
                print(f"[ERROR] {error_msg}")
                return

            self.viewer.set_view(bounds.xlim, bounds.ylim)
            print(
                f"[INFO] Applied custom view bounds: X({bounds.xlim[0]:.3f}, {bounds.xlim[1]:.3f}), Y({bounds.ylim[0]:.3f}, {bounds.ylim[1]:.3f})"
            )

    def immediate_exit(self) -> None:
        print("[INFO] Exit button pressed, closing viewer.")
        self.viewer.close()

    def apply_offset_values(self):
        """Apply offset values from spinboxes to the selected plot(s) or group."""
        with self.viewer.busy_manager.busy_operation("Applying plot offset"):
            x_offset, y_offset = self.viewer.control_bar_manager.get_offset_values()

            if self.viewer.plot_manager.is_group_selected():
                group_id = self.viewer.plot_manager.selected_group_id
                self.viewer.plot_manager.set_group_property(group_id, "offset_x", x_offset)
                self.viewer.plot_manager.set_group_property(group_id, "offset_y", y_offset)

                group_info = self.viewer.plot_manager.get_group_info(group_id)
                if group_info:
                    print(
                        f"[INFO] Applied offset to group '{group_info.group_name}': ({x_offset:.3f}, {y_offset:.3f})"
                    )
            else:
                plot_index = self.viewer.plot_manager.selected_plot_index
                self.viewer.plot_manager.set_plot_property(plot_index, "offset_x", x_offset)
                self.viewer.plot_manager.set_plot_property(plot_index, "offset_y", y_offset)

                plot_info = self.viewer.plot_manager.get_plot_info(plot_index)
                if plot_info:
                    print(
                        f"[INFO] Applied offset to plot {plot_index + 1}: ({x_offset:.3f}, {y_offset:.3f})"
                    )

    def on_color_field_changed(self, field_name: str) -> None:
        """Switch the color field for the selected plot(s) or group."""
        with self.viewer.busy_manager.busy_operation(
            f"Changing color field to {field_name}"
        ):
            if self.viewer.plot_manager.is_group_selected():
                group_id = self.viewer.plot_manager.selected_group_id
                group_info = self.viewer.plot_manager.get_group_info(group_id)
                if not group_info:
                    return
                plot_indices = group_info.plot_indices
            else:
                plot_indices = [self.viewer.plot_manager.selected_plot_index]

            # group selection: shared color range across all member plots
            global_color_min = None
            global_color_max = None

            if self.viewer.plot_manager.is_group_selected():
                for plot_index in plot_indices:
                    array_index = (
                        self.viewer.array_field_integration.array_index_for_plot(
                            plot_index
                        )
                    )
                    if array_index is None:
                        continue

                    array_info = self.viewer.array_field_integration.array_field_manager.get_array_info(
                        array_index
                    )
                    if not array_info:
                        continue

                    data = array_info["data"]
                    if field_name not in data.dtype.names:
                        continue

                    field_data = data[field_name].astype(np.float32)
                    if len(field_data) > 0:
                        local_min = float(field_data.min())
                        local_max = float(field_data.max())

                        if global_color_min is None:
                            global_color_min = local_min
                            global_color_max = local_max
                        else:
                            global_color_min = min(global_color_min, local_min)
                            global_color_max = max(global_color_max, local_max)

            for plot_index in plot_indices:
                array_index = (
                    self.viewer.array_field_integration.array_index_for_plot(
                        plot_index
                    )
                )
                if array_index is None:
                    print(f"[WARNING] Could not find array for plot {plot_index}")
                    continue

                array_info = self.viewer.array_field_integration.array_field_manager.get_array_info(
                    array_index
                )
                if not array_info:
                    print(f"[WARNING] Could not get array info for array {array_index}")
                    continue

                data = array_info["data"]
                if field_name not in data.dtype.names:
                    print(
                        f"[ERROR] Field '{field_name}' not found in array {array_index}"
                    )
                    continue

                plot = self.viewer.plot_manager.plots[plot_index]
                plot.color_data = data[field_name].astype(np.float32)
                array_info["properties"]["color_field"] = field_name

                if global_color_min is not None and global_color_max is not None:
                    self.viewer.plot_manager.plot_global_color_ranges[plot_index] = (
                        global_color_min,
                        global_color_max,
                    )
                    array_info["properties"]["global_color_min"] = global_color_min
                    array_info["properties"]["global_color_max"] = global_color_max
                else:
                    self.viewer.plot_manager.plot_global_color_ranges.pop(
                        plot_index, None
                    )
                    array_info["properties"].pop("global_color_min", None)
                    array_info["properties"].pop("global_color_max", None)

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

            print(
                f"[INFO] Color field changed to '{field_name}' for {len(plot_indices)} plot(s)"
            )
