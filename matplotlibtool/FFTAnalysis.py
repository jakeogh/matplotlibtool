#!/usr/bin/env python3
# tab-width:4

"""
FFTAnalysis - spectrum of the in-view samples of one plot, via pyfft.

The view selects the record: whatever x window is on screen is what gets
transformed, so zooming is how a segment is chosen. The x axis only supplies
the sample spacing; it must be uniform, because a DFT of unevenly spaced
samples is not the spectrum of the signal and pretending otherwise would be
worse than refusing.

The frequency axis is in Hz when the viewer's sample rate is set (x-units are
then samples), otherwise cycles per x-unit. The amplitude axis is in the
viewer's y unit when one is set (dBV for volts, with the floor reported as a
density in V/sqrt(Hz)), otherwise in whatever the y values are, typically ADC
codes. Long records are reduced to an averaged Blackman-Harris power spectrum
sized for a stable noise floor; short records get a single full-length
transform. The DC bin and its window skirt are removed after mean detrending
- they carry no spectral information and would otherwise own the dB
autoscale.

Two views come out of one analysis. The amplitude spectrum has one bin width
and reads tones directly in volts peak; its bottom bin is the sample rate
over 2^20, a couple of hertz at 2 MS/s however long the record. The density
is multiresolution: the same full-rate spectrum at the top, then decimated
stages down to a few bins above 1/T, on a log frequency axis in V/sqrt(Hz),
which is the datasheet unit for noise and the only way a 0.1 Hz line and a
1 MHz floor sit on one plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyfft import Density
from pyfft import NoiseFloor
from pyfft import Peak
from pyfft import Spectrum
from pyfft import average_spectrum
from pyfft import compute_spectrum
from pyfft import find_peaks
from pyfft import multiresolution_density
from pyfft import noise_floor


class FFTAnalysisError(ValueError):
    """
    The data cannot support a spectral analysis.

    Distinct from a bug: callers catch this by name to report the reason and
    carry on, while anything else propagates and crashes as it should.
    """


WINDOW = "blackmanharris"
TARGET_SEGMENTS = 8         # below this many nfft spans, one full-length FFT
OVERLAP = 0.75
MIN_SAMPLES = 32
MIN_NFFT = 1024
MAX_NFFT = 1 << 20
SPACING_TOL = 1e-3          # relative deviation of dx that still counts as uniform
REPORT_PEAKS = 8
PEAK_LABEL_COLOR = "#ff8c00"


class FFTPeakArtists:
    """Markers and frequency labels for analyzed peaks, drawn on one axes."""

    def __init__(self, ax):
        self.ax = ax
        self._artists: list = []

    def clear(self) -> None:
        for artist in self._artists:
            artist.remove()
        self._artists = []

    def draw(self, peaks: tuple[Peak, ...], formatter) -> None:
        """`formatter` turns a Peak into its label text."""
        self.clear()
        for peak in peaks:
            self._artists.append(
                self.ax.plot(
                    [peak.frequency],
                    [peak.db],
                    marker="o",
                    markersize=3,
                    color=PEAK_LABEL_COLOR,
                    linestyle="none",
                    zorder=999,
                )[0]
            )
            self._artists.append(
                self.ax.annotate(
                    formatter(peak),
                    (peak.frequency, peak.db),
                    textcoords="offset points",
                    xytext=(6, -2),
                    ha="left",
                    va="top",
                    fontsize=8,
                    color=PEAK_LABEL_COLOR,
                    zorder=1000,
                )
            )


@dataclass(frozen=True)
class BandFloor:
    low: float
    high: float
    asd_median: float           # median amplitude spectral density in the band


@dataclass(frozen=True)
class FFTResult:
    spectrum: Spectrum          # DC bin and skirt removed
    floor: NoiseFloor
    peaks: tuple[Peak, ...]
    n_samples: int
    x0: float
    x1: float
    dx: float                   # sample spacing, x-units
    samplerate: float           # in frequency_unit * 2 Nyquist terms
    frequency_unit: str         # Hz when the viewer sample rate is set, else cyc/x
    y_unit: str                 # unit of the amplitudes, "" when y is unscaled
    density: Density            # multiresolution asd, 1/T-ish to Nyquist
    bands: tuple[BandFloor, ...]    # floor density per decade, from the density

    @property
    def db_unit(self) -> str:
        """Label for the dB axis: dBV for volts, dB re unit otherwise."""
        if self.y_unit == "V":
            return "dBV"
        if self.y_unit:
            return f"dB re {self.y_unit}"
        return "dB"


def decade_floors(density: Density) -> tuple[BandFloor, ...]:
    """Median amplitude spectral density per decade: the shape of a noise
    floor in a few numbers, where a full-band rms hides a 1/f rise or a
    shelf."""
    frequencies = density.frequencies
    top = float(frequencies[-1])
    low = 10.0 ** math.floor(math.log10(float(frequencies[0])))
    bands = []
    while low < top:
        high = low * 10.0
        mask = (frequencies >= low) & (frequencies < min(high, top * (1.0 + 1e-9)))
        if np.count_nonzero(mask) >= 8:
            bands.append(BandFloor(low, min(high, top), float(np.median(density.asd[mask]))))
        low = high
    return tuple(bands)


def analyze_fft(
    x: np.ndarray,
    y: np.ndarray,
    view_xlim: tuple[float, float],
    sample_rate_hz: float | None,
    y_unit_scale: float = 1.0,
    y_unit: str = "",
) -> FFTResult:
    """`y_unit_scale` is physical units per y value (volts per ADC code, for
    a raw capture) and `y_unit` names them; both default to the y values as
    they are."""
    order = np.argsort(x, kind="stable")
    x = np.asarray(x, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float64)[order] * y_unit_scale

    sel = (x >= view_xlim[0]) & (x <= view_xlim[1])
    xs = x[sel]
    ys = y[sel]
    n = xs.size
    if n < MIN_SAMPLES:
        raise FFTAnalysisError(
            f"fft: {n} samples in the x window, need >= {MIN_SAMPLES}"
        )

    dx = np.diff(xs)
    dx_med = float(np.median(dx))
    if dx_med <= 0.0:
        raise FFTAnalysisError("fft: duplicate x values, sample spacing is zero")
    bad = int(np.count_nonzero(np.abs(dx - dx_med) > SPACING_TOL * dx_med))
    if bad:
        raise FFTAnalysisError(
            f"fft: x is not uniformly sampled: {bad} of {dx.size} intervals "
            f"deviate more than {SPACING_TOL:.0%} from the median spacing "
            f"{dx_med:.6g}"
        )

    if sample_rate_hz:
        samplerate = sample_rate_hz / dx_med   # x-units are samples
        frequency_unit = "Hz"
    else:
        samplerate = 1.0 / dx_med
        frequency_unit = "cyc/x"

    if n >= TARGET_SEGMENTS * MIN_NFFT:
        nfft = min(1 << int(math.log2(n // TARGET_SEGMENTS)), MAX_NFFT)
        spec = average_spectrum(
            ys,
            samplerate,
            nfft,
            overlap=OVERLAP,
            window=WINDOW,
            mode="power",
        )
    else:
        spec = compute_spectrum(ys, samplerate, window=WINDOW)

    density = multiresolution_density(
        ys, samplerate, nfft=spec.nfft, window=WINDOW, overlap=OVERLAP,
        first=spec if spec.averages > 1 else None,
    )

    # bin 0 holds only the detrend residual; the skirt above it is window
    # leakage of that residual, not signal
    spec = spec[spec.binwidth * 0.5 :].cut_dc()

    return FFTResult(
        spectrum=spec,
        floor=noise_floor(spec),
        peaks=tuple(find_peaks(spec, count=REPORT_PEAKS)),
        n_samples=n,
        x0=float(xs[0]),
        x1=float(xs[-1]),
        dx=dx_med,
        samplerate=samplerate,
        frequency_unit=frequency_unit,
        y_unit=y_unit,
        density=density,
        bands=decade_floors(density),
    )
