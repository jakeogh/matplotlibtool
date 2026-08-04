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
then samples), otherwise cycles per x-unit. Long records are reduced to an
averaged Blackman-Harris power spectrum sized for a stable noise floor; short
records get a single full-length transform. The DC bin and its window skirt
are removed after mean detrending - they carry no spectral information and
would otherwise own the dB autoscale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyfft import NoiseFloor
from pyfft import Peak
from pyfft import Spectrum
from pyfft import average_spectrum
from pyfft import compute_spectrum
from pyfft import find_peaks
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


def analyze_fft(
    x: np.ndarray,
    y: np.ndarray,
    view_xlim: tuple[float, float],
    sample_rate_hz: float | None,
) -> FFTResult:
    order = np.argsort(x, kind="stable")
    x = np.asarray(x, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float64)[order]

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
    )
