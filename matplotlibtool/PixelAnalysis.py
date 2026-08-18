#!/usr/bin/env python3
# tab-width:4

"""
PixelAnalysis - dwell-domain measurements on a captured detector readout.

Operates on a structured array carrying pixel, frame and one value field. A
dwell is a maximal run of records sharing a pixel index; the hardware supplies
that index, so dwell boundaries are ground truth rather than something to
detect.

Four independent measurements, none of which depends on the others:

  geometry     how long the dwells are and which ones are usable
  profile      settling within a dwell, as median residual against index
  crosstalk    how much of the previous pixel leaks into this one, per start
  frames       repeat measurements of the same pixel, split into the part that
               advances every frame and the part that does not

The window choice follows from crosstalk, not from noise: the residual per
sample sets how much averaging helps, while the leak from the preceding pixel
sets where averaging is allowed to begin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SETTLED_TAIL_GUARD = 2      # trailing records excluded from the settled reference
SETTLED_TAIL_SPAN = 8       # records forming the settled reference
SETTLE_FLOOR_MULT = 0.5     # settled once the fitted decay is this fraction of the floor
FIT_FLOOR_MULT = 3.0        # profile fit stops this far above the noise floor
FIT_LO = 2                  # index 0 and 1 straddle the pixel clock edge
MIN_WINDOW = 4              # records kept when the dwell is too short to settle
MIN_GROUPS = 32
MIN_FIT_POINTS = 5

# How the window was found. A well formed capture always yields a window; the
# verdict says whether the decay fit produced it (measured), the profile never
# stood clear of the noise floor so every start is equally settled (flat), the
# residual never decays and the window is the latest the dwell can hold
# (no decay), or the dwell has no room for a settled reference at all and the
# window is the latest records it can hold (short).
Verdict = Literal["measured", "flat", "no decay", "short"]


class PixelAnalysisError(ValueError):
    """The capture cannot support a dwell-domain analysis."""


@dataclass(frozen=True)
class DwellGeometry:
    modal_length: int
    n_groups: int
    n_analysed: int
    length_counts: dict[int, int]
    excluded_long: int          # dwells far longer than modal, e.g. idle pixel 0
    excluded_short: int


@dataclass(frozen=True)
class SettleProfile:
    residual: np.ndarray        # median |value - settled| per within-dwell index
    transient: np.ndarray       # signed mean per index; pixel steps cancel out
    floor: float                # per-sample sigma implied by the profile floor


@dataclass(frozen=True)
class SettleFit:
    """The log-residual decay fit; present only when the profile resolves one."""

    tau: float                  # samples, from the log-residual slope
    fit_lo: int
    fit_hi: int
    fit_slope: float            # decades per sample
    fit_intercept: float
    fit_rms: float


@dataclass(frozen=True)
class CrosstalkCurve:
    """
    Two separate costs of starting the average early.

    lag_gain is the part proportional to the preceding pixel step: dynamic
    crosstalk, which smears the image along the readout direction.

    bias_sigma is the spread of the remaining offset across pixels. Its mean is
    a constant an offset calibration removes; its spread is not, and appears as
    fixed pattern. Both terms are diagnostics, not the window criterion: once
    the candidate window overlaps the settled reference the two share samples
    and both collapse toward zero for reasons that have nothing to do with
    settling, so they are reported as nan there rather than as a small number.
    """

    start: np.ndarray           # candidate first averaged index
    lag_gain: np.ndarray        # fraction of the preceding pixel step
    mean_bias: np.ndarray       # step-independent offset, in value units
    bias_sigma: np.ndarray      # spread of that offset across pixels


@dataclass(frozen=True)
class FrameDecomposition:
    """
    Repeat measurements of one pixel across frames, split by how they behave.

    A value that advances by the same amount every frame is integrating: a
    continuously integrating CTIA with no reset accumulates its own leakage, so
    each pixel ramps at its own rate. A value that changes independently every
    frame is noise. The two are separable because a ramp makes consecutive
    frame-to-frame differences agree while noise makes them anticorrelate.
    """

    frames: tuple[int, ...]
    n_pixels: int
    ramp_sigma: float           # per-frame advance, spread across pixels
    ramp_mean: float
    random_sigma: float         # per-measurement, does not reduce with averaging
    neighbour_corr: float       # spatial correlation of one frame's residual
    monotonic_fraction: float


@dataclass(frozen=True)
class PixelReport:
    value_field: str
    geometry: DwellGeometry
    profile: SettleProfile | None   # None when the dwell is too short to hold
    fit: SettleFit | None           # a settled reference
    crosstalk: CrosstalkCurve | None
    frames: FrameDecomposition | None
    recommended_start: int
    recommended_length: int
    settled: bool               # False when the window is the latest the dwell
                                # can hold rather than a settled one
    verdict: Verdict


def segment_dwells(pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end index of every maximal run of equal pixel."""
    change = np.flatnonzero(np.diff(pixel) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(pixel)]))
    return starts, ends


def measure_dwells(
    pixel: np.ndarray,
    *,
    idle_pixel: int = 0,
) -> tuple[DwellGeometry, np.ndarray, np.ndarray, np.ndarray]:
    """Dwell geometry plus (starts, ends, usable) for every dwell."""
    starts, ends = segment_dwells(pixel)
    lengths = ends - starts
    px = pixel[starts]
    counts = dict(zip(*np.unique(lengths[px != idle_pixel], return_counts=True)))
    if not counts:
        raise PixelAnalysisError("pixel analysis: no dwells outside the idle pixel")
    counts = {int(k): int(v) for k, v in counts.items()}
    modal = max(counts, key=counts.__getitem__)

    # a dwell one record long either way is the sample clock beating against the
    # pixel clock; anything further off is not a readout dwell
    usable = (px != idle_pixel) & (lengths >= modal) & (lengths <= modal + 1)
    if usable.sum() < MIN_GROUPS:
        raise PixelAnalysisError(
            f"pixel analysis: {usable.sum()} usable dwells of {modal} records, "
            f"need at least {MIN_GROUPS}"
        )
    return (
        DwellGeometry(
            modal_length=modal,
            n_groups=int(len(starts)),
            n_analysed=int(usable.sum()),
            length_counts=counts,
            excluded_long=int(((px != idle_pixel) & (lengths > modal + 1)).sum()),
            excluded_short=int(((px != idle_pixel) & (lengths < modal)).sum()),
        ),
        starts,
        ends,
        usable,
    )


def _stack(value: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    offsets = np.arange(length)
    return value[starts[:, None] + offsets[None, :]]


def _settled(matrix: np.ndarray) -> np.ndarray:
    hi = matrix.shape[1] - SETTLED_TAIL_GUARD
    lo = hi - SETTLED_TAIL_SPAN
    if lo < 1:
        raise PixelAnalysisError(
            f"pixel analysis: dwell of {matrix.shape[1]} records is too short to "
            f"hold a settled reference"
        )
    return matrix[:, lo:hi].mean(axis=1)


def _profile(
    matrix: np.ndarray,
    settled: np.ndarray,
) -> tuple[SettleProfile, Verdict, SettleFit | None, float]:
    """Settling profile plus the verdict of the decay fit over it."""
    deviation = matrix - settled[:, None]
    residual = np.median(np.abs(deviation), axis=0)
    transient = deviation.mean(axis=0)

    tail = residual[-(SETTLED_TAIL_GUARD + SETTLED_TAIL_SPAN) : -SETTLED_TAIL_GUARD]
    flat = float(np.median(tail))
    profile = SettleProfile(
        residual=residual,
        transient=transient,
        floor=flat / 0.6745,                    # median |x| of a normal -> sigma
    )
    if flat <= 0.0:
        # identically zero residual: constant data, settled everywhere
        return profile, "flat", None, flat

    above = np.flatnonzero(residual > FIT_FLOOR_MULT * flat)
    fit_hi = int(above[-1]) if above.size else FIT_LO
    if fit_hi - FIT_LO + 1 < MIN_FIT_POINTS:
        # the profile never stands clear of the floor: settling is faster than
        # the dwell resolves, so every start is equally settled
        return profile, "flat", None, flat

    k = np.arange(FIT_LO, fit_hi + 1)
    slope, intercept = np.polyfit(k, np.log10(residual[k]), 1)
    if not slope < 0.0:                         # also catches a degenerate nan fit
        # rising residual: it never settles within a dwell
        return profile, "no decay", None, flat
    rms = float(np.sqrt(np.mean((np.log10(residual[k]) - (slope * k + intercept)) ** 2)))
    fit = SettleFit(
        tau=float(np.log10(np.e) / -slope),
        fit_lo=FIT_LO,
        fit_hi=fit_hi,
        fit_slope=float(slope),
        fit_intercept=float(intercept),
        fit_rms=rms,
    )
    return profile, "measured", fit, flat


def _crosstalk(
    matrix: np.ndarray,
    settled: np.ndarray,
    pixel_of_group: np.ndarray,
) -> CrosstalkCurve:
    step = np.empty_like(settled)
    step[0] = 0.0
    step[1:] = settled[1:] - settled[:-1]
    end = matrix.shape[1] - SETTLED_TAIL_GUARD

    order = np.argsort(pixel_of_group, kind="stable")
    bounds = np.flatnonzero(np.diff(pixel_of_group[order]) != 0) + 1
    per_pixel = np.split(order, bounds)

    settled_lo = end - SETTLED_TAIL_SPAN
    starts = np.arange(0, end - 1)
    gains = np.full(len(starts), np.nan)
    biases = np.full(len(starts), np.nan)
    sigmas = np.full(len(starts), np.nan)
    for i, st in enumerate(starts):
        if st >= settled_lo:
            continue
        error = matrix[:, st:end].mean(axis=1) - settled
        if step.min() == step.max():
            # every pixel stepped by the same amount, so the crosstalk slope
            # has nothing to lever against and the fit is singular
            raise PixelAnalysisError(
                "crosstalk: pixel to pixel steps are all equal, so there is no "
                "gradient to fit"
            )
        gains[i], biases[i] = np.polyfit(step, error, 1)
        # averaging each pixel over its frames leaves the part that repeats
        sigmas[i] = float(np.std([error[g].mean() for g in per_pixel]))
    return CrosstalkCurve(
        start=starts, lag_gain=gains, mean_bias=biases, bias_sigma=sigmas
    )


def _frames(
    matrix: np.ndarray,
    pixel_of_group: np.ndarray,
    frame_of_group: np.ndarray,
    start: int,
    end: int,
    drop_first_frame: bool,
) -> FrameDecomposition | None:
    frames = np.unique(frame_of_group)
    if drop_first_frame and len(frames) > 1:
        # the frame following a global reset has not reached its working point
        frames = frames[1:]
    if len(frames) < 3:
        return None

    per_frame = {int(f): {} for f in frames}
    for j, (p, f) in enumerate(zip(pixel_of_group, frame_of_group)):
        if int(f) in per_frame:
            per_frame[int(f)][int(p)] = j
    common = sorted(set.intersection(*(set(v) for v in per_frame.values())))
    if len(common) < MIN_GROUPS:
        return None

    values = np.stack(
        [
            np.array([matrix[per_frame[int(f)][p], start:end].mean() for p in common])
            for f in frames
        ]
    )
    delta = np.diff(values, axis=0)
    var_delta = float(delta.var())
    cov = float(
        np.mean([np.cov(delta[i], delta[i + 1])[0, 1] for i in range(len(delta) - 1)])
    )
    # delta = ramp + n[k+1] - n[k]:  var = var_ramp + 2 var_n,  cov = var_ramp - var_n
    var_ramp = max((var_delta + 2 * cov) / 3.0, 0.0)
    var_noise = max((var_delta - cov) / 3.0, 0.0)

    residual = values[0] - values[0].mean()
    return FrameDecomposition(
        frames=tuple(int(f) for f in frames),
        n_pixels=len(common),
        ramp_sigma=float(np.sqrt(var_ramp)),
        ramp_mean=float(delta.mean()),
        random_sigma=float(np.sqrt(var_noise)),
        neighbour_corr=float(np.corrcoef(residual[:-1], residual[1:])[0, 1]),
        monotonic_fraction=float(np.mean(np.sign(delta[:-1]) == np.sign(delta[1:])))
        if len(delta) > 1
        else float("nan"),
    )


def analyse_pixels(
    data: np.ndarray,
    *,
    value_field: str = "in0",
    pixel_field: str = "pixel",
    frame_field: str = "frame",
    idle_pixel: int = 0,
    settle_floor_mult: float = SETTLE_FLOOR_MULT,
    drop_first_frame: bool = True,
) -> PixelReport:
    """Measure dwell geometry, settling, crosstalk and frame repeatability."""
    for field in (value_field, pixel_field):
        if field not in data.dtype.names:
            raise PixelAnalysisError(f"pixel analysis: array has no {field!r} field")

    pixel = data[pixel_field]
    geometry, starts, ends, usable = measure_dwells(pixel, idle_pixel=idle_pixel)

    value = data[value_field].astype(np.float64)
    group_starts = starts[usable]
    matrix = _stack(value, group_starts, geometry.modal_length)

    # A well formed capture always yields a window: the operator captured it
    # and wants a number out of it, not a refusal. The verdict says how the
    # window was found.
    if geometry.modal_length <= SETTLED_TAIL_GUARD + SETTLED_TAIL_SPAN + MIN_WINDOW:
        # No room for a settled reference, so nothing about settling is
        # measurable. Average the latest records the dwell can hold, keeping
        # the tail guard while any sample remains before it, down to a single
        # sample.
        profile, fit, crosstalk = None, None, None
        verdict: Verdict = "short"
        settled = False
        usable_end = max(geometry.modal_length - SETTLED_TAIL_GUARD, 1)
        recommended_length = min(MIN_WINDOW, usable_end)
        recommended_start = usable_end - recommended_length
        return _finish_report(
            data, value_field, frame_field, drop_first_frame,
            geometry, matrix, pixel, group_starts,
            profile, fit, crosstalk,
            recommended_start, recommended_length, settled, verdict,
        )

    settled_values = _settled(matrix)

    profile, verdict, fit, flat = _profile(matrix, settled_values)
    crosstalk = _crosstalk(matrix, settled_values, pixel[group_starts])

    latest = geometry.modal_length - SETTLED_TAIL_GUARD - MIN_WINDOW
    if verdict == "measured":
        # settled where the fitted decay reaches a fraction of the noise floor.
        # Extrapolating the fit rather than reading the first index under a
        # threshold matters: the profile approaches its floor asymptotically,
        # so a crossing index there is decided by noise on an almost flat curve
        # and moves by several samples between subsets of the same capture,
        # while the fit uses every point of the decay and does not.
        crossing = (
            np.log10(flat * settle_floor_mult) - fit.fit_intercept
        ) / fit.fit_slope
        if not np.isfinite(crossing):
            verdict, fit = "no decay", None
    if verdict == "measured":
        # a crossing before FIT_LO settled inside the clock-edge straddle; the
        # cap at the dwell length only guards the ceil, the latest-window clamp
        # below decides the unsettled case
        crossing = min(max(crossing, float(FIT_LO)), float(geometry.modal_length))
        recommended_start = int(np.ceil(crossing))
        settled = recommended_start <= latest
        if not settled:
            recommended_start = latest
    elif verdict == "flat":
        recommended_start = FIT_LO
        settled = True
    else:
        recommended_start = latest
        settled = False
    recommended_length = geometry.modal_length - SETTLED_TAIL_GUARD - recommended_start

    return _finish_report(
        data, value_field, frame_field, drop_first_frame,
        geometry, matrix, pixel, group_starts,
        profile, fit, crosstalk,
        recommended_start, recommended_length, settled, verdict,
    )


def _finish_report(
    data: np.ndarray,
    value_field: str,
    frame_field: str,
    drop_first_frame: bool,
    geometry: DwellGeometry,
    matrix: np.ndarray,
    pixel: np.ndarray,
    group_starts: np.ndarray,
    profile: SettleProfile | None,
    fit: SettleFit | None,
    crosstalk: CrosstalkCurve | None,
    recommended_start: int,
    recommended_length: int,
    settled: bool,
    verdict: Verdict,
) -> PixelReport:
    frames = None
    if frame_field in data.dtype.names:
        frames = _frames(
            matrix,
            pixel[group_starts],
            data[frame_field][group_starts],
            recommended_start,
            recommended_start + recommended_length,
            drop_first_frame,
        )

    return PixelReport(
        value_field=value_field,
        geometry=geometry,
        profile=profile,
        fit=fit,
        crosstalk=crosstalk,
        frames=frames,
        recommended_start=recommended_start,
        recommended_length=recommended_length,
        settled=settled,
        verdict=verdict,
    )


def format_report(
    report: PixelReport,
    *,
    sample_rate_hz: float | None = None,
    volts_per_code: float | None = None,
) -> str:
    """Render a report as text. Units are added where the scaling is known."""

    def t(samples: float) -> str:
        if sample_rate_hz:
            return f"{samples:.3f} samples ({samples / sample_rate_hz * 1e6:.3f} us)"
        return f"{samples:.3f} samples"

    def v(codes: float) -> str:
        if volts_per_code:
            return f"{codes * volts_per_code * 1e6:,.0f} uV"
        return f"{codes:,.0f} codes"

    g, p, c = report.geometry, report.profile, report.crosstalk
    out = [f"pixel analysis: {report.value_field}"]
    out.append(
        f"  dwells:   {g.n_analysed:,} of {g.n_groups:,} usable, modal length "
        f"{g.modal_length} records"
        + (f" = {t(g.modal_length)}" if sample_rate_hz else "")
    )
    out.append(
        f"            lengths {dict(sorted(g.length_counts.items())[:4])}, "
        f"excluded {g.excluded_long} long / {g.excluded_short} short"
    )
    if report.fit is not None:
        out.append(
            f"  settling: tau {t(report.fit.tau)} from indices "
            f"{report.fit.fit_lo}..{report.fit.fit_hi}, "
            f"rms {report.fit.fit_rms:.4f} dec"
        )
    elif report.verdict == "short":
        out.append(
            "  settling: the dwell is too short to hold a settled reference; "
            "nothing about settling is measurable"
        )
    elif report.verdict == "flat":
        out.append(
            "  settling: faster than the dwell resolves; the profile never "
            "stands clear of the noise floor"
        )
    else:
        out.append("  settling: the residual does not decay within a dwell")
    if p is not None:
        out.append(f"            per-sample noise {v(p.floor)} rms")
    if c is not None:
        out.append("  cost of starting the average at index k (nan once k reaches")
        out.append("  the settled reference, where the two share samples):")
        for st in range(0, len(c.start), 4):
            out.append(
                f"            start {c.start[st]:>2}: leak {c.lag_gain[st] * 100:>+8.4f}%   "
                f"bias {v(c.mean_bias[st]):>13}   fixed pattern "
                f"{v(c.bias_sigma[st]):>10}"
            )
        out.append(
            f"  window:   start {report.recommended_start}, length "
            f"{report.recommended_length} "
            f"(leak {c.lag_gain[report.recommended_start] * 100:+.4f}%, "
            f"fixed pattern {v(c.bias_sigma[report.recommended_start])})"
        )
    else:
        out.append(
            f"  window:   start {report.recommended_start}, length "
            f"{report.recommended_length}"
        )
    if report.verdict == "short":
        out.append(
            "            NOT settled: the window is the latest the dwell can "
            "hold; lengthen the dwell to measure settling"
        )
    elif report.verdict == "no decay":
        out.append(
            "            NOT settled: the residual never decays; this window "
            "is the latest the dwell can hold"
        )
    elif not report.settled:
        out.append(
            "            NOT settled: the dwell ends before the residual reaches "
            "the noise floor; this window is the latest the dwell can hold"
        )

    f = report.frames
    if f is None:
        out.append("  frames:   fewer than 3 settled frames, repeatability not measured")
        return "\n".join(out)

    out.append(f"  frames:   {f.frames} over {f.n_pixels:,} shared pixels")
    out.append(
        f"            per-pixel ramp {v(f.ramp_sigma)}/frame spread, "
        f"mean {v(f.ramp_mean)}/frame"
    )
    out.append(f"            random per measurement {v(f.random_sigma)} rms")
    out.append(
        f"            neighbour correlation {f.neighbour_corr:+.3f}, "
        f"monotonic {f.monotonic_fraction * 100:.0f}%"
    )
    if p is not None:
        averaged = p.floor / np.sqrt(max(report.recommended_length, 1))
        out.append(
            f"            averaging {report.recommended_length} samples puts the ADC "
            f"contribution at {v(averaged)}, which is "
            f"{f.random_sigma / max(averaged, 1e-12):.0f}x below the random term"
        )
    return "\n".join(out)
