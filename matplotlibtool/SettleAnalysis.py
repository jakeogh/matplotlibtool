#!/usr/bin/env python3
# tab-width:4

"""
SettleAnalysis - segmentation and single-pole fit of a step-settling trace.

The view only selects the event: the largest |dy| edge inside the x window.
Every boundary is then walked outward through the full record to event-defined
limits (previous/next edge-threshold crossing), so results are independent of
zoom level. All thresholds are tied to measured noise statistics, never to
positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter

EDGE_K = 8.0        # edge threshold: median + k * robust sigma of |dy|
EVENT_GAP = 8       # quiet samples required to separate two events
SETTLED_M = 4.0     # settled band: m * baseline noise sigma
FLOOR_C = 8.0       # linear-fit floor: c * baseline noise sigma
TOP_F = 0.9         # linear-fit ceiling: fraction of step height
TRIM_K = 3.0        # end-trim points deviating > k * fit rms
MIN_FIT_POINTS = 6
RISE_TAU_RATIO = float(np.log(9.0))   # t(10-90%) / tau for a single pole

EDGE_COLOR = "#ff8c00"
RISE_COLOR = "#ffd700"
FIT_COLOR = "#00bfff"
SETTLED_COLOR = "#adff2f"
FLOOR_COLOR = "#888888"
POLE_COLOR = "#ff4040"
MIN_BASELINE = 8
MIN_SETTLED = 8


@dataclass(frozen=True)
class SettleSegments:
    y_pre: float
    y_final: float
    step_height: float          # signed, y_final - y_pre
    noise_sigma: float
    baseline_n: int
    baseline_x0: float
    edge_start_x: float
    edge_end_x: float
    linear_start_x: float
    linear_end_x: float
    n_fit_points: int
    rise_10_90: float           # measured 10-90% transition, x-units
    rise_20_80: float           # measured 20-80% transition, x-units
    rise_x10: float             # x of the 10% crossing
    rise_x90: float             # x of the 90% crossing
    slope: float                # decades per x-unit (negative for a decay)
    slope_first_half: float
    slope_second_half: float
    lead_trim_decades: float    # decades of the top of the settle the fit rejected
    tail_trim_decades: float    # decades above the floor the fit rejected
    tau: float                  # e-fold time constant, x-units
    fit_x0: float
    fit_intercept: float        # log10 residual at fit_x0
    fit_rms: float              # rms about the fit, decades
    settled_x: float            # first x of the persistent SETTLED_M-sigma band
    settling_time: float        # settled_x - edge_start_x
    span_x1: float              # end of the analyzed post-step span


def _text_color(ax) -> str:
    """Readable foreground for the axes background, so light mode works too."""
    r, g, b, _ = to_rgba(ax.get_facecolor())
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"


def x_formatter(sample_rate_hz: float | None):
    """Render an x-unit interval, adding seconds when a sample rate is known."""
    if not sample_rate_hz:
        return lambda value: f"{value:.4g} x-units"
    eng = EngFormatter(unit="s", places=3)
    return lambda value: f"{value:.4g} x-units ({eng(value / sample_rate_hz)})"


def _robust_sigma(values: np.ndarray) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad > 0.0:
        return 1.4826 * mad
    return float(values.std())


def _crossing_x(
    x: np.ndarray,
    frac: np.ndarray,
    level: float,
    i_ref: int,
    forward: bool,
) -> float:
    """
    Interpolated x where frac crosses level, searched outward from i_ref.

    Anchoring the search at the 50% index and walking outward makes the result
    immune to noise excursions elsewhere in the window.
    """
    if forward:
        hits = np.flatnonzero(frac[i_ref:] >= level)
        if hits.size == 0:
            raise ValueError(f"settle analysis: transition never reaches {level:.0%}")
        i = int(hits[0]) + i_ref
        lo = i - 1
    else:
        hits = np.flatnonzero(frac[: i_ref + 1] <= level)
        if hits.size == 0:
            raise ValueError(f"settle analysis: transition never starts below {level:.0%}")
        lo = int(hits[-1])
        i = lo + 1

    if lo < 0 or i >= len(frac):
        raise ValueError(
            f"settle analysis: the {level:.0%} crossing is outside the captured edge"
        )
    span = frac[i] - frac[lo]
    if span == 0.0:
        return float(x[i])
    return float(x[lo] + (x[i] - x[lo]) * (level - frac[lo]) / span)


def analyze_settle(
    x: np.ndarray,
    y: np.ndarray,
    view_xlim: tuple[float, float],
) -> SettleSegments:
    order = np.argsort(x, kind="stable")
    x = np.asarray(x, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float64)[order]
    if len(x) < MIN_BASELINE + MIN_SETTLED + MIN_FIT_POINTS:
        raise ValueError(f"settle analysis: only {len(x)} samples in record")

    dy = np.diff(y)
    ady = np.abs(dy)
    d_med = float(np.median(ady))
    d_sigma = _robust_sigma(ady)
    if d_sigma == 0.0:
        raise ValueError("settle analysis: trace is constant, no event to analyze")
    thresh = d_med + EDGE_K * d_sigma

    # dy[i] spans x[i] -> x[i+1]; restrict the event search to the view
    view_d = np.flatnonzero((x[:-1] >= view_xlim[0]) & (x[:-1] <= view_xlim[1]))
    if view_d.size == 0:
        raise ValueError("settle analysis: no samples in the current x window")
    peak = int(view_d[np.argmax(ady[view_d])])
    if ady[peak] <= thresh:
        raise ValueError(
            f"settle analysis: no step in view (max |dy| {ady[peak]:.4g} "
            f"<= threshold {thresh:.4g})"
        )

    # event cluster containing the peak: exceedances merged across gaps
    # shorter than EVENT_GAP, so a settle tail's threshold chatter stays one
    # event and the next cluster is a genuine new step
    idxs = np.flatnonzero(ady > thresh)
    pos = int(np.searchsorted(idxs, peak))
    lo_i = pos
    while lo_i - 1 >= 0 and idxs[lo_i - 1] >= idxs[lo_i] - EVENT_GAP:
        lo_i -= 1
    hi_i = pos
    while hi_i + 1 < len(idxs) and idxs[hi_i + 1] <= idxs[hi_i] + EVENT_GAP:
        hi_i += 1
    edge_start = int(idxs[lo_i])  # last pre-step sample index

    # pre-step baseline: the quiet run back to the previous event cluster
    b0 = int(idxs[lo_i - 1]) + 1 if lo_i > 0 else 0
    baseline = y[b0 : edge_start + 1]
    if baseline.size < MIN_BASELINE:
        raise ValueError(
            f"settle analysis: pre-step baseline has {baseline.size} samples, "
            f"need >= {MIN_BASELINE}"
        )
    y_pre = float(np.median(baseline))
    noise_sigma = _robust_sigma(baseline)
    if noise_sigma == 0.0:
        nz = ady[ady > 0.0]
        noise_sigma = 0.5 * float(nz.min())  # quantization floor

    # post-step span: from the transition to the start of the next event
    # cluster (or end of record)
    next_event = int(idxs[hi_i + 1]) if hi_i + 1 < len(idxs) else len(y) - 1
    post_x = x[peak + 1 : next_event + 1]
    post_y = y[peak + 1 : next_event + 1]
    if post_y.size < MIN_SETTLED + MIN_FIT_POINTS:
        raise ValueError(
            f"settle analysis: only {post_y.size} samples between the edge and "
            f"the next event"
        )

    # converged final value: mean of the trailing run inside the settled band
    y_final = float(np.median(post_y[post_y.size // 2 :]))
    converged = False
    for _ in range(50):
        outside = np.flatnonzero(np.abs(post_y - y_final) > SETTLED_M * noise_sigma)
        start = int(outside[-1]) + 1 if outside.size else 0
        if post_y.size - start < MIN_SETTLED:
            raise ValueError(
                "settle analysis: trace does not settle within the record after "
                "the step; capture more post-step data"
            )
        new = float(post_y[start:].mean())
        if abs(new - y_final) <= 0.05 * noise_sigma:
            y_final = new
            converged = True
            break
        y_final = new
    if not converged:
        raise ValueError("settle analysis: final-value estimate did not converge")

    step = y_final - y_pre
    if abs(step) <= FLOOR_C * noise_sigma:
        raise ValueError(
            f"settle analysis: step height {abs(step):.4g} is inside the noise "
            f"floor ({FLOOR_C} sigma = {FLOOR_C * noise_sigma:.4g})"
        )

    # settled point: first entry into the band that persists, so a rare noise
    # excursion late in a long record cannot inflate the settling time
    inside = (np.abs(post_y - y_final) <= SETTLED_M * noise_sigma).astype(np.int32)
    runs = np.convolve(inside, np.ones(MIN_SETTLED, dtype=np.int32), "valid")
    settled_i = int(np.flatnonzero(runs == MIN_SETTLED)[0])

    residual = np.abs(post_y - y_final)
    below_top = np.flatnonzero(residual < TOP_F * abs(step))
    if below_top.size == 0:
        raise ValueError(
            "settle analysis: residual never drops below the fit ceiling"
        )
    edge_end_x = float(post_x[below_top[0]])  # transition over, exponential begins

    cand = np.flatnonzero(
        (residual > FLOOR_C * noise_sigma) & (residual < TOP_F * abs(step))
    )
    if cand.size < MIN_FIT_POINTS:
        raise ValueError(
            f"settle analysis: {cand.size} samples in the linear window, need "
            f">= {MIN_FIT_POINTS}; the settle may be faster than the sample rate"
        )

    fit_x0 = float(post_x[cand[0]])
    log_r = np.log10(residual[cand])
    xs = post_x[cand] - fit_x0
    lead0, tail0 = float(log_r[0]), float(log_r[-1])
    while True:
        slope, intercept = np.polyfit(xs, log_r, 1)
        dev = log_r - (slope * xs + intercept)
        rms = float(np.sqrt(np.mean(dev * dev)))
        band = TRIM_K * max(rms, 1e-12)
        if abs(dev[0]) > band and len(xs) - 1 >= MIN_FIT_POINTS:
            xs, log_r = xs[1:], log_r[1:]
            continue
        if abs(dev[-1]) > band and len(xs) - 1 >= MIN_FIT_POINTS:
            xs, log_r = xs[:-1], log_r[:-1]
            continue
        break
    lead_trim = lead0 - float(log_r[0])
    tail_trim = float(log_r[-1]) - tail0
    if slope >= 0.0:
        raise ValueError(
            f"settle analysis: log residual is not decreasing (slope "
            f"{slope:.4g}); no exponential settle found"
        )

    # rise time across the transition itself, from the pre-step baseline to the
    # settled point, normalized so the sign of the step drops out
    seg_lo = int(np.searchsorted(x, x[edge_start]))
    seg_hi = peak + 1 + settled_i
    seg_x = x[seg_lo : seg_hi + 1]
    frac = (y[seg_lo : seg_hi + 1] - y_pre) / step
    i_mid = int(np.flatnonzero(frac >= 0.5)[0])
    rise_x10 = _crossing_x(seg_x, frac, 0.10, i_mid, forward=False)
    rise_x90 = _crossing_x(seg_x, frac, 0.90, i_mid, forward=True)
    rise_x20 = _crossing_x(seg_x, frac, 0.20, i_mid, forward=False)
    rise_x80 = _crossing_x(seg_x, frac, 0.80, i_mid, forward=True)

    half = len(xs) // 2
    if half >= 3:
        slope_a = float(np.polyfit(xs[:half], log_r[:half], 1)[0])
        slope_b = float(np.polyfit(xs[half:], log_r[half:], 1)[0])
    else:
        slope_a = slope_b = float(slope)

    return SettleSegments(
        y_pre=y_pre,
        y_final=y_final,
        step_height=float(step),
        noise_sigma=float(noise_sigma),
        baseline_n=int(baseline.size),
        baseline_x0=float(x[b0]),
        edge_start_x=float(x[edge_start]),
        edge_end_x=edge_end_x,
        linear_start_x=fit_x0 + float(xs[0]),
        linear_end_x=fit_x0 + float(xs[-1]),
        n_fit_points=int(len(xs)),
        rise_10_90=rise_x90 - rise_x10,
        rise_20_80=rise_x80 - rise_x20,
        rise_x10=rise_x10,
        rise_x90=rise_x90,
        slope=float(slope),
        slope_first_half=slope_a,
        slope_second_half=slope_b,
        lead_trim_decades=lead_trim,
        tail_trim_decades=tail_trim,
        tau=float(np.log10(np.e) / -slope),
        fit_x0=fit_x0,
        fit_intercept=float(intercept),
        fit_rms=rms,
        settled_x=float(post_x[settled_i]),
        settling_time=float(post_x[settled_i] - x[edge_start]),
        span_x1=float(post_x[-1]),
    )


class SettleAnalysisArtists:
    """Segmentation overlay drawn in the log-residual display space."""

    def __init__(self, ax):
        self.ax = ax
        self._artists: list = []

    def clear(self) -> None:
        for artist in self._artists:
            artist.remove()
        self._artists = []

    def draw(self, seg: SettleSegments, sample_rate_hz: float | None = None) -> None:
        self.clear()
        ax = self.ax
        fmt = x_formatter(sample_rate_hz)
        color = _text_color(ax)

        for xpos, line_color in (
            (seg.edge_start_x, EDGE_COLOR),
            (seg.rise_x10, RISE_COLOR),
            (seg.rise_x90, RISE_COLOR),
            (seg.linear_start_x, FIT_COLOR),
            (seg.linear_end_x, FIT_COLOR),
            (seg.settled_x, SETTLED_COLOR),
        ):
            self._artists.append(
                ax.axvline(
                    xpos, color=line_color, linewidth=1.0, linestyle="--", alpha=0.8
                )
            )

        self._artists.append(
            ax.axhline(
                np.log10(seg.noise_sigma),
                color=FLOOR_COLOR,
                linewidth=1.0,
                linestyle=":",
                alpha=0.8,
            )
        )

        fit_x = np.array([seg.linear_start_x, seg.linear_end_x])
        fit_y = seg.slope * (fit_x - seg.fit_x0) + seg.fit_intercept
        self._artists.append(
            ax.plot(fit_x, fit_y, color=POLE_COLOR, linewidth=1.2, alpha=0.9)[0]
        )

        key = ax.legend(
            handles=[
                Line2D([], [], color=EDGE_COLOR, linestyle="--",
                       label="edge start (last pre-step sample)"),
                Line2D([], [], color=RISE_COLOR, linestyle="--",
                       label="10% and 90% crossings"),
                Line2D([], [], color=FIT_COLOR, linestyle="--",
                       label="fitted region"),
                Line2D([], [], color=SETTLED_COLOR, linestyle="--",
                       label=f"settled ({SETTLED_M:.0f}\u03c3 band)"),
                Line2D([], [], color=POLE_COLOR, label="fitted single pole"),
                Line2D([], [], color=FLOOR_COLOR, linestyle=":",
                       label="noise floor (1\u03c3)"),
            ],
            loc="upper right",
            fontsize=8,
            facecolor=ax.get_facecolor(),
            edgecolor=color,
            labelcolor=color,
            framealpha=0.75,
        )
        key.set_zorder(1000)
        self._artists.append(key)

        self._artists.append(
            ax.text(
                0.02,
                0.02,
                (
                    f"rise 10-90% = {fmt(seg.rise_10_90)}"
                    f"  (tau {fmt(seg.rise_10_90 / RISE_TAU_RATIO)})\n"
                    f"tau fitted = {fmt(seg.tau)}"
                    f"  (rms {seg.fit_rms:.3g} dec)\n"
                    f"settled({SETTLED_M:.0f}\u03c3) after {fmt(seg.settling_time)}"
                ),
                transform=ax.transAxes,
                color=color,
                fontsize=9,
                verticalalignment="bottom",
                zorder=1000,
            )
        )
