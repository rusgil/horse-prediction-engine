"""
Output-space calibration for the win model.

The logistic regression + softmax pipeline produces model_pct values
that can be sharply miscalibrated at the top of the range: when the
model outputs 70%, the historical hit rate for its 70% predictions
might be 40%. Venue calibration corrects for per-venue drift but not
global over-confidence. The reactive multipliers in engine.py (midfield,
going, thin-record, feature-completeness) only fire on whitelisted
conditions.

Isotonic regression on stored predictions vs actual outcomes fits a
monotonic curve that maps the model's confidence to empirical
frequency. Standard reliability-diagram / Platt-scaling / isotonic-
calibration approach (Zadrozny & Elkan 2002, Guo et al. 2017).

Compute nightly. Apply at inference between the raw model output and
the reactive multipliers — so multipliers still fire on the calibrated
value, but the raw softmax spikes are already dampened by the time
they hit them.

Curve representation: list of (input_pct, output_pct) breakpoints in
sorted, monotonic-non-decreasing order. Apply via piecewise linear
interpolation. First/last points anchor the domain.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


# Minimum samples per bucket before we trust its empirical rate.
# Below this, the bucket blends toward the identity (raw model_pct).
_MIN_BUCKET_SAMPLES = 20

# Minimum total samples before we ship a curve at all. Guards against
# a cold-start pipeline shipping a wildly off curve on n=50.
_MIN_TOTAL_SAMPLES = 500

# Bucket edges — narrower at the top where over-confidence lives, wider
# at the bottom where nothing important happens. Values are model_pct
# in [0, 100].
_BUCKET_EDGES = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100
]


def _bucket_of(pct: float) -> int:
    """Return index in [0, len(_BUCKET_EDGES)-2] for a model_pct."""
    for i in range(len(_BUCKET_EDGES) - 1):
        if pct < _BUCKET_EDGES[i + 1]:
            return i
    return len(_BUCKET_EDGES) - 2


def _pav(y: list[float], w: list[float]) -> list[float]:
    """
    Pool-Adjacent-Violators isotonic regression (non-decreasing).

    y: values to smooth (empirical win rates per bucket).
    w: weights (sample counts per bucket).
    Returns list of same length, monotonically non-decreasing.

    Standard PAV: walk left-to-right, whenever the next value violates
    monotonicity, pool it with the previous block using weighted mean;
    repeat pooling back until monotone or start-of-array.
    """
    if not y:
        return []
    blocks: list[tuple[float, float]] = []  # (weighted_sum, weight)
    for yi, wi in zip(y, w):
        wi = max(wi, 1e-9)
        blocks.append((yi * wi, wi))
        # Pool backward while last two are decreasing.
        while len(blocks) >= 2:
            (s1, w1), (s2, w2) = blocks[-2], blocks[-1]
            if s1 / w1 <= s2 / w2:
                break
            blocks[-2:] = [(s1 + s2, w1 + w2)]
    out: list[float] = []
    for s, w_ in blocks:
        v = s / w_
        # Each pooled block expands back to its original length,
        # but blocks track only pooled sums — so we need to remember
        # the original sizes. Instead, re-derive by counting from y/w.
        out.append(v)
    # Expand: each block should replay across its constituent original
    # entries. Rebuild by walking again with pool tracking.
    result: list[float] = []
    i = 0
    for s, tot_w in blocks:
        # Count how many of the original entries were pooled into this
        # block by matching cumulative weight.
        acc = 0.0
        pooled = 0
        while i < len(w) and acc < tot_w - 1e-9:
            acc += max(w[i], 1e-9)
            pooled += 1
            i += 1
        result.extend([s / tot_w] * max(pooled, 1))
    # Safety: pad if rounding drifted.
    while len(result) < len(y):
        result.append(result[-1] if result else 0.0)
    return result[: len(y)]


def compute_calibration_curve(
    samples: list[tuple[float, int]],
) -> Optional[list[tuple[float, float]]]:
    """
    Build an isotonic calibration curve from (model_pct, won) pairs.

    samples: list of (model_pct, won_int) — model_pct in [0,100],
             won_int in {0, 1}. Any horse in a settled race qualifies;
             pass rank-1 only if you want to calibrate the top-pick
             sub-slice specifically.

    Returns: list of (input_pct, calibrated_pct) breakpoints in
    monotone-non-decreasing order, or None if there isn't enough data
    to trust the fit.
    """
    if len(samples) < _MIN_TOTAL_SAMPLES:
        return None

    n_buckets = len(_BUCKET_EDGES) - 1
    counts = [0] * n_buckets
    wins = [0] * n_buckets
    midpoints = [
        (_BUCKET_EDGES[i] + _BUCKET_EDGES[i + 1]) / 2.0
        for i in range(n_buckets)
    ]
    for pct, won in samples:
        b = _bucket_of(pct)
        counts[b] += 1
        wins[b] += 1 if won else 0

    # Empirical rate per bucket; buckets below _MIN_BUCKET_SAMPLES fall
    # back to the identity (bucket midpoint) so a sparse extreme doesn't
    # yank the curve to noise.
    empirical: list[float] = []
    weights: list[float] = []
    for i in range(n_buckets):
        if counts[i] >= _MIN_BUCKET_SAMPLES:
            empirical.append(100.0 * wins[i] / counts[i])
            weights.append(float(counts[i]))
        else:
            empirical.append(midpoints[i])
            weights.append(float(counts[i]) * 0.25)  # low trust

    smoothed = _pav(empirical, weights)

    # Build breakpoints: (bucket_midpoint, smoothed_value). Anchor start
    # at 0 and end at 100 so extrapolation is well-defined.
    curve = [(0.0, smoothed[0] if smoothed else 0.0)]
    for i, s in enumerate(smoothed):
        curve.append((midpoints[i], s))
    curve.append((100.0, smoothed[-1] if smoothed else 100.0))

    # Deduplicate co-located points, keep monotone.
    seen_x: set[float] = set()
    dedup: list[tuple[float, float]] = []
    for x, y in curve:
        if x in seen_x:
            continue
        seen_x.add(x)
        if dedup and y < dedup[-1][1]:
            y = dedup[-1][1]  # enforce non-decreasing
        dedup.append((x, y))

    return dedup


def apply_calibration_curve(
    win_pcts: list[float],
    curve: Optional[list[tuple[float, float]]],
) -> list[float]:
    """
    Map each model_pct through the isotonic curve via piecewise
    linear interpolation between adjacent breakpoints.

    win_pcts: model output percentages [0, 100].
    curve: list from compute_calibration_curve, or None (identity).

    Returns calibrated percentages, same length.
    """
    if not curve or not win_pcts:
        return list(win_pcts)

    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]

    def interp(x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        # Binary search would be faster but curve has <25 points.
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                if xs[i + 1] == xs[i]:
                    return ys[i]
                t = (x - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    return [round(max(0.0, min(100.0, interp(p))), 4) for p in win_pcts]
