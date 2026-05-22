"""
Venue-level calibration multipliers.

The logistic regression model has systematic over/under-confidence at
certain venues (e.g. Eagle Farm 22% actual vs ~50% global, Casterton 87%).
This module computes a per-venue multiplier from historical results and
applies it to raw win probabilities before saving predictions.

Multiplier formula (dampened to avoid over-correction):
    raw_mult = venue_win_rate / global_win_rate
    multiplier = clamp(0.5 + 0.5 * raw_mult, 0.55, 1.45)

Min 15 races required before a venue gets a non-neutral multiplier.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_MIN_RACES = 15
_MULT_MIN = 0.55
_MULT_MAX = 1.45


def compute_venue_multipliers(venue_stats: dict[str, dict]) -> dict[str, float]:
    """
    venue_stats: {venue_code: {"races": int, "wins": int}}
    Returns {venue_code: multiplier}.
    Venues with < _MIN_RACES data get multiplier 1.0 (neutral).
    """
    qualified = {v: s for v, s in venue_stats.items() if s["races"] >= _MIN_RACES}
    if not qualified:
        return {}

    total_races = sum(s["races"] for s in qualified.values())
    total_wins = sum(s["wins"] for s in qualified.values())
    global_rate = total_wins / total_races if total_races else 0.50

    multipliers: dict[str, float] = {}
    for venue, stats in qualified.items():
        venue_rate = stats["wins"] / stats["races"]
        raw_mult = venue_rate / max(global_rate, 0.01)
        dampened = 0.5 + 0.5 * raw_mult
        clamped = max(_MULT_MIN, min(_MULT_MAX, dampened))
        multipliers[venue] = round(clamped, 4)
        if abs(clamped - 1.0) > 0.05:
            log.debug(
                "Venue calibration %s: %.1f%% actual vs %.1f%% global → ×%.3f",
                venue, venue_rate * 100, global_rate * 100, clamped,
            )

    return multipliers


def apply_venue_calibration(
    win_probs: list[float],
    venue: str,
    multipliers: dict[str, float],
) -> list[float]:
    """Scale win probabilities by venue multiplier. Clamps result to [0, 1]."""
    mult = multipliers.get(venue, 1.0)
    if mult == 1.0:
        return win_probs
    return [max(0.0, min(1.0, p * mult)) for p in win_probs]
