"""
Track surface and barrier features.

Barrier advantage is highly track- and distance-specific.
We use a hardcoded table for major Australian venues based on
known historical bias data.
"""
from __future__ import annotations

from horse_engine.pedigree.sire_profiles import parse_condition_category

# ── Barrier bias tables ────────────────────────────────────────────────────
# Values represent historical win% advantage vs average from that barrier.
# Positive = advantaged, negative = disadvantaged.
# Source: Compiled from publicly available Australian racing analysis.

BARRIER_BIAS: dict[str, dict[int, dict[str, float]]] = {
    # key: venue_lower → {distance: {barrier: advantage}}
    "randwick": {
        1200: {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.0, 6: -0.1, 7: -0.2, 8: -0.3,
               9: -0.3, 10: -0.4, 11: -0.5, 12: -0.5, 13: -0.5, 14: -0.6},
        1600: {1: 0.2, 2: 0.2, 3: 0.1, 4: 0.1, 5: 0.0, 6: 0.0, 7: -0.1, 8: -0.2,
               9: -0.2, 10: -0.3, 11: -0.3, 12: -0.4, 13: -0.4, 14: -0.5},
        2000: {1: 0.1, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.1, 6: 0.0, 7: 0.0, 8: -0.1,
               9: -0.1, 10: -0.2, 11: -0.2, 12: -0.3, 13: -0.3, 14: -0.4},
    },
    "flemington": {
        1000: {1: -0.1, 2: 0.1, 3: 0.3, 4: 0.3, 5: 0.2, 6: 0.1, 7: 0.0, 8: -0.1,
               9: -0.2, 10: -0.2, 11: -0.3, 12: -0.4, 13: -0.4, 14: -0.5},
        1600: {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.0, 6: -0.1, 7: -0.1, 8: -0.2,
               9: -0.2, 10: -0.3, 11: -0.3, 12: -0.4, 13: -0.4, 14: -0.5},
        2400: {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.2, 5: 0.1, 6: 0.1, 7: 0.0, 8: -0.1,
               9: -0.1, 10: -0.2, 11: -0.2, 12: -0.2, 13: -0.3, 14: -0.3},
    },
    "moonee valley": {
        # MV is tight, small circuit — favours low barriers and on-pace runners
        1600: {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.0, 7: -0.2, 8: -0.3,
               9: -0.4, 10: -0.5, 11: -0.6, 12: -0.7, 13: -0.8, 14: -0.9},
        2040: {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.0, 6: 0.0, 7: -0.1, 8: -0.2,
               9: -0.3, 10: -0.3, 11: -0.4, 12: -0.5, 13: -0.5, 14: -0.6},
    },
    "eagle farm": {
        1200: {1: 0.3, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.0, 6: -0.1, 7: -0.2, 8: -0.3,
               9: -0.3, 10: -0.4, 11: -0.5, 12: -0.5, 13: -0.6, 14: -0.6},
        1800: {1: 0.1, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.0, 6: 0.0, 7: -0.1, 8: -0.1,
               9: -0.2, 10: -0.2, 11: -0.3, 12: -0.3, 13: -0.4, 14: -0.4},
    },
    "doomben": {
        1200: {1: 0.2, 2: 0.2, 3: 0.1, 4: 0.0, 5: -0.1, 6: -0.1, 7: -0.2, 8: -0.3,
               9: -0.3, 10: -0.4, 11: -0.5, 12: -0.5, 13: -0.6, 14: -0.6},
    },
    "caulfield": {
        1400: {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.0, 6: -0.1, 7: -0.2, 8: -0.3,
               9: -0.3, 10: -0.4, 11: -0.4, 12: -0.5, 13: -0.5, 14: -0.6},
        1800: {1: 0.2, 2: 0.2, 3: 0.1, 4: 0.0, 5: 0.0, 6: -0.1, 7: -0.1, 8: -0.2,
               9: -0.2, 10: -0.3, 11: -0.3, 12: -0.4, 13: -0.4, 14: -0.5},
    },
    "rosehill": {
        1200: {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.0, 6: -0.1, 7: -0.2, 8: -0.3,
               9: -0.3, 10: -0.4, 11: -0.5, 12: -0.5, 13: -0.6, 14: -0.6},
        2000: {1: 0.1, 2: 0.2, 3: 0.1, 4: 0.1, 5: 0.0, 6: 0.0, 7: -0.1, 8: -0.1,
               9: -0.2, 10: -0.2, 11: -0.3, 12: -0.3, 13: -0.3, 14: -0.4},
    },
}

# Rail position modifiers: where the rail is placed affects inside vs outside
RAIL_BIAS: dict[str, float] = {
    "true": 0.0,
    "out 3m": 0.05,   # slight outside advantage
    "out 5m": 0.08,
    "out 8m": 0.12,
    "in 3m": -0.05,   # slight inside advantage
    "in 5m": -0.08,
}


def barrier_score(
    venue: str,
    distance: int,
    barrier: int,
    rail_position: str = "true",
    field_size: int = 14,
) -> float:
    """
    0–1 score: barrier advantage relative to this race's field.
    0.5 = neutral, >0.5 = favoured, <0.5 = disadvantaged.
    """
    venue_key = venue.lower()
    venue_data = BARRIER_BIAS.get(venue_key)

    if venue_data:
        # Find closest distance in table
        closest = min(venue_data.keys(), key=lambda d: abs(d - distance))
        dist_data = venue_data[closest]
        bias = dist_data.get(min(barrier, max(dist_data.keys())), -0.1)
    else:
        # Generic: low barriers slightly favoured (default assumption for most Aus tracks)
        if barrier <= 4:
            bias = 0.15
        elif barrier <= 8:
            bias = 0.0
        elif barrier <= 12:
            bias = -0.15
        else:
            bias = -0.3

    # Rail position adjustment
    rail_key = rail_position.lower()
    rail_adj = 0.0
    for key, adj in RAIL_BIAS.items():
        if key in rail_key:
            # Wide barriers benefit from inside rail moving out
            if barrier > field_size / 2:
                rail_adj = adj
            else:
                rail_adj = -adj
            break

    raw = 0.5 + (bias + rail_adj) * 0.5
    return round(max(0.0, min(1.0, raw)), 3)


def track_bias_advantage(
    speed_map_position: str,
    rail_position: str,
    venue: str,
) -> float:
    """
    -1 to +1 advantage for a running style given today's rail and venue.
    Leaders benefit from rail being out (fresh grass, wider circuit).
    Backmarkers benefit from rail being in (shorter circuit, less ground to make up).
    """
    is_out = any(x in rail_position.lower() for x in ["out 5", "out 8", "out 10"])
    is_in = any(x in rail_position.lower() for x in ["in 3", "in 5", "in 8"])

    # Moonee Valley heavily favours leaders regardless of rail
    is_mv = "moonee" in venue.lower() or "mv" in venue.lower()

    if speed_map_position in ("leader", "on_pace"):
        if is_out:
            return 0.3
        elif is_in:
            return 0.1   # on-pace still ok on inside rail
        elif is_mv:
            return 0.5
        return 0.1
    elif speed_map_position == "backmarker":
        if is_in:
            return 0.1   # slightly easier to make up ground
        elif is_out:
            return -0.2  # wide out, backmarkers find it tough
        return -0.1
    return 0.0  # midfield — neutral
