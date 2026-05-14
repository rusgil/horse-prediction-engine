"""
Speed map analysis.

Determines where each horse is likely to settle in the running,
and whether a pace scenario favours frontrunners or backmarkers.

Running style is inferred from:
  - In-running comments in last 3 starts
  - Sire's on_pace_tendency score
  - Barrier (low barriers = more likely to lead)
  - Distance (shorter = more pace horses)
"""
from __future__ import annotations

import re
from collections import Counter

from horse_engine.models.race import Runner
from horse_engine.pedigree.sire_profiles import get_sire_profile

# Keywords that suggest running position
LEADER_KEYWORDS = ["led", "led throughout", "led easily", "made all", "on pace", "jumped well", "disputed"]
ON_PACE_KEYWORDS = ["settled 2nd", "settled 3rd", "handy", "on speed", "midfield on pace", "one back"]
MIDFIELD_KEYWORDS = ["settled midfield", "4th", "5th", "one pace", "midfield"]
BACK_KEYWORDS = ["settled last", "back", "backmarker", "settled well back", "7th", "8th", "9th", "10th", "tailed off"]


def infer_running_style(runner: Runner, race_distance: int) -> str:
    """
    Infer running style from in-running comments and pedigree.
    Returns: "leader" | "on_pace" | "midfield" | "backmarker"
    """
    style_votes: Counter = Counter()

    # Parse in-running comments from last 3 starts
    for s in runner.last_10_starts[:3]:
        comment = (s.in_running or "") + " " + (s.comment or "")
        c = comment.lower()

        if any(kw in c for kw in LEADER_KEYWORDS):
            style_votes["leader"] += 2
        elif any(kw in c for kw in ON_PACE_KEYWORDS):
            style_votes["on_pace"] += 2
        elif any(kw in c for kw in MIDFIELD_KEYWORDS):
            style_votes["midfield"] += 1
        elif any(kw in c for kw in BACK_KEYWORDS):
            style_votes["backmarker"] += 2

    # Pedigree on-pace signal
    sire_p = get_sire_profile(runner.pedigree.sire if runner.pedigree else "")
    if sire_p:
        pace = sire_p.get("on_pace", 5)
        if pace >= 8:
            style_votes["leader"] += 1
        elif pace >= 6:
            style_votes["on_pace"] += 1
        elif pace <= 3:
            style_votes["backmarker"] += 1

    # Low barrier → slight leader/on-pace lean for short distances
    if race_distance <= 1400:
        if runner.barrier <= 3:
            style_votes["leader"] += 1
        elif runner.barrier <= 6:
            style_votes["on_pace"] += 1

    if not style_votes:
        return "midfield"

    return style_votes.most_common(1)[0][0]


def build_speed_map(runners: list[Runner], race_distance: int) -> dict[str, str]:
    """Returns {horse_name: running_style} for all runners."""
    return {r.horse_name: infer_running_style(r, race_distance) for r in runners}


def pace_scenario_score(speed_map: dict[str, str]) -> dict[str, float]:
    """
    Given the full speed map, compute a scenario advantage for each horse.

    Hot pace (many leaders) → backmarkers / closers benefit.
    Slow pace (few leaders) → on-pace runners benefit.

    Returns {horse_name: advantage (-1 to +1)}
    """
    counts = Counter(speed_map.values())
    leaders_and_pace = counts.get("leader", 0) + counts.get("on_pace", 0)
    total = len(speed_map)
    if total == 0:
        return {h: 0.0 for h in speed_map}

    pace_ratio = leaders_and_pace / total  # 0 = no pace, 1 = all pace

    result = {}
    for horse, style in speed_map.items():
        if pace_ratio >= 0.5:
            # Hot pace scenario — advantages closers
            adv = {
                "backmarker": 0.3,
                "midfield": 0.1,
                "on_pace": -0.1,
                "leader": -0.3,
            }.get(style, 0.0)
        else:
            # Slow pace scenario — advantages on-pace
            adv = {
                "leader": 0.3,
                "on_pace": 0.2,
                "midfield": -0.1,
                "backmarker": -0.3,
            }.get(style, 0.0)
        result[horse] = adv

    return result
