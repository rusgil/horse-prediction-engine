"""
Form guide feature extraction.

Converts a horse's last_10_starts into quantitative features.
All scoring is from the horse's own record — no inter-horse comparison here,
that happens in the feature vector normalisation step.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional

from horse_engine.models.race import FormStart, Runner
from horse_engine.pedigree.sire_profiles import parse_condition_category

# Weights for recency: most recent start = 1.0, older = decayed
RECENCY_WEIGHTS = [1.0, 0.85, 0.72, 0.60, 0.50, 0.42, 0.35, 0.28, 0.22, 0.18]


def _condition_cat(tc: str) -> str:
    return parse_condition_category(tc)


def weighted_form_score(starts: list[FormStart]) -> float:
    """
    0–1 score: 1.0 = won every recent start, 0.0 = finished last every time.
    Uses recency-weighted normalised finish position.
    """
    if not starts:
        return 0.3  # no form data — slight negative prior
    scores = []
    for i, s in enumerate(starts[:10]):
        if s.finishers <= 1:
            continue
        w = RECENCY_WEIGHTS[i]
        # Normalise position: 1st = 1.0, last = 0.0
        pos_score = (s.finishers - s.position) / (s.finishers - 1)
        scores.append(w * pos_score)
    if not scores:
        return 0.3
    total_w = sum(RECENCY_WEIGHTS[:len(scores)])
    return round(sum(scores) / total_w, 4)


def win_rate_at_distance(starts: list[FormStart], race_distance: int, tolerance: int = 200) -> float:
    relevant = [s for s in starts if abs(s.distance - race_distance) <= tolerance]
    if not relevant:
        return 0.0
    wins = sum(1 for s in relevant if s.position == 1)
    return round(wins / len(relevant), 4)


def win_rate_at_track(starts: list[FormStart], venue: str) -> float:
    relevant = [s for s in starts if s.track.lower() == venue.lower()]
    if not relevant:
        return 0.0
    wins = sum(1 for s in relevant if s.position == 1)
    return round(wins / len(relevant), 4)


def win_rate_at_class(starts: list[FormStart], race_class: str) -> float:
    relevant = [s for s in starts if s.race_class.upper() == race_class.upper()]
    if not relevant:
        return 0.0
    wins = sum(1 for s in relevant if s.position == 1)
    return round(wins / len(relevant), 4)


def avg_beaten_margin_last5(starts: list[FormStart]) -> float:
    recent = starts[:5]
    if not recent:
        return 10.0  # unknown, assume beaten far
    return round(sum(s.beaten_margin for s in recent) / len(recent), 2)


def wet_track_record(starts: list[FormStart]) -> float:
    """Win % on soft/heavy going."""
    wet = [s for s in starts if _condition_cat(s.track_condition) in ("soft", "heavy")]
    if not wet:
        return 0.0
    return round(sum(1 for s in wet if s.position == 1) / len(wet), 4)


def dry_track_record(starts: list[FormStart]) -> float:
    """Win % on good/firm/synthetic going."""
    dry = [s for s in starts if _condition_cat(s.track_condition) in ("good", "firm", "synthetic")]
    if not dry:
        return 0.0
    return round(sum(1 for s in dry if s.position == 1) / len(dry), 4)


def surface_match_score(starts: list[FormStart], current_condition: str) -> float:
    """
    0–1: how well today's going matches what this horse has performed well on historically.
    """
    cat = _condition_cat(current_condition)
    if cat in ("soft", "heavy"):
        record = wet_track_record(starts)
        all_record = wet_track_record(starts) + dry_track_record(starts)
        if all_record == 0:
            return 0.5
        return round(record / max(all_record, 0.01), 3)
    else:
        record = dry_track_record(starts)
        all_record = wet_track_record(starts) + dry_track_record(starts)
        if all_record == 0:
            return 0.5
        return round(record / max(all_record, 0.01), 3)


def days_since_last_run(starts: list[FormStart]) -> int:
    """Days since most recent start. -1 if no prior starts."""
    if not starts:
        return -1
    last_date_str = starts[0].date
    try:
        last_date = datetime.strptime(last_date_str[:10], "%Y-%m-%d").date()
        return (date.today() - last_date).days
    except Exception:
        return -1


def runs_this_prep(starts: list[FormStart], spell_threshold_days: int = 60) -> int:
    """
    Count runs since last spell (>= spell_threshold_days gap between starts).
    0 = first-up (resuming), 1 = second-up, etc.
    """
    if not starts:
        return 0

    sorted_starts = sorted(starts, key=lambda s: s.date, reverse=True)
    count = 0
    prev_date = None

    for s in sorted_starts:
        try:
            d = datetime.strptime(s.date[:10], "%Y-%m-%d").date()
        except Exception:
            continue

        if prev_date is not None:
            gap = (prev_date - d).days
            if gap >= spell_threshold_days:
                break
        count += 1
        prev_date = d

    return count


def spell_weeks(starts: list[FormStart]) -> Optional[int]:
    """
    If horse is resuming, how many weeks was the spell?
    Returns None if in an ongoing prep.
    """
    d = days_since_last_run(starts)
    if d < 0:
        return None
    if d >= 60:
        return d // 7
    return None


def class_change_flag(starts: list[FormStart], current_class: str) -> int:
    """
    -1 = stepping up in class, 0 = same, +1 = stepping down.
    This is counterintuitive — a drop in class is a positive (easier race).
    """
    CLASS_RANK = {
        "maidens": 0, "maiden": 0,
        "class 1": 1, "0hcp": 1,
        "class 2": 2, "mdnhcp": 2,
        "class 3": 3, "bm58": 3, "bm60": 3,
        "bm64": 4, "bm65": 4,
        "bm70": 5, "bm72": 5,
        "bm78": 6, "bm80": 6,
        "bm88": 7, "bm90": 7,
        "listed": 8, "listed race": 8,
        "g3": 9, "group 3": 9,
        "g2": 10, "group 2": 10,
        "g1": 11, "group 1": 11,
    }

    def rank(c: str) -> int:
        return CLASS_RANK.get(c.lower(), 4)

    if not starts:
        return 0
    last_class = starts[0].race_class
    curr_rank = rank(current_class)
    last_rank = rank(last_class)
    if curr_rank < last_rank:
        return 1   # easier race = positive
    elif curr_rank > last_rank:
        return -1  # tougher race = negative
    return 0


def best_finish_at_distance(starts: list[FormStart], race_distance: int, tolerance: int = 200) -> int:
    relevant = [s for s in starts if abs(s.distance - race_distance) <= tolerance]
    if not relevant:
        return 99
    return min(s.position for s in relevant)


def career_win_rate(runner: "Runner") -> float:
    if runner.career_starts == 0:
        return 0.0
    return round(runner.career_wins / runner.career_starts, 4)
