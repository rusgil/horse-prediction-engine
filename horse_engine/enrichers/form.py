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


# Layoff thresholds for the form-score discount below. A normal racing
# spell is 3-6 months; anything past 120 days starts to compromise the
# signal that "recent form" is conveying. Past a year off the track the
# horse hasn't shown the public anything in a long time and we treat
# the form record as effectively unknown.
LAYOFF_DISCOUNT_START_DAYS = 120
# 365 → 250 (2026-07-28): 180-day backtest of 2,426 rank-1 picks showed
# 250-365d picks winning 19.2% vs 23.2% claimed (-4pt) while 121-249d was
# calibrated (-0.8pt) — the old taper still credited too much stale form in
# its back half. Full discount to baseline now lands at 250 days.
LAYOFF_DISCOUNT_FULL_DAYS = 250
LAYOFF_BASELINE_SCORE = 0.3  # matches weighted_form_score's no-data fallback


def discount_form_for_thin_record(form_score: float, starts_last_10: int) -> float:
    """Discount form_score toward the no-form baseline when the recent
    record is too thin to support a confident reading.

    Triggered by OSAKA CASTLE/sportsbet-ballarat R5 on 2026-06-12: 2
    starts in the last 10, form score 0.62, model gave it 31% and rank
    1. SP $31 (market rank 1, so the market was also confused), finished
    10th of 11.

    Extended on 2026-07-10 after COAL SEAM/mackay R1: 3 starts (1 win,
    3 places = 100% place strike rate), 182 days off, form score
    0.7495, model gave it 68.8% and rank 1 with an 11× confidence gap
    to rank-2. Finished 6th of 8. Three starts is still thin — a
    1-3-3 record looks elite but two of those places could have been
    5-length beaten seconds and we'd never know. The old '≥3 starts
    → unchanged' rule treated 3 starts as a workable sample; it isn't.

    Discount weights:
        ≥ 4 starts → form_score unchanged (workable sample)
        3 starts   → pull 15% toward the no-form baseline (0.3) — NEW
        2 starts   → pull 35% toward the no-form baseline
        1 start    → pull 70% toward the no-form baseline
        0 starts   → return baseline (handled by weighted_form_score's
                     no-form prior already, but defended here too)

    Like the layoff discount, this is a no-op for horses already at or
    below the baseline — thin record shouldn't make a poor performer
    look worse.
    """
    if starts_last_10 is None:
        return form_score
    if starts_last_10 >= 4:
        return form_score
    if form_score <= LAYOFF_BASELINE_SCORE:
        return form_score
    if starts_last_10 <= 0:
        return LAYOFF_BASELINE_SCORE
    weight = {1: 0.7, 2: 0.35, 3: 0.15}.get(starts_last_10, 0.0)
    return round(form_score - (form_score - LAYOFF_BASELINE_SCORE) * weight, 4)


# Extreme-layoff floor — applied beyond LAYOFF_DISCOUNT_FULL_DAYS. Past
# a year off the track the recency signal is effectively zero; past two
# years it's actively misleading. Drop the score below the no-form
# baseline so the model can tell the difference between 'never raced
# recently' and 'so far in the past that we shouldn't trust it'.
# 730 → 550 (2026-07-28, same backtest): 366d+ picks won 14.6% vs 20.9%
# claimed (-6.3pt) — the crawl from baseline to the extreme floor was too
# slow. The floor now lands at ~18 months instead of 2 years.
LAYOFF_EXTREME_DAYS = 550  # ~18 months
LAYOFF_EXTREME_SCORE = 0.15


def discount_form_for_layoff(form_score: float, days_since_last_run: int) -> float:
    """Discount form_score toward the no-form baseline when the horse has
    been off the track for an unusually long time.

    Triggered by BULLETIN BEAU/Pinjarra R5 on 2026-06-24: 468 days off
    the track, model rank 1 (31%, +23pt edge), market rank 6 (the market
    knew better), finished 12th of 12.

    Curve (tightened 2026-07-28 — 180d backtest, 2,426 rank-1 picks:
    250-365d actual 19.2% vs 23.2% claimed, 366d+ 14.6% vs 20.9%;
    COMIC CULTURE/Bathurst R4 at 304d was the trigger case):
        ≤ 120 days     → form_score unchanged (normal racing spell)
        120-250 days   → linear taper toward the no-form baseline (0.3)
                         (was 120-365; the back half under-discounted)
        250-550 days   → linear taper from 0.3 down to 0.15
                         (was 365-730 — OFFENBACH/Northam R7 2026-06-25
                         at 426 days, rank-1 form 0.71 → 12th of 13)
        ≥ 550 days     → 0.15 (treat as actively misleading)

    A horse whose form is already at/below the floor is left alone — a
    layoff shouldn't make a poor performer look worse.
    """
    if days_since_last_run is None or days_since_last_run < 0:
        return form_score
    if days_since_last_run <= LAYOFF_DISCOUNT_START_DAYS:
        return form_score
    if form_score <= LAYOFF_EXTREME_SCORE:
        return form_score
    # First segment: 120-365 days, taper toward baseline 0.3
    if days_since_last_run < LAYOFF_DISCOUNT_FULL_DAYS:
        span = LAYOFF_DISCOUNT_FULL_DAYS - LAYOFF_DISCOUNT_START_DAYS
        progress = (days_since_last_run - LAYOFF_DISCOUNT_START_DAYS) / span
        target = form_score - (form_score - LAYOFF_BASELINE_SCORE) * progress
        return round(target, 4)
    # Second segment: 365-730 days, taper from baseline 0.3 → extreme floor 0.15
    if days_since_last_run < LAYOFF_EXTREME_DAYS:
        # Anchor: at 365d we're at baseline (or form_score, whichever lower).
        start_score = min(form_score, LAYOFF_BASELINE_SCORE)
        if start_score <= LAYOFF_EXTREME_SCORE:
            return round(start_score, 4)
        span = LAYOFF_EXTREME_DAYS - LAYOFF_DISCOUNT_FULL_DAYS
        progress = (days_since_last_run - LAYOFF_DISCOUNT_FULL_DAYS) / span
        target = start_score - (start_score - LAYOFF_EXTREME_SCORE) * progress
        return round(target, 4)
    # Third segment: ≥730d → extreme floor
    return LAYOFF_EXTREME_SCORE


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
