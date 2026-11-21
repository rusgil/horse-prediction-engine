"""Single source of truth for the Sharp gate.

ONE definition that every surface POINTS to — both `horse_engine.api.main` and
`horse_engine.models.database` import `is_sharp_gate` from here — so the Sharp
classification can never drift between code paths again.

History: Sharp used to be coded in THREE places — the gate function in main.py
plus two inline copies inside `save_race_predictions` that silently omitted the
clear-favourite and late-non-metro rules, so a race could be flagged Sharp
differently depending on which code path froze it. Consolidated here 2026-11-21.

Recalibration (2026-11-21): the win_prob-based gate had lost its edge — over a
6-month sample Sharp (~36% win) ≈ the field (~34%), and out-of-sample it decayed
(≈50% train → 32% test) because the model's top-confidence band is anti-predictive
(claimed ≥46% win → 29% actual). A market-price slice was both higher AND
OOS-stable (short-priced favourites ≈54-59% win, train≈test). So from
`_SHARP_GATE_ODDS_FROM`, Sharp = the model's rank-1 pick is a short-priced market
favourite (`best_available_odds < _SHARP_MAX_ODDS`). Earlier races keep the old
win_prob gate EXACTLY (date-versioned → frozen history is never re-judged).
See memory note `sharp-filter-auto-recalibration` for the full analysis; the
planned weekly slice-scanner will tune these thresholds automatically.
"""

# Legacy win_prob gate constants (pre-recalibration races only).
_SHARP_MIN_GAP = 0.06
_SHARP_MIN_GAP_PCT = 6.0
_SHARP_GATE_LATE_NONMETRO_FROM = "2026-08-02"
_SHARP_GATE_CLEAR_FAV_FROM = "2026-08-14"

# Market-price recalibration go-live. On/after this date Sharp is defined by the
# pre-jump market price, not win_prob.
_SHARP_GATE_ODDS_FROM = "2026-11-21"
# The model's rank-1 pick is Sharp iff its best_available_odds is strictly below
# this (a short-priced favourite). Calibrated to ~sp<2.3 — ≈54% win, OOS-stable,
# ~5-10 Sharp races/day. Tune via the slice-scanner, not by hand drift.
_SHARP_MAX_ODDS = 2.3


def is_sharp_gate(
    rank1_prob,
    rank2_prob,
    top3_sum,
    days_since_last_run=None,
    is_metro=True,
    race_number=None,
    meeting_max_race=0,
    race_date=None,
    best_available_odds=None,
):
    """Is a race's rank-1 pick Sharp? ALL probabilities on the 0-1 scale
    (callers in 0-100 percent must divide by 100 first).

    DATE-VERSIONED so a historical race is judged exactly as it was flagged then
    — pass `race_date` ('YYYY-MM-DD') for settled races; leave None for
    live/upcoming to get the newest gate.

    On/after `_SHARP_GATE_ODDS_FROM` (market-price gate):
      - the pick must be PRICED (best_available_odds > 1.0) and short —
        best_available_odds < `_SHARP_MAX_ODDS`  (a blind/unpriced race is never Sharp), AND
      - not a long layoff (days_since_last_run <= 180).
    Before that date (legacy win_prob gate — unchanged):
      1. rank1 >= 0.30 OR top-3 sum >= 0.60
      2. clear favourite (rank1 - rank2) >= `_SHARP_MIN_GAP`  (from 2026-08-14)
      3. not a long layoff (days_since_last_run <= 180)
      4. not a late race at a non-metro meeting               (from 2026-08-02)
    """
    # Long-layoff guard applies in BOTH eras.
    if isinstance(days_since_last_run, (int, float)) and days_since_last_run > 180:
        return False

    # ── Market-price gate (live/upcoming, and settled races on/after go-live) ──
    if race_date is None or race_date >= _SHARP_GATE_ODDS_FROM:
        try:
            odds = float(best_available_odds) if best_available_odds is not None else None
        except (TypeError, ValueError):
            odds = None
        if odds is None or odds <= 1.0:
            return False  # unpriced / blind race is never Sharp
        return odds < _SHARP_MAX_ODDS

    # ── Legacy win_prob gate (historical races before go-live only) ──
    r1 = rank1_prob or 0.0
    if not (r1 >= 0.30 or (top3_sum or 0.0) >= 0.60):
        return False
    if race_date >= _SHARP_GATE_CLEAR_FAV_FROM:
        if (r1 - (rank2_prob or 0.0)) < _SHARP_MIN_GAP:
            return False
    if race_date >= _SHARP_GATE_LATE_NONMETRO_FROM:
        if (not is_metro) and race_number is not None and (meeting_max_race or 0) >= 5 \
                and race_number >= (meeting_max_race or 0) - 1:
            return False
    return True
