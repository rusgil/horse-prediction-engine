"""
Market signal enrichment.

Converts raw odds into predictive market features:
- Overround-free implied probabilities
- Steam/drift detection
- Best available odds across books
- Overlay calculation vs model
"""
from __future__ import annotations

from horse_engine.clients.odds import (
    best_odds,
    compute_odds_movement,
    implied_probability,
    overround_free_probability,
)
from horse_engine.models.race import Runner

STEAM_THRESHOLD = 1.0   # shortened by $1+ = steamed
DRIFT_THRESHOLD = 1.5   # drifted by $1.50+ = drifted


def compute_market_features(
    runners: list[Runner],
) -> dict[str, dict]:
    """
    Returns a dict keyed by horse_name with market features for each runner.
    All-runner context is required to normalise field-level probabilities.
    """
    # Collect best available odds for each runner
    tab_odds = [r.fixed_win_odds or r.tote_win_odds or 0.0 for r in runners]
    sb_odds = [r.sportsbet_odds or 0.0 for r in runners]
    lb_odds = [r.ladbrokes_odds or 0.0 for r in runners]

    # Best odds across books per runner
    best = []
    best_books = []
    for i, r in enumerate(runners):
        b, book = best_odds(tab_odds[i] or None, sb_odds[i] or None, lb_odds[i] or None)
        best.append(b)
        best_books.append(book)

    # Overround-free probabilities from TAB (primary book)
    valid_tab = [o if o > 1.0 else 999.0 for o in tab_odds]
    orf_probs = overround_free_probability(valid_tab)

    # Market rank (1 = favourite)
    ranked = sorted(range(len(runners)), key=lambda i: tab_odds[i] if tab_odds[i] > 0 else 9999)
    rank_map = {runners[i].horse_name: rank + 1 for rank, i in enumerate(ranked)}

    result = {}
    for i, r in enumerate(runners):
        opening = r.odds_opening
        current = tab_odds[i] or None
        movement = compute_odds_movement(opening, current)
        is_steamed = movement >= STEAM_THRESHOLD
        is_drifted = movement <= -DRIFT_THRESHOLD

        result[r.horse_name] = {
            "market_rank": rank_map.get(r.horse_name, len(runners)),
            "market_implied_prob": orf_probs[i] if i < len(orf_probs) else 0.0,
            "odds_movement": movement,
            "best_available_odds": best[i],
            "best_book": best_books[i],
            "is_steamed": is_steamed,
            "is_drifted": is_drifted,
            "overlay": 0.0,  # filled in after model run
        }

    return result
