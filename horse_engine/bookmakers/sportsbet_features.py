"""Static snapshot of Sportsbet's published racing-feature rules.

Refresh quarterly when their published ladders change. No live scraping;
all our math is done off the model + TAB odds, then translated to
Sportsbet-equivalent dollars using the rules below.
"""

# Multi-boost ladder for cross-race racing multis.
# Source: Sportsbet's published racing-multi promotional rate card.
# Returns the boost as a multiplier on raw multi odds.
MULTI_BOOST_BY_LEGS: dict[int, float] = {
    2: 0.05,
    3: 0.10,
    4: 0.20,
    5: 0.25,
    6: 0.30,
    7: 0.40,
    8: 0.50,
}

def multi_boost_pct(legs: int) -> float:
    """Returns the boost as a fraction (0.20 = +20%) for a given leg count."""
    if legs < 2:
        return 0.0
    if legs in MULTI_BOOST_BY_LEGS:
        return MULTI_BOOST_BY_LEGS[legs]
    # Cap at the highest published rate
    return max(MULTI_BOOST_BY_LEGS.values())

def boosted_multi_odds(raw_multi: float, legs: int) -> float:
    """Apply Sportsbet's leg-based boost to raw multiplied odds."""
    if raw_multi <= 1.0:
        return raw_multi
    return round(raw_multi * (1 + multi_boost_pct(legs)), 2)


# Bonus bet mechanics.
# - Stake is NOT returned on win (typical bonus bet behavior).
# - Max odds Sportsbet allows on bonus bet placement.
# - Excluded from cash-out and most other promos.
BONUS_BET_MAX_ODDS = 50.0
BONUS_BET_STAKE_RETURNED = False

def bonus_bet_return(stake: float, odds: float) -> float:
    """Return on a winning bonus bet: stake × (odds - 1)."""
    if odds < 1.0 or stake <= 0:
        return 0.0
    if odds > BONUS_BET_MAX_ODDS:
        odds = BONUS_BET_MAX_ODDS
    return round(stake * (odds - 1), 2)


# Top-N market availability — Sportsbet offers each when field size meets threshold.
TOP4_MIN_FIELD_SIZE = 8
TOP3_MIN_FIELD_SIZE = 6
TOP2_MIN_FIELD_SIZE = 4

# Approximate Sportsbet over-round on Top 4 fixed prices.
# Use to translate our model top-4 probability into a Sportsbet-equivalent
# price quote (purely indicative — actual prices vary by 5-10%).
TOP4_OVERROUND_PCT = 0.12


# Currently published racing promotions (informational; not used in math).
RACING_PROMOS: list[str] = [
    "money_back_2nd_to_fav_metro_features",
    "early_payout_lead_at_100m",
    "best_tote_plus_sp",
    "extra_place_8plus_runners",
    "multi_boost_2_to_8_legs",
]
