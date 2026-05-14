"""
Build the feature vector for the logistic regression model.

Feature order must remain stable — it defines the weight vector indices.
"""
from __future__ import annotations

from horse_engine.models.enriched import EnrichedRunner

FEATURE_NAMES = [
    "form_score",               # 0
    "win_rate_career",          # 1
    "win_rate_distance",        # 2
    "win_rate_track",           # 3
    "surface_match_score",      # 4
    "barrier_score",            # 5
    "days_since_last_run_norm", # 6  — normalised: 0=optimal freshness, 1=stale
    "runs_in_prep_norm",        # 7  — 0=first-up, 1=6+ runs (fully wound up vs needing run)
    "class_change",             # 8  — -1/0/1
    "trainer_overall_rate",     # 9
    "trainer_track_rate",       # 10
    "trainer_jockey_combo_rate",# 11
    "jockey_overall_rate",      # 12
    "jockey_track_rate",        # 13
    "jockey_wins_today_norm",   # 14 — hot hand: 0–1
    "pedigree_distance_match",  # 15
    "pedigree_wet_score_norm",  # 16 — 0–1 (from 0–10)
    "market_rank_norm",         # 17 — 1/market_rank (fav=1.0, 2nd fav=0.5 etc)
    "odds_movement_norm",       # 18 — sigmoid of odds_movement
    "speed_map_advantage",      # 19 — -1 to +1
    "gear_change_score",        # 20
    "weight_vs_field_norm",     # 21 — -1 to +1
    "jockey_booking_significance", # 22
    "stable_form",              # 23
    "track_bias_advantage",     # 24
]

NUM_FEATURES = len(FEATURE_NAMES)

# Default weights (priors) — these will be overridden after retraining
DEFAULT_WEIGHTS = [
    0.8,   # form_score
    0.6,   # win_rate_career
    0.7,   # win_rate_distance
    0.5,   # win_rate_track
    0.4,   # surface_match_score
    0.3,   # barrier_score
   -0.2,   # days_since_last_run_norm (fresh = good)
    0.2,   # runs_in_prep_norm (horse that's had a run or two = usually better)
    0.3,   # class_change
    0.5,   # trainer_overall_rate
    0.6,   # trainer_track_rate
    0.4,   # trainer_jockey_combo_rate
    0.5,   # jockey_overall_rate
    0.6,   # jockey_track_rate
    0.3,   # jockey_wins_today_norm
    0.4,   # pedigree_distance_match
    0.3,   # pedigree_wet_score_norm
    0.9,   # market_rank_norm (market is a strong prior)
    0.5,   # odds_movement_norm
    0.3,   # speed_map_advantage
    0.2,   # gear_change_score
   -0.2,   # weight_vs_field_norm (heavier = slight disadvantage)
    0.3,   # jockey_booking_significance
    0.4,   # stable_form
    0.3,   # track_bias_advantage
]


def build_feature_vector(er: EnrichedRunner) -> list[float]:
    import math

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    # Days since last run: 14–28 days = optimal (0.0), very long/short = worse
    days = er.days_since_last_run
    if days < 0:
        days_norm = 0.3  # first-ever start
    elif days < 7:
        days_norm = 0.3  # too fresh, not fit
    elif 14 <= days <= 28:
        days_norm = 0.0  # optimal
    elif days <= 60:
        days_norm = 0.2  # slightly stale
    else:
        days_norm = 0.8  # long spell, needs a run

    # Runs in prep: 0=first-up (0.0), 2=usually peaking (~0.5), 6+=fully wound (1.0)
    runs_norm = min(er.runs_this_prep / 6.0, 1.0)

    # Market rank normalised
    market_rank_norm = 1.0 / max(er.market_rank, 1)

    # Odds movement sigmoid
    odds_mov_norm = sigmoid(er.odds_movement * 0.5)

    # Jockey hot hand
    jockey_wins_today_norm = min(er.jockey_wins_today / 3.0, 1.0)

    # Weight vs field (-2kg to +2kg → -1 to +1)
    weight_norm = max(-1.0, min(1.0, er.weight_vs_field_avg / 2.0))

    return [
        er.form_score,
        er.win_rate_career,
        er.win_rate_distance,
        er.win_rate_track,
        er.surface_match_score,
        er.barrier_score,
        days_norm,
        runs_norm,
        float(er.class_change),
        er.trainer_overall_rate / 100.0,  # convert % to 0–1
        er.trainer_track_rate / 100.0,
        er.trainer_jockey_combo_rate / 100.0,
        er.jockey_overall_rate / 100.0,
        er.jockey_track_rate / 100.0,
        jockey_wins_today_norm,
        er.pedigree_distance_match,
        er.pedigree_wet_score / 10.0,
        market_rank_norm,
        odds_mov_norm,
        er.speed_map_advantage,
        er.gear_change_score,
        weight_norm,
        er.jockey_booking_significance,
        er.stable_form,
        er.track_bias_advantage,
    ]
