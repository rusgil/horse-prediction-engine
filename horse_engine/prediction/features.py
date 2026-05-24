"""
Build the feature vector for the logistic regression model.

Feature order must remain stable — it defines the weight vector indices.

Odds movement features (indices 25-29) are gated by USE_ODDS_MOVEMENT.
Set to False to exclude them and compare model performance with/without.
Only meaningful once odds_snapshots data has been collected for several months.
"""
from __future__ import annotations

from horse_engine.models.enriched import EnrichedRunner

# ── Toggle: include odds movement snapshot features ───────────────────────────
# Set False to train/predict without snapshot features (A/B comparison).
# These features are stubs until odds_snapshots has sufficient history.
USE_ODDS_MOVEMENT_FEATURES = False

FEATURE_NAMES_BASE = [
    "form_score",               # 0
    "win_rate_career",          # 1
    "win_rate_distance",        # 2
    "win_rate_track",           # 3
    "surface_match_score",      # 4
    "barrier_score",            # 5
    "days_since_last_run_norm", # 6
    "runs_in_prep_norm",        # 7
    "class_change",             # 8
    "trainer_overall_rate",     # 9
    "trainer_track_rate",       # 10
    "trainer_jockey_combo_rate",# 11
    "jockey_overall_rate",      # 12
    "jockey_track_rate",        # 13
    "jockey_wins_today_norm",   # 14
    "pedigree_distance_match",  # 15
    "pedigree_wet_score_norm",  # 16
    "market_rank_norm",         # 17
    "odds_movement_norm",       # 18 — single delta (open → current)
    "speed_map_advantage",      # 19
    "gear_change_score",        # 20
    "weight_vs_field_norm",     # 21
    "jockey_booking_significance", # 22
    "stable_form",              # 23
    "track_bias_advantage",     # 24
    "win_rate_track_distance",  # 25 — win % at this exact track+distance (Punters stats)
    "condition_win_rate",       # 26 — win % in today's going (good/soft/heavy/firm)
    "first_up_win_rate",        # 27 — win % when first-up (only relevant if resuming)
]

FEATURE_NAMES_ODDS_MOVEMENT = [
    "steam_60",                 # 25 — odds change T-60min → T-5min
    "steam_30",                 # 26 — odds change T-30min → T-5min
    "drift_flag",               # 27 — 1.0 if opened short, now longer
    "odds_velocity",            # 28 — avg rate of change per minute (last 60min)
    "late_money",               # 29 — odds drop in final 15min
]

FEATURE_NAMES = FEATURE_NAMES_BASE + (FEATURE_NAMES_ODDS_MOVEMENT if USE_ODDS_MOVEMENT_FEATURES else [])
NUM_FEATURES = len(FEATURE_NAMES)

DEFAULT_WEIGHTS_BASE = [
    0.8,   # form_score
    0.6,   # win_rate_career
    0.7,   # win_rate_distance
    0.5,   # win_rate_track
    0.4,   # surface_match_score
    0.3,   # barrier_score
   -0.2,   # days_since_last_run_norm
    0.2,   # runs_in_prep_norm
    0.3,   # class_change
    0.5,   # trainer_overall_rate
    0.6,   # trainer_track_rate
    0.4,   # trainer_jockey_combo_rate
    0.5,   # jockey_overall_rate
    0.6,   # jockey_track_rate
    0.3,   # jockey_wins_today_norm
    0.4,   # pedigree_distance_match
    0.3,   # pedigree_wet_score_norm
    0.9,   # market_rank_norm
    0.5,   # odds_movement_norm
    0.3,   # speed_map_advantage
    0.2,   # gear_change_score
   -0.2,   # weight_vs_field_norm
    0.3,   # jockey_booking_significance
    0.4,   # stable_form
    0.3,   # track_bias_advantage
    0.8,   # win_rate_track_distance — strongest per-horse signal
    0.5,   # condition_win_rate
    0.4,   # first_up_win_rate
]

DEFAULT_WEIGHTS_ODDS_MOVEMENT = [
    0.4,   # steam_60
    0.5,   # steam_30
   -0.3,   # drift_flag (drifting = negative signal)
    0.3,   # odds_velocity
    0.6,   # late_money (late steam = strongest signal)
]

DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_BASE + (DEFAULT_WEIGHTS_ODDS_MOVEMENT if USE_ODDS_MOVEMENT_FEATURES else [])

# Place model defaults — same features, weights tuned for P(position ≤ 3)
# Emphasises consistency, stamina, condition form over explosive speed/barrier
DEFAULT_PLACE_WEIGHTS_BASE = [
    0.7,   # form_score — consistency signal
    0.3,   # win_rate_career — less relevant; place rate matters more
    0.6,   # win_rate_distance
    0.5,   # win_rate_track
    0.6,   # surface_match_score — condition consistency
    0.15,  # barrier_score — less important for placing than winning
   -0.15,  # days_since_last_run_norm
    0.25,  # runs_in_prep_norm
    0.2,   # class_change
    0.5,   # trainer_overall_rate
    0.6,   # trainer_track_rate
    0.3,   # trainer_jockey_combo_rate
    0.4,   # jockey_overall_rate
    0.5,   # jockey_track_rate
    0.25,  # jockey_wins_today_norm
    0.5,   # pedigree_distance_match — stamina for running on
    0.35,  # pedigree_wet_score_norm
    0.7,   # market_rank_norm
    0.4,   # odds_movement_norm
    0.4,   # speed_map_advantage
    0.15,  # gear_change_score
   -0.15,  # weight_vs_field_norm
    0.25,  # jockey_booking_significance
    0.5,   # stable_form — in-form stables place consistently
    0.35,  # track_bias_advantage
    0.7,   # win_rate_track_distance — track+dist consistency
    0.65,  # condition_win_rate — performs in today's going
    0.3,   # first_up_win_rate
]

DEFAULT_PLACE_WEIGHTS = DEFAULT_PLACE_WEIGHTS_BASE + (DEFAULT_WEIGHTS_ODDS_MOVEMENT if USE_ODDS_MOVEMENT_FEATURES else [])


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
        er.win_rate_track_distance,
        er.win_rate_condition,
        er.first_up_win_rate if er.is_resuming else 0.0,
    ]
