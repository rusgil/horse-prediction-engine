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
# Enabled once odds_snapshots has sufficient history (weeks of pre-race data).
USE_ODDS_MOVEMENT_FEATURES = True

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
    "win_rate_track_distance",  # 25 — win % at this exact track+distance (Racing Australia stats)
    "condition_win_rate",       # 26 — win % in today's going (good/soft/heavy/firm)
    "first_up_win_rate",        # 27 — win % when first-up (only relevant if resuming)
    "going_preference",         # 28 — signed differential: how much better on today's going vs opposite
    "trainer_first_up_fitness", # 29 — trainer_first_up_rate × is_resuming (specialised trainer signal)
    "class_drop_in_form",       # 30 — class drop × form_score (dropping in class while running well)
    "career_place_rate",        # 31
    "second_up_win_rate",       # 32
    "market_implied_prob",      # 33
    "avg_beaten_margin_norm",   # 34
    "jockey_distance_rate",     # 35
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
    0.3,   # win_rate_distance — reduced: noisy below 3 starts (ablation +6.4%)
    0.5,   # win_rate_track
    0.4,   # surface_match_score
    0.3,   # barrier_score
   -0.2,   # days_since_last_run_norm
    0.0,   # runs_in_prep_norm — zeroed: trained sign was inverted vs domain knowledge
    0.3,   # class_change
    0.5,   # trainer_overall_rate
    0.6,   # trainer_track_rate
    0.4,   # trainer_jockey_combo_rate
    0.5,   # jockey_overall_rate
    0.6,   # jockey_track_rate
    0.3,   # jockey_wins_today_norm
    0.4,   # pedigree_distance_match
    0.3,   # pedigree_wet_score_norm
    0.35,  # market_rank_norm
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
    0.6,   # going_preference — wet/dry differential signed for today's going
    0.5,   # trainer_first_up_fitness — trainer specialisation when horse is resuming
    0.4,   # class_drop_in_form — class relief for a horse already running well
    0.4,   # career_place_rate — consistency signal
    0.3,   # second_up_win_rate — second-up peak in Australian racing
    0.0,   # market_implied_prob — available but handled via market_rank_norm
    0.2,   # avg_beaten_margin_norm — close-up horses
    0.2,   # jockey_distance_rate — distance specialist jockey
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
    0.3,   # win_rate_distance — reduced: noisy below 3 starts
    0.5,   # win_rate_track
    0.6,   # surface_match_score — condition consistency
    0.15,  # barrier_score — less important for placing than winning
   -0.15,  # days_since_last_run_norm
    0.0,   # runs_in_prep_norm — zeroed: inverted signal
    0.2,   # class_change
    0.5,   # trainer_overall_rate
    0.6,   # trainer_track_rate
    0.3,   # trainer_jockey_combo_rate
    0.4,   # jockey_overall_rate
    0.5,   # jockey_track_rate
    0.25,  # jockey_wins_today_norm
    0.5,   # pedigree_distance_match — stamina for running on
    0.35,  # pedigree_wet_score_norm
    0.3,   # market_rank_norm
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
    0.65,  # going_preference — stronger signal for placing (conditions matter more)
    0.55,  # trainer_first_up_fitness
    0.35,  # class_drop_in_form
    0.5,   # career_place_rate — strong for placing
    0.35,  # second_up_win_rate
    0.0,   # market_implied_prob
    0.25,  # avg_beaten_margin_norm
    0.2,   # jockey_distance_rate
]

DEFAULT_PLACE_WEIGHTS = DEFAULT_PLACE_WEIGHTS_BASE + (DEFAULT_WEIGHTS_ODDS_MOVEMENT if USE_ODDS_MOVEMENT_FEATURES else [])

# Exotic model defaults — tuned for trifecta box coverage
# Emphasises consistency and runs-in-prep over barrier/win-rate peak signals
DEFAULT_EXOTIC_WEIGHTS_BASE = [
    0.75,  # form_score
    0.25,  # win_rate_career
    0.3,   # win_rate_distance — reduced: noisy below 3 starts
    0.55,  # win_rate_track
    0.65,  # surface_match_score
    0.10,  # barrier_score — less important in exotics
   -0.10,  # days_since_last_run_norm
    0.15,  # runs_in_prep_norm — positive for exotics: seasoned prep = consistent 3-runner coverage
    0.15,  # class_change
    0.55,  # trainer_overall_rate
    0.65,  # trainer_track_rate
    0.30,  # trainer_jockey_combo_rate
    0.40,  # jockey_overall_rate
    0.50,  # jockey_track_rate
    0.20,  # jockey_wins_today_norm
    0.55,  # pedigree_distance_match
    0.40,  # pedigree_wet_score_norm
    0.3,   # market_rank_norm
    0.35,  # odds_movement_norm
    0.45,  # speed_map_advantage
    0.15,  # gear_change_score
   -0.10,  # weight_vs_field_norm
    0.20,  # jockey_booking_significance
    0.55,  # stable_form
    0.40,  # track_bias_advantage
    0.75,  # win_rate_track_distance
    0.70,  # condition_win_rate
    0.25,  # first_up_win_rate
    0.60,  # going_preference
    0.45,  # trainer_first_up_fitness
    0.35,  # class_drop_in_form
    0.45,  # career_place_rate
    0.3,   # second_up_win_rate
    0.0,   # market_implied_prob
    0.2,   # avg_beaten_margin_norm
    0.15,  # jockey_distance_rate
]

DEFAULT_EXOTIC_WEIGHTS = DEFAULT_EXOTIC_WEIGHTS_BASE + (DEFAULT_WEIGHTS_ODDS_MOVEMENT if USE_ODDS_MOVEMENT_FEATURES else [])


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

    # Going preference: signed differential — how much better on today's going vs opposite.
    # Wet track day → wet_record - dry_record (positive = wet-tracker, good today)
    # Dry track day → dry_record - wet_record (positive = dry-tracker, good today)
    # Capped to [-1, 1]; 0.0 when no form on either surface.
    cond = er.track_condition_category
    if cond in ("soft", "heavy"):
        going_pref = er.wet_track_record - er.dry_track_record
    elif cond in ("good", "firm", "synthetic"):
        going_pref = er.dry_track_record - er.wet_track_record
    else:
        going_pref = 0.0
    going_pref = max(-1.0, min(1.0, going_pref))

    # trainer_first_up_fitness: trainer's first-up strike rate × is_resuming flag
    trainer_first_up_fitness = (er.trainer_first_up_rate / 100.0) if er.is_resuming else 0.0

    # class_drop_in_form: class relief amplified when horse is already running well
    # class_change = -1 means dropping down; form_score in [0,1]
    class_drop_in_form = max(0.0, -float(er.class_change)) * er.form_score

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
        going_pref,
        trainer_first_up_fitness,
        class_drop_in_form,
        er.career_place_rate,
        er.second_up_win_rate,
        er.market_implied_prob,
        max(0.0, 1.0 - min(er.avg_beaten_margin_last5 / 10.0, 1.0)),  # avg_beaten_margin_norm: higher = closer up
        er.jockey_distance_rate / 100.0,
        er.steam_60,
        er.steam_30,
        er.drift_flag,
        er.odds_velocity,
        er.late_money,
    ]
