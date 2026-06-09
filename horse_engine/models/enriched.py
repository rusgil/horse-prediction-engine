"""Enriched runner model — all features computed, ready for prediction."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class EnrichedRunner(BaseModel):
    horse_name: str
    barrier: int
    tab_number: int
    jockey: str
    trainer: str
    weight: float

    # ── Form features ──────────────────────────────────────────────────────
    form_score: float               # weighted avg finish position (0–1, higher = better)
    win_rate_career: float          # career win %
    win_rate_distance: float        # win % at races within ±200m of today
    win_rate_track: float           # win % at today's track
    win_rate_class: float           # win % at this class level
    avg_beaten_margin_last5: float  # avg lengths beaten (0 = winner every time)
    best_finish_distance: int       # position in best run at this distance
    days_since_last_run: int        # -1 = first-ever start
    runs_this_prep: int             # 0 = first-up, 1 = second-up, etc.
    class_change: int               # -1 step down, 0 same, +1 step up
    prizemoney_career: int

    # ── Barrier features ───────────────────────────────────────────────────
    barrier_score: float            # historical win% from this barrier at track/dist

    # ── Weight features ────────────────────────────────────────────────────
    weight_vs_field_avg: float      # kg above/below field average (positive = heavier)

    # ── Surface / track condition ──────────────────────────────────────────
    wet_track_record: float         # win % on soft/heavy tracks
    dry_track_record: float         # win % on good/firm tracks
    surface_match_score: float      # 0–1: how well today's condition suits this horse
    track_condition_category: str   # "firm" | "good" | "soft" | "heavy" | "synthetic"

    # ── Trainer features ───────────────────────────────────────────────────
    trainer_overall_rate: float
    trainer_track_rate: float
    trainer_distance_rate: float
    trainer_first_up_rate: float    # relevant if runs_this_prep == 0
    trainer_wet_rate: float
    trainer_jockey_combo_rate: float

    # ── Jockey features ────────────────────────────────────────────────────
    jockey_overall_rate: float
    jockey_track_rate: float
    jockey_distance_rate: float
    jockey_barrier_rate: float      # win% from today's barrier range
    jockey_wins_today: int          # hot hand at this meeting

    # ── Pedigree features ──────────────────────────────────────────────────
    pedigree_distance_match: float  # 0–1: how well sire's ideal range matches distance
    pedigree_wet_score: float       # 0–10 wet track affinity from bloodlines
    pedigree_first_up_score: float  # 0–10 first-up readiness from bloodlines
    sire_name: str
    dam_sire_name: str
    distance_aptitude: str          # "sprint" | "mile" | "middle" | "staying"

    # ── Market features ────────────────────────────────────────────────────
    market_rank: int                # 1 = favourite
    market_implied_prob: float      # 1/odds (overround-free)
    odds_movement: float            # morning_price - current (positive = shortened/steam)
    best_available_odds: float
    overlay: float                  # model_prob - market_implied_prob (computed after model)
    is_steamed: bool                # significant money in
    is_drifted: bool                # significant money out

    # ── Odds movement features (populated from odds_snapshots once data exists) ──
    steam_60: float = 0.0           # odds change T-60min → T-5min (positive = shortened)
    steam_30: float = 0.0           # odds change T-30min → T-5min
    drift_flag: float = 0.0         # 1.0 if opened short and now longer, else 0.0
    odds_velocity: float = 0.0      # average rate of change per minute (last 60 min)
    late_money: float = 0.0         # odds drop in final 15 min (positive = firming late)

    # ── Speed map ──────────────────────────────────────────────────────────
    speed_map_position: str         # "leader" | "on_pace" | "midfield" | "backmarker"
    speed_map_advantage: float      # -1 to +1: benefit of likely run position for this track

    # ── Gear / misc ────────────────────────────────────────────────────────
    gear_changes: list[str]         # ["blinkers_on", "tongue_tie_on"] etc.
    gear_change_score: float        # -1 to +1 impact estimate
    is_first_start: bool
    is_resuming: bool               # first-up after spell
    spell_weeks: Optional[int]      # weeks since last run (None if ongoing prep)
    age_sex_weight_factor: float    # weight-for-age scale adjustment

    # ── Unique signals ─────────────────────────────────────────────────────
    dosage_index: float             # pedigree-derived speed/stamina ratio (>4 = sprinter)
    international_form: bool        # has recent Northern Hemisphere or Singapore form
    stable_form: float              # trainer's last-7-days win rate (stable in form?)
    jockey_booking_significance: float  # 0–1: was a top jockey booked late?
    track_bias_advantage: float     # -1 to +1: does today's rail pos suit this runner style?

    # ── Punters direct stats (new features) ───────────────────────────────
    win_rate_track_distance: float = 0.0  # win % at this exact track+distance combo
    win_rate_condition: float = 0.0       # win % in today's going (good/soft/heavy/firm)
    first_up_win_rate: float = 0.0        # win % when first-up from spell
    second_up_win_rate: float = 0.0       # win % when second-up
    career_place_rate: float = 0.0        # career places / career starts
    wins_last_10: int = 0                 # wins in last 10 starts
    starts_last_10: int = 0              # actual starts in last 10 (may be < 10 for lightly raced)
