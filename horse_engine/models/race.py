"""Core Pydantic domain models for horse racing data."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class FormStart(BaseModel):
    """One historical race start."""
    date: str
    track: str
    distance: int
    track_condition: str       # e.g. "Good 4", "Soft 6", "Heavy 8"
    barrier: int
    weight: float
    jockey: str
    position: int
    finishers: int             # total runners
    beaten_margin: float       # lengths (0 = winner)
    race_class: str            # e.g. "BM64", "G1", "Maiden"
    prize_money: int
    starting_price: Optional[float] = None
    sectional_last600: Optional[float] = None  # last 600m split in seconds
    in_running: Optional[str] = None           # "Settled 4th, 3W turn"
    comment: Optional[str] = None


class TrainerStats(BaseModel):
    name: str
    win_rate_overall: float       # %
    win_rate_track: float         # % at today's track
    win_rate_distance: float      # % at today's distance class
    win_rate_first_up: float      # % when horse is resuming
    win_rate_second_up: float
    win_rate_wet: float           # % on soft/heavy
    prizemoney_season: int        # total prize money this season
    runners_season: int
    wins_season: int


class JockeyStats(BaseModel):
    name: str
    win_rate_overall: float
    win_rate_track: float
    win_rate_distance: float
    win_rate_barrier_low: float   # barriers 1-5
    win_rate_barrier_mid: float   # barriers 6-10
    win_rate_barrier_wide: float  # barriers 11+
    wins_today: int               # "hot hand" — wins at this meeting so far
    prizemoney_season: int
    wins_season: int
    trainer_jockey_combo_rate: float  # win % of this exact trainer+jockey combo


class PedigreeProfile(BaseModel):
    sire: str
    dam: str
    dam_sire: str
    distance_aptitude: str        # "sprint" | "mile" | "middle" | "staying"
    distance_min: int             # minimum ideal distance (m)
    distance_max: int             # maximum ideal distance (m)
    wet_track_score: float        # 0–10: 10 = loves wet
    first_up_score: float         # 0–10: 10 = first-up specialist
    second_up_score: float        # 0–10
    on_pace_tendency: float       # 0–10: 10 = loves to lead
    stamina_index: float          # 0–10: 10 = pure stayer
    brilliance_index: float       # 0–10: 10 = explosive sprinter


class Runner(BaseModel):
    """A horse competing in a race."""
    barrier: int
    tab_number: int               # scratching-safe number
    horse_name: str
    age: int
    sex: str                      # "G" gelding, "M" mare, "C" colt, "F" filly, "H" horse
    colour: str
    weight: float                 # kg
    jockey: str
    trainer: str
    country: str                  # "AUS", "NZ", "IRE" etc
    career_starts: int
    career_wins: int
    career_places: int

    last_10_starts: list[FormStart] = []
    trainer_stats: Optional[TrainerStats] = None
    jockey_stats: Optional[JockeyStats] = None
    pedigree: Optional[PedigreeProfile] = None

    # Market
    fixed_win_odds: Optional[float] = None
    tote_win_odds: Optional[float] = None
    tote_place_odds: Optional[float] = None
    sportsbet_odds: Optional[float] = None
    ladbrokes_odds: Optional[float] = None
    best_available_odds: Optional[float] = None
    odds_opening: Optional[float] = None
    market_rank: Optional[int] = None  # 1 = favourite


class Race(BaseModel):
    race_id: str                  # unique: {date}_{venue}_{race_num}
    date: str
    venue: str
    state: str
    race_number: int
    race_name: str
    race_class: str
    distance: int
    track_condition: str
    rail_position: str
    prize_money: int
    scheduled_time: str
    race_type: str                # "R" thoroughbred, "H" harness, "G" greyhound

    runners: list[Runner] = []


class Meeting(BaseModel):
    date: str
    venue: str
    state: str
    track_condition: str
    races: list[Race] = []
