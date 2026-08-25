"""SQLAlchemy ORM models and DB helpers."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, Float, Integer, String, Text, Boolean, DateTime, Index, UniqueConstraint, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from horse_engine.config import settings


def _make_engine():
    url = settings.async_database_url
    if url.startswith("postgresql"):
        return create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=1800,
        )
    return create_async_engine(url, echo=False)

engine = _make_engine()
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def sched_to_utc_naive(value) -> "datetime | None":
    """Parse a scheduled_time string to NAIVE UTC for guard comparisons.

    THE bug this kills (2026-07-28): race times arrive as "…T13:40:00+10:00";
    `fromisoformat(...).replace(tzinfo=None)` kept the LOCAL clock face, so
    `utcnow() > sched` believed every race was ~10h in the future — letting
    post-race re-enrichments rewrite mutable AND history snapshots ("MR
    CACCIATORE became our pick after winning"). Z-format strings were fine,
    which is why the corruption was intermittent (depends on which feed
    stamped the times that day). ALWAYS convert via the offset.
    """
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class PredictionIntegrityRow(Base):
    """Independent post-race tamper audit. Captures a race's model 1st/2nd/3rd
    picks ONCE just after it jumps (the pre-race record), then re-reads at
    11pm; a mismatch means the frozen prediction changed after the race
    started — the write-guard trigger failed. Belt-and-braces on that trigger."""
    __tablename__ = "prediction_integrity"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True, unique=True)
    race_date = Column(String, index=True)
    scheduled_time = Column(String, nullable=True)
    jump_top1 = Column(String, nullable=True)   # rank-1 horse at/just-after jump
    jump_top2 = Column(String, nullable=True)
    jump_top3 = Column(String, nullable=True)
    jump_captured_at = Column(DateTime, nullable=True)
    baseline_field = Column(Text, nullable=True)  # JSON: all runner names at baseline (detects late additions)
    eod_top1 = Column(String, nullable=True)     # rank-1 re-read at 11pm
    eod_top2 = Column(String, nullable=True)
    eod_top3 = Column(String, nullable=True)
    eod_captured_at = Column(DateTime, nullable=True)
    mismatch = Column(Boolean, default=False, nullable=True, index=True)
    detail = Column(Text, nullable=True)
    # Sharp-flag integrity — same idea as the 1·2·3 check, for the rank-1 pick's
    # Sharp flag: baselined pre-jump, re-read from frozen history at 23:15. A
    # post-jump change is a breach (the flag is a performance-tracked feature).
    jump_is_sharp = Column(Boolean, nullable=True)
    eod_is_sharp = Column(Boolean, nullable=True)
    sharp_mismatch = Column(Boolean, default=False, nullable=True, index=True)


class RaceExoticDividendRow(Base):
    """Real tote exotic dividends captured post-race (quinella first-class,
    plus exacta/trifecta/first-four when available). Source of truth for
    measuring exotic-bet EV — win/place odds live elsewhere. One row per
    race_id, upserted; nulls where a pool wasn't returned by the source."""
    __tablename__ = "race_exotic_dividends"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True, unique=True)
    quinella = Column(Float, nullable=True)
    exacta = Column(Float, nullable=True)
    trifecta = Column(Float, nullable=True)
    first_four = Column(Float, nullable=True)
    source = Column(String, nullable=True)          # "tab" | "ra"
    captured_at = Column(DateTime, default=datetime.utcnow)


class RacePredictionRow(Base):
    __tablename__ = "race_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True)          # {date}_{venue}_{race_num}
    date = Column(String, index=True)
    venue = Column(String)
    state = Column(String)
    race_number = Column(Integer)
    race_name = Column(String)
    race_class = Column(String)
    distance = Column(Integer)
    track_condition = Column(String)
    prize_money = Column(Integer)
    scheduled_time = Column(String)
    field_size = Column(Integer)
    runners_json = Column(Text)                    # serialised list[RunnerPrediction]
    enriched_at = Column(DateTime, default=datetime.utcnow)


class RunnerPredictionRow(Base):
    __tablename__ = "runner_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True)
    horse_name = Column(String, index=True)
    tab_number = Column(Integer)
    barrier = Column(Integer)
    jockey = Column(String)
    trainer = Column(String)
    weight = Column(Float)

    win_probability = Column(Float)
    place_probability = Column(Float)
    # Raw softmax output from HorseModel — the pipeline stores this as the
    # *input* to isotonic calibration (which now runs first). Multipliers
    # then transform the calibrated value into win_probability. Kept as its
    # own column so the nightly calibration curve can be re-fit on
    # raw→actual (monotone by construction), avoiding the plateau failure
    # mode where post-multiplier win_probability was fed back into isotonic.
    win_prob_raw = Column(Float, nullable=True)
    model_rank = Column(Integer)
    market_rank = Column(Integer)
    overlay = Column(Float)
    best_available_odds = Column(Float)
    value_rating = Column(Float)         # composite value score

    narrative = Column(Text, nullable=True)
    key_flags = Column(Text)             # JSON list of flag strings
    enriched_json = Column(Text)         # full EnrichedRunner JSON
    place_model_rank = Column(Integer, nullable=True)
    exotic_model_rank = Column(Integer, nullable=True)
    scheduled_time = Column(String, nullable=True)
    enriched_at = Column(DateTime, default=datetime.utcnow)
    cancelled = Column(Boolean, default=False, nullable=True)

    venue = Column(String, nullable=True)
    state = Column(String, nullable=True)
    race_number = Column(Integer, nullable=True)
    race_name = Column(String, nullable=True)
    distance = Column(Integer, nullable=True)
    track_condition = Column(String, nullable=True)
    field_size = Column(Integer, nullable=True)
    prize_money = Column(Integer, nullable=True)
    rail_position = Column(String, nullable=True)
    class_change = Column(Integer, nullable=True)


class ModelWeightCandidateRow(Base):
    """Candidate model weights awaiting human review.

    Follows the release-gate pattern established by WinCalibrationCurveRow.
    Retrain endpoints write here with status='candidate'; an operator
    reviews the backtest and either promotes (copies weights into the
    matching active table — model_weights / place_model_weights /
    exotic_model_weights) or rejects. No retrain path is allowed to
    write to the active tables directly — see
    [[feedback_model_release_process]].

    One row per (batch_id, feature_name). All rows in the same batch
    represent one candidate model. batch_id is a UUID generated by
    the retrain sweep.
    """
    __tablename__ = "model_weight_candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String, nullable=False, index=True)
    # 'win' | 'place' | 'exotic' — routes promotion to the correct
    # active table.
    model_type = Column(String, nullable=False, index=True)
    feature_name = Column(String, nullable=False)
    weight = Column(Float, nullable=False)

    # Batch-level metadata is DUPLICATED across every row in the batch
    # so a single SELECT recovers everything. Cheap given ~41 features
    # per model.
    # 'candidate' | 'active' | 'archived' | 'rejected'
    status = Column(String, default="candidate", nullable=False, index=True)
    sample_days = Column(Integer, nullable=True)
    sample_size = Column(Integer, nullable=True)
    holdout_days = Column(Integer, nullable=True)
    best_window = Column(Integer, nullable=True)
    training_window_start = Column(String, nullable=True)   # ISO date
    training_window_end = Column(String, nullable=True)     # ISO date
    # Backtest result JSON — { races, wins, roi, mse, delta_vs_active, ... }.
    backtest_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_note = Column(Text, nullable=True)


class SecurityFindingRow(Base):
    """Security events reported by the autonomous security agents — Dr Evil
    (red-team: vulnerability + anomaly findings) and Thor (blue-team: real-time
    detections + verifications). Read surface for the admin dashboard Security
    tab: agents POST findings, a human triages via the status endpoint.

    Append-mostly event log. Nothing here grants any capability, and callers
    MUST mask secrets before posting — this is not a place for live credentials.
    """
    __tablename__ = "security_findings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent = Column(String, nullable=False, index=True)      # 'dr_evil' | 'thor'
    severity = Column(String, nullable=False, index=True)   # critical|high|medium|low|info
    category = Column(String, nullable=True)                # sqli|auth|anomaly|cve|...
    title = Column(String, nullable=False)
    target = Column(String, nullable=True)                  # host / route / file
    detail = Column(Text, nullable=True)                    # description + MASKED evidence
    remediation = Column(Text, nullable=True)               # suggested fix
    threat = Column(Text, nullable=True)                    # Dr Evil's villainous blast-radius line (attack -> capability -> consequence)
    # open | verified | fixed | dismissed
    status = Column(String, default="open", nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


class AutoPromotionRow(Base):
    """Audit trail for the guarded auto-promotion pipeline. One row per
    auto-promotion: which candidate went live, the prior batch (for rollback),
    the backtest margin that cleared the guardrails, and the live-regression
    outcome. Only written when AUTO_PROMOTE_ENABLED — the human gate otherwise.
    """
    __tablename__ = "auto_promotions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String, nullable=False, index=True)  # win | place | exotic
    batch_id = Column(String, nullable=False)                # the promoted candidate
    prior_batch_id = Column(String, nullable=True)           # for auto-rollback
    delta_pp = Column(Float, nullable=True)                  # OOS margin vs incumbent at promotion
    races = Column(Integer, nullable=True)                   # backtest sample size
    expected_hit_rate = Column(Float, nullable=True)         # candidate hit rate at promotion
    # active | superseded | rolled_back
    status = Column(String, default="active", nullable=False, index=True)
    promoted_at = Column(DateTime, default=datetime.utcnow, index=True)
    rolled_back_at = Column(DateTime, nullable=True)
    note = Column(Text, nullable=True)


class ModelWeightRow(Base):
    __tablename__ = "model_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, unique=True)
    weight = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


class PlaceModelWeightRow(Base):
    __tablename__ = "place_model_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, unique=True)
    weight = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ExoticModelWeightRow(Base):
    __tablename__ = "exotic_model_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, unique=True)
    weight = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


class HistoricalResultRow(Base):
    __tablename__ = "historical_results"
    # Uniqueness on (race_id, LOWER(horse_name)) enforced at init_db
    # migration time — see the CREATE UNIQUE INDEX statement there.
    # LOWER() collapses "Subarashii Express" and "SUBARASHII EXPRESS"
    # (RA is inconsistent about case between Acceptances and Results).

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True)
    horse_name = Column(String, index=True)
    position = Column(Integer)
    beaten_margin = Column(Float)
    winner = Column(Boolean)
    placed = Column(Boolean)
    starting_price = Column(Float, nullable=True)
    feature_vector_json = Column(Text, nullable=True)   # for retraining

    # Runner context — populated at seed time for stats computation
    jockey = Column(String, nullable=True, index=True)
    trainer = Column(String, nullable=True, index=True)
    venue = Column(String, nullable=True, index=True)
    state = Column(String, nullable=True)
    distance = Column(Integer, nullable=True)
    track_condition = Column(String, nullable=True)
    barrier = Column(Integer, nullable=True)
    tab_number = Column(Integer, nullable=True)
    weight = Column(Float, nullable=True)
    age = Column(Integer, nullable=True)
    sex = Column(String, nullable=True)
    race_class = Column(String, nullable=True)
    prize_money = Column(Integer, nullable=True)
    field_size = Column(Integer, nullable=True)
    race_number = Column(Integer, nullable=True)

    recorded_at = Column(DateTime, default=datetime.utcnow)


class BacktestResultRow(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True)
    race_date = Column(String, index=True)
    venue = Column(String)
    horse_name = Column(String)
    model_rank = Column(Integer)
    win_probability = Column(Float)
    starting_price = Column(Float, nullable=True)
    actual_position = Column(Integer, nullable=True)
    winner = Column(Boolean)
    source = Column(String)   # "backtest" (retroactive) or "live" (predicted before race)
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestStateRow(Base):
    """Single-row table tracking backtest progress across restarts."""
    __tablename__ = "backtest_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_date = Column(String)
    end_date = Column(String)
    last_completed_date = Column(String, nullable=True)  # last day fully written
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CalibrationRow(Base):
    __tablename__ = "calibrations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ran_at = Column(DateTime, default=datetime.utcnow, index=True)
    holdout_days = Column(Integer)
    best_window = Column(Integer)           # winning training window in days
    win_rate = Column(Float)
    place_rate = Column(Float)
    value_roi = Column(Float)               # SP-based value bet ROI (per $1)
    value_bets = Column(Integer)
    total_races = Column(Integer)
    drift_flag = Column(Boolean, default=False)
    drift_reason = Column(Text, nullable=True)
    all_results_json = Column(Text)         # JSON list of per-window stats
    # BAO-based value bet ROI (2026-07-15). Uses RunnerPredictionHistoryRow.
    # best_available_odds when populated, falls back to starting_price so
    # rows with sparse BAO (pre-2026-07-15 odds-snapshot fix) still count
    # against the same denominator. bao_coverage_pct tells the reviewer
    # how much of the number is real BAO vs SP fallback.
    value_roi_bao = Column(Float, nullable=True)
    bao_coverage_pct = Column(Float, nullable=True)


class WinCalibrationCurveRow(Base):
    """Persisted isotonic calibration curve for the win model output.

    Under the release-gate architecture (2026-07-15), nightly rebuilds
    now INSERT new rows with status='candidate' rather than overwriting
    the active curve. A backtest is computed against the incumbent
    (status='active') curve and stored in backtest_json. A follow-up
    row is created for human approval — promotion flips the candidate
    to status='active' and the previous active to status='archived'.
    Never trust an unattended cron to promote a model artefact.

    curve_json holds a list of (input_pct, calibrated_pct) breakpoints
    in monotone-non-decreasing order — see
    horse_engine.prediction.output_calibration for the format.
    """
    __tablename__ = "win_calibration_curve"

    id = Column(Integer, primary_key=True, autoincrement=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    sample_days = Column(Integer)          # holdout window used
    sample_size = Column(Integer)          # total (pct, won) pairs fit
    curve_json = Column(Text)              # JSON list of [input_pct, calibrated_pct]
    # 'active' | 'candidate' | 'archived' | 'rejected'
    status = Column(String, default="active", nullable=True, index=True)
    # JSON blob with candidate-vs-incumbent backtest metrics:
    # {"races": N, "candidate_wins": X, "incumbent_wins": Y, "candidate_win_pct": ..., ...}
    backtest_json = Column(Text, nullable=True)
    # Populated when a human promotes or rejects. Free-form audit trail.
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_note = Column(Text, nullable=True)


class OddsSnapshotRow(Base):
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True)
    horse_name = Column(String)
    snapshotted_at = Column(DateTime, default=datetime.utcnow, index=True)
    minutes_to_jump = Column(Integer, nullable=True)   # negative = post-jump
    win_odds = Column(Float, nullable=True)
    place_odds = Column(Float, nullable=True)
    source = Column(String, default="flucs")           # "tote" or "flucs"


class VenueCoordinateRow(Base):
    __tablename__ = "venue_coordinates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    venue_key = Column(String, unique=True, index=True)  # "VenueName|STATE"
    latitude = Column(Float)
    longitude = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ExoticBacktestRow(Base):
    __tablename__ = "exotic_backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ran_at = Column(DateTime, default=datetime.utcnow, index=True)
    best_window = Column(Integer)
    best_holdout_box_hit_rate = Column(Float)
    holdout_races = Column(Integer)
    holdout_days = Column(Integer)
    results_json = Column(Text)   # full response JSON


class RunnerPredictionHistoryRow(Base):
    """
    Immutable snapshot of pre-race predictions — written once, never updated or deleted.
    Performance endpoints (track record, premium stats) read exclusively from this table.
    Only rows where enriched_at < scheduled_time are written here, ensuring no post-race
    data contaminates historical accuracy reporting.
    """
    __tablename__ = "runner_prediction_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True)
    horse_name = Column(String, index=True)
    tab_number = Column(Integer)
    barrier = Column(Integer)
    jockey = Column(String)
    trainer = Column(String)
    weight = Column(Float)

    win_probability = Column(Float)
    place_probability = Column(Float)
    # See RunnerPredictionRow.win_prob_raw. Copied verbatim by the pre-race
    # snapshot cron so the calibration training set can be built purely
    # from history without touching mutable state.
    win_prob_raw = Column(Float, nullable=True)
    model_rank = Column(Integer, index=True)
    market_rank = Column(Integer)
    overlay = Column(Float)
    best_available_odds = Column(Float)
    value_rating = Column(Float)

    narrative = Column(Text, nullable=True)
    key_flags = Column(Text)
    enriched_json = Column(Text)
    place_model_rank = Column(Integer, nullable=True)
    exotic_model_rank = Column(Integer, nullable=True)
    scheduled_time = Column(String, nullable=True)
    enriched_at = Column(DateTime)          # when the prediction was originally made
    cancelled = Column(Boolean, default=False, nullable=True)

    venue = Column(String, nullable=True)
    state = Column(String, nullable=True)
    race_number = Column(Integer, nullable=True)
    race_name = Column(String, nullable=True)
    distance = Column(Integer, nullable=True)
    track_condition = Column(String, nullable=True)
    field_size = Column(Integer, nullable=True)
    prize_money = Column(Integer, nullable=True)
    rail_position = Column(String, nullable=True)
    class_change = Column(Integer, nullable=True)
    model_score = Column(Float, nullable=True)
    source = Column(String, default="live", nullable=True)  # "live" | "validation"

    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)  # when history was written
    batch_id = Column(String, nullable=True, index=True)    # UUID shared by all runners in one enrichment run
    # Data-quality flag. True marks rows written during a period known to
    # be corrupted (pipeline chaos, odds bug, WAF outage). Retrains and
    # backtests filter these out by default. See init_db migrations for
    # the specific windows currently marked.
    contaminated = Column(Boolean, default=False, nullable=True, index=True)
    # Sharp eligibility frozen at snapshot time. Set on rank-1 picks
    # using the Sharp gate active when the row was written. Once written
    # it is never recomputed — future gate refinements only affect new
    # races. /api/performance?sharp=true reads this flag instead of
    # re-evaluating the current gate against historical data.
    is_sharp = Column(Boolean, nullable=True, index=True)


class RaCalendarCacheRow(Base):
    """Persistent cache for RA Calendar.aspx HTML + the slug→ra_key map
    it produces. Survives Railway redeploys so we don't fanout 32 RA
    requests on every cold start. Each row is one (date, state).
    """
    __tablename__ = "ra_calendar_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_date = Column(String, index=True, nullable=False)  # YYYY-MM-DD
    state = Column(String, nullable=False)                  # NSW / VIC / etc.
    meetings_json = Column(Text, nullable=False)            # list of meeting dicts
    slug_to_key_json = Column(Text, nullable=False)         # {slug: ra_key}
    fetched_at = Column(DateTime, default=datetime.utcnow, index=True)


class RaceConditionsRow(Base):
    """Per-race going + weather, for analysis. `track_rating` is the numeric
    going (1-2 Firm, 3-4 Good, 5-7 Soft, 8-10 Heavy) — filter/join on the number
    instead of the string (e.g. Good = track_rating IN (3,4)). Weather is the
    day's forecast for the venue, captured at enrich time."""
    __tablename__ = "race_conditions"

    race_id = Column(String, primary_key=True)
    venue = Column(String, nullable=True)
    state = Column(String, nullable=True)
    race_date = Column(String, index=True, nullable=True)   # YYYY-MM-DD
    track_condition = Column(String, nullable=True)         # e.g. "Good 4"
    track_rating = Column(Integer, index=True, nullable=True)  # 1-10
    rain_mm = Column(Float, nullable=True)
    temp_max = Column(Float, nullable=True)
    temp_min = Column(Float, nullable=True)
    wind_kmh = Column(Float, nullable=True)
    weather_condition = Column(String, nullable=True)
    weather_icon = Column(String, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class VenueBadgeRow(Base):
    """Dynamic per-venue quality badge (gold/silver), recomputed weekly from a
    rolling window of rank-1 results. gold = win% > 40, silver = win% > 30."""
    __tablename__ = "venue_badges"

    venue = Column(String, primary_key=True)
    state = Column(String, nullable=True)
    badge = Column(String, nullable=True)          # 'gold' | 'silver' | None
    win_pct = Column(Float, nullable=True)
    place_pct = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=True)
    window_days = Column(Integer, nullable=True)
    computed_at = Column(DateTime, default=datetime.utcnow)


class WeeklyReviewFollowUpRow(Base):
    """Deferred follow-up on a weekly-review suggestion — records what was
    applied, what to measure, when to re-check, and (once measured) the
    outcome + suggested next action.

    Populated by /api/admin/weekly-review-followup/create and read by
    the Follow-Ups dashboard tab. The Sunday cron
    _scheduled_weekly_review_followup_check walks rows where
    scheduled_for <= today AND measured_at IS NULL, runs the specific
    measurement, and fills in the result columns."""
    __tablename__ = "weekly_review_followups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    scheduled_for = Column(String, index=True)         # 'YYYY-MM-DD' — when to measure

    # What
    title = Column(String)
    context_md = Column(Text)                          # what the review said
    action_md = Column(Text)                           # what we shipped

    # How to measure
    measurement_type = Column(String)                  # e.g. 'market_disagreed_losses_7d'
    baseline_value = Column(Float)                     # the number at time of applying
    target_below = Column(Float, nullable=True)       # want measured < target_below
    target_above = Column(Float, nullable=True)       # or measured > target_above

    # Filled by the cron
    measured_at = Column(DateTime, nullable=True, index=True)
    measured_value = Column(Float, nullable=True)
    verdict = Column(String, nullable=True)            # 'fixed' | 'partial' | 'unchanged' | 'worse'
    next_action_md = Column(Text, nullable=True)


class QualityCheckRow(Base):
    """Nightly data-integrity report — one row per date. Populated by
    _scheduled_quality_check (04:00 AEST). Payload is the full report
    (critical/warning/info categorized findings). Queryable history so
    we can see when a specific integrity issue first appeared."""
    __tablename__ = "quality_checks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    check_date = Column(String, index=True)                # 'YYYY-MM-DD' the report is about
    ran_at = Column(DateTime, default=datetime.utcnow, index=True)
    critical_count = Column(Integer, default=0)
    warning_count = Column(Integer, default=0)
    info_count = Column(Integer, default=0)
    report_json = Column(Text)                             # full response body


class ResponseCacheRow(Base):
    """Persistent response-cache snapshot. Keyed by name (e.g. 'edge').
    Used to survive Railway redeploys: the new container hydrates the
    in-memory cache from this row before serving any user request,
    eliminating the 30-60s cold-cache window after every deploy.
    """
    __tablename__ = "response_cache"

    cache_key = Column(String, primary_key=True)        # 'edge', 'track_record', ...
    payload_json = Column(Text, nullable=False)          # serialized response body
    cache_version = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, index=True)


# ── Membership + auth (Phase 1 — added 2026-07-20) ────────────────────

class UserRow(Base):
    """Every registered member. Created on first successful email
    verification, not on the /request-code POST (avoids polluting the
    table with unverified spam signups).

    role determines site-wide capability:
      - 'member'      — regular paying customer
      - 'power_user'  — read-only access + a test_plan_override that
                        bypasses Stripe for testing production flows
      - 'admin'       — full CRUD via /api/admin/*

    member_number is assigned when the user's subscription first goes
    active (trial start counts as "seat taken" per the model). It's the
    sequence 1..MEMBER_CAP and identifies founding members
    (member_number <= 100). Null while user is pre-trial or lapsed.

    seat_active tracks whether they currently occupy a seat against the
    MEMBER_CAP. Flips false when trial expires unconverted OR when
    subscription lapses. Waitlist promotion checks COUNT(seat_active=true).
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, unique=True, index=True)
    # Display name — collected later (Stripe checkout, profile edit).
    # Kept nullable so passwordless signup doesn't demand it up front.
    name = Column(String, nullable=True)
    # Collected at signup (sign-in page form). Nullable for pre-existing users.
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    referral_source = Column(String, nullable=True)   # "How did you find us?"
    # Personalised preferences (account page).
    mobile_number = Column(String, nullable=True)
    marketing_opt_in = Column(Boolean, nullable=False, default=True)
    role = Column(String, nullable=False, default="member", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Membership seat + founding status. Both nullable until first
    # subscription-active event.
    member_number = Column(Integer, nullable=True, unique=True, index=True)
    seat_active = Column(Boolean, nullable=False, default=False, index=True)
    founding = Column(Boolean, nullable=False, default=False)  # member_number <= 100 at seat allocation
    founding_coupon_issued_at = Column(DateTime, nullable=True)  # set once, at 1-year anniversary

    # Invite lineage — never mutated after signup. Helpful for tracing
    # organic growth patterns + attribution.
    invited_by_user_id = Column(Integer, nullable=True, index=True)

    # Power-user-only override. Ignored for role != 'power_user'.
    # Format: comma-separated plan tags, e.g. "punter_pro,labs,founding".
    # Auth middleware reads this in place of Stripe subscription state.
    test_plan_override = Column(String, nullable=True)

    # Remaining invites this member can issue. Defaults to 20 per the
    # Phase 2 spec ("same 20 invites for anyone"). Admins can bulk-mint
    # via /api/admin/invites/mint without touching this counter, so the
    # column reflects member-facing invite capacity only. Decremented
    # atomically on POST /api/invites/create; NOT refunded on revoke —
    # the invite existed and the seat may already have been claimed by
    # a different flow. Refill via admin adjustment if genuinely needed.
    invites_remaining = Column(Integer, nullable=False, default=20)

    # ── Paid access (freemium 5-day pass, 2026-08-24) ────────────────
    # THE single source of truth for "can this user see full picks".
    # Set by billing grant_access() on a completed payment, or by the
    # admin test-grant endpoint. NULL = never paid. Access is live while
    # access_until > utcnow(); the paywall gate checks exactly this and
    # nothing else. Deliberately provider-agnostic — the provider only
    # ever calls grant_access(), which extends this column. See
    # AccessGrantRow for the append-only payment ledger behind it.
    access_until = Column(DateTime, nullable=True, index=True)


class MagicLinkRow(Base):
    """Short-lived (15 min) one-time-use tokens sent by email to prove
    ownership of an email address. Consumed on GET /api/auth/verify
    which then creates a SessionRow. Never used as a session itself —
    the session lives in its own table with its own token.

    Token stored as SHA-256 hex hash so a DB compromise doesn't leak
    usable login codes.
    """
    __tablename__ = "magic_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, index=True)   # not user_id — sometimes user doesn't exist yet
    token_hash = Column(String, nullable=False, unique=True, index=True)
    intent = Column(String, nullable=False)              # 'login' | 'signup' — determines redirect target
    invite_token_hash = Column(String, nullable=True)    # tie-in with an InviteRow when intent='signup'
    # Signup profile carried from request-code → applied on account creation.
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    referral_source = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    used_at = Column(DateTime, nullable=True)


class SessionRow(Base):
    """The 7-day authenticated session. HttpOnly cookie value on client
    is hashed and stored here. Middleware reads cookie → hashes → looks
    up in this table → returns user or 401.

    On logout: delete this row (single session). Admin can 'revoke all
    sessions for user X' by deleting all rows for that user_id.
    """
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    cookie_hash = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    last_seen_at = Column(DateTime, default=datetime.utcnow)   # touched on every authed request; helps identify stale devices
    user_agent = Column(String, nullable=True)                 # useful for account-security review
    ip_address = Column(String, nullable=True)


class InviteRow(Base):
    """One invite code — either issued by a member to a specific friend
    or bulk-minted by an admin without a target email. Consumed on the
    verify path when the recipient completes their magic-link flow.

    We store SHA-256(code) — never the raw code — so a DB leak can't be
    replayed as a valid invite. The raw code lives only in the URL the
    member shared and in the recipient's inbox / clipboard.

    Lifecycle:
      created → (optionally) revoked_at set by issuer  →  consumed_at
                                                          set on verify.
    A revoked or consumed invite fails resolve_invite() from that point
    forward. expires_at defaults to 30 days; adjust when minting bulk.
    """
    __tablename__ = "invites"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code_hash = Column(String, nullable=False, unique=True, index=True)
    # Nullable — bulk admin invites don't target a specific email.
    # When set, request-code enforces that the recipient email matches
    # (case-insensitive) before consuming.
    issued_to_email = Column(String, nullable=True, index=True)
    # Nullable — bulk admin invites may not attribute to a specific
    # admin (e.g., minted by cron for a promo campaign).
    issued_by_user_id = Column(Integer, nullable=True, index=True)
    # Role the recipient gets on user creation. 'member' = normal (billed
    # eventually via Stripe), 'guest' = comp'd account with full read
    # access but no trial/subscription tie-in. Admin-only field — the
    # member-facing /api/invites/create ignores it and always mints
    # role='member'. Populated on consumption: verify sets
    # user.role = invite.role when this row is atomically consumed.
    role = Column(String, nullable=False, default="member")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    consumed_at = Column(DateTime, nullable=True)
    consumed_by_user_id = Column(Integer, nullable=True, index=True)
    revoked_at = Column(DateTime, nullable=True)


class WaitlistRow(Base):
    """People who tried to sign in without an invite. They land here so
    an admin can either promote them (mint an invite → email it) or
    surface them in a 'people waiting for a seat' view. Distinct from
    'people with valid invites who hit the member_cap' — that path
    creates a UserRow but leaves seat_active=False.
    """
    __tablename__ = "waitlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    # e.g. 'public_landing', 'invite_landing_no_code', 'cap_hit'.
    # Lets us slice conversion later.
    source = Column(String, nullable=True)
    # Optional free-text — user's own reason if we ever offer a textbox,
    # or admin notes attached during outreach.
    notes = Column(String, nullable=True)
    invited_at = Column(DateTime, nullable=True)  # set when admin mints an invite for this waitlister


class AccessGrantRow(Base):
    """Append-only ledger of paid-access grants — one row per settled
    payment (plus admin/comp grants). The row is the audit trail; the
    live access state lives on UserRow.access_until, which grant_access()
    extends.

    Provider-agnostic ON PURPOSE (2026-08-24): a billing-provider swap
    (Paddle → Stripe → …) just writes rows with a different `provider`
    tag — no schema change, and the access check never learns the
    provider's name. Historical Paddle rows stay interpretable forever.

    `external_txn_id` is the provider's transaction/order id and is
    UNIQUE, which gives idempotent webhooks for free: a retried delivery
    of the same transaction can't double-grant (grant_access() no-ops if
    the id already exists). Grants match a user by OUR user_id — passed
    to the provider as checkout metadata — never by the provider's
    customer id, so the provider's customer graph is never load-bearing.
    """
    __tablename__ = "access_grants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    provider = Column(String, nullable=False)              # 'paddle' | 'stripe' | 'admin' | 'comp'
    external_txn_id = Column(String, nullable=False, unique=True, index=True)
    days_granted = Column(Integer, nullable=False)
    amount = Column(Float, nullable=True)                  # decimal major units (e.g. 10.00)
    currency = Column(String, nullable=True)               # 'AUD'
    # What access_until became right after applying this grant — pure
    # audit, so the ledger fully explains the current access state.
    access_until_after = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class RAVenueKeyCacheRow(Base):
    """Persistent cache of the (date, state, clean_venue) → RA key mapping.

    RA identifies each meeting by a compound string like
    "2026Jul18,NSW,TAB Grafton" that we can't derive from our clean venue
    name because the sponsor prefix varies per meeting. Historically the
    RA client tries the plain name plus 6 sponsor variants until one
    resolves, then caches the answer in RAM. That in-memory cache dies
    on every Railway redeploy, so the sponsor-variant fanout burns ~1500
    extra RA requests per day — the exact pattern that got us WAF-flagged.

    Persisting to this table makes every resolved (date, state, venue)
    mapping survive redeploys. First lookup fans out once; subsequent
    lookups (and all future process starts) hit the DB directly.

    Row TTL: implicit — dates roll off naturally, we keep everything.
    """
    __tablename__ = "ra_venue_key_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_date = Column(String, nullable=False, index=True)   # "2026-07-18"
    state = Column(String, nullable=False)                    # "NSW"
    clean_venue = Column(String, nullable=False)              # "Grafton"
    ra_key = Column(String, nullable=False)                   # "2026Jul18,NSW,TAB Grafton"
    resolved_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("race_date", "state", "clean_venue", name="uq_ra_venue_key"),
    )


class RAFormCacheRow(Base):
    """Persistent cache of parsed HorseFullForm / JockeyLastRuns /
    TrainerLastRuns responses. Same structural fix as ra_venue_key_cache:
    the RA client had per-code RAM caches with 1h TTL, which the enrichment
    schedule (8:30 + 10:30 + 11:30 AEST) blows through and every Railway
    redeploy wipes.

    kind identifies the source page: 'h' HorseFullForm, 'j' JockeyLastRuns,
    't' TrainerLastRuns. Cache reads apply a per-kind TTL (see the RA
    client) — horse form is safe to cache longer since horses only race
    every 2-4 weeks; jockey/trainer stats update daily as they ride/train.

    payload_json holds the already-parsed dict, so hits skip both the RA
    fetch AND the BeautifulSoup parse.
    """
    __tablename__ = "ra_form_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(1), nullable=False)      # 'h' | 'j' | 't'
    code = Column(String, nullable=False)          # horsecode / jockeycode / trainercode
    payload_json = Column(Text, nullable=False)    # serialized parsed form dict
    cached_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("kind", "code", name="uq_ra_form_kind_code"),
    )


class BetRecommendationRow(Base):
    """One row per recommended trifecta box bet. Paper-trading ledger —
    no real money. Settled after the race using the trifecta dividend
    parsed from RA's Results.aspx."""
    __tablename__ = "bet_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True, nullable=False)
    strategy_label = Column(String, nullable=False)  # core_top3 / core_top4 / value_runner1 / value_runner2 / no_favourite_hedge
    box_horses_json = Column(Text, nullable=False)   # JSON array of tab numbers
    box_horse_names_json = Column(Text, nullable=False)  # JSON array of names — display only
    num_permutations = Column(Integer, nullable=False)
    stake_dollars = Column(Float, nullable=False, default=2.0)
    recommended_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Settlement (filled by settlement job after results land)
    settled = Column(Boolean, default=False, index=True)
    is_hit = Column(Boolean, nullable=True)
    actual_top3_json = Column(Text, nullable=True)  # JSON array of tab numbers in finishing order
    trifecta_dividend = Column(Float, nullable=True)  # listed pool dividend
    payout_dollars = Column(Float, nullable=True)     # (stake / num_perms) * dividend if hit
    pnl_dollars = Column(Float, nullable=True)        # payout - stake
    settled_at = Column(DateTime, nullable=True)
    # Voided: the box contained a runner that was scratched between
    # generation and the race jumping. Real-world TAB would refund the
    # corresponding share of the stake — for our paper-trading the bet
    # neither counts as a hit nor a loss. is_hit stays False; voided=True
    # excludes the row from hit-rate and P&L aggregates.
    voided = Column(Boolean, default=False, nullable=True)
    # True when the dividend was Harville-estimated from model probabilities
    # because TAB's race endpoint didn't return a trifecta dividend. Lets
    # the UI flag the resulting P&L as 'estimated' so users don't mistake
    # it for actual TAB settlement.
    dividend_estimated = Column(Boolean, default=False, nullable=True)


class NightlyReviewRow(Base):
    """One row per (review_date, source). Nightly Python analyser writes
    source='python' rows; weekly Claude agent writes source='claude' rows.
    suggestions_json is the canonical list of structured suggestions —
    each item carries its own id, status, and notes so the admin UI can
    tick items off independently."""
    __tablename__ = "nightly_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_date = Column(String, index=True, nullable=False)  # YYYY-MM-DD (date being reviewed)
    source = Column(String, nullable=False, default="python")  # 'python' | 'claude'
    summary_markdown = Column(Text, nullable=False, default="")
    suggestions_json = Column(Text, nullable=False, default="[]")
    headline_stats_json = Column(Text, nullable=True)  # {top1_win_pct, top3_strike_pct, anomalies_count, ...}
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


async def init_db() -> None:
    import logging as _logging
    _log = _logging.getLogger(__name__)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Run each DDL statement in its own transaction so one failure never
    # blocks the rest. All statements are idempotent (IF NOT EXISTS / IF EXISTS).
    migrations = [
        # ── History write-once guard (2026-07-28, MR CACCIATORE incident) ──
        # DB-level backstop independent of app guards: after a race jumps,
        # (a) new "live" snapshot batches are contaminated AT BIRTH (readers
        #     already filter contaminated), and
        # (b) UPDATEs may not change horse/rank/probabilities — the trigger
        #     silently keeps the OLD values (cancelled/track_condition/is_sharp
        #     syncs still pass). Every catch is logged to
        #     history_guard_incidents, surfaced by the nightly quality check.
        """
        CREATE TABLE IF NOT EXISTS history_guard_incidents (
            id SERIAL PRIMARY KEY,
            race_id TEXT,
            horse_name TEXT,
            kind TEXT,
            detail TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """,
        # Per-race going + weather for analysis. track_rating = numeric going
        # (1-2 Firm, 3-4 Good, 5-7 Soft, 8-10 Heavy) so you filter/join on the
        # number, not the string (e.g. Good = track_rating IN (3,4)).
        """
        CREATE TABLE IF NOT EXISTS race_conditions (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            state TEXT,
            race_date TEXT,
            track_condition TEXT,
            track_rating INTEGER,
            rain_mm REAL,
            temp_max REAL,
            temp_min REAL,
            wind_kmh REAL,
            weather_condition TEXT,
            weather_icon TEXT,
            recorded_at TIMESTAMPTZ DEFAULT now()
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_race_conditions_track_rating ON race_conditions(track_rating)",
        "CREATE INDEX IF NOT EXISTS ix_race_conditions_race_date ON race_conditions(race_date)",
        # Dynamic per-venue quality badge, recomputed weekly from a rolling window
        # of rank-1 results. gold = win% > 40, silver = win% > 30 (min sample).
        """
        CREATE TABLE IF NOT EXISTS venue_badges (
            venue TEXT PRIMARY KEY,
            state TEXT,
            badge TEXT,
            win_pct REAL,
            place_pct REAL,
            sample_size INTEGER,
            window_days INTEGER,
            computed_at TIMESTAMPTZ DEFAULT now()
        )
        """,
        """
        CREATE OR REPLACE FUNCTION fiq_history_write_guard() RETURNS trigger AS $fn$
        DECLARE jumped boolean := false;
        BEGIN
          BEGIN
            jumped := NEW.scheduled_time IS NOT NULL
                      AND (NEW.scheduled_time)::timestamptz < (now() - interval '5 minutes');
          EXCEPTION WHEN others THEN
            jumped := false;
          END;
          IF TG_OP = 'INSERT' THEN
            IF jumped AND COALESCE(NEW.source, 'live') = 'live'
               AND EXISTS (
                 SELECT 1 FROM runner_prediction_history h
                 WHERE h.race_id = NEW.race_id
                   AND COALESCE(h.source, 'live') = 'live'
                   AND h.batch_id IS DISTINCT FROM NEW.batch_id
               ) THEN
              NEW.contaminated := true;
              INSERT INTO history_guard_incidents(race_id, horse_name, kind, detail)
              VALUES (NEW.race_id, NEW.horse_name, 'post_race_insert',
                      'live snapshot after jump — auto-contaminated at birth');
            END IF;
            RETURN NEW;
          ELSE
            IF jumped AND (
                 NEW.horse_name IS DISTINCT FROM OLD.horse_name
              OR NEW.model_rank IS DISTINCT FROM OLD.model_rank
              OR NEW.win_probability IS DISTINCT FROM OLD.win_probability
              OR NEW.place_probability IS DISTINCT FROM OLD.place_probability
              OR NEW.is_sharp IS DISTINCT FROM OLD.is_sharp
            ) THEN
              INSERT INTO history_guard_incidents(race_id, horse_name, kind, detail)
              VALUES (OLD.race_id, OLD.horse_name, 'post_race_update_blocked',
                      'kept pre-race values; attempted rank='
                      || COALESCE(NEW.model_rank::text, '?')
                      || ' win_prob=' || COALESCE(NEW.win_probability::text, '?'));
              NEW.horse_name := OLD.horse_name;
              NEW.model_rank := OLD.model_rank;
              NEW.win_probability := OLD.win_probability;
              NEW.place_probability := OLD.place_probability;
              NEW.place_model_rank := OLD.place_model_rank;
              NEW.exotic_model_rank := OLD.exotic_model_rank;
              NEW.is_sharp := OLD.is_sharp;   -- lock the Sharp flag post-jump (tracked feature)
            END IF;
            -- A runner that already ran cannot be scratched after the jump.
            -- Block a NEW post-jump cancellation (a false scratch that would drop
            -- the pick from the board and skew win/Sharp stats). Un-cancelling
            -- (True->False) stays allowed so the result-reconciliation heal can
            -- still restore genuine false scratches.
            IF jumped AND COALESCE(OLD.cancelled, false) = false AND NEW.cancelled IS TRUE THEN
              INSERT INTO history_guard_incidents(race_id, horse_name, kind, detail)
              VALUES (OLD.race_id, OLD.horse_name, 'post_race_cancel_blocked',
                      'kept pre-race cancelled=false; a runner cannot be scratched after the jump');
              NEW.cancelled := OLD.cancelled;
            END IF;
            RETURN NEW;
          END IF;
        END
        $fn$ LANGUAGE plpgsql
        """,
        """
        DROP TRIGGER IF EXISTS trg_fiq_history_write_guard ON runner_prediction_history
        """,
        """
        CREATE TRIGGER trg_fiq_history_write_guard
        BEFORE INSERT OR UPDATE ON runner_prediction_history
        FOR EACH ROW EXECUTE FUNCTION fiq_history_write_guard()
        """,
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS cancelled BOOLEAN DEFAULT FALSE",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS place_model_rank INTEGER",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS exotic_model_rank INTEGER",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS scheduled_time TEXT",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS venue TEXT",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS state TEXT",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS race_number INTEGER",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS race_name TEXT",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS distance INTEGER",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS track_condition TEXT",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS field_size INTEGER",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS prize_money INTEGER",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS rail_position TEXT",
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS class_change INTEGER",
        "ALTER TABLE runner_prediction_history ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'live'",
        "ALTER TABLE runner_prediction_history ADD COLUMN IF NOT EXISTS batch_id TEXT",
        # Isotonic-on-raw-softmax pipeline (2026-07-15). Populated going
        # forward by predict_race; rows written before this deploy stay
        # NULL and are filtered out of the calibration fit.
        "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS win_prob_raw DOUBLE PRECISION",
        "ALTER TABLE runner_prediction_history ADD COLUMN IF NOT EXISTS win_prob_raw DOUBLE PRECISION",
        # Release-gate architecture for model artefacts (2026-07-15).
        # Nightly rebuilds now write candidate rows instead of overwriting
        # active. Existing single-row rows default to status='active' so
        # the load path finds them unchanged on first boot after deploy.
        "ALTER TABLE win_calibration_curve ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active'",
        "ALTER TABLE win_calibration_curve ADD COLUMN IF NOT EXISTS backtest_json TEXT",
        "ALTER TABLE win_calibration_curve ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP",
        "ALTER TABLE win_calibration_curve ADD COLUMN IF NOT EXISTS reviewed_note TEXT",
        "CREATE INDEX IF NOT EXISTS ix_win_cal_status ON win_calibration_curve (status)",
        "UPDATE win_calibration_curve SET status = 'active' WHERE status IS NULL",
        # BAO variant on calibration trend (2026-07-15).
        "ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS value_roi_bao DOUBLE PRECISION",
        "ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS bao_coverage_pct DOUBLE PRECISION",
        # Model-weight release gate (2026-07-17). model_weight_candidates
        # table is auto-created by Base.metadata.create_all above; these
        # indexes speed up "load a candidate's full weight set" (batch_id)
        # and "list candidates for a model type" (model_type + status).
        "CREATE INDEX IF NOT EXISTS ix_mwcand_batch ON model_weight_candidates (batch_id)",
        "CREATE INDEX IF NOT EXISTS ix_mwcand_type_status ON model_weight_candidates (model_type, status)",
        "CREATE INDEX IF NOT EXISTS ix_hist_batch_id ON runner_prediction_history (batch_id)",
        # Security findings — Dr Evil's threat/blast-radius line (2026-08-11).
        # Table is auto-created by create_all; this ALTER adds the column to the
        # already-existing table on redeploy.
        "ALTER TABLE security_findings ADD COLUMN IF NOT EXISTS threat TEXT",
        # UNIQUE index creation must be robust to pre-existing duplicates.
        # The 2026-07-18 incident: this CREATE UNIQUE INDEX ran on a table
        # that had duplicate (race_id, horse_name) rows from a prior settle
        # sweep, threw a duplicate-key error, and Railway kept trying to
        # restart the app until Postgres itself was killed. Dedupe first
        # (keep the lowest id, drop the rest) then attempt the index.
        # DELETE runs unconditionally — a no-op on a clean table.
        """
        DELETE FROM runner_prediction_history
        WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY race_id, horse_name
                    ORDER BY id
                ) AS rn
                FROM runner_prediction_history
            ) t WHERE rn > 1
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_history_race_horse ON runner_prediction_history (race_id, horse_name)",
        "CREATE INDEX IF NOT EXISTS ix_runner_pred_race_rank ON runner_predictions (race_id, model_rank)",
        "CREATE INDEX IF NOT EXISTS ix_runner_pred_hist_race_rank ON runner_prediction_history (race_id, model_rank)",
        # OBS-G: composite (source, race_id, cancelled) supports the very common
        # premium / edge / calibration / backtest reads, which all filter on
        # source='live' + cancelled NULL/false and scan races_id ranges. Postgres
        # can ignore the prefix here; SQLite uses leftmost prefix matching so the
        # ordering still benefits common workloads.
        "CREATE INDEX IF NOT EXISTS ix_hist_source_race_cancelled ON runner_prediction_history (source, race_id, cancelled)",
        # Data-quality flag on history rows. Set to True for rows written
        # during known-corrupted windows (pipeline chaos, odds bug, WAF
        # outage). Retrains and backtests filter these out by default.
        # ORM added the column on 2026-07-16; this migration ships the
        # ALTER so DB matches. Any read of runner_prediction_history was
        # 500'ing until this landed.
        "ALTER TABLE runner_prediction_history ADD COLUMN IF NOT EXISTS contaminated BOOLEAN DEFAULT FALSE",
        "CREATE INDEX IF NOT EXISTS ix_hist_contaminated ON runner_prediction_history (contaminated)",
        # RA venue-key cache (2026-07-18). Table itself is auto-created by
        # Base.metadata.create_all above. This index speeds up the very hot
        # "have I resolved this (date, state, venue) already?" lookup that
        # every find_results call now performs to skip the sponsor-variant
        # fanout. The uq_ra_venue_key unique constraint doubles as the
        # index for the equality lookup.
        "CREATE INDEX IF NOT EXISTS ix_ra_venue_key_date ON ra_venue_key_cache (race_date)",
        # RA form cache (2026-07-18) — persist horse/jockey/trainer form
        # parses across process restarts to skip repeated RA fetches. The
        # uq_ra_form_kind_code unique constraint provides the equality
        # lookup index; the cached_at index supports TTL / cleanup queries.
        "CREATE INDEX IF NOT EXISTS ix_ra_form_cached_at ON ra_form_cache (cached_at)",
        # ── Membership + auth (Phase 1 — 2026-07-20) ─────────────────
        # New tables (users, magic_links, sessions) are auto-created by
        # Base.metadata.create_all above. These migrations only need to
        # exist for indexes not already declared inline on the model
        # (email lookups, session expiry sweeps, magic-link cleanup).
        "CREATE INDEX IF NOT EXISTS ix_users_role_seat ON users (role, seat_active)",
        "CREATE INDEX IF NOT EXISTS ix_magic_links_email_created ON magic_links (email, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_sessions_user_expires ON sessions (user_id, expires_at)",
        # ── Membership + auth (Phase 2 — invites + waitlist) ─────────
        # invites_remaining lands on users; new tables invites + waitlist
        # are created by Base.metadata.create_all above. Backfill sets
        # any pre-Phase-2 users to the default cap so existing accounts
        # (right now: just the bootstrap admin) can start issuing.
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS invites_remaining INTEGER NOT NULL DEFAULT 20",
        "UPDATE users SET invites_remaining = 20 WHERE invites_remaining IS NULL",
        # Give the first admin a much larger pool so bootstrap invites
        # for the initial cohort don't chew through the default 20. Only
        # bumps admins whose pool is still at the default — hand-adjusted
        # values won't be clobbered.
        "UPDATE users SET invites_remaining = 500 WHERE role = 'admin' AND invites_remaining = 20",
        # Composite so 'issuer's active invites' listing (dashboard) hits
        # an index — the exact filter is (issued_by_user_id, revoked_at
        # IS NULL, consumed_at IS NULL). Postgres can use this prefix.
        "CREATE INDEX IF NOT EXISTS ix_invites_issuer_active ON invites (issued_by_user_id, consumed_at, revoked_at)",
        "CREATE INDEX IF NOT EXISTS ix_invites_expires ON invites (expires_at)",
        # Guest-invite support: role stamped on the invite propagates to
        # user.role on consumption. Defaults to 'member' so any pre-Phase-2.1
        # invite still creates a normal member account.
        "ALTER TABLE invites ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'member'",
        "CREATE INDEX IF NOT EXISTS ix_waitlist_created ON waitlist (created_at)",
        # ── Paid access (freemium 5-day pass, 2026-08-24) ────────────
        # access_until on users + the provider-agnostic access_grants
        # ledger (the table itself is auto-created by create_all above;
        # these statements add the column on older deployments and the
        # supporting indexes / idempotency unique). TIMESTAMP is correct
        # on Postgres and accepted by SQLite.
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS access_until TIMESTAMP",
        "CREATE INDEX IF NOT EXISTS ix_users_access_until ON users (access_until)",
        # Guarantee no duplicate accounts / member numbers even on tables that
        # predate the model's unique=True. Idempotent; in Postgres NULL
        # member_number rows (pre-allocation) are exempt from the unique check.
        # A failure (e.g. pre-existing dupes) is logged and skipped, not fatal.
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON users (email)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_member_number ON users (member_number)",
        # Signup profile fields (name + referral source), on users and carried
        # on the magic-link row from request-code through to account creation.
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name TEXT",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name TEXT",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS referral_source TEXT",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS mobile_number TEXT",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS marketing_opt_in BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE magic_links ADD COLUMN IF NOT EXISTS first_name TEXT",
        "ALTER TABLE magic_links ADD COLUMN IF NOT EXISTS last_name TEXT",
        "ALTER TABLE magic_links ADD COLUMN IF NOT EXISTS referral_source TEXT",
        # One-shot backfill: give existing accounts a sequential member number
        # (earliest join = lowest number), continuing after the current max.
        # Idempotent — only touches rows still NULL, so it's a no-op once run.
        """
        UPDATE users u SET member_number = s.newnum FROM (
            SELECT id,
                   (SELECT COALESCE(MAX(member_number), 0) FROM users)
                   + row_number() OVER (ORDER BY created_at NULLS FIRST, id) AS newnum
            FROM users WHERE member_number IS NULL
        ) s WHERE u.id = s.id
        """,
        # Founding = first 100 member numbers.
        "UPDATE users SET founding = TRUE WHERE member_number IS NOT NULL AND member_number <= 100 AND founding = FALSE",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_access_grants_external_txn ON access_grants (external_txn_id)",
        "CREATE INDEX IF NOT EXISTS ix_access_grants_user ON access_grants (user_id)",
        "CREATE INDEX IF NOT EXISTS ix_hist_results_race_winner ON historical_results (race_id, winner)",
        "CREATE INDEX IF NOT EXISTS ix_hist_results_race_placed ON historical_results (race_id, placed)",
        # Composite for common top-3/top-4 fetches: SELECT ... WHERE race_id = X AND position IN (1,2,3)
        "CREATE INDEX IF NOT EXISTS ix_hist_results_race_position ON historical_results (race_id, position)",
        # text_pattern_ops so `race_id LIKE 'YYYY-MM-DD_%'` prefix queries
        # (used every daily aggregation) can use index scans. Default
        # locale-aware btree can't do LIKE prefix on non-C locale.
        "CREATE INDEX IF NOT EXISTS ix_hist_results_race_id_pattern ON historical_results (race_id text_pattern_ops)",
        # Partial index: settle path filters "races with at least one
        # top-3 finisher whose tab_number is populated" — small subset
        # of total rows, but the exact hot filter for bet settlement.
        "CREATE INDEX IF NOT EXISTS ix_hist_results_race_with_tab ON historical_results (race_id) WHERE tab_number IS NOT NULL",
        # Note: dedupe + CREATE UNIQUE INDEX is deliberately NOT run here.
        # The dedup is destructive and needs a preview + sign-off first.
        # Use /api/admin/dedupe-historical-results?apply=false to preview
        # counts, then apply=true to run. Once the historic table is clean
        # add back a "CREATE UNIQUE INDEX IF NOT EXISTS
        # ux_historical_results_race_horse ON historical_results
        # (race_id, LOWER(horse_name))" here so future deploys stay safe.
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_bet_reco_race_strategy ON bet_recommendations (race_id, strategy_label)",
        "CREATE INDEX IF NOT EXISTS ix_bet_reco_settled ON bet_recommendations (settled)",
        "ALTER TABLE bet_recommendations ADD COLUMN IF NOT EXISTS voided BOOLEAN DEFAULT FALSE",
        "ALTER TABLE bet_recommendations ADD COLUMN IF NOT EXISTS dividend_estimated BOOLEAN DEFAULT FALSE",
        # response_cache table is created by Base.metadata.create_all above,
        # but make sure the cache_version column exists on older deployments.
        "ALTER TABLE response_cache ADD COLUMN IF NOT EXISTS cache_version INTEGER DEFAULT 0",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_ra_calendar_date_state ON ra_calendar_cache (race_date, state)",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS jockey TEXT",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS trainer TEXT",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS venue TEXT",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS state TEXT",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS distance INTEGER",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS track_condition TEXT",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS barrier INTEGER",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS tab_number INTEGER",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS weight REAL",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS age INTEGER",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS sex TEXT",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS race_class TEXT",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS prize_money INTEGER",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS field_size INTEGER",
        "ALTER TABLE historical_results ADD COLUMN IF NOT EXISTS race_number INTEGER",
        "CREATE INDEX IF NOT EXISTS ix_hist_results_jockey ON historical_results (jockey)",
        "CREATE INDEX IF NOT EXISTS ix_hist_results_trainer ON historical_results (trainer)",
        "CREATE INDEX IF NOT EXISTS ix_hist_results_venue ON historical_results (venue)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_nightly_review_date_source ON nightly_reviews (review_date, source)",
        # Sharp eligibility snapshot — see RunnerPredictionHistoryRow.is_sharp.
        "ALTER TABLE runner_prediction_history ADD COLUMN IF NOT EXISTS is_sharp BOOLEAN",
        "CREATE INDEX IF NOT EXISTS ix_hist_is_sharp ON runner_prediction_history (is_sharp)",
        # prediction_integrity — the table was created (6acb488) before
        # baseline_field was added to the model (0eefb46), and there was no
        # ALTER migration, so the column was missing in production and EVERY
        # read of the integrity audit 500'd ("column ... does not exist";
        # the /admin/prediction-integrity endpoint SELECTs all columns).
        # Ship IF NOT EXISTS ALTERs for the whole nullable column set so the
        # DB matches the model regardless of which deploy created the table.
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS scheduled_time TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS jump_top1 TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS jump_top2 TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS jump_top3 TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS jump_captured_at TIMESTAMP",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS baseline_field TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS eod_top1 TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS eod_top2 TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS eod_top3 TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS eod_captured_at TIMESTAMP",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS mismatch BOOLEAN DEFAULT FALSE",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS detail TEXT",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS jump_is_sharp BOOLEAN",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS eod_is_sharp BOOLEAN",
        "ALTER TABLE prediction_integrity ADD COLUMN IF NOT EXISTS sharp_mismatch BOOLEAN DEFAULT FALSE",
        "CREATE INDEX IF NOT EXISTS ix_prediction_integrity_mismatch ON prediction_integrity (mismatch)",
    ]
    for stmt in migrations:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        except Exception as e:
            _log.warning("[init_db] migration skipped (%s): %s", stmt[:60], e)

    # One-shot backfill: populate is_sharp for existing rows using the
    # ORIGINAL gate (model_pct >= 0.30 OR top3_sum >= 0.60), NOT the
    # newer days_off ≤ 180 rule. This pins historical Sharp percentages
    # to whatever they were before the gate refinement landed, so past
    # numbers stop shifting. Only runs once per row — once is_sharp is
    # set (True or False), it's never touched.
    backfills = [
        # rank-1 with prob ≥ 0.30 → Sharp regardless of top-3 sum
        """
        UPDATE runner_prediction_history
        SET is_sharp = TRUE
        WHERE model_rank = 1
          AND is_sharp IS NULL
          AND win_probability >= 0.30
        """,
        # rank-1 with prob < 0.30 but race's top-3 sum ≥ 0.60 → Sharp
        # Compute top-3 sum per race using a correlated subquery that
        # works on both SQLite and Postgres (avoids window functions).
        """
        UPDATE runner_prediction_history AS h
        SET is_sharp = TRUE
        WHERE h.model_rank = 1
          AND h.is_sharp IS NULL
          AND (
            SELECT COALESCE(SUM(win_probability), 0)
            FROM runner_prediction_history
            WHERE race_id = h.race_id
              AND model_rank IN (1, 2, 3)
              AND (cancelled IS FALSE OR cancelled IS NULL)
          ) >= 0.60
        """,
        # Everything else on rank-1 → not Sharp. Sets explicit FALSE so
        # the IS NULL guard above doesn't re-run on every startup.
        """
        UPDATE runner_prediction_history
        SET is_sharp = FALSE
        WHERE model_rank = 1 AND is_sharp IS NULL
        """,
        # Non-rank-1 rows: leave is_sharp NULL (the Sharp gate only
        # applies to rank-1 picks per the definition).
    ]
    for stmt in backfills:
        try:
            async with engine.begin() as conn:
                result = await conn.execute(text(stmt))
                _log.info("[init_db] is_sharp backfill: %s rows", result.rowcount)
        except Exception as e:
            _log.warning("[init_db] is_sharp backfill skipped: %s", e)

    # ── First-admin bootstrap ─────────────────────────────────────────
    # Idempotent: only inserts if there is currently no admin AND the
    # target email doesn't already exist. Subsequent config changes to
    # first_admin_email are no-ops (won't demote an existing admin,
    # won't create a second admin). Post-launch, admins are managed by
    # existing admins via the admin dashboard.
    try:
        from horse_engine.config import settings as _settings
        target_email = (_settings.first_admin_email or "").strip().lower()
        if target_email:
            async with engine.begin() as conn:
                any_admin = (await conn.execute(
                    text("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1")
                )).scalar()
                if not any_admin:
                    existing = (await conn.execute(
                        text("SELECT id FROM users WHERE email = :e"),
                        {"e": target_email},
                    )).scalar()
                    if existing:
                        await conn.execute(
                            text("UPDATE users SET role = 'admin' WHERE id = :id"),
                            {"id": existing},
                        )
                        _log.info("[init_db] Promoted existing user %s to admin", target_email)
                    else:
                        await conn.execute(
                            text("INSERT INTO users (email, role, created_at, seat_active, founding) "
                                 "VALUES (:e, 'admin', :now, FALSE, FALSE)"),
                            {"e": target_email, "now": datetime.utcnow()},
                        )
                        _log.info("[init_db] Bootstrapped first admin: %s", target_email)
    except Exception as e:
        _log.warning("[init_db] first-admin bootstrap skipped: %s", e)


async def backfill_prediction_history(session: AsyncSession) -> int:
    """
    One-time back-fill: copy all pre-race RunnerPredictionRows into history.
    Safe to call repeatedly — skips races already in history.
    Returns number of races copied.
    """
    from sqlalchemy import func
    already = (await session.execute(
        select(func.distinct(RunnerPredictionHistoryRow.race_id))
    )).scalars().all()
    already_set = set(already)

    result = await session.execute(
        select(RunnerPredictionRow)
        .where(RunnerPredictionRow.enriched_at.isnot(None))
        .where(RunnerPredictionRow.scheduled_time.isnot(None))
    )
    rows = result.scalars().all()

    copied = 0
    seen_races: set[str] = set()
    race_rows: dict[str, list] = {}
    for row in rows:
        try:
            sched = sched_to_utc_naive(row.scheduled_time)
            if sched and row.enriched_at < sched and row.race_id not in already_set:
                race_rows.setdefault(row.race_id, []).append(row)
        except (ValueError, TypeError):
            continue

    for race_id, rrows in race_rows.items():
        for row in rrows:
            session.add(RunnerPredictionHistoryRow(
                race_id=row.race_id,
                horse_name=row.horse_name,
                tab_number=row.tab_number,
                barrier=row.barrier,
                jockey=row.jockey,
                trainer=row.trainer,
                weight=row.weight,
                win_probability=row.win_probability,
                place_probability=row.place_probability,
                model_rank=row.model_rank,
                market_rank=row.market_rank,
                overlay=row.overlay,
                best_available_odds=row.best_available_odds,
                value_rating=row.value_rating,
                narrative=row.narrative,
                key_flags=row.key_flags,
                enriched_json=row.enriched_json,
                place_model_rank=row.place_model_rank,
                exotic_model_rank=row.exotic_model_rank,
                scheduled_time=row.scheduled_time,
                enriched_at=row.enriched_at,
                cancelled=row.cancelled,
                venue=getattr(row, "venue", None),
                state=getattr(row, "state", None),
                race_number=getattr(row, "race_number", None),
                race_name=getattr(row, "race_name", None),
                distance=getattr(row, "distance", None),
                track_condition=getattr(row, "track_condition", None),
                field_size=getattr(row, "field_size", None),
                prize_money=getattr(row, "prize_money", None),
                rail_position=getattr(row, "rail_position", None),
                class_change=getattr(row, "class_change", None),
                model_score=getattr(row, "model_score", None),
                source="live",
                recorded_at=datetime.utcnow(),
            ))
        copied += 1
    await session.commit()
    return copied


async def save_race_predictions(session: AsyncSession, race_id: str, predictions: list[dict], force: bool = False) -> None:
    from sqlalchemy import delete, func

    # Check if an immutable history snapshot already exists for this race
    history_exists = (await session.execute(
        select(func.count()).select_from(RunnerPredictionHistoryRow)
        .where(RunnerPredictionHistoryRow.race_id == race_id)
    )).scalar() or 0

    # Block post-race re-enrichment only when a pre-race snapshot already exists.
    # If history_exists=False the race has never been snapshotted — allow the write
    # so late-listed meetings can still get their first prediction captured.
    # force=True bypasses the guard AND mirrors the fresh values into the
    # latest history snapshot too. Use ONLY for admin repair when the
    # existing snapshot is known-broken (e.g. captured during an outage);
    # normal callers must leave force=False so Ground Rule 1 holds.
    if predictions and history_exists and not force:
        scheduled_time = predictions[0].get("scheduled_time")
        if scheduled_time:
            try:
                sched = sched_to_utc_naive(scheduled_time)
                if sched is not None and datetime.utcnow() > sched:
                    return  # Pre-race snapshot exists and race has started — don't overwrite
            except (ValueError, TypeError):
                pass

    # Preserve known-good market data across re-enrichment. The save below
    # is a delete-then-insert (line 482), so if the upstream odds source
    # was unreachable this cycle, p["best_available_odds"] arrives as 0
    # and would clobber a perfectly good prior value. Capture per-horse
    # odds/derived fields before the delete and restore them if the new
    # values are empty — the next live-odds refresh will overwrite with
    # fresh data when upstream recovers.
    #
    # Also preserve cancelled=True across re-enrichment: when upstream lags
    # behind Sportsbet/stewards on a scratching, the morning enrich would
    # otherwise resurrect a scratched horse (e.g. Winchman/Doomben R3 on
    # 2026-06-24). The cancellation only clears when an admin explicitly
    # calls /api/admin/restore-cancelled.
    existing_market: dict[str, dict] = {}
    existing_cancelled: set[str] = set()
    existing_rows = (await session.execute(
        select(
            RunnerPredictionRow.horse_name,
            RunnerPredictionRow.best_available_odds,
            RunnerPredictionRow.market_rank,
            RunnerPredictionRow.overlay,
            RunnerPredictionRow.value_rating,
            RunnerPredictionRow.cancelled,
        ).where(RunnerPredictionRow.race_id == race_id)
    )).fetchall()
    for horse_name, bao, mrank, overlay, vrating, cancelled in existing_rows:
        if bao and bao > 1.0:
            existing_market[horse_name] = {
                "best_available_odds": bao,
                "market_rank": mrank,
                "overlay": overlay,
                "value_rating": vrating,
            }
        if cancelled and horse_name:
            existing_cancelled.add(horse_name)

    # Write to mutable table
    await session.execute(delete(RunnerPredictionRow).where(RunnerPredictionRow.race_id == race_id))
    rows = []
    for p in predictions:
        new_odds = p.get("best_available_odds") or 0
        if new_odds <= 1.0:
            prev = existing_market.get(p.get("horse_name"))
            if prev:
                p = dict(p)
                p["best_available_odds"] = prev["best_available_odds"]
                if prev.get("market_rank") is not None:
                    p["market_rank"] = prev["market_rank"]
                if prev.get("overlay") is not None:
                    p["overlay"] = prev["overlay"]
                if prev.get("value_rating") is not None:
                    p["value_rating"] = prev["value_rating"]
        if p.get("horse_name") in existing_cancelled:
            p = dict(p)
            p["cancelled"] = True
        row = RunnerPredictionRow(**p)
        session.add(row)
        rows.append(row)
    await session.commit()

    # Cross-entry handling (2026-07-31 root fix — false-scratch incident).
    # A horse nominated in more than one race at a meeting (cross-entry) RUNS
    # exactly one of them, decided by the trainer/stewards — and it is very
    # often the EARLIER race, with the later entries scratched. The old code
    # here assumed the opposite ("keep the highest-numbered race, cancel the
    # earlier ones") and so falsely cancelled horses that actually ran: on
    # 2026-07-30 alone it killed Lego Master (grange R1, ran), Sleepy Joe /
    # Qickwit (gatton R3, ran) and Coral Cove (grange R2, ran) — every one the
    # earlier leg of a cross-entry. The guess is unrecoverable pre-race and only
    # the nightly result-reconciliation caught it after the fact.
    #
    # There is no reliable way to know which leg a cross-entered horse runs from
    # the race number alone, so we NO LONGER guess. A cross-entry is left active
    # in every race until an EXPLICIT scratch marker removes it from the ones it
    # doesn't run — handled per-race by _apply_oddspro_scratches (OddsPro
    # status=='SCRATCHED'). Same-race re-enrichment can't resurrect a real
    # scratch either: existing cancelled=True rows are preserved above
    # (existing_cancelled). Explicit signals only — never a race-number guess.

    # Force-mirror path: history already exists and the caller asked to overwrite.
    # Update the LATEST snapshot batch's rows with the fresh probabilities/rank so
    # settled-race consumers (/api/edge for races with results) see the corrected
    # values. Only fields the pipeline recomputes are touched — batch_id, source,
    # enriched_at, recorded_at are left alone so the audit trail of the original
    # snapshot moment is preserved.
    if force and history_exists and predictions:
        from sqlalchemy import update as sa_update
        latest_at = (await session.execute(
            select(func.max(RunnerPredictionHistoryRow.enriched_at))
            .where(RunnerPredictionHistoryRow.race_id == race_id)
        )).scalar()
        if latest_at is not None:
            by_name = {p.get("horse_name"): p for p in predictions if p.get("horse_name")}
            # Recompute race-level is_sharp for the rank-1 pick, mirroring the
            # gate used at snapshot time (see _snapshot_prerace_predictions).
            active = [p for p in predictions if not p.get("cancelled")]
            active_sorted = sorted(active, key=lambda p: p.get("model_rank") or 99)
            rank1 = active_sorted[0] if active_sorted else None
            top3_sum = sum((p.get("win_probability") or 0) for p in active_sorted[:3])
            race_is_sharp = None
            if rank1 is not None:
                r1_conf = (rank1.get("win_probability") or 0) >= 0.30 or top3_sum >= 0.60
                r1_days_off = rank1.get("days_since_last_run")
                r1_layoff_ok = r1_days_off is None or r1_days_off <= 180
                race_is_sharp = bool(r1_conf and r1_layoff_ok)
            hist_rows = (await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.race_id == race_id)
                .where(RunnerPredictionHistoryRow.enriched_at == latest_at)
            )).scalars().all()
            for hrow in hist_rows:
                p = by_name.get(hrow.horse_name)
                if p is None:
                    continue
                new_values = {
                    "win_probability": p.get("win_probability"),
                    "place_probability": p.get("place_probability"),
                    "model_rank": p.get("model_rank"),
                    "place_model_rank": p.get("place_model_rank"),
                    "exotic_model_rank": p.get("exotic_model_rank"),
                    "best_available_odds": p.get("best_available_odds"),
                    "overlay": p.get("overlay"),
                    "value_rating": p.get("value_rating"),
                    "cancelled": p.get("cancelled"),
                }
                # Only rank-1 carries the Sharp flag (race-level property).
                if p.get("model_rank") == 1:
                    new_values["is_sharp"] = race_is_sharp
                await session.execute(
                    sa_update(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.id == hrow.id)
                    .values(**new_values)
                )
            await session.commit()

    # Push to immutable history if this is a pre-race prediction and not already recorded
    if not history_exists and predictions:
        first = predictions[0]
        scheduled_time = first.get("scheduled_time")
        # enriched_at is set by _prediction_to_db_dict at prediction time; fall back to now
        enriched_at = first.get("enriched_at") or datetime.utcnow()
        if not isinstance(enriched_at, datetime):
            try:
                enriched_at = datetime.fromisoformat(str(enriched_at))
            except (ValueError, TypeError):
                enriched_at = datetime.utcnow()
        is_pre_race = False
        if scheduled_time:
            try:
                sched = sched_to_utc_naive(scheduled_time)
                is_pre_race = sched is not None and enriched_at < sched
            except (ValueError, TypeError):
                pass
        if is_pre_race:
            batch_id = str(uuid.uuid4())
            now = datetime.utcnow()
            # Compute is_sharp for the rank-1 row — same gate as
            # _snapshot_prerace_predictions in main.py so history rows
            # written by this path get the flag set, not left NULL.
            # Gates: (rank-1 win_prob ≥0.30 OR top-3 sum ≥0.60)
            #        AND rank-1 days_since_last_run ≤180.
            active = [pp for pp in predictions if not pp.get("cancelled")]
            active_sorted = sorted(active, key=lambda pp: pp.get("model_rank") or 99)
            rank1 = active_sorted[0] if active_sorted else None
            top3_sum = sum((pp.get("win_probability") or 0) for pp in active_sorted[:3])
            race_is_sharp = None
            if rank1 is not None:
                _high_conf = ((rank1.get("win_probability") or 0) >= 0.30) or (top3_sum >= 0.60)
                _days_off = None
                _enriched_json = rank1.get("enriched_json")
                if _enriched_json:
                    try:
                        _e = json.loads(_enriched_json) if isinstance(_enriched_json, str) else _enriched_json
                        _days_off = _e.get("days_since_last_run") if isinstance(_e, dict) else None
                    except Exception:
                        _days_off = None
                _layoff_ok = not (isinstance(_days_off, (int, float)) and _days_off > 180)
                race_is_sharp = bool(_high_conf and _layoff_ok)
            for p in predictions:
                try:
                    session.add(RunnerPredictionHistoryRow(
                        race_id=p.get("race_id"), horse_name=p.get("horse_name"),
                        tab_number=p.get("tab_number"), barrier=p.get("barrier"),
                        jockey=p.get("jockey"), trainer=p.get("trainer"), weight=p.get("weight"),
                        win_probability=p.get("win_probability"),
                        # Snapshot the pre-multiplier raw softmax so the
                        # isotonic-on-raw calibration measurer + nightly
                        # rebuild have data to fit against. Without this,
                        # every history row written via save_race_predictions
                        # (the 8:30/10:30/11:30 AEST enrich crons + the
                        # 15-min combined cron) landed with win_prob_raw
                        # NULL — starving both _measure_isotonic_raw_sample_size
                        # and _compute_output_calibration_curve. The
                        # 09:00 AEST snapshot code path was already writing
                        # it, but that only runs when no history row exists
                        # yet for a race, which is rarely true after the
                        # 08:30 enrich has fired.
                        win_prob_raw=p.get("win_prob_raw"),
                        place_probability=p.get("place_probability"),
                        model_rank=p.get("model_rank"), market_rank=p.get("market_rank"),
                        overlay=p.get("overlay"), best_available_odds=p.get("best_available_odds"),
                        value_rating=p.get("value_rating"),
                        narrative=p.get("narrative"), key_flags=p.get("key_flags"),
                        enriched_json=p.get("enriched_json"),
                        place_model_rank=p.get("place_model_rank"),
                        exotic_model_rank=p.get("exotic_model_rank"),
                        scheduled_time=p.get("scheduled_time"),
                        enriched_at=enriched_at, cancelled=p.get("cancelled"),
                        venue=p.get("venue"), state=p.get("state"),
                        race_number=p.get("race_number"), race_name=p.get("race_name"),
                        distance=p.get("distance"), track_condition=p.get("track_condition"),
                        field_size=p.get("field_size"), prize_money=p.get("prize_money"),
                        rail_position=p.get("rail_position"), class_change=p.get("class_change"),
                        model_score=p.get("model_score"),
                        source="live",
                        batch_id=batch_id,
                        recorded_at=now,
                        # Only the rank-1 row carries the Sharp flag.
                        is_sharp=race_is_sharp if p.get("model_rank") == 1 else None,
                    ))
                except Exception:
                    pass  # unique constraint: row already exists for this race+horse, skip
            await session.commit()


async def load_race_predictions(session: AsyncSession, race_id: str) -> list[RunnerPredictionRow]:
    result = await session.execute(
        select(RunnerPredictionRow).where(RunnerPredictionRow.race_id == race_id)
        .order_by(RunnerPredictionRow.model_rank)
    )
    return list(result.scalars().all())


async def load_ra_venue_key(
    session: AsyncSession, race_date: str, state: str, clean_venue: str
) -> Optional[str]:
    """Return the cached RA key for a (date, state, venue) triple, or None
    if we've never resolved it. Survives Railway redeploys — the RAM cache
    in the RA client does not.
    """
    row = (await session.execute(
        select(RAVenueKeyCacheRow.ra_key)
        .where(RAVenueKeyCacheRow.race_date == race_date)
        .where(RAVenueKeyCacheRow.state == state)
        .where(RAVenueKeyCacheRow.clean_venue == clean_venue)
        .limit(1)
    )).scalar_one_or_none()
    return row


async def save_ra_venue_key(
    session: AsyncSession, race_date: str, state: str, clean_venue: str, ra_key: str
) -> None:
    """Persist a resolved (date, state, venue) → RA key mapping. Idempotent
    via the uq_ra_venue_key unique constraint — concurrent writers race
    without corrupting the cache."""
    from sqlalchemy import insert
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    try:
        stmt = pg_insert(RAVenueKeyCacheRow).values(
            race_date=race_date,
            state=state,
            clean_venue=clean_venue,
            ra_key=ra_key,
        ).on_conflict_do_nothing(index_elements=["race_date", "state", "clean_venue"])
        await session.execute(stmt)
        await session.commit()
    except Exception:
        # SQLite fallback path — no ON CONFLICT support, just try/except
        await session.rollback()
        try:
            session.add(RAVenueKeyCacheRow(
                race_date=race_date,
                state=state,
                clean_venue=clean_venue,
                ra_key=ra_key,
            ))
            await session.commit()
        except Exception:
            await session.rollback()


async def load_ra_form(
    session: AsyncSession, kind: str, code: str, max_age_seconds: int
) -> Optional[dict]:
    """Return a cached form payload if we have one for (kind, code) that's
    younger than max_age_seconds. Returns None if no row exists or the row
    is too old — caller falls back to a fresh RA fetch.

    Kind is 'h' (horse), 'j' (jockey), or 't' (trainer). Same one-letter
    tags used by the RA client for its RAM caches.
    """
    row = (await session.execute(
        select(RAFormCacheRow.payload_json, RAFormCacheRow.cached_at)
        .where(RAFormCacheRow.kind == kind)
        .where(RAFormCacheRow.code == code)
        .limit(1)
    )).first()
    if row is None:
        return None
    payload_json, cached_at = row
    age = (datetime.utcnow() - cached_at).total_seconds()
    if age > max_age_seconds:
        return None
    try:
        return json.loads(payload_json)
    except Exception:
        return None


async def save_ra_form(
    session: AsyncSession, kind: str, code: str, payload: dict
) -> None:
    """Persist a parsed form payload. UPSERT on (kind, code) so a re-fetch
    refreshes cached_at rather than piling up rows. Idempotent under
    concurrent writers via ON CONFLICT."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    payload_json = json.dumps(payload or {})
    now = datetime.utcnow()
    try:
        stmt = pg_insert(RAFormCacheRow).values(
            kind=kind, code=code, payload_json=payload_json, cached_at=now,
        ).on_conflict_do_update(
            index_elements=["kind", "code"],
            set_=dict(payload_json=payload_json, cached_at=now),
        )
        await session.execute(stmt)
        await session.commit()
    except Exception:
        # SQLite fallback — no ON CONFLICT DO UPDATE, so do it manually.
        await session.rollback()
        try:
            existing = (await session.execute(
                select(RAFormCacheRow).where(RAFormCacheRow.kind == kind).where(RAFormCacheRow.code == code)
            )).scalars().first()
            if existing:
                existing.payload_json = payload_json
                existing.cached_at = now
            else:
                session.add(RAFormCacheRow(kind=kind, code=code, payload_json=payload_json, cached_at=now))
            await session.commit()
        except Exception:
            await session.rollback()


async def load_model_weights(session: AsyncSession) -> dict[str, float]:
    result = await session.execute(select(ModelWeightRow))
    rows = result.scalars().all()
    return {r.feature_name: r.weight for r in rows}


async def save_model_weights(session: AsyncSession, weights: dict[str, float]) -> None:
    from sqlalchemy import delete
    await session.execute(delete(ModelWeightRow))
    for name, w in weights.items():
        session.add(ModelWeightRow(feature_name=name, weight=w))
    await session.commit()


async def save_weight_candidate(
    session: AsyncSession,
    model_type: str,
    weights: dict[str, float],
    batch_id: str,
    meta: dict,
) -> None:
    """Write a candidate weight set (all features in one batch).

    Only path that a retrain endpoint should touch. Live active tables
    (model_weights / place_model_weights / exotic_model_weights) are
    swapped by the promote endpoint, never by retrain itself.

    meta: {sample_days, sample_size, holdout_days, best_window,
           training_window_start, training_window_end}
    """
    for name, w in weights.items():
        session.add(ModelWeightCandidateRow(
            batch_id=batch_id,
            model_type=model_type,
            feature_name=name,
            weight=float(w),
            status="candidate",
            sample_days=meta.get("sample_days"),
            sample_size=meta.get("sample_size"),
            holdout_days=meta.get("holdout_days"),
            best_window=meta.get("best_window"),
            training_window_start=meta.get("training_window_start"),
            training_window_end=meta.get("training_window_end"),
        ))
    await session.commit()


async def load_weight_candidate(
    session: AsyncSession,
    batch_id: str,
) -> dict[str, float]:
    """Load a candidate batch's full weight set. Returns empty dict if
    no such batch exists."""
    result = await session.execute(
        select(ModelWeightCandidateRow).where(ModelWeightCandidateRow.batch_id == batch_id)
    )
    rows = result.scalars().all()
    return {r.feature_name: r.weight for r in rows}


async def promote_weight_candidate(
    session: AsyncSession,
    batch_id: str,
    reviewer_note: str = "",
) -> tuple[str, int]:
    """Copy a candidate batch's weights into the matching active table.
    Marks the candidate rows status='active' and any prior active
    batch's rows status='archived'. Returns (model_type, features_copied).

    Only path from candidate → live production weights.
    """
    from sqlalchemy import delete
    result = await session.execute(
        select(ModelWeightCandidateRow).where(ModelWeightCandidateRow.batch_id == batch_id)
    )
    rows = result.scalars().all()
    if not rows:
        raise ValueError(f"no candidate batch {batch_id!r}")
    model_type = rows[0].model_type

    weights = {r.feature_name: r.weight for r in rows}
    if model_type == "win":
        await session.execute(delete(ModelWeightRow))
        for name, w in weights.items():
            session.add(ModelWeightRow(feature_name=name, weight=w))
    elif model_type == "place":
        await session.execute(delete(PlaceModelWeightRow))
        for name, w in weights.items():
            session.add(PlaceModelWeightRow(feature_name=name, weight=w))
    elif model_type == "exotic":
        await session.execute(delete(ExoticModelWeightRow))
        for name, w in weights.items():
            session.add(ExoticModelWeightRow(feature_name=name, weight=w))
    else:
        raise ValueError(f"unknown model_type {model_type!r}")

    # Archive prior active batch(es) for this model type
    prior = (await session.execute(
        select(ModelWeightCandidateRow)
        .where(ModelWeightCandidateRow.model_type == model_type)
        .where(ModelWeightCandidateRow.status == "active")
    )).scalars().all()
    for p in prior:
        p.status = "archived"

    now = datetime.utcnow()
    for r in rows:
        r.status = "active"
        r.reviewed_at = now
        r.reviewed_note = reviewer_note or "promoted"
    await session.commit()
    return model_type, len(weights)


async def reject_weight_candidate(
    session: AsyncSession,
    batch_id: str,
    reviewer_note: str = "",
) -> int:
    """Mark all rows in a candidate batch as rejected. Returns rows affected."""
    result = await session.execute(
        select(ModelWeightCandidateRow).where(ModelWeightCandidateRow.batch_id == batch_id)
    )
    rows = result.scalars().all()
    if not rows:
        raise ValueError(f"no candidate batch {batch_id!r}")
    now = datetime.utcnow()
    for r in rows:
        r.status = "rejected"
        r.reviewed_at = now
        r.reviewed_note = reviewer_note or "rejected"
    await session.commit()
    return len(rows)


async def load_place_model_weights(session: AsyncSession) -> dict[str, float]:
    result = await session.execute(select(PlaceModelWeightRow))
    rows = result.scalars().all()
    return {r.feature_name: r.weight for r in rows}


async def save_place_model_weights(session: AsyncSession, weights: dict[str, float]) -> None:
    from sqlalchemy import delete
    await session.execute(delete(PlaceModelWeightRow))
    for name, w in weights.items():
        session.add(PlaceModelWeightRow(feature_name=name, weight=w))
    await session.commit()


async def load_exotic_model_weights(session: AsyncSession) -> dict[str, float]:
    result = await session.execute(select(ExoticModelWeightRow))
    rows = result.scalars().all()
    return {r.feature_name: r.weight for r in rows}


async def save_exotic_model_weights(session: AsyncSession, weights: dict[str, float]) -> None:
    from sqlalchemy import delete
    await session.execute(delete(ExoticModelWeightRow))
    for name, w in weights.items():
        session.add(ExoticModelWeightRow(feature_name=name, weight=w))
    await session.commit()
