"""SQLAlchemy ORM models and DB helpers."""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import Column, Float, Integer, String, Text, Boolean, DateTime, select, text
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
    value_roi = Column(Float)
    value_bets = Column(Integer)
    total_races = Column(Integer)
    drift_flag = Column(Boolean, default=False)
    drift_reason = Column(Text, nullable=True)
    all_results_json = Column(Text)         # JSON list of per-window stats


class WinCalibrationCurveRow(Base):
    """Persisted isotonic calibration curve for the win model output.

    Recomputed nightly from the last N days of predictions vs actuals.
    curve_json holds a list of (input_pct, calibrated_pct) breakpoints
    in monotone-non-decreasing order — see
    horse_engine.prediction.output_calibration for the format.

    Single-row table (id=1) updated in place; keeps history in
    updated_at rather than as multiple rows so the loader can always
    do a scalar-one read.
    """
    __tablename__ = "win_calibration_curve"

    id = Column(Integer, primary_key=True, autoincrement=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    sample_days = Column(Integer)          # holdout window used
    sample_size = Column(Integer)          # total (pct, won) pairs fit
    curve_json = Column(Text)              # JSON list of [input_pct, calibrated_pct]


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
        "CREATE INDEX IF NOT EXISTS ix_hist_batch_id ON runner_prediction_history (batch_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_history_race_horse ON runner_prediction_history (race_id, horse_name)",
        "CREATE INDEX IF NOT EXISTS ix_runner_pred_race_rank ON runner_predictions (race_id, model_rank)",
        "CREATE INDEX IF NOT EXISTS ix_runner_pred_hist_race_rank ON runner_prediction_history (race_id, model_rank)",
        # OBS-G: composite (source, race_id, cancelled) supports the very common
        # premium / edge / calibration / backtest reads, which all filter on
        # source='live' + cancelled NULL/false and scan races_id ranges. Postgres
        # can ignore the prefix here; SQLite uses leftmost prefix matching so the
        # ordering still benefits common workloads.
        "CREATE INDEX IF NOT EXISTS ix_hist_source_race_cancelled ON runner_prediction_history (source, race_id, cancelled)",
        "CREATE INDEX IF NOT EXISTS ix_hist_results_race_winner ON historical_results (race_id, winner)",
        "CREATE INDEX IF NOT EXISTS ix_hist_results_race_placed ON historical_results (race_id, placed)",
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
            sched = datetime.fromisoformat(row.scheduled_time.replace("Z", "+00:00")).replace(tzinfo=None)
            if row.enriched_at < sched and row.race_id not in already_set:
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


async def save_race_predictions(session: AsyncSession, race_id: str, predictions: list[dict]) -> None:
    from sqlalchemy import delete, func

    # Check if an immutable history snapshot already exists for this race
    history_exists = (await session.execute(
        select(func.count()).select_from(RunnerPredictionHistoryRow)
        .where(RunnerPredictionHistoryRow.race_id == race_id)
    )).scalar() or 0

    # Block post-race re-enrichment only when a pre-race snapshot already exists.
    # If history_exists=False the race has never been snapshotted — allow the write
    # so late-listed meetings can still get their first prediction captured.
    if predictions and history_exists:
        scheduled_time = predictions[0].get("scheduled_time")
        if scheduled_time:
            try:
                sched = datetime.fromisoformat(str(scheduled_time).replace("Z", "+00:00")).replace(tzinfo=None)
                if datetime.utcnow() > sched:
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

    # Auto-cancel duplicates — bidirectional.
    # A horse should only appear in the HIGHEST-numbered race it's nominated in.
    # Cancel earlier race entries when this race is saved, and cancel THIS race's entry
    # if the horse already has an uncancelled entry in a later race (handles re-enrichment
    # of an earlier race after the later race was already saved).
    date_venue = race_id.rsplit("_R", 1)[0]  # e.g. "2026-06-09_scone"
    horse_names = [p.get("horse_name") for p in predictions if p.get("horse_name")]
    try:
        current_race_num = int(race_id.rsplit("_R", 1)[1])
    except (IndexError, ValueError):
        current_race_num = 0
    if horse_names and date_venue:
        from sqlalchemy import update as sa_update
        # 1. Cancel same horses in any EARLIER race
        await session.execute(
            sa_update(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{date_venue}_R%"))
            .where(RunnerPredictionRow.race_id != race_id)
            .where(RunnerPredictionRow.horse_name.in_(horse_names))
            .where(RunnerPredictionRow.race_number < current_race_num)
            .values(cancelled=True)
        )
        # 2. If any of these horses already exist uncancelled in a LATER race,
        #    cancel them in THIS race (re-enrichment of an earlier race must not
        #    resurrect a horse that was correctly moved to a higher race number).
        in_later = (await session.execute(
            select(RunnerPredictionRow.horse_name)
            .where(RunnerPredictionRow.race_id.like(f"{date_venue}_R%"))
            .where(RunnerPredictionRow.race_id != race_id)
            .where(RunnerPredictionRow.horse_name.in_(horse_names))
            .where(RunnerPredictionRow.race_number > current_race_num)
            .where(
                RunnerPredictionRow.cancelled.is_(False)
                | RunnerPredictionRow.cancelled.is_(None)
            )
        )).scalars().all()
        if in_later:
            await session.execute(
                sa_update(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id == race_id)
                .where(RunnerPredictionRow.horse_name.in_(in_later))
                .values(cancelled=True)
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
                sched = datetime.fromisoformat(str(scheduled_time).replace("Z", "+00:00")).replace(tzinfo=None)
                is_pre_race = enriched_at < sched
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
