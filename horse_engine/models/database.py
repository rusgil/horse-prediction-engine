"""SQLAlchemy ORM models and DB helpers."""
from __future__ import annotations

import json
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


async def init_db() -> None:
    import logging as _logging
    _log = _logging.getLogger(__name__)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            for col in [
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
                "CREATE INDEX IF NOT EXISTS ix_runner_pred_race_rank ON runner_predictions (race_id, model_rank)",
                "CREATE INDEX IF NOT EXISTS ix_runner_pred_hist_race_rank ON runner_prediction_history (race_id, model_rank)",
                "CREATE INDEX IF NOT EXISTS ix_hist_results_race_winner ON historical_results (race_id, winner)",
                "CREATE INDEX IF NOT EXISTS ix_hist_results_race_placed ON historical_results (race_id, placed)",
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
            ]:
                await conn.execute(text(col))
    except Exception as e:
        _log.warning("[init_db] schema migration skipped — will retry next deploy: %s", e)


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

    # Write to mutable table
    await session.execute(delete(RunnerPredictionRow).where(RunnerPredictionRow.race_id == race_id))
    rows = []
    for p in predictions:
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
        enriched_at = first.get("enriched_at") or datetime.utcnow()
        is_pre_race = False
        if scheduled_time and enriched_at:
            try:
                sched = datetime.fromisoformat(str(scheduled_time).replace("Z", "+00:00")).replace(tzinfo=None)
                ea = enriched_at if isinstance(enriched_at, datetime) else datetime.fromisoformat(str(enriched_at))
                is_pre_race = ea < sched
            except (ValueError, TypeError):
                pass
        if is_pre_race:
            for p in predictions:
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
                    recorded_at=datetime.utcnow(),
                ))
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
