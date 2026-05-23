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
    enriched_at = Column(DateTime, default=datetime.utcnow)
    cancelled = Column(Boolean, default=False, nullable=True)


class ModelWeightRow(Base):
    __tablename__ = "model_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, unique=True)
    weight = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


class HistoricalResultRow(Base):
    __tablename__ = "historical_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, index=True)
    horse_name = Column(String)
    position = Column(Integer)
    beaten_margin = Column(Float)
    winner = Column(Boolean)
    placed = Column(Boolean)
    starting_price = Column(Float, nullable=True)
    feature_vector_json = Column(Text, nullable=True)   # for retraining
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


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Idempotent column migrations for existing tables
        await conn.execute(text(
            "ALTER TABLE runner_predictions ADD COLUMN IF NOT EXISTS cancelled BOOLEAN DEFAULT FALSE"
        ))


async def save_race_predictions(session: AsyncSession, race_id: str, predictions: list[dict]) -> None:
    from sqlalchemy import delete
    await session.execute(delete(RunnerPredictionRow).where(RunnerPredictionRow.race_id == race_id))
    for p in predictions:
        row = RunnerPredictionRow(**p)
        session.add(row)
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
