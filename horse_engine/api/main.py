"""
FunkyIQ Horse Prediction Engine — FastAPI application.

Endpoints:
  GET  /api/meetings/{date}                   — list all thoroughbred meetings
  GET  /api/meetings/{date}/{venue}           — meeting detail with all races
  GET  /api/races/{race_id}                   — full race prediction
  POST /api/races/{race_id}/enrich            — trigger enrichment for one race
  POST /api/meetings/{date}/{venue}/enrich    — enrich all races at a meeting
  POST /api/retrain                           — retrain model on historical results
  POST /api/admin/results/{date}             — seed race results (training data)
  POST /api/cron/enrich                       — daily cron enrichment
  GET  /api/health                            — liveness check
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from horse_engine.api.database import get_session
from horse_engine.clients.tab import TABClient
from horse_engine.config import settings
from horse_engine.models.database import (
    HistoricalResultRow,
    RunnerPredictionRow,
    RacePredictionRow,
    init_db,
    load_model_weights,
    save_model_weights,
    save_race_predictions,
)
from horse_engine.pipeline import enrich_and_predict_race, enrich_meeting
from horse_engine.prediction.model import HorseModel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="FunkyIQ Horse Prediction Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _load_model(session) -> HorseModel:
    weights = await load_model_weights(session)
    if weights:
        return HorseModel.from_weights_dict(weights)
    return HorseModel()


def _today() -> str:
    return date.today().isoformat()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ── Meetings ──────────────────────────────────────────────────────────────────

@app.get("/api/meetings/{race_date}")
async def list_meetings(race_date: str = _today()):
    """List all thoroughbred meetings for the given date."""
    client = TABClient()
    try:
        meetings = await client.get_meetings(race_date)
    except Exception as e:
        raise HTTPException(502, f"TAB API error: {e}")

    return {
        "date": race_date,
        "meetings": [
            {
                "venue": m.get("venueName"),
                "venue_code": m.get("meetingCode"),
                "state": m.get("location", {}).get("state"),
                "track_condition": m.get("trackCondition", {}).get("name"),
                "races": m.get("numRaces"),
                "first_race": m.get("raceStartTime"),
            }
            for m in meetings
        ],
    }


@app.get("/api/meetings/{race_date}/{venue_code}")
async def get_meeting(race_date: str, venue_code: str):
    """Get all races at a meeting with current predictions if available."""
    async with get_session() as session:
        result = await session.execute(
            select(RacePredictionRow)
            .where(RacePredictionRow.date == race_date)
            .where(RacePredictionRow.venue == venue_code)
            .order_by(RacePredictionRow.race_number)
        )
        races = result.scalars().all()

    if not races:
        # Fall through to TAB for bare race card (no predictions yet)
        client = TABClient()
        raw_races = await client.get_meeting_races(race_date, venue_code)
        return {
            "date": race_date,
            "venue": venue_code,
            "enriched": False,
            "races": [
                {
                    "race_id": f"{race_date}_{venue_code}_R{r.get('raceNumber')}",
                    "race_number": r.get("raceNumber"),
                    "race_name": r.get("raceName"),
                    "distance": r.get("raceDistance"),
                    "time": r.get("raceStartTime"),
                }
                for r in raw_races
            ],
        }

    return {
        "date": race_date,
        "venue": venue_code,
        "enriched": True,
        "races": [_race_summary(r) for r in races],
    }


# ── Races ─────────────────────────────────────────────────────────────────────

@app.get("/api/races/{race_id}")
async def get_race(race_id: str):
    """Return full race prediction for a given race_id."""
    async with get_session() as session:
        result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id == race_id)
            .order_by(RunnerPredictionRow.model_rank)
        )
        runners = result.scalars().all()

    if not runners:
        raise HTTPException(404, f"No predictions for race {race_id}. Trigger /enrich first.")

    return {
        "race_id": race_id,
        "runners": [_runner_response(r) for r in runners],
    }


@app.post("/api/races/{race_id}/enrich")
async def enrich_race(race_id: str, force: bool = Query(False)):
    """Enrich a specific race. race_id format: {date}_{venue_code}_R{num}"""
    parts = race_id.split("_")
    if len(parts) < 3:
        raise HTTPException(400, "Invalid race_id format. Expected: {date}_{venue}_R{num}")

    race_date = parts[0]
    venue_code = parts[1]
    try:
        race_num = int(parts[2].replace("R", ""))
    except ValueError:
        raise HTTPException(400, "Invalid race number in race_id")

    async with get_session() as session:
        # Check if already enriched (unless force)
        if not force:
            existing = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id == race_id)
                .limit(1)
            )
            if existing.scalars().first():
                return {"status": "already_enriched", "race_id": race_id}

        model = await _load_model(session)

    try:
        tab = TABClient()
        raw_race = await tab.get_race(race_date, venue_code, race_num)
        if not raw_race:
            raise HTTPException(404, f"Race not found: {race_id}")

        # Determine meeting metadata
        meetings = await tab.get_meetings(race_date)
        meeting_meta = next((m for m in meetings if m.get("meetingCode") == venue_code), {})
        venue_name = meeting_meta.get("venueName", venue_code)
        state = meeting_meta.get("location", {}).get("state", "")
        track_condition = meeting_meta.get("trackCondition", {}).get("name", "Good 4")

        race = tab.parse_race(raw_race, race_date, venue_name, state)
        race.track_condition = track_condition

        predictions, race_row_data = await enrich_and_predict_race(race, model)

        async with get_session() as session:
            await save_race_predictions(
                session,
                race_id,
                [_prediction_to_db_dict(p, race_id) for p in predictions],
            )

        return {
            "status": "enriched",
            "race_id": race_id,
            "runners": len(predictions),
            "top_pick": predictions[0].runner.horse_name if predictions else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Enrich failed for %s", race_id)
        raise HTTPException(500, str(e))


@app.post("/api/meetings/{race_date}/{venue_code}/enrich")
async def enrich_meeting_endpoint(race_date: str, venue_code: str):
    """Enrich all races at a meeting."""
    tab = TABClient()
    raw_races = await tab.get_meeting_races(race_date, venue_code)
    if not raw_races:
        raise HTTPException(404, f"No races found for {venue_code} on {race_date}")

    meetings = await tab.get_meetings(race_date)
    meeting_meta = next((m for m in meetings if m.get("meetingCode") == venue_code), {})
    venue_name = meeting_meta.get("venueName", venue_code)
    state = meeting_meta.get("location", {}).get("state", "")

    results = []
    async with get_session() as session:
        model = await _load_model(session)

    for raw_race in raw_races:
        race_num = raw_race.get("raceNumber")
        race_id = f"{race_date}_{venue_code}_R{race_num}"
        try:
            full_raw = await tab.get_race(race_date, venue_code, race_num)
            if not full_raw:
                continue
            race = tab.parse_race(full_raw, race_date, venue_name, state)
            predictions, _ = await enrich_and_predict_race(race, model)
            async with get_session() as session:
                await save_race_predictions(
                    session,
                    race_id,
                    [_prediction_to_db_dict(p, race_id) for p in predictions],
                )
            results.append({"race_id": race_id, "status": "ok", "runners": len(predictions)})
        except Exception as e:
            log.warning("Failed to enrich %s: %s", race_id, e)
            results.append({"race_id": race_id, "status": "error", "error": str(e)})

    return {"venue": venue_code, "date": race_date, "races": results}


# ── Retrain ───────────────────────────────────────────────────────────────────

@app.post("/api/retrain")
async def retrain_model():
    """Retrain logistic regression on stored historical results."""
    async with get_session() as session:
        result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.feature_vector_json.isnot(None))
        )
        rows = result.scalars().all()

    if len(rows) < 50:
        raise HTTPException(400, f"Need at least 50 labelled results to retrain (have {len(rows)})")

    training_data = []
    for row in rows:
        try:
            fv = json.loads(row.feature_vector_json)
            label = 1 if row.winner else 0
            training_data.append((fv, label))
        except Exception:
            continue

    model = HorseModel()
    stats = model.train(training_data)

    async with get_session() as session:
        await save_model_weights(session, {k: v for k, v in zip(
            [k for k in stats["weights"]], [v for v in stats["weights"].values()]
        )})

    return {"status": "retrained", **stats}


# ── Admin: seed results ───────────────────────────────────────────────────────

@app.post("/api/admin/results/{race_date}")
async def seed_results(race_date: str, x_cron_secret: Optional[str] = Header(None)):
    """
    Fetch race results from TAB for a past date and store as training data.
    Matched against stored RunnerPrediction feature vectors.
    """
    # Basic auth for admin
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")

    tab = TABClient()
    meetings = await tab.get_meetings(race_date)
    seeded = 0

    for meeting in meetings:
        venue_code = meeting.get("meetingCode")
        raw_races = await tab.get_meeting_races(race_date, venue_code)
        for raw_race in raw_races:
            race_num = raw_race.get("raceNumber")
            race_id = f"{race_date}_{venue_code}_R{race_num}"

            full = await tab.get_race(race_date, venue_code, race_num)
            if not full:
                continue

            for runner_raw in full.get("runners", []):
                if runner_raw.get("scratched"):
                    continue
                position = runner_raw.get("finishingPosition") or runner_raw.get("position")
                if not position:
                    continue
                horse = runner_raw.get("runnerName", "")
                sp = runner_raw.get("startingPrice") or runner_raw.get("spPrice")
                beaten = runner_raw.get("margin", 0)

                # Look up stored feature vector
                async with get_session() as session:
                    fv_result = await session.execute(
                        select(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == race_id)
                        .where(RunnerPredictionRow.horse_name == horse)
                        .limit(1)
                    )
                    fv_row = fv_result.scalars().first()
                    fv_json = fv_row.enriched_json if fv_row else None

                    row = HistoricalResultRow(
                        race_id=race_id,
                        horse_name=horse,
                        position=int(position),
                        beaten_margin=float(beaten or 0),
                        winner=int(position) == 1,
                        placed=int(position) <= 3,
                        starting_price=float(sp) if sp else None,
                        feature_vector_json=fv_json,
                    )
                    session.add(row)
                    await session.commit()
                    seeded += 1

    return {"status": "seeded", "results": seeded}


# ── Cron ──────────────────────────────────────────────────────────────────────

@app.post("/api/cron/enrich")
async def cron_enrich(x_cron_secret: Optional[str] = Header(None)):
    """Daily cron: enrich all today's meetings."""
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")

    today = _today()
    tab = TABClient()
    meetings = await tab.get_meetings(today)

    summary = []
    async with get_session() as session:
        model = await _load_model(session)

    for m in meetings:
        venue_code = m.get("meetingCode")
        try:
            raw_races = await tab.get_meeting_races(today, venue_code)
            venue_name = m.get("venueName", venue_code)
            state = m.get("location", {}).get("state", "")

            for raw_race in raw_races:
                race_num = raw_race.get("raceNumber")
                race_id = f"{today}_{venue_code}_R{race_num}"
                full_raw = await tab.get_race(today, venue_code, race_num)
                if not full_raw:
                    continue
                race = tab.parse_race(full_raw, today, venue_name, state)
                predictions, _ = await enrich_and_predict_race(race, model)
                async with get_session() as session:
                    await save_race_predictions(
                        session,
                        race_id,
                        [_prediction_to_db_dict(p, race_id) for p in predictions],
                    )
            summary.append({"venue": venue_code, "status": "ok"})
        except Exception as e:
            summary.append({"venue": venue_code, "status": "error", "error": str(e)})

    return {"date": today, "summary": summary}


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _prediction_to_db_dict(pred, race_id: str) -> dict:
    return {
        "race_id": race_id,
        "horse_name": pred.runner.horse_name,
        "tab_number": pred.runner.tab_number,
        "barrier": pred.runner.barrier,
        "jockey": pred.runner.jockey,
        "trainer": pred.runner.trainer,
        "weight": pred.runner.weight,
        "win_probability": round(pred.win_prob, 4),
        "place_probability": round(pred.place_prob, 4),
        "model_rank": pred.model_rank,
        "market_rank": pred.enriched.market_rank,
        "overlay": round(pred.overlay, 4),
        "best_available_odds": pred.enriched.best_available_odds,
        "value_rating": round(pred.value_rating, 4),
        "narrative": pred.narrative,
        "key_flags": json.dumps(pred.key_flags),
        "enriched_json": pred.enriched.model_dump_json(),
    }


def _runner_response(row: RunnerPredictionRow) -> dict:
    enriched = {}
    if row.enriched_json:
        try:
            enriched = json.loads(row.enriched_json)
        except Exception:
            pass
    return {
        "tab_number": row.tab_number,
        "barrier": row.barrier,
        "horse_name": row.horse_name,
        "jockey": row.jockey,
        "trainer": row.trainer,
        "weight": row.weight,
        "model_rank": row.model_rank,
        "market_rank": row.market_rank,
        "win_probability": row.win_probability,
        "place_probability": row.place_probability,
        "best_available_odds": row.best_available_odds,
        "overlay": row.overlay,
        "value_rating": row.value_rating,
        "narrative": row.narrative,
        "key_flags": json.loads(row.key_flags or "[]"),
        "form_score": enriched.get("form_score"),
        "distance_aptitude": enriched.get("distance_aptitude"),
        "sire_name": enriched.get("sire_name"),
        "pedigree_distance_match": enriched.get("pedigree_distance_match"),
        "pedigree_wet_score": enriched.get("pedigree_wet_score"),
        "speed_map_position": enriched.get("speed_map_position"),
        "is_steamed": enriched.get("is_steamed", False),
        "is_drifted": enriched.get("is_drifted", False),
        "trainer_overall_rate": enriched.get("trainer_overall_rate"),
        "jockey_overall_rate": enriched.get("jockey_overall_rate"),
        "days_since_last_run": enriched.get("days_since_last_run"),
        "runs_this_prep": enriched.get("runs_this_prep"),
        "wet_track_record": enriched.get("wet_track_record"),
        "dosage_index": enriched.get("dosage_index"),
        "track_condition_category": enriched.get("track_condition_category"),
        "enriched_at": row.enriched_at.isoformat() if row.enriched_at else None,
    }


def _race_summary(row: RacePredictionRow) -> dict:
    runners_raw = []
    if row.runners_json:
        try:
            runners_raw = json.loads(row.runners_json)
        except Exception:
            pass
    return {
        "race_id": row.race_id,
        "race_number": row.race_number,
        "race_name": row.race_name,
        "distance": row.distance,
        "track_condition": row.track_condition,
        "prize_money": row.prize_money,
        "scheduled_time": row.scheduled_time,
        "field_size": row.field_size,
        "enriched_at": row.enriched_at.isoformat() if row.enriched_at else None,
    }
