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

Venue codes are the punters.com.au venue slug, e.g. "werribee", "randwick", "flemington".
Meeting slugs follow the pattern "{venue}-{date}" e.g. "werribee-20260514".
Race IDs follow the pattern "{date}_{venue}_R{num}" e.g. "2026-05-14_werribee_R3".
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from horse_engine.api.database import get_session
from horse_engine.clients.factory import get_tab_client
from horse_engine.config import settings
from horse_engine.models.database import (
    CalibrationRow,
    HistoricalResultRow,
    RunnerPredictionRow,
    RacePredictionRow,
    init_db,
    load_model_weights,
    save_model_weights,
    save_race_predictions,
)
from horse_engine.models.enriched import EnrichedRunner
from horse_engine.pipeline import enrich_and_predict_race, enrich_meeting
from horse_engine.prediction.features import build_feature_vector
from horse_engine.prediction.model import HorseModel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


async def _scheduled_enrich():
    """Run by APScheduler — enrich today + next 2 days."""
    log.info("[scheduler] Running scheduled enrichment")
    try:
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)
        for i in range(3):
            race_date = (date.today() + timedelta(days=i)).isoformat()
            log.info("[scheduler] Enriching %s", race_date)
            await _enrich_date(race_date, client, model)
        log.info("[scheduler] Enrichment complete")
    except Exception as e:
        log.exception("[scheduler] Enrichment failed: %s", e)


async def _seed_results_for_date(race_date: str) -> int:
    """Fetch settled results for race_date and store as training data. Returns count seeded."""
    client = get_tab_client()
    meetings = await client.get_meetings(race_date)
    seeded = 0
    date_sfx = f"-{race_date.replace('-', '')}"
    for meeting in meetings:
        slug = meeting.get("slug", "")
        venue_code = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else ""
        raw_events = await client.get_meeting_races(slug)
        for raw_event in raw_events:
            race_num = raw_event.get("eventNumber")
            race_id = f"{race_date}_{venue_code}_R{race_num}"
            full_event = await client.get_race(slug, race_num)
            if not full_event:
                continue
            for sel in full_event.get("selections", []):
                if (sel.get("status") or "").upper() == "SCRATCHED":
                    continue
                position = sel.get("selectionResult")
                if not position or int(position) <= 0:
                    continue
                horse = (sel.get("competitor") or {}).get("name", "")
                sp = sel.get("startingPrice")
                beaten = sel.get("officialMargin", 0)
                async with get_session() as session:
                    existing = await session.execute(
                        select(HistoricalResultRow)
                        .where(HistoricalResultRow.race_id == race_id)
                        .where(HistoricalResultRow.horse_name == horse)
                        .limit(1)
                    )
                    if existing.scalars().first():
                        continue
                    fv_result = await session.execute(
                        select(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == race_id)
                        .where(RunnerPredictionRow.horse_name == horse)
                        .limit(1)
                    )
                    fv_row = fv_result.scalars().first()
                    session.add(HistoricalResultRow(
                        race_id=race_id,
                        horse_name=horse,
                        position=int(position),
                        beaten_margin=float(beaten or 0),
                        winner=int(position) == 1,
                        placed=int(position) <= 3,
                        starting_price=float(sp) if sp else None,
                        feature_vector_json=fv_row.enriched_json if fv_row else None,
                    ))
                    await session.commit()
                    seeded += 1
    return seeded


async def _scheduled_seed_results():
    """Run by APScheduler nightly — seed yesterday's settled results."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    log.info("[scheduler] Seeding results for %s", yesterday)
    try:
        n = await _seed_results_for_date(yesterday)
        log.info("[scheduler] Seeded %d results for %s", n, yesterday)
    except Exception as e:
        log.exception("[scheduler] Result seeding failed for %s: %s", yesterday, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Schedule enrichment at 6am, 10am, 1pm AEST (UTC+10 = subtract 10h)
    scheduler = AsyncIOScheduler(timezone="Australia/Sydney")
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=6,  minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=10, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=13, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=23, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_calibrate, CronTrigger(day_of_week="sun", hour=2, minute=0, timezone="Australia/Sydney"))
    scheduler.start()
    log.info("[scheduler] Cron jobs scheduled: 6am/10am/1pm enrich, 11pm seed results, 2am Sun calibration")

    # Enrich today on startup so deploys don't leave races un-loaded
    asyncio.create_task(_scheduled_enrich())

    yield

    scheduler.shutdown()
    log.info("[scheduler] Shutdown")


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


def _meeting_slug(venue: str, race_date: str) -> str:
    """Build the punters.com.au meeting slug from venue slug and date."""
    return f"{venue}-{race_date.replace('-', '')}"


# ── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse("frontend/index.html")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ── Meetings ──────────────────────────────────────────────────────────────────

@app.get("/api/meetings/{race_date}")
async def list_meetings(race_date: str = _today()):
    """List all Australian thoroughbred meetings for the given date."""
    client = get_tab_client()
    try:
        meetings = await client.get_meetings(race_date)
    except Exception as e:
        raise HTTPException(502, f"Data fetch error: {e}")

    date_suffix = f"-{race_date.replace('-', '')}"
    return {
        "date": race_date,
        "meetings": [
            {
                "venue": m.get("venue"),
                "venue_code": (
                    m["slug"][: -len(date_suffix)]
                    if (slug := m.get("slug", "")) and slug.endswith(date_suffix)
                    else slug.split("-")[0] if slug else m.get("name", "").lower().replace(" ", "-")
                ),
                "state": m.get("state"),
                "rail_position": m.get("rail_position"),
                "slug": m.get("slug"),
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
        client = get_tab_client()
        slug = _meeting_slug(venue_code, race_date)
        raw_races = await client.get_meeting_races(slug)
        return {
            "date": race_date,
            "venue": venue_code,
            "enriched": False,
            "races": [
                {
                    "race_id": f"{race_date}_{venue_code}_R{r.get('eventNumber')}",
                    "race_number": r.get("eventNumber"),
                    "race_name": r.get("name"),
                    "distance": r.get("distance"),
                    "time": r.get("startTime"),
                    "status": r.get("status"),
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


@app.get("/api/races/{race_id}/live-odds")
async def live_odds(race_id: str):
    """
    Re-fetch current tote odds from punters for a race and compute updated overlays.
    Fast (~1s) — does not regenerate model predictions, just refreshes market data.
    """
    parts = race_id.split("_")
    if len(parts) < 3:
        raise HTTPException(400, "Invalid race_id format")

    race_date = parts[0]
    venue_code = parts[1]
    try:
        race_num = int(parts[2].replace("R", ""))
    except ValueError:
        raise HTTPException(400, "Invalid race number in race_id")

    # Load stored model predictions
    async with get_session() as session:
        result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id == race_id)
            .order_by(RunnerPredictionRow.model_rank)
        )
        stored = result.scalars().all()

    if not stored:
        raise HTTPException(404, f"No predictions for {race_id} — enrich first")

    model_probs = {r.horse_name: r.win_probability or 0.0 for r in stored}

    # Fetch current odds from punters
    try:
        client = get_tab_client()
        slug = _meeting_slug(venue_code, race_date)
        raw_event = await client.get_race(slug, race_num)
    except Exception as e:
        raise HTTPException(502, f"Could not fetch live odds: {e}")

    if not raw_event:
        raise HTTPException(404, "Race not found on punters")

    # Build updated odds snapshot — also capture settled result if available
    runners_odds = []
    all_tote = []
    for sel in raw_event.get("selections", []):
        if (sel.get("status") or "").upper() == "SCRATCHED":
            continue
        horse = (sel.get("competitor") or {}).get("name", "")
        tote_win = sel.get("topToteWin")
        sp = sel.get("startingPrice")
        current_odds = float(tote_win) if tote_win else (float(sp) if sp else None)
        result_pos = sel.get("selectionResult")
        actual_position = int(result_pos) if result_pos and int(result_pos) > 0 else None
        all_tote.append((horse, current_odds, actual_position))

    # Overround-free implied probs from current tote
    valid_odds = [o for _, o, _ in all_tote if o and o > 1.0]
    total_implied = sum(1 / o for o in valid_odds) if valid_odds else 0
    scale = 1.0 / total_implied if total_implied > 0 else 1.0

    # Find winner for model-correct flag
    winner_name = next((h for h, _, pos in all_tote if pos == 1), None)
    top_model_pick = stored[0].horse_name if stored else None
    model_correct = (winner_name == top_model_pick) if winner_name else None

    for horse, current_odds, actual_position in all_tote:
        model_prob = model_probs.get(horse, 0.0)
        if current_odds and current_odds > 1.0:
            raw_implied = 1.0 / current_odds
            orf_implied = round(raw_implied * scale, 4)
        else:
            orf_implied = 0.0
        overlay = round(model_prob - orf_implied, 4) if orf_implied else 0.0
        runners_odds.append({
            "horse_name": horse,
            "current_tote_win": current_odds,
            "implied_prob": orf_implied,
            "model_win_prob": round(model_prob, 4),
            "overlay": overlay,
            "value": overlay > 0.05 and current_odds and current_odds >= 3.0,
            "actual_position": actual_position,
        })

    runners_odds.sort(key=lambda x: x["model_win_prob"], reverse=True)

    return {
        "race_id": race_id,
        "fetched_at": datetime.utcnow().isoformat(),
        "settled": winner_name is not None,
        "winner": winner_name,
        "model_correct": model_correct,
        "runners": runners_odds,
    }


@app.post("/api/races/{race_id}/enrich")
async def enrich_race(race_id: str, force: bool = Query(False)):
    """Enrich a specific race. race_id format: {date}_{venue}_R{num}"""
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
        client = get_tab_client()
        slug = _meeting_slug(venue_code, race_date)
        raw_event = await client.get_race(slug, race_num)
        if not raw_event:
            raise HTTPException(404, f"Race not found: {race_id}")

        meeting = raw_event.get("_meeting", {})
        venue_obj = meeting.get("venue") or {}
        venue_name = venue_obj.get("name", venue_code)
        state = venue_obj.get("state", "")

        race = client.parse_race(raw_event, race_date, venue_name, state)
        predictions, _ = await enrich_and_predict_race(race, model)

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
    client = get_tab_client()
    slug = _meeting_slug(venue_code, race_date)

    raw_events = await client.get_meeting_races(slug)
    if not raw_events:
        raise HTTPException(404, f"No races found for {venue_code} on {race_date}")

    meeting_detail = await client.get_meeting_by_slug(slug)
    venue_obj = (meeting_detail or {}).get("venue") or {}
    venue_name = venue_obj.get("name", venue_code)
    state = venue_obj.get("state", "")

    results = []
    async with get_session() as session:
        model = await _load_model(session)

    for raw_event in raw_events:
        race_num = raw_event.get("eventNumber")
        race_id = f"{race_date}_{venue_code}_R{race_num}"
        try:
            full_event = await client.get_race(slug, race_num)
            if not full_event:
                continue
            race = client.parse_race(full_event, race_date, venue_name, state)
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
async def retrain_model(days: int = Query(0, ge=0, le=365)):
    """
    Retrain logistic regression on stored historical results.
    days=0 (default) uses all available data. days=N uses only the last N days.
    """
    async with get_session() as session:
        hr_query = select(HistoricalResultRow)
        if days > 0:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            hr_query = hr_query.where(HistoricalResultRow.race_id >= cutoff)
        hr_result = await session.execute(hr_query)
        hr_rows = hr_result.scalars().all()

        pred_result = await session.execute(
            select(RunnerPredictionRow).where(RunnerPredictionRow.enriched_json.isnot(None))
        )
        pred_rows = pred_result.scalars().all()

    if len(hr_rows) < 50:
        raise HTTPException(400, f"Need at least 50 labelled results to retrain (have {len(hr_rows)})")

    # Join on (race_id, horse_name) — enriched_json lives on the prediction row
    pred_by_key = {(p.race_id, p.horse_name): p for p in pred_rows}

    training_data = []
    for row in hr_rows:
        pred = pred_by_key.get((row.race_id, row.horse_name))
        if not pred:
            continue
        try:
            er = EnrichedRunner(**json.loads(pred.enriched_json))
            fv = build_feature_vector(er)
            label = 1 if row.winner else 0
            training_data.append((fv, label))
        except Exception as e:
            log.debug("Skipping retrain row %s/%s: %s", row.race_id, row.horse_name, e)

    if not training_data:
        raise HTTPException(400, f"No matched training examples (have {len(hr_rows)} results, {len(pred_rows)} predictions — check race_id/horse_name alignment)")

    model = HorseModel()
    stats = model.train(training_data)

    async with get_session() as session:
        await save_model_weights(session, stats["weights"])

    return {"status": "retrained", "training_days": days or "all", "training_examples": len(training_data), **stats}


# ── Admin: seed results ───────────────────────────────────────────────────────

@app.post("/api/admin/results/{race_date}")
async def seed_results(race_date: str, x_cron_secret: Optional[str] = Header(None)):
    """Fetch race results from punters for a past date and store as training data."""
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")
    seeded = await _seed_results_for_date(race_date)
    return {"status": "seeded", "results": seeded}


# ── Backfill ──────────────────────────────────────────────────────────────────

_backfill: dict = {"running": False, "done": False, "current": None,
                   "completed": [], "errors": [], "meetings": 0, "races": 0, "runners": 0}


async def _run_backfill(days: int, x_secret: Optional[str], force: bool = False):
    global _backfill
    _backfill.update({"running": True, "done": False, "current": None,
                      "completed": [], "errors": [], "meetings": 0, "races": 0, "runners": 0,
                      "started_at": datetime.utcnow().isoformat(), "total_days": days})
    try:
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)

        for i in range(1, days + 1):
            race_date = (date.today() - timedelta(days=i)).isoformat()
            _backfill["current"] = race_date
            if not force:
                async with get_session() as session:
                    already = await session.execute(
                        select(HistoricalResultRow)
                        .where(HistoricalResultRow.race_id.like(f"{race_date}_%"))
                        .limit(1)
                    )
                    if already.scalars().first():
                        log.info("[backfill] Skipping %s — already loaded", race_date)
                        _backfill["completed"].append(race_date)
                        continue
            log.info("[backfill] Processing %s", race_date)
            try:
                meetings = await client.get_meetings(race_date)
                for m in meetings:
                    slug = m.get("slug", "")
                    date_sfx = f"-{race_date.replace('-', '')}"
                    venue_code = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else ""
                    venue_name = m.get("venue", venue_code)
                    state = m.get("state", "")

                    meeting_detail = await client.get_meeting_by_slug(slug)
                    if not meeting_detail:
                        continue
                    full = await client._fetch_meeting_full(meeting_detail["id"])
                    if not full:
                        continue

                    for event in full.get("events", []):
                        race_num = event.get("eventNumber")
                        race_id = f"{race_date}_{venue_code}_R{race_num}"
                        event["_meeting"] = full
                        try:
                            race = client.parse_race(event, race_date, venue_name, state)
                            if not race.runners:
                                continue
                            predictions, _ = await enrich_and_predict_race(
                                race, model, generate_narratives=False
                            )
                            async with get_session() as session:
                                await save_race_predictions(
                                    session, race_id,
                                    [_prediction_to_db_dict(p, race_id) for p in predictions],
                                )
                            # Seed actual results
                            for sel in event.get("selections", []):
                                position = sel.get("selectionResult")
                                if not position or int(position) <= 0:
                                    continue
                                horse = (sel.get("competitor") or {}).get("name", "")
                                sp = sel.get("startingPrice")
                                beaten = sel.get("officialMargin") or 0
                                async with get_session() as session:
                                    # Skip if historical result already exists
                                    existing_hr = await session.execute(
                                        select(HistoricalResultRow)
                                        .where(HistoricalResultRow.race_id == race_id)
                                        .where(HistoricalResultRow.horse_name == horse)
                                        .limit(1)
                                    )
                                    if not existing_hr.scalars().first():
                                        fv_q = await session.execute(
                                            select(RunnerPredictionRow)
                                            .where(RunnerPredictionRow.race_id == race_id)
                                            .where(RunnerPredictionRow.horse_name == horse)
                                            .limit(1)
                                        )
                                        fv_row = fv_q.scalars().first()
                                        hr = HistoricalResultRow(
                                            race_id=race_id,
                                            horse_name=horse,
                                            position=int(position),
                                            beaten_margin=float(beaten),
                                            winner=int(position) == 1,
                                            placed=int(position) <= 3,
                                            starting_price=float(sp) if sp else None,
                                            feature_vector_json=fv_row.enriched_json if fv_row else None,
                                        )
                                        session.add(hr)
                                        await session.commit()
                                _backfill["runners"] += 1
                            _backfill["races"] += 1
                        except Exception as e:
                            log.warning("[backfill] Race %s failed: %s", race_id, e)

                    _backfill["meetings"] += 1
                    await asyncio.sleep(random.uniform(1, 3))

                _backfill["completed"].append(race_date)
            except Exception as e:
                log.warning("[backfill] Date %s failed: %s", race_date, e)
                _backfill["errors"].append({"date": race_date, "error": str(e)})
    finally:
        _backfill.update({"running": False, "done": True, "current": None,
                          "finished_at": datetime.utcnow().isoformat()})
        log.info("[backfill] Done — %d meetings, %d races, %d runners",
                 _backfill["meetings"], _backfill["races"], _backfill["runners"])


@app.post("/api/admin/backfill")
async def start_backfill(
    days: int = Query(14, ge=1, le=365),
    force: bool = Query(False),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Start background backfill of past N days.
    force=true re-runs predictions even for dates already in the DB (useful after model updates).
    Historical results are never duplicated regardless.
    """
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")
    if _backfill["running"]:
        raise HTTPException(409, "Backfill already running")
    asyncio.create_task(_run_backfill(days, x_cron_secret, force=force))
    return {"status": "started", "days": days, "force": force, "message": "Check /api/admin/backfill/status for progress"}


@app.get("/api/admin/backfill/status")
async def backfill_status():
    """Current backfill progress."""
    return _backfill


# ── Backtest ──────────────────────────────────────────────────────────────────

@app.get("/api/admin/backtest")
async def backtest_report(
    days: int = Query(14, ge=1, le=90),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Performance report: join predictions vs actual results.
    Returns top-pick win/place rate, value P&L, and per-condition breakdown.
    """
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")

    cutoff = (date.today() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        # Fetch all historical results within range
        hr_result = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()

        # Fetch all runner predictions for those race_ids
        race_ids = list({r.race_id for r in hr_rows})
        if not race_ids:
            return {
                "days": days,
                "races_with_results": 0,
                "message": "No historical results found — run backfill first",
            }

        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
        )
        pred_rows = pred_result.scalars().all()

    # Index predictions by (race_id, horse_name) and track top-pick per race
    pred_by_key: dict[tuple, RunnerPredictionRow] = {}
    top_pick: dict[str, RunnerPredictionRow] = {}   # race_id -> model_rank=1
    for p in pred_rows:
        pred_by_key[(p.race_id, p.horse_name)] = p
        if p.model_rank == 1 or p.race_id not in top_pick:
            if p.model_rank == 1:
                top_pick[p.race_id] = p
            elif p.race_id not in top_pick:
                top_pick[p.race_id] = p

    # Index results by (race_id, horse_name)
    result_by_key: dict[tuple, HistoricalResultRow] = {
        (r.race_id, r.horse_name): r for r in hr_rows
    }

    races_with_predictions = set(top_pick.keys()) & set(r.race_id for r in hr_rows)
    total_races = len(races_with_predictions)

    top_pick_wins = 0
    top_pick_places = 0
    value_pnl = 0.0
    value_bets = 0

    # Per-condition breakdown
    condition_stats: dict[str, dict] = {}

    recent_picks = []

    for race_id in sorted(races_with_predictions, reverse=True):
        pick = top_pick.get(race_id)
        if not pick:
            continue
        actual = result_by_key.get((race_id, pick.horse_name))
        if not actual:
            continue

        won = actual.winner
        placed = actual.placed
        sp = actual.starting_price or 0.0
        overlay = pick.overlay or 0.0

        if won:
            top_pick_wins += 1
        if placed:
            top_pick_places += 1

        # Value bet: only when model ranks it #1 and overlay > 0.15
        if overlay > 0.15 and sp > 0:
            value_bets += 1
            value_pnl += (sp - 1.0) if won else -1.0

        # Track condition breakdown — derive from race_id prefix
        parts = race_id.split("_")
        # track_condition is on RacePredictionRow which we don't fetch here.
        # Use "all" as the only bucket for now.
        bucket = "all"
        if bucket not in condition_stats:
            condition_stats[bucket] = {"races": 0, "wins": 0, "places": 0}
        condition_stats[bucket]["races"] += 1
        if won:
            condition_stats[bucket]["wins"] += 1
        if placed:
            condition_stats[bucket]["places"] += 1

        # Recent 20 picks for the detail table
        if len(recent_picks) < 20:
            recent_picks.append({
                "race_id": race_id,
                "top_pick": pick.horse_name,
                "model_rank": pick.model_rank,
                "win_prob": round(pick.win_probability or 0, 3),
                "overlay": round(overlay, 3),
                "sp": sp,
                "actual_position": actual.position,
                "won": won,
                "placed": placed,
            })

    win_rate = round(top_pick_wins / total_races, 3) if total_races else 0
    place_rate = round(top_pick_places / total_races, 3) if total_races else 0
    value_roi = round(value_pnl / value_bets, 3) if value_bets else 0

    return {
        "days": days,
        "total_races": total_races,
        "top_pick_wins": top_pick_wins,
        "top_pick_win_rate": win_rate,
        "top_pick_places": top_pick_places,
        "top_pick_place_rate": place_rate,
        "value_bets": value_bets,
        "value_pnl": round(value_pnl, 2),
        "value_roi_per_bet": value_roi,
        "recent_picks": recent_picks,
    }


@app.get("/api/performance")
async def performance_summary(days: int = Query(5, ge=1, le=30)):
    """
    Per-day performance strip for the last N days.
    Shows top-pick win rate, place rate, and value P&L per day.
    No auth required — displayed publicly on the frontend.
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()

        if not hr_rows:
            return {"days": days, "summary": [], "overall_win_rate": None}

        race_ids = list({r.race_id for r in hr_rows})
        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        top_picks = {p.race_id: p for p in pred_result.scalars().all()}

    result_by_key = {(r.race_id, r.horse_name): r for r in hr_rows}

    # Group by date
    by_date: dict[str, dict] = {}
    for race_id, pick in top_picks.items():
        race_date = race_id[:10]
        actual = result_by_key.get((race_id, pick.horse_name))
        if not actual:
            continue
        d = by_date.setdefault(race_date, {"races": 0, "wins": 0, "places": 0, "value_pnl": 0.0, "value_bets": 0})
        d["races"] += 1
        if actual.winner:
            d["wins"] += 1
        if actual.placed:
            d["places"] += 1
        sp = actual.starting_price or 0.0
        overlay = pick.overlay or 0.0
        if overlay > 0.05 and sp >= 3.0:
            d["value_bets"] += 1
            d["value_pnl"] += (sp - 1) if actual.winner else -1.0

    summary = []
    for day_str in sorted(by_date.keys(), reverse=True):
        d = by_date[day_str]
        races = d["races"]
        summary.append({
            "date": day_str,
            "races": races,
            "wins": d["wins"],
            "win_rate": round(d["wins"] / races, 3) if races else 0,
            "place_rate": round(d["places"] / races, 3) if races else 0,
            "value_bets": d["value_bets"],
            "value_pnl": round(d["value_pnl"], 2),
        })

    total_races = sum(d["races"] for d in by_date.values())
    total_wins = sum(d["wins"] for d in by_date.values())
    return {
        "days": days,
        "overall_win_rate": round(total_wins / total_races, 3) if total_races else None,
        "overall_races": total_races,
        "summary": summary,
    }


# ── Calibration ───────────────────────────────────────────────────────────────

_CANDIDATE_WINDOWS = [30, 60, 90, 180, 270]
_DRIFT_THRESHOLD = 0.05   # 5% drop vs 4-week rolling avg triggers flag


async def _run_calibration_sweep(holdout_days: int = 14) -> dict:
    """
    For each candidate training window, train the model excluding the holdout
    period, then score holdout races in-memory to get true out-of-sample stats.
    Saves the best window's weights to the DB and writes a CalibrationRow.
    """
    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        # All historical results
        hr_result = await session.execute(select(HistoricalResultRow))
        all_hr = hr_result.scalars().all()
        # All predictions with enriched data
        pred_result = await session.execute(
            select(RunnerPredictionRow).where(RunnerPredictionRow.enriched_json.isnot(None))
        )
        all_pred = pred_result.scalars().all()

    pred_by_key = {(p.race_id, p.horse_name): p for p in all_pred}

    # Group predictions by race_id for holdout scoring
    holdout_races: dict[str, list] = {}
    holdout_results: dict[tuple, HistoricalResultRow] = {}
    for r in all_hr:
        if r.race_id >= holdout_cutoff:
            holdout_results[(r.race_id, r.horse_name)] = r
    for p in all_pred:
        if p.race_id >= holdout_cutoff:
            holdout_races.setdefault(p.race_id, []).append(p)

    window_results = []
    best_window = None
    best_score = float("-inf")
    best_weights = None

    for window in _CANDIDATE_WINDOWS:
        train_cutoff = (today - timedelta(days=window)).isoformat()

        # Training data: within window, outside holdout
        training_data = []
        for row in all_hr:
            if row.race_id < train_cutoff or row.race_id >= holdout_cutoff:
                continue
            pred = pred_by_key.get((row.race_id, row.horse_name))
            if not pred:
                continue
            try:
                er = EnrichedRunner(**json.loads(pred.enriched_json))
                fv = build_feature_vector(er)
                training_data.append((fv, 1 if row.winner else 0))
            except Exception:
                continue

        if len(training_data) < 50:
            window_results.append({
                "window_days": window,
                "training_examples": len(training_data),
                "skipped": True,
                "reason": "insufficient training data",
            })
            continue

        model = HorseModel()
        stats = model.train(training_data)

        # Score holdout races with candidate model
        win_picks = place_picks = value_bets = total_races = 0
        value_pnl = 0.0

        for race_id, runners in holdout_races.items():
            runner_fvs = []
            for r in runners:
                try:
                    er = EnrichedRunner(**json.loads(r.enriched_json))
                    runner_fvs.append((r, build_feature_vector(er)))
                except Exception:
                    continue
            if not runner_fvs:
                continue

            win_probs, _ = model.predict_field([fv for _, fv in runner_fvs])
            best_idx = win_probs.index(max(win_probs))
            top_runner = runner_fvs[best_idx][0]
            top_prob = win_probs[best_idx]

            actual = holdout_results.get((race_id, top_runner.horse_name))
            if not actual:
                continue

            total_races += 1
            if actual.winner:
                win_picks += 1
            if actual.placed:
                place_picks += 1

            sp = actual.starting_price or 0
            implied = 1 / sp if sp > 1 else 0
            if (top_prob - implied) > 0.05 and sp > 0:
                value_bets += 1
                value_pnl += (sp - 1.0) if actual.winner else -1.0

        win_rate = round(win_picks / total_races, 3) if total_races else 0
        place_rate = round(place_picks / total_races, 3) if total_races else 0
        roi = round(value_pnl / value_bets, 3) if value_bets else 0

        result = {
            "window_days": window,
            "training_examples": len(training_data),
            "training_accuracy": stats["accuracy"],
            "holdout_races": total_races,
            "win_rate": win_rate,
            "place_rate": place_rate,
            "value_bets": value_bets,
            "value_pnl": round(value_pnl, 2),
            "value_roi": roi,
        }
        window_results.append(result)
        log.info("[calibrate] window=%d win=%.1f%% roi=%.3f", window, win_rate * 100, roi)

        # Best = highest ROI when we have enough value bets, else highest win rate
        score = roi if value_bets >= 10 else win_rate
        if score > best_score:
            best_score = score
            best_window = window
            best_weights = stats["weights"]

    if not best_weights:
        return {"error": "no valid windows", "window_results": window_results}

    # Save best weights
    async with get_session() as session:
        await save_model_weights(session, best_weights)

    # Drift detection: compare vs last 4 calibrations
    best_result = next((r for r in window_results if r.get("window_days") == best_window), {})
    drift_flag = False
    drift_reason = None

    async with get_session() as session:
        hist = await session.execute(
            select(CalibrationRow).order_by(CalibrationRow.ran_at.desc()).limit(4)
        )
        prev_runs = hist.scalars().all()

    if len(prev_runs) >= 4:
        avg_win = sum(r.win_rate for r in prev_runs) / len(prev_runs)
        avg_roi = sum(r.value_roi for r in prev_runs if r.value_roi is not None) / max(len(prev_runs), 1)
        cur_win = best_result.get("win_rate", 0)
        cur_roi = best_result.get("value_roi", 0)
        if avg_win - cur_win > _DRIFT_THRESHOLD:
            drift_flag = True
            drift_reason = f"Win rate dropped {avg_win - cur_win:.1%} below 4-week avg ({avg_win:.1%} → {cur_win:.1%})"
        elif avg_roi - cur_roi > _DRIFT_THRESHOLD:
            drift_flag = True
            drift_reason = f"Value ROI dropped {avg_roi - cur_roi:.3f} below 4-week avg ({avg_roi:.3f} → {cur_roi:.3f})"

    # Persist calibration record
    cal_row = CalibrationRow(
        ran_at=datetime.utcnow(),
        holdout_days=holdout_days,
        best_window=best_window,
        win_rate=best_result.get("win_rate"),
        place_rate=best_result.get("place_rate"),
        value_roi=best_result.get("value_roi"),
        value_bets=best_result.get("value_bets"),
        total_races=best_result.get("holdout_races"),
        drift_flag=drift_flag,
        drift_reason=drift_reason,
        all_results_json=json.dumps(window_results),
    )
    async with get_session() as session:
        session.add(cal_row)
        await session.commit()

    log.info("[calibrate] Best window=%d days, win=%.1f%%, roi=%.3f, drift=%s",
             best_window, best_result.get("win_rate", 0) * 100,
             best_result.get("value_roi", 0), drift_flag)

    return {
        "best_window": best_window,
        "best_score": round(best_score, 3),
        "drift_flag": drift_flag,
        "drift_reason": drift_reason,
        "window_results": window_results,
    }


async def _scheduled_calibrate():
    """Run by APScheduler every Sunday at 2am AEST."""
    log.info("[scheduler] Running weekly calibration sweep")
    try:
        result = await _run_calibration_sweep(holdout_days=14)
        if result.get("drift_flag"):
            log.warning("[calibrate] DRIFT DETECTED: %s", result.get("drift_reason"))
        log.info("[calibrate] Complete. Best window: %d days", result.get("best_window", 0))
    except Exception as e:
        log.exception("[calibrate] Weekly calibration failed: %s", e)


_calibration_status: dict = {"running": False, "done": False, "result": None, "error": None}


async def _run_calibration_task(holdout_days: int):
    global _calibration_status
    _calibration_status = {"running": True, "done": False, "result": None, "error": None,
                           "started_at": datetime.utcnow().isoformat()}
    try:
        result = await _run_calibration_sweep(holdout_days=holdout_days)
        _calibration_status.update({"running": False, "done": True, "result": result,
                                    "finished_at": datetime.utcnow().isoformat()})
        log.info("[calibrate] Background task complete. Best window: %s", result.get("best_window"))
    except Exception as e:
        log.exception("[calibrate] Background task failed: %s", e)
        _calibration_status.update({"running": False, "done": True, "error": str(e),
                                    "finished_at": datetime.utcnow().isoformat()})


@app.post("/api/admin/calibrate")
async def run_calibration(
    holdout_days: int = Query(14, ge=7, le=30),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Start background calibration sweep. Check /api/admin/calibrate/status for progress.
    Saves winning weights to DB when done.
    """
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")
    if _calibration_status.get("running"):
        raise HTTPException(409, "Calibration already running")
    asyncio.create_task(_run_calibration_task(holdout_days))
    return {"status": "started", "holdout_days": holdout_days,
            "message": "Check /api/admin/calibrate/status for progress"}


@app.get("/api/admin/calibrate/status")
async def calibration_task_status(x_cron_secret: Optional[str] = Header(None)):
    """Current calibration task progress and result when done."""
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")
    return _calibration_status


@app.get("/api/admin/calibration/history")
async def calibration_history(
    limit: int = Query(10, ge=1, le=50),
    x_cron_secret: Optional[str] = Header(None),
):
    """Return the last N calibration runs with drift history."""
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")
    async with get_session() as session:
        result = await session.execute(
            select(CalibrationRow).order_by(CalibrationRow.ran_at.desc()).limit(limit)
        )
        rows = result.scalars().all()
    return {
        "calibrations": [
            {
                "ran_at": r.ran_at.isoformat(),
                "best_window": r.best_window,
                "win_rate": r.win_rate,
                "place_rate": r.place_rate,
                "value_roi": r.value_roi,
                "value_bets": r.value_bets,
                "total_races": r.total_races,
                "drift_flag": r.drift_flag,
                "drift_reason": r.drift_reason,
                "window_results": json.loads(r.all_results_json or "[]"),
            }
            for r in rows
        ]
    }


# ── Cron ──────────────────────────────────────────────────────────────────────

async def _enrich_date(race_date: str, client, model) -> list[dict]:
    """Enrich all meetings for a single date. Returns summary list."""
    meetings = await client.get_meetings(race_date)
    summary = []
    for m in meetings:
        slug = m.get("slug", "")
        date_sfx = f"-{race_date.replace('-', '')}"
        venue_code = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else m.get("name", "").lower().replace(" ", "-")
        venue_name = m.get("venue", venue_code)
        state = m.get("state", "")
        try:
            raw_events = await client.get_meeting_races(slug)
            for raw_event in raw_events:
                race_num = raw_event.get("eventNumber")
                race_id = f"{race_date}_{venue_code}_R{race_num}"
                full_event = await client.get_race(slug, race_num)
                if not full_event:
                    continue
                race = client.parse_race(full_event, race_date, venue_name, state)
                predictions, _ = await enrich_and_predict_race(race, model)
                async with get_session() as session:
                    await save_race_predictions(
                        session,
                        race_id,
                        [_prediction_to_db_dict(p, race_id) for p in predictions],
                    )
            summary.append({"venue": venue_code, "status": "ok"})
        except Exception as e:
            log.warning("Cron failed for %s on %s: %s", venue_code, race_date, e)
            summary.append({"venue": venue_code, "status": "error", "error": str(e)})
    return summary


@app.post("/api/cron/enrich")
async def cron_enrich(
    days: int = Query(3, ge=1, le=7),
    x_cron_secret: Optional[str] = Header(None),
):
    """Cron: enrich all meetings for today + next N days (default 3)."""
    if settings.cron_secret and x_cron_secret != settings.cron_secret:
        raise HTTPException(403, "Forbidden")

    client = get_tab_client()
    async with get_session() as session:
        model = await _load_model(session)

    results = {}
    for i in range(days):
        race_date = (date.today() + timedelta(days=i)).isoformat()
        log.info("[cron] Enriching %s", race_date)
        results[race_date] = await _enrich_date(race_date, client, model)

    return {"dates": results}


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
