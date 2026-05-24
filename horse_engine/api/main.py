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
import re
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from horse_engine.api.database import get_session
from horse_engine.clients.factory import get_tab_client
from horse_engine.config import settings
from horse_engine.models.database import (
    BacktestResultRow,
    BacktestStateRow,
    CalibrationRow,
    HistoricalResultRow,
    OddsSnapshotRow,
    RunnerPredictionRow,
    RacePredictionRow,
    init_db,
    load_model_weights,
    save_model_weights,
    load_place_model_weights,
    save_place_model_weights,
    load_exotic_model_weights,
    save_exotic_model_weights,
    save_race_predictions,
)
from horse_engine.models.enriched import EnrichedRunner
from horse_engine.pipeline import enrich_and_predict_race, enrich_meeting
from horse_engine.prediction.engine import _value_rating
from horse_engine.prediction.features import build_feature_vector
from horse_engine.prediction.model import HorseModel, PlaceModel, ExoticModel
from horse_engine.prediction.venue_calibration import compute_venue_multipliers

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VENUE_RE = re.compile(r"^[a-z0-9-]{1,60}$")

_AEST = ZoneInfo("Australia/Sydney")

def _today_aest() -> date:
    return datetime.now(_AEST).date()


def _validate_date(race_date: str) -> str:
    if not _DATE_RE.match(race_date):
        raise HTTPException(400, "Invalid date format — expected YYYY-MM-DD")
    return race_date


def _validate_venue(venue_code: str) -> str:
    if not _VENUE_RE.match(venue_code):
        raise HTTPException(400, "Invalid venue code")
    return venue_code


def _check_admin(x_secret: Optional[str]) -> None:
    """Fail-closed admin auth: requires CRON_SECRET env var to be set."""
    if not settings.cron_secret:
        raise HTTPException(403, "Admin access not configured")
    if not secrets.compare_digest(x_secret or "", settings.cron_secret):
        raise HTTPException(403, "Forbidden")


def _like_safe(value: str) -> str:
    """Escape LIKE metacharacters to prevent pattern injection."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def _scheduled_odds_snapshot():
    """Snapshot current odds for all upcoming races within the next 3 hours."""
    log.info("[odds-snapshot] Running odds snapshot")
    try:
        client = get_tab_client()
        now_aest = datetime.now(_AEST)
        today = now_aest.date().isoformat()
        tomorrow = (now_aest.date() + timedelta(days=1)).isoformat()

        for target_date in [today, tomorrow]:
            prefix = f"{target_date}_"
            async with get_session() as session:
                result = await session.execute(
                    select(RunnerPredictionRow)
                    .where(RunnerPredictionRow.model_rank == 1)
                    .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
                )
                picks = result.scalars().all()

            if not picks:
                continue

            slugs: dict[str, list[RunnerPredictionRow]] = {}
            for p in picks:
                _, venue_code, _ = _parse_race_id(p.race_id)
                slug = _meeting_slug(venue_code, target_date)
                slugs.setdefault(slug, []).append(p)

            for slug, slug_picks in slugs.items():
                try:
                    events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=30)
                    snapped_at = datetime.utcnow()

                    # Build lookup: horse_name → (win_odds, place_odds, source, jump_time)
                    horse_data: dict[str, dict] = {}
                    for event in events:
                        start_time = event.get("startTime")
                        for sel in event.get("selections") or []:
                            name = (sel.get("competitor") or {}).get("name")
                            if not name:
                                continue
                            tote = sel.get("topToteWin")
                            place = sel.get("topTotePlace")
                            flucs = sel.get("flucs") or {}
                            if tote:
                                win = float(tote)
                                source = "tote"
                            elif flucs.get("low"):
                                win = float(flucs["low"])
                                source = "flucs"
                            elif flucs.get("open"):
                                win = float(flucs["open"])
                                source = "flucs"
                            else:
                                continue
                            horse_data[name] = {
                                "win": win,
                                "place": float(place) if place else None,
                                "source": source,
                                "start_time": start_time,
                            }

                    async with get_session() as session:
                        for pick in slug_picks:
                            hd = horse_data.get(pick.horse_name)
                            if not hd:
                                continue
                            mins_to_jump = None
                            if hd["start_time"]:
                                try:
                                    jump = datetime.fromisoformat(hd["start_time"].replace("Z", "+00:00"))
                                    mins_to_jump = round((jump.timestamp() - snapped_at.replace(tzinfo=None).timestamp()) / 60)
                                except Exception:
                                    pass
                            session.add(OddsSnapshotRow(
                                race_id=pick.race_id,
                                horse_name=pick.horse_name,
                                snapshotted_at=snapped_at,
                                minutes_to_jump=mins_to_jump,
                                win_odds=hd["win"],
                                place_odds=hd["place"],
                                source=hd["source"],
                            ))
                        await session.commit()
                    log.info("[odds-snapshot] Snapped %d runners for %s", len(horse_data), slug)
                except Exception as e:
                    log.warning("[odds-snapshot] Failed for %s: %s", slug, e)
    except Exception as e:
        log.exception("[odds-snapshot] Snapshot failed: %s", e)


async def _scheduled_enrich():
    """Run by APScheduler — enrich today + next 2 days, then seed today's results."""
    log.info("[scheduler] Running scheduled enrichment")
    try:
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)
        for i in range(3):
            race_date = (_today_aest() + timedelta(days=i)).isoformat()
            log.info("[scheduler] Enriching %s", race_date)
            await _enrich_date(race_date, client, model)
        # Seed yesterday + today so every startup/deploy auto-backfills the most recent gap
        for offset in (-1, 0):
            seed_date = (_today_aest() + timedelta(days=offset)).isoformat()
            n = await _seed_results_for_date(seed_date)
            if n:
                log.info("[scheduler] Seeded %d results for %s", n, seed_date)
        log.info("[scheduler] Enrichment complete")
    except Exception as e:
        log.exception("[scheduler] Enrichment failed: %s", e)


async def _scheduled_pre_race_enrich():
    """
    Re-enrich any race starting within the next 2 hours.
    Runs every 15 min during racing hours. Waits a random 0-10 min delay
    before hitting Punters to avoid predictable request patterns.
    Also detects meetings removed from Punters and marks those races cancelled.
    """
    from sqlalchemy import update as sa_update
    delay = random.uniform(0, 600)
    log.info("[pre-race] Waiting %.0fs before Punters requests", delay)
    await asyncio.sleep(delay)
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    horizon = now_utc + timedelta(hours=2)
    today = _today_aest().isoformat()
    date_sfx = f"-{today.replace('-', '')}"
    log.info("[pre-race] Scanning for races within 2 hours")
    try:
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)
            place_model = await _load_place_model(session)
        meetings = await client.get_meetings(today)

        # Build set of active venue codes from Punters
        active_venue_codes: set[str] = set()
        for m in meetings:
            slug = m.get("slug", "")
            vc = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else ""
            if vc:
                active_venue_codes.add(vc)

        # Detect venues in our DB that Punters has dropped — mark their races cancelled
        async with get_session() as session:
            db_rows = (await session.execute(
                select(RunnerPredictionRow.race_id)
                .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
                .where(RunnerPredictionRow.model_rank == 1)
                .distinct()
            )).scalars().all()

        for race_id in db_rows:
            _, venue_code, _ = _parse_race_id(race_id)
            if venue_code not in active_venue_codes:
                async with get_session() as session:
                    await session.execute(
                        sa_update(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == race_id)
                        .values(cancelled=True)
                    )
                    await session.commit()
                log.info("[pre-race] Marked %s CANCELLED (venue dropped from Punters)", race_id)

        enriched_count = 0
        for m in meetings:
            slug = m.get("slug", "")
            venue_code = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else m.get("name", "").lower().replace(" ", "-")
            venue_name = m.get("venue", venue_code)
            state = m.get("state", "")
            try:
                raw_events = await client.get_meeting_races(slug)
                for raw_event in raw_events:
                    start_raw = raw_event.get("startTime")
                    if not start_raw:
                        continue
                    try:
                        jump = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if not (now_utc <= jump <= horizon):
                        continue
                    race_num = raw_event.get("eventNumber")
                    race_id = f"{today}_{venue_code}_R{race_num}"
                    full_event = await client.get_race(slug, race_num)
                    if not full_event:
                        continue
                    race = await client.parse_race(full_event, today, venue_name, state)
                    predictions, _ = await enrich_and_predict_race(race, model, place_model=place_model)
                    async with get_session() as session:
                        await save_race_predictions(
                            session,
                            race_id,
                            [_prediction_to_db_dict(p, race_id) for p in predictions],
                        )
                    log.info("[pre-race] Re-enriched %s (jump %s)", race_id, start_raw)
                    enriched_count += 1
            except Exception as e:
                log.warning("[pre-race] Failed for %s: %s", slug, e)
        if enriched_count:
            log.info("[pre-race] Re-enriched %d races", enriched_count)
    except Exception as e:
        log.exception("[pre-race] Pre-race enrich failed: %s", e)


async def _seed_results_for_date(race_date: str) -> int:
    """Fetch settled results for race_date and store as training data. Returns count seeded."""
    client = get_tab_client()
    meetings = await client.get_meetings(race_date)
    seeded = 0
    date_sfx = f"-{race_date.replace('-', '')}"
    for meeting in meetings:
        slug = meeting.get("slug", "")
        venue_code = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else ""
        # Resolve meeting id: prefer id from get_meetings, fall back to slug lookup
        meeting_id = meeting.get("id")
        if not meeting_id:
            meta = await client.get_meeting_by_slug(slug)
            meeting_id = (meta or {}).get("id")
        if not meeting_id:
            continue
        # Fetch full meeting once — avoids re-fetching per race (was O(N) calls, now O(1))
        full = await client._fetch_meeting_full(meeting_id)
        if not full:
            continue
        for event in full.get("events", []):
            race_num = event.get("eventNumber")
            race_id = f"{race_date}_{venue_code}_R{race_num}"
            for sel in event.get("selections", []):
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
    yesterday = (_today_aest() - timedelta(days=1)).isoformat()
    log.info("[scheduler] Seeding results for %s", yesterday)
    try:
        n = await _seed_results_for_date(yesterday)
        log.info("[scheduler] Seeded %d results for %s", n, yesterday)
    except Exception as e:
        log.exception("[scheduler] Result seeding failed for %s: %s", yesterday, e)


async def _scheduled_exotic_retrain():
    """Run by APScheduler at 3am AEST — retrain exotic model after nightly calibration."""
    log.info("[scheduler] Running nightly exotic model retrain")
    try:
        async with get_session() as session:
            hr_result = await session.execute(select(HistoricalResultRow))
            hr_rows = hr_result.scalars().all()
            pred_result = await session.execute(
                select(RunnerPredictionRow).where(RunnerPredictionRow.enriched_json.isnot(None))
            )
            pred_rows = pred_result.scalars().all()

        pred_by_key = {(p.race_id, p.horse_name): p for p in pred_rows}

        race_data: dict[str, list] = {}
        for row in hr_rows:
            pred = pred_by_key.get((row.race_id, row.horse_name))
            if not pred:
                continue
            try:
                er = EnrichedRunner(**json.loads(pred.enriched_json))
                fv = build_feature_vector(er)
                race_data.setdefault(row.race_id, []).append((fv, 1 if row.placed else 0))
            except Exception:
                continue

        race_groups = [
            runners for runners in race_data.values()
            if len(runners) >= 7 and sum(1 for _, lbl in runners if lbl == 1) == 3
        ]

        if not race_groups:
            log.warning("[scheduler] Exotic retrain: no eligible races found")
            return

        m = ExoticModel()
        stats = m.train_exotic(race_groups)
        async with get_session() as session:
            await save_exotic_model_weights(session, stats["weights"])
        log.info(
            "[scheduler] Exotic retrain complete — %d races, box_hit_rate=%.3f",
            len(race_groups), stats.get("box_hit_rate", 0),
        )
    except Exception as e:
        log.exception("[scheduler] Exotic retrain failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Schedule enrichment at 6am, 10am, 1pm AEST (UTC+10 = subtract 10h)
    scheduler = AsyncIOScheduler(timezone="Australia/Sydney")
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=6,  minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=10, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=13, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=15, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=17, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=23, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_calibrate,      CronTrigger(hour=2, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_exotic_retrain, CronTrigger(hour=3, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(
        _scheduled_odds_snapshot,
        CronTrigger(hour="9-18", minute="0,15,30,45", timezone="Australia/Sydney")
    )
    scheduler.start()
    log.info("[scheduler] Cron jobs scheduled: 6am/10am/1pm enrich, 3pm/5pm/11pm seed results, 2am calibration, 3am exotic retrain, every 15min odds snapshots 9am-6pm")

    # Enrich today on startup so deploys don't leave races un-loaded
    asyncio.create_task(_scheduled_enrich())
    # Backfill last 3 days — enrich any un-enriched meetings then seed results
    async def _startup_backfill():
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)
        for offset in (-3, -2, -1, 0):
            seed_date = (_today_aest() + timedelta(days=offset)).isoformat()
            try:
                # Enrich any meetings that are missing predictions
                await _enrich_date(seed_date, client, model)
                # Then seed results
                n = await _seed_results_for_date(seed_date)
                if n:
                    log.info("[startup] Seeded %d results for %s", n, seed_date)
            except Exception as e:
                log.warning("[startup] Backfill failed for %s: %s", seed_date, e)
    asyncio.create_task(_startup_backfill())

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

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; "
            "font-src https://fonts.gstatic.com; "
            "connect-src 'self'; "
            "img-src 'self' data:; "
            "frame-ancestors 'none'"
        )
        return response

app.add_middleware(SecurityHeadersMiddleware)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _load_model(session) -> HorseModel:
    weights = await load_model_weights(session)
    if weights:
        return HorseModel.from_weights_dict(weights)
    return HorseModel()


async def _load_place_model(session) -> PlaceModel:
    weights = await load_place_model_weights(session)
    if weights:
        return PlaceModel.from_weights_dict(weights)
    return PlaceModel()


async def _load_exotic_model(session) -> ExoticModel:
    weights = await load_exotic_model_weights(session)
    if weights:
        return ExoticModel.from_weights_dict(weights)
    return ExoticModel()


def _today() -> str:
    return _today_aest().isoformat()


def _meeting_slug(venue: str, race_date: str) -> str:
    """Build the punters.com.au meeting slug from venue slug and date."""
    return f"{venue}-{race_date.replace('-', '')}"


# ── Edge picks ───────────────────────────────────────────────────────────────

_CALIBRATED_WIN_RATES = [(50, 88), (45, 82), (40, 76), (35, 71), (30, 66)]

def _parse_race_id(race_id: str) -> tuple[str, str, int | None]:
    """Parse race_id '{date}_{venue}_R{num}' → (date, venue, race_number)."""
    try:
        parts = race_id.split("_R")
        race_num = int(parts[-1])
        rest = "_R".join(parts[:-1])          # handles venues with underscores
        date_part = rest[:10]
        venue_part = rest[11:]
        return date_part, venue_part, race_num
    except Exception:
        return "", race_id, None


# Cache meeting start times for 5 min to avoid hitting punters on every edge load
_edge_times_cache: dict[str, tuple[datetime, dict[int, str]]] = {}

async def _fetch_race_times(client, slug: str) -> dict[int, str]:
    """Return {race_number: startTime ISO string} for a meeting slug. Cached 5 min."""
    cached = _edge_times_cache.get(slug)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < 300:
        return cached[1]
    try:
        events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=20)
        times = {e["eventNumber"]: e.get("startTime") for e in events if e.get("eventNumber")}
        _edge_times_cache[slug] = (datetime.utcnow(), times)
        return times
    except Exception:
        return {}


@app.get("/api/edge")
async def get_edge_picks():
    """High-confidence picks for today + next 3 days. Threshold: model win_probability >= 29.5% (rounds to 30%)."""
    threshold = 0.295
    picks = []
    today = _today_aest()
    client = get_tab_client()

    for i in range(4):
        target_date = (today + timedelta(days=i)).isoformat()
        prefix = f"{target_date}_"
        async with get_session() as session:
            result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.model_rank == 1)
                .where(RunnerPredictionRow.win_probability >= threshold)
                .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
                .order_by(RunnerPredictionRow.win_probability.desc())
            )
            rows = result.scalars().all()

            if not rows:
                continue

            # Batch-fetch place model runners for trifecta legs
            race_ids = [r.race_id for r in rows]
            place_result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id.in_(race_ids))
                .where(RunnerPredictionRow.place_model_rank >= 1)
                .where(RunnerPredictionRow.place_model_rank <= 4)
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            )
            place_rows_list = place_result.scalars().all()

            # Batch-fetch exotic model top-3 for alignment check
            exotic_result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id.in_(race_ids))
                .where(RunnerPredictionRow.exotic_model_rank >= 1)
                .where(RunnerPredictionRow.exotic_model_rank <= 3)
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            )
            exotic_rows_list = exotic_result.scalars().all()

        # Build place-model lookup: race_id -> sorted list by place_model_rank
        trifecta_map: dict[str, list] = {}
        for pr in place_rows_list:
            if pr.place_model_rank:
                trifecta_map.setdefault(pr.race_id, []).append(pr)
        for key in trifecta_map:
            trifecta_map[key].sort(key=lambda r: r.place_model_rank)

        # Build exotic top-3 lookup: race_id -> set of horse names
        exotic_top3_map: dict[str, set] = {}
        for er in exotic_rows_list:
            exotic_top3_map.setdefault(er.race_id, set()).add(er.horse_name)

        # Fetch scheduled times per unique meeting in parallel
        unique_venues = {_parse_race_id(r.race_id)[1] for r in rows}
        slug_map = {v: _meeting_slug(v, target_date) for v in unique_venues}
        time_results = await asyncio.gather(*[_fetch_race_times(client, slug) for slug in slug_map.values()])
        race_times: dict[str, str | None] = {}  # race_id → startTime
        for venue, times in zip(slug_map.keys(), time_results):
            for race_num, start_time in times.items():
                race_times[f"{target_date}_{venue}_R{race_num}"] = start_time

        for runner_row in rows:
            odds = runner_row.best_available_odds or 0
            model_pct = round(runner_row.win_probability * 100, 1)
            market_implied_pct = round((1 / odds) * 100, 1) if odds else None
            edge_pct = round(model_pct - market_implied_pct, 1) if market_implied_pct else None
            calibrated = next((r for t, r in _CALIBRATED_WIN_RATES if model_pct >= t), 66)
            hot = model_pct >= 45
            _, venue_code, race_num = _parse_race_id(runner_row.race_id)

            # Build trifecta legs: win pick + top 2 place-model picks (excluding win)
            place_runners = trifecta_map.get(runner_row.race_id, [])
            place_excl = [pr for pr in place_runners if pr.horse_name != runner_row.horse_name]
            def _leg(pr):
                return {
                    "tab_number": pr.tab_number,
                    "horse_name": pr.horse_name,
                    "place_pct": round(pr.place_probability * 100, 1) if pr.place_probability else None,
                }
            win_leg = {
                "tab_number": runner_row.tab_number,
                "horse_name": runner_row.horse_name,
                "place_pct": round(runner_row.place_probability * 100, 1) if runner_row.place_probability else None,
            }
            tri_legs = [win_leg] + [_leg(pr) for pr in place_excl[:2]]
            ff_legs  = [win_leg] + [_leg(pr) for pr in place_excl[:3]]
            # Approximate combined hit rate: product of individual place probabilities
            tri_probs = [l["place_pct"] for l in tri_legs if l["place_pct"] is not None]
            tri_combined = round(tri_probs[0] * tri_probs[1] * tri_probs[2] / 10000, 1) if len(tri_probs) == 3 else None
            ff_probs = [l["place_pct"] for l in ff_legs if l["place_pct"] is not None]
            ff_combined = round(ff_probs[0] * ff_probs[1] * ff_probs[2] * ff_probs[3] / 1000000, 1) if len(ff_probs) == 4 else None
            # Exotic alignment: compare exotic model's top-3 against win pick
            exotic_top3 = exotic_top3_map.get(runner_row.race_id, set())
            if not exotic_top3:
                exotic_alignment = "no_exotic"
            elif runner_row.horse_name in exotic_top3:
                # Check if win horse is exotic leg 1 (rank 1)
                exotic_leg1 = next(
                    (er.horse_name for er in exotic_rows_list
                     if er.race_id == runner_row.race_id and er.exotic_model_rank == 1),
                    None,
                )
                exotic_alignment = "confirmed" if runner_row.horse_name == exotic_leg1 else "partial"
            else:
                exotic_alignment = "diverge"

            trifecta = {
                "legs": tri_legs,
                "combined_pct": tri_combined,
                "first_four": ff_legs if len(ff_legs) >= 4 else None,
                "first_four_combined_pct": ff_combined,
                "exotic_alignment": exotic_alignment,
            } if len(tri_legs) >= 3 else None

            picks.append({
                "date": target_date,
                "race_id": runner_row.race_id,
                "venue": venue_code,
                "state": None,
                "race_number": race_num,
                "race_name": None,
                "distance": None,
                "track_condition": None,
                "scheduled_time": race_times.get(runner_row.race_id),
                "horse_name": runner_row.horse_name,
                "jockey": runner_row.jockey,
                "trainer": runner_row.trainer,
                "barrier": runner_row.barrier,
                "weight": runner_row.weight,
                "model_pct": model_pct,
                "calibrated_win_rate": calibrated,
                "best_available_odds": odds,
                "market_implied_pct": market_implied_pct,
                "edge_pct": edge_pct,
                "hot_pick": hot,
                "overlay": runner_row.overlay,
                "value_rating": runner_row.value_rating,
                "place_probability": round(runner_row.place_probability * 100, 1) if runner_row.place_probability else None,
                "cancelled": bool(runner_row.cancelled),
                "trifecta": trifecta,
            })

    # For picks whose race has already jumped, fetch results and annotate trifecta legs
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    finished_picks = [
        p for p in picks
        if p["scheduled_time"] and datetime.fromisoformat(p["scheduled_time"].replace("Z", "+00:00")) < now_utc
    ]
    if finished_picks:
        finished_venues: dict[str, str] = {}  # venue_code → date
        for p in finished_picks:
            finished_venues[p["venue"]] = p["date"]

        async def _fetch_today_results(venue: str, date: str) -> dict:
            slug = _meeting_slug(venue, date)
            try:
                events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=30)
                out = {}
                for event in events:
                    rn = event.get("eventNumber")
                    for sel in event.get("selections") or []:
                        name = (sel.get("competitor") or {}).get("name")
                        if name:
                            pos = sel.get("selectionResult")
                            out[(venue, rn, name)] = {
                                "position": int(pos) if isinstance(pos, (int, float)) and pos > 0 else None,
                                "scratched": sel.get("status") == "SCRATCHED",
                            }
                return out
            except Exception:
                return {}

        result_batches = await asyncio.gather(*[
            _fetch_today_results(v, d) for v, d in finished_venues.items()
        ])
        today_results: dict = {}
        for rb in result_batches:
            today_results.update(rb)

        for p in finished_picks:
            tri = p.get("trifecta")
            if not tri:
                continue
            venue_code, race_num = p["venue"], p["race_number"]

            def _annotate_legs(legs):
                out = []
                for l in legs:
                    res = today_results.get((venue_code, race_num, l["horse_name"]), {})
                    out.append({**l, "position": res.get("position"), "scratched": res.get("scratched", False)})
                return out

            tri["legs"] = _annotate_legs(tri["legs"])
            tri_positions = {l["position"] for l in tri["legs"] if l["position"] and not l["scratched"]}
            tri["hit"] = tri_positions == {1, 2, 3}

            if tri.get("first_four"):
                tri["first_four"] = _annotate_legs(tri["first_four"])
                ff_positions = {l["position"] for l in tri["first_four"] if l["position"] and not l["scratched"]}
                tri["first_four_hit"] = ff_positions == {1, 2, 3, 4}

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "threshold_pct": int(threshold * 100),
        "picks": picks,
    }


_odds_refresh_last: datetime | None = None
_ODDS_REFRESH_COOLDOWN = 120  # seconds — prevents hammering punters API

@app.post("/api/edge/refresh-odds")
async def refresh_edge_odds():
    """
    Fetch fresh odds from punters for upcoming edge picks and update DB.
    No Claude calls — zero AI cost. Only updates best_available_odds.
    Rate-limited to once per 2 minutes globally.
    """
    global _odds_refresh_last
    now = datetime.utcnow()
    if _odds_refresh_last and (now - _odds_refresh_last).total_seconds() < _ODDS_REFRESH_COOLDOWN:
        return {"updated": {}, "count": 0, "cached": True}
    threshold = 0.295
    today = _today_aest()
    client = get_tab_client()
    updated: dict[str, float] = {}  # race_id → new odds

    for i in range(4):  # today + next 3 days — covers weekend picks
        target_date = (today + timedelta(days=i)).isoformat()
        prefix = f"{target_date}_"

        async with get_session() as session:
            result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.model_rank == 1)
                .where(RunnerPredictionRow.win_probability >= threshold)
                .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
            )
            picks = result.scalars().all()

        # Group by meeting slug so we only call punters once per meeting
        slugs: dict[str, list[RunnerPredictionRow]] = {}
        for p in picks:
            _, venue_code, _ = _parse_race_id(p.race_id)
            slug = _meeting_slug(venue_code, target_date)
            slugs.setdefault(slug, []).append(p)

        for slug, slug_picks in slugs.items():
            try:
                events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=30)
                # Build horse → odds map from all selections across all events
                horse_odds: dict[str, float] = {}
                for event in events:
                    for sel in event.get("selections") or []:
                        name = (sel.get("competitor") or {}).get("name")
                        if not name:
                            continue
                        tote = sel.get("topToteWin")
                        sp = sel.get("startingPrice")
                        flucs = sel.get("flucs") or {}
                        best = (float(tote) if tote else
                                float(flucs["low"]) if flucs.get("low") else
                                float(flucs["open"]) if flucs.get("open") else
                                float(sp) if sp else 0.0)
                        if best:
                            horse_odds[name] = best

                # Update DB rows that have a fresh odds value or stale value_rating
                async with get_session() as session:
                    for pick in slug_picks:
                        new_odds = horse_odds.get(pick.horse_name)
                        odds_changed = new_odds and new_odds != pick.best_available_odds
                        stale_rating = pick.best_available_odds and (not pick.value_rating)
                        if odds_changed or stale_rating:
                            pick_row = await session.get(RunnerPredictionRow, pick.id)
                            if pick_row:
                                if odds_changed:
                                    pick_row.best_available_odds = new_odds
                                else:
                                    new_odds = pick_row.best_available_odds
                                market_implied = 1.0 / new_odds if new_odds else 0.0
                                pick_row.overlay = round(pick_row.win_probability - market_implied, 4)
                                pick_row.value_rating = _value_rating(pick_row.win_probability, new_odds, pick_row.overlay)
                                updated[pick.race_id] = new_odds
                    await session.commit()
            except Exception as e:
                log.warning("refresh-odds failed for %s: %s", slug, e)

    _odds_refresh_last = now
    return {"updated": updated, "count": len(updated)}


@app.get("/api/edge/yesterday")
async def get_edge_yesterday(for_date: Optional[str] = Query(None, alias="date")):
    """Qualifying picks with actual results and SP odds from punters.
    Accepts ?date=YYYY-MM-DD (defaults to yesterday)."""
    target_date = for_date or (_today_aest() - timedelta(days=1)).isoformat()
    threshold = 0.295
    prefix = f"{target_date}_"
    stake = 10

    async with get_session() as session:
        result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.model_rank == 1)
            .where(RunnerPredictionRow.win_probability >= threshold)
            .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
            .order_by(RunnerPredictionRow.win_probability.desc())
        )
        picks = result.scalars().all()

        if not picks:
            return {"date": target_date, "picks": [], "summary": None}

        # Batch-fetch place model runners for trifecta legs
        yst_race_ids = [p.race_id for p in picks]
        yst_place_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(yst_race_ids))
            .where(RunnerPredictionRow.place_model_rank >= 1)
            .where(RunnerPredictionRow.place_model_rank <= 4)
        )
        yst_place_rows = yst_place_result.scalars().all()

    yst_trifecta_map: dict[str, list] = {}
    for pr in yst_place_rows:
        if pr.place_model_rank:
            yst_trifecta_map.setdefault(pr.race_id, []).append(pr)
    for key in yst_trifecta_map:
        yst_trifecta_map[key].sort(key=lambda r: r.place_model_rank)

    client = get_tab_client()
    unique_venues = {_parse_race_id(p.race_id)[1] for p in picks}

    async def fetch_results(venue: str) -> dict:
        slug = _meeting_slug(venue, target_date)
        try:
            events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=30)
            out = {}
            for event in events:
                race_num = event.get("eventNumber")
                for sel in event.get("selections") or []:
                    name = (sel.get("competitor") or {}).get("name")
                    if name:
                        pos = sel.get("selectionResult")
                        sp = sel.get("startingPrice")
                        out[(venue, race_num, name)] = {
                            "position": int(pos) if isinstance(pos, (int, float)) and pos > 0 else None,
                            "sp": float(sp) if sp else None,
                            "winner": pos == 1,
                            "scratched": sel.get("status") == "SCRATCHED",
                        }
            return out
        except Exception as e:
            log.warning("yesterday results fetch failed for %s: %s", venue, e)
            return {}

    results_list = await asyncio.gather(*[fetch_results(v) for v in unique_venues])
    all_results: dict = {}
    for r in results_list:
        all_results.update(r)

    output = []
    for p in picks:
        _, venue_code, race_num = _parse_race_id(p.race_id)
        r = all_results.get((venue_code, race_num, p.horse_name), {})
        sp = r.get("sp") or p.best_available_odds or None
        winner = r.get("winner", False)
        position = r.get("position")
        scratched = r.get("scratched", False)
        model_pct = round(p.win_probability * 100, 1)
        payout = round(sp * stake, 2) if winner and sp else 0
        profit = round(payout - stake, 2) if winner and sp else -stake

        placed = bool(position and position <= 3 and not scratched)

        # Find the actual race winner when our pick didn't win
        winner_name = None
        if not winner and not scratched:
            for (v, rn, name), res in all_results.items():
                if v == venue_code and rn == race_num and res.get("winner"):
                    winner_name = name
                    break

        place_pct = round(p.place_probability * 100, 1) if p.place_probability else None

        # Build trifecta legs for yesterday display
        yst_place_runners = yst_trifecta_map.get(p.race_id, [])
        yst_place_excl = [pr for pr in yst_place_runners if pr.horse_name != p.horse_name]
        def _yst_leg(pr):
            return {
                "tab_number": pr.tab_number,
                "horse_name": pr.horse_name,
                "place_pct": round(pr.place_probability * 100, 1) if pr.place_probability else None,
            }
        yst_win_leg = {
            "tab_number": p.tab_number,
            "horse_name": p.horse_name,
            "place_pct": place_pct,
        }
        yst_tri_legs = [yst_win_leg] + [_yst_leg(pr) for pr in yst_place_excl[:2]]
        yst_ff_legs  = [yst_win_leg] + [_yst_leg(pr) for pr in yst_place_excl[:3]]
        yst_tri_probs = [l["place_pct"] for l in yst_tri_legs if l["place_pct"] is not None]
        yst_tri_combined = round(yst_tri_probs[0] * yst_tri_probs[1] * yst_tri_probs[2] / 10000, 1) if len(yst_tri_probs) == 3 else None
        yst_ff_probs = [l["place_pct"] for l in yst_ff_legs if l["place_pct"] is not None]
        yst_ff_combined = round(yst_ff_probs[0] * yst_ff_probs[1] * yst_ff_probs[2] * yst_ff_probs[3] / 1000000, 1) if len(yst_ff_probs) == 4 else None
        # Check actual finishing positions for each trifecta leg
        def _leg_result(leg):
            res = all_results.get((venue_code, race_num, leg["horse_name"]), {})
            return {**leg, "position": res.get("position"), "scratched": res.get("scratched", False)}

        yst_tri_legs_result = [_leg_result(l) for l in yst_tri_legs]
        tri_positions = {l["position"] for l in yst_tri_legs_result if l["position"] and not l["scratched"]}
        tri_hit = tri_positions == {1, 2, 3}

        yst_ff_legs_result = [_leg_result(l) for l in yst_ff_legs] if len(yst_ff_legs) >= 4 else None
        ff_positions = {l["position"] for l in yst_ff_legs_result if l["position"] and not l["scratched"]} if yst_ff_legs_result else set()
        ff_hit = ff_positions == {1, 2, 3, 4}

        yst_trifecta = {
            "legs": yst_tri_legs_result,
            "combined_pct": yst_tri_combined,
            "hit": tri_hit,
            "first_four": yst_ff_legs_result if yst_ff_legs_result else None,
            "first_four_combined_pct": yst_ff_combined,
            "first_four_hit": ff_hit,
        } if len(yst_tri_legs) >= 3 else None

        output.append({
            "race_id": p.race_id,
            "venue": venue_code,
            "race_number": race_num,
            "horse_name": p.horse_name,
            "jockey": p.jockey,
            "trainer": p.trainer,
            "barrier": p.barrier,
            "weight": p.weight,
            "model_pct": model_pct,
            "place_probability": place_pct,
            "calibrated_win_rate": next((r2 for t, r2 in _CALIBRATED_WIN_RATES if model_pct >= t), 66),
            "sp": sp,
            "winner": winner,
            "placed": placed,
            "position": position,
            "winner_name": winner_name,
            "scratched": scratched,
            "payout": payout,
            "profit": profit,
            "stake": stake,
            "trifecta": yst_trifecta,
        })

    active = [o for o in output if not o["scratched"]]
    wins = [o for o in active if o["winner"]]
    placed_picks = [o for o in active if o["placed"] and not o["winner"]]
    total_staked = len(active) * stake
    total_returns = sum(o["payout"] for o in active)
    pnl = round(total_returns - total_staked, 2)

    return {
        "date": target_date,
        "picks": output,
        "summary": {
            "total": len(active),
            "wins": len(wins),
            "placed": len(placed_picks),
            "losses": len(active) - len(wins) - len(placed_picks),
            "win_rate": round(len(wins) / len(active) * 100, 1) if active else 0,
            "place_rate": round((len(wins) + len(placed_picks)) / len(active) * 100, 1) if active else 0,
            "pnl": pnl,
            "total_staked": total_staked,
            "total_returns": round(total_returns, 2),
            "roi_pct": round((pnl / total_staked) * 100, 1) if total_staked else 0,
        },
    }


def _assign_trifecta_tiers(picks: list[dict]) -> None:
    """
    Assign Hot/High/Strong tiers by percentile rank within the pick list,
    sorted descending by combined_pct.  Top 25% = Hot, next 35% = High,
    rest = Strong.  Requires picks already filtered to field_size >= 7.
    Mutates picks in-place.
    """
    picks.sort(key=lambda p: p["combined_pct"], reverse=True)
    n = len(picks)
    for i, pick in enumerate(picks):
        rank = i / n if n > 1 else 0
        if rank < 0.25:
            pick["tier"] = "hot"
        elif rank < 0.60:
            pick["tier"] = "high"
        else:
            pick["tier"] = "strong"
        pick["premium"] = pick["tier"] in ("hot", "high") and any(
            leg.get("overlay") and leg["overlay"] > 0.03 for leg in pick["legs"]
        )


@app.get("/api/edge/trifectas")
async def get_edge_trifectas():
    """Standalone trifecta picks ranked by combined place probability, tiered like win picks."""
    today = _today_aest()
    picks = []

    for i in range(4):
        target_date = (today + timedelta(days=i)).isoformat()
        prefix = f"{target_date}_"

        async with get_session() as session:
            exotic_result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
                .where(RunnerPredictionRow.exotic_model_rank >= 1)
                .where(RunnerPredictionRow.exotic_model_rank <= 4)
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            )
            exotic_rows = exotic_result.scalars().all()

            if not exotic_rows:
                # Fall back to place_model_rank until exotic model is trained
                exotic_result = await session.execute(
                    select(RunnerPredictionRow)
                    .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
                    .where(RunnerPredictionRow.place_model_rank >= 1)
                    .where(RunnerPredictionRow.place_model_rank <= 4)
                    .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                )
                exotic_rows = exotic_result.scalars().all()
                using_fallback = True
            else:
                using_fallback = False

            if not exotic_rows:
                continue

            race_ids = list({r.race_id for r in exotic_rows})
            race_result = await session.execute(
                select(RacePredictionRow).where(RacePredictionRow.race_id.in_(race_ids))
            )
            race_lookup = {r.race_id: r for r in race_result.scalars().all()}
            # Count actual non-cancelled runners per race for field size
            count_result = await session.execute(
                select(RunnerPredictionRow.race_id, func.count(RunnerPredictionRow.id))
                .where(RunnerPredictionRow.race_id.in_(race_ids))
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                .group_by(RunnerPredictionRow.race_id)
            )
            field_size_lookup = {row[0]: row[1] for row in count_result.all()}

            # Only fetch edge-qualifying win picks (same threshold as /api/edge)
            win_result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id.in_(race_ids))
                .where(RunnerPredictionRow.model_rank == 1)
                .where(RunnerPredictionRow.win_probability >= 0.295)
            )
            win_lookup = {r.race_id: r.horse_name for r in win_result.scalars().all()}

        # Group by race, sort by exotic_model_rank (or place_model_rank fallback)
        rank_key = (lambda r: r.place_model_rank or 99) if using_fallback else (lambda r: r.exotic_model_rank or 99)
        race_map: dict[str, list] = {}
        for row in exotic_rows:
            race_map.setdefault(row.race_id, []).append(row)

        for race_id, runners in race_map.items():
            runners.sort(key=rank_key)
            if len(runners) < 3:
                continue

            legs = runners[:3]
            probs = [r.place_probability for r in legs if r.place_probability]
            if len(probs) < 3:
                continue

            combined = probs[0] * probs[1] * probs[2]
            combined_pct = round(combined * 100, 1)

            race = race_lookup.get(race_id)
            field_size = (field_size_lookup.get(race_id)
                          or (race.field_size if race and race.field_size else 10))

            # Skip small fields and very low confidence picks
            if field_size < 7 or combined_pct < 5.0:
                continue

            n_fs = max(field_size, 3)
            multiplier = round(combined / (6.0 / (n_fs * (n_fs - 1) * (n_fs - 2))), 1)

            ff = runners[:4] if len(runners) >= 4 else None
            ff_probs = [r.place_probability for r in ff] if ff else []
            ff_combined_pct = round(
                ff_probs[0] * ff_probs[1] * ff_probs[2] * ff_probs[3] * 100, 1
            ) if len(ff_probs) == 4 and all(ff_probs) else None

            _, venue_code, race_num = _parse_race_id(race_id)

            # Alignment: compare exotic top-3 horses vs win model pick
            win_horse = win_lookup.get(race_id)
            exotic_horses = {r.horse_name for r in legs}
            if win_horse is None:
                alignment = "no_edge_pick"
            elif win_horse == legs[0].horse_name:
                alignment = "confirmed"       # win pick is exotic leg 1
            elif win_horse in exotic_horses:
                alignment = "partial"         # win pick is in exotic top 3 but not leg 1
            else:
                alignment = "diverge"         # win pick not in exotic top 3

            picks.append({
                "date": target_date,
                "race_id": race_id,
                "venue": venue_code,
                "state": race.state if race else None,
                "race_number": race_num,
                "race_name": race.race_name if race else None,
                "distance": race.distance if race else None,
                "track_condition": race.track_condition if race else None,
                "scheduled_time": race.scheduled_time if race else None,
                "combined_pct": combined_pct,
                "multiplier": multiplier,
                "field_size": field_size,
                "tier": "strong",
                "premium": False,
                "exotic_alignment": alignment,
                "win_horse": win_horse,
                "using_fallback_model": using_fallback,
                "legs": [
                    {
                        "tab_number": r.tab_number,
                        "horse_name": r.horse_name,
                        "barrier": r.barrier,
                        "jockey": r.jockey,
                        "trainer": r.trainer,
                        "weight": r.weight,
                        "place_pct": round(r.place_probability * 100, 1) if r.place_probability else None,
                        "win_pct": round(r.win_probability * 100, 1) if r.win_probability else None,
                        "best_available_odds": r.best_available_odds or 0,
                        "overlay": r.overlay,
                    }
                    for r in legs
                ],
                "first_four": [
                    {
                        "tab_number": r.tab_number,
                        "horse_name": r.horse_name,
                        "place_pct": round(r.place_probability * 100, 1) if r.place_probability else None,
                        "best_available_odds": r.best_available_odds or 0,
                    }
                    for r in ff
                ] if ff else None,
                "first_four_combined_pct": ff_combined_pct,
            })

    _assign_trifecta_tiers(picks)   # sorts + assigns Hot/High/Strong by percentile

    # Annotate finished races with actual positions
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    today_str = _today_aest().isoformat()
    finished_picks = [
        p for p in picks
        if (
            p["scheduled_time"] and datetime.fromisoformat(p["scheduled_time"].replace("Z", "+00:00")) < now_utc
        ) or (
            not p["scheduled_time"] and p["date"] == today_str
        )
    ]
    if finished_picks:
        finished_venues: dict[str, str] = {}
        for p in finished_picks:
            finished_venues[p["venue"]] = p["date"]

        async def _fetch_tri_results(venue: str, date: str) -> dict:
            slug = _meeting_slug(venue, date)
            try:
                client = get_tab_client()
                events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=30)
                out = {}
                for event in events:
                    rn = event.get("eventNumber")
                    for sel in event.get("selections") or []:
                        name = (sel.get("competitor") or {}).get("name")
                        if name:
                            pos = sel.get("selectionResult")
                            out[(venue, rn, name)] = {
                                "position": int(pos) if isinstance(pos, (int, float)) and pos > 0 else None,
                                "scratched": sel.get("status") == "SCRATCHED",
                            }
                return out
            except Exception:
                return {}

        result_batches = await asyncio.gather(*[
            _fetch_tri_results(v, d) for v, d in finished_venues.items()
        ])
        today_results: dict = {}
        for rb in result_batches:
            today_results.update(rb)

        for p in finished_picks:
            venue_code, race_num = p["venue"], p["race_number"]

            def _annotate(legs):
                out = []
                for l in legs:
                    res = today_results.get((venue_code, race_num, l["horse_name"]), {})
                    out.append({**l, "position": res.get("position"), "scratched": res.get("scratched", False)})
                return out

            p["legs"] = _annotate(p["legs"])
            tri_positions = {l["position"] for l in p["legs"] if l["position"] and not l["scratched"]}
            p["hit"] = tri_positions == {1, 2, 3}

            if p.get("first_four"):
                p["first_four"] = _annotate(p["first_four"])
                ff_positions = {l["position"] for l in p["first_four"] if l["position"] and not l["scratched"]}
                p["first_four_hit"] = ff_positions == {1, 2, 3, 4}

    return {"generated_at": datetime.utcnow().isoformat(), "picks": picks}


@app.get("/api/track-record")
async def get_track_record():
    """Public endpoint — tier win rates derived from live + backtest data."""
    async with get_session() as session:
        bt_result = await session.execute(
            select(BacktestResultRow.win_probability, BacktestResultRow.winner)
            .where(BacktestResultRow.source == "backtest")
            .where(BacktestResultRow.winner.isnot(None))
        )
        bt_rows = [{"win_prob": r.win_probability, "winner": bool(r.winner)} for r in bt_result.all()]

        hr_result = await session.execute(select(HistoricalResultRow))
        hr_map = {(r.race_id, r.horse_name): r for r in hr_result.scalars().all()}

        live_result = await session.execute(
            select(RunnerPredictionRow).where(RunnerPredictionRow.model_rank == 1)
        )
        live_rows = []
        for r in live_result.scalars().all():
            hr = hr_map.get((r.race_id, r.horse_name))
            if hr:
                live_rows.append({"win_prob": r.win_probability, "winner": bool(hr.winner)})

    unified = bt_rows + live_rows

    tiers = [
        {"badge": "hot",      "min": 0.45, "max": 1.0,  "conf_min": 45, "conf_max": None},
        {"badge": "high",     "min": 0.35, "max": 0.45, "conf_min": 35, "conf_max": 45},
        {"badge": "standard", "min": 0.30, "max": 0.35, "conf_min": 30, "conf_max": 35},
    ]
    output = []
    for tier in tiers:
        picks = [r for r in unified if tier["min"] <= r["win_prob"] < tier["max"]]
        wins  = [r for r in picks if r["winner"]]
        win_pct = round(len(wins) / len(picks) * 100) if picks else 0
        output.append({
            "badge":    tier["badge"],
            "win_pct":  win_pct,
            "races":    len(picks),
            "conf_min": tier["conf_min"],
            "conf_max": tier["conf_max"],
        })
    return {"tiers": output, "generated_at": datetime.utcnow().isoformat()}


# ── Frontend ─────────────────────────────────────────────────────────────────

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse("frontend/index.html", headers=_NO_CACHE)

@app.get("/edge", include_in_schema=False)
async def edge_page():
    return FileResponse("frontend/edge.html", headers=_NO_CACHE)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/check")
async def check():
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ── Meetings ──────────────────────────────────────────────────────────────────

@app.get("/api/meetings/{race_date}")
async def list_meetings(race_date: str = _today()):
    """List all Australian thoroughbred meetings for the given date."""
    _validate_date(race_date)
    client = get_tab_client()
    try:
        meetings = await client.get_meetings(race_date)
    except Exception as e:
        log.exception("list_meetings failed for %s", race_date)
        raise HTTPException(502, "Failed to fetch meetings from data provider")

    from horse_engine.clients.weather import get_weather_for_venue

    date_suffix = f"-{race_date.replace('-', '')}"
    items = [
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
    ]

    weathers = await asyncio.gather(
        *[get_weather_for_venue(it["venue"] or "", it["state"] or "", race_date) for it in items],
        return_exceptions=True,
    )
    for it, w in zip(items, weathers):
        it["weather"] = w if isinstance(w, dict) else None

    return {"date": race_date, "meetings": items}


@app.get("/api/meetings/{race_date}/{venue_code}")
async def get_meeting(race_date: str, venue_code: str):
    _validate_date(race_date)
    _validate_venue(venue_code)
    """Get all races at a meeting with current predictions if available."""
    client = get_tab_client()
    slug = _meeting_slug(venue_code, race_date)
    raw_races = await client.get_meeting_races(slug)

    race_list = [
        {
            "race_id": f"{race_date}_{venue_code}_R{r.get('eventNumber')}",
            "race_number": r.get("eventNumber"),
            "race_name": r.get("name"),
            "distance": r.get("distance"),
            "scheduled_time": r.get("startTime"),
            "time": r.get("startTime"),
            "status": r.get("status"),
            "enriched_at": None,
            "track_condition": None,
            "field_size": None,
            "prize_money": None,
        }
        for r in raw_races
    ]

    # For past dates punters returns nothing — derive race list from DB
    if not race_list:
        prefix = f"{_like_safe(race_date)}_{_like_safe(venue_code)}_R"
        async with get_session() as session:
            db_result = await session.execute(
                select(RunnerPredictionRow.race_id)
                .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
                .where(RunnerPredictionRow.model_rank == 1)
                .order_by(RunnerPredictionRow.race_id)
            )
            db_race_ids = [row[0] for row in db_result]
        # Also check historical results if no predictions
        if not db_race_ids:
            async with get_session() as session:
                hr_result = await session.execute(
                    select(HistoricalResultRow.race_id)
                    .where(HistoricalResultRow.race_id.like(f"{prefix}%"))
                    .distinct()
                    .order_by(HistoricalResultRow.race_id)
                )
                db_race_ids = [row[0] for row in hr_result]
        for rid in db_race_ids:
            try:
                rnum = int(rid.split("_R")[-1])
            except ValueError:
                continue
            race_list.append({
                "race_id": rid,
                "race_number": rnum,
                "race_name": None,
                "distance": None,
                "scheduled_time": None,
                "time": None,
                "status": "closed",
                "enriched_at": None,
                "track_condition": None,
                "field_size": None,
                "prize_money": None,
            })

    race_ids = [r["race_id"] for r in race_list]

    async with get_session() as session:
        # Which races have been enriched
        enriched_result = await session.execute(
            select(RunnerPredictionRow.race_id, RunnerPredictionRow.enriched_at)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        enriched_rows = {row.race_id: row.enriched_at for row in enriched_result}

        # Top pick per race
        top_picks = {race_id: None for race_id in race_ids}
        top_win_probs = {race_id: None for race_id in race_ids}
        tp_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        for p in tp_result.scalars().all():
            top_picks[p.race_id] = p.horse_name
            top_win_probs[p.race_id] = p.win_probability

        # Winners and placers per race from historical results
        hr_result = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.in_(race_ids))
            .where(HistoricalResultRow.winner == True)
        )
        winners = {r.race_id: r.horse_name for r in hr_result.scalars().all()}

        hr_placed = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.in_(race_ids))
            .where(HistoricalResultRow.placed == True)
        )
        placers = {}
        for r in hr_placed.scalars().all():
            placers.setdefault(r.race_id, set()).add(r.horse_name)

        # Top place probability per race
        tp_place_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        top_place_probs = {p.race_id: p.place_probability for p in tp_place_result.scalars().all()}

    enriched = bool(enriched_rows)

    def _model_correct(race_id: str):
        pick = top_picks.get(race_id)
        winner = winners.get(race_id)
        if not pick or not winner:
            return None
        return pick == winner

    def _model_placed(race_id: str):
        pick = top_picks.get(race_id)
        race_placers = placers.get(race_id)
        if not pick or not race_placers:
            return None
        return pick in race_placers

    races_out = []
    for r in race_list:
        rid = r["race_id"]
        races_out.append({
            **r,
            "enriched_at": enriched_rows.get(rid).isoformat() if enriched_rows.get(rid) else None,
            "model_correct": _model_correct(rid),
            "model_placed": _model_placed(rid),
            "top_win_probability": top_win_probs.get(rid),
            "top_place_probability": top_place_probs.get(rid),
        })

    return {
        "date": race_date,
        "venue": venue_code,
        "enriched": enriched,
        "races": races_out,
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
    stored_odds = {r.horse_name: r.best_available_odds for r in stored}
    stored_overlay = {r.horse_name: r.overlay or 0.0 for r in stored}

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
        flucs = sel.get("flucs") or {}
        fluc_low = flucs.get("low")
        current_odds = (float(tote_win) if tote_win
                        else float(fluc_low) if fluc_low
                        else float(sp) if sp
                        else None)
        result_pos = sel.get("selectionResult")
        actual_position = int(result_pos) if result_pos and int(result_pos) > 0 else None
        all_tote.append((horse, current_odds, actual_position))

    # Overround-free implied probs from current tote
    valid_odds = [o for _, o, _ in all_tote if o and o > 1.0]
    total_implied = sum(1 / o for o in valid_odds) if valid_odds else 0
    scale = 1.0 / total_implied if total_implied > 0 else 1.0

    # Find winner and placers for model flags
    winner_name = next((h for h, _, pos in all_tote if pos == 1), None)
    placed_names = {h for h, _, pos in all_tote if pos and pos <= 3}
    top_model_pick = stored[0].horse_name if stored else None
    model_correct = (winner_name == top_model_pick) if winner_name else None
    model_placed = (top_model_pick in placed_names) if (placed_names and top_model_pick) else None

    for horse, current_odds, actual_position in all_tote:
        model_prob = model_probs.get(horse, 0.0)
        if current_odds and current_odds > 1.0:
            raw_implied = 1.0 / current_odds
            orf_implied = round(raw_implied * scale, 4)
            overlay = round(model_prob - orf_implied, 4)
            display_odds = current_odds
        else:
            # No live tote or flucs data — last resort fallback to stored enrichment values
            overlay = stored_overlay.get(horse, 0.0)
            display_odds = stored_odds.get(horse)
            orf_implied = round(1.0 / display_odds, 4) if display_odds and display_odds > 1.0 else 0.0
        runners_odds.append({
            "horse_name": horse,
            "current_tote_win": display_odds,
            "implied_prob": orf_implied,
            "model_win_prob": round(model_prob, 4),
            "overlay": overlay,
            "value": overlay > 0.05 and display_odds and display_odds >= 3.0,
            "actual_position": actual_position,
            "is_top_pick": horse == top_model_pick,
        })

    runners_odds.sort(key=lambda x: x["model_win_prob"], reverse=True)

    return {
        "race_id": race_id,
        "fetched_at": datetime.utcnow().isoformat(),
        "settled": winner_name is not None,
        "winner": winner_name,
        "model_correct": model_correct,
        "model_placed": model_placed,
        "runners": runners_odds,
    }


@app.get("/api/races/{race_id}/odds-trend")
async def odds_trend(race_id: str, horse: str = Query(...)):
    """Return the 15-min odds snapshot history for a horse in a race."""
    async with get_session() as session:
        result = await session.execute(
            select(OddsSnapshotRow)
            .where(OddsSnapshotRow.race_id == race_id)
            .where(OddsSnapshotRow.horse_name == horse)
            .order_by(OddsSnapshotRow.snapshotted_at.asc())
        )
        rows = result.scalars().all()
    return {
        "race_id": race_id,
        "horse_name": horse,
        "snapshots": [
            {
                "time": r.snapshotted_at.isoformat(),
                "win_odds": r.win_odds,
                "minutes_to_jump": r.minutes_to_jump,
                "source": r.source,
            }
            for r in rows
        ],
    }


_RACE_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z0-9-]{1,60}_R\d{1,2}$")

@app.post("/api/races/{race_id}/enrich")
async def enrich_race(race_id: str, force: bool = Query(False)):
    """Enrich a specific race. race_id format: {date}_{venue}_R{num}"""
    if not _RACE_ID_RE.match(race_id):
        raise HTTPException(400, "Invalid race_id format. Expected: YYYY-MM-DD_venue_RN")
    parts = race_id.split("_")
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

        race = await client.parse_race(raw_event, race_date, venue_name, state)
        venue_cal = await _load_venue_calibration()
        predictions, _ = await enrich_and_predict_race(race, model, venue_calibration=venue_cal)

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
    except Exception:
        log.exception("Enrich failed for %s", race_id)
        raise HTTPException(500, "Enrichment failed — check server logs")


@app.post("/api/meetings/{race_date}/{venue_code}/enrich")
async def enrich_meeting_endpoint(race_date: str, venue_code: str):
    """Enrich all races at a meeting."""
    _validate_date(race_date)
    _validate_venue(venue_code)
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
            race = await client.parse_race(full_event, race_date, venue_name, state)
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
async def retrain_model(
    days: int = Query(0, ge=0, le=365),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Retrain logistic regression on stored historical results.
    days=0 (default) uses all available data. days=N uses only the last N days.
    """
    _check_admin(x_cron_secret)
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

    async def _do_retrain():
        m = HorseModel()
        s = m.train(training_data)
        async with get_session() as sess:
            await save_model_weights(sess, s["weights"])
        log.info("[retrain] complete — %d examples, accuracy=%.3f", len(training_data), s.get("accuracy", 0))

    asyncio.create_task(_do_retrain())
    return {"status": "retrain_started", "training_days": days or "all", "training_examples": len(training_data)}


@app.post("/api/admin/retrain-place")
async def retrain_place_model(
    days: int = Query(0, ge=0, le=365),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Train the place model on P(position ≤ 3) labels.
    Uses the same feature vectors as the win model but with placed=True as the target.
    """
    _check_admin(x_cron_secret)
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
        raise HTTPException(400, f"Need at least 50 results to train (have {len(hr_rows)})")

    pred_by_key = {(p.race_id, p.horse_name): p for p in pred_rows}

    training_data = []
    for row in hr_rows:
        pred = pred_by_key.get((row.race_id, row.horse_name))
        if not pred:
            continue
        try:
            er = EnrichedRunner(**json.loads(pred.enriched_json))
            fv = build_feature_vector(er)
            label = 1 if row.placed else 0   # ← placed label, not winner
            training_data.append((fv, label))
        except Exception as e:
            log.debug("Skipping place retrain row %s/%s: %s", row.race_id, row.horse_name, e)

    if not training_data:
        raise HTTPException(400, "No matched training examples for place model")

    placed_count = sum(1 for _, label in training_data if label == 1)
    log.info("[place-retrain] %d examples, %d placed (%.1f%%)",
             len(training_data), placed_count, placed_count / len(training_data) * 100)

    async def _do_retrain():
        m = PlaceModel()
        s = m.train(training_data)
        async with get_session() as sess:
            await save_place_model_weights(sess, s["weights"])
        log.info("[place-retrain] complete — %d examples, accuracy=%.3f", len(training_data), s.get("accuracy", 0))

    asyncio.create_task(_do_retrain())
    return {
        "status": "place_retrain_started",
        "training_examples": len(training_data),
        "placed_examples": placed_count,
        "placed_rate_pct": round(placed_count / len(training_data) * 100, 1),
    }


@app.post("/api/admin/retrain-exotic")
async def retrain_exotic_model(
    days: int = Query(0, ge=0, le=365),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Train the exotic model using race-grouped trifecta-aware loss.
    Only uses field_size >= 7 races (trifecta-eligible fields).
    Groups runners by race so the trifecta box objective can be applied.
    """
    _check_admin(x_cron_secret)
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
        raise HTTPException(400, f"Need at least 50 results to train (have {len(hr_rows)})")

    pred_by_key = {(p.race_id, p.horse_name): p for p in pred_rows}

    # Group by race_id for race-level trifecta training
    race_data: dict[str, list[tuple[list[float], int, int | None]]] = {}
    for row in hr_rows:
        pred = pred_by_key.get((row.race_id, row.horse_name))
        if not pred:
            continue
        try:
            er = EnrichedRunner(**json.loads(pred.enriched_json))
            fv = build_feature_vector(er)
            label = 1 if row.placed else 0
            race_data.setdefault(row.race_id, []).append((fv, label, row.position))
        except Exception as e:
            log.debug("Skipping exotic retrain row %s/%s: %s", row.race_id, row.horse_name, e)

    # Filter to field_size >= 7 with complete top-3 data
    race_groups = []
    for race_id, runners in race_data.items():
        if len(runners) < 7:
            continue
        top3_count = sum(1 for _, lbl, _ in runners if lbl == 1)
        if top3_count != 3:
            continue
        race_groups.append([(fv, lbl) for fv, lbl, _ in runners])

    if not race_groups:
        raise HTTPException(400, "No eligible trifecta races found for exotic training")

    total_runners = sum(len(r) for r in race_groups)
    log.info("[exotic-retrain] %d races, %d runners", len(race_groups), total_runners)

    async def _do_retrain():
        m = ExoticModel()
        s = m.train_exotic(race_groups)
        async with get_session() as sess:
            await save_exotic_model_weights(sess, s["weights"])
        log.info(
            "[exotic-retrain] complete — %d races, box_hit_rate=%.3f",
            len(race_groups), s.get("box_hit_rate", 0),
        )

    asyncio.create_task(_do_retrain())
    return {
        "status": "exotic_retrain_started",
        "eligible_races": len(race_groups),
        "total_runners": total_runners,
    }


@app.get("/api/admin/backtest-exotic")
async def backtest_exotic(x_cron_secret: Optional[str] = Header(None)):
    """
    Window sweep + feature ablation for the exotic (trifecta box) model.

    For each training window in [30, 60, 90, 180, 270]:
      - Train a fresh ExoticModel on races outside the 14-day holdout
      - Score holdout races, compute trifecta box hit rate by tier (Hot/High/Strong)

    Then, using the best window model:
      - Zero out each feature in turn and re-score holdout
      - Report delta hit rate so we can identify valuable vs noisy features

    Returns: per-window results, best window, feature ablation table.
    """
    _check_admin(x_cron_secret)
    import math

    holdout_days = 14
    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(select(HistoricalResultRow))
        all_hr = hr_result.scalars().all()
        pred_result = await session.execute(
            select(RunnerPredictionRow).where(RunnerPredictionRow.enriched_json.isnot(None))
        )
        all_pred = pred_result.scalars().all()

    pred_by_key = {(p.race_id, p.horse_name): p for p in all_pred}

    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))

    def _score_field(model: ExoticModel, fvs: list[list[float]]) -> list[float]:
        return [_sigmoid(sum(w * f for w, f in zip(model.weights, fv)) + model.bias) for fv in fvs]

    def _box_hit_rate_by_tier(model: ExoticModel, holdout_groups: list[tuple]) -> dict:
        """Score holdout race groups, compute box hit rate overall and by tier."""
        combined_scores = []
        for race_id, fvs, positions in holdout_groups:
            probs = _score_field(model, fvs)
            ranked = sorted(zip(probs, positions), reverse=True)
            top3_probs = [p for p, _ in ranked[:3]]
            if len(top3_probs) < 3:
                continue
            combined = top3_probs[0] * top3_probs[1] * top3_probs[2]
            if combined * 100 < 5.0:
                continue
            combined_scores.append((race_id, combined, fvs, positions, probs))

        if not combined_scores:
            return {"races": 0, "box_hit_rate": 0.0, "hot": {}, "high": {}, "strong": {}}

        sorted_by_combined = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        n = len(sorted_by_combined)
        hot_threshold  = sorted_by_combined[int(n * 0.25)][1] if n > 4 else float("inf")
        high_threshold = sorted_by_combined[int(n * 0.60)][1] if n > 4 else float("inf")

        tiers: dict[str, dict] = {
            "hot":    {"races": 0, "hits": 0},
            "high":   {"races": 0, "hits": 0},
            "strong": {"races": 0, "hits": 0},
        }
        total_hits = 0

        for _, combined, fvs, positions, probs in sorted_by_combined:
            ranked = sorted(zip(probs, range(len(probs))), reverse=True)
            predicted_top3_idx = {idx for _, idx in ranked[:3]}
            actual_top3_idx = {i for i, pos in enumerate(positions) if pos in (1, 2, 3)}

            if len(actual_top3_idx) != 3:
                continue

            hit = predicted_top3_idx == actual_top3_idx

            if combined >= hot_threshold:
                tier = "hot"
            elif combined >= high_threshold:
                tier = "high"
            else:
                tier = "strong"

            tiers[tier]["races"] += 1
            if hit:
                tiers[tier]["hits"] += 1
                total_hits += 1

        total_races = sum(t["races"] for t in tiers.values())

        def _rate(t):
            return round(t["hits"] / t["races"] * 100, 1) if t["races"] else 0.0

        return {
            "races": total_races,
            "box_hit_rate": round(total_hits / total_races * 100, 1) if total_races else 0.0,
            "hot":    {**tiers["hot"],    "hit_rate_pct": _rate(tiers["hot"])},
            "high":   {**tiers["high"],   "hit_rate_pct": _rate(tiers["high"])},
            "strong": {**tiers["strong"], "hit_rate_pct": _rate(tiers["strong"])},
        }

    # Build holdout race groups: field_size >= 7, exactly 3 top-3 finishers, within holdout window
    holdout_race_data: dict[str, list[tuple[list[float], int]]] = {}
    for row in all_hr:
        if row.race_id < holdout_cutoff:
            continue
        pred = pred_by_key.get((row.race_id, row.horse_name))
        if not pred or row.position is None:
            continue
        try:
            er = EnrichedRunner(**json.loads(pred.enriched_json))
            fv = build_feature_vector(er)
            holdout_race_data.setdefault(row.race_id, []).append((fv, row.position))
        except Exception:
            continue

    holdout_groups: list[tuple] = []
    for race_id, runners in holdout_race_data.items():
        if len(runners) < 7:
            continue
        top3_count = sum(1 for _, pos in runners if pos in (1, 2, 3))
        if top3_count != 3:
            continue
        fvs = [fv for fv, _ in runners]
        positions = [pos for _, pos in runners]
        holdout_groups.append((race_id, fvs, positions))

    if not holdout_groups:
        return {"error": "No eligible holdout races found (need field_size >= 7 with complete top-3 results)"}

    # Window sweep
    window_results = []
    best_window = None
    best_hit_rate = -1.0
    best_model_weights: dict | None = None

    for window in _CANDIDATE_WINDOWS:
        train_cutoff = (today - timedelta(days=window)).isoformat()

        race_train_data: dict[str, list[tuple[list[float], int]]] = {}
        for row in all_hr:
            if row.race_id < train_cutoff or row.race_id >= holdout_cutoff:
                continue
            pred = pred_by_key.get((row.race_id, row.horse_name))
            if not pred:
                continue
            try:
                er = EnrichedRunner(**json.loads(pred.enriched_json))
                fv = build_feature_vector(er)
                label = 1 if row.placed else 0
                race_train_data.setdefault(row.race_id, []).append((fv, label))
            except Exception:
                continue

        race_groups_train = []
        for race_id, runners in race_train_data.items():
            if len(runners) < 7:
                continue
            top3_count = sum(1 for _, lbl in runners if lbl == 1)
            if top3_count != 3:
                continue
            race_groups_train.append(runners)

        if len(race_groups_train) < 20:
            window_results.append({
                "window_days": window,
                "training_races": len(race_groups_train),
                "skipped": True,
                "reason": "insufficient training races",
            })
            continue

        m = ExoticModel()
        stats = m.train_exotic(race_groups_train)
        holdout_stats = _box_hit_rate_by_tier(m, holdout_groups)

        result = {
            "window_days": window,
            "training_races": len(race_groups_train),
            "training_box_hit_rate": round(stats.get("box_hit_rate", 0) * 100, 1),
            "holdout_races": holdout_stats["races"],
            "holdout_box_hit_rate": holdout_stats["box_hit_rate"],
            "by_tier": {
                "hot":    holdout_stats["hot"],
                "high":   holdout_stats["high"],
                "strong": holdout_stats["strong"],
            },
        }
        window_results.append(result)
        log.info("[backtest-exotic] window=%d train_races=%d holdout_box_hit=%.1f%%",
                 window, len(race_groups_train), holdout_stats["box_hit_rate"])

        if holdout_stats["box_hit_rate"] > best_hit_rate:
            best_hit_rate = holdout_stats["box_hit_rate"]
            best_window = window
            best_model_weights = {"weights": list(m.weights), "bias": m.bias}

    if best_model_weights is None:
        return {
            "error": "No valid training windows found",
            "holdout_races": len(holdout_groups),
            "window_results": window_results,
        }

    # Feature ablation using best window model
    from horse_engine.prediction.features import FEATURE_NAMES, NUM_FEATURES

    best_m = ExoticModel()
    best_m.weights = best_model_weights["weights"]
    best_m.bias = best_model_weights["bias"]
    baseline_stats = _box_hit_rate_by_tier(best_m, holdout_groups)
    baseline_hit_rate = baseline_stats["box_hit_rate"]

    ablation_results = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        # Zero out this feature in all holdout race groups
        zeroed_groups = []
        for race_id, fvs, positions in holdout_groups:
            zeroed_fvs = []
            for fv in fvs:
                zfv = list(fv)
                if feat_idx < len(zfv):
                    zfv[feat_idx] = 0.0
                zeroed_fvs.append(zfv)
            zeroed_groups.append((race_id, zeroed_fvs, positions))

        ablated_stats = _box_hit_rate_by_tier(best_m, zeroed_groups)
        delta = round(ablated_stats["box_hit_rate"] - baseline_hit_rate, 1)
        ablation_results.append({
            "feature": feat_name,
            "feature_idx": feat_idx,
            "weight": round(best_m.weights[feat_idx] if feat_idx < len(best_m.weights) else 0.0, 4),
            "baseline_hit_rate": baseline_hit_rate,
            "ablated_hit_rate": ablated_stats["box_hit_rate"],
            "delta": delta,
            "verdict": "valuable" if delta < -1.0 else ("noisy/harmful" if delta > 1.0 else "neutral"),
        })

    ablation_results.sort(key=lambda x: x["delta"])

    return {
        "holdout_races": len(holdout_groups),
        "holdout_days": holdout_days,
        "best_window": best_window,
        "best_holdout_box_hit_rate": best_hit_rate,
        "window_results": window_results,
        "feature_ablation": ablation_results,
    }


# ── Admin: seed results ───────────────────────────────────────────────────────

@app.post("/api/admin/results/{race_date}")
async def seed_results(race_date: str, x_cron_secret: Optional[str] = Header(None)):
    """Fetch race results from punters for a past date and store as training data."""
    _check_admin(x_cron_secret)
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
                            race = await client.parse_race(event, race_date, venue_name, state)
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
    _check_admin(x_cron_secret)
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
    _check_admin(x_cron_secret)

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


# ── Backtest (retroactive) ────────────────────────────────────────────────────

_backtest_state: dict = {"running": False, "processed": 0, "total": 0, "errors": 0, "started_at": None}


async def _save_backtest_state(start_date: str, end_date: str, last_completed: str | None) -> None:
    from sqlalchemy import delete as sa_delete2
    async with get_session() as session:
        await session.execute(sa_delete2(BacktestStateRow))
        session.add(BacktestStateRow(
            start_date=start_date,
            end_date=end_date,
            last_completed_date=last_completed,
        ))
        await session.commit()


async def _load_backtest_state() -> dict | None:
    async with get_session() as session:
        result = await session.execute(select(BacktestStateRow))
        row = result.scalars().first()
        if not row:
            return None
        return {
            "start_date": row.start_date,
            "end_date": row.end_date,
            "last_completed_date": row.last_completed_date,
        }


async def _run_backtest_range(start_date: str, end_date: str) -> None:
    """Retroactively run model on historical races and store in backtest_results."""
    global _backtest_state
    client = get_tab_client()
    async with get_session() as session:
        model = await _load_model(session)

    # Resume from last completed date if available
    saved = await _load_backtest_state()
    if saved and saved["last_completed_date"] and saved["last_completed_date"] >= start_date:
        resume_from = (date.fromisoformat(saved["last_completed_date"]) + timedelta(days=1)).isoformat()
        log.info("Resuming backtest from %s (last completed: %s)", resume_from, saved["last_completed_date"])
        current = date.fromisoformat(resume_from)
    else:
        current = date.fromisoformat(start_date)

    await _save_backtest_state(start_date, end_date, saved["last_completed_date"] if saved else None)

    end = date.fromisoformat(end_date)
    total_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    days_done = (current - date.fromisoformat(start_date)).days
    _backtest_state.update({"running": True, "processed": days_done, "total": total_days, "errors": 0})

    while current <= end:
        date_str = current.isoformat()
        try:
            meetings = await asyncio.wait_for(client.get_meetings(date_str), timeout=90)
            for m in meetings:
                slug = m.get("slug", "")
                venue_code = slug.replace(f"-{date_str.replace('-', '')}", "") if slug else ""
                venue_name = m.get("venue", venue_code)
                state_code = m.get("state", "")
                try:
                    events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=120)
                    rows_to_insert = []
                    for event in events:
                        selections = event.get("selections") or []
                        # Only process races with official results
                        if not any(
                            isinstance(s.get("selectionResult"), int) and s["selectionResult"] >= 1
                            for s in selections
                        ):
                            continue
                        # Actual positions from selectionResult
                        actuals = {
                            (s.get("competitor") or {}).get("name"): s["selectionResult"]
                            for s in selections
                            if isinstance(s.get("selectionResult"), int) and s["selectionResult"] >= 1
                        }
                        try:
                            race_num = event.get("eventNumber")
                            race_id = f"{date_str}_{venue_code}_R{race_num}"
                            event["_meeting"] = {"slug": slug, "railPosition": m.get("rail_position", "")}
                            race = await asyncio.wait_for(
                                client.parse_race(event, date_str, venue_name, state_code), timeout=60
                            )
                            predictions, _ = await enrich_and_predict_race(race, model)
                            for pred in predictions:
                                pos = actuals.get(pred.runner.horse_name)
                                rows_to_insert.append(BacktestResultRow(
                                    race_id=race_id,
                                    race_date=date_str,
                                    venue=venue_code,
                                    horse_name=pred.runner.horse_name,
                                    model_rank=pred.model_rank,
                                    win_probability=round(pred.win_prob, 4),
                                    starting_price=pred.runner.fixed_win_odds or pred.runner.best_available_odds,
                                    actual_position=pos,
                                    winner=(pos == 1),
                                    source="backtest",
                                ))
                        except Exception as e:
                            log.debug("Backtest race error %s R%s: %s", slug, event.get("eventNumber"), e)
                            _backtest_state["errors"] += 1
                    if rows_to_insert:
                        async with get_session() as session:
                            from sqlalchemy import delete as sa_delete
                            await session.execute(
                                sa_delete(BacktestResultRow)
                                .where(BacktestResultRow.race_date == date_str)
                                .where(BacktestResultRow.venue == venue_code)
                            )
                            for row in rows_to_insert:
                                session.add(row)
                            await session.commit()
                except Exception as e:
                    log.warning("Backtest meeting error %s: %s", slug, e)
                    _backtest_state["errors"] += 1
        except Exception as e:
            log.warning("Backtest date error %s: %s", date_str, e)
            _backtest_state["errors"] += 1

        _backtest_state["processed"] += 1
        await _save_backtest_state(start_date, end_date, date_str)  # persist progress
        current += timedelta(days=1)
        await asyncio.sleep(0.1)  # be polite to punters API

    _backtest_state["running"] = False


@app.post("/api/admin/backtest/run")
async def run_backtest(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    force: bool = Query(False),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Retroactively run model on historical races. Fires in background — check /status.
    If start_date/end_date omitted, resumes from saved DB state automatically.
    """
    _check_admin(x_cron_secret)
    if _backtest_state["running"] and not force:
        return {"status": "already_running", "state": _backtest_state}

    # Load saved state if no dates provided
    if not start_date or not end_date:
        saved = await _load_backtest_state()
        if not saved:
            raise HTTPException(400, "No saved backtest state — provide start_date and end_date")
        start_date = saved["start_date"]
        end_date = saved["end_date"]

    _validate_date(start_date)
    _validate_date(end_date)
    _backtest_state["running"] = False
    _backtest_state["started_at"] = datetime.utcnow().isoformat()
    asyncio.create_task(_run_backtest_range(start_date, end_date))
    return {"status": "started", "start_date": start_date, "end_date": end_date}


@app.get("/api/admin/backtest/status")
async def backtest_status(x_cron_secret: Optional[str] = Header(None)):
    _check_admin(x_cron_secret)
    async with get_session() as session:
        count_result = await session.execute(
            select(func.count()).select_from(BacktestResultRow)
        )
        total_rows = count_result.scalar() or 0
    saved = await _load_backtest_state()
    return {
        **_backtest_state,
        "total_rows_stored": total_rows,
        "saved_state": saved,
    }


@app.get("/api/admin/backtest/analysis")
async def backtest_analysis(x_cron_secret: Optional[str] = Header(None)):
    """
    Threshold analysis across both retroactive backtest and live predictions.
    Returns: cutover_date, and for each threshold the bet count, win rate, and ROI.
    """
    _check_admin(x_cron_secret)

    async with get_session() as session:
        # Cutover date = earliest date we have live predictions
        cutover_result = await session.execute(
            select(func.min(RunnerPredictionRow.race_id))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        earliest_race_id = cutover_result.scalar()
        cutover_date = earliest_race_id[:10] if earliest_race_id else None

        # --- Retroactive backtest rows ---
        bt_result = await session.execute(
            select(BacktestResultRow).where(BacktestResultRow.source == "backtest")
        )
        bt_rows = bt_result.scalars().all()

        # --- Live: runner_predictions joined with historical_results ---
        hr_result = await session.execute(select(HistoricalResultRow))
        hr_map = {(r.race_id, r.horse_name): r for r in hr_result.scalars().all()}

        live_result = await session.execute(
            select(RunnerPredictionRow).where(RunnerPredictionRow.model_rank == 1)
        )
        live_rows = live_result.scalars().all()

    # Build unified row list: (win_probability, starting_price, winner, source)
    unified = []
    for r in bt_rows:
        if r.win_probability is not None:
            unified.append({
                "win_prob": r.win_probability,
                "sp": r.starting_price,
                "winner": bool(r.winner),
                "source": "backtest",
                "race_date": r.race_date,
            })
    for r in live_rows:
        hr = hr_map.get((r.race_id, r.horse_name))
        if hr and r.win_probability is not None:
            unified.append({
                "win_prob": r.win_probability,
                "sp": hr.starting_price,
                "winner": bool(hr.winner),
                "source": "live",
                "race_date": r.race_id[:10],
            })

    thresholds = [20, 25, 30, 35, 40, 45, 50]
    analysis = []
    for t in thresholds:
        t_frac = t / 100.0
        subset = [r for r in unified if r["win_prob"] >= t_frac]
        if not subset:
            analysis.append({"threshold_pct": t, "bets": 0})
            continue
        wins = sum(1 for r in subset if r["winner"])
        sp_list = [r["sp"] for r in subset if r["sp"] and r["sp"] > 1.0]
        pnl = sum((r["sp"] - 1.0) if r["winner"] and r["sp"] else (-1.0) for r in subset)
        roi = round(pnl / len(subset) * 100, 1) if subset else 0
        bt_count = sum(1 for r in subset if r["source"] == "backtest")
        live_count = sum(1 for r in subset if r["source"] == "live")
        analysis.append({
            "threshold_pct": t,
            "bets": len(subset),
            "wins": wins,
            "win_pct": round(wins / len(subset) * 100, 1),
            "roi_pct": roi,
            "avg_sp": round(sum(sp_list) / len(sp_list), 2) if sp_list else None,
            "backtest_bets": bt_count,
            "live_bets": live_count,
        })

    return {
        "cutover_date": cutover_date,
        "note": "Predictions before cutover_date are retroactive (backtest). After are live.",
        "total_unified_rows": len(unified),
        "backtest_rows": sum(1 for r in unified if r["source"] == "backtest"),
        "live_rows": sum(1 for r in unified if r["source"] == "live"),
        "thresholds": analysis,
    }


@app.get("/api/admin/trifecta-analysis")
async def trifecta_analysis(x_cron_secret: Optional[str] = Header(None)):
    """
    Assess the model's ability to pick trifectas.
    Uses both retroactive backtest data and live prediction data.
    """
    _check_admin(x_cron_secret)

    from collections import defaultdict

    async with get_session() as session:
        # --- Backtest rows (have model_rank + actual_position) ---
        bt_result = await session.execute(
            select(BacktestResultRow)
            .where(BacktestResultRow.actual_position.isnot(None))
            .where(BacktestResultRow.model_rank <= 3)
        )
        bt_rows = bt_result.scalars().all()

        # --- Live: runner_predictions (ranks 1-3) + historical_results ---
        hr_result = await session.execute(select(HistoricalResultRow))
        hr_map = {(r.race_id, r.horse_name): r for r in hr_result.scalars().all()}

        live_result = await session.execute(
            select(RunnerPredictionRow).where(RunnerPredictionRow.model_rank <= 3)
        )
        live_rows = live_result.scalars().all()

    # Build per-race maps: {race_id: {model_rank: actual_position}}
    def _analyse(race_map: dict[str, dict[int, int]]) -> dict:
        races_with_full_top3 = 0
        exacta_hits = 0       # rank1→pos1, rank2→pos2
        exacta_box_hits = 0   # {rank1, rank2} positions == {1, 2}
        trifecta_hits = 0     # rank1→1, rank2→2, rank3→3
        trifecta_box_hits = 0 # positions of rank1+2+3 == {1, 2, 3}
        rank1_wins = 0
        rank1_places = 0      # pos <= 3
        races_with_top1 = 0

        for race_id, ranks in race_map.items():
            pos1 = ranks.get(1)
            pos2 = ranks.get(2)
            pos3 = ranks.get(3)

            if pos1 is not None:
                races_with_top1 += 1
                if pos1 == 1:
                    rank1_wins += 1
                if pos1 <= 3:
                    rank1_places += 1

            if pos1 is not None and pos2 is not None:
                if {pos1, pos2} == {1, 2}:
                    exacta_box_hits += 1
                if pos1 == 1 and pos2 == 2:
                    exacta_hits += 1

            if pos1 is not None and pos2 is not None and pos3 is not None:
                races_with_full_top3 += 1
                if {pos1, pos2, pos3} == {1, 2, 3}:
                    trifecta_box_hits += 1
                if pos1 == 1 and pos2 == 2 and pos3 == 3:
                    trifecta_hits += 1

        def pct(n, d):
            return round(n / d * 100, 1) if d else None

        return {
            "races_with_top1_result": races_with_top1,
            "races_with_full_top3_result": races_with_full_top3,
            "rank1_win_rate_pct": pct(rank1_wins, races_with_top1),
            "rank1_place_rate_pct": pct(rank1_places, races_with_top1),
            "exacta_straight_hits": exacta_hits,
            "exacta_straight_hit_rate_pct": pct(exacta_hits, races_with_full_top3),
            "exacta_boxed_hits": exacta_box_hits,
            "exacta_boxed_hit_rate_pct": pct(exacta_box_hits, races_with_full_top3),
            "trifecta_straight_hits": trifecta_hits,
            "trifecta_straight_hit_rate_pct": pct(trifecta_hits, races_with_full_top3),
            "trifecta_boxed_hits": trifecta_box_hits,
            "trifecta_boxed_hit_rate_pct": pct(trifecta_box_hits, races_with_full_top3),
        }

    # Build backtest race map
    bt_race_map: dict[str, dict[int, int]] = defaultdict(dict)
    for r in bt_rows:
        if r.actual_position and r.model_rank:
            bt_race_map[r.race_id][r.model_rank] = r.actual_position

    # Build live race map
    live_race_map: dict[str, dict[int, int]] = defaultdict(dict)
    for r in live_rows:
        hr = hr_map.get((r.race_id, r.horse_name))
        if hr and hr.position and r.model_rank:
            live_race_map[r.race_id][r.model_rank] = hr.position

    # Combined
    combined_map: dict[str, dict[int, int]] = defaultdict(dict)
    for race_id, ranks in bt_race_map.items():
        combined_map[race_id].update(ranks)
    for race_id, ranks in live_race_map.items():
        combined_map[race_id].update(ranks)

    return {
        "note": "Positions sourced from historical_results (live) and backtest_results (retroactive).",
        "backtest": _analyse(bt_race_map),
        "live": _analyse(live_race_map),
        "combined": _analyse(combined_map),
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
        d = by_date.setdefault(race_date, {
            "races": 0, "wins": 0, "places": 0, "value_pnl": 0.0, "value_bets": 0,
            "tier_premium": 0, "tier_hot": 0, "tier_high": 0, "tier_strong": 0,
        })
        d["races"] += 1
        if actual.winner:
            d["wins"] += 1
        if actual.placed:
            d["places"] += 1
        sp = actual.starting_price or 0.0
        overlay = pick.overlay or 0.0
        model_pct = round((pick.win_probability or 0) * 100, 1)
        if overlay > 0.05 and sp >= 3.0:
            d["value_bets"] += 1
            d["value_pnl"] += (sp - 1) if actual.winner else -1.0
            if model_pct >= 30:
                d["tier_premium"] += 1
            elif model_pct >= 45:
                d["tier_hot"] += 1
            elif model_pct >= 35:
                d["tier_high"] += 1
            else:
                d["tier_strong"] += 1

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
            "tier_premium": d["tier_premium"],
            "tier_hot": d["tier_hot"],
            "tier_high": d["tier_high"],
            "tier_strong": d["tier_strong"],
        })

    total_races = sum(d["races"] for d in by_date.values())
    total_wins = sum(d["wins"] for d in by_date.values())
    return {
        "days": days,
        "overall_win_rate": round(total_wins / total_races, 3) if total_races else None,
        "overall_races": total_races,
        "summary": summary,
    }


@app.get("/api/admin/backtest-trifecta")
async def backtest_trifecta(x_cron_secret: Optional[str] = Header(None)):
    """
    Backtest the place model's ability to box trifectas and first fours.
    Re-scores all HistoricalResultRow entries through the current place model
    (using stored feature_vector_json), then checks whether the top 3/4 picks
    finished in positions 1-3/1-4.  Breaks results down by confidence tier.
    """
    _check_admin(x_cron_secret)
    import math

    async with get_session() as session:
        place_weights_dict = await load_place_model_weights(session)
        pm = PlaceModel.from_weights_dict(place_weights_dict) if place_weights_dict else PlaceModel()

        hist_result = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.position.isnot(None))
            .where(HistoricalResultRow.feature_vector_json.isnot(None))
        )
        hist_rows = hist_result.scalars().all()

    if not hist_rows:
        return {"error": "No historical results with feature vectors found. Run /api/retrain first."}

    from horse_engine.prediction.features import NUM_FEATURES

    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))

    def _score(fv_raw: str) -> float | None:
        try:
            data = json.loads(fv_raw)
            if isinstance(data, list):
                fv = data
                if len(fv) < NUM_FEATURES:
                    fv = fv + [0.0] * (NUM_FEATURES - len(fv))
                elif len(fv) > NUM_FEATURES:
                    fv = fv[:NUM_FEATURES]
            elif isinstance(data, dict):
                # Stored as EnrichedRunner JSON — rebuild feature vector
                er = EnrichedRunner(**{k: v for k, v in data.items() if k in EnrichedRunner.model_fields})
                fv = build_feature_vector(er)
            else:
                return None
            z = sum(w * f for w, f in zip(pm.weights, fv)) + pm.bias
            return _sigmoid(z)
        except Exception:
            return None

    # Group by race_id
    race_map: dict[str, list] = {}
    for row in hist_rows:
        race_map.setdefault(row.race_id, []).append(row)

    totals = {"tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}}
    field_size_dist: dict[int, int] = {}

    # First pass: collect combined probs for all qualifying races (field >= 7)
    # to compute percentile-based tier thresholds (same logic as endpoint)
    qualifying: list[tuple[str, float, list, int, list]] = []  # (race_id, combined, valid, field_size, rows)
    for race_id, rows in race_map.items():
        valid = [(r, _score(r.feature_vector_json)) for r in rows if r.position is not None]
        valid = [(r, s) for r, s in valid if s is not None]
        if len(valid) < 3:
            continue
        field_size = len(rows)
        if field_size < 7:
            continue
        valid.sort(key=lambda x: x[1], reverse=True)
        top3_probs = [s for _, s in valid[:3]]
        tri_combined = top3_probs[0] * top3_probs[1] * top3_probs[2]
        if tri_combined * 100 < 5.0:
            continue
        qualifying.append((race_id, tri_combined, valid, field_size, rows))

    # Derive percentile thresholds: top 25% = hot, top 60% = high, rest = strong
    qualifying_sorted = sorted(qualifying, key=lambda x: x[1], reverse=True)
    n_q = len(qualifying_sorted)
    hot_threshold  = qualifying_sorted[int(n_q * 0.25)][1] if n_q > 4 else 0.0
    high_threshold = qualifying_sorted[int(n_q * 0.60)][1] if n_q > 4 else 0.0

    tiers = {
        "hot":    {"label": f"🔥 Hot (top 25%, combined ≥{hot_threshold*100:.1f}%)",   "tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}},
        "high":   {"label": f"⚡ High Confidence (top 60%, combined ≥{high_threshold*100:.1f}%)", "tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}},
        "strong": {"label": "📈 Strong (bottom 40%)",                                  "tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}},
        "below":  {"label": "Below threshold / small field",                           "tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}},
    }

    # Account for small fields separately
    for race_id, rows in race_map.items():
        field_size = len(rows)
        if field_size < 7:
            field_size_dist[field_size] = field_size_dist.get(field_size, 0) + 1

    for race_id, tri_combined, valid, field_size, rows in qualifying_sorted:
        field_size_dist[field_size] = field_size_dist.get(field_size, 0) + 1

        top3 = valid[:3]
        top3_positions = {r.position for r, _ in top3}

        if tri_combined >= hot_threshold:
            tier_key = "hot"
        elif tri_combined >= high_threshold:
            tier_key = "high"
        else:
            tier_key = "strong"

        tri_hit = top3_positions == {1, 2, 3}
        tiers[tier_key]["tri"]["races"] += 1
        if tri_hit:
            tiers[tier_key]["tri"]["hits"] += 1
        totals["tri"]["races"] += 1
        if tri_hit:
            totals["tri"]["hits"] += 1

        if field_size >= 4:
            top4 = valid[:4]
            top4_positions = {r.position for r, _ in top4}
            ff_hit = top4_positions == {1, 2, 3, 4}
            tiers[tier_key]["ff"]["races"] += 1
            if ff_hit:
                tiers[tier_key]["ff"]["hits"] += 1
            totals["ff"]["races"] += 1
            if ff_hit:
                totals["ff"]["hits"] += 1

    def _rate(d):
        return round(d["hits"] / d["races"] * 100, 1) if d["races"] else 0

    tier_results = []
    for key, t in tiers.items():
        tri_r, ff_r = t["tri"]["races"], t["ff"]["races"]
        if tri_r == 0:
            continue
        tier_results.append({
            "tier": t["label"],
            "trifecta": {**t["tri"], "hit_rate_pct": _rate(t["tri"])},
            "first_four": {**t["ff"], "hit_rate_pct": _rate(t["ff"])},
        })

    return {
        "races_analysed": len(race_map),
        "runners_with_data": len(hist_rows),
        "overall": {
            "trifecta": {**totals["tri"], "hit_rate_pct": _rate(totals["tri"])},
            "first_four": {**totals["ff"], "hit_rate_pct": _rate(totals["ff"])},
        },
        "by_tier": tier_results,
        "field_size_distribution": dict(sorted(field_size_dist.items())),
    }


@app.get("/api/performance/by-venue")
async def performance_by_venue(days: int = Query(30, ge=1, le=90)):
    """Per-venue top-pick win rate for the last N days."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()
        if not hr_rows:
            return {"days": days, "venues": []}
        race_ids = list({r.race_id for r in hr_rows})
        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        top_picks = {p.race_id: p for p in pred_result.scalars().all()}

    result_by_key = {(r.race_id, r.horse_name): r for r in hr_rows}
    by_venue: dict[str, dict] = {}
    for race_id, pick in top_picks.items():
        _, venue, _ = _parse_race_id(race_id)
        result = result_by_key.get((race_id, pick.horse_name))
        if not result:
            continue
        if venue not in by_venue:
            by_venue[venue] = {"races": 0, "wins": 0, "placed": 0}
        by_venue[venue]["races"] += 1
        if result.winner:
            by_venue[venue]["wins"] += 1
        if result.position and result.position <= 3:
            by_venue[venue]["placed"] += 1

    venues = sorted([
        {
            "venue": v,
            "races": d["races"],
            "wins": d["wins"],
            "placed": d["placed"],
            "win_rate": round(d["wins"] / d["races"] * 100, 1),
            "place_rate": round(d["placed"] / d["races"] * 100, 1),
        }
        for v, d in by_venue.items() if d["races"] >= 2
    ], key=lambda x: x["win_rate"], reverse=True)

    return {"days": days, "venues": venues}


@app.get("/api/performance/premium")
async def premium_performance(days: int = Query(30, ge=1, le=365), x_cron_secret: Optional[str] = Header(None)):
    """
    P&L analysis for Premium picks: model_pct >= 30%, SP >= $3.00, overlay > 5%.
    Requires admin auth.
    """
    _check_admin(x_cron_secret)
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()

        if not hr_rows:
            return {"days": days, "picks": [], "summary": {"bets": 0, "wins": 0, "win_pct": None, "pnl": 0.0, "roi_pct": None, "avg_sp": None}}

        race_ids = list({r.race_id for r in hr_rows})
        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        top_picks = {p.race_id: p for p in pred_result.scalars().all()}

    result_by_key = {(r.race_id, r.horse_name): r for r in hr_rows}

    picks = []
    for race_id, pick in top_picks.items():
        actual = result_by_key.get((race_id, pick.horse_name))
        if not actual:
            continue
        sp = actual.starting_price or 0.0
        overlay = pick.overlay or 0.0
        model_pct = round((pick.win_probability or 0) * 100, 1)
        if model_pct >= 30 and sp >= 3.0 and overlay > 0.05:
            pnl = (sp - 1.0) if actual.winner else -1.0
            picks.append({
                "date": race_id[:10],
                "race_id": race_id,
                "horse_name": pick.horse_name,
                "model_pct": model_pct,
                "sp": sp,
                "overlay_pct": round(overlay * 100, 1),
                "winner": actual.winner,
                "pnl": round(pnl, 2),
            })

    picks.sort(key=lambda x: x["date"])
    bets = len(picks)
    wins = sum(1 for p in picks if p["winner"])
    total_pnl = sum(p["pnl"] for p in picks)
    sp_list = [p["sp"] for p in picks]

    return {
        "days": days,
        "summary": {
            "bets": bets,
            "wins": wins,
            "win_pct": round(wins / bets * 100, 1) if bets else None,
            "pnl_at_10": round(total_pnl * 10, 2),
            "roi_pct": round(total_pnl / bets * 100, 1) if bets else None,
            "avg_sp": round(sum(sp_list) / len(sp_list), 2) if sp_list else None,
        },
        "picks": picks,
    }


@app.get("/api/performance/premium/public")
async def premium_performance_public():
    """Public summary of rolling 30-day Premium pick performance (no auth required)."""
    days = 30
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()
        if not hr_rows:
            return {"days": days, "bets": 0, "wins": 0, "win_pct": None, "pnl_at_10": 0.0, "roi_pct": None, "avg_sp": None}

        race_ids = list({r.race_id for r in hr_rows})
        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        top_picks = {p.race_id: p for p in pred_result.scalars().all()}

    result_by_key = {(r.race_id, r.horse_name): r for r in hr_rows}
    bets = wins = 0
    total_pnl = 0.0
    sp_list = []
    for race_id, pick in top_picks.items():
        actual = result_by_key.get((race_id, pick.horse_name))
        if not actual:
            continue
        sp = actual.starting_price or 0.0
        overlay = pick.overlay or 0.0
        model_pct = (pick.win_probability or 0) * 100
        if model_pct >= 30 and sp >= 3.0 and overlay > 0.05:
            bets += 1
            if actual.winner:
                wins += 1
                total_pnl += sp - 1.0
            else:
                total_pnl -= 1.0
            sp_list.append(sp)

    return {
        "days": days,
        "bets": bets,
        "wins": wins,
        "win_pct": round(wins / bets * 100, 1) if bets else None,
        "pnl_at_10": round(total_pnl * 10, 2),
        "roi_pct": round(total_pnl / bets * 100, 1) if bets else None,
        "avg_sp": round(sum(sp_list) / len(sp_list), 2) if sp_list else None,
    }


@app.get("/api/performance/premium/monthly")
async def premium_performance_monthly():
    """Public monthly breakdown of Premium pick P&L for last 6 months inc MTD (no auth required)."""
    today = date.today()
    # First day of the month 5 months ago (gives current month + 5 prior = 6 total)
    first_of_current = today.replace(day=1)
    cutoff_month = first_of_current
    for _ in range(5):
        cutoff_month = (cutoff_month - timedelta(days=1)).replace(day=1)
    cutoff = cutoff_month.isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()
        if not hr_rows:
            return {"months": []}

        race_ids = list({r.race_id for r in hr_rows})
        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        top_picks = {p.race_id: p for p in pred_result.scalars().all()}

    result_by_key = {(r.race_id, r.horse_name): r for r in hr_rows}
    monthly: dict[str, dict] = {}

    for race_id, pick in top_picks.items():
        actual = result_by_key.get((race_id, pick.horse_name))
        if not actual:
            continue
        sp = actual.starting_price or 0.0
        overlay = pick.overlay or 0.0
        model_pct = (pick.win_probability or 0) * 100
        if model_pct >= 30 and sp >= 3.0 and overlay > 0.05:
            month = race_id[:7]  # YYYY-MM
            if month not in monthly:
                monthly[month] = {"bets": 0, "wins": 0, "pnl": 0.0}
            monthly[month]["bets"] += 1
            if actual.winner:
                monthly[month]["wins"] += 1
                monthly[month]["pnl"] += sp - 1.0
            else:
                monthly[month]["pnl"] -= 1.0

    months_out = []
    for month in sorted(monthly.keys()):
        m = monthly[month]
        bets, wins, pnl = m["bets"], m["wins"], m["pnl"]
        spent = bets * 10
        returned = round(spent + pnl * 10, 2)
        months_out.append({
            "month": month,
            "bets": bets,
            "wins": wins,
            "win_pct": round(wins / bets * 100, 1) if bets else None,
            "spent": spent,
            "returned": returned,
            "pnl_at_10": round(pnl * 10, 2),
            "roi_pct": round(pnl / bets * 100, 1) if bets else None,
        })

    return {"months": months_out}


@app.get("/api/performance/premium/daily")
async def premium_performance_daily():
    """Public daily breakdown of Premium pick P&L for last 5 days (no auth required)."""
    today = _today_aest()
    cutoff = (today - timedelta(days=4)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()
        if not hr_rows:
            return {"days": []}

        race_ids = list({r.race_id for r in hr_rows})
        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        top_picks = {p.race_id: p for p in pred_result.scalars().all()}

    result_by_key = {(r.race_id, r.horse_name): r for r in hr_rows}
    daily: dict[str, dict] = {}

    for race_id, pick in top_picks.items():
        actual = result_by_key.get((race_id, pick.horse_name))
        if not actual:
            continue
        sp = actual.starting_price or 0.0
        overlay = pick.overlay or 0.0
        model_pct = (pick.win_probability or 0) * 100
        if model_pct >= 30 and sp >= 3.0 and overlay > 0.05:
            day = race_id[:10]
            if day not in daily:
                daily[day] = {"bets": 0, "wins": 0, "pnl": 0.0}
            daily[day]["bets"] += 1
            if actual.winner:
                daily[day]["wins"] += 1
                daily[day]["pnl"] += sp - 1.0
            else:
                daily[day]["pnl"] -= 1.0

    # Ensure all 5 days are present even if no picks
    days_out = []
    for i in range(4, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        d = daily.get(day, {"bets": 0, "wins": 0, "pnl": 0.0})
        bets, wins, pnl = d["bets"], d["wins"], d["pnl"]
        days_out.append({
            "date": day,
            "bets": bets,
            "wins": wins,
            "win_pct": round(wins / bets * 100, 1) if bets else None,
            "spent": bets * 10,
            "returned": round(bets * 10 + pnl * 10, 2),
            "pnl_at_10": round(pnl * 10, 2),
            "roi_pct": round(pnl / bets * 100, 1) if bets else None,
        })

    return {"days": days_out}


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
    _check_admin(x_cron_secret)
    if _calibration_status.get("running"):
        raise HTTPException(409, "Calibration already running")
    asyncio.create_task(_run_calibration_task(holdout_days))
    return {"status": "started", "holdout_days": holdout_days,
            "message": "Check /api/admin/calibrate/status for progress"}


@app.get("/api/admin/calibrate/status")
async def calibration_task_status(x_cron_secret: Optional[str] = Header(None)):
    """Current calibration task progress and result when done."""
    _check_admin(x_cron_secret)
    return _calibration_status


@app.get("/api/admin/calibration/history")
async def calibration_history(
    limit: int = Query(10, ge=1, le=50),
    x_cron_secret: Optional[str] = Header(None),
):
    """Return the last N calibration runs with drift history."""
    _check_admin(x_cron_secret)
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

async def _load_venue_calibration() -> dict[str, float]:
    """Compute venue win-rate multipliers from last 60 days of historical results."""
    cutoff = (date.today() - timedelta(days=60)).isoformat()
    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()
        if not hr_rows:
            return {}
        race_ids = list({r.race_id for r in hr_rows})
        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        top_picks = {p.race_id: p for p in pred_result.scalars().all()}

    result_by_key = {(r.race_id, r.horse_name): r for r in hr_rows}
    venue_stats: dict[str, dict] = {}
    for race_id, pick in top_picks.items():
        _, venue, _ = _parse_race_id(race_id)
        result = result_by_key.get((race_id, pick.horse_name))
        if not result:
            continue
        if venue not in venue_stats:
            venue_stats[venue] = {"races": 0, "wins": 0}
        venue_stats[venue]["races"] += 1
        if result.winner:
            venue_stats[venue]["wins"] += 1

    multipliers = compute_venue_multipliers(venue_stats)
    log.info("[venue_calibration] %d venues calibrated", len(multipliers))
    return multipliers


async def _enrich_date(race_date: str, client, model, force: bool = False, place_model: PlaceModel | None = None, exotic_model: ExoticModel | None = None) -> list[dict]:
    """Enrich all meetings for a single date. Returns summary list."""
    venue_cal = await _load_venue_calibration()
    if place_model is None:
        async with get_session() as session:
            place_model = await _load_place_model(session)
    if exotic_model is None:
        async with get_session() as session:
            exotic_model = await _load_exotic_model(session)
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
                # Skip if already enriched (unless forced)
                if not force:
                    async with get_session() as session:
                        already = await session.execute(
                            select(RunnerPredictionRow)
                            .where(RunnerPredictionRow.race_id == race_id)
                            .limit(1)
                        )
                        if already.scalars().first():
                            continue
                full_event = await client.get_race(slug, race_num)
                if not full_event:
                    continue
                race = await client.parse_race(full_event, race_date, venue_name, state)
                predictions, _ = await enrich_and_predict_race(race, model, venue_calibration=venue_cal, place_model=place_model, exotic_model=exotic_model)
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
    _check_admin(x_cron_secret)

    client = get_tab_client()
    async with get_session() as session:
        model = await _load_model(session)

    results = {}
    for i in range(days):
        race_date = (date.today() + timedelta(days=i)).isoformat()
        log.info("[cron] Enriching %s", race_date)
        results[race_date] = await _enrich_date(race_date, client, model)

    return {"dates": results}


@app.post("/api/admin/reenrich")
async def admin_reenrich(
    x_cron_secret: Optional[str] = Header(None),
):
    """Force re-enrich all of today's races (fire-and-forget)."""
    _check_admin(x_cron_secret)
    today = _today_aest().isoformat()

    async def _do_reenrich():
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)
        await _enrich_date(today, client, model, force=True)
        log.info("[reenrich] Completed re-enrich for %s", today)

    asyncio.create_task(_do_reenrich())
    return {"status": "reenrich_started", "date": today}


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
        "place_model_rank": pred.place_model_rank if pred.place_model_rank else None,
        "exotic_model_rank": pred.exotic_model_rank if pred.exotic_model_rank else None,
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
        "place_model_rank": row.place_model_rank,
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
        "win_rate_distance": enriched.get("win_rate_distance"),
        "class_change": enriched.get("class_change"),
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
