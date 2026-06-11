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

Data sources:
  Racing Australia — meeting / race-card / results feed (primary)
  OddsPro          — multi-book live odds + steam/drift inputs
  Betfair          — optional REST metadata + live streaming odds (when credentials set)
  TAB direct API   — fallback odds lookup for picks not covered by OddsPro

Venue codes are lowercase venue slugs, e.g. "werribee", "randwick", "flemington".
Meeting slugs follow the pattern "{venue}-{date}" e.g. "werribee-20260514".
Race IDs follow the pattern "{date}_{venue}_R{num}" e.g. "2026-05-14_werribee_R3".
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
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
    ExoticBacktestRow,
    HistoricalResultRow,
    OddsSnapshotRow,
    RunnerPredictionRow,
    RunnerPredictionHistoryRow,
    RacePredictionRow,
    init_db,
    backfill_prediction_history,
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
    delay = random.uniform(0, 600)
    log.info("[odds-snapshot] Waiting %.0fs before RA requests", delay)
    await asyncio.sleep(delay)
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

                    snapped_race_ids: set[str] = set()
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
                            snapped_race_ids.add(pick.race_id)
                        await session.commit()
                    log.info("[odds-snapshot] Snapped %d runners for %s", len(horse_data), slug)
                    # Recompute steam features from full snapshot history for these races
                    asyncio.create_task(_update_steam_features(list(snapped_race_ids)))
                except Exception as e:
                    log.warning("[odds-snapshot] Failed for %s: %s", slug, e)
    except Exception as e:
        log.exception("[odds-snapshot] Snapshot failed: %s", e)


def _compute_steam_from_snapshots(snaps: list) -> dict:
    """
    Compute steam/drift features from a horse's snapshot history.
    snaps: list of OddsSnapshotRow, any order, pre-race only (minutes_to_jump > 0).
    Returns dict matching EnrichedRunner steam field names.
    """
    pre = [s for s in snaps if s.minutes_to_jump is not None and s.minutes_to_jump > 0 and s.win_odds]
    if len(pre) < 2:
        return {"steam_60": 0.0, "steam_30": 0.0, "drift_flag": 0.0, "odds_velocity": 0.0, "late_money": 0.0}

    pre.sort(key=lambda s: s.minutes_to_jump)  # ascending: smallest mtj = closest to jump

    def _closest(target_mtj: int):
        return min(pre, key=lambda s: abs(s.minutes_to_jump - target_mtj)).win_odds

    odds_t5  = _closest(5)
    odds_t15 = _closest(15) if any(s.minutes_to_jump >= 10 for s in pre) else None
    odds_t30 = _closest(30) if any(s.minutes_to_jump >= 25 for s in pre) else None
    odds_t60 = _closest(60) if any(s.minutes_to_jump >= 50 for s in pre) else None
    odds_open = pre[-1].win_odds  # highest minutes_to_jump = earliest/opening snapshot

    steam_60 = round(odds_t60 - odds_t5, 2) if odds_t60 else 0.0   # +ve = shortened
    steam_30 = round(odds_t30 - odds_t5, 2) if odds_t30 else 0.0
    late_money = round(odds_t15 - odds_t5, 2) if odds_t15 else 0.0
    drift_flag = 1.0 if (odds_t5 - odds_open) > 1.5 else 0.0       # drifted out $1.50+

    span = pre[-1].minutes_to_jump - pre[0].minutes_to_jump
    if span > 0 and odds_t60:
        odds_velocity = round((odds_t60 - odds_t5) / span, 4)
    else:
        odds_velocity = 0.0

    return {
        "steam_60": steam_60,
        "steam_30": steam_30,
        "drift_flag": drift_flag,
        "odds_velocity": odds_velocity,
        "late_money": late_money,
    }


async def _update_steam_features(race_ids: list[str]) -> None:
    """
    For each race_id, load all snapshots, compute steam features per horse,
    and patch the steam fields into enriched_json on runner_predictions rows.
    Called after each odds snapshot run.
    """
    if not race_ids:
        return
    from sqlalchemy import update as sa_update

    async with get_session() as session:
        snap_result = await session.execute(
            select(OddsSnapshotRow)
            .where(OddsSnapshotRow.race_id.in_(race_ids))
        )
        all_snaps = snap_result.scalars().all()

        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.enriched_json.isnot(None))
        )
        pred_rows = pred_result.scalars().all()

    # Group snapshots by (race_id, horse_name)
    snap_map: dict[tuple, list] = {}
    for s in all_snaps:
        snap_map.setdefault((s.race_id, s.horse_name), []).append(s)

    updated = 0
    async with get_session() as session:
        for pred in pred_rows:
            key = (pred.race_id, pred.horse_name)
            horse_snaps = snap_map.get(key, [])
            if not horse_snaps:
                continue
            steam = _compute_steam_from_snapshots(horse_snaps)
            try:
                er_data = json.loads(pred.enriched_json)
                er_data.update(steam)
                pred.enriched_json = json.dumps(er_data)
                session.add(pred)
                updated += 1
            except Exception:
                continue
        await session.commit()

    if updated:
        log.info("[steam-update] Updated steam features for %d runners across %d races", updated, len(race_ids))


async def _cancel_abandoned_meetings(
    client,
    today: str,
    meetings: list | None = None,
    venue_events: dict[str, list] | None = None,
) -> None:
    """
    Compare today's DB predictions against Racing Australia's live meeting list.
    Mark runners cancelled for any venue that has been dropped or where
    all races are abandoned/closed without having resulted.

    Pass pre-fetched ``meetings`` and ``venue_events`` to skip redundant API calls
    when called from a combined job that has already fetched this data.
    """
    from sqlalchemy import update as sa_update
    date_sfx = f"-{today.replace('-', '')}"

    if meetings is None:
        try:
            meetings = await asyncio.wait_for(client.get_meetings(today), timeout=20)
        except Exception as e:
            log.warning("[cancel-check] Could not fetch meetings: %s", e)
            return

    # Guard: if RA returns nothing, it's likely blocked — never mass-cancel on empty response
    if not meetings:
        log.warning("[cancel-check] RA returned 0 meetings for %s — skipping to avoid false cancellations", today)
        return

    active_venue_codes: set[str] = set()
    ra_race_statuses: dict[str, set[str]] = {}
    for m in meetings:
        slug = m.get("slug", "")
        vc = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug
        active_venue_codes.add(vc)
        if venue_events is not None and vc in venue_events:
            ra_race_statuses[vc] = {(e.get("status") or "").lower() for e in venue_events[vc]}
        else:
            try:
                raw_events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=15)
                ra_race_statuses[vc] = {(e.get("status") or "").lower() for e in raw_events}
            except Exception:
                pass

    async with get_session() as session:
        db_race_ids = (await session.execute(
            select(RunnerPredictionRow.race_id)
            .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
            .where(RunnerPredictionRow.model_rank == 1)
            .distinct()
        )).scalars().all()

    db_venue_codes = {_parse_race_id(rid)[1] for rid in db_race_ids}
    ra_count = len(active_venue_codes)
    db_count = len(db_venue_codes)
    # Only trust "dropped" signal if RA returned >= 80% of known DB venues.
    # A partial RA response (e.g. 2 of 5 venues when blocked) must NOT cancel the rest.
    trust_drop_signal = ra_count >= max(1, round(db_count * 0.8))
    if not trust_drop_signal:
        log.warning(
            "[cancel-check] RA returned %d venues but DB has %d — partial response, ignoring 'dropped' signal",
            ra_count, db_count,
        )

    for race_id in db_race_ids:
        _, venue_code, _ = _parse_race_id(race_id)
        statuses = ra_race_statuses.get(venue_code, set())
        dropped = trust_drop_signal and (venue_code not in active_venue_codes)
        all_abandoned = bool(statuses) and statuses.issubset({"abandoned", "cancelled", "closed"}) and "open" not in statuses and "resulted" not in statuses

        if not dropped and not all_abandoned:
            # Only restore if EVERY runner in the race was cancelled — that's a venue-level
            # block from a prior run. Leave individually-cancelled runners (manual scratches) alone.
            # The same guard is applied separately to mutable and history so an individual
            # scratch in either table prevents a mass-uncancel from clobbering it.
            restored = False
            async with get_session() as session:
                total = (await session.execute(
                    select(func.count()).where(RunnerPredictionRow.race_id == race_id)
                )).scalar_one()
                n_cancelled = (await session.execute(
                    select(func.count())
                    .where(RunnerPredictionRow.race_id == race_id)
                    .where(RunnerPredictionRow.cancelled.is_(True))
                )).scalar_one()
                if total > 0 and n_cancelled == total:
                    await session.execute(
                        sa_update(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == race_id)
                        .where(RunnerPredictionRow.cancelled.is_(True))
                        .values(cancelled=False)
                    )
                    restored = True
                # Same all-cancelled guard for history. Without this, a mass-cancel that
                # was previously mirrored into history (FIX-J / BUG-14) stays sticky in
                # history even after mutable is restored.
                hist_total = (await session.execute(
                    select(func.count()).where(RunnerPredictionHistoryRow.race_id == race_id)
                )).scalar_one()
                hist_n_cancelled = (await session.execute(
                    select(func.count())
                    .where(RunnerPredictionHistoryRow.race_id == race_id)
                    .where(RunnerPredictionHistoryRow.cancelled.is_(True))
                )).scalar_one()
                if hist_total > 0 and hist_n_cancelled == hist_total:
                    await session.execute(
                        sa_update(RunnerPredictionHistoryRow)
                        .where(RunnerPredictionHistoryRow.race_id == race_id)
                        .where(RunnerPredictionHistoryRow.cancelled.is_(True))
                        .values(cancelled=False)
                    )
                    restored = True
                await session.commit()
            if restored:
                # Cancellation state changed — drop cached entries for this venue
                # so the restoration is visible on the next request (BUG-30).
                _invalidate_meeting_caches(today, venue_code)
        elif dropped or all_abandoned:
            async with get_session() as session:
                await session.execute(
                    sa_update(RunnerPredictionRow)
                    .where(RunnerPredictionRow.race_id == race_id)
                    .values(cancelled=True)
                )
                # Mirror mass-cancel into history so settled-race readers (which filter
                # on history.cancelled per BUG-13) also exclude the abandoned race.
                await session.execute(
                    sa_update(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id == race_id)
                    .values(cancelled=True)
                )
                await session.commit()
            reason = "venue dropped" if dropped else "all races abandoned/closed"
            log.info("[cancel-check] Marked %s CANCELLED (%s)", race_id, reason)
            # State changed — drop cached list + per-venue detail so the next
            # request reflects the cancellation (BUG-30).
            _invalidate_meeting_caches(today, venue_code)


async def _scheduled_enrich():
    """Run by APScheduler — enrich today + next 2 days, then seed today's results."""
    log.info("[scheduler] Running scheduled enrichment")
    try:
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)

        # Pre-warm jockey/trainer cache for today so parse_race() fires zero extra GQL calls
        today = _today_aest().isoformat()
        if hasattr(client, "prefetch_people_for_date"):
            log.info("[scheduler] Pre-fetching jockey/trainer stats for %s", today)
            await client.prefetch_people_for_date(today)

        for i in range(3):
            race_date = (_today_aest() + timedelta(days=i)).isoformat()
            log.info("[scheduler] Enriching %s (force=%s)", race_date, i == 0)
            # Force today's races to always re-enrich so stale pre-enrichments
            # (from days-out bulk runs with no market data) get replaced with
            # fresh odds and features each morning.
            await _enrich_date(race_date, client, model, force=(i == 0))
        # Check for abandoned meetings after enrichment
        await _cancel_abandoned_meetings(client, _today_aest().isoformat())
        # Snapshot BEFORE seeding results — captures predictions before result
        # data would be visible. No intraday retrain so predictions are clean.
        n_snap = await _snapshot_prerace_predictions()
        if n_snap:
            log.info("[scheduler] Snapshotted %d races into history after enrichment", n_snap)
        # Seed yesterday + today after snapshot
        for offset in (-1, 0):
            seed_date = (_today_aest() + timedelta(days=offset)).isoformat()
            n = await _seed_results_for_date(seed_date)
            if n:
                log.info("[scheduler] Seeded %d results for %s", n, seed_date)
        log.info("[scheduler] Enrichment complete")
    except Exception as e:
        log.exception("[scheduler] Enrichment failed: %s", e)


async def _scheduled_pre_race_enrich():
    """Re-enrich any race starting within the next 2 hours. Runs every 15 min during racing hours."""
    from sqlalchemy import update as sa_update
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

        # Check for abandoned meetings (dropped venues or all races closed)
        await _cancel_abandoned_meetings(client, today)

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
                    async with get_session() as session:
                        await _inject_accumulated_stats(race, session)
                    predictions, _ = await enrich_and_predict_race(race, model, place_model=place_model)
                    async with get_session() as session:
                        await save_race_predictions(
                            session,
                            race_id,
                            [_prediction_to_db_dict(p, race_id, start_raw, race=race) for p in predictions],
                        )
                    log.info("[pre-race] Re-enriched %s (jump %s)", race_id, start_raw)
                    enriched_count += 1
            except Exception as e:
                log.warning("[pre-race] Failed for %s: %s", slug, e)
        if enriched_count:
            log.info("[pre-race] Re-enriched %d races", enriched_count)
    except Exception as e:
        log.exception("[pre-race] Pre-race enrich failed: %s", e)


async def _check_scratches_today() -> int:
    """
    Lightweight scratch detection — no ML inference.
    Checks races starting within the next 4 hours (wider than the 2-hour pre-race
    enrich window so scratches are caught before enrichment runs).
    Returns count of newly cancelled runners.
    """
    from sqlalchemy import update as sa_update
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    horizon = now_utc + timedelta(hours=4)
    today = _today_aest().isoformat()
    date_sfx = f"-{today.replace('-', '')}"
    total_cancelled = 0

    try:
        client = get_tab_client()
        # Clear RA meeting cache so we always see fresh runner statuses
        if hasattr(client, "_ra") and hasattr(client._ra, "_meeting_cache"):
            client._ra._meeting_cache.clear()

        meetings = await client.get_meetings(today)

        for m in meetings:
            slug = m.get("slug", "")
            if not slug:
                continue
            venue_code = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0]
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
                    # Only check races that haven't jumped yet (or jumped <30 min ago)
                    if jump > horizon or jump < now_utc - timedelta(minutes=30):
                        continue
                    race_num = raw_event.get("eventNumber")
                    race_id = f"{today}_{venue_code}_R{race_num}"

                    full_event = await client.get_race(slug, race_num)
                    if not full_event:
                        continue

                    # RA returns status="" for all runners (active and scratched alike),
                    # so status-based detection never fires. Instead: compare the current
                    # RA field against DB predictions — any runner absent from RA is scratched.
                    current_field = {
                        (sel.get("competitor") or {}).get("name", "").strip()
                        for sel in full_event.get("selections", [])
                        if (sel.get("competitor") or {}).get("name", "").strip()
                    }
                    if not current_field:
                        continue  # empty field = bad data, skip

                    async with get_session() as session:
                        db_runners = (await session.execute(
                            select(RunnerPredictionRow.horse_name)
                            .where(RunnerPredictionRow.race_id == race_id)
                            .where(
                                RunnerPredictionRow.cancelled.is_(False)
                                | RunnerPredictionRow.cancelled.is_(None)
                            )
                        )).scalars().all()

                        scratched_names = {n for n in db_runners if n not in current_field}
                        if not scratched_names:
                            continue

                        result = await session.execute(
                            sa_update(RunnerPredictionRow)
                            .where(RunnerPredictionRow.race_id == race_id)
                            .where(RunnerPredictionRow.horse_name.in_(scratched_names))
                            .values(cancelled=True)
                        )
                        # Mirror scratch into history so settled-race edge picks also exclude them
                        await session.execute(
                            sa_update(RunnerPredictionHistoryRow)
                            .where(RunnerPredictionHistoryRow.race_id == race_id)
                            .where(RunnerPredictionHistoryRow.horse_name.in_(scratched_names))
                            .values(cancelled=True)
                        )
                        if result.rowcount:
                            await session.commit()
                            log.info("[scratch-check] %s: cancelled %d runner(s): %s",
                                     race_id, result.rowcount, scratched_names)
                            total_cancelled += result.rowcount
            except Exception as e:
                log.debug("[scratch-check] Failed for %s: %s", slug, e)
    except Exception as e:
        log.exception("[scratch-check] Failed: %s", e)

    # Sync step: catch any mutable-cancelled runners whose history row predates this fix.
    # Runs every call so retroactive scratches (cancelled before this code was deployed) propagate.
    try:
        async with get_session() as session:
            already_cancelled_mut = (await session.execute(
                select(RunnerPredictionRow.race_id, RunnerPredictionRow.horse_name)
                .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
                .where(RunnerPredictionRow.cancelled.is_(True))
            )).fetchall()
            if already_cancelled_mut:
                for race_id, horse_name in already_cancelled_mut:
                    await session.execute(
                        sa_update(RunnerPredictionHistoryRow)
                        .where(RunnerPredictionHistoryRow.race_id == race_id)
                        .where(RunnerPredictionHistoryRow.horse_name == horse_name)
                        .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                        .values(cancelled=True)
                    )
                await session.commit()
    except Exception as e:
        log.warning("[scratch-check] History sync failed: %s", e)

    return total_cancelled


async def _scheduled_check_scratches():
    try:
        n = await _check_scratches_today()
        if n:
            log.info("[scheduler] Scratch check: cancelled %d runner(s)", n)
    except Exception as e:
        log.exception("[scheduler] Scratch check failed: %s", e)


async def _scheduled_pre_race_enrich_and_scratch():
    """
    Combined 15-min job: re-enrich races within 2h, detect scratches within 4h,
    and check for abandoned meetings — all from a single get_meetings +
    get_meeting_races pass. Eliminates ~2,100 redundant API calls/day vs running
    these as separate crons.
    """
    from sqlalchemy import update as sa_update
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    enrich_horizon = now_utc + timedelta(hours=2)
    scratch_horizon = now_utc + timedelta(hours=4)
    today = _today_aest().isoformat()
    date_sfx = f"-{today.replace('-', '')}"
    log.info("[pre-race] Combined enrich+scratch scan starting")

    try:
        client = get_tab_client()

        # Clear RA meeting cache so scratch detection sees fresh runner lists
        if hasattr(client, "_ra") and hasattr(client._ra, "_meeting_cache"):
            client._ra._meeting_cache.clear()

        async with get_session() as session:
            model = await _load_model(session)
            place_model = await _load_place_model(session)

        meetings = await client.get_meetings(today)
        if not meetings:
            log.warning("[pre-race] No meetings returned for %s", today)
            return

        # Single get_meeting_races pass per meeting — shared by enrich, scratch, and cancel-check
        meeting_meta: dict[str, tuple] = {}   # vc -> (slug, venue_name, state)
        venue_raw_events: dict[str, list] = {}  # vc -> raw_events
        for m in meetings:
            slug = m.get("slug", "")
            if not slug:
                continue
            vc = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0]
            meeting_meta[vc] = (slug, m.get("venue", vc), m.get("state", ""))
            try:
                venue_raw_events[vc] = await client.get_meeting_races(slug)
            except Exception as e:
                log.debug("[pre-race] get_meeting_races failed for %s: %s", slug, e)

        # Cancel-check using pre-fetched data — no extra API calls
        await _cancel_abandoned_meetings(client, today, meetings=meetings, venue_events=venue_raw_events)

        enriched_count = 0
        scratch_count = 0

        for vc, raw_events in venue_raw_events.items():
            slug, venue_name, state = meeting_meta[vc]
            for raw_event in raw_events:
                start_raw = raw_event.get("startTime")
                if not start_raw:
                    continue
                try:
                    jump = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                except ValueError:
                    continue

                in_enrich = now_utc <= jump <= enrich_horizon
                in_scratch = now_utc - timedelta(minutes=30) <= jump <= scratch_horizon

                if not in_enrich and not in_scratch:
                    continue

                race_num = raw_event.get("eventNumber")
                race_id = f"{today}_{vc}_R{race_num}"

                try:
                    full_event = await client.get_race(slug, race_num)
                except Exception as e:
                    log.warning("[pre-race] get_race failed for %s: %s", race_id, e)
                    continue
                if not full_event:
                    continue

                if in_enrich:
                    try:
                        race = await client.parse_race(full_event, today, venue_name, state)
                        async with get_session() as session:
                            await _inject_accumulated_stats(race, session)
                        predictions, _ = await enrich_and_predict_race(race, model, place_model=place_model)
                        async with get_session() as session:
                            await save_race_predictions(
                                session,
                                race_id,
                                [_prediction_to_db_dict(p, race_id, start_raw, race=race) for p in predictions],
                            )
                        log.info("[pre-race] Re-enriched %s (jump %s)", race_id, start_raw)
                        enriched_count += 1
                    except Exception as e:
                        log.warning("[pre-race] Enrich failed for %s: %s", race_id, e)

                if in_scratch:
                    try:
                        current_field = {
                            (sel.get("competitor") or {}).get("name", "").strip()
                            for sel in full_event.get("selections", [])
                            if (sel.get("competitor") or {}).get("name", "").strip()
                        }
                        if not current_field:
                            continue
                        async with get_session() as session:
                            # Compare current RA field against the union of uncancelled
                            # rows in both mutable and history. This catches horses that
                            # only live in history (e.g. snapshotted at 9am but already
                            # removed from mutable by a subsequent re-enrichment) as well
                            # as horses still present in mutable.
                            mut_runners = (await session.execute(
                                select(RunnerPredictionRow.horse_name)
                                .where(RunnerPredictionRow.race_id == race_id)
                                .where(
                                    RunnerPredictionRow.cancelled.is_(False)
                                    | RunnerPredictionRow.cancelled.is_(None)
                                )
                            )).scalars().all()
                            hist_runners = (await session.execute(
                                select(RunnerPredictionHistoryRow.horse_name)
                                .where(RunnerPredictionHistoryRow.race_id == race_id)
                                .where(
                                    RunnerPredictionHistoryRow.cancelled.is_(False)
                                    | RunnerPredictionHistoryRow.cancelled.is_(None)
                                )
                            )).scalars().all()
                            scratched_names = {
                                n for n in set(mut_runners) | set(hist_runners)
                                if n not in current_field
                            }
                            if scratched_names:
                                mut_result = await session.execute(
                                    sa_update(RunnerPredictionRow)
                                    .where(RunnerPredictionRow.race_id == race_id)
                                    .where(RunnerPredictionRow.horse_name.in_(scratched_names))
                                    .values(cancelled=True)
                                )
                                # Mirror into history so downstream readers (which filter on
                                # history.cancelled per BUG-13) actually exclude scratches.
                                hist_result = await session.execute(
                                    sa_update(RunnerPredictionHistoryRow)
                                    .where(RunnerPredictionHistoryRow.race_id == race_id)
                                    .where(RunnerPredictionHistoryRow.horse_name.in_(scratched_names))
                                    .values(cancelled=True)
                                )
                                if mut_result.rowcount or hist_result.rowcount:
                                    await session.commit()
                                    log.info(
                                        "[scratch-check] %s: cancelled %d mutable / %d history row(s): %s",
                                        race_id, mut_result.rowcount, hist_result.rowcount, scratched_names,
                                    )
                                    scratch_count += len(scratched_names)
                                    # Drop the per-venue meeting cache so the
                                    # scratch is visible on the next request
                                    # (BUG-30). _list_meetings_cache is also
                                    # dropped — cheap insurance for the rare
                                    # case where a scratch causes a venue to
                                    # fall off the DB-merged extension list.
                                    _invalidate_meeting_caches(today, vc)
                    except Exception as e:
                        log.debug("[scratch-check] Failed for %s: %s", race_id, e)

        if enriched_count:
            log.info("[pre-race] Re-enriched %d races", enriched_count)
        if scratch_count:
            log.info("[scratch-check] Cancelled %d runner(s) total", scratch_count)

        # Snapshot any newly-enriched races that don't yet have a history row.
        # Covers WA venues (Belmont etc.) that aren't listed at 9am AEST — their
        # first enrichment happens here, so we snapshot immediately rather than
        # waiting for the next _scheduled_enrich (10am/1pm) by which point they
        # may already be mid-race and blocked by the time guard.
        if enriched_count:
            try:
                n_snap = await _snapshot_prerace_predictions()
                if n_snap:
                    log.info("[pre-race] Snapshotted %d runner(s) after combined enrich", n_snap)
            except Exception as snap_err:
                log.warning("[pre-race] Snapshot after enrich failed: %s", snap_err)

    except Exception as e:
        log.exception("[pre-race] Combined enrich+scratch job failed: %s", e)


async def _scheduled_live_odds_refresh():
    """
    Refresh OddsPro odds for all runners in races starting within the next 3 hours.
    Runs every 20 min. Self-limits when no upcoming races are in the window.
    Updates best_available_odds, overlay, value_rating, market_rank only — no re-enrichment.
    """
    from horse_engine.clients.oddspro import OddsProClient

    now_utc = datetime.utcnow()
    now_aest = datetime.now(_AEST)
    today = now_aest.date().isoformat()
    window_end = now_utc + timedelta(hours=3)

    try:
        async with get_session() as session:
            result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
                .where(RunnerPredictionRow.scheduled_time.isnot(None))
            )
            all_rows = result.scalars().all()

        upcoming = []
        for row in all_rows:
            try:
                sched = datetime.fromisoformat(row.scheduled_time.replace("Z", "+00:00")).replace(tzinfo=None)
                if now_utc <= sched <= window_end:
                    upcoming.append(row)
            except Exception:
                continue

        if not upcoming:
            log.debug("[live-odds] No races within 3-hour window, skipping")
            return

        log.info("[live-odds] %d runners in next 3 hours across %d races",
                 len(upcoming), len({r.race_id for r in upcoming}))

        op = OddsProClient()
        tracks = await op.get_tracks(today)

        by_venue: dict[str, list[RunnerPredictionRow]] = {}
        for row in upcoming:
            venue_key = row.venue or _parse_race_id(row.race_id)[1]
            if venue_key:
                by_venue.setdefault(venue_key, []).append(row)

        total_updated = 0
        for venue, rows in by_venue.items():
            op_track = op.find_matching_track(venue, tracks)
            if not op_track:
                log.debug("[live-odds] No OddsPro track match for '%s'", venue)
                continue

            odds_map = await op.get_track_odds(op_track)
            if not odds_map:
                log.debug("[live-odds] Empty odds for track '%s'", op_track)
                continue

            by_race: dict[int, list[RunnerPredictionRow]] = {}
            for row in rows:
                if row.race_number:
                    by_race.setdefault(row.race_number, []).append(row)

            async with get_session() as session:
                for race_num, race_rows in by_race.items():
                    new_odds: dict[int, float] = {}
                    steam_map: dict[int, dict] = {}  # row.id → steam features
                    for row in race_rows:
                        name_lower = row.horse_name.lower()
                        norm_name = _normalize_horse(row.horse_name)
                        op_runner = odds_map.get((race_num, name_lower)) or odds_map.get((race_num, norm_name))
                        if not op_runner:
                            continue
                        raw = op_runner.get("currentBestOdds")
                        try:
                            val = float(raw) if raw else 0.0
                            if val > 1.0:
                                new_odds[row.id] = val
                        except (TypeError, ValueError):
                            pass
                        # Compute steam/drift from firstPrice vs currentBestOdds
                        try:
                            first = float(op_runner.get("firstPrice") or 0)
                            current = float(op_runner.get("currentBestOdds") or 0)
                            pct = float(op_runner.get("movementPercentage") or 0)
                            if first > 1.0 and current > 1.0:
                                price_move = round(first - current, 2)  # +ve = steamed in
                                drift = 1.0 if (current - first) > 1.5 else 0.0
                                # odds_velocity: positive = steaming, negative = drifting
                                ov = round(pct / 100 * (1 if price_move >= 0 else -1), 4)
                                steam_map[row.id] = {
                                    "steam_60": price_move,
                                    "steam_30": price_move,
                                    "late_money": price_move,
                                    "drift_flag": drift,
                                    "odds_velocity": ov,
                                }
                        except (TypeError, ValueError):
                            pass

                    if not new_odds and not steam_map:
                        continue

                    # Recompute market_rank across the FULL active field, not just the
                    # OddsPro-updated runners (BUG-20). Use new_odds where we have it,
                    # falling back to the row's existing best_available_odds, so the
                    # ranking is coherent even when OddsPro only covers part of the
                    # field. Cancelled runners are excluded from the ranking.
                    final_odds: dict[int, float] = {}
                    for row in race_rows:
                        if row.cancelled:
                            continue
                        o = new_odds.get(row.id) or row.best_available_odds or 0
                        if o and o > 1.0:
                            final_odds[row.id] = o
                    sorted_ids = sorted(final_odds, key=lambda rid: final_odds[rid])
                    rank_map = {rid: i + 1 for i, rid in enumerate(sorted_ids)}

                    for row in race_rows:
                        db_row = await session.get(RunnerPredictionRow, row.id)
                        if not db_row:
                            continue
                        new_o = new_odds.get(row.id)
                        if new_o:
                            db_row.best_available_odds = new_o
                            market_implied = 1.0 / new_o
                            db_row.overlay = round(db_row.win_probability - market_implied, 4)
                            db_row.value_rating = _value_rating(db_row.win_probability, new_o, db_row.overlay)
                            total_updated += 1
                        # Apply the recomputed market_rank to every active runner that
                        # appeared in the ranking, not just those with refreshed odds.
                        new_rank = rank_map.get(row.id)
                        if new_rank is not None:
                            db_row.market_rank = new_rank
                        steam = steam_map.get(row.id)
                        if steam and db_row.enriched_json:
                            try:
                                er = json.loads(db_row.enriched_json)
                                er.update(steam)
                                db_row.enriched_json = json.dumps(er)
                            except Exception:
                                pass

                await session.commit()

        log.info("[live-odds] Updated %d runners across %d venues", total_updated, len(by_venue))

    except Exception as e:
        log.exception("[live-odds] Failed: %s", e)


_MIN_JOCKEY_TRAINER_SAMPLES = 10  # require at least this many starts before using computed rate
_MIN_HORSE_SAMPLES = 5


async def _inject_accumulated_stats(race, session) -> None:
    """
    Inject real jockey/trainer/career win rates from historical_results into Race runners.
    Runs after parse_race() so it overrides the flat 10.0 defaults from RA/Betfair.
    Only updates fields where we have >= MIN_SAMPLES observations.

    All aggregates exclude historical_results rows on or after race.date so that
    today's already-resulted races don't leak into the features for races later
    today (and so backtests are deterministic).
    """
    from sqlalchemy import text as sa_text

    venue = (race.venue or "").lower()
    distance = race.distance or 0
    # race_id format is "YYYY-MM-DD_venue_RN"; lex-comparing race_id < race.date
    # correctly excludes today's races and any future seeded results.
    race_date_prefix = race.date or date.today().isoformat()

    jockeys = list({r.jockey for r in race.runners if r.jockey})
    trainers = list({r.trainer for r in race.runners if r.trainer})
    horses = list({r.horse_name for r in race.runners if r.horse_name})

    if not jockeys and not trainers and not horses:
        return

    # ── Jockey stats ─────────────────────────────────────────────────────
    jockey_stats_map: dict[str, dict] = {}
    if jockeys:
        placeholders = ", ".join(f":j{i}" for i in range(len(jockeys)))
        params = {f"j{i}": j for i, j in enumerate(jockeys)}
        params["venue"] = venue
        params["dist_lo"] = distance - 200
        params["dist_hi"] = distance + 200
        params["race_date_prefix"] = race_date_prefix
        rows = (await session.execute(sa_text(f"""
            SELECT
                jockey,
                COUNT(*) AS starts,
                SUM(CASE WHEN winner THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN LOWER(venue) = :venue THEN 1 ELSE 0 END) AS track_starts,
                SUM(CASE WHEN LOWER(venue) = :venue AND winner THEN 1 ELSE 0 END) AS track_wins,
                SUM(CASE WHEN distance BETWEEN :dist_lo AND :dist_hi THEN 1 ELSE 0 END) AS dist_starts,
                SUM(CASE WHEN distance BETWEEN :dist_lo AND :dist_hi AND winner THEN 1 ELSE 0 END) AS dist_wins,
                SUM(CASE WHEN barrier <= 5 THEN 1 ELSE 0 END) AS bl_starts,
                SUM(CASE WHEN barrier <= 5 AND winner THEN 1 ELSE 0 END) AS bl_wins,
                SUM(CASE WHEN barrier BETWEEN 6 AND 10 THEN 1 ELSE 0 END) AS bm_starts,
                SUM(CASE WHEN barrier BETWEEN 6 AND 10 AND winner THEN 1 ELSE 0 END) AS bm_wins,
                SUM(CASE WHEN barrier > 10 THEN 1 ELSE 0 END) AS bw_starts,
                SUM(CASE WHEN barrier > 10 AND winner THEN 1 ELSE 0 END) AS bw_wins,
                SUM(wins_season_placeholder) AS wins_season
            FROM (
                SELECT jockey, winner, venue, distance, barrier, 0 AS wins_season_placeholder
                FROM historical_results
                WHERE jockey IN ({placeholders}) AND jockey IS NOT NULL
                  AND race_id < :race_date_prefix
            ) sub
            GROUP BY jockey
        """), params)).fetchall()
        for row in rows:
            jockey_stats_map[row[0]] = {
                "starts": row[1], "wins": row[2],
                "track_starts": row[3], "track_wins": row[4],
                "dist_starts": row[5], "dist_wins": row[6],
                "bl_starts": row[7], "bl_wins": row[8],
                "bm_starts": row[9], "bm_wins": row[10],
                "bw_starts": row[11], "bw_wins": row[12],
            }

    # ── Trainer stats ─────────────────────────────────────────────────────
    trainer_stats_map: dict[str, dict] = {}
    if trainers:
        placeholders = ", ".join(f":t{i}" for i in range(len(trainers)))
        params = {f"t{i}": t for i, t in enumerate(trainers)}
        params["venue"] = venue
        params["dist_lo"] = distance - 200
        params["dist_hi"] = distance + 200
        params["race_date_prefix"] = race_date_prefix
        rows = (await session.execute(sa_text(f"""
            SELECT
                trainer,
                COUNT(*) AS starts,
                SUM(CASE WHEN winner THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN LOWER(venue) = :venue THEN 1 ELSE 0 END) AS track_starts,
                SUM(CASE WHEN LOWER(venue) = :venue AND winner THEN 1 ELSE 0 END) AS track_wins,
                SUM(CASE WHEN distance BETWEEN :dist_lo AND :dist_hi THEN 1 ELSE 0 END) AS dist_starts,
                SUM(CASE WHEN distance BETWEEN :dist_lo AND :dist_hi AND winner THEN 1 ELSE 0 END) AS dist_wins,
                SUM(CASE WHEN LOWER(track_condition) LIKE 'soft%' OR LOWER(track_condition) LIKE 'heavy%' THEN 1 ELSE 0 END) AS wet_starts,
                SUM(CASE WHEN (LOWER(track_condition) LIKE 'soft%' OR LOWER(track_condition) LIKE 'heavy%') AND winner THEN 1 ELSE 0 END) AS wet_wins
            FROM historical_results
            WHERE trainer IN ({placeholders}) AND trainer IS NOT NULL
              AND race_id < :race_date_prefix
            GROUP BY trainer
        """), params)).fetchall()
        for row in rows:
            trainer_stats_map[row[0]] = {
                "starts": row[1], "wins": row[2],
                "track_starts": row[3], "track_wins": row[4],
                "dist_starts": row[5], "dist_wins": row[6],
                "wet_starts": row[7], "wet_wins": row[8],
            }

    # ── Horse career stats ────────────────────────────────────────────────
    horse_stats_map: dict[str, dict] = {}
    if horses:
        placeholders = ", ".join(f":h{i}" for i in range(len(horses)))
        params = {f"h{i}": h.lower() for i, h in enumerate(horses)}
        params["venue"] = venue
        params["dist_lo"] = distance - 200
        params["dist_hi"] = distance + 200
        params["race_date_prefix"] = race_date_prefix
        rows = (await session.execute(sa_text(f"""
            SELECT
                LOWER(horse_name) AS hname,
                COUNT(*) AS starts,
                SUM(CASE WHEN winner THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN placed THEN 1 ELSE 0 END) AS places,
                SUM(CASE WHEN LOWER(venue) = :venue THEN 1 ELSE 0 END) AS track_starts,
                SUM(CASE WHEN LOWER(venue) = :venue AND winner THEN 1 ELSE 0 END) AS track_wins,
                SUM(CASE WHEN distance BETWEEN :dist_lo AND :dist_hi THEN 1 ELSE 0 END) AS dist_starts,
                SUM(CASE WHEN distance BETWEEN :dist_lo AND :dist_hi AND winner THEN 1 ELSE 0 END) AS dist_wins,
                SUM(CASE WHEN LOWER(track_condition) LIKE 'soft%' OR LOWER(track_condition) LIKE 'heavy%' THEN 1 ELSE 0 END) AS wet_starts,
                SUM(CASE WHEN (LOWER(track_condition) LIKE 'soft%' OR LOWER(track_condition) LIKE 'heavy%') AND winner THEN 1 ELSE 0 END) AS wet_wins
            FROM historical_results
            WHERE LOWER(horse_name) IN ({placeholders})
              AND race_id < :race_date_prefix
            GROUP BY LOWER(horse_name)
        """), params)).fetchall()
        for row in rows:
            horse_stats_map[row[0]] = {
                "starts": row[1], "wins": row[2], "places": row[3],
                "track_starts": row[4], "track_wins": row[5],
                "dist_starts": row[6], "dist_wins": row[7],
                "wet_starts": row[8], "wet_wins": row[9],
            }

    def _rate(wins, starts, default=10.0):
        return round(100.0 * wins / starts, 1) if starts >= _MIN_JOCKEY_TRAINER_SAMPLES else default

    def _horse_rate(wins, starts, default=0):
        return wins if starts >= _MIN_HORSE_SAMPLES else default

    # ── Inject into runners ───────────────────────────────────────────────
    for runner in race.runners:
        # Career stats from DB
        hs = horse_stats_map.get((runner.horse_name or "").lower())
        if hs and hs["starts"] >= _MIN_HORSE_SAMPLES:
            runner.career_starts = hs["starts"]
            runner.career_wins = hs["wins"]
            runner.career_places = hs["places"]
            runner.track_starts = hs["track_starts"]
            runner.track_wins = hs["track_wins"]
            runner.distance_starts = hs["dist_starts"]
            runner.distance_wins = hs["dist_wins"]

            # Condition (wet/dry) stats from DB — use career wet record, not just last 10
            condition_cat = (race.track_condition or "").lower()
            is_wet_day = any(w in condition_cat for w in ("soft", "heavy"))
            wet_s = hs.get("wet_starts") or 0
            wet_w = hs.get("wet_wins") or 0
            if is_wet_day:
                runner.condition_starts = wet_s
                runner.condition_wins = wet_w
            else:
                runner.condition_starts = max(0, hs["starts"] - wet_s)
                runner.condition_wins = max(0, hs["wins"] - wet_w)

        # Jockey stats from DB
        js = jockey_stats_map.get(runner.jockey or "")
        if js and runner.jockey_stats:
            runner.jockey_stats.win_rate_overall = _rate(js["wins"], js["starts"])
            runner.jockey_stats.win_rate_track = _rate(js["track_wins"], js["track_starts"])
            runner.jockey_stats.win_rate_distance = _rate(js["dist_wins"], js["dist_starts"])
            runner.jockey_stats.win_rate_barrier_low = _rate(js["bl_wins"], js["bl_starts"])
            runner.jockey_stats.win_rate_barrier_mid = _rate(js["bm_wins"], js["bm_starts"])
            runner.jockey_stats.win_rate_barrier_wide = _rate(js["bw_wins"], js["bw_starts"])
            runner.jockey_stats.wins_season = js["wins"]

        # Trainer stats from DB
        ts = trainer_stats_map.get(runner.trainer or "")
        if ts and runner.trainer_stats:
            runner.trainer_stats.win_rate_overall = _rate(ts["wins"], ts["starts"])
            runner.trainer_stats.win_rate_track = _rate(ts["track_wins"], ts["track_starts"])
            runner.trainer_stats.win_rate_distance = _rate(ts["dist_wins"], ts["dist_starts"])
            runner.trainer_stats.win_rate_wet = _rate(ts["wet_wins"], ts["wet_starts"])
            runner.trainer_stats.wins_season = ts["wins"]

    # ── Form history (last_10_starts) from historical_results ─────────────────
    # Only inject if runner has no existing form (RA doesn't provide it)
    form_by_horse: dict[str, list] = {}
    if horses and all(len(r.last_10_starts) == 0 for r in race.runners):
        from horse_engine.models.race import FormStart
        today_prefix = race.date or date.today().isoformat()
        placeholders = ", ".join(f":h{i}" for i in range(len(horses)))
        params = {f"h{i}": h.lower() for i, h in enumerate(horses)}
        params["today_prefix"] = today_prefix

        form_rows = (await session.execute(sa_text(f"""
            SELECT hname, race_id, position, beaten_margin, venue, distance,
                   track_condition, barrier, weight, prize_money, race_class,
                   field_size, starting_price
            FROM (
                SELECT LOWER(horse_name) AS hname, race_id, position, beaten_margin,
                       venue, distance, track_condition, barrier, weight, prize_money,
                       race_class, field_size, starting_price,
                       ROW_NUMBER() OVER (
                           PARTITION BY LOWER(horse_name) ORDER BY race_id DESC
                       ) AS rn
                FROM historical_results
                WHERE LOWER(horse_name) IN ({placeholders})
                  AND race_id < :today_prefix
            ) sub
            WHERE rn <= 20
            ORDER BY hname, race_id DESC
        """), params)).fetchall()

        # Group into per-horse form history
        for row in form_rows:
            form_by_horse.setdefault(row[0], []).append(row)

        for runner in race.runners:
            hname = (runner.horse_name or "").lower()
            rows = form_by_horse.get(hname, [])
            if not rows:
                continue

            starts: list[FormStart] = []
            for row in rows:
                try:
                    starts.append(FormStart(
                        date=row[1][:10],                     # race_id[:10] = YYYY-MM-DD
                        track=row[4] or "",                   # venue
                        distance=int(row[5] or 0),
                        track_condition=row[6] or "Good",
                        barrier=int(row[7] or 0),
                        weight=float(row[8] or 0),
                        jockey="",
                        position=int(row[2] or 0),
                        finishers=int(row[11] or max(int(row[2] or 1) + 3, 8)),
                        beaten_margin=float(row[3] or 0),
                        race_class=row[10] or "",
                        prize_money=int(row[9] or 0),
                        starting_price=float(row[12]) if row[12] else None,
                    ))
                except Exception:
                    continue

            if not starts:
                continue

            runner.last_10_starts = starts[:10]

            # Detect first-up and second-up runs and populate stats
            sorted_asc = sorted(starts, key=lambda s: s.date)
            fu_starts = fu_wins = 0
            su_starts = su_wins = 0
            td_starts = td_wins = 0
            prev_was_fu = False
            for i, s in enumerate(sorted_asc):
                if i == 0:
                    is_fu = True
                    is_su = False
                else:
                    try:
                        prev_date = date.fromisoformat(sorted_asc[i - 1].date)
                        curr_date = date.fromisoformat(s.date)
                        gap = (curr_date - prev_date).days
                        is_fu = gap >= 60
                        is_su = prev_was_fu and gap < 60
                    except Exception:
                        is_fu = is_su = False
                if is_fu:
                    fu_starts += 1
                    if s.position == 1:
                        fu_wins += 1
                if is_su:
                    su_starts += 1
                    if s.position == 1:
                        su_wins += 1
                prev_was_fu = is_fu

                # Track+distance: within 200m at same venue
                if (venue and s.track.lower() == venue and
                        distance and abs(s.distance - distance) <= 200):
                    td_starts += 1
                    if s.position == 1:
                        td_wins += 1

            if fu_starts > 0:
                runner.first_up_starts = fu_starts
                runner.first_up_wins = fu_wins
            if su_starts > 0:
                runner.second_up_starts = su_starts
                runner.second_up_wins = su_wins
            if td_starts > 0:
                runner.track_distance_starts = td_starts
                runner.track_distance_wins = td_wins

        # ── Trainer first-up / second-up stats from form history ─────────────────
        # Compute per-trainer first-up and second-up stats from the full form history.
        # This covers what RA enrichment provides, using DB data for backfill rows.
        if form_by_horse:
            trainer_fu_map: dict[str, list[tuple[bool, bool, bool]]] = {}  # trainer → [(won, is_fu, is_su), ...]

            for runner in race.runners:
                if not runner.trainer or not runner.trainer_stats:
                    continue
                hname = (runner.horse_name or "").lower()
                rows = form_by_horse.get(hname, [])
                if not rows:
                    continue

                sorted_form = sorted(rows, key=lambda r: r[1])  # sort by race_id asc
                prev_was_fu_t = False
                for j, row in enumerate(sorted_form):
                    won = row[2] == 1  # position == 1
                    if j == 0:
                        is_fu_t = True
                        is_su_t = False
                    else:
                        try:
                            prev_d = date.fromisoformat(sorted_form[j-1][1][:10])
                            curr_d = date.fromisoformat(row[1][:10])
                            gap_t = (curr_d - prev_d).days
                            is_fu_t = gap_t >= 60
                            is_su_t = prev_was_fu_t and gap_t < 60
                        except Exception:
                            is_fu_t = is_su_t = False

                    if is_fu_t or is_su_t:
                        trainer_fu_map.setdefault(runner.trainer, []).append((won, is_fu_t, is_su_t))
                    prev_was_fu_t = is_fu_t

            # Inject trainer first-up/second-up rates
            for runner in race.runners:
                if not runner.trainer or not runner.trainer_stats:
                    continue
                entries = trainer_fu_map.get(runner.trainer, [])
                fu_entries = [(won, True) for won, is_fu_t, _ in entries if is_fu_t]
                su_entries = [(won, True) for won, _, is_su_t in entries if is_su_t]

                fu_n = len(fu_entries)
                fu_w = sum(1 for won, _ in fu_entries if won)
                if fu_n >= _MIN_JOCKEY_TRAINER_SAMPLES:
                    runner.trainer_stats.win_rate_first_up = round(100.0 * fu_w / fu_n, 1)

                su_n = len(su_entries)
                su_w = sum(1 for won, _ in su_entries if won)
                if su_n >= _MIN_JOCKEY_TRAINER_SAMPLES:
                    runner.trainer_stats.win_rate_second_up = round(100.0 * su_w / su_n, 1)


async def _seed_results_for_date(race_date: str) -> int:
    """Fetch settled results for race_date and store as training data. Returns count seeded."""
    client = get_tab_client()

    # Fast path for past dates: use stored venue+state to find RA results directly.
    # RA Calendar.aspx only lists future/current meetings — it can't resolve yesterday.
    async with get_session() as session:
        pred_rows_for_date = (await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{race_date}_%"))
            .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
        )).scalars().all()
        already_seeded_ra = set(
            (await session.execute(
                select(HistoricalResultRow.race_id)
                .where(HistoricalResultRow.race_id.like(f"{race_date}_%"))
                .distinct()
            )).scalars().all()
        )
    venue_state_map_ra: dict[tuple[str, str], set[str]] = {}
    for row in pred_rows_for_date:
        v = (row.venue or "").strip()
        s = (row.state or "").strip().upper()
        if v and s:
            venue_state_map_ra.setdefault((v, s), set()).add(row.race_id)
    ra_seeded_total = 0
    ra = client._ra
    for (venue_name, state), race_ids in venue_state_map_ra.items():
        try:
            _, results = await ra.find_results(race_date, state, venue_name)
        except Exception:
            results = {}
        if not results:
            continue
        venue_code = _parse_race_id(list(race_ids)[0])[1]
        matched_preds = {p.race_id: p for p in pred_rows_for_date
                        if _parse_race_id(p.race_id)[1] == venue_code}
        async with get_session() as session:
            for race_num, race_data in results.items():
                race_id = f"{race_date}_{venue_code}_R{race_num}"
                if race_id in already_seeded_ra:
                    continue
                runners_data = race_data.get("runners", {})
                if not runners_data:
                    continue
                for name_lower, rd in runners_data.items():
                    pos = rd.get("position")
                    if not pos or pos <= 0:
                        continue
                    matched = next(
                        (p for p in matched_preds.values()
                         if p.race_id == race_id
                         and _normalize_horse(p.horse_name) == _normalize_horse(name_lower)),
                        None,
                    )
                    existing_at_pos = (await session.execute(
                        select(HistoricalResultRow.horse_name)
                        .where(HistoricalResultRow.race_id == race_id)
                        .where(HistoricalResultRow.position == pos)
                    )).scalars().all()
                    if any(_normalize_horse(h) == _normalize_horse(name_lower) for h in existing_at_pos):
                        continue
                    display_name = matched.horse_name if matched else name_lower.title()
                    session.add(HistoricalResultRow(
                        race_id=race_id,
                        horse_name=display_name,
                        position=pos,
                        beaten_margin=float(rd.get("margin") or 0),
                        winner=pos == 1,
                        placed=pos <= 3,
                        starting_price=rd.get("sp"),
                        feature_vector_json=matched.enriched_json if matched else None,
                    ))
                    ra_seeded_total += 1
                already_seeded_ra.add(race_id)
            await session.commit()
    if ra_seeded_total:
        log.info("[seed-results] RA direct-key seeded %d entries for %s", ra_seeded_total, race_date)

    # Primary source: meetings from Racing Australia (works reliably for today/upcoming)
    meetings = await client.get_meetings(race_date)
    date_sfx = f"-{race_date.replace('-', '')}"
    ra_slugs = {m.get("slug", ""): m for m in meetings if m.get("slug")}

    # Secondary source: venues we already have predictions for — RA form guide
    # doesn't reliably list past country meetings so many venues get missed.
    async with get_session() as session:
        db_race_ids = (await session.execute(
            select(RunnerPredictionRow.race_id, RunnerPredictionRow.venue)
            .where(RunnerPredictionRow.race_id.like(f"{race_date}_%"))
            .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            .distinct()
        )).all()
    db_venue_codes = {}
    for row in db_race_ids:
        _, vc, _ = _parse_race_id(row.race_id)
        if vc:
            db_venue_codes[vc] = row.venue or vc

    # Build union of meetings: RA slugs + DB-only venues (synthesise slug for DB venues)
    all_meetings = list(meetings)
    ra_vcs = set()
    for m in meetings:
        slug = m.get("slug", "")
        vc = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else ""
        if vc:
            ra_vcs.add(vc)
    for vc, venue_name in db_venue_codes.items():
        if vc not in ra_vcs:
            synth_slug = f"{vc}{date_sfx}"
            all_meetings.append({"slug": synth_slug, "id": None, "name": venue_name, "state": "", "rail_position": ""})
            log.info("[seed-results] Adding DB-only venue %s (slug: %s) for %s", vc, synth_slug, race_date)

    seeded = 0
    for meeting in all_meetings:
        slug = meeting.get("slug", "")
        venue_code = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else ""
        races_with_results: set[str] = set()
        raw_races = await client.get_meeting_races(slug)
        for raw_race in raw_races:
            race_num = raw_race.get("eventNumber")
            race_id = f"{race_date}_{venue_code}_R{race_num}"
            full_race = await client.get_race(slug, race_num)
            if not full_race:
                continue

            # Build selection lookup: horse_name → selection dict (has jockey, trainer, barrier, etc.)
            sel_by_name = {}
            for sel in full_race.get("selections") or []:
                sel_name = ((sel.get("competitor") or {}).get("name") or "").lower()
                if sel_name:
                    sel_by_name[sel_name] = sel

            race_meeting = full_race.get("_meeting") or {}
            race_venue = race_meeting.get("venue") or venue_code
            race_state = race_meeting.get("state") or meeting.get("state") or ""
            race_distance = int(raw_race.get("distance") or 0) or None
            race_condition = (raw_race.get("trackCondition") or {}).get("overall") or ""
            if race_condition and (raw_race.get("trackCondition") or {}).get("rating"):
                race_condition = f"{race_condition} {(raw_race.get('trackCondition') or {}).get('rating')}".strip()
            race_class = raw_race.get("eventClass") or ""
            runners_list = full_race.get("runners") or []
            race_field_size = len([r for r in runners_list if not r.get("scratched")])

            for r in runners_list:
                if r.get("scratched"):
                    continue
                position = r.get("finishingPosition")
                if not position or int(position) <= 0:
                    continue
                horse = r.get("runnerName", "")
                beaten = float(r.get("margin", 0) or 0)
                sp = None
                for p in r.get("prices", []):
                    if p.get("priceType") in ("StartingPrice", "SP", "Win"):
                        sp = float(p.get("winPrice", 0) or 0) or None
                        break

                sel = sel_by_name.get(horse.lower()) or {}
                jockey = (sel.get("jockey") or {}).get("name") or ""
                trainer = (sel.get("trainer") or {}).get("name") or ""
                barrier = int(sel.get("barrierNumber") or 0) or None
                tab_num = int(sel.get("competitorNumber") or 0) or None
                weight = float(sel.get("weight") or 0) or None
                comp = sel.get("competitor") or {}
                age = int(comp.get("age") or 0) or None
                sex = comp.get("sex") or ""
                prize = int(full_race.get("prize_money") or 0) or None

                async with get_session() as session:
                    existing = await session.execute(
                        select(HistoricalResultRow)
                        .where(HistoricalResultRow.race_id == race_id)
                        .where(HistoricalResultRow.horse_name == horse)
                        .limit(1)
                    )
                    existing_row = existing.scalars().first()
                    if existing_row:
                        # Patch SP and any missing context fields
                        updated = False
                        if existing_row.starting_price is None and sp:
                            existing_row.starting_price = sp
                            updated = True
                        if not existing_row.jockey and jockey:
                            existing_row.jockey = jockey
                            updated = True
                        if not existing_row.trainer and trainer:
                            existing_row.trainer = trainer
                            updated = True
                        if not existing_row.venue and race_venue:
                            existing_row.venue = race_venue
                            updated = True
                        if not existing_row.distance and race_distance:
                            existing_row.distance = race_distance
                            updated = True
                        if not existing_row.track_condition and race_condition:
                            existing_row.track_condition = race_condition
                            updated = True
                        if existing_row.barrier is None and barrier:
                            existing_row.barrier = barrier
                            updated = True
                        if not existing_row.state and race_state:
                            existing_row.state = race_state
                            updated = True
                        if updated:
                            await session.commit()
                        races_with_results.add(race_id)
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
                        beaten_margin=beaten,
                        winner=int(position) == 1,
                        placed=int(position) <= 3,
                        starting_price=sp,
                        feature_vector_json=fv_row.enriched_json if fv_row else None,
                        jockey=jockey or None,
                        trainer=trainer or None,
                        venue=race_venue or None,
                        state=race_state or None,
                        distance=race_distance,
                        track_condition=race_condition or None,
                        barrier=barrier,
                        tab_number=tab_num,
                        weight=weight,
                        age=age,
                        sex=sex or None,
                        race_class=race_class or None,
                        prize_money=prize,
                        field_size=race_field_size or None,
                        race_number=race_num,
                    ))
                    await session.commit()
                    seeded += 1
                    races_with_results.add(race_id)

        # Clear stale cancelled flags only for mass-cancelled races (all runners cancelled).
        # Never clear individual dedup cancellations — only restore when the whole race
        # was mass-cancelled by _cancel_abandoned_meetings due to a feed block.
        if races_with_results:
            from sqlalchemy import update as sa_update
            async with get_session() as session:
                for rid in list(races_with_results):
                    total = (await session.execute(
                        select(func.count()).select_from(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == rid)
                    )).scalar_one()
                    n_cancelled = (await session.execute(
                        select(func.count()).select_from(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == rid)
                        .where(RunnerPredictionRow.cancelled.is_(True))
                    )).scalar_one()
                    if total > 0 and n_cancelled == total:
                        await session.execute(
                            sa_update(RunnerPredictionRow)
                            .where(RunnerPredictionRow.race_id == rid)
                            .where(RunnerPredictionRow.cancelled.is_(True))
                            .values(cancelled=False)
                        )
                await session.commit()
    return seeded + ra_seeded_total


async def _seed_race_results_on_demand(race_ids: list[str]) -> dict[tuple, int]:
    """
    For finished races not yet in HistoricalResultRow, fetch from RA once and persist.
    Returns {(race_id, horse_name): position} for all seeded entries.
    Only called on first page load after a race finishes — subsequent loads use DB.
    """
    if not race_ids:
        return {}

    async with get_session() as session:
        existing_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name, HistoricalResultRow.position)
            .where(HistoricalResultRow.race_id.in_(race_ids))
        )).all()

    db_positions: dict[tuple, int] = {(r.race_id, r.horse_name): r.position for r in existing_rows}
    seeded_ids: set[str] = {r.race_id for r in existing_rows}
    unseeded = [rid for rid in race_ids if rid not in seeded_ids]

    if unseeded:
        client = get_tab_client()
        # Group unseeded race_ids by (date, venue_code)
        venue_date_map: dict[tuple, list[str]] = {}
        for rid in unseeded:
            date_str, vc, _ = _parse_race_id(rid)
            venue_date_map.setdefault((vc, date_str), []).append(rid)

        for (vc, date_str), target_race_ids in venue_date_map.items():
            slug = _meeting_slug(vc, date_str)
            try:
                events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=20)
                for event in events or []:
                    rn = event.get("eventNumber")
                    race_id = f"{date_str}_{vc}_R{rn}"
                    if race_id not in target_race_ids:
                        continue
                    full_race = await asyncio.wait_for(client.get_race(slug, rn), timeout=15)
                    if not full_race:
                        continue
                    rows_to_add = []
                    for r in full_race.get("runners", []):
                        if r.get("scratched"):
                            continue
                        pos = r.get("finishingPosition")
                        if not pos or int(pos) <= 0:
                            continue
                        horse = r.get("runnerName", "")
                        if not horse:
                            continue
                        sp = None
                        for p in r.get("prices", []):
                            if p.get("priceType") in ("StartingPrice", "SP"):
                                sp = float(p.get("winPrice", 0) or 0) or None
                                break
                        rows_to_add.append(HistoricalResultRow(
                            race_id=race_id,
                            horse_name=horse,
                            position=int(pos),
                            beaten_margin=float(r.get("margin", 0) or 0),
                            winner=int(pos) == 1,
                            placed=int(pos) <= 3,
                            starting_price=sp,
                        ))
                        db_positions[(race_id, horse)] = int(pos)
                    if rows_to_add:
                        from sqlalchemy import update as sa_update
                        async with get_session() as session:
                            for row in rows_to_add:
                                existing = (await session.execute(
                                    select(HistoricalResultRow.id, HistoricalResultRow.starting_price)
                                    .where(HistoricalResultRow.race_id == row.race_id)
                                    .where(HistoricalResultRow.horse_name == row.horse_name)
                                    .limit(1)
                                )).first()
                                if not existing:
                                    session.add(row)
                                elif existing.starting_price is None and row.starting_price is not None:
                                    # Row was written by _persist_live_results without SP; fill it in now
                                    await session.execute(
                                        sa_update(HistoricalResultRow)
                                        .where(HistoricalResultRow.id == existing.id)
                                        .values(starting_price=row.starting_price)
                                    )
                            await session.commit()
                        log.info("[on-demand-seed] Seeded %d results for %s", len(rows_to_add), race_id)
            except Exception as e:
                log.warning("[on-demand-seed] Failed to fetch %s/%s: %s", vc, date_str, e)

    return db_positions


async def _persist_live_results(race_id: str, all_tote: list) -> None:
    """Persist settled race results seen in a live-odds fetch to HistoricalResultRow.
    Called as a background task — does not block the live-odds response.

    Seeding authority (highest → lowest):
      1. _seed_results_for_date  — cron, RA, has feature_vector_json and real SP
      2. _seed_race_results_on_demand — on first page view; RA, real SP from race data
      3. _persist_live_results   — fastest but no official SP; writes starting_price=None
    """
    try:
        async with get_session() as session:
            for horse, _current_odds, position in all_tote:
                if not position or not horse:
                    continue
                existing = (await session.execute(
                    select(HistoricalResultRow.id)
                    .where(HistoricalResultRow.race_id == race_id)
                    .where(HistoricalResultRow.horse_name == horse)
                    .limit(1)
                )).scalar()
                if not existing:
                    session.add(HistoricalResultRow(
                        race_id=race_id,
                        horse_name=horse,
                        position=position,
                        beaten_margin=0.0,
                        winner=position == 1,
                        placed=position <= 3,
                        starting_price=None,  # live tote ≠ official SP; cron fills this in
                    ))
            await session.commit()
        log.info("[live-odds] Persisted results for %s", race_id)
    except Exception as e:
        log.warning("[live-odds] Failed to persist results for %s: %s", race_id, e)


async def _scheduled_seed_results():
    """Snapshot any unrecorded pre-race predictions, then seed settled results."""
    today     = _today_aest().isoformat()
    yesterday = (_today_aest() - timedelta(days=1)).isoformat()
    # Snapshot BEFORE seeding — the time guard in _snapshot_prerace_predictions ensures
    # only races that haven't started yet are captured. Seeding first would mark races
    # as resulted, and a snapshot without the time guard would capture post-race state.
    try:
        n = await _snapshot_prerace_predictions()
        if n:
            log.info("[scheduler] Pre-seed snapshot: %d races written", n)
    except Exception as e:
        log.exception("[scheduler] Pre-seed snapshot failed: %s", e)
    for race_date in (today, yesterday):
        log.info("[scheduler] Seeding results for %s", race_date)
        try:
            n = await _seed_results_for_date(race_date)
            log.info("[scheduler] Seeded %d results for %s", n, race_date)
        except Exception as e:
            log.exception("[scheduler] Result seeding failed for %s: %s", race_date, e)


async def _scheduled_exotic_retrain():
    """Run by APScheduler at 3am AEST — retrain exotic model after nightly calibration."""
    log.info("[scheduler] Running nightly exotic model retrain")
    try:
        async with get_session() as session:
            hr_result = await session.execute(select(HistoricalResultRow))
            hr_rows = hr_result.scalars().all()
            hist_result = await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
                .where(
                    RunnerPredictionHistoryRow.cancelled.is_(False)
                    | RunnerPredictionHistoryRow.cancelled.is_(None)
                )
            )
            hist_rows = hist_result.scalars().all()

        # Result lookup: (race_id, normalized_name) → (placed, position)
        result_lookup: dict[tuple, tuple] = {
            (r.race_id, _normalize_horse(r.horse_name)): (bool(r.placed), r.position)
            for r in hr_rows
        }

        race_data: dict[str, list] = {}
        for row in hist_rows:
            key = (row.race_id, _normalize_horse(row.horse_name))
            outcome = result_lookup.get(key)
            if not outcome:
                continue
            placed, position = outcome
            try:
                er = EnrichedRunner(**json.loads(row.enriched_json))
                fv = build_feature_vector(er)
                race_data.setdefault(row.race_id, []).append((fv, 1 if placed else 0, position))
            except Exception:
                continue

        race_groups = [
            runners for runners in race_data.values()
            if len(runners) >= 7 and sum(1 for _, lbl, _ in runners if lbl == 1) == 3
        ]

        if not race_groups:
            log.warning("[scheduler] Exotic retrain: no eligible races found")
            return

        m = ExoticModel()
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, m.train_exotic, race_groups)
        async with get_session() as session:
            await save_exotic_model_weights(session, stats["weights"])
        log.info(
            "[scheduler] Exotic retrain complete — %d races, tri_box=%.3f ff_box=%.3f",
            len(race_groups), stats.get("tri_box_hit_rate", 0), stats.get("ff_box_hit_rate", 0),
        )
    except Exception as e:
        log.exception("[scheduler] Exotic retrain failed: %s", e)


async def _snapshot_prerace_predictions() -> int:
    """
    9am AEST job: write RunnerPredictionHistoryRow for every today race that
    (a) has no history snapshot yet, and (b) hasn't started yet.
    Idempotent — safe to call multiple times.
    """
    today = _today_aest().isoformat()
    now_utc = datetime.utcnow()
    written = 0

    # Build race_id → scheduled_time from Racing Australia (authoritative start times)
    sched_map: dict[str, datetime] = {}
    try:
        client = get_tab_client()
        date_sfx = f"-{today.replace('-', '')}"
        meetings = await asyncio.wait_for(client.get_meetings(today), timeout=20)
        for m in meetings:
            slug = m.get("slug", "")
            vc = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0] if slug else ""
            try:
                races_raw = await asyncio.wait_for(client.get_meeting_races(slug), timeout=15)
                for ev in races_raw:
                    rnum = ev.get("eventNumber")
                    start = ev.get("startTime") or ev.get("scheduledTime")
                    if rnum and start:
                        rid = f"{today}_{vc}_R{rnum}"
                        sched_map[rid] = datetime.fromisoformat(start.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                pass
    except Exception as e:
        log.warning("[snapshot] Could not fetch RA start times: %s", e)

    async with get_session() as session:
        already_result = await session.execute(
            select(RunnerPredictionHistoryRow.race_id)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{today}_%"))
            .distinct()
        )
        already_set = set(already_result.scalars().all())

        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
            .where(RunnerPredictionRow.win_probability.isnot(None))
            .where(RunnerPredictionRow.enriched_json.isnot(None))
        )
        rows = pred_result.scalars().all()

    # Group by race — skip races already in history OR that have already started.
    # The time guard (BUG-01 fix) prevents post-race mutable state from being
    # snapshotted as if it were a pre-race prediction.
    races: dict[str, list[RunnerPredictionRow]] = {}
    for r in rows:
        if r.race_id in already_set:
            continue
        # Time guard: skip if race has already started
        sched = sched_map.get(r.race_id)
        if sched is None and r.scheduled_time:
            try:
                sched = datetime.fromisoformat(str(r.scheduled_time).replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass
        if sched and sched <= now_utc:
            continue
        races.setdefault(r.race_id, []).append(r)

    if not races:
        log.info("[snapshot] No unsnapshotted pre-race races for %s", today)
        return 0

    import uuid as _uuid
    async with get_session() as session:
        for race_id, runners in races.items():
            batch_id = str(_uuid.uuid4())  # shared across all runners in this enrichment batch
            for r in runners:
                try:
                    session.add(RunnerPredictionHistoryRow(
                        race_id=r.race_id,
                        horse_name=r.horse_name,
                        tab_number=r.tab_number,
                        barrier=r.barrier,
                        jockey=r.jockey,
                        trainer=r.trainer,
                        weight=r.weight,
                        win_probability=r.win_probability,
                        place_probability=r.place_probability,
                        model_rank=r.model_rank,
                        place_model_rank=r.place_model_rank,
                        exotic_model_rank=r.exotic_model_rank,
                        market_rank=r.market_rank,
                        overlay=r.overlay,
                        best_available_odds=r.best_available_odds,
                        value_rating=r.value_rating,
                        key_flags=r.key_flags,
                        enriched_json=r.enriched_json,
                        scheduled_time=r.scheduled_time,
                        enriched_at=r.enriched_at or now_utc,
                        cancelled=r.cancelled,
                        venue=r.venue,
                        state=r.state,
                        race_number=r.race_number,
                        race_name=r.race_name,
                        distance=r.distance,
                        track_condition=r.track_condition,
                        field_size=r.field_size,
                        prize_money=r.prize_money,
                        rail_position=getattr(r, "rail_position", None),
                        class_change=getattr(r, "class_change", None),
                        source="live",
                        batch_id=batch_id,
                        recorded_at=now_utc,
                    ))
                except Exception:
                    pass  # unique constraint: row already exists for this race+horse, skip
            await session.commit()
            written += 1

    log.info("[snapshot] Snapshotted %d pre-race races for %s", written, today)
    return written


async def _scheduled_prerace_snapshot():
    """APScheduler wrapper for _snapshot_prerace_predictions."""
    try:
        n = await _snapshot_prerace_predictions()
        log.info("[scheduler] Pre-race snapshot: %d races written", n)
    except Exception as e:
        log.exception("[scheduler] Pre-race snapshot failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = AsyncIOScheduler(timezone="Australia/Sydney")
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=6,  minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=10, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=13, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_prerace_snapshot, CronTrigger(hour=9, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=14, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=15, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=17, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=19, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_seed_results, CronTrigger(hour=23, minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_calibrate,      CronTrigger(hour=2,  minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_exotic_retrain, CronTrigger(hour=3,  minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(
        _scheduled_odds_snapshot,
        CronTrigger(hour="9-20", minute="0,15,30,45", timezone="Australia/Sydney")
    )
    scheduler.add_job(
        _scheduled_pre_race_enrich_and_scratch,
        CronTrigger(hour="9-20", minute="0,15,30,45", timezone="Australia/Sydney")
    )
    scheduler.add_job(
        _scheduled_live_odds_refresh,
        CronTrigger(hour="9-20", minute="0,20,40", timezone="Australia/Sydney")
    )
    scheduler.start()
    log.info("[scheduler] Cron jobs scheduled")

    # Enrich today on startup
    asyncio.create_task(_scheduled_enrich())

    # Backfill last 3 days — catch up on any missed enrichments/results
    async def _startup_backfill():
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)
        for offset in (-3, -2, -1):
            seed_date = (_today_aest() + timedelta(days=offset)).isoformat()
            try:
                await _enrich_date(seed_date, client, model)
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
    """Build the Racing Australia meeting slug from venue slug and date."""
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


_COUNTRY_CODE_RE = re.compile(r"\s*\([A-Za-z]{2,3}\)\s*$")

def _normalize_horse(name: str) -> str:
    """Lowercase and strip country code suffixes like (FR), (NZ), (IRE), (Nz) for consistent matching."""
    return _COUNTRY_CODE_RE.sub("", name).lower().strip()


# Cache meeting start times + live odds for 5 min
_edge_times_cache: dict[str, tuple[datetime, dict[int, str]]] = {}
_edge_odds_cache: dict[str, tuple[datetime, dict[str, float]]] = {}  # race_id → {horse_name: flucs_win}

# Cache full list_meetings response for 10 min (weather + RA calls are expensive)
_list_meetings_cache: dict[str, tuple[datetime, dict]] = {}  # date → (ts, response)

# Cache per-venue meeting response for 2 min — prevents thundering herd from _loadMeetingWinRate
_get_meeting_cache: dict[str, tuple[datetime, dict]] = {}  # "date/venue" → (ts, response)


def _invalidate_meeting_caches(race_date: str, venue_code: str | None = None) -> None:
    """Drop cached meeting list / detail entries whenever cancellation state for a
    date changes (admin cancel/restore, mass-cancel by _cancel_abandoned_meetings,
    scratch detection). Without this the 10-min list cache and 2-min per-venue
    cache can keep showing a cancelled meeting or a scratched runner as live.

    When venue_code is None, every per-venue entry for the date is dropped.
    """
    _list_meetings_cache.pop(race_date, None)
    if venue_code:
        _get_meeting_cache.pop(f"{race_date}/{venue_code}", None)
        return
    prefix = f"{race_date}/"
    for key in list(_get_meeting_cache.keys()):
        if key.startswith(prefix):
            _get_meeting_cache.pop(key, None)

async def _fetch_live_odds(client, race_id: str) -> dict[str, float]:
    """Return {horse_name: flucs_win_odds} for a race. Cached 5 min."""
    cached = _edge_odds_cache.get(race_id)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < 300:
        return cached[1]
    try:
        _, venue, race_num = _parse_race_id(race_id)
        date_part = race_id[:10]
        slug = _meeting_slug(venue, date_part)
        event = await asyncio.wait_for(client.get_race(slug, race_num), timeout=20)
        if not event:
            return {}
        odds: dict[str, float] = {}
        for sel in event.get("selections", []):
            if (sel.get("status") or "").upper() == "SCRATCHED":
                continue
            comp = sel.get("competitor") or {}
            name = comp.get("name")
            flucs = sel.get("flucs") or {}
            win = sel.get("topToteWin") or flucs.get("low") or flucs.get("open") or sel.get("startingPrice")
            if name and win:
                odds[name] = float(win)
        _edge_odds_cache[race_id] = (datetime.utcnow(), odds)
        return odds
    except Exception:
        return {}

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


def _compute_hedge(pick_odds: float, field_size: int, hedge_horses: list[dict]) -> dict | None:
    """
    Compute Dutch place-bet insurance for a win pick.
    Tries up to 4 hedge horses with graduated recovery targets.
    Stakes normalised to a $10 win bet — UI scales proportionally.
    """
    if not pick_odds or pick_odds < 4.0 or not hedge_horses:
        return None

    WIN_STAKE = 10.0
    divisor = 4 if field_size >= 8 else 3 if field_size >= 5 else 2

    # Recovery targets per number of hedge horses (more horses = accept lower recovery)
    RECOVERY_BY_N = {1: 0.70, 2: 0.70, 3: 0.55, 4: 0.45}

    # Estimate place odds for each hedge candidate
    candidates = []
    for h in hedge_horses:
        w = h.get("win_odds") or 0
        if w <= 1.0:
            continue
        p_est = round((w - 1) / divisor + 1, 3)
        candidates.append({**h, "place_est": p_est})

    if not candidates:
        return None

    LABELS = {1: "single", 2: "double", 3: "triple", 4: "quad"}
    options = {}
    for n in range(1, min(len(candidates), 4) + 1):
        subset = candidates[:n]
        recovery = RECOVERY_BY_N.get(n, 0.45)
        sum_inv = sum(1 / h["place_est"] for h in subset)
        factor = recovery * sum_inv
        if factor >= 1.0:
            continue
        R = recovery * WIN_STAKE / (1 - factor)
        total_hedge = R * sum_inv
        total_out = WIN_STAKE + total_hedge
        win_net = round(WIN_STAKE * (pick_odds - 1) - total_hedge, 2)
        if win_net < 0:
            continue
        horses = [
            {
                "horse_name": h["horse_name"],
                "tab_number": h["tab_number"],
                "win_odds": h["win_odds"],
                "place_est": h["place_est"],
                "stake": round(R / h["place_est"], 2),
                "return_if_places": round(R, 2),
            }
            for h in subset
        ]
        key = LABELS.get(n, f"{n}-horse")
        options[key] = {
            "horses": horses,
            "total_hedge": round(total_hedge, 2),
            "total_outlay": round(total_out, 2),
            "recovery_if_fires": round(R, 2),
            "recovery_pct": round(R / total_out * 100),
            "win_net": win_net,
        }

    if not options:
        return None

    return {
        "win_stake": WIN_STAKE,
        "pick_odds": pick_odds,
        "field_size": field_size,
        "divisor": divisor,
        "options": options,
    }


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
            # Identify settled races for this date (have historical results)
            settled_q = await session.execute(
                select(HistoricalResultRow.race_id)
                .where(HistoricalResultRow.race_id.like(f"{prefix}%"))
                .distinct()
            )
            settled_race_ids: set[str] = {r for (r,) in settled_q.fetchall()}

            # For settled races, query history rank-1 — threshold evaluated against pre-race win_probability
            hist_rows: list = []
            if settled_race_ids:
                hr_q = await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id.in_(settled_race_ids))
                    .where(RunnerPredictionHistoryRow.model_rank == 1)
                    .where(RunnerPredictionHistoryRow.win_probability >= threshold)
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                    .order_by(RunnerPredictionHistoryRow.win_probability.desc())
                )
                hist_rows = hr_q.scalars().all()

            # For upcoming races, query mutable as before
            mut_q = (
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.model_rank == 1)
                .where(RunnerPredictionRow.win_probability >= threshold)
                .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                .order_by(RunnerPredictionRow.win_probability.desc())
            )
            if settled_race_ids:
                mut_q = mut_q.where(~RunnerPredictionRow.race_id.in_(settled_race_ids))
            mut_rows: list = (await session.execute(mut_q)).scalars().all()

            # For settled-race history picks with 0 odds, fall back to mutable best_available_odds.
            # History is snapshotted at 9am before the edge page triggers a refresh; mutable
            # has up-to-date odds. Use a dict — never mutate history row objects in-session.
            hist_odds_override: dict[tuple[str, str], float] = {}
            hist_zero_odds = [r for r in hist_rows if not (r.best_available_odds or 0)]
            if hist_zero_odds:
                mut_odds_q = await session.execute(
                    select(RunnerPredictionRow.race_id, RunnerPredictionRow.horse_name,
                           RunnerPredictionRow.best_available_odds)
                    .where(RunnerPredictionRow.race_id.in_({r.race_id for r in hist_zero_odds}))
                    .where(RunnerPredictionRow.horse_name.in_({r.horse_name for r in hist_zero_odds}))
                )
                for race_id, horse_name, bao in mut_odds_q.fetchall():
                    if (bao or 0) > 1.0:
                        hist_odds_override[(race_id, horse_name)] = bao

            rows = hist_rows + mut_rows
            # Exclude trial/trackwork venues
            rows = [r for r in rows if not re.search(r"-(trial|trail|jumpout)s?[_-]", r.race_id, re.IGNORECASE)]

            if not rows:
                continue

            hist_race_ids = {r.race_id for r in hist_rows}
            mut_race_ids_list = [r.race_id for r in mut_rows]

            # Batch-fetch place model runners for trifecta legs
            place_rows_list: list = []
            if hist_race_ids:
                place_rows_list.extend((await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id.in_(hist_race_ids))
                    .where(RunnerPredictionHistoryRow.place_model_rank >= 1)
                    .where(RunnerPredictionHistoryRow.place_model_rank <= 4)
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                )).scalars().all())
            if mut_race_ids_list:
                place_rows_list.extend((await session.execute(
                    select(RunnerPredictionRow)
                    .where(RunnerPredictionRow.race_id.in_(mut_race_ids_list))
                    .where(RunnerPredictionRow.place_model_rank >= 1)
                    .where(RunnerPredictionRow.place_model_rank <= 4)
                    .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                )).scalars().all())

            # Batch-fetch exotic model top-3 for alignment check
            exotic_rows_list: list = []
            if hist_race_ids:
                exotic_rows_list.extend((await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id.in_(hist_race_ids))
                    .where(RunnerPredictionHistoryRow.exotic_model_rank >= 1)
                    .where(RunnerPredictionHistoryRow.exotic_model_rank <= 3)
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                )).scalars().all())
            if mut_race_ids_list:
                exotic_rows_list.extend((await session.execute(
                    select(RunnerPredictionRow)
                    .where(RunnerPredictionRow.race_id.in_(mut_race_ids_list))
                    .where(RunnerPredictionRow.exotic_model_rank >= 1)
                    .where(RunnerPredictionRow.exotic_model_rank <= 3)
                    .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                )).scalars().all())

            # Batch-fetch win model ranks 2–5 for hedge insurance calculations
            hedge_rank_rows: list = []
            if hist_race_ids:
                hedge_rank_rows.extend((await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id.in_(hist_race_ids))
                    .where(RunnerPredictionHistoryRow.model_rank.in_([2, 3, 4, 5]))
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                )).scalars().all())
            if mut_race_ids_list:
                hedge_rank_rows.extend((await session.execute(
                    select(RunnerPredictionRow)
                    .where(RunnerPredictionRow.race_id.in_(mut_race_ids_list))
                    .where(RunnerPredictionRow.model_rank.in_([2, 3, 4, 5]))
                    .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                )).scalars().all())

            # Field sizes (active runner count per race) for place divisor
            field_sizes: dict[str, int] = {}
            if hist_race_ids:
                field_sizes.update(dict((await session.execute(
                    select(RunnerPredictionHistoryRow.race_id, func.count(RunnerPredictionHistoryRow.id))
                    .where(RunnerPredictionHistoryRow.race_id.in_(hist_race_ids))
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                    .group_by(RunnerPredictionHistoryRow.race_id)
                )).fetchall()))
            if mut_race_ids_list:
                field_sizes.update(dict((await session.execute(
                    select(RunnerPredictionRow.race_id, func.count(RunnerPredictionRow.id))
                    .where(RunnerPredictionRow.race_id.in_(mut_race_ids_list))
                    .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                    .group_by(RunnerPredictionRow.race_id)
                )).fetchall()))

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

        # Build hedge lookup: race_id -> sorted list of rank-2/3 runners
        hedge_map: dict[str, list] = {}
        for hr in hedge_rank_rows:
            hedge_map.setdefault(hr.race_id, []).append(hr)
        for key in hedge_map:
            hedge_map[key].sort(key=lambda r: r.model_rank)

        # Fetch scheduled times per unique meeting in parallel
        unique_venues = {_parse_race_id(r.race_id)[1] for r in rows}
        slug_map = {v: _meeting_slug(v, target_date) for v in unique_venues}
        time_results = await asyncio.gather(*[_fetch_race_times(client, slug) for slug in slug_map.values()])
        race_times: dict[str, str | None] = {}  # race_id → startTime
        for venue, times in zip(slug_map.keys(), time_results):
            for race_num, start_time in times.items():
                race_times[f"{target_date}_{venue}_R{race_num}"] = start_time

        # For races where rank-2/3 hedge candidates have 0 odds, fetch live odds in parallel
        races_needing_live_odds = [
            r.race_id for r in rows
            if any((hr.best_available_odds or 0) <= 1.0 for hr in hedge_map.get(r.race_id, []))
        ]
        live_odds_results = await asyncio.gather(*[
            _fetch_live_odds(client, rid) for rid in races_needing_live_odds
        ])
        live_odds_by_race: dict[str, dict[str, float]] = dict(
            zip(races_needing_live_odds, live_odds_results)
        )

        for runner_row in rows:
            odds = runner_row.best_available_odds or hist_odds_override.get((runner_row.race_id, runner_row.horse_name), 0)
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

            # Insurance hedge: only computed when pick odds >= threshold
            hedge_runners = hedge_map.get(runner_row.race_id, [])
            live_race_odds = live_odds_by_race.get(runner_row.race_id, {})
            hedge_candidates = []
            for hr in hedge_runners:
                w = hr.best_available_odds or 0
                if w <= 1.0:
                    w = live_race_odds.get(hr.horse_name, 0)
                if w > 1.0:
                    hedge_candidates.append({
                        "horse_name": hr.horse_name,
                        "tab_number": hr.tab_number,
                        "win_odds": w,
                    })
            hedge = _compute_hedge(odds, field_sizes.get(runner_row.race_id, 8), hedge_candidates)

            picks.append({
                "date": target_date,
                "race_id": runner_row.race_id,
                "venue": venue_code,
                "state": None,
                "race_number": race_num,
                "race_name": None,
                "distance": None,
                "track_condition": None,
                "scheduled_time": race_times.get(runner_row.race_id) or runner_row.scheduled_time,
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
                "hedge": hedge,
            })

    # Annotate finished picks with actual race results
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    finished_picks = [
        p for p in picks
        if p["scheduled_time"] and datetime.fromisoformat(p["scheduled_time"].replace("Z", "+00:00")) < now_utc
    ]
    if finished_picks:
        finished_race_ids = list({p["race_id"] for p in finished_picks})
        db_positions = await _seed_race_results_on_demand(finished_race_ids)
        seeded_race_ids: set[str] = {rid for (rid, _) in db_positions}

        # Fetch SP for finished picks
        async with get_session() as session:
            sp_rows = (await session.execute(
                select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
                       HistoricalResultRow.starting_price)
                .where(HistoricalResultRow.race_id.in_(finished_race_ids))
            )).all()
        db_sp: dict[tuple, float | None] = {(r.race_id, r.horse_name): r.starting_price for r in sp_rows}

        # Annotate main pick with result
        for p in finished_picks:
            pos = db_positions.get((p["race_id"], p["horse_name"]))
            if pos is not None:
                p["actual_position"] = pos
                p["won"] = pos == 1
                p["placed"] = pos <= 3
                p["sp"] = db_sp.get((p["race_id"], p["horse_name"]))

        for p in finished_picks:
            tri = p.get("trifecta")
            if not tri:
                continue
            race_id = p["race_id"]

            def _annotate_legs(legs):
                return [{**l, "position": db_positions.get((race_id, l["horse_name"])), "scratched": False}
                        for l in legs]

            tri["legs"] = _annotate_legs(tri["legs"])
            if race_id in seeded_race_ids:
                tri_positions = {l["position"] for l in tri["legs"] if l["position"]}
                tri["hit"] = tri_positions == {1, 2, 3}

            if tri.get("first_four"):
                tri["first_four"] = _annotate_legs(tri["first_four"])
                if race_id in seeded_race_ids:
                    ff_positions = {l["position"] for l in tri["first_four"] if l["position"]}
                    tri["first_four_hit"] = ff_positions == {1, 2, 3, 4}

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "threshold_pct": int(threshold * 100),
        "picks": picks,
    }


_odds_refresh_last: datetime | None = None
_ODDS_REFRESH_COOLDOWN = 120  # seconds — prevents hammering RA/OddsPro

# Venue name aliases: our internal name → what TAB/OddsPro use
# Key is lowercased; value is the canonical name to search with
_VENUE_ALIASES: dict[str, str] = {
    "kensington": "randwick",  # Kensington track is the inner track at Royal Randwick
    "royal randwick": "randwick",
    "canberra": "thoroughbred park",
    "gold coast": "the gold coast",
}

async def _update_odds_from_oddspro(
    op: "OddsProClient",
    target_date: str,
    all_picks: list,
    updated: dict,
    diag: list,
) -> set[str]:
    """
    Fetch OddsPro odds for all picks on target_date.
    Returns set of race_ids that got odds so callers can skip them in fallback.
    """
    covered: set[str] = set()
    try:
        # One call gets all AU meetings with full odds (not just movers)
        meeting_odds = await op.get_meeting_odds(target_date)
    except Exception as e:
        diag.append({"date": target_date, "source": "oddspro", "error": f"get_meeting_odds: {e}"})
        return covered

    diag.append({"date": target_date, "source": "oddspro", "tracks_with_odds": list(meeting_odds.keys())})

    by_venue: dict[str, list] = {}
    for row in all_picks:
        venue_key = row.venue or _parse_race_id(row.race_id)[1]
        if venue_key:
            by_venue.setdefault(venue_key, []).append(row)

    for venue, rows in by_venue.items():
        venue_lower = venue.lower()
        alias_lower = _VENUE_ALIASES.get(venue_lower, venue_lower)
        # Match against OddsPro track keys (try exact, alias, then partial)
        track_key = None
        if venue_lower in meeting_odds:
            track_key = venue_lower
        elif alias_lower in meeting_odds:
            track_key = alias_lower
        else:
            for k in meeting_odds:
                if venue_lower in k or k in venue_lower or alias_lower in k or k in alias_lower:
                    track_key = k
                    break

        venue_diag: dict = {"date": target_date, "venue": venue, "source": "oddspro", "op_track": track_key}
        if not track_key:
            diag.append(venue_diag)
            continue

        odds_map = meeting_odds[track_key]
        venue_diag["runners_in_odds"] = len(odds_map)
        try:
            by_race: dict[int, list] = {}
            for row in rows:
                if row.race_number:
                    by_race.setdefault(row.race_number, []).append(row)
            async with get_session() as session:
                for race_num, race_rows in by_race.items():
                    new_odds_map: dict[int, float] = {}
                    for row in race_rows:
                        name_lower = row.horse_name.lower()
                        norm_name = _normalize_horse(row.horse_name)
                        val = odds_map.get((race_num, name_lower)) or odds_map.get((race_num, norm_name))
                        if val and val > 1.0:
                            new_odds_map[row.id] = val
                    sorted_ids = sorted(new_odds_map, key=lambda rid: new_odds_map[rid])
                    for row in race_rows:
                        new_o = new_odds_map.get(row.id)
                        if not new_o:
                            continue
                        db_row = await session.get(RunnerPredictionRow, row.id)
                        if not db_row:
                            continue
                        db_row.best_available_odds = new_o
                        market_implied = 1.0 / new_o
                        db_row.overlay = round(db_row.win_probability - market_implied, 4)
                        db_row.value_rating = _value_rating(db_row.win_probability, new_o, db_row.overlay)
                        db_row.market_rank = sorted_ids.index(row.id) + 1 if row.id in sorted_ids else db_row.market_rank
                        updated[row.race_id] = new_o
                        covered.add(row.race_id)
                await session.commit()
            diag.append(venue_diag)
        except Exception as e:
            venue_diag["error"] = str(e)
            diag.append(venue_diag)
            log.warning("refresh-odds OddsPro failed for %s: %s", venue, e)
    return covered


async def _update_odds_from_tab(
    target_date: str,
    all_picks: list,
    skip_race_ids: set[str],
    updated: dict,
    diag: list,
) -> None:
    """
    Fallback: fetch TAB fixed-win odds directly for picks not covered by OddsPro.
    Bypasses TABClient slug resolution — fetches meetings API per jurisdiction,
    matches venue by name, then fetches each race with the correct meetingCode.
    """
    import httpx as _httpx
    _TAB = "https://api.tab.com.au/v1/tab-info-service"
    _JURS = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
    _HDR = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}

    picks = [p for p in all_picks if p.race_id not in skip_race_ids]
    if not picks:
        return

    # Step 1: build venue_name_lower → {code, jur} from all jurisdictions
    meeting_map: dict[str, dict] = {}
    try:
        async with _httpx.AsyncClient(headers=_HDR, timeout=15) as client:
            tasks = [
                client.get(f"{_TAB}/racing/dates/{target_date}/meetings", params={"jurisdiction": j})
                for j in _JURS
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        for jur, resp in zip(_JURS, responses):
            if isinstance(resp, Exception) or resp.status_code != 200:
                continue
            for m in resp.json().get("meetings", []):
                if m.get("raceType") == "R" and m.get("meetingCode"):
                    name = (m.get("venueName") or "").lower().strip()
                    if name and name not in meeting_map:
                        meeting_map[name] = {"code": m["meetingCode"], "jur": jur}
    except Exception as e:
        diag.append({"date": target_date, "source": "tab", "error": f"meetings fetch: {e}"})
        return

    # Step 2: group picks by (meetingCode, jur, race_num)
    by_race: dict[tuple[str, str, int], list] = {}
    for row in picks:
        _, venue_code, race_num = _parse_race_id(row.race_id)
        if not race_num:
            continue
        venue_key = (row.venue or venue_code).lower().strip()
        alias_key = _VENUE_ALIASES.get(venue_key, venue_key)
        meeting = meeting_map.get(alias_key) or meeting_map.get(venue_key)
        if not meeting:
            for k, v in meeting_map.items():
                if venue_code in k or k in venue_code or alias_key in k or k in alias_key:
                    meeting = v
                    break
        if not meeting:
            diag.append({"date": target_date, "venue": venue_key, "tab_venues": list(meeting_map.keys())[:15], "source": "tab", "error": "no meeting match"})
            continue
        by_race.setdefault((meeting["code"], meeting["jur"], race_num), []).append(row)

    # Step 3: fetch each race and update odds
    async with _httpx.AsyncClient(headers=_HDR, timeout=15) as client:
        for (code, jur, race_num), race_rows in by_race.items():
            tab_diag: dict = {"date": target_date, "code": code, "race": race_num, "source": "tab", "jur": jur}
            try:
                resp = await client.get(
                    f"{_TAB}/racing/dates/{target_date}/meetings/{code}/R/races/{race_num}",
                    params={"jurisdiction": jur},
                )
                if resp.status_code != 200:
                    tab_diag["error"] = f"HTTP {resp.status_code}"
                    diag.append(tab_diag)
                    continue
                raw = resp.json()
                horse_odds: dict[str, float] = {}
                for runner in raw.get("runners", []):
                    name = (runner.get("runnerName") or "").upper()
                    if not name:
                        continue
                    for p in runner.get("prices", []):
                        if p.get("priceType") in ("FixedWin", "Win"):
                            val = float(p.get("winPrice") or 0)
                            if val > 1.0 and name not in horse_odds:
                                horse_odds[name] = val
                tab_diag["runners_with_odds"] = len(horse_odds)
                async with get_session() as session:
                    for row in race_rows:
                        new_o = horse_odds.get(row.horse_name.upper())
                        if not new_o:
                            continue
                        db_row = await session.get(RunnerPredictionRow, row.id)
                        if not db_row:
                            continue
                        db_row.best_available_odds = new_o
                        market_implied = 1.0 / new_o
                        db_row.overlay = round(db_row.win_probability - market_implied, 4)
                        db_row.value_rating = _value_rating(db_row.win_probability, new_o, db_row.overlay)
                        updated[row.race_id] = new_o
                    await session.commit()
            except Exception as e:
                tab_diag["error"] = str(e)
                log.warning("TAB odds failed for %s/%s R%s: %s", code, target_date, race_num, e)
            diag.append(tab_diag)


@app.post("/api/edge/refresh-odds")
async def refresh_edge_odds(force: bool = False):
    """
    Fetch odds for all upcoming edge picks — today + next 3 days.
    Primary: OddsPro movers. Fallback: TAB API (all runners, no auth).
    Rate-limited to once per 2 minutes globally. Pass force=true to bypass.
    """
    from horse_engine.clients.oddspro import OddsProClient
    global _odds_refresh_last
    now = datetime.utcnow()
    if not force and _odds_refresh_last and (now - _odds_refresh_last).total_seconds() < _ODDS_REFRESH_COOLDOWN:
        return {"updated": {}, "count": 0, "cached": True}

    threshold = 0.295
    today = _today_aest()
    op = OddsProClient()
    updated: dict[str, float] = {}
    diag: list[dict] = []

    for i in range(4):  # today + next 3 days
        target_date = (today + timedelta(days=i)).isoformat()
        prefix = f"{target_date}_"

        async with get_session() as session:
            settled_ids = set((await session.execute(
                select(HistoricalResultRow.race_id)
                .where(HistoricalResultRow.race_id.like(f"{prefix}%"))
                .distinct()
            )).scalars().all())
            q = (
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.model_rank == 1)
                .where(RunnerPredictionRow.win_probability >= threshold)
                .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            )
            if settled_ids:
                q = q.where(RunnerPredictionRow.race_id.notin_(settled_ids))
            result = await session.execute(q)
            all_picks = result.scalars().all()
            all_picks = [p for p in all_picks if not re.search(r"-(trial|trail|jumpout)s?[_-]", p.race_id, re.IGNORECASE)]

        if not all_picks:
            continue

        covered = await _update_odds_from_oddspro(op, target_date, all_picks, updated, diag)
        await _update_odds_from_tab(target_date, all_picks, covered, updated, diag)

    _odds_refresh_last = now
    return {"updated": updated, "count": len(updated), "debug": diag}


_results_refresh_last: datetime | None = None
_results_refresh_cooldown: float = 100.0  # refreshed each call with jitter

@app.post("/api/edge/refresh-results")
async def refresh_edge_results():
    """
    Seed today's settled results on demand. Rate-limited globally with random jitter
    (100–130s) so TAB sees one semi-regular user, not a clock-perfect bot.
    Only runs during race hours 12pm–8pm AEST. Returns cached=True outside that window.
    """
    global _results_refresh_last, _results_refresh_cooldown
    now_aest = datetime.now(_AEST)
    # Only run during race hours
    if not (12 <= now_aest.hour < 20):
        return {"seeded": 0, "cached": True, "reason": "outside race hours"}

    now_utc = datetime.utcnow()
    if _results_refresh_last and (now_utc - _results_refresh_last).total_seconds() < _results_refresh_cooldown:
        return {"seeded": 0, "cached": True}

    # Refresh cooldown: 100s base + random 0–30s jitter
    _results_refresh_cooldown = 100.0 + random.uniform(0, 30)
    _results_refresh_last = now_utc

    today = _today_aest().isoformat()
    try:
        n = await _seed_results_for_date(today)
        log.info("[refresh-results] Seeded %d results for %s", n, today)
        return {"seeded": n, "cached": False}
    except Exception as e:
        log.warning("[refresh-results] Failed: %s", e)
        return {"seeded": 0, "cached": False, "error": str(e)}


@app.get("/api/edge/yesterday")
async def get_edge_yesterday(for_date: Optional[str] = Query(None, alias="date")):
    """Qualifying picks with actual results and SP odds from Racing Australia.
    Accepts ?date=YYYY-MM-DD (defaults to yesterday)."""
    target_date = for_date or (_today_aest() - timedelta(days=1)).isoformat()
    threshold = 0.295
    prefix = f"{target_date}_"
    stake = 10

    async with get_session() as session:
        # cancelled filter (BUG-25) + source="live" filter (BUG-26) — exclude
        # scratched horses (whose result lookup would silently drop the race)
        # and validation-backtest rows (so retroactive scores don't surface
        # alongside genuine pre-race picks).
        result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.win_probability >= threshold)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{prefix}%"))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.win_probability.desc())
        )
        picks = result.scalars().all()
        picks = [p for p in picks if not re.search(r"-(trial|trail|jumpout)s?[_-]", p.race_id, re.IGNORECASE)]

        if not picks:
            return {"date": target_date, "picks": [], "summary": None}

        # Batch-fetch place model runners for trifecta legs — same filters so
        # scratched horses or validation rows don't appear as trifecta legs.
        yst_race_ids = [p.race_id for p in picks]
        yst_place_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(yst_race_ids))
            .where(RunnerPredictionHistoryRow.place_model_rank >= 1)
            .where(RunnerPredictionHistoryRow.place_model_rank <= 4)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )
        yst_place_rows = yst_place_result.scalars().all()

    yst_trifecta_map: dict[str, list] = {}
    for pr in yst_place_rows:
        if pr.place_model_rank:
            yst_trifecta_map.setdefault(pr.race_id, []).append(pr)
    for key in yst_trifecta_map:
        yst_trifecta_map[key].sort(key=lambda r: r.place_model_rank)

    # Seed any missing results on first load, then query DB
    all_race_ids = list({p.race_id for p in picks} | {pr.race_id for pr in yst_place_rows})
    await _seed_race_results_on_demand(all_race_ids)
    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.in_(all_race_ids))
        )
        hr_rows = hr_result.scalars().all()

    # Key: (venue_code, race_num, horse_name) → result dict
    all_results: dict = {}
    seeded_race_ids: set[str] = {hr.race_id for hr in hr_rows}
    for hr in hr_rows:
        _, venue_code, race_num = _parse_race_id(hr.race_id)
        pos = hr.position
        all_results[(venue_code, race_num, _normalize_horse(hr.horse_name))] = {
            "position": pos,
            "sp": hr.starting_price,
            "winner": pos == 1,
            "placed": bool(pos and pos <= 3),
            "scratched": False,
        }

    output = []
    for p in picks:
        _, venue_code, race_num = _parse_race_id(p.race_id)
        r = all_results.get((venue_code, race_num, _normalize_horse(p.horse_name)), {})
        sp = r.get("sp") or p.best_available_odds or None
        winner = r.get("winner", False)
        position = r.get("position")
        # Race seeded but horse absent → scratched (HistoricalResultRow skips scratched runners)
        scratched = bool(p.race_id in seeded_race_ids and not r)
        # Race not seeded at all → result unavailable (RA had no data; don't show as Unplaced)
        no_result = bool(p.race_id not in seeded_race_ids and not r)
        model_pct = round(p.win_probability * 100, 1)
        payout = round(sp * stake, 2) if winner and sp else 0
        profit = 0 if (scratched or no_result) else (round(payout - stake, 2) if winner and sp else -stake)

        placed = r.get("placed", False) or bool(position and position <= 3 and not scratched)

        # Find the actual race winner when our pick didn't win
        winner_name = None
        if not winner and not scratched:
            for (v, rn, name), res in all_results.items():
                if v == venue_code and rn == race_num and res.get("winner"):
                    winner_name = name.title()
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
            res = all_results.get((venue_code, race_num, _normalize_horse(leg["horse_name"])), {})
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
            "no_result": no_result,
            "payout": payout,
            "profit": profit,
            "stake": stake,
            "trifecta": yst_trifecta,
        })

    active = [o for o in output if not o["scratched"] and not o["no_result"]]
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
            # Identify settled races for this date (have historical results)
            settled_q = await session.execute(
                select(HistoricalResultRow.race_id)
                .where(HistoricalResultRow.race_id.like(f"{prefix}%"))
                .distinct()
            )
            settled_race_ids: set[str] = {r for (r,) in settled_q.fetchall()}

            # Exotic/place model legs — history for settled, mutable for upcoming
            exotic_rows: list = []
            using_fallback = False

            def _exotic_q_hist(rank_col, lo, hi):
                q = (select(RunnerPredictionHistoryRow)
                     .where(RunnerPredictionHistoryRow.race_id.in_(settled_race_ids))
                     .where(rank_col >= lo).where(rank_col <= hi)
                     .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None)))
                return q

            def _exotic_q_mut(race_filter, rank_col, lo, hi):
                q = (select(RunnerPredictionRow)
                     .where(race_filter)
                     .where(rank_col >= lo).where(rank_col <= hi)
                     .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None)))
                return q

            mut_filter = RunnerPredictionRow.race_id.like(f"{prefix}%")
            if settled_race_ids:
                mut_filter = mut_filter & ~RunnerPredictionRow.race_id.in_(settled_race_ids)

            # Try exotic_model_rank first
            if settled_race_ids:
                exotic_rows.extend((await session.execute(
                    _exotic_q_hist(RunnerPredictionHistoryRow.exotic_model_rank, 1, 4)
                )).scalars().all())
            exotic_rows.extend((await session.execute(
                _exotic_q_mut(mut_filter, RunnerPredictionRow.exotic_model_rank, 1, 4)
            )).scalars().all())

            if not exotic_rows:
                # Fall back to place_model_rank until exotic model is trained
                if settled_race_ids:
                    exotic_rows.extend((await session.execute(
                        _exotic_q_hist(RunnerPredictionHistoryRow.place_model_rank, 1, 4)
                    )).scalars().all())
                exotic_rows.extend((await session.execute(
                    _exotic_q_mut(mut_filter, RunnerPredictionRow.place_model_rank, 1, 4)
                )).scalars().all())
                using_fallback = True

            exotic_rows = [r for r in exotic_rows if not re.search(r"-(trial|trail|jumpout)s?[_-]", r.race_id, re.IGNORECASE)]
            if not exotic_rows:
                continue

            race_ids = list({r.race_id for r in exotic_rows})
            hist_exotic_ids = {r.race_id for r in exotic_rows if r.race_id in settled_race_ids}
            mut_exotic_ids = [r for r in race_ids if r not in settled_race_ids]

            race_result = await session.execute(
                select(RacePredictionRow).where(RacePredictionRow.race_id.in_(race_ids))
            )
            race_lookup = {r.race_id: r for r in race_result.scalars().all()}

            # Field sizes — history for settled, mutable for upcoming
            field_size_lookup: dict[str, int] = {}
            if hist_exotic_ids:
                field_size_lookup.update(dict((await session.execute(
                    select(RunnerPredictionHistoryRow.race_id, func.count(RunnerPredictionHistoryRow.id))
                    .where(RunnerPredictionHistoryRow.race_id.in_(hist_exotic_ids))
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                    .group_by(RunnerPredictionHistoryRow.race_id)
                )).fetchall()))
            if mut_exotic_ids:
                field_size_lookup.update(dict((await session.execute(
                    select(RunnerPredictionRow.race_id, func.count(RunnerPredictionRow.id))
                    .where(RunnerPredictionRow.race_id.in_(mut_exotic_ids))
                    .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                    .group_by(RunnerPredictionRow.race_id)
                )).fetchall()))

            # Win picks — history for settled (threshold on pre-race prob), mutable for upcoming
            win_lookup: dict[str, str] = {}
            if hist_exotic_ids:
                seen: set[str] = set()
                for p in (await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id.in_(hist_exotic_ids))
                    .where(RunnerPredictionHistoryRow.model_rank == 1)
                    .where(RunnerPredictionHistoryRow.win_probability >= 0.295)
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                    .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
                )).scalars().all():
                    if p.race_id not in seen:
                        seen.add(p.race_id)
                        win_lookup[p.race_id] = p.horse_name
            if mut_exotic_ids:
                for p in (await session.execute(
                    select(RunnerPredictionRow)
                    .where(RunnerPredictionRow.race_id.in_(mut_exotic_ids))
                    .where(RunnerPredictionRow.model_rank == 1)
                    .where(RunnerPredictionRow.win_probability >= 0.295)
                    .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                )).scalars().all():
                    win_lookup[p.race_id] = p.horse_name

        # Group by race, sort by exotic_model_rank (or place_model_rank fallback)
        rank_key = (lambda r: r.place_model_rank or 99) if using_fallback else (lambda r: r.exotic_model_rank or 99)
        race_map: dict[str, list] = {}
        for row in exotic_rows:
            race_map.setdefault(row.race_id, []).append(row)

        # Race times come from DB (RunnerPredictionRow.scheduled_time) — no RA call needed

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
                "scheduled_time": runners[0].scheduled_time if runners else None,
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

    # Annotate finished races from DB — HistoricalResultRow seeded at 3/5/11pm
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    finished_picks = [
        p for p in picks
        if p["scheduled_time"] and datetime.fromisoformat(p["scheduled_time"].replace("Z", "+00:00")) < now_utc
    ]
    if finished_picks:
        finished_race_ids = list({p["race_id"] for p in finished_picks})
        # Fetch from DB; on first load after a race finishes, seeds missing results from RA once
        db_positions = await _seed_race_results_on_demand(finished_race_ids)
        seeded_race_ids: set[str] = {rid for (rid, _) in db_positions}

        for p in finished_picks:
            race_id = p["race_id"]

            def _annotate(legs):
                return [{**l, "position": db_positions.get((race_id, l["horse_name"])), "scratched": False}
                        for l in legs]

            p["legs"] = _annotate(p["legs"])
            if race_id in seeded_race_ids:
                tri_positions = {l["position"] for l in p["legs"] if l["position"]}
                p["hit"] = tri_positions == {1, 2, 3}

            if p.get("first_four"):
                p["first_four"] = _annotate(p["first_four"])
                if race_id in seeded_race_ids:
                    ff_positions = {l["position"] for l in p["first_four"] if l["position"]}
                    p["first_four_hit"] = ff_positions == {1, 2, 3, 4}

    return {"generated_at": datetime.utcnow().isoformat(), "picks": picks}


@app.get("/api/track-record")
async def get_track_record():
    """Public endpoint — tier win rates from 30-day backtest."""
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_map = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_result.scalars().all()}

        # History table — written once pre-race, never overwritten by re-enrichments.
        # Filter cancelled (BUG-31) so scratched horses don't surface as the tier
        # pick; exclude validation-backtest rows; dedup-in-Python on latest
        # enriched_at to avoid the BUG-09 exact-timestamp drop.
        hist_picks = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= cutoff)
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.win_probability.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).scalars().all()

        top_picks: dict[str, RunnerPredictionHistoryRow] = {}
        for r in hist_picks:
            if r.race_id not in top_picks:
                top_picks[r.race_id] = r

        all_rows = []
        for r in top_picks.values():
            hr = hr_map.get((r.race_id, _normalize_horse(r.horse_name)))
            if hr:
                all_rows.append({
                    "win_prob": r.win_probability,
                    "winner": hr.position == 1,
                })

    tiers = [
        {"badge": "hot",      "min": 0.45, "max": 1.0,  "conf_min": 45, "conf_max": None},
        {"badge": "high",     "min": 0.35, "max": 0.45, "conf_min": 35, "conf_max": 45},
        {"badge": "standard", "min": 0.30, "max": 0.35, "conf_min": 30, "conf_max": 35},
    ]
    output = []
    for tier in tiers:
        picks = [r for r in all_rows if tier["min"] <= r["win_prob"] < tier["max"]]
        wins  = [r for r in picks if r["winner"]]
        win_pct = round(len(wins) / len(picks) * 100) if picks else 0
        output.append({
            "badge":    tier["badge"],
            "win_pct":  win_pct,
            "races":    len(picks),
            "conf_min": tier["conf_min"],
            "conf_max": tier["conf_max"],
        })
    return {
        "tiers": output,
        "total_races": len(all_rows),
        "generated_at": datetime.utcnow().isoformat(),
    }


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
    cached = _list_meetings_cache.get(race_date)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < 600:
        return cached[1]
    client = get_tab_client()
    try:
        meetings = await client.get_meetings(race_date)
    except Exception as e:
        log.exception("list_meetings failed for %s", race_date)
        meetings = []

    from horse_engine.clients.weather import get_weather_for_venue

    date_suffix = f"-{race_date.replace('-', '')}"
    items = []
    for m in meetings:
        slug = m.get("slug", "")
        if slug and slug.endswith(date_suffix):
            vc = slug[: -len(date_suffix)]
        elif slug:
            vc = slug.split("-")[0]
        else:
            vc = m.get("name", "").lower().replace(" ", "-")
        items.append({
            "venue": m.get("venue") or vc.replace("-", " ").title(),
            "venue_code": vc,
            "state": m.get("state"),
            "rail_position": m.get("rail_position"),
            "slug": slug,
        })

    # Merge DB-enriched meetings — ensures venues still appear when RA is blocked
    active_codes = {it["venue_code"] for it in items}
    async with get_session() as session:
        # RacePredictionRow has venue/state metadata from enrichment
        rp_meta = (await session.execute(
            select(RacePredictionRow.race_id, RacePredictionRow.venue, RacePredictionRow.state)
            .where(RacePredictionRow.race_id.like(f"{race_date}_%"))
        )).all()
        # RunnerPredictionRow covers future dates enriched before RacePredictionRow was written
        runner_race_ids = (await session.execute(
            select(RunnerPredictionRow.race_id)
            .where(RunnerPredictionRow.race_id.like(f"{race_date}_%"))
            .where(RunnerPredictionRow.model_rank == 1)
            .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            .distinct()
        )).scalars().all()

    seen_vc: set[str] = set()
    # Build venue→metadata map from RacePredictionRow first
    rp_venue_meta: dict[str, tuple] = {}
    for row in rp_meta:
        _, vc, _ = _parse_race_id(row.race_id)
        if vc:
            rp_venue_meta.setdefault(vc, (row.venue, row.state))

    # Fallback venue→state for common AU venues when state wasn't stored
    _VENUE_STATE_FALLBACK: dict[str, str] = {
        "wellington": "NSW", "hamilton": "VIC", "albury": "NSW", "wagga": "NSW",
        "tamworth": "NSW", "dubbo": "NSW", "orange": "NSW", "bathurst": "NSW",
        "canberra": "ACT", "goulburn": "NSW", "nowra": "NSW", "grafton": "NSW",
        "lismore": "NSW", "armidale": "NSW", "mudgee": "NSW", "moruya": "NSW",
        "eagle-farm": "QLD", "doomben": "QLD", "ipswich": "QLD", "sunshine-coast": "QLD",
        "toowoomba": "QLD", "gold-coast": "QLD", "rockhampton": "QLD", "cairns": "QLD",
        "townsville": "QLD", "mackay": "QLD", "emerald": "QLD", "longreach": "QLD",
        "flemington": "VIC", "caulfield": "VIC", "moonee-valley": "VIC", "sandown": "VIC",
        "mornington": "VIC", "geelong": "VIC", "bendigo": "VIC", "ballarat": "VIC",
        "echuca": "VIC", "sale": "VIC", "warrnambool": "VIC", "wodonga": "VIC",
        "morphettville": "SA", "murray-bridge": "SA", "gawler": "SA", "port-augusta": "SA",
        "ascot": "WA", "belmont": "WA", "bunbury": "WA", "kalgoorlie": "WA",
        "randwick": "NSW", "rosehill": "NSW", "warwick-farm": "NSW", "hawkesbury": "NSW",
        "gosford": "NSW", "newcastle": "NSW", "kembla-grange": "NSW",
        "darwin": "NT", "alice-springs": "NT",
        "launceston": "TAS", "hobart": "TAS", "devonport": "TAS",
    }

    # Union both sources
    _TRIAL_VC_RE = re.compile(r"(trial|trail|jumpout)s?", re.IGNORECASE)
    all_db_vcs = {vc for vc in rp_venue_meta.keys() if not _TRIAL_VC_RE.search(vc)}
    for rid in runner_race_ids:
        _, vc, _ = _parse_race_id(rid)
        if vc and not _TRIAL_VC_RE.search(vc):
            all_db_vcs.add(vc)

    for vc in all_db_vcs:
        if vc not in active_codes and vc not in seen_vc:
            seen_vc.add(vc)
            venue_name, state = rp_venue_meta.get(vc, (None, None))
            # Resolve state from fallback table when not stored in DB
            if not state:
                state = _VENUE_STATE_FALLBACK.get(vc)
            items.append({
                "venue": venue_name or vc.replace("-", " ").title(),
                "venue_code": vc,
                "state": state,
                "rail_position": None,
                "slug": None,
            })
    if seen_vc:
        log.info("list_meetings: added %d DB-only venues for %s: %s", len(seen_vc), race_date, seen_vc)

    # Add any DB-mass-cancelled meetings that are no longer on RA.
    # BUG-35: previously identified "cancelled venues" by "rank-1 mutable row is
    # cancelled", which fires false-positives when a single scratch happens to
    # hit the rank-1 horse. Now uses an aggregate over the venue's full field —
    # a venue is mass-cancelled only when EVERY runner across all its races for
    # the date is cancelled.
    active_codes = {it["venue_code"] for it in items}
    async with get_session() as session:
        pred_status_rows = (await session.execute(
            select(RunnerPredictionRow.race_id, RunnerPredictionRow.cancelled)
            .where(RunnerPredictionRow.race_id.like(f"{race_date}_%"))
        )).all()

        # Venues with actual results in HistoricalResultRow are NOT cancelled —
        # the enrichment cron sometimes marks races abandoned prematurely.
        resulted_race_ids = (await session.execute(
            select(HistoricalResultRow.race_id)
            .where(HistoricalResultRow.race_id.like(f"{race_date}_%"))
            .distinct()
        )).scalars().all()

    resulted_vcs = set()
    for rid in resulted_race_ids:
        _, vc, _ = _parse_race_id(rid)
        if vc:
            resulted_vcs.add(vc)

    # Aggregate cancellation state per venue.
    venue_totals: dict[str, dict[str, int]] = {}
    for race_id, cancelled in pred_status_rows:
        _, vc, _ = _parse_race_id(race_id)
        if not vc:
            continue
        bucket = venue_totals.setdefault(vc, {"total": 0, "cancelled": 0})
        bucket["total"] += 1
        if cancelled:
            bucket["cancelled"] += 1

    fully_cancelled_vcs = {
        vc for vc, b in venue_totals.items()
        if b["total"] > 0 and b["cancelled"] == b["total"]
    }

    seen_cancelled: set[str] = set()
    for vc in fully_cancelled_vcs:
        if vc not in active_codes and vc not in seen_cancelled:
            seen_cancelled.add(vc)
            venue_name_c, state_c = rp_venue_meta.get(vc, (None, None))
            if not state_c:
                state_c = _VENUE_STATE_FALLBACK.get(vc)
            # If HistoricalResultRow has results for this venue, it actually ran
            actually_cancelled = vc not in resulted_vcs
            items.append({
                "venue": venue_name_c or vc.replace("-", " ").title(),
                "venue_code": vc,
                "state": state_c,
                "rail_position": None,
                "slug": None,
                "cancelled": actually_cancelled,
                "weather": None,
            })

    weathers = await asyncio.gather(
        *[get_weather_for_venue(it["venue"] or "", it["state"] or "", race_date) for it in items if not it.get("cancelled")],
        return_exceptions=True,
    )
    live_items = [it for it in items if not it.get("cancelled")]
    for it, w in zip(live_items, weathers):
        it["weather"] = w if isinstance(w, dict) else None

    result = {"date": race_date, "meetings": items}
    _list_meetings_cache[race_date] = (datetime.utcnow(), result)
    return result


@app.get("/api/meetings/{race_date}/{venue_code}")
async def get_meeting(race_date: str, venue_code: str):
    _validate_date(race_date)
    _validate_venue(venue_code)
    """Get all races at a meeting with current predictions if available."""
    _cache_key = f"{race_date}/{venue_code}"
    _cached = _get_meeting_cache.get(_cache_key)
    if _cached and (datetime.utcnow() - _cached[0]).total_seconds() < 30:
        return _cached[1]
    prefix = f"{_like_safe(race_date)}_{_like_safe(venue_code)}_R"

    # ── Step 1: build race list from DB (always available) ───────────────────
    async with get_session() as session:
        rp_result = await session.execute(
            select(RacePredictionRow)
            .where(RacePredictionRow.race_id.like(f"{prefix}%"))
            .order_by(RacePredictionRow.race_number)
        )
        rp_rows = {r.race_id: r for r in rp_result.scalars().all()}

        # Fall back to RunnerPredictionRow race_ids if RacePredictionRow is empty
        if not rp_rows:
            db_result = await session.execute(
                select(
                    RunnerPredictionRow.race_id,
                    RunnerPredictionRow.scheduled_time,
                    RunnerPredictionRow.race_name,
                    RunnerPredictionRow.distance,
                    RunnerPredictionRow.track_condition,
                )
                .where(RunnerPredictionRow.race_id.like(f"{prefix}%"))
                .where(RunnerPredictionRow.model_rank == 1)
                .order_by(RunnerPredictionRow.race_id)
            )
            fallback_rows = {row[0]: row for row in db_result}
            fallback_ids = list(fallback_rows.keys())
            if not fallback_ids:
                hr_result = await session.execute(
                    select(HistoricalResultRow.race_id)
                    .where(HistoricalResultRow.race_id.like(f"{prefix}%"))
                    .distinct().order_by(HistoricalResultRow.race_id)
                )
                fallback_ids = [row[0] for row in hr_result]
                fallback_rows = {}
            for rid in fallback_ids:
                try:
                    rnum = int(rid.split("_R")[-1])
                except ValueError:
                    continue
                rp_rows[rid] = fallback_rows.get(rid)  # row tuple or None

    race_list = []
    for race_id, rp in rp_rows.items():
        try:
            rnum = int(race_id.split("_R")[-1])
        except ValueError:
            continue
        # rp is a RacePredictionRow ORM object, a RunnerPredictionRow Row tuple, or None
        sched      = getattr(rp, "scheduled_time", None) if rp is not None else None
        race_name  = getattr(rp, "race_name", None)      if rp is not None else None
        dist       = getattr(rp, "distance", None)       if rp is not None else None
        tc         = getattr(rp, "track_condition", None) if rp is not None else None
        field_size = getattr(rp, "field_size", None)     if rp is not None else None
        prize_money = getattr(rp, "prize_money", None)   if rp is not None else None
        race_list.append({
            "race_id": race_id,
            "race_number": rnum,
            "race_name": race_name,
            "distance": dist,
            "scheduled_time": sched,
            "time": sched,
            "status": None,  # filled by RA below if available
            "track_condition": tc,
            "field_size": field_size,
            "prize_money": prize_money,
        })
    race_list.sort(key=lambda r: r["race_number"])

    # ── Step 2: top-up with live RA data (best-effort) ───────────────────────
    # Adds: live status for open/closed races + any races not yet enriched
    ra_times: dict[str, str] = {}  # race_id → startTime, for DB back-fill
    try:
        client = get_tab_client()
        slug = _meeting_slug(venue_code, race_date)
        raw_races = await asyncio.wait_for(client.get_meeting_races(slug), timeout=25)
        ra_by_num = {r.get("eventNumber"): r for r in raw_races}
        existing_nums = {r["race_number"] for r in race_list}
        for r_num, r in ra_by_num.items():
            race_id = f"{race_date}_{venue_code}_R{r_num}"
            start_time = r.get("startTime")
            if start_time:
                ra_times[race_id] = start_time
            if r_num in existing_nums:
                for item in race_list:
                    if item["race_number"] == r_num:
                        item["status"] = r.get("status")
                        if not item["race_name"]:   item["race_name"]   = r.get("name")
                        if not item["distance"]:    item["distance"]    = r.get("distance")
                        if not item["scheduled_time"]: item["scheduled_time"] = start_time
                        if not item["time"]:        item["time"]        = start_time
                        break
            else:
                race_list.append({
                    "race_id": race_id,
                    "race_number": r_num,
                    "race_name": r.get("name"),
                    "distance": r.get("distance"),
                    "scheduled_time": start_time,
                    "time": start_time,
                    "status": r.get("status"),
                    "track_condition": None,
                    "field_size": None,
                    "prize_money": None,
                })
        race_list.sort(key=lambda r: r["race_number"])
    except Exception as e:
        log.warning("[get_meeting] RA fallback failed for %s/%s: %s", venue_code, race_date, e)

    race_ids = [r["race_id"] for r in race_list]

    # Seed resulted races before the main DB read so winners are available immediately
    resulted_race_ids = {r["race_id"] for r in race_list if r.get("status") == "resulted"}
    if resulted_race_ids:
        await _seed_race_results_on_demand(list(resulted_race_ids))

    async with get_session() as session:
        # Which races have been enriched
        enriched_result = await session.execute(
            select(RunnerPredictionRow.race_id, RunnerPredictionRow.enriched_at)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )
        enriched_rows = {row.race_id: row.enriched_at for row in enriched_result}

        # Back-fill scheduled_time into DB for enriched races that were missing it
        if ra_times:
            from sqlalchemy import update as sa_update
            missing_time_ids = [rid for rid in enriched_rows if rid in ra_times]
            if missing_time_ids:
                for rid in missing_time_ids:
                    await session.execute(
                        sa_update(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == rid)
                        .where(RunnerPredictionRow.scheduled_time.is_(None))
                        .values(scheduled_time=ra_times[rid])
                    )
                await session.commit()
                log.info("[get_meeting] Back-filled scheduled_time for %d races at %s/%s",
                         len(missing_time_ids), venue_code, race_date)

        # Winners and placers — loaded first so completed races are known before
        # we decide which prediction table to consult.
        hr_all = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.in_(race_ids))
            .where(HistoricalResultRow.position.isnot(None))
            .order_by(HistoricalResultRow.id)
        )
        winners: dict[str, str] = {}
        placers: dict[str, set] = {}
        for r in hr_all.scalars().all():
            if r.position == 1:
                winners[r.race_id] = r.horse_name
            if r.position <= 3:
                placers.setdefault(r.race_id, set()).add(r.horse_name)
        log.info("[get_meeting] position-based winners for %s/%s: %s", venue_code, race_date, winners)

        # Completed = HistoricalResultRow exists for the race (Ground Rule 5).
        # Completed races read from the immutable history snapshot so the displayed
        # top pick matches the pick used for model_correct / result banners. Upcoming
        # races read from mutable so post-9am enrichments (fresh form, scratches,
        # updated odds) are visible. Mutable is protected from post-race contamination
        # by the history_exists + scheduled_time guard in save_race_predictions.
        settled_ids_q = await session.execute(
            select(HistoricalResultRow.race_id)
            .where(HistoricalResultRow.race_id.in_(race_ids))
            .distinct()
        )
        settled_ids: set[str] = set(settled_ids_q.scalars().all())

        completed_ids = set(winners.keys()) | settled_ids

        # Top pick per race:
        # - Completed races → history table only (written once pre-race; re-enrichments
        #   never overwrite it, so it always holds the genuine pre-race prediction).
        # - Upcoming races  → mutable table (reflects latest retrain/enrichment).
        top_picks = {race_id: None for race_id in race_ids}
        top_win_probs = {race_id: None for race_id in race_ids}
        top_place_probs = {race_id: None for race_id in race_ids}

        if completed_ids:
            hist_tp_result = await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.race_id.in_(completed_ids))
                .where(RunnerPredictionHistoryRow.model_rank == 1)
                .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
            )
            seen_races: set[str] = set()
            for p in hist_tp_result.scalars().all():
                if p.race_id not in seen_races:
                    seen_races.add(p.race_id)
                    top_picks[p.race_id] = p.horse_name
                    top_win_probs[p.race_id] = p.win_probability
                    top_place_probs[p.race_id] = p.place_probability

        upcoming_ids = [rid for rid in race_ids if rid not in completed_ids]
        if upcoming_ids:
            tp_result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id.in_(upcoming_ids))
                .where(RunnerPredictionRow.model_rank == 1)
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            )
            for p in tp_result.scalars().all():
                top_picks[p.race_id] = p.horse_name
                top_win_probs[p.race_id] = p.win_probability
                top_place_probs[p.race_id] = p.place_probability

    enriched = bool(enriched_rows)

    def _model_correct(race_id: str):
        pick = top_picks.get(race_id)
        winner = winners.get(race_id)
        if not pick or not winner:
            return None
        return _normalize_horse(pick) == _normalize_horse(winner)

    def _model_placed(race_id: str):
        pick = top_picks.get(race_id)
        race_placers = placers.get(race_id)
        if not pick or not race_placers:
            return None
        return _normalize_horse(pick) in {_normalize_horse(h) for h in race_placers}

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

    result = {
        "date": race_date,
        "venue": venue_code,
        "enriched": enriched,
        "races": races_out,
    }
    _get_meeting_cache[_cache_key] = (datetime.utcnow(), result)
    return result


# ── Races ─────────────────────────────────────────────────────────────────────

@app.get("/api/races/{race_id}")
async def get_race(race_id: str):
    """Return full race prediction for a given race_id."""
    async with get_session() as session:
        # Settled = HistoricalResultRow exists (Ground Rule 5). Settled races read from
        # the immutable history snapshot so the rank-1 shown on the card matches the
        # pick used for model_correct / result banners. Unsettled races read from
        # mutable so post-9am enrichments (fresh form, scratches, updated odds) are
        # visible. Mutable is protected from post-race contamination by the
        # history_exists + scheduled_time guard in save_race_predictions.
        settled = (await session.execute(
            select(HistoricalResultRow.race_id)
            .where(HistoricalResultRow.race_id == race_id)
            .limit(1)
        )).scalar() is not None

        if settled:
            max_at = (await session.execute(
                select(func.max(RunnerPredictionHistoryRow.enriched_at))
                .where(RunnerPredictionHistoryRow.race_id == race_id)
            )).scalar()
            if max_at:
                hist_result = await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id == race_id)
                    .where(RunnerPredictionHistoryRow.enriched_at == max_at)
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                    .order_by(RunnerPredictionHistoryRow.model_rank)
                )
                runners = hist_result.scalars().all()
            else:
                runners = []
        else:
            mutable_result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id == race_id)
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
                .order_by(RunnerPredictionRow.model_rank)
            )
            runners = mutable_result.scalars().all()

            if not runners:
                hist_result = await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id == race_id)
                    .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                    .order_by(RunnerPredictionHistoryRow.model_rank)
                )
                runners = hist_result.scalars().all()

    if not runners:
        raise HTTPException(404, f"No predictions for race {race_id}. Trigger /enrich first.")

    # Last 10 + most recent run date from historical_results.
    # race_id format "YYYY-MM-DD_venue_RN" sorts chronologically, so the first
    # result per horse is the most recent tracked run.
    horse_names = [r.horse_name for r in runners]
    two_years_ago = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    async with get_session() as session:
        hist_rows = (await session.execute(
            select(HistoricalResultRow.horse_name, HistoricalResultRow.winner,
                   HistoricalResultRow.placed, HistoricalResultRow.race_id)
            .where(HistoricalResultRow.horse_name.in_(horse_names))
            .where(HistoricalResultRow.race_id != race_id)
            .where(HistoricalResultRow.race_id >= two_years_ago)
            .order_by(HistoricalResultRow.race_id.desc())
        )).all()

    hist_by_horse: dict[str, list[tuple[bool, bool]]] = {}
    hist_last_run: dict[str, int] = {}  # horse_name → days since last tracked run
    for horse_name, winner, placed, hist_race_id in hist_rows:
        bucket = hist_by_horse.setdefault(horse_name, [])
        if len(bucket) < 10:
            bucket.append((bool(winner), bool(placed)))
        if horse_name not in hist_last_run:
            run_date_str = hist_race_id[:10]  # "YYYY-MM-DD"
            try:
                run_date = datetime.strptime(run_date_str, "%Y-%m-%d").date()
                hist_last_run[horse_name] = (datetime.strptime(today_str, "%Y-%m-%d").date() - run_date).days
            except ValueError:
                pass

    last10 = {
        name: {
            "wins_last_10": sum(1 for w, _ in starts if w),
            "places_last_10": sum(1 for w, p in starts if w or p),
            "starts_last_10": len(starts),
            "days_since_last_run_hist": hist_last_run.get(name),
        }
        for name, starts in hist_by_horse.items()
    }
    # Also include hist_last_run for horses that have history but no last10 bucket
    for name, days in hist_last_run.items():
        if name not in last10:
            last10[name] = {"days_since_last_run_hist": days}

    return {
        "race_id": race_id,
        "runners": [_runner_response(r, last10.get(r.horse_name)) for r in runners],
    }


@app.get("/api/races/{race_id}/live-odds")
async def live_odds(race_id: str):
    """
    Re-fetch current tote odds from Racing Australia for a race and compute updated overlays.
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

    # Load model predictions — use the most-recent enrichment batch (by max enriched_at)
    # so duplicate history rows from re-enrichment don't cause inconsistent top picks.
    async with get_session() as session:
        # Get latest enriched_at for this race in history table
        max_hist_at = (await session.execute(
            select(func.max(RunnerPredictionHistoryRow.enriched_at))
            .where(RunnerPredictionHistoryRow.race_id == race_id)
        )).scalar()
        if max_hist_at:
            hist_result = await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.race_id == race_id)
                .where(RunnerPredictionHistoryRow.enriched_at == max_hist_at)
                .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                .order_by(RunnerPredictionHistoryRow.model_rank)
            )
        else:
            hist_result = None
        hist_stored = hist_result.scalars().all() if hist_result else []

        mutable_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id == race_id)
            .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            .order_by(RunnerPredictionRow.model_rank)
        )
        mut_stored = mutable_result.scalars().all()

    if not hist_stored and not mut_stored:
        raise HTTPException(404, f"No predictions for {race_id} — enrich first")

    stored = mut_stored or hist_stored
    top_model_pick_hist = hist_stored[0].horse_name if hist_stored else (stored[0].horse_name if stored else None)
    top_model_pick_mut = mut_stored[0].horse_name if mut_stored else top_model_pick_hist

    stored_odds = {r.horse_name: r.best_available_odds for r in stored}
    stored_overlay = {r.horse_name: r.overlay or 0.0 for r in stored}
    # model_probs deferred — assigned after db_settled is known so settled races use history

    # ── Step 1: load settled results from DB (HistoricalResultRow) ───────────
    async with get_session() as session:
        hr_rows = (await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id == race_id)
        )).scalars().all()
    db_results = {r.horse_name: r for r in hr_rows}
    db_settled = bool(db_results)

    # For settled races use pre-race history probs so overlay isn't contaminated by re-enrichment
    model_probs_src = hist_stored if (db_settled and hist_stored) else stored
    model_probs = {r.horse_name: r.win_probability or 0.0 for r in model_probs_src}

    # ── Step 2: try TAB for live odds + any missing positions ────────────────
    ra_tote: dict[str, tuple] = {}   # horse → (current_odds, actual_position)
    ra_ok = False
    try:
        client = get_tab_client()
        slug = _meeting_slug(venue_code, race_date)
        raw_event = await asyncio.wait_for(client.get_race(slug, race_num), timeout=15)
        if raw_event:
            for r in raw_event.get("runners", []):
                if r.get("scratched"):
                    continue
                horse = r.get("runnerName", "")
                tote_win = next((float(p["winPrice"]) for p in r.get("prices", []) if p.get("priceType") == "Win" and p.get("winPrice")), None)
                fixed_win = next((float(p["winPrice"]) for p in r.get("prices", []) if p.get("priceType") == "FixedWin" and p.get("winPrice")), None)
                current_odds = fixed_win or tote_win
                pos_raw = r.get("finishingPosition")
                actual_position = int(pos_raw) if pos_raw and int(pos_raw) > 0 else None
                ra_tote[horse] = (current_odds, actual_position)
            ra_ok = True
    except Exception:
        pass  # use DB data only

    # ── Step 3: merge — DB results are authoritative for settled races ─────────
    all_horses = set(model_probs.keys()) | set(ra_tote.keys())
    all_tote = []
    for horse in all_horses:
        p_odds, p_pos = ra_tote.get(horse, (None, None))
        db_r = db_results.get(horse)
        # Position: RA live > DB historical
        actual_position = p_pos or (db_r.position if db_r else None)
        # Odds: RA live > DB historical SP > stored enrichment odds
        current_odds = p_odds or (db_r.starting_price if db_r else None) or stored_odds.get(horse)
        all_tote.append((horse, current_odds, actual_position))

    # Overround-free implied probs from current tote
    valid_odds = [o for _, o, _ in all_tote if o and o > 1.0]
    total_implied = sum(1 / o for o in valid_odds) if valid_odds else 0
    scale = 1.0 / total_implied if total_implied > 0 else 1.0

    winner_name = next((h for h, _, pos in all_tote if pos == 1), None)
    placed_names = {h for h, _, pos in all_tote if pos and pos <= 3}
    settled = bool(winner_name) or db_settled
    # Completed races: history rank 1 is the genuine pre-race pick.
    # Upcoming races: use mutable (latest enrichment).
    top_model_pick = top_model_pick_hist if settled else top_model_pick_mut
    model_correct = (winner_name == top_model_pick) if winner_name else None
    model_placed = (top_model_pick in placed_names) if (placed_names and top_model_pick) else None

    # Persist results to HistoricalResultRow when seen here for the first time
    # so performance stats update without waiting for the next scheduled seed cron
    if settled and not db_settled and any(pos for _, _, pos in all_tote if pos):
        asyncio.create_task(_persist_live_results(race_id, all_tote))

    runners_odds = []
    for horse, current_odds, actual_position in all_tote:
        model_prob = model_probs.get(horse, 0.0)
        if current_odds and current_odds > 1.0:
            raw_implied = 1.0 / current_odds
            orf_implied = round(raw_implied * scale, 4)
            overlay = round(model_prob - orf_implied, 4)
        else:
            overlay = stored_overlay.get(horse, 0.0)
            current_odds = stored_odds.get(horse)
            orf_implied = round(1.0 / current_odds, 4) if current_odds and current_odds > 1.0 else 0.0
        runners_odds.append({
            "horse_name": horse,
            "current_tote_win": current_odds,
            "implied_prob": orf_implied,
            "model_win_prob": round(model_prob, 4),
            "overlay": overlay,
            "value": overlay > 0.05 and current_odds and current_odds >= 3.0,
            "actual_position": actual_position,
            "is_top_pick": horse == top_model_pick,
        })

    runners_odds.sort(key=lambda x: x["model_win_prob"], reverse=True)

    return {
        "race_id": race_id,
        "fetched_at": datetime.utcnow().isoformat(),
        "settled": settled,
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
        async with get_session() as session:
            await _inject_accumulated_stats(race, session)
        venue_cal = await _load_venue_calibration()
        predictions, _ = await enrich_and_predict_race(race, model, venue_calibration=venue_cal)

        async with get_session() as session:
            await save_race_predictions(
                session,
                race_id,
                [_prediction_to_db_dict(p, race_id, race.scheduled_time, race=race) for p in predictions],
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
            async with get_session() as session:
                await _inject_accumulated_stats(race, session)
            predictions, _ = await enrich_and_predict_race(race, model)
            async with get_session() as session:
                await save_race_predictions(
                    session,
                    race_id,
                    [_prediction_to_db_dict(p, race_id, race.scheduled_time, race=race) for p in predictions],
                )
            results.append({"race_id": race_id, "status": "ok", "runners": len(predictions)})
        except Exception as e:
            log.warning("Failed to enrich %s: %s", race_id, e)
            results.append({"race_id": race_id, "status": "error", "error": str(e)})

    return {"venue": venue_code, "date": race_date, "races": results}


# ── Retrain ───────────────────────────────────────────────────────────────────

@app.post("/api/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    days: int = Query(0, ge=0, le=365),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Retrain win model using race-grouped softmax (conditional logit).
    Returns immediately; all data loading and training runs in a background task.
    days=0 (default) uses all available data.
    """
    _check_admin(x_cron_secret)
    cutoff = (date.today() - timedelta(days=days)).isoformat() if days > 0 else None

    async def _do_win_retrain():
        from collections import defaultdict as _defaultdict
        from horse_engine.prediction.clean_features import (
            AggregateIndex, recompute_clean_feature_vector,
        )
        # FIX-S filter trio enforced at SQL level (BUG-38): cancelled NULL/false
        # and source="live". Without source="live" any prior
        # _run_validation_backtest rows (source="validation") would be ingested
        # as additional training examples, bleeding backtest-fit signal into
        # production weights.
        # Note: hr_rows is loaded over the FULL history range (no cutoff) so the
        # BUG-18-clean aggregate recompute has access to every result strictly
        # before each training example's race date. Without that, a 30-day
        # cutoff training set would see only 30 days of HR which defeats the
        # purpose of computing accurate career/track/distance rates.
        async with get_session() as session:
            hist_query = (
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
                .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                .where(RunnerPredictionHistoryRow.source == "live")
            )
            if cutoff:
                hist_query = hist_query.where(RunnerPredictionHistoryRow.race_id >= cutoff)
            hist_rows = (await session.execute(hist_query)).scalars().all()

            hr_rows = (await session.execute(select(HistoricalResultRow))).scalars().all()

        # Build the BUG-18-clean aggregate index from the FULL historical
        # results pool, then recompute each training row's feature vector
        # using only results strictly before that row's race date.
        index = AggregateIndex(hr_rows)

        # Use position == 1 directly — winner boolean can be stale from re-seedings
        winners: dict[str, str] = {}
        for r in hr_rows:
            if r.position == 1:
                winners[r.race_id] = _normalize_horse(r.horse_name)
        by_race: dict[str, list] = _defaultdict(list)
        for row in hist_rows:
            by_race[row.race_id].append(row)

        _today_r = date.today()
        race_groups: list[list[tuple]] = []
        race_weights: list[float] = []
        contam_repaired = 0
        for race_id, runners in by_race.items():
            winner_name = winners.get(race_id)
            if not winner_name:
                continue
            race: list[tuple] = []
            for row in runners:
                try:
                    fv = recompute_clean_feature_vector(row, index)
                    if fv is None:
                        # Fallback to the original enriched_json path so a single
                        # bad row doesn't drop the race from the training set.
                        er = EnrichedRunner(**json.loads(row.enriched_json))
                        fv = build_feature_vector(er)
                    else:
                        contam_repaired += 1
                    label = 1 if _normalize_horse(row.horse_name) == winner_name else 0
                    race.append((fv, label))
                except Exception as e:
                    log.debug("Skipping history row %s/%s: %s", race_id, row.horse_name, e)
            if sum(l for _, l in race) != 1:
                continue
            race_groups.append(race)
            try:
                days_ago = (_today_r - date.fromisoformat(race_id[:10])).days
            except Exception:
                days_ago = 30
            race_weights.append(math.exp(-days_ago / 30.0))
        log.info(
            "[retrain] BUG-18 recompute repaired %d / %d runner feature vectors",
            contam_repaired, sum(len(r) for r in race_groups) or 1,
        )

        if len(race_groups) < 50:
            log.error("[retrain] Need at least 50 races, have %d", len(race_groups))
            return
        log.info("[retrain] %d races assembled for race-grouped training", len(race_groups))
        m = HorseModel()
        s = await asyncio.to_thread(m.train_race_grouped, race_groups, sample_weights=race_weights)
        async with get_session() as sess:
            await save_model_weights(sess, s["weights"])
        log.info("[retrain] complete — %d races, top1=%.3f", s.get("races", 0), s.get("top1_hit_rate", 0))

    background_tasks.add_task(_do_win_retrain)
    return {"status": "retrain_started", "training_days": days or "all"}


@app.post("/api/admin/cancel-meeting")
async def cancel_meeting(
    venue: str = Query(..., description="Venue code, e.g. 'taree'"),
    date: Optional[str] = Query(None, description="YYYY-MM-DD, defaults to today"),
    x_cron_secret: Optional[str] = Header(None),
):
    """Mark all runner predictions for a venue+date as cancelled (abandoned meeting)."""
    from sqlalchemy import update as sa_update
    _check_admin(x_cron_secret)
    target_date = date or _today_aest().isoformat()
    async with get_session() as session:
        result = await session.execute(
            sa_update(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{target_date}_{venue}_%"))
            .values(cancelled=True)
        )
        # Mirror into history so settled-race reads also exclude the cancelled meeting.
        hist_result = await session.execute(
            sa_update(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{target_date}_{venue}_%"))
            .values(cancelled=True)
        )
        await session.commit()
        affected = result.rowcount
        hist_affected = hist_result.rowcount
    # Invalidate cached meeting list + the per-venue detail so the change is
    # immediately visible (BUG-30).
    _invalidate_meeting_caches(target_date, venue)
    log.info("[admin] cancel-meeting: marked %d mutable / %d history row(s) cancelled for %s on %s",
             affected, hist_affected, venue, target_date)
    return {
        "status": "cancelled",
        "venue": venue,
        "date": target_date,
        "runners_affected": affected,
        "history_affected": hist_affected,
    }


@app.post("/api/admin/retrain-place")
async def retrain_place_model(
    background_tasks: BackgroundTasks,
    days: int = Query(0, ge=0, le=365),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Train the place model on P(position ≤ 3) labels.
    Returns immediately; all data loading and training runs in a background task.
    """
    _check_admin(x_cron_secret)
    cutoff = (date.today() - timedelta(days=days)).isoformat() if days > 0 else None

    async def _do_place_retrain():
        from horse_engine.prediction.clean_features import (
            AggregateIndex, recompute_clean_feature_vector,
        )
        # FIX-S filter trio enforced at SQL level (BUG-39): cancelled NULL/false
        # and source="live". Mirrors the BUG-38 fix on the win retrain.
        # hr_rows loaded over the FULL historical range (no cutoff) so the
        # BUG-18-clean aggregate recompute has access to every prior result.
        async with get_session() as session:
            hist_query = (
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
                .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                .where(RunnerPredictionHistoryRow.source == "live")
            )
            if cutoff:
                hist_query = hist_query.where(RunnerPredictionHistoryRow.race_id >= cutoff)
            hist_rows = (await session.execute(hist_query)).scalars().all()

            hr_rows = (await session.execute(select(HistoricalResultRow))).scalars().all()

        index = AggregateIndex(hr_rows)

        # Use position <= 3 directly — placed boolean can be stale from re-seedings
        placed_lookup = {
            (r.race_id, _normalize_horse(r.horse_name)): (r.position is not None and r.position <= 3)
            for r in hr_rows if r.position is not None
        }
        training_data = []
        place_sample_weights = []
        _today_p = date.today()
        contam_repaired = 0
        for row in hist_rows:
            placed = placed_lookup.get((row.race_id, _normalize_horse(row.horse_name)))
            if placed is None:
                continue
            try:
                fv = recompute_clean_feature_vector(row, index)
                if fv is None:
                    er = EnrichedRunner(**json.loads(row.enriched_json))
                    fv = build_feature_vector(er)
                else:
                    contam_repaired += 1
                training_data.append((fv, 1 if placed else 0))
                try:
                    days_ago = (_today_p - date.fromisoformat(row.race_id[:10])).days
                except Exception:
                    days_ago = 30
                place_sample_weights.append(math.exp(-days_ago / 30.0))
            except Exception as e:
                log.debug("Skipping place retrain row %s/%s: %s", row.race_id, row.horse_name, e)
        log.info(
            "[place-retrain] BUG-18 recompute repaired %d / %d feature vectors",
            contam_repaired, len(training_data) or 1,
        )

        if not training_data:
            log.error("[place-retrain] No matched training examples")
            return
        placed_count = sum(1 for _, label in training_data if label == 1)
        log.info("[place-retrain] %d examples, %d placed (%.1f%%)",
                 len(training_data), placed_count, placed_count / len(training_data) * 100)
        m = PlaceModel()
        s = await asyncio.to_thread(m.train, training_data, sample_weights=place_sample_weights)
        async with get_session() as sess:
            await save_place_model_weights(sess, s["weights"])
        log.info("[place-retrain] complete — %d examples, accuracy=%.3f", len(training_data), s.get("accuracy", 0))

    background_tasks.add_task(_do_place_retrain)
    return {"status": "place_retrain_started", "cutoff": cutoff or "all"}


@app.post("/api/admin/retrain-exotic")
async def retrain_exotic_model(
    background_tasks: BackgroundTasks,
    days: int = Query(0, ge=0, le=365),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Train the exotic model using race-grouped trifecta-aware loss.
    Returns immediately; all data loading and training runs in a background task.
    """
    _check_admin(x_cron_secret)
    cutoff = (date.today() - timedelta(days=days)).isoformat() if days > 0 else None

    async def _do_exotic_retrain():
        async with get_session() as session:
            hr_query = select(HistoricalResultRow)
            if cutoff:
                hr_query = hr_query.where(HistoricalResultRow.race_id >= cutoff)
            hr_rows = (await session.execute(hr_query)).scalars().all()

            hist_query = select(RunnerPredictionHistoryRow).where(
                RunnerPredictionHistoryRow.enriched_json.isnot(None)
            ).where(
                RunnerPredictionHistoryRow.cancelled.is_(False)
                | RunnerPredictionHistoryRow.cancelled.is_(None)
            )
            if cutoff:
                hist_query = hist_query.where(RunnerPredictionHistoryRow.race_id >= cutoff)
            hist_rows = (await session.execute(hist_query)).scalars().all()

        result_lookup: dict[tuple, tuple] = {
            (r.race_id, _normalize_horse(r.horse_name)): (bool(r.placed), r.position)
            for r in hr_rows
        }
        race_data: dict[str, list[tuple[list[float], int, int | None]]] = {}
        for row in hist_rows:
            key = (row.race_id, _normalize_horse(row.horse_name))
            outcome = result_lookup.get(key)
            if not outcome:
                continue
            placed, position = outcome
            try:
                er = EnrichedRunner(**json.loads(row.enriched_json))
                fv = build_feature_vector(er)
                race_data.setdefault(row.race_id, []).append((fv, 1 if placed else 0, position))
            except Exception as e:
                log.debug("Skipping exotic retrain row %s/%s: %s", row.race_id, row.horse_name, e)

        race_groups = []
        for race_id, runners in race_data.items():
            if len(runners) < 7:
                continue
            top3_count = sum(1 for _, lbl, _ in runners if lbl == 1)
            if top3_count != 3:
                continue
            race_groups.append([(fv, lbl, pos) for fv, lbl, pos in runners])

        if not race_groups:
            log.error("[exotic-retrain] No eligible trifecta races found")
            return
        total_runners = sum(len(r) for r in race_groups)
        log.info("[exotic-retrain] %d races, %d runners", len(race_groups), total_runners)
        m = ExoticModel()
        s = await asyncio.get_event_loop().run_in_executor(None, m.train_exotic, race_groups)
        async with get_session() as sess:
            await save_exotic_model_weights(sess, s["weights"])
        log.info("[exotic-retrain] complete — %d races, tri_box=%.3f ff_box=%.3f",
                 len(race_groups), s.get("tri_box_hit_rate", 0), s.get("ff_box_hit_rate", 0))

    background_tasks.add_task(_do_exotic_retrain)
    return {"status": "exotic_retrain_started", "cutoff": cutoff or "all"}


@app.get("/api/admin/model-weights/status")
async def model_weights_status(x_cron_secret: Optional[str] = Header(None)):
    """Return feature count, last-updated timestamp, and training data availability."""
    _check_admin(x_cron_secret)
    from sqlalchemy import text as _text
    results = {}
    async with get_session() as session:
        for label, table in [
            ("win", "model_weights"),
            ("place", "place_model_weights"),
            ("exotic", "exotic_model_weights"),
        ]:
            row = (await session.execute(
                _text(f"SELECT COUNT(*), MAX(updated_at) FROM {table}")
            )).one()
            results[label] = {
                "feature_count": row[0],
                "last_updated_utc": row[1].isoformat() if row[1] else None,
            }
        enriched_count = (await session.execute(
            _text("SELECT COUNT(*) FROM runner_prediction_history WHERE enriched_json IS NOT NULL")
        )).scalar()
        results["training_data"] = {"history_rows_with_enriched_json": enriched_count}
    return results


_exotic_backtest_state: dict = {"running": False, "progress": "", "error": None}


@app.get("/api/admin/backtest-exotic/status")
async def backtest_exotic_status(x_cron_secret: Optional[str] = Header(None)):
    _check_admin(x_cron_secret)
    return _exotic_backtest_state


@app.post("/api/admin/backtest-exotic/run")
async def backtest_exotic_run(background_tasks: BackgroundTasks, x_cron_secret: Optional[str] = Header(None)):
    _check_admin(x_cron_secret)
    if _exotic_backtest_state["running"]:
        return {"status": "already_running", "progress": _exotic_backtest_state["progress"]}
    background_tasks.add_task(_run_exotic_backtest_bg)
    return {"status": "started"}


async def _run_exotic_backtest_bg():
    global _exotic_backtest_state
    _exotic_backtest_state = {"running": True, "progress": "Loading training data…", "error": None}
    try:
        await _do_exotic_backtest()
    except Exception as e:
        log.exception("[backtest-exotic] Failed: %s", e)
        _exotic_backtest_state = {"running": False, "progress": "error", "error": str(e)}


@app.get("/api/admin/backtest-exotic")
async def backtest_exotic(x_cron_secret: Optional[str] = Header(None)):
    _check_admin(x_cron_secret)
    return await _do_exotic_backtest()


async def _do_exotic_backtest():
    global _exotic_backtest_state
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

    _exotic_backtest_state["progress"] = f"Training {len(_CANDIDATE_WINDOWS)} windows…"

    for window in _CANDIDATE_WINDOWS:
        _exotic_backtest_state["progress"] = f"Training {window}d window…"
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
                race_train_data.setdefault(row.race_id, []).append((fv, label, row.position))
            except Exception:
                continue

        race_groups_train = []
        for race_id, runners in race_train_data.items():
            if len(runners) < 7:
                continue
            top3_count = sum(1 for _, lbl, _ in runners if lbl == 1)
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
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, m.train_exotic, race_groups_train)
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
        _exotic_backtest_state = {"running": False, "progress": "done", "error": "No valid training windows found"}
        return {
            "error": "No valid training windows found",
            "holdout_races": len(holdout_groups),
            "window_results": window_results,
        }

    _exotic_backtest_state["progress"] = "Running feature ablation…"
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

    result = {
        "holdout_races": len(holdout_groups),
        "holdout_days": holdout_days,
        "best_window": best_window,
        "best_holdout_box_hit_rate": best_hit_rate,
        "window_results": window_results,
        "feature_ablation": ablation_results,
    }

    from horse_engine.models.database import ExoticBacktestRow
    async with get_session() as session:
        session.add(ExoticBacktestRow(
            best_window=best_window,
            best_holdout_box_hit_rate=best_hit_rate,
            holdout_races=len(holdout_groups),
            holdout_days=holdout_days,
            results_json=json.dumps(result),
        ))
        await session.commit()

    _exotic_backtest_state = {"running": False, "progress": "done", "error": None}
    return result


@app.get("/api/admin/backtest-exotic/last")
async def backtest_exotic_last(x_cron_secret: Optional[str] = Header(None)):
    if x_cron_secret != settings.cron_secret:
        raise HTTPException(403)
    from horse_engine.models.database import ExoticBacktestRow
    async with get_session() as session:
        result = await session.execute(
            select(ExoticBacktestRow).order_by(ExoticBacktestRow.ran_at.desc()).limit(1)
        )
        row = result.scalars().first()
    if not row:
        return {"data": None}
    return {"data": json.loads(row.results_json), "ran_at": row.ran_at.isoformat()}


@app.get("/api/admin/backtest-exotic/history")
async def backtest_exotic_history(
    limit: int = Query(14, ge=1, le=50),
    x_cron_secret: Optional[str] = Header(None),
):
    _check_admin(x_cron_secret)
    from horse_engine.models.database import ExoticBacktestRow
    async with get_session() as session:
        result = await session.execute(
            select(ExoticBacktestRow).order_by(ExoticBacktestRow.ran_at.desc()).limit(limit)
        )
        rows = result.scalars().all()
    if not rows:
        return {"history": [], "drift": False}

    history = [
        {
            "ran_at": r.ran_at.isoformat(),
            "best_window": r.best_window,
            "best_holdout_box_hit_rate": r.best_holdout_box_hit_rate,
            "holdout_races": r.holdout_races,
        }
        for r in rows
    ]

    # Drift: current hit rate vs average of prior 3 runs
    current = history[0]["best_holdout_box_hit_rate"]
    prior = [h["best_holdout_box_hit_rate"] for h in history[1:4] if h["best_holdout_box_hit_rate"]]
    drift = bool(prior and current < (sum(prior) / len(prior)) - 1.5)

    return {"history": history, "drift": drift, "drift_reason": "Below recent avg" if drift else None}


@app.get("/api/admin/exotic-daily-performance")
async def exotic_daily_performance(
    days: int = Query(14, ge=1, le=90),
    x_cron_secret: Optional[str] = Header(None),
):
    _check_admin(x_cron_secret)
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        # All historical results in the window with a position
        hr_result = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id >= cutoff)
            .where(HistoricalResultRow.position.isnot(None))
        )
        all_hr = hr_result.scalars().all()

        # Exotic predictions from history table — pre-race ranks, never post-race
        # contaminated. Cancelled (BUG-34) keeps scratched horses out of the
        # "predicted top 3" set so they don't depress the trifecta hit rate
        # by guaranteeing a mismatch against actuals. Source="live" excludes
        # validation-backtest rows.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= cutoff)
            .where(
                (RunnerPredictionHistoryRow.exotic_model_rank >= 1) |
                (RunnerPredictionHistoryRow.place_model_rank >= 1)
            )
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )
        all_pred = pred_result.scalars().all()

    # Build: race_id → predicted top 3 horse names
    pred_top3: dict[str, list[str]] = {}
    for p in all_pred:
        rank = p.exotic_model_rank if p.exotic_model_rank else p.place_model_rank
        if rank and rank <= 3:
            pred_top3.setdefault(p.race_id, []).append((rank, p.horse_name))
    predicted: dict[str, set[str]] = {
        rid: {name for _, name in sorted(runners)[:3]}
        for rid, runners in pred_top3.items()
    }

    # Build: race_id → actual top 3 horse names
    actual_top3: dict[str, set[str]] = {}
    for r in all_hr:
        if r.position and 1 <= r.position <= 3:
            actual_top3.setdefault(r.race_id, set()).add(r.horse_name)

    # Group by date
    from collections import defaultdict
    by_date: dict[str, dict] = defaultdict(lambda: {"eligible": 0, "hits": 0})
    for race_id in predicted:
        if race_id not in actual_top3:
            continue
        actual = actual_top3[race_id]
        pred = predicted[race_id]
        if len(actual) < 3 or len(pred) < 3:
            continue
        race_date_str = race_id[:10]  # "YYYY-MM-DD"
        by_date[race_date_str]["eligible"] += 1
        if actual == pred:
            by_date[race_date_str]["hits"] += 1

    summary = []
    for d in sorted(by_date.keys()):
        row = by_date[d]
        hit_rate = round(row["hits"] / row["eligible"] * 100, 1) if row["eligible"] else 0
        summary.append({
            "date": d,
            "eligible_races": row["eligible"],
            "box_hits": row["hits"],
            "hit_rate_pct": hit_rate,
        })

    return {"summary": summary, "days": days}


# ── Admin: TAB API probe ──────────────────────────────────────────────────────

@app.get("/api/admin/probe-ras")
async def probe_ras(slug: str = "warwick-farm", date: str = "", race_num: int = 1,
                    x_cron_secret: Optional[str] = Header(None)):
    """Probe racingandsports.com.au for form data availability."""
    _check_admin(x_cron_secret)
    import httpx as _httpx
    target_date = date or _today_aest().isoformat()

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.racingandsports.com.au/",
    }

    results = {}

    # Test 1: enhanced form page (HTML)
    url_html = f"https://www.racingandsports.com.au/form-guide/thoroughbred/australia/{slug}/{target_date}/R{race_num}/enhanced-form"
    try:
        async with _httpx.AsyncClient(headers=HEADERS, timeout=20, follow_redirects=True) as c:
            resp = await c.get(url_html)
            results["enhanced_form"] = {
                "url": url_html, "status": resp.status_code,
                "content_length": len(resp.content),
                "snippet": resp.text[:300] if resp.status_code == 200 else resp.text[:200],
            }
    except Exception as e:
        results["enhanced_form"] = {"url": url_html, "error": str(e)}

    # Test 2: try the API endpoint that their SPA likely calls
    url_api = f"https://www.racingandsports.com.au/api/form-guide/thoroughbred/australia/{slug}/{target_date}/R{race_num}"
    try:
        async with _httpx.AsyncClient(headers={**HEADERS, "Accept": "application/json"}, timeout=20, follow_redirects=True) as c:
            resp = await c.get(url_api)
            results["api_endpoint"] = {
                "url": url_api, "status": resp.status_code,
                "content_type": resp.headers.get("content-type", ""),
                "snippet": resp.text[:300],
            }
    except Exception as e:
        results["api_endpoint"] = {"url": url_api, "error": str(e)}

    # Test 3: old ASP form guide
    url_asp = f"https://www.racingandsports.com.au/en/form-guide/meeting.asp"
    try:
        async with _httpx.AsyncClient(headers=HEADERS, timeout=20, follow_redirects=True) as c:
            resp = await c.get(url_asp)
            results["asp_guide"] = {"url": url_asp, "status": resp.status_code, "snippet": resp.text[:200]}
    except Exception as e:
        results["asp_guide"] = {"url": url_asp, "error": str(e)}

    return results


@app.post("/api/admin/cancel-runner")
async def cancel_runner(
    race_id: str = Query(...),
    horse_name: str = Query(...),
    x_cron_secret: Optional[str] = Header(None),
):
    """Mark a specific runner as cancelled (e.g. scratched from wrong race)."""
    _check_admin(x_cron_secret)
    from sqlalchemy import update as sa_update
    async with get_session() as session:
        result = await session.execute(
            sa_update(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id == race_id)
            .where(RunnerPredictionRow.horse_name == horse_name)
            .values(cancelled=True)
        )
        # Mirror into history so settled-race reads also exclude this runner.
        hist_result = await session.execute(
            sa_update(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id == race_id)
            .where(RunnerPredictionHistoryRow.horse_name == horse_name)
            .values(cancelled=True)
        )
        await session.commit()
    # Clear the per-venue meeting cache so the scratch is immediately visible.
    # The helper also drops the list cache for this date — strictly unnecessary
    # for a single-runner scratch (the venue list rarely changes) but the cost
    # of an extra RA fetch on the next list_meetings call is trivial.
    date_part, venue_part, _ = _parse_race_id(race_id)
    _invalidate_meeting_caches(date_part, venue_part)
    return {
        "updated": result.rowcount,
        "history_updated": hist_result.rowcount,
        "race_id": race_id,
        "horse_name": horse_name,
    }


@app.get("/api/admin/debug-odds")
async def debug_odds(venue: str = "", date: str = "", x_cron_secret: Optional[str] = Header(None)):
    """Probe OddsPro + Betfair for a venue. Returns raw odds data for diagnosis."""
    _check_admin(x_cron_secret)
    from horse_engine.clients.oddspro import OddsProClient
    target_date = date or _today_aest().isoformat()

    op = OddsProClient()
    result: dict = {"date": target_date, "venue_query": venue}

    # OddsPro — full meeting odds (not just movers)
    try:
        meeting_odds = await op.get_meeting_odds(target_date)
        result["op_tracks"] = list(meeting_odds.keys())
        # Find matching track
        venue_lower = (venue or "").lower()
        alias_lower = _VENUE_ALIASES.get(venue_lower, venue_lower)
        track_key = None
        if venue_lower in meeting_odds:
            track_key = venue_lower
        elif alias_lower in meeting_odds:
            track_key = alias_lower
        else:
            for k in meeting_odds:
                if venue_lower in k or k in venue_lower or alias_lower in k or k in alias_lower:
                    track_key = k
                    break
        result["op_track_matched"] = track_key
        if track_key:
            odds_map = meeting_odds[track_key]
            result["op_runners_total"] = len(odds_map)
            result["op_sample"] = [
                {"race": k[0], "runner": k[1], "best_price": v}
                for k, v in list(odds_map.items())[:20]
            ]
    except Exception as e:
        result["op_error"] = str(e)

    # TAB meetings (raw venue names from API)
    try:
        import httpx as _httpx
        _TAB = "https://api.tab.com.au/v1/tab-info-service"
        _JURS = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
        _HDR = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        async with _httpx.AsyncClient(headers=_HDR, timeout=10) as client:
            tasks = [
                client.get(f"{_TAB}/racing/dates/{target_date}/meetings", params={"jurisdiction": j})
                for j in _JURS
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        tab_venues = []
        for jur, resp in zip(_JURS, responses):
            if isinstance(resp, Exception) or resp.status_code != 200:
                continue
            for m in resp.json().get("meetings", []):
                if m.get("raceType") == "R" and m.get("meetingCode"):
                    tab_venues.append({
                        "name": m.get("venueName"),
                        "code": m.get("meetingCode"),
                        "jur": jur,
                    })
        result["tab_venues"] = tab_venues
    except Exception as e:
        result["tab_error"] = str(e)

    # Betfair
    try:
        from horse_engine.config import settings
        result["bf_credentials"] = bool(settings.betfair_app_key and settings.betfair_username and settings.betfair_password)
        if result["bf_credentials"]:
            from horse_engine.clients.betfair import BetfairClient
            bf = BetfairClient()
            login_ok = await bf._login()
            result["bf_login"] = login_ok
            if login_ok:
                meetings = await bf.get_meetings(target_date)
                result["bf_meetings"] = [{"slug": m["slug"], "name": m["name"]} for m in meetings]
    except Exception as e:
        result["bf_error"] = str(e)

    return result


@app.get("/api/admin/debug-betfair")
async def debug_betfair(date: str = "", x_cron_secret: Optional[str] = Header(None)):
    """Test Betfair connection: credentials, auth, market count, and meeting slugs."""
    _check_admin(x_cron_secret)
    from horse_engine.config import settings
    target_date = date or _today_aest().isoformat()

    info: dict = {
        "date": target_date,
        "credentials_set": bool(settings.betfair_app_key and settings.betfair_username and settings.betfair_password),
        "app_key_prefix": settings.betfair_app_key[:4] + "..." if settings.betfair_app_key else None,
    }
    if not info["credentials_set"]:
        return info

    try:
        import httpx as _httpx
        from horse_engine.config import settings as _s
        async with _httpx.AsyncClient(timeout=15.0) as _c:
            _r = await _c.post(
                "https://identitysso.betfair.com.au/api/login",
                data={"username": _s.betfair_username, "password": _s.betfair_password},
                headers={"X-Application": _s.betfair_app_key, "Accept": "application/json",
                         "Content-Type": "application/x-www-form-urlencoded"},
            )
            _body = _r.json()
            info["login_status"] = _body.get("status")
            info["login_error"] = _body.get("error")
            info["login_ok"] = _body.get("status") == "SUCCESS"
        if not info["login_ok"]:
            return info
        from horse_engine.clients.betfair import BetfairClient
        bf = BetfairClient()
        bf._session_token = _body.get("token")
        login_ok = True

        markets = await bf._load_catalogue(target_date)
        info["market_count"] = len(markets)
        meetings = await bf.get_meetings(target_date)
        info["meetings"] = [{"slug": m["slug"], "name": m["name"], "state": m["state"]} for m in meetings]
    except Exception as e:
        info["error"] = str(e)

    return info


@app.get("/api/admin/probe-tab")
async def probe_tab(race_id: str = "", date: str = "", x_cron_secret: Optional[str] = Header(None)):
    """Probe the TAB API. Pass race_id=YYYY-MM-DD_venue_RN for race data, or date=YYYY-MM-DD to list meetings."""
    _check_admin(x_cron_secret)
    import httpx as _httpx
    from horse_engine.clients.tab import _AU_JURISDICTIONS
    TAB_BASE = "https://api.tab.com.au/v1/tab-info-service"
    HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}

    target_date = date or (race_id.split("_")[0] if race_id else _today_aest().isoformat())

    # Step 1: list meetings to discover real venue codes
    meetings_by_code: dict[str, dict] = {}
    http_statuses: dict[str, int] = {}
    for jur in _AU_JURISDICTIONS:
        try:
            async with _httpx.AsyncClient(headers=HEADERS, timeout=15) as c:
                resp = await c.get(f"{TAB_BASE}/racing/dates/{target_date}/meetings",
                                   params={"jurisdiction": jur})
                http_statuses[jur] = resp.status_code
                if resp.status_code == 200:
                    for m in resp.json().get("meetings", []):
                        if m.get("raceType") == "R" and m.get("meetingCode"):
                            meetings_by_code[m["meetingCode"]] = {
                                "venue": m.get("venueName"), "jurisdiction": jur,
                                "state": (m.get("location") or {}).get("state"),
                            }
        except Exception as e:
            http_statuses[jur] = f"error: {e}"

    if not race_id:
        return {"date": target_date, "meetings": meetings_by_code, "http_statuses": http_statuses}

    # Step 2: fetch specific race using the real meeting code
    parts = race_id.split("_")
    venue_slug = parts[1] if len(parts) >= 3 else ""
    race_num = int(parts[2].replace("R", "")) if len(parts) >= 3 else 1

    # Match slug to TAB meeting code (fuzzy: strip hyphens, case-insensitive)
    slug_clean = venue_slug.replace("-", "").lower()
    matched_code = next(
        (code for code, info in meetings_by_code.items()
         if info["venue"] and info["venue"].replace(" ", "").lower() == slug_clean),
        None
    )
    if not matched_code:
        # Fall back to trying the slug uppercased as the code
        matched_code = venue_slug.upper().replace("-", "")

    jur = (meetings_by_code.get(matched_code) or {}).get("jurisdiction", "NSW")
    try:
        async with _httpx.AsyncClient(headers=HEADERS, timeout=15) as c:
            resp = await c.get(
                f"{TAB_BASE}/racing/dates/{target_date}/meetings/{matched_code}/R/races/{race_num}",
                params={"jurisdiction": jur}
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "venue_code_tried": matched_code,
                        "url": str(resp.url), "available_codes": list(meetings_by_code.keys())}
            raw = resp.json()
    except Exception as e:
        return {"error": str(e), "available_codes": list(meetings_by_code.keys())}

    r0 = (raw.get("runners") or [{}])[0]
    return {
        "jurisdiction": jur,
        "venue_code": matched_code,
        "race_name": raw.get("raceName"),
        "runner_count": len(raw.get("runners", [])),
        "sample_runner_keys": list(r0.keys()),
        "form_keys": list((r0.get("form") or {}).keys()),
        "career_overall": (r0.get("form") or {}).get("overall"),
        "jockey_keys": list((r0.get("jockey") or {}).keys()),
        "trainer_keys": list((r0.get("trainer") or {}).keys()),
        "recent_starts_count": len((r0.get("form") or {}).get("recentStarts") or []),
        "first_recent_start": ((r0.get("form") or {}).get("recentStarts") or [None])[0],
        "sample_runner": r0,
    }


@app.get("/api/admin/test-ra-fetch")
async def test_ra_fetch(ra_key: str = "2026Jun08,NSW,Canterbury Park",
                        x_cron_secret: Optional[str] = Header(None)):
    """Directly call ra.get_results() with a key and return raw parse output."""
    _check_admin(x_cron_secret)
    client = get_tab_client()
    ra = client._ra
    from urllib.parse import quote
    url = f"https://www.racingaustralia.horse/FreeFields/Results.aspx?Key={quote(ra_key, safe='')}"
    try:
        html = await ra._get(url)
        http_ok = True
        html_len = len(html)
        snippet = html[:500]
    except Exception as e:
        return {"error": str(e), "url": url}
    from horse_engine.clients.racing_australia import _parse_results_page
    parsed = _parse_results_page(html)
    races_found = len(parsed)
    sample = {}
    for rn, rd in list(parsed.items())[:3]:
        w = [(k, v["position"]) for k, v in rd["runners"].items() if v.get("position") == 1]
        sample[rn] = {"runners": len(rd["runners"]), "winner": w}
    return {
        "url": url,
        "ra_key": ra_key,
        "html_len": html_len,
        "html_snippet": snippet,
        "races_found": races_found,
        "sample": sample,
    }


@app.get("/api/admin/debug-meeting-picks/{race_date}/{venue_code}")
async def debug_meeting_picks(
    race_date: str,
    venue_code: str,
    x_cron_secret: Optional[str] = Header(None),
):
    """Dump raw top_picks vs winners for every race in a meeting to diagnose model_correct=False."""
    _check_admin(x_cron_secret)
    async with get_session() as session:
        pred_rows = (await session.execute(
            select(RunnerPredictionRow.race_id)
            .where(RunnerPredictionRow.race_id.like(f"{race_date}_{venue_code}_%"))
            .distinct()
        )).scalars().all()
        race_ids = sorted(set(pred_rows))

        # top picks from history
        max_at_result = await session.execute(
            select(RunnerPredictionHistoryRow.race_id,
                   func.max(RunnerPredictionHistoryRow.enriched_at).label("max_at"))
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .group_by(RunnerPredictionHistoryRow.race_id)
        )
        max_at_by_race = {row.race_id: row.max_at for row in max_at_result}

        hist_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
        )).scalars().all()
        top_picks_hist = {}
        for p in hist_rows:
            if p.enriched_at == max_at_by_race.get(p.race_id):
                top_picks_hist[p.race_id] = p.horse_name

        mut_rows = (await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.in_(race_ids))
            .where(RunnerPredictionRow.model_rank == 1)
        )).scalars().all()
        top_picks_mut = {p.race_id: p.horse_name for p in mut_rows}

        hist_results = (await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.in_(race_ids))
            .order_by(HistoricalResultRow.position)
        )).scalars().all()
        winners_raw = {}
        all_results = {}
        for r in hist_results:
            all_results.setdefault(r.race_id, []).append(
                {"horse_name": r.horse_name, "position": r.position}
            )
            if r.position == 1:
                winners_raw[r.race_id] = r.horse_name

    out = []
    for rid in race_ids:
        pick_hist = top_picks_hist.get(rid)
        pick_mut = top_picks_mut.get(rid)
        pick = pick_hist or pick_mut
        winner = winners_raw.get(rid)
        match = (
            _normalize_horse(pick) == _normalize_horse(winner)
            if pick and winner else None
        )
        out.append({
            "race_id": rid,
            "top_pick_history": pick_hist,
            "top_pick_mutable": pick_mut,
            "effective_pick": pick,
            "normalized_pick": _normalize_horse(pick) if pick else None,
            "winner_raw": winner,
            "normalized_winner": _normalize_horse(winner) if winner else None,
            "model_correct": match,
            "all_results": all_results.get(rid, []),
        })
    return {"race_date": race_date, "venue_code": venue_code, "races": out}


@app.get("/api/admin/probe-ra-results")
async def probe_ra_results(race_date: str = "", x_cron_secret: Optional[str] = Header(None)):
    """
    Diagnostic: for each venue on race_date, show stored venue/state, keys tried, and
    whether RA returned results. Helps diagnose why seed-ra-results gets 0.
    """
    _check_admin(x_cron_secret)
    from horse_engine.clients.racing_australia import _ra_date as _make_ra_date
    from urllib.parse import quote

    target = race_date or (_today_aest() - timedelta(days=1)).isoformat()
    client = get_tab_client()
    ra = client._ra
    ra_date_str = _make_ra_date(target)
    _BASE_RA = "https://www.racingaustralia.horse"

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(RunnerPredictionRow.race_id, RunnerPredictionRow.venue, RunnerPredictionRow.state)
            .where(RunnerPredictionRow.race_id.like(f"{target}_%"))
            .distinct()
        )).all()
        hist_race_ids = set((await session.execute(
            select(HistoricalResultRow.race_id)
            .where(HistoricalResultRow.race_id.like(f"{target}_%"))
            .distinct()
        )).scalars().all())

    venue_state: dict[tuple[str, str], list[str]] = {}
    no_venue: list[str] = []
    for row in pred_rows:
        v = (row.venue or "").strip()
        s = (row.state or "").strip().upper()
        if v and s:
            venue_state.setdefault((v, s), []).append(row.race_id)
        else:
            no_venue.append(row.race_id)

    detail = []
    prefixes = [""] + [p for p in ra._SPONSOR_PREFIXES]
    for (venue_name, state), race_ids in venue_state.items():
        tried = []
        found_key = None
        for prefix in prefixes:
            ra_key = f"{ra_date_str},{state},{prefix}{venue_name}"
            url = f"{_BASE_RA}/Results.aspx?Key={quote(ra_key, safe='')}"
            try:
                html = await ra._get(url)
                from horse_engine.clients.racing_australia import _parse_results_page
                parsed = _parse_results_page(html)
                races_found = len(parsed)
                runners_found = sum(len(r["runners"]) for r in parsed.values())
            except Exception as e:
                races_found = 0
                runners_found = 0
                html = str(e)
            tried.append({"key": ra_key, "url": url, "races": races_found, "runners": runners_found,
                          "html_snippet": html[:200] if isinstance(html, str) else ""})
            if races_found > 0:
                found_key = ra_key
                break

        already = sum(1 for rid in race_ids if rid in hist_race_ids)
        detail.append({
            "venue": venue_name, "state": state,
            "pred_race_ids": sorted(set(race_ids)),
            "already_in_historical": already,
            "found_key": found_key,
            "tried": tried,
        })

    return {
        "date": target,
        "venues_with_predictions": len(venue_state),
        "race_ids_missing_venue": no_venue[:10],
        "detail": detail,
    }


@app.get("/api/admin/contamination-audit")
async def contamination_audit(
    sample: int = Query(5, ge=1, le=50),
    x_cron_secret: Optional[str] = Header(None),
):
    """Pick `sample` random pre-2026-06-11 history rows and show the BUG-18
    contamination delta for each aggregate-rate field.

    Use this to verify the recompute helper before committing to a retrain,
    and to quantify how big the bias actually is for your specific data.
    """
    _check_admin(x_cron_secret)
    from horse_engine.prediction.clean_features import (
        AggregateIndex, contamination_diff,
    )
    import random as _random

    async with get_session() as session:
        # Pull from before the BUG-18 fix so the delta is real.
        cutoff = "2026-06-11"
        candidates = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .where(RunnerPredictionHistoryRow.race_id < cutoff)
        )).scalars().all()
        hr_rows = (await session.execute(select(HistoricalResultRow))).scalars().all()

    if not candidates:
        return {"sample": 0, "rows": [], "note": "no pre-fix history rows available"}

    chosen = _random.sample(candidates, min(sample, len(candidates)))
    index = AggregateIndex(hr_rows)
    diffs = [contamination_diff(row, index) for row in chosen]

    # Summary stats — which fields shifted most and by how much
    field_deltas: dict[str, list[float]] = {}
    for d in diffs:
        for field, vals in d.get("fields", {}).items():
            delta = vals.get("delta")
            if delta is not None:
                field_deltas.setdefault(field, []).append(abs(delta))
    summary = sorted(
        [
            {
                "field": f,
                "mean_abs_delta": round(sum(v) / len(v), 4),
                "max_abs_delta": round(max(v), 4),
                "n_with_change": sum(1 for x in v if x > 1e-6),
            }
            for f, v in field_deltas.items()
        ],
        key=lambda x: x["mean_abs_delta"],
        reverse=True,
    )

    return {
        "sample": len(chosen),
        "available_pre_fix_rows": len(candidates),
        "field_shift_summary": summary,
        "rows": diffs,
    }


@app.get("/api/admin/data-coverage")
async def data_coverage(x_cron_secret: Optional[str] = Header(None)):
    """Check how much jockey/trainer/result data we have for building self-accumulated stats."""
    _check_admin(x_cron_secret)
    from sqlalchemy import func, text as sa_text
    async with get_session() as session:
        hist_count = (await session.execute(
            select(func.count()).select_from(HistoricalResultRow)
        )).scalar()
        hist_winners = (await session.execute(
            select(func.count()).select_from(HistoricalResultRow)
            .where(HistoricalResultRow.winner == True)
        )).scalar()
        hist_date_range = (await session.execute(
            select(func.min(HistoricalResultRow.race_id), func.max(HistoricalResultRow.race_id))
        )).one()

        snap_count = (await session.execute(
            select(func.count()).select_from(RunnerPredictionHistoryRow)
        )).scalar()
        snap_with_jockey = (await session.execute(
            select(func.count()).select_from(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.jockey.isnot(None))
            .where(RunnerPredictionHistoryRow.jockey != "")
        )).scalar()
        distinct_jockeys = (await session.execute(
            select(func.count(func.distinct(RunnerPredictionHistoryRow.jockey)))
            .select_from(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.jockey.isnot(None))
            .where(RunnerPredictionHistoryRow.jockey != "")
        )).scalar()
        distinct_trainers = (await session.execute(
            select(func.count(func.distinct(RunnerPredictionHistoryRow.trainer)))
            .select_from(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.trainer.isnot(None))
            .where(RunnerPredictionHistoryRow.trainer != "")
        )).scalar()

        # Count joinable rows: history snapshot + result both exist for same race+horse
        joinable = (await session.execute(sa_text("""
            SELECT COUNT(*) FROM runner_prediction_history h
            JOIN historical_results r ON h.race_id = r.race_id AND LOWER(h.horse_name) = LOWER(r.horse_name)
            WHERE h.jockey IS NOT NULL AND h.jockey != ''
        """))).scalar()

        snap_date_range = (await session.execute(
            select(func.min(RunnerPredictionHistoryRow.race_id), func.max(RunnerPredictionHistoryRow.race_id))
        )).one()

    return {
        "historical_results": {"total_runners": hist_count, "winners": hist_winners,
                               "date_range": [hist_date_range[0], hist_date_range[1]]},
        "prediction_history": {"total_snapshots": snap_count, "with_jockey": snap_with_jockey,
                               "distinct_jockeys": distinct_jockeys, "distinct_trainers": distinct_trainers,
                               "date_range": [snap_date_range[0], snap_date_range[1]]},
        "joinable_rows": joinable,
        "note": "joinable_rows = rows where we have both jockey/trainer AND race result — usable for win rate computation",
    }


@app.post("/api/admin/backfill-context")
async def backfill_context(x_cron_secret: Optional[str] = Header(None)):
    """
    Patch existing historical_results rows that have NULL jockey/trainer/venue/distance
    by joining against runner_predictions and runner_prediction_history tables.
    Pure DB operation — no API calls. Run once after deploying new columns.
    """
    _check_admin(x_cron_secret)
    from sqlalchemy import text as sa_text

    async with get_session() as session:
        # Pass 1: patch from mutable runner_predictions (has venue, distance, track_condition etc.)
        r1 = await session.execute(sa_text("""
            UPDATE historical_results
            SET
                jockey          = COALESCE(historical_results.jockey,          rp.jockey),
                trainer         = COALESCE(historical_results.trainer,         rp.trainer),
                venue           = COALESCE(historical_results.venue,           rp.venue),
                state           = COALESCE(historical_results.state,           rp.state),
                distance        = COALESCE(historical_results.distance,        rp.distance),
                track_condition = COALESCE(historical_results.track_condition, rp.track_condition),
                barrier         = COALESCE(historical_results.barrier,         rp.barrier),
                tab_number      = COALESCE(historical_results.tab_number,      rp.tab_number),
                weight          = COALESCE(historical_results.weight,          rp.weight),
                race_class      = COALESCE(historical_results.race_class,      rp.race_name),
                prize_money     = COALESCE(historical_results.prize_money,     rp.prize_money),
                field_size      = COALESCE(historical_results.field_size,      rp.field_size),
                race_number     = COALESCE(historical_results.race_number,     rp.race_number)
            FROM runner_predictions rp
            WHERE historical_results.race_id = rp.race_id
              AND LOWER(historical_results.horse_name) = LOWER(rp.horse_name)
              AND (historical_results.jockey IS NULL OR historical_results.venue IS NULL)
        """))
        patched_from_pred = r1.rowcount

        # Pass 2: patch from immutable prediction history (fills gaps where mutable was overwritten)
        r2 = await session.execute(sa_text("""
            UPDATE historical_results
            SET
                jockey          = COALESCE(historical_results.jockey,          rph.jockey),
                trainer         = COALESCE(historical_results.trainer,         rph.trainer),
                venue           = COALESCE(historical_results.venue,           rph.venue),
                state           = COALESCE(historical_results.state,           rph.state),
                distance        = COALESCE(historical_results.distance,        rph.distance),
                track_condition = COALESCE(historical_results.track_condition, rph.track_condition),
                barrier         = COALESCE(historical_results.barrier,         rph.barrier),
                tab_number      = COALESCE(historical_results.tab_number,      rph.tab_number),
                weight          = COALESCE(historical_results.weight,          rph.weight),
                race_class      = COALESCE(historical_results.race_class,      rph.race_name),
                prize_money     = COALESCE(historical_results.prize_money,     rph.prize_money),
                field_size      = COALESCE(historical_results.field_size,      rph.field_size),
                race_number     = COALESCE(historical_results.race_number,     rph.race_number)
            FROM runner_prediction_history rph
            WHERE historical_results.race_id = rph.race_id
              AND LOWER(historical_results.horse_name) = LOWER(rph.horse_name)
              AND (historical_results.jockey IS NULL OR historical_results.venue IS NULL)
        """))
        patched_from_history = r2.rowcount

        await session.commit()

        # Check remaining gaps
        still_null = (await session.execute(sa_text(
            "SELECT COUNT(*) FROM historical_results WHERE jockey IS NULL"
        ))).scalar()
        total = (await session.execute(sa_text(
            "SELECT COUNT(*) FROM historical_results"
        ))).scalar()

    return {
        "patched_from_runner_predictions": patched_from_pred,
        "patched_from_prediction_history": patched_from_history,
        "still_missing_jockey": still_null,
        "total_rows": total,
        "coverage_pct": round(100 * (total - still_null) / total, 1) if total else 0,
    }


# ── Admin: purge trial rows ───────────────────────────────────────────────────

@app.delete("/api/admin/purge-trials")
async def purge_trial_rows(x_cron_secret: Optional[str] = Header(None), dry_run: bool = Query(True)):
    """Find and delete RunnerPredictionRow + HistoricalResultRow entries for trial/trail/jumpout venues."""
    _check_admin(x_cron_secret)
    from sqlalchemy import delete as sa_delete
    trial_re = re.compile(r"-(trial|trail|jumpout)s?[_-]", re.IGNORECASE)

    async with get_session() as session:
        pred_rows = (await session.execute(select(RunnerPredictionRow.race_id).distinct())).scalars().all()
        snap_rows = (await session.execute(select(RunnerPredictionHistoryRow.race_id).distinct())).scalars().all()
        hist_rows = (await session.execute(select(HistoricalResultRow.race_id).distinct())).scalars().all()

    trial_pred_ids = [rid for rid in pred_rows if trial_re.search(rid)]
    trial_snap_ids = [rid for rid in snap_rows if trial_re.search(rid)]
    trial_hist_ids = [rid for rid in hist_rows if trial_re.search(rid)]

    if dry_run:
        return {"dry_run": True, "pred_race_ids": trial_pred_ids, "snap_race_ids": trial_snap_ids, "hist_race_ids": trial_hist_ids}

    async with get_session() as session:
        if trial_pred_ids:
            await session.execute(
                sa_delete(RunnerPredictionRow).where(RunnerPredictionRow.race_id.in_(trial_pred_ids))
            )
        if trial_snap_ids:
            await session.execute(
                sa_delete(RunnerPredictionHistoryRow).where(RunnerPredictionHistoryRow.race_id.in_(trial_snap_ids))
            )
        if trial_hist_ids:
            await session.execute(
                sa_delete(HistoricalResultRow).where(HistoricalResultRow.race_id.in_(trial_hist_ids))
            )
        await session.commit()

    return {"dry_run": False, "deleted_pred_race_ids": trial_pred_ids, "deleted_snap_race_ids": trial_snap_ids, "deleted_hist_race_ids": trial_hist_ids}


@app.delete("/api/admin/purge-venue/{venue_code}")
async def purge_venue_rows(venue_code: str, x_cron_secret: Optional[str] = Header(None), dry_run: bool = Query(True)):
    """Delete all RunnerPredictionRow + HistoricalResultRow entries for a venue_code slug."""
    _check_admin(x_cron_secret)
    from sqlalchemy import delete as sa_delete
    pattern = f"%_{venue_code}_%"
    async with get_session() as session:
        pred_ids = (await session.execute(
            select(RunnerPredictionRow.race_id).where(RunnerPredictionRow.race_id.like(pattern)).distinct()
        )).scalars().all()
        hist_ids = (await session.execute(
            select(HistoricalResultRow.race_id).where(HistoricalResultRow.race_id.like(pattern)).distinct()
        )).scalars().all()
        if not dry_run:
            if pred_ids:
                await session.execute(sa_delete(RunnerPredictionRow).where(RunnerPredictionRow.race_id.like(pattern)))
            if hist_ids:
                await session.execute(sa_delete(HistoricalResultRow).where(HistoricalResultRow.race_id.like(pattern)))
            await session.commit()
    return {"dry_run": dry_run, "pred_race_ids": list(pred_ids), "hist_race_ids": list(hist_ids)}


@app.delete("/api/admin/purge-results/{race_date}/{venue_code}")
async def purge_results_for_venue(
    race_date: str,
    venue_code: str,
    x_cron_secret: Optional[str] = Header(None),
    dry_run: bool = Query(True),
):
    """Delete HistoricalResultRow entries for a specific date + venue only (predictions kept)."""
    _check_admin(x_cron_secret)
    _validate_date(race_date)
    _validate_venue(venue_code)
    from sqlalchemy import delete as sa_delete
    pattern = f"{race_date}_{venue_code}_R%"
    async with get_session() as session:
        hist_ids = (await session.execute(
            select(HistoricalResultRow.race_id)
            .where(HistoricalResultRow.race_id.like(pattern))
            .distinct()
        )).scalars().all()
        if not dry_run and hist_ids:
            await session.execute(
                sa_delete(HistoricalResultRow)
                .where(HistoricalResultRow.race_id.like(pattern))
            )
            await session.commit()
    return {"dry_run": dry_run, "deleted_from_race_ids": list(hist_ids)}


# ── Admin: seed results ───────────────────────────────────────────────────────

@app.post("/api/admin/results/{race_date}")
async def seed_results(race_date: str, x_cron_secret: Optional[str] = Header(None)):
    """Fetch race results from Racing Australia for a past date and store as training data."""
    _check_admin(x_cron_secret)
    seeded = await _seed_results_for_date(race_date)
    return {"status": "seeded", "results": seeded}


@app.post("/api/admin/seed-ra-results/{race_date}")
async def seed_ra_results(
    race_date: str,
    x_cron_secret: Optional[str] = Header(None),
    force: bool = Query(False),
):
    """
    Seed past results directly from Racing Australia Results.aspx, bypassing Calendar.aspx.
    Uses stored venue + state from predictions to construct RA keys — works for past dates
    where Calendar.aspx no longer lists meetings.
    force=true deletes existing HistoricalResultRow for each race before re-inserting,
    so stale/wrong rows are replaced with fresh RA data.
    """
    _check_admin(x_cron_secret)
    from sqlalchemy import delete as sa_delete

    client = get_tab_client()
    ra = client._ra

    # Find all predictions for this date. Mutable is the canonical source of
    # "which venues we predicted" — we need it to scope the venue/state lookup.
    # History is also fetched so feature_vector_json on the new HistoricalResultRow
    # rows can be sourced from the immutable pre-race snapshot when available
    # (BUG-36) rather than mutable, which can carry post-race contamination for
    # races without a snapshot.
    async with get_session() as session:
        pred_rows = (await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{race_date}_%"))
            .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
        )).scalars().all()

        hist_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{race_date}_%"))
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )).scalars().all()

    if not pred_rows:
        return {"status": "ok", "seeded": 0, "detail": "no predictions for this date"}

    # Index history by (race_id, normalized horse name) for fv_json lookup.
    hist_by_key: dict[tuple[str, str], RunnerPredictionHistoryRow] = {
        (h.race_id, _normalize_horse(h.horse_name)): h for h in hist_rows
    }

    # Group predictions by (venue_name, state) → list of race_ids
    venue_state_map: dict[tuple[str, str], set[str]] = {}
    for row in pred_rows:
        v = (row.venue or "").strip()
        s = (row.state or "").strip().upper()
        if v and s:
            venue_state_map.setdefault((v, s), set()).add(row.race_id)

    seeded_total = 0
    detail: list[dict] = []

    for (venue_name, state), race_ids in venue_state_map.items():
        ra_key, results = await ra.find_results(race_date, state, venue_name)
        if not results:
            detail.append({"venue": venue_name, "state": state, "races_found": 0})
            continue

        venue_code = _parse_race_id(list(race_ids)[0])[1]
        seeded_here = 0
        races_detail = []

        for race_num, race_data in results.items():
            race_id = f"{race_date}_{venue_code}_R{race_num}"
            runners = race_data.get("runners", {})  # {name_lower: {position, margin, sp}}
            runners_with_pos = [(n, rd["position"]) for n, rd in runners.items() if rd.get("position") and rd["position"] > 0]
            if not runners:
                races_detail.append({"race_id": race_id, "skip": "no runners"})
                continue

            deleted = 0
            async with get_session() as session:
                if force:
                    # Wipe existing rows for this race so fresh RA data replaces stale ones
                    res = await session.execute(
                        sa_delete(HistoricalResultRow)
                        .where(HistoricalResultRow.race_id == race_id)
                    )
                    deleted = res.rowcount
                    await session.commit()

                race_seeded = 0
                for name_lower, rd in runners.items():
                    pos = rd.get("position")
                    if not pos or pos <= 0:
                        continue
                    sp = rd.get("sp")
                    margin = float(rd.get("margin") or 0)

                    matched_pred = next(
                        (p for p in pred_rows if p.race_id == race_id
                         and _normalize_horse(p.horse_name) == _normalize_horse(name_lower)),
                        None,
                    )

                    if not force:
                        existing_at_pos = (await session.execute(
                            select(HistoricalResultRow.horse_name)
                            .where(HistoricalResultRow.race_id == race_id)
                            .where(HistoricalResultRow.position == pos)
                        )).scalars().all()
                        if any(_normalize_horse(h) == _normalize_horse(name_lower) for h in existing_at_pos):
                            continue

                    display_name = matched_pred.horse_name if matched_pred else name_lower.title()
                    # Prefer history's enriched_json (immutable pre-race snapshot)
                    # over mutable's, which may be post-race-contaminated for
                    # races without a snapshot (BUG-36).
                    matched_hist = hist_by_key.get((race_id, _normalize_horse(name_lower)))
                    fv_json = (matched_hist.enriched_json if matched_hist
                               else (matched_pred.enriched_json if matched_pred else None))
                    session.add(HistoricalResultRow(
                        race_id=race_id,
                        horse_name=display_name,
                        position=pos,
                        beaten_margin=margin,
                        winner=pos == 1,
                        placed=pos <= 3,
                        starting_price=sp,
                        feature_vector_json=fv_json,
                    ))
                    race_seeded += 1
                await session.commit()

            seeded_here += race_seeded
            races_detail.append({
                "race_id": race_id,
                "runners_total": len(runners),
                "runners_with_pos": len(runners_with_pos),
                "deleted": deleted,
                "seeded": race_seeded,
                "winner_from_ra": runners_with_pos[0] if runners_with_pos else None,
            })

        seeded_total += seeded_here
        detail.append({"venue": venue_name, "state": state, "ra_key": ra_key,
                       "races_found": len(results), "seeded": seeded_here, "races": races_detail})

    return {"status": "ok", "seeded": seeded_total, "detail": detail}


@app.post("/api/admin/patch-sp/{race_date}")
async def patch_sp(race_date: str, x_cron_secret: Optional[str] = Header(None)):
    """Patch null starting_price on existing HistoricalResultRow entries using RA Results.aspx."""
    _check_admin(x_cron_secret)
    from horse_engine.clients.racing_australia import _ra_date as _make_ra_date

    client = get_tab_client()
    ra = client._ra

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{race_date}_%"))
        )).scalars().all()

        null_sp_rows = (await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.like(f"{race_date}_%"))
            .where(HistoricalResultRow.starting_price.is_(None))
        )).scalars().all()

    if not null_sp_rows:
        return {"status": "ok", "patched": 0, "detail": "no null SPs to patch"}

    # Group predictions by venue/state for RA key construction
    venue_state_map: dict[tuple[str, str], str] = {}
    for row in pred_rows:
        v = (row.venue or "").strip()
        s = (row.state or "").strip().upper()
        _, vc, _ = _parse_race_id(row.race_id)
        if v and s and vc:
            venue_state_map[vc] = (v, s)

    ra_date_str = _make_ra_date(race_date)
    patched = 0
    detail: list[dict] = []

    # Group null-SP rows by venue_code
    vc_rows: dict[str, list[HistoricalResultRow]] = {}
    for hr in null_sp_rows:
        _, vc, _ = _parse_race_id(hr.race_id)
        if vc:
            vc_rows.setdefault(vc, []).append(hr)

    for vc, hr_list in vc_rows.items():
        vs = venue_state_map.get(vc)
        if not vs:
            detail.append({"venue_code": vc, "skipped": "no venue/state in predictions"})
            continue
        venue_name, state = vs
        ra_key = f"{ra_date_str},{state},{venue_name}"
        results = await ra.get_results(ra_key)
        if not results:
            detail.append({"venue_code": vc, "ra_key": ra_key, "skipped": "no RA results"})
            continue

        patched_here = 0
        for hr in hr_list:
            _, _, race_num = _parse_race_id(hr.race_id)
            race_data = results.get(race_num, {})
            runners = race_data.get("runners", {})
            ra_runner = runners.get(_normalize_horse(hr.horse_name))
            if not ra_runner:
                continue
            sp = ra_runner.get("sp")
            if not sp:
                continue
            async with get_session() as session:
                row = (await session.execute(
                    select(HistoricalResultRow).where(HistoricalResultRow.id == hr.id)
                )).scalars().first()
                if row and row.starting_price is None:
                    row.starting_price = sp
                    await session.commit()
                    patched_here += 1

        patched += patched_here
        detail.append({"venue_code": vc, "ra_key": ra_key, "patched": patched_here})

    return {"status": "ok", "patched": patched, "detail": detail}


@app.get("/api/meetings/{race_date}/{venue_code}/results")
async def get_meeting_results(race_date: str, venue_code: str):
    """
    Return today's race results for a single venue, fetched live from Racing Australia.
    Also seeds HistoricalResultRow so model-correct dots appear immediately.
    """
    _validate_date(race_date)
    _validate_venue(venue_code)

    from horse_engine.clients.composite import CompositeClient
    client = get_tab_client()
    slug = _meeting_slug(venue_code, race_date)

    # Prime slug→key cache then get the RA internal key
    await client.get_meeting_by_slug(slug)
    ra_key = client._ra._slug_to_key.get(slug) if hasattr(client, "_ra") else None
    if not ra_key:
        return {"date": race_date, "venue": venue_code, "races": [], "seeded": 0}

    ra_results = await client._ra.get_results(ra_key)
    if not ra_results:
        return {"date": race_date, "venue": venue_code, "races": [], "seeded": 0}

    # Seed HistoricalResultRow from RA results
    seeded = 0
    races_with_results: set[str] = set()
    for race_num, race_data in ra_results.items():
        race_id = f"{race_date}_{venue_code}_R{race_num}"
        for name_lower, rd in race_data.get("runners", {}).items():
            position = rd.get("position")
            if not position or position <= 0:
                continue
            # RA result names may include country codes (NZ, FR, IRE) — match on both
            # exact name and normalized (country-code-stripped) name to avoid duplicates
            norm_name = _normalize_horse(name_lower)
            async with get_session() as session:
                existing_rows = (await session.execute(
                    select(HistoricalResultRow.horse_name)
                    .where(HistoricalResultRow.race_id == race_id)
                    .where(HistoricalResultRow.position == position)
                )).scalars().all()
                already_exists = any(
                    _normalize_horse(h) == norm_name for h in existing_rows
                )
                if already_exists:
                    races_with_results.add(race_id)
                    continue
                fv_result = await session.execute(
                    select(RunnerPredictionRow)
                    .where(RunnerPredictionRow.race_id == race_id)
                    .where(func.lower(RunnerPredictionRow.horse_name).in_([name_lower, norm_name]))
                    .limit(1)
                )
                fv_row = fv_result.scalars().first()
                horse_name = fv_row.horse_name if fv_row else name_lower.upper()
                session.add(HistoricalResultRow(
                    race_id=race_id,
                    horse_name=horse_name,
                    position=position,
                    beaten_margin=float(rd.get("margin") or 0),
                    winner=position == 1,
                    placed=position <= 3,
                    starting_price=rd.get("sp"),
                    feature_vector_json=fv_row.enriched_json if fv_row else None,
                ))
                await session.commit()
                seeded += 1
                races_with_results.add(race_id)

    # Load top model picks from history — history is written once pre-race and
    # is never overwritten, so it always holds the genuine pre-race prediction.
    top_picks: dict[str, str] = {}
    if races_with_results:
        async with get_session() as session:
            hist_rows = (await session.execute(
                select(RunnerPredictionHistoryRow.race_id, RunnerPredictionHistoryRow.horse_name,
                       RunnerPredictionHistoryRow.enriched_at)
                .where(RunnerPredictionHistoryRow.race_id.in_(list(races_with_results)))
                .where(RunnerPredictionHistoryRow.model_rank == 1)
                .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
            )).all()
        seen: set[str] = set()
        for r in hist_rows:
            if r.race_id not in seen:
                seen.add(r.race_id)
                top_picks[r.race_id] = r.horse_name

    # Build response
    races_out = []
    for race_num in sorted(ra_results.keys()):
        race_data = ra_results[race_num]
        race_id = f"{race_date}_{venue_code}_R{race_num}"
        runners_raw = race_data.get("runners", {})
        runners_sorted = sorted(
            [(name, rd) for name, rd in runners_raw.items() if rd.get("position")],
            key=lambda x: x[1]["position"],
        )
        winner = runners_sorted[0][0] if runners_sorted else None
        top_pick = top_picks.get(race_id) or ""
        # Normalize both sides to strip country codes (NZ, FR, IRE etc.) before comparing
        norm_pick = _normalize_horse(top_pick)
        model_correct = (_normalize_horse(norm_pick) == _normalize_horse(winner)) if (norm_pick and winner) else None
        model_placed = (norm_pick in {_normalize_horse(n) for n, rd in runners_sorted if rd["position"] <= 3}) if norm_pick else None

        races_out.append({
            "race_number": race_num,
            "race_id": race_id,
            "track_condition": race_data.get("track_condition", ""),
            "has_result": bool(runners_sorted),
            "model_correct": model_correct,
            "model_placed": model_placed,
            "top_pick": top_picks.get(race_id),
            "runners": [
                {
                    "position": rd["position"],
                    "name": name.upper(),
                    "margin": rd.get("margin", 0),
                    "sp": rd.get("sp"),
                }
                for name, rd in runners_sorted
            ],
        })

    return {"date": race_date, "venue": venue_code, "races": races_out, "seeded": seeded}


# ── Backfill ──────────────────────────────────────────────────────────────────

_backfill: dict = {"running": False, "done": False, "current": None,
                   "completed": [], "errors": [], "meetings": 0, "races": 0, "runners": 0}


async def _run_backfill(days: int, x_secret: Optional[str], force: bool = False, holdout_from: str = "2026-05-01"):
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
            if race_date >= holdout_from:
                log.info("[backfill] Skipping %s — holdout period (>= %s)", race_date, holdout_from)
                continue
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

                    raw_races = await client.get_meeting_races(slug)
                    for raw_race in raw_races:
                        race_num = raw_race.get("eventNumber")
                        race_id = f"{race_date}_{venue_code}_R{race_num}"
                        full_race = await client.get_race(slug, race_num)
                        if not full_race:
                            continue
                        try:
                            race = await client.parse_race(full_race, race_date, venue_name, state)
                            if not race.runners:
                                continue

                            # Inject SP as tote_win_odds so compute_market_features derives
                            # correct market_rank for historical races (no live odds available).
                            sp_by_name: dict[str, float] = {}
                            for r in full_race.get("runners", []):
                                if r.get("scratched"):
                                    continue
                                horse = (r.get("runnerName") or "").lower()
                                for p in r.get("prices", []):
                                    if p.get("priceType") in ("StartingPrice", "SP", "Win"):
                                        try:
                                            sp_val = float(p.get("winPrice") or 0)
                                            if sp_val > 1.0 and horse not in sp_by_name:
                                                sp_by_name[horse] = sp_val
                                                break
                                        except (TypeError, ValueError):
                                            pass
                            if sp_by_name:
                                for runner in race.runners:
                                    h = runner.horse_name.lower()
                                    if h in sp_by_name and not (runner.fixed_win_odds or runner.tote_win_odds):
                                        runner.tote_win_odds = sp_by_name[h]
                                log.debug("[backfill] %s R%s: injected SP for %d/%d runners",
                                          venue_code, race_num, len(sp_by_name), len(race.runners))

                            async with get_session() as session:
                                await _inject_accumulated_stats(race, session)
                            predictions, _ = await enrich_and_predict_race(
                                race, model
                            )
                            db_dicts = [_prediction_to_db_dict(p, race_id, race.scheduled_time, race=race) for p in predictions]
                            async with get_session() as session:
                                await save_race_predictions(session, race_id, db_dicts)
                                # Write immutable history snapshot for training (skip if live snapshot exists)
                                existing_hist = (await session.execute(
                                    select(RunnerPredictionHistoryRow.id)
                                    .where(RunnerPredictionHistoryRow.race_id == race_id)
                                    .limit(1)
                                )).scalar()
                                if not existing_hist:
                                    now = datetime.utcnow()
                                    for p, d in zip(predictions, db_dicts):
                                        session.add(RunnerPredictionHistoryRow(
                                            race_id=race_id,
                                            horse_name=d["horse_name"],
                                            tab_number=d.get("tab_number"),
                                            barrier=d.get("barrier"),
                                            jockey=d.get("jockey"),
                                            trainer=d.get("trainer"),
                                            weight=d.get("weight"),
                                            win_probability=d["win_probability"],
                                            place_probability=d.get("place_probability"),
                                            model_rank=d["model_rank"],
                                            place_model_rank=d.get("place_model_rank"),
                                            exotic_model_rank=d.get("exotic_model_rank"),
                                            market_rank=d.get("market_rank"),
                                            overlay=d.get("overlay"),
                                            best_available_odds=d.get("best_available_odds"),
                                            value_rating=d.get("value_rating"),
                                            key_flags=d.get("key_flags"),
                                            enriched_json=d.get("enriched_json"),
                                            scheduled_time=d.get("scheduled_time"),
                                            venue=d.get("venue"),
                                            state=d.get("state"),
                                            race_number=d.get("race_number"),
                                            race_name=d.get("race_name"),
                                            distance=d.get("distance"),
                                            track_condition=d.get("track_condition"),
                                            field_size=d.get("field_size"),
                                            prize_money=d.get("prize_money"),
                                            rail_position=d.get("rail_position"),
                                            class_change=d.get("class_change"),
                                            enriched_at=now,
                                            source="backfill",
                                        ))
                                    await session.commit()
                                    log.debug("[backfill] Wrote %d history rows for %s", len(predictions), race_id)
                            # Seed actual results from runner finishing positions
                            # Build selection lookup for rich context
                            bf_sel_by_name = {}
                            for sel in full_race.get("selections") or []:
                                sel_name = ((sel.get("competitor") or {}).get("name") or "").lower()
                                if sel_name:
                                    bf_sel_by_name[sel_name] = sel
                            bf_meeting = full_race.get("_meeting") or {}
                            bf_venue = bf_meeting.get("venue") or venue_code
                            bf_state = bf_meeting.get("state") or ""
                            raw_race_obj = next((rr for rr in raw_races if rr.get("eventNumber") == race_num), {})
                            bf_dist = int(raw_race_obj.get("distance") or 0) or None
                            bf_tc = (raw_race_obj.get("trackCondition") or {})
                            bf_cond = f"{bf_tc.get('overall','')} {bf_tc.get('rating','')}".strip() or None
                            bf_class = raw_race_obj.get("eventClass") or None
                            bf_field = len([rr for rr in full_race.get("runners", []) if not rr.get("scratched")]) or None

                            for r in full_race.get("runners", []):
                                if r.get("scratched"):
                                    continue
                                position = r.get("finishingPosition")
                                if not position or int(position) <= 0:
                                    continue
                                horse = r.get("runnerName", "")
                                beaten = float(r.get("margin", 0) or 0)
                                sp = None
                                for p in r.get("prices", []):
                                    if p.get("priceType") in ("StartingPrice", "SP"):
                                        sp = float(p.get("winPrice", 0) or 0) or None
                                        break
                                sel = bf_sel_by_name.get(horse.lower()) or {}
                                jockey = (sel.get("jockey") or {}).get("name") or None
                                trainer = (sel.get("trainer") or {}).get("name") or None
                                barrier = int(sel.get("barrierNumber") or 0) or None
                                tab_num = int(sel.get("competitorNumber") or 0) or None
                                wt = float(sel.get("weight") or 0) or None
                                comp = sel.get("competitor") or {}
                                age = int(comp.get("age") or 0) or None
                                sex = comp.get("sex") or None
                                async with get_session() as session:
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
                                        session.add(HistoricalResultRow(
                                            race_id=race_id,
                                            horse_name=horse,
                                            position=int(position),
                                            beaten_margin=beaten,
                                            winner=int(position) == 1,
                                            placed=int(position) <= 3,
                                            starting_price=sp,
                                            feature_vector_json=fv_row.enriched_json if fv_row else None,
                                            jockey=jockey,
                                            trainer=trainer,
                                            venue=bf_venue or None,
                                            state=bf_state or None,
                                            distance=bf_dist,
                                            track_condition=bf_cond,
                                            barrier=barrier,
                                            tab_number=tab_num,
                                            weight=wt,
                                            age=age,
                                            sex=sex,
                                            race_class=bf_class,
                                            field_size=bf_field,
                                            race_number=race_num,
                                        ))
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
    holdout_from: str = Query("2026-05-01", description="Skip dates >= this date (holdout period for evaluation)"),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Start background backfill of past N days.
    force=true re-runs predictions even for dates already in the DB (useful after model updates).
    holdout_from skips dates >= that value so May 2026 stays clean for evaluation.
    Historical results are never duplicated regardless.
    Writes to RunnerPredictionHistoryRow with source='backfill' for training.
    SP from race results is injected as tote_win_odds so market_rank_norm is real.
    """
    _check_admin(x_cron_secret)
    if _backfill["running"]:
        raise HTTPException(409, "Backfill already running")
    asyncio.create_task(_run_backfill(days, x_cron_secret, force=force, holdout_from=holdout_from))
    return {"status": "started", "days": days, "force": force, "holdout_from": holdout_from,
            "message": "Check /api/admin/backfill/status for progress"}


@app.get("/api/admin/backfill/status")
async def backfill_status():
    """Current backfill progress."""
    return _backfill


# ── DB-based backfill: build training history from historical_results ─────────

_db_backfill: dict = {"running": False, "done": False}


async def _run_db_backfill(holdout_from: str = "2026-05-01", batch_size: int = 50, force: bool = False,
                           offset: int = 0, limit: int = 0) -> None:
    """
    Build RunnerPredictionHistoryRow training snapshots directly from historical_results.
    Does not call any external API — works entirely from stored DB data.
    Processes races in chronological order, oldest first.
    offset/limit: process a slice of races (for chunked runs). limit=0 means all.
    """
    global _db_backfill
    from horse_engine.models.race import Race, Runner
    from sqlalchemy import delete as _sa_delete, update as _sa_update

    _db_backfill.update({
        "running": True, "done": False, "races": 0, "runners": 0,
        "skipped": 0, "errors": 0, "started_at": datetime.utcnow().isoformat(),
        "offset": offset, "limit": limit,
    })
    try:
        async with get_session() as session:
            model = await _load_model(session)

        # Fetch all eligible race_ids from historical_results (before holdout, has results)
        async with get_session() as session:
            rows = (await session.execute(
                select(HistoricalResultRow.race_id)
                .where(HistoricalResultRow.race_id < holdout_from)
                .where(HistoricalResultRow.position.isnot(None))
                .distinct()
                .order_by(HistoricalResultRow.race_id)
            )).scalars().all()
        all_race_ids = list(rows)
        race_ids = all_race_ids[offset: offset + limit] if limit > 0 else all_race_ids[offset:]
        _db_backfill["total_races"] = len(all_race_ids)
        _db_backfill["chunk_races"] = len(race_ids)
        log.info("[db-backfill] %d total eligible races, processing %d (offset=%d limit=%d)",
                 len(all_race_ids), len(race_ids), offset, limit)

        for race_id in race_ids:
            try:
                # Skip if history snapshot already exists (unless force=True)
                async with get_session() as session:
                    exists = (await session.execute(
                        select(RunnerPredictionHistoryRow.id)
                        .where(RunnerPredictionHistoryRow.race_id == race_id)
                        .limit(1)
                    )).scalar()
                if exists:
                    if not force:
                        _db_backfill["skipped"] += 1
                        continue
                    # force=True: fall through — will UPDATE enriched_json on existing rows

                # Load all runners for this race from historical_results
                async with get_session() as session:
                    hr_rows = (await session.execute(
                        select(HistoricalResultRow)
                        .where(HistoricalResultRow.race_id == race_id)
                        .where(HistoricalResultRow.position.isnot(None))
                        .order_by(HistoricalResultRow.tab_number, HistoricalResultRow.barrier)
                    )).scalars().all()

                if len(hr_rows) < 2:
                    _db_backfill["skipped"] += 1
                    continue

                race_date, venue_code, race_num = _parse_race_id(race_id)
                if not race_date or not race_num:
                    continue

                # Use first row for race-level context
                ref = hr_rows[0]
                venue = ref.venue or venue_code.replace("-", " ").title()
                state = ref.state or "NSW"
                distance = ref.distance or 1200
                track_cond = ref.track_condition or "Good"
                race_class = ref.race_class or ""
                prize_money = ref.prize_money or 0
                field_size = ref.field_size or len(hr_rows)

                # Build Runner objects — SP injected as tote_win_odds for real market_rank
                runners: list[Runner] = []
                for hr in hr_rows:
                    runners.append(Runner(
                        horse_name=hr.horse_name or "",
                        tab_number=hr.tab_number or 0,
                        barrier=hr.barrier or 0,
                        jockey=hr.jockey or "",
                        trainer=hr.trainer or "",
                        weight=float(hr.weight or 58.0),
                        age=int(hr.age or 0),
                        sex=hr.sex or "G",
                        colour="",
                        country="AUS",
                        career_starts=0,
                        career_wins=0,
                        career_places=0,
                        tote_win_odds=float(hr.starting_price) if hr.starting_price and hr.starting_price > 1.0 else None,
                    ))

                race = Race(
                    race_id=race_id,
                    date=race_date,
                    venue=venue,
                    state=state,
                    race_number=race_num,
                    race_name=f"Race {race_num}",
                    race_class=race_class,
                    distance=distance,
                    track_condition=track_cond,
                    rail_position="",
                    prize_money=prize_money,
                    scheduled_time=f"{race_date}T10:00:00",
                    race_type="R",
                    runners=runners,
                )

                # Inject accumulated stats from DB (jockey/trainer/career/form)
                async with get_session() as session:
                    await _inject_accumulated_stats(race, session)

                predictions, _ = await enrich_and_predict_race(race, model)
                if not predictions:
                    _db_backfill["skipped"] += 1
                    continue

                db_dicts = [_prediction_to_db_dict(p, race_id, race.scheduled_time, race=race) for p in predictions]

                async with get_session() as session:
                    # Write to mutable predictions table
                    await save_race_predictions(session, race_id, db_dicts)

                    if exists and force:
                        # UPDATE enriched_json on existing rows (no duplicates)
                        for d in db_dicts:
                            if d.get("enriched_json"):
                                await session.execute(
                                    _sa_update(RunnerPredictionHistoryRow)
                                    .where(RunnerPredictionHistoryRow.race_id == race_id)
                                    .where(func.lower(RunnerPredictionHistoryRow.horse_name) == func.lower(d["horse_name"]))
                                    .values(enriched_json=d["enriched_json"])
                                )
                    else:
                        # INSERT new history snapshot rows
                        now = datetime.utcnow()
                        for hr, d in zip(hr_rows, db_dicts):
                            session.add(RunnerPredictionHistoryRow(
                                race_id=race_id,
                                horse_name=d["horse_name"],
                                tab_number=d.get("tab_number"),
                                barrier=d.get("barrier"),
                                jockey=d.get("jockey"),
                                trainer=d.get("trainer"),
                                weight=d.get("weight"),
                                win_probability=d["win_probability"],
                                place_probability=d.get("place_probability"),
                                model_rank=d["model_rank"],
                                place_model_rank=d.get("place_model_rank"),
                                exotic_model_rank=d.get("exotic_model_rank"),
                                market_rank=d.get("market_rank"),
                                overlay=d.get("overlay"),
                                best_available_odds=d.get("best_available_odds"),
                                value_rating=d.get("value_rating"),
                                key_flags=d.get("key_flags"),
                                enriched_json=d.get("enriched_json"),
                                scheduled_time=d.get("scheduled_time"),
                                venue=d.get("venue"),
                                state=d.get("state"),
                                race_number=d.get("race_number"),
                                race_name=d.get("race_name"),
                                distance=d.get("distance"),
                                track_condition=d.get("track_condition"),
                                field_size=d.get("field_size"),
                                prize_money=d.get("prize_money"),
                                rail_position=d.get("rail_position"),
                                class_change=d.get("class_change"),
                                enriched_at=now,
                                source="backfill",
                            ))

                    # Update feature_vector_json on historical_results
                    for hr in hr_rows:
                        d = next((x for x in db_dicts if x["horse_name"] == hr.horse_name), None)
                        if d and d.get("enriched_json"):
                            hr_row = (await session.execute(
                                select(HistoricalResultRow)
                                .where(HistoricalResultRow.id == hr.id)
                            )).scalars().first()
                            if hr_row:
                                hr_row.feature_vector_json = d["enriched_json"]

                    await session.commit()

                _db_backfill["races"] += 1
                _db_backfill["runners"] += len(predictions)

                if _db_backfill["races"] % batch_size == 0:
                    log.info("[db-backfill] %d races done, %d runners, %d skipped",
                             _db_backfill["races"], _db_backfill["runners"], _db_backfill["skipped"])
                    await asyncio.sleep(0.1)  # yield to event loop

            except Exception as e:
                log.warning("[db-backfill] Race %s failed: %s", race_id, e)
                _db_backfill["errors"] += 1

    finally:
        _db_backfill.update({
            "running": False, "done": True,
            "finished_at": datetime.utcnow().isoformat(),
        })
        log.info("[db-backfill] Done — %d races, %d runners, %d skipped, %d errors",
                 _db_backfill["races"], _db_backfill["runners"],
                 _db_backfill["skipped"], _db_backfill["errors"])


@app.post("/api/admin/backfill/from-db")
async def start_db_backfill(
    holdout_from: str = Query("2026-05-01", description="Skip races on or after this date"),
    force: bool = Query(False, description="Overwrite existing enriched_json (refreshes feature vectors)"),
    offset: int = Query(0, ge=0, description="Skip this many races (for chunked runs)"),
    limit: int = Query(0, ge=0, description="Process at most this many races (0 = all)"),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Build RunnerPredictionHistoryRow training data directly from historical_results.
    No external API calls — processes stored race rows using DB-joined stats.
    Use offset+limit to process in chunks (e.g. limit=500 per call) to avoid Railway timeouts.
    Use force=true to regenerate enriched_json for already-processed races.
    """
    _check_admin(x_cron_secret)
    if _db_backfill.get("running"):
        raise HTTPException(409, "DB backfill already running")
    asyncio.create_task(_run_db_backfill(holdout_from=holdout_from, force=force, offset=offset, limit=limit))
    return {"status": "started", "holdout_from": holdout_from, "offset": offset, "limit": limit,
            "message": "Check /api/admin/backfill/from-db/status for progress"}


@app.get("/api/admin/backfill/from-db/status")
async def db_backfill_status():
    """Current DB backfill progress."""
    return _db_backfill


@app.post("/api/admin/history/snapshot")
async def trigger_prerace_snapshot(x_cron_secret: Optional[str] = Header(None)):
    """Manually trigger the pre-race snapshot job for today."""
    _check_admin(x_cron_secret)
    written = await _snapshot_prerace_predictions()
    return {"status": "ok", "races_written": written}


@app.post("/api/admin/history/backfill")
async def backfill_history(x_cron_secret: Optional[str] = Header(None)):
    """
    Copy all valid pre-race RunnerPredictionRows into the immutable history table.
    Safe to re-run — skips races already recorded. Returns count of races copied.
    """
    _check_admin(x_cron_secret)
    async with get_session() as session:
        copied = await backfill_prediction_history(session)
    return {"status": "ok", "races_copied": copied}


@app.post("/api/admin/history/snapshot-backfill")
async def snapshot_backfill(
    date: Optional[str] = Query(None),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Retroactively fill missing history snapshots for a given date from RunnerPredictionRow.
    Only writes races NOT already in history — never modifies existing rows.
    Rows are flagged source='late' so they can be audited separately.
    Safe to run multiple times (idempotent).
    Defaults to yesterday if no date supplied.
    """
    _check_admin(x_cron_secret)
    target_date = date or (_today_aest() - timedelta(days=1)).isoformat()
    now_utc = datetime.utcnow()

    async with get_session() as session:
        already_result = await session.execute(
            select(RunnerPredictionHistoryRow.race_id)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{target_date}_%"))
            .distinct()
        )
        already_set = set(already_result.scalars().all())

        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{target_date}_%"))
            .where(RunnerPredictionRow.win_probability.isnot(None))
            .where(RunnerPredictionRow.enriched_json.isnot(None))
        )
        rows = pred_result.scalars().all()

    if not rows:
        return {"status": "ok", "date": target_date, "written": 0, "message": "No predictions found for this date"}

    race_groups: dict[str, list] = {}
    for r in rows:
        if r.race_id not in already_set:
            race_groups.setdefault(r.race_id, []).append(r)

    if not race_groups:
        return {"status": "ok", "date": target_date, "written": 0,
                "skipped_already_in_history": len(already_set),
                "message": "All races already have history snapshots"}

    written = 0
    async with get_session() as session:
        for race_id, runners in race_groups.items():
            # Copy model_rank / place_model_rank / exotic_model_rank verbatim from
            # mutable per BUG-07's fix. Recomputing rank by sorting on win_probability
            # would lose any overlay-adjusted or tie-broken rank ordering that
            # mutable holds. Fall back to a win_probability sort only when mutable
            # has no rank populated (legacy rows).
            for r in runners:
                session.add(RunnerPredictionHistoryRow(
                    race_id=r.race_id, horse_name=r.horse_name,
                    tab_number=r.tab_number, barrier=r.barrier,
                    jockey=r.jockey, trainer=r.trainer, weight=r.weight,
                    win_probability=r.win_probability, place_probability=r.place_probability,
                    model_rank=r.model_rank,
                    place_model_rank=r.place_model_rank,
                    exotic_model_rank=r.exotic_model_rank,
                    market_rank=r.market_rank, overlay=r.overlay,
                    best_available_odds=r.best_available_odds, value_rating=r.value_rating,
                    key_flags=r.key_flags, enriched_json=r.enriched_json,
                    scheduled_time=r.scheduled_time, enriched_at=r.enriched_at or now_utc,
                    cancelled=r.cancelled, venue=r.venue, state=r.state,
                    race_number=r.race_number, race_name=r.race_name,
                    distance=r.distance, track_condition=r.track_condition,
                    field_size=r.field_size, prize_money=r.prize_money,
                    rail_position=getattr(r, "rail_position", None),
                    class_change=getattr(r, "class_change", None),
                    source="late",  # retroactive fill — flagged for auditing
                    recorded_at=now_utc,
                ))
            # Fallback: if mutable has no model_rank for any runner in this race,
            # derive one from win_probability so downstream readers can still
            # locate a rank-1 row (legacy pre-rank-tracking data).
            if all((r.model_rank is None) for r in runners):
                from sqlalchemy import update as sa_update
                wp_sorted = sorted(runners, key=lambda x: x.win_probability or 0, reverse=True)
                for rank, r in enumerate(wp_sorted, 1):
                    await session.execute(
                        sa_update(RunnerPredictionHistoryRow)
                        .where(RunnerPredictionHistoryRow.race_id == r.race_id)
                        .where(RunnerPredictionHistoryRow.horse_name == r.horse_name)
                        .values(model_rank=rank)
                    )
            await session.commit()
            written += 1

    log.info("[snapshot-backfill] Wrote %d races for %s", written, target_date)
    return {
        "status": "ok",
        "date": target_date,
        "written": written,
        "skipped_already_in_history": len(already_set),
    }


@app.post("/api/admin/history/clear-stale")
async def clear_stale_history(
    date: str = Query(..., description="Date to clear stale history for (YYYY-MM-DD)"),
    before_date: str = Query(..., description="Delete history rows with enriched_at before this date (YYYY-MM-DD), e.g. today's date to remove rows written before today"),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Delete history rows for a given race date where enriched_at predates before_date.
    Used to clear stale pre-enriched snapshots (e.g., races enriched 2 days early with
    null odds/features) so fresh pre-race snapshots can be written in their place.
    Only deletes rows for unsettled races (no HistoricalResultRow).
    """
    _check_admin(x_cron_secret)
    from sqlalchemy import delete as sa_delete

    cutoff_dt = datetime.fromisoformat(before_date)

    async with get_session() as session:
        # Safety: only delete rows for races with no result yet
        settled = {r for r, in (await session.execute(
            select(HistoricalResultRow.race_id)
            .where(HistoricalResultRow.race_id.like(f"{date}_%"))
            .distinct()
        )).all()}

        result = await session.execute(
            sa_delete(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{date}_%"))
            .where(RunnerPredictionHistoryRow.enriched_at < cutoff_dt)
            .where(RunnerPredictionHistoryRow.race_id.notin_(settled))
        )
        await session.commit()
        deleted = result.rowcount

    return {"status": "ok", "date": date, "deleted_rows": deleted, "protected_settled": len(settled)}


@app.post("/api/admin/history/patch-nulls")
async def patch_history_nulls(x_cron_secret: Optional[str] = Header(None)):
    """
    Patch NULL race-level fields (venue, state, distance, etc.) in existing history rows
    by joining against RacePredictionRow and enriched_json.
    """
    from sqlalchemy import update as sa_update
    _check_admin(x_cron_secret)
    patched = 0
    async with get_session() as session:
        # Find history rows with missing race-level fields
        null_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.venue.is_(None))
        )).scalars().all()

        if not null_rows:
            return {"status": "ok", "patched": 0}

        # Load RacePredictionRow lookup
        race_ids = list({r.race_id for r in null_rows})
        rp_rows = (await session.execute(
            select(RacePredictionRow).where(RacePredictionRow.race_id.in_(race_ids))
        )).scalars().all()
        rp_map = {r.race_id: r for r in rp_rows}

        for row in null_rows:
            rp = rp_map.get(row.race_id)
            updates: dict = {}
            if rp:
                updates["venue"] = rp.venue
                updates["state"] = rp.state
                updates["race_number"] = rp.race_number
                updates["race_name"] = rp.race_name
                updates["distance"] = rp.distance
                updates["track_condition"] = rp.track_condition
                updates["field_size"] = rp.field_size
                updates["prize_money"] = rp.prize_money
            # Pull class_change from enriched_json
            if row.class_change is None and row.enriched_json:
                try:
                    ej = json.loads(row.enriched_json)
                    cc = ej.get("class_change")
                    if cc is not None:
                        updates["class_change"] = int(cc)
                except Exception:
                    pass
            if updates:
                await session.execute(
                    sa_update(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.id == row.id)
                    .values(**updates)
                )
                patched += 1

        await session.commit()
    return {"status": "ok", "patched": patched}


@app.post("/api/admin/backfill-odds-from-sp")
async def backfill_odds_from_sp(x_cron_secret: Optional[str] = Header(None)):
    """
    Backfill best_available_odds from starting_price for RunnerPredictionRow rows
    where best_available_odds is NULL. Also patches RunnerPredictionHistoryRow.
    Recomputes overlay and value_rating from SP. Safe to re-run — skips rows
    that already have best_available_odds set.
    """
    from sqlalchemy import update as sa_update
    _check_admin(x_cron_secret)

    updated_live = 0
    updated_hist = 0

    # ── 1. RunnerPredictionRow (mutable table) ────────────────────────────────
    async with get_session() as session:
        null_rows = (await session.execute(
            select(RunnerPredictionRow)
            .where(
                (RunnerPredictionRow.best_available_odds.is_(None)) |
                (RunnerPredictionRow.best_available_odds <= 1.0)
            )
            .where(RunnerPredictionRow.race_id.isnot(None))
        )).scalars().all()

        if null_rows:
            race_ids = list({r.race_id for r in null_rows})
            hr_rows = (await session.execute(
                select(HistoricalResultRow)
                .where(HistoricalResultRow.race_id.in_(race_ids))
                .where(HistoricalResultRow.starting_price.isnot(None))
            )).scalars().all()
            sp_map: dict[tuple[str, str], float] = {
                (r.race_id, r.horse_name.lower()): r.starting_price
                for r in hr_rows if r.starting_price and r.starting_price > 1.0
            }

            for row in null_rows:
                sp = sp_map.get((row.race_id, row.horse_name.lower()))
                if not sp:
                    continue
                db_row = await session.get(RunnerPredictionRow, row.id)
                if not db_row:
                    continue
                db_row.best_available_odds = sp
                market_implied = 1.0 / sp
                db_row.overlay = round(db_row.win_probability - market_implied, 4)
                db_row.value_rating = _value_rating(db_row.win_probability, sp, db_row.overlay)
                updated_live += 1

        await session.commit()

    # ── 2. RunnerPredictionHistoryRow (immutable history) ─────────────────────
    async with get_session() as session:
        null_hist = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(
                (RunnerPredictionHistoryRow.best_available_odds.is_(None)) |
                (RunnerPredictionHistoryRow.best_available_odds <= 1.0)
            )
            .where(RunnerPredictionHistoryRow.race_id.isnot(None))
        )).scalars().all()

        if null_hist:
            race_ids = list({r.race_id for r in null_hist})
            hr_rows = (await session.execute(
                select(HistoricalResultRow)
                .where(HistoricalResultRow.race_id.in_(race_ids))
                .where(HistoricalResultRow.starting_price.isnot(None))
            )).scalars().all()
            sp_map_hist: dict[tuple[str, str], float] = {
                (r.race_id, r.horse_name.lower()): r.starting_price
                for r in hr_rows if r.starting_price and r.starting_price > 1.0
            }

            for row in null_hist:
                sp = sp_map_hist.get((row.race_id, row.horse_name.lower()))
                if not sp:
                    continue
                db_row = await session.get(RunnerPredictionHistoryRow, row.id)
                if not db_row:
                    continue
                db_row.best_available_odds = sp
                market_implied = 1.0 / sp
                db_row.overlay = round(db_row.win_probability - market_implied, 4)
                db_row.value_rating = _value_rating(db_row.win_probability, sp, db_row.overlay)
                updated_hist += 1

        await session.commit()

    log.info("[backfill-odds] Patched %d live rows, %d history rows from SP", updated_live, updated_hist)
    return {"status": "ok", "updated_live": updated_live, "updated_history": updated_hist}


@app.post("/api/admin/patch-betfair-bsp")
async def patch_betfair_bsp(
    payload: dict,
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Accept a batch of Betfair BSP + LTP snapshot data and patch the DB.

    Payload:
        {
          "races": [
            {
              "race_id": "2026-05-15_warwick-farm_R3",
              "runners": [
                {
                  "name": "Dark Fox",
                  "bsp": 4.2,
                  "snapshots": [
                    {"minutes_to_jump": 62.1, "snapshotted_at": "2026-05-15T02:00:00Z", "win_odds": 5.5},
                    ...
                  ]
                },
                ...
              ]
            },
            ...
          ]
        }

    Idempotent: skips rows that already have starting_price set; skips snapshots
    that already exist within 2 minutes of the stored time.
    """
    from sqlalchemy import text
    _check_admin(x_cron_secret)
    races = payload.get("races") or []
    bsp_patched = bsp_skipped = pred_patched = snap_inserted = 0

    _debug_err = None
    async with get_session() as session:
        for race in races:
            race_id = race.get("race_id", "")
            for runner in race.get("runners") or []:
                name = runner.get("name", "")
                bsp = runner.get("bsp")
                snapshots = runner.get("snapshots") or []

                # Wrap each runner in a savepoint so an error only rolls back that runner
                try:
                    async with session.begin_nested():
                        if bsp:
                            # Patch historical_results.starting_price
                            res = await session.execute(text(
                                "UPDATE historical_results SET starting_price = :bsp "
                                "WHERE race_id = :rid AND LOWER(horse_name) = LOWER(:name) "
                                "AND (starting_price IS NULL OR starting_price = 0)"
                            ), {"bsp": float(bsp), "rid": race_id, "name": name})
                            if res.rowcount:
                                bsp_patched += res.rowcount
                            else:
                                bsp_skipped += 1

                            # Patch runner_prediction_history.best_available_odds
                            res2 = await session.execute(text(
                                "UPDATE runner_prediction_history SET best_available_odds = :bsp "
                                "WHERE race_id = :rid AND LOWER(horse_name) = LOWER(:name) "
                                "AND (best_available_odds IS NULL OR best_available_odds = 0)"
                            ), {"bsp": float(bsp), "rid": race_id, "name": name})
                            pred_patched += res2.rowcount

                        for snap in snapshots:
                            mtj = int(round(float(snap.get("minutes_to_jump") or 0)))
                            snap_at_str = snap.get("snapshotted_at", "")
                            win_odds_val = snap.get("win_odds")
                            if not (snap_at_str and win_odds_val):
                                continue
                            try:
                                snap_dt = datetime.fromisoformat(snap_at_str.replace("Z", "+00:00"))
                                snap_dt = snap_dt.replace(tzinfo=None)
                            except Exception:
                                continue
                            dup = (await session.execute(text(
                                "SELECT 1 FROM odds_snapshots "
                                "WHERE race_id = :rid AND LOWER(horse_name) = LOWER(:name) "
                                "AND minutes_to_jump BETWEEN :lo AND :hi LIMIT 1"
                            ), {"rid": race_id, "name": name, "lo": mtj - 2, "hi": mtj + 2})).fetchone()
                            if dup:
                                continue
                            await session.execute(text(
                                "INSERT INTO odds_snapshots "
                                "(race_id, horse_name, snapshotted_at, minutes_to_jump, win_odds, source) "
                                "VALUES (:rid, :name, :snap_dt, :mtj, :odds, 'betfair_ltp')"
                            ), {"rid": race_id, "name": name, "snap_dt": snap_dt,
                                "mtj": mtj, "odds": float(win_odds_val)})
                            snap_inserted += 1
                except Exception as _runner_err:
                    _debug_err = repr(_runner_err)
                    log.warning("[patch-betfair-bsp] runner error (savepoint rolled back): %s", _runner_err)

        await session.commit()

    log.info("[patch-betfair-bsp] bsp=%d skipped=%d pred=%d snaps=%d",
             bsp_patched, bsp_skipped, pred_patched, snap_inserted)
    return {
        "status": "ok",
        "bsp_patched": bsp_patched,
        "bsp_skipped": bsp_skipped,
        "pred_patched": pred_patched,
        "snap_inserted": snap_inserted,
        "debug_err": _debug_err,
    }


_validation_bt_state: dict = {"running": False, "status": "idle", "result": None}


async def _run_validation_backtest(train_days_start: int, train_days_end: int) -> None:
    global _validation_bt_state
    _validation_bt_state = {"running": True, "status": "training", "result": None}

    today = _today_aest()
    train_start = (today - timedelta(days=train_days_start)).isoformat()
    train_end   = (today - timedelta(days=train_days_end)).isoformat()
    test_start  = (today - timedelta(days=train_days_end)).isoformat()
    test_end    = today.isoformat()

    # ── 1. Build training data ────────────────────────────────────────────────
    # Read features from RunnerPredictionHistoryRow per Ground Rule 1 (BUG-27).
    # Mutable for past races without snapshots can hold post-race-contaminated
    # feature vectors; training on those biases the model toward post-race
    # information. Source="live" excludes any prior validation-backtest rows.
    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id >= train_start)
            .where(HistoricalResultRow.race_id < train_end)
        )
        hr_rows = hr_result.scalars().all()

        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= train_start)
            .where(RunnerPredictionHistoryRow.race_id < train_end)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )
        pred_rows = pred_result.scalars().all()

    pred_by_key = {(p.race_id, p.horse_name): p for p in pred_rows}
    hr_by_key   = {(r.race_id, r.horse_name): r for r in hr_rows}

    win_training: list[tuple[list[float], int]] = []
    place_training: list[tuple[list[float], int]] = []

    for (race_id, horse_name), hr in hr_by_key.items():
        pred = pred_by_key.get((race_id, horse_name))
        if not pred or not pred.enriched_json:
            continue
        try:
            er = EnrichedRunner(**json.loads(pred.enriched_json))
            fv = build_feature_vector(er)
            # Use position == 1 over hr.winner to avoid stale-boolean risk from
            # re-seedings (same hardening as the win retrain path).
            win_training.append((fv, 1 if hr.position == 1 else 0))
            place_training.append((fv, 1 if hr.placed else 0))
        except Exception:
            continue

    if len(win_training) < 100:
        _validation_bt_state.update({"running": False, "status": "error",
                                     "result": {"detail": f"Insufficient training data: {len(win_training)} examples"}})
        return

    # ── 2. Train fresh models ─────────────────────────────────────────────────
    loop = asyncio.get_event_loop()

    win_model = HorseModel()
    place_model_v = PlaceModel()

    await loop.run_in_executor(None, win_model.train, win_training)
    await loop.run_in_executor(None, place_model_v.train, place_training)

    log.info("[validation-bt] Trained on %d win / %d place examples (%s to %s)",
             len(win_training), len(place_training), train_start, train_end)

    # ── 3. Load test-set predictions ─────────────────────────────────────────
    async with get_session() as session:
        test_pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id >= test_start)
            .where(RunnerPredictionRow.race_id < test_end)
            .where(RunnerPredictionRow.enriched_json.isnot(None))
        )
        test_rows = test_pred_result.scalars().all()

        # Races already in history (skip)
        already_result = await session.execute(
            select(RunnerPredictionHistoryRow.race_id)
            .where(RunnerPredictionHistoryRow.race_id >= test_start)
            .where(RunnerPredictionHistoryRow.race_id < test_end)
            .distinct()
        )
        already_set = set(already_result.scalars().all())

        # Race-level metadata from RacePredictionRow
        rp_result = await session.execute(
            select(RacePredictionRow)
            .where(RacePredictionRow.race_id >= test_start)
            .where(RacePredictionRow.race_id < test_end)
        )
        rp_map = {r.race_id: r for r in rp_result.scalars().all()}

    # Group by race
    races: dict[str, list] = {}
    for row in test_rows:
        if row.race_id in already_set:
            continue
        races.setdefault(row.race_id, []).append(row)

    # ── 4. Score and write to history ─────────────────────────────────────────
    written_races = 0
    written_runners = 0
    async with get_session() as session:
        for race_id, runners in races.items():
            scored = []
            for row in runners:
                try:
                    er = EnrichedRunner(**json.loads(row.enriched_json))
                    fv = build_feature_vector(er)
                    win_prob  = float(win_model.predict_proba([fv])[0])
                    place_prob = float(place_model_v.predict_proba([fv])[0])
                    scored.append((row, fv, win_prob, place_prob))
                except Exception:
                    continue

            if not scored:
                continue

            # Rank by win prob
            scored.sort(key=lambda x: x[2], reverse=True)
            place_sorted = sorted(scored, key=lambda x: x[3], reverse=True)
            place_rank_map = {row.horse_name: i + 1 for i, (row, _, _, _) in enumerate(place_sorted)}

            rp = rp_map.get(race_id)
            for rank, (row, fv, win_prob, place_prob) in enumerate(scored, 1):
                odds = row.best_available_odds or 0.0
                market_implied = (1.0 / odds) if odds > 1.0 else 0.0
                overlay = round(win_prob - market_implied, 4)
                vr = _value_rating(win_prob, odds, overlay)

                session.add(RunnerPredictionHistoryRow(
                    race_id=race_id,
                    horse_name=row.horse_name,
                    tab_number=row.tab_number,
                    barrier=row.barrier,
                    jockey=row.jockey,
                    trainer=row.trainer,
                    weight=row.weight,
                    win_probability=round(win_prob, 4),
                    place_probability=round(place_prob, 4),
                    model_rank=rank,
                    place_model_rank=place_rank_map.get(row.horse_name),
                    market_rank=row.market_rank,
                    overlay=overlay,
                    best_available_odds=odds or None,
                    value_rating=round(vr, 4),
                    key_flags=row.key_flags,
                    enriched_json=row.enriched_json,
                    scheduled_time=row.scheduled_time,
                    enriched_at=row.enriched_at or datetime.utcnow(),
                    cancelled=row.cancelled,
                    venue=row.venue or (rp.venue if rp else None),
                    state=row.state or (rp.state if rp else None),
                    race_number=row.race_number or (rp.race_number if rp else None),
                    race_name=row.race_name or (rp.race_name if rp else None),
                    distance=row.distance or (rp.distance if rp else None),
                    track_condition=row.track_condition or (rp.track_condition if rp else None),
                    field_size=row.field_size or (rp.field_size if rp else None),
                    prize_money=row.prize_money or (rp.prize_money if rp else None),
                    source="validation",
                    recorded_at=datetime.utcnow(),
                ))
                written_runners += 1

            await session.commit()
            written_races += 1

    log.info("[validation-bt] Written %d races / %d runners to history", written_races, written_runners)
    _validation_bt_state.update({
        "running": False,
        "status": "done",
        "result": {
            "status": "ok",
            "train_window": {"start": train_start, "end": train_end, "examples": len(win_training)},
            "test_window":  {"start": test_start, "end": test_end},
            "written_races": written_races,
            "written_runners": written_runners,
            "skipped_already_in_history": len(already_set),
        },
    })


@app.post("/api/admin/history/validation-backtest")
async def start_validation_backtest(
    train_days_start: int = Query(270, ge=60, le=730),
    train_days_end: int = Query(30, ge=7, le=365),
    x_cron_secret: Optional[str] = Header(None),
):
    _check_admin(x_cron_secret)
    if _validation_bt_state["running"]:
        return {"status": "already_running", "state": _validation_bt_state}
    asyncio.create_task(_run_validation_backtest(train_days_start, train_days_end))
    return {"status": "started", "train_days_start": train_days_start, "train_days_end": train_days_end}


@app.get("/api/admin/history/validation-backtest/status")
async def validation_backtest_status(x_cron_secret: Optional[str] = Header(None)):
    _check_admin(x_cron_secret)
    return _validation_bt_state


# ── Backtest ──────────────────────────────────────────────────────────────────

@app.get("/api/admin/backtest")
async def backtest_report(
    days: int = Query(14, ge=1, le=90),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Performance report: join predictions vs actual results.
    Returns top-pick win/place rate, value P&L, and per-condition breakdown.

    Reads top picks from RunnerPredictionHistoryRow per Ground Rule 1 — mutable
    can be overwritten by post-race re-enrichment for races without snapshots,
    which would inflate metrics. Uses the dedup-in-Python latest-enriched_at
    pattern, filters on cancelled, and excludes validation-backtest rows.
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
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        pred_rows = pred_result.scalars().all()

    # Dedup-in-Python on latest enriched_at — keeps the most recent pre-race row
    # per race and avoids the BUG-09 exact-timestamp pitfall.
    top_pick: dict[str, RunnerPredictionHistoryRow] = {}
    for p in pred_rows:
        if p.race_id not in top_pick:
            top_pick[p.race_id] = p

    # Index results by (race_id, normalized_horse_name)
    result_by_key: dict[tuple, HistoricalResultRow] = {
        (r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows
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
        actual = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
        if not actual:
            continue

        won = actual.position == 1
        placed = bool(actual.position and actual.position <= 3)
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
                    race_stubs = await asyncio.wait_for(client.get_meeting_races(slug), timeout=120)
                    rows_to_insert = []
                    for stub in race_stubs:
                        race_num = stub.get("eventNumber")
                        full_race = await asyncio.wait_for(client.get_race(slug, race_num), timeout=60)
                        if not full_race:
                            continue
                        runners = full_race.get("runners", [])
                        # Only process races with official results
                        actuals = {
                            r["runnerName"]: int(r["finishingPosition"])
                            for r in runners
                            if r.get("finishingPosition") and int(r["finishingPosition"]) >= 1
                        }
                        if not actuals:
                            continue
                        try:
                            race_id = f"{date_str}_{venue_code}_R{race_num}"
                            race = await asyncio.wait_for(
                                client.parse_race(full_race, date_str, venue_name, state_code), timeout=60
                            )
                            async with get_session() as session:
                                await _inject_accumulated_stats(race, session)
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
        await asyncio.sleep(0.1)  # be polite to the RA API

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
        # Cutover date = earliest date we have live predictions (use history per
        # Ground Rule 1; mutable rank-1 for past races may be post-race
        # contaminated for races without a snapshot).
        cutover_result = await session.execute(
            select(func.min(RunnerPredictionHistoryRow.race_id))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.source == "live")
        )
        earliest_race_id = cutover_result.scalar()
        cutover_date = earliest_race_id[:10] if earliest_race_id else None

        # --- Retroactive backtest rows ---
        bt_result = await session.execute(
            select(BacktestResultRow).where(BacktestResultRow.source == "backtest")
        )
        bt_rows = bt_result.scalars().all()

        # --- Live: history rank-1 joined with historical_results (BUG-32).
        # Previously read from RunnerPredictionRow which can hold post-race
        # mutable for races without snapshots. cancelled filter + source="live"
        # for consistency with the rest of the Ground Rule 1 sweep; dedup-in-
        # Python on latest enriched_at avoids the BUG-09 timestamp pitfall.
        hr_result = await session.execute(select(HistoricalResultRow))
        hr_map = {(r.race_id, r.horse_name): r for r in hr_result.scalars().all()}

        live_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        live_top_picks: dict[str, RunnerPredictionHistoryRow] = {}
        for p in live_result.scalars().all():
            if p.race_id not in live_top_picks:
                live_top_picks[p.race_id] = p
        live_rows = list(live_top_picks.values())

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
            # Position == 1 over hr.winner (BUG-28 pattern) — boolean can drift
            # under re-seedings; position is authoritative.
            unified.append({
                "win_prob": r.win_probability,
                "sp": hr.starting_price,
                "winner": hr.position == 1,
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


@app.get("/api/admin/backtest/feature-ablation")
async def win_feature_ablation(
    holdout_days: int = Query(14, ge=7, le=30),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Feature ablation for the win model: zero out each feature in turn on holdout races
    and measure the drop in top-pick win rate. Shows which features actually matter.
    """
    _check_admin(x_cron_secret)
    import math as _math
    from horse_engine.prediction.features import FEATURE_NAMES, NUM_FEATURES

    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()
    train_cutoff = (today - timedelta(days=30)).isoformat()

    async with get_session() as session:
        weights_dict = await load_model_weights(session)
        hr_result = await session.execute(select(HistoricalResultRow))
        all_hr = hr_result.scalars().all()
        pred_result = await session.execute(
            select(RunnerPredictionRow).where(RunnerPredictionRow.enriched_json.isnot(None))
        )
        all_pred = pred_result.scalars().all()

    model = HorseModel.from_weights_dict(weights_dict) if weights_dict else HorseModel()

    pred_by_key = {(p.race_id, p.horse_name): p for p in all_pred}
    holdout_races: dict[str, list] = {}
    holdout_results: dict[tuple, HistoricalResultRow] = {}
    for r in all_hr:
        if r.race_id >= holdout_cutoff:
            holdout_results[(r.race_id, r.horse_name)] = r
    for p in all_pred:
        if p.race_id >= holdout_cutoff:
            holdout_races.setdefault(p.race_id, []).append(p)

    # Build holdout feature vectors once
    holdout_fvs: dict[str, list[tuple]] = {}
    for race_id, runners in holdout_races.items():
        fvs = []
        for r in runners:
            try:
                er = EnrichedRunner(**json.loads(r.enriched_json))
                fvs.append((r, build_feature_vector(er)))
            except Exception:
                continue
        if fvs:
            holdout_fvs[race_id] = fvs

    def _win_rate(fv_sets):
        wins = races = 0
        for race_id, runner_fvs in fv_sets.items():
            win_probs, _ = model.predict_field([fv for _, fv in runner_fvs])
            best_idx = win_probs.index(max(win_probs))
            top_runner = runner_fvs[best_idx][0]
            actual = holdout_results.get((race_id, top_runner.horse_name))
            if not actual:
                continue
            races += 1
            if actual.winner:
                wins += 1
        return round(wins / races * 100, 1) if races else 0.0, races

    baseline_rate, total_races = _win_rate(holdout_fvs)

    ablation = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        zeroed = {}
        for race_id, runner_fvs in holdout_fvs.items():
            zeroed[race_id] = [
                (r, [v if i != feat_idx else 0.0 for i, v in enumerate(fv)])
                for r, fv in runner_fvs
            ]
        ablated_rate, _ = _win_rate(zeroed)
        delta = round(ablated_rate - baseline_rate, 1)
        ablation.append({
            "feature": feat_name,
            "weight": round(model.weights[feat_idx] if feat_idx < len(model.weights) else 0.0, 4),
            "baseline_win_rate": baseline_rate,
            "ablated_win_rate": ablated_rate,
            "delta": delta,
            "verdict": "valuable" if delta < -1.0 else ("noisy/harmful" if delta > 1.0 else "neutral"),
        })

    ablation.sort(key=lambda x: x["delta"])
    return {
        "holdout_races": total_races,
        "holdout_days": holdout_days,
        "baseline_win_rate_pct": baseline_rate,
        "feature_ablation": ablation,
    }


@app.get("/api/admin/backtest-place")
async def backtest_place(
    holdout_days: int = Query(14, ge=7, le=60),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Place model backtest: scores holdout races through the place model,
    reports top-pick place rate by tier and overall.

    Reads from RunnerPredictionHistoryRow per Ground Rule 1 — mutable can be
    overwritten by post-race re-enrichment for races without snapshots, which
    would contaminate the holdout feature vectors. Filters on cancelled and
    excludes validation-backtest rows. Per-race runners are deduplicated by
    keeping only the latest enriched_at batch so a re-snapshot edge case
    can't yield a mixed-batch feature set.
    """
    _check_admin(x_cron_secret)
    from horse_engine.prediction.features import FEATURE_NAMES

    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        weights_dict = await load_place_model_weights(session)
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= holdout_cutoff)
        )
        hr_rows = hr_result.scalars().all()
        race_ids = list({r.race_id for r in hr_rows})
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        pred_rows = pred_result.scalars().all()

    pm = PlaceModel.from_weights_dict(weights_dict) if weights_dict else PlaceModel()
    hr_map = {(r.race_id, r.horse_name): r for r in hr_rows}

    # Group by race, keeping only the latest-enriched_at batch per race so a
    # race with multiple snapshots (legacy) can't mix feature vectors from
    # different enrichment runs.
    races: dict[str, list] = {}
    latest_at_by_race: dict[str, datetime] = {}
    for p in pred_rows:
        existing = latest_at_by_race.get(p.race_id)
        if existing is None:
            latest_at_by_race[p.race_id] = p.enriched_at
        if p.enriched_at == latest_at_by_race[p.race_id]:
            races.setdefault(p.race_id, []).append(p)

    total = place_hits = 0
    sum_predicted_prob = 0.0

    for race_id, runners in races.items():
        runner_fvs = []
        for r in runners:
            try:
                er = EnrichedRunner(**json.loads(r.enriched_json))
                runner_fvs.append((r, build_feature_vector(er)))
            except Exception:
                continue
        if not runner_fvs:
            continue

        # BUG-41 fix: PlaceModel.predict_field returns (trained_p_top3, heuristic).
        # Use the FIRST tuple element (the trained P(top-3) output) for ranking
        # — the previous `_, place_probs = ...` discarded the trained output and
        # ranked by the win model's softmax(raw×0.5) heuristic applied to
        # PlaceModel weights, a fundamentally different quantity.
        place_probs, _ = pm.predict_field([fv for _, fv in runner_fvs])
        best_idx = place_probs.index(max(place_probs))
        top_runner = runner_fvs[best_idx][0]
        top_prob = place_probs[best_idx]
        actual = hr_map.get((race_id, top_runner.horse_name))
        if not actual:
            continue

        total += 1
        sum_predicted_prob += top_prob
        if bool(actual.placed):
            place_hits += 1

    # Tier breakdown removed — the previous premium/hot/high/strong thresholds
    # were copied from the WIN model's UI shortlist (model_pct >= 30 AND sp >=
    # 3.0 AND overlay > 5%) and applied here without the odds/overlay
    # components. That made "premium tier in backtest_place" a confidence
    # percentile, not a bet-selection criterion — measuring something different
    # from what the public-facing premium picks actually represent. Premium
    # pick performance is tracked separately by /api/performance/premium*
    # endpoints with the real win-criteria; this backtest now just reports
    # overall place rate and a single calibration gap.
    overall_place_rate = (place_hits / total * 100) if total else 0.0
    avg_predicted = (sum_predicted_prob / total * 100) if total else 0.0
    return {
        "holdout_days": holdout_days,
        "total_races": total,
        "place_hits": place_hits,
        "overall_place_rate_pct": round(overall_place_rate, 1),
        "avg_predicted_place_pct": round(avg_predicted, 1),
        "calibration_gap_pct": round(avg_predicted - overall_place_rate, 1),
    }


@app.get("/api/admin/odds-snapshots/health")
async def odds_snapshots_health(x_cron_secret: Optional[str] = Header(None)):
    """
    Validate odds snapshot collection: count rows, non-zero values,
    coverage by race, and recency — to confirm the cron is working pre-race.
    """
    _check_admin(x_cron_secret)
    cutoff_7d = (date.today() - timedelta(days=7)).isoformat()

    async with get_session() as session:
        total_result = await session.execute(
            select(func.count(OddsSnapshotRow.id))
        )
        total_snapshots = total_result.scalar() or 0

        nonzero_result = await session.execute(
            select(func.count(OddsSnapshotRow.id))
            .where(OddsSnapshotRow.win_odds.isnot(None))
            .where(OddsSnapshotRow.win_odds > 0)
        )
        nonzero_win_odds = nonzero_result.scalar() or 0

        recent_result = await session.execute(
            select(func.count(OddsSnapshotRow.id))
            .where(OddsSnapshotRow.race_id >= cutoff_7d)
        )
        recent_snapshots = recent_result.scalar() or 0

        recent_races_result = await session.execute(
            select(func.count(func.distinct(OddsSnapshotRow.race_id)))
            .where(OddsSnapshotRow.race_id >= cutoff_7d)
        )
        recent_races = recent_races_result.scalar() or 0

        latest_result = await session.execute(
            select(OddsSnapshotRow.snapshotted_at, OddsSnapshotRow.race_id, OddsSnapshotRow.win_odds)
            .order_by(OddsSnapshotRow.snapshotted_at.desc())
            .limit(5)
        )
        latest = latest_result.all()

        # Check runners with non-zero steam features in enriched_json
        pred_sample = await session.execute(
            select(RunnerPredictionRow.enriched_json)
            .where(RunnerPredictionRow.enriched_json.isnot(None))
            .where(RunnerPredictionRow.race_id >= cutoff_7d)
            .limit(200)
        )
        sample_rows = pred_sample.scalars().all()

    steam_nonzero = 0
    for ejson in sample_rows:
        try:
            d = json.loads(ejson)
            if any(d.get(f, 0.0) != 0.0 for f in ("steam_60", "steam_30", "drift_flag", "odds_velocity", "late_money")):
                steam_nonzero += 1
        except Exception:
            continue

    return {
        "total_snapshots": total_snapshots,
        "nonzero_win_odds": nonzero_win_odds,
        "zero_or_null_win_odds": total_snapshots - nonzero_win_odds,
        "last_7d_snapshots": recent_snapshots,
        "last_7d_races_covered": recent_races,
        "avg_snapshots_per_race_7d": round(recent_snapshots / recent_races, 1) if recent_races else 0,
        "enriched_runners_last_7d_sampled": len(sample_rows),
        "runners_with_nonzero_steam_features": steam_nonzero,
        "steam_feature_coverage_pct": round(steam_nonzero / len(sample_rows) * 100, 1) if sample_rows else 0,
        "latest_snapshots": [
            {"snapshotted_at": r[0].isoformat(), "race_id": r[1], "win_odds": r[2]}
            for r in latest
        ],
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
async def performance_summary(days: int = Query(5, ge=1, le=365)):
    """
    Per-day performance strip for the last N days.
    Shows top-pick win rate, place rate, and value P&L per day.
    No auth required — displayed publicly on the frontend.
    Uses the immutable pre-race snapshot table so stats are stable across retrains.
    """
    cutoff = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()

        if not hr_rows:
            return {"days": days, "summary": [], "overall_win_rate": None}

        race_ids = list({r.race_id for r in hr_rows})

        # History table — pre-race snapshot, unaffected by post-race re-enrichments.
        # Dedup in Python (latest enriched_at first) to avoid BUG-09 exact timestamp
        # match silently dropping picks when microsecond precision differs.
        hist_pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        top_picks: dict[str, RunnerPredictionHistoryRow] = {}
        for p in hist_pred_result.scalars().all():
            if p.race_id not in top_picks:
                top_picks[p.race_id] = p

    # Winner per race (position==1) — used for accurate act_won comparison
    winners: dict[str, str] = {}
    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}
    for r in hr_rows:
        if r.position == 1:
            winners[r.race_id] = r.horse_name

    # result races per date (for display — total_races_ran field)
    result_race_ids_by_date: dict[str, set] = {}
    for r in hr_rows:
        result_race_ids_by_date.setdefault(r.race_id[:10], set()).add(r.race_id)
    result_races_by_date = {d: len(ids) for d, ids in result_race_ids_by_date.items()}

    # predicted+settled intersection — correct denominator for data_complete
    result_race_id_flat: set[str] = {rid for ids in result_race_ids_by_date.values() for rid in ids}
    predicted_settled_by_date: dict[str, int] = {}
    for rid in top_picks:
        if rid in result_race_id_flat:
            d = rid[:10]
            predicted_settled_by_date[d] = predicted_settled_by_date.get(d, 0) + 1

    # Group by date
    by_date: dict[str, dict] = {}
    for race_id, pick in top_picks.items():
        race_date = race_id[:10]
        winner = winners.get(race_id)
        if not winner:
            continue  # race has no known winner yet
        d = by_date.setdefault(race_date, {
            "races": 0, "wins": 0, "places": 0, "value_pnl": 0.0, "value_bets": 0,
            "tier_premium": 0, "tier_hot": 0, "tier_high": 0, "tier_strong": 0,
        })
        d["races"] += 1
        act_won = _normalize_horse(pick.horse_name) == _normalize_horse(winner)
        # Look up pick horse for SP and place check (may be absent if scratched)
        actual = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
        act_placed = bool(actual and actual.position and actual.position <= 3) or act_won
        if act_won:
            d["wins"] += 1
        if act_placed:
            d["places"] += 1
        sp = (actual.starting_price if actual else None) or 0.0
        overlay = pick.overlay or 0.0
        model_pct = round((pick.win_probability or 0) * 100, 1)
        if overlay > 0.05 and sp >= 3.0:
            d["value_bets"] += 1
            d["value_pnl"] += (sp - 1) if act_won else -1.0
            if model_pct >= 50:
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
        total_predicted_settled = predicted_settled_by_date.get(day_str, races)
        total_ran = result_races_by_date.get(day_str, races)
        # Incomplete if history snapshots cover <85% of races we predicted that also settled
        # Incomplete if history snapshots cover <85% of races that actually ran
        data_complete = (races >= total_ran * 0.85) if total_ran else False
        summary.append({
            "date": day_str,
            "races": races,
            "total_races_ran": total_ran,
            "wins": d["wins"],
            "win_rate": round(d["wins"] / races, 3) if races else 0,
            "place_rate": round(d["places"] / races, 3) if races else 0,
            "value_bets": d["value_bets"],
            "value_pnl": round(d["value_pnl"], 2),
            "tier_premium": d["tier_premium"],
            "tier_hot": d["tier_hot"],
            "tier_high": d["tier_high"],
            "tier_strong": d["tier_strong"],
            "data_complete": data_complete,
        })

    total_races = sum(d["races"] for d in by_date.values())
    total_wins = sum(d["wins"] for d in by_date.values())
    return {
        "days": days,
        "overall_win_rate": round(total_wins / total_races, 3) if total_races else None,
        "overall_races": total_races,
        "summary": summary,
    }


@app.get("/api/admin/backtest-win")
async def backtest_win(
    split_pct: float = Query(0.7, ge=0.5, le=0.9),
    source: str = Query("all"),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Unbiased walk-forward win model backtest.

    source=history  — immutable pre-race snapshots only (~days of data, very clean)
    source=all      — also includes mutable rows where enriched_at < scheduled_time
                      (much larger dataset, same pre-race guarantee for rows with scheduled_time)

    Algorithm:
    1. Load pre-race feature snapshots, deduplicate by (race_id, horse_name).
    2. Sort chronologically, split at split_pct (default 70% train / 30% test).
    3. Train a scratch HorseModel on train races using race-grouped softmax.
    4. Re-score every runner in every test race with the train-only model.
    5. Compare top model pick vs actual winner and vs market favourite baseline.
    """
    _check_admin(x_cron_secret)

    async with get_session() as session:
        hist_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(
                RunnerPredictionHistoryRow.cancelled.is_(False)
                | RunnerPredictionHistoryRow.cancelled.is_(None)
            )
        )
        hist_rows = hist_result.scalars().all()

        # Extend with mutable rows confirmed pre-race via enriched_at < scheduled_time
        mutable_rows = []
        if source != "history":
            mut_result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.enriched_json.isnot(None))
                .where(RunnerPredictionRow.enriched_at.isnot(None))
                .where(RunnerPredictionRow.scheduled_time.isnot(None))
                .where(
                    RunnerPredictionRow.cancelled.is_(False)
                    | RunnerPredictionRow.cancelled.is_(None)
                )
            )
            for row in mut_result.scalars().all():
                try:
                    sched = datetime.fromisoformat(
                        row.scheduled_time.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    if row.enriched_at < sched:
                        mutable_rows.append(row)
                except (ValueError, TypeError):
                    pass

        hr_result = await session.execute(select(HistoricalResultRow))
        hr_rows = hr_result.scalars().all()

    # Build lookups — derive winner from position == 1 rather than the winner
    # boolean (BUG-28). The boolean can be stale after re-seedings; position is
    # the authoritative source. Matches the win retrain path's hardening.
    winners: dict[str, str] = {
        r.race_id: _normalize_horse(r.horse_name) for r in hr_rows if r.position == 1
    }
    sp_lookup: dict[str, float] = {
        r.race_id: r.starting_price
        for r in hr_rows if r.position == 1 and r.starting_price
    }

    # Merge rows — history table takes precedence; mutable fills gaps
    # Deduplicate by (race_id, normalized_horse_name)
    from collections import defaultdict as _ddict
    seen_keys: set[tuple] = set()
    merged_rows = []
    for row in hist_rows:
        key = (row.race_id, _normalize_horse(row.horse_name))
        seen_keys.add(key)
        merged_rows.append(row)
    for row in mutable_rows:
        key = (row.race_id, _normalize_horse(row.horse_name))
        if key not in seen_keys:
            seen_keys.add(key)
            merged_rows.append(row)

    # Group runners by race, build feature vectors
    by_race: dict[str, list[dict]] = _ddict(list)
    for row in merged_rows:
        try:
            er = EnrichedRunner(**json.loads(row.enriched_json))
            fv = build_feature_vector(er)
            by_race[row.race_id].append({
                "horse": _normalize_horse(row.horse_name),
                "fv": fv,
                "market_rank": er.market_rank,
                "best_odds": er.best_available_odds,
                "sp": sp_lookup.get(row.race_id),
            })
        except Exception:
            pass

    # Only races with a settled result and ≥2 runners
    settled_races = sorted(
        [(rid, runners) for rid, runners in by_race.items() if rid in winners and len(runners) >= 2],
        key=lambda x: x[0],
    )

    if len(settled_races) < 20:
        raise HTTPException(400, f"Only {len(settled_races)} settled races — need ≥20 for a meaningful split")

    split_idx = max(10, int(len(settled_races) * split_pct))
    train_races = settled_races[:split_idx]
    test_races  = settled_races[split_idx:]

    log.info("[backtest-win] %d train races, %d test races", len(train_races), len(test_races))

    # Build race groups for training (same format as retrain endpoint)
    _today = date.today()
    train_groups: list[list[tuple]] = []
    train_weights: list[float] = []
    for race_id, runners in train_races:
        winner = winners[race_id]
        group = [(r["fv"], 1 if r["horse"] == winner else 0) for r in runners]
        if sum(l for _, l in group) != 1:
            continue
        train_groups.append(group)
        try:
            days_ago = (_today - date.fromisoformat(race_id[:10])).days
        except Exception:
            days_ago = 30
        train_weights.append(math.exp(-days_ago / 30.0))

    # Train a scratch model on train set only (in thread — CPU bound)
    def _run_backtest():
        m = HorseModel()
        m.train_race_grouped(train_groups, sample_weights=train_weights)

        # Score test races with train-only model
        by_day: dict[str, dict] = {}
        race_results = []

        for race_id, runners in test_races:
            winner = winners[race_id]
            scores = [(r, m.raw_score(r["fv"])) for r in runners]
            scores.sort(key=lambda x: x[1], reverse=True)

            model_pick = scores[0][0]["horse"]
            fav_pick   = next(
                (r["horse"] for r in runners if r["market_rank"] == 1),
                runners[0]["horse"],
            )
            model_wins = model_pick == winner
            fav_wins   = fav_pick == winner
            sp = sp_lookup.get(race_id)

            day = race_id[:10]
            d = by_day.setdefault(day, {
                "races": 0,
                "model_wins": 0, "fav_wins": 0,
                "model_pnl": 0.0, "fav_pnl": 0.0,
                "sp_races": 0,
            })
            d["races"] += 1
            if model_wins:
                d["model_wins"] += 1
            if fav_wins:
                d["fav_wins"] += 1
            if sp:
                d["sp_races"] += 1
                d["model_pnl"] += (sp - 1.0) if model_wins else -1.0
                d["fav_pnl"]   += (sp - 1.0) if fav_wins   else -1.0

            race_results.append({
                "race_id": race_id,
                "winner": winner,
                "model_pick": model_pick,
                "fav_pick": fav_pick,
                "model_correct": model_wins,
                "fav_correct": fav_wins,
                "sp": sp,
                "field_size": len(runners),
            })

        # Summary
        total   = len(race_results)
        m_wins  = sum(1 for r in race_results if r["model_correct"])
        fav_wins_t = sum(1 for r in race_results if r["fav_correct"])
        sp_rows = [r for r in race_results if r["sp"]]
        m_pnl   = sum(((r["sp"] - 1.0) if r["model_correct"] else -1.0) for r in sp_rows)
        f_pnl   = sum(((r["sp"] - 1.0) if r["fav_correct"]   else -1.0) for r in sp_rows)

        days_out = []
        for day in sorted(by_day):
            d = by_day[day]
            n = d["races"]
            days_out.append({
                "date": day,
                "races": n,
                "model_win_rate_pct": round(d["model_wins"] / n * 100, 1),
                "fav_win_rate_pct":   round(d["fav_wins"]   / n * 100, 1),
                "model_pnl": round(d["model_pnl"], 2) if d["sp_races"] else None,
                "fav_pnl":   round(d["fav_pnl"],   2) if d["sp_races"] else None,
            })

        return {
            "split": {
                "train_races": len(train_races),
                "test_races": len(test_races),
                "split_pct": split_pct,
                "train_period": f"{train_races[0][0][:10]} → {train_races[-1][0][:10]}",
                "test_period":  f"{test_races[0][0][:10]}  → {test_races[-1][0][:10]}",
            },
            "summary": {
                "total_races": total,
                "model_wins": m_wins,
                "model_win_rate_pct": round(m_wins / total * 100, 1) if total else 0,
                "fav_wins": fav_wins_t,
                "fav_win_rate_pct": round(fav_wins_t / total * 100, 1) if total else 0,
                "sp_races": len(sp_rows),
                "model_pnl_at_1": round(m_pnl, 2),
                "fav_pnl_at_1": round(f_pnl, 2),
                "model_roi_pct": round(m_pnl / len(sp_rows) * 100, 1) if sp_rows else None,
                "fav_roi_pct":   round(f_pnl / len(sp_rows) * 100, 1) if sp_rows else None,
                "model_beats_fav": m_wins > fav_wins_t,
                "random_baseline_win_rate_pct": round(
                    100 / (sum(r["field_size"] for r in race_results) / total), 1
                ) if total else None,
            },
            "by_day": days_out,
            "races": race_results,
        }

    result = await asyncio.to_thread(_run_backtest)
    return result


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
        "hot":    {"label": f"🔥 Hot (top 25%, combined ≥{hot_threshold*100:.1f}%)",   "tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}, "field_size_sum": 0},
        "high":   {"label": f"⚡ High Confidence (top 60%, combined ≥{high_threshold*100:.1f}%)", "tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}, "field_size_sum": 0},
        "strong": {"label": "📈 Strong (bottom 40%)",                                  "tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}, "field_size_sum": 0},
        "below":  {"label": "Below threshold / small field",                           "tri": {"races": 0, "hits": 0}, "ff": {"races": 0, "hits": 0}, "field_size_sum": 0},
    }

    # Account for small fields separately
    for race_id, rows in race_map.items():
        field_size = len(rows)
        if field_size < 7:
            field_size_dist[field_size] = field_size_dist.get(field_size, 0) + 1

    totals_field_size_sum = 0
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

        tiers[tier_key]["field_size_sum"] += field_size
        totals_field_size_sum += field_size

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

    def _tri_random_baseline(n: float) -> float:
        """P(random box trifecta) = 6 / (n*(n-1)*(n-2)) × 100%"""
        if n < 3:
            return 100.0
        return 600.0 / (n * (n - 1) * (n - 2))

    def _ff_random_baseline(n: float) -> float:
        """P(random box first four) = 24 / (n*(n-1)*(n-2)*(n-3)) × 100%"""
        if n < 4:
            return 100.0
        return 2400.0 / (n * (n - 1) * (n - 2) * (n - 3))

    def _edge(hit_rate_pct: float, random_pct: float) -> float:
        return round(hit_rate_pct / random_pct, 2) if random_pct > 0 else 0.0

    tier_results = []
    for key, t in tiers.items():
        tri_r = t["tri"]["races"]
        if tri_r == 0:
            continue
        avg_n = t["field_size_sum"] / tri_r if tri_r else 8.0
        tri_hr = _rate(t["tri"])
        ff_hr = _rate(t["ff"])
        tri_rb = _tri_random_baseline(avg_n)
        ff_rb = _ff_random_baseline(avg_n)
        tier_results.append({
            "tier": t["label"],
            "avg_field_size": round(avg_n, 1),
            "trifecta": {
                **t["tri"],
                "hit_rate_pct": tri_hr,
                "random_baseline_pct": round(tri_rb, 2),
                "edge_multiple": _edge(tri_hr, tri_rb),
            },
            "first_four": {
                **t["ff"],
                "hit_rate_pct": ff_hr,
                "random_baseline_pct": round(_ff_random_baseline(avg_n), 2),
                "edge_multiple": _edge(ff_hr, ff_rb),
            },
        })

    overall_avg_n = totals_field_size_sum / totals["tri"]["races"] if totals["tri"]["races"] else 8.0
    overall_tri_hr = _rate(totals["tri"])
    overall_ff_hr = _rate(totals["ff"])
    return {
        "races_analysed": len(race_map),
        "runners_with_data": len(hist_rows),
        "overall": {
            "trifecta": {
                **totals["tri"],
                "hit_rate_pct": overall_tri_hr,
                "random_baseline_pct": round(_tri_random_baseline(overall_avg_n), 2),
                "edge_multiple": _edge(overall_tri_hr, _tri_random_baseline(overall_avg_n)),
            },
            "first_four": {
                **totals["ff"],
                "hit_rate_pct": overall_ff_hr,
                "random_baseline_pct": round(_ff_random_baseline(overall_avg_n), 2),
                "edge_multiple": _edge(overall_ff_hr, _ff_random_baseline(overall_avg_n)),
            },
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
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
        )
        top_picks = {p.race_id: p for p in pred_result.scalars().all()}

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}
    by_venue: dict[str, dict] = {}
    for race_id, pick in top_picks.items():
        _, venue, _ = _parse_race_id(race_id)
        result = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
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
    Reads from the immutable history table to guarantee pre-race predictions only.
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
        # Immutable history table — guaranteed pre-race snapshots. Use model_rank == 1
        # (BUG-24: was max(win_probability), which diverges from the race-card top pick
        # when ranks were tie-broken or overlay-adjusted). Filter cancelled (BUG-25),
        # exclude validation-backtest rows, dedup-in-Python by latest enriched_at.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.win_probability.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        top_picks: dict[str, RunnerPredictionHistoryRow] = {}
        for p in pred_result.scalars().all():
            if p.race_id not in top_picks:
                top_picks[p.race_id] = p

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}

    picks = []
    for race_id, pick in top_picks.items():
        actual = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
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
        # Immutable history table — use model_rank == 1 (BUG-24), filter cancelled
        # (BUG-25), exclude validation-backtest rows, dedup-in-Python by latest
        # enriched_at.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.win_probability.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        top_picks: dict[str, RunnerPredictionHistoryRow] = {}
        for p in pred_result.scalars().all():
            if p.race_id not in top_picks:
                top_picks[p.race_id] = p

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}
    bets = wins = 0
    total_pnl = 0.0
    sp_list = []
    for race_id, pick in top_picks.items():
        actual = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
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
        # Filter cancelled (BUG-25), dedup-in-Python by latest enriched_at.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        top_picks: dict[str, RunnerPredictionHistoryRow] = {}
        for p in pred_result.scalars().all():
            if p.race_id not in top_picks:
                top_picks[p.race_id] = p

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}
    monthly: dict[str, dict] = {}

    for race_id, pick in top_picks.items():
        actual = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
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
        # Filter cancelled (BUG-25), dedup-in-Python by latest enriched_at.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        top_picks: dict[str, RunnerPredictionHistoryRow] = {}
        for p in pred_result.scalars().all():
            if p.race_id not in top_picks:
                top_picks[p.race_id] = p

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}
    daily: dict[str, dict] = {}

    for race_id, pick in top_picks.items():
        actual = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
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
_EXOTIC_CANDIDATE_WINDOWS = [30, 60, 90]   # capped: exotic model only has ~30d of real outcome data
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
        # History table only — written once pre-race, never overwritten by re-enrichments,
        # so enriched_json always reflects the genuine pre-race feature vector.
        # Cancelled filter added per BUG-29 — without it, a scratched runner can be
        # the holdout-rank-1 pick; the result lookup then fails and the race is
        # silently dropped from the sample. Source="live" excludes prior
        # validation-backtest rows from contaminating the calibration.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
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

        # Training data: build per-race groups for race-grouped softmax training
        # (ARCH-2). Per-runner binary CE — what the calibration sweep used
        # before — ignores that horses compete within the field, so the
        # production weights it set were systematically less aligned with the
        # softmax inference step than the admin retrain endpoint's were.
        # Now both training paths use the same loss as inference.
        import math as _math
        races_in_window: dict[str, list[tuple[list[float], int]]] = {}
        race_weight_by_id: dict[str, float] = {}
        for row in all_hr:
            if row.race_id < train_cutoff or row.race_id >= holdout_cutoff:
                continue
            pred = pred_by_key.get((row.race_id, row.horse_name))
            if not pred:
                continue
            if pred.enriched_at and pred.scheduled_time:
                try:
                    sched = datetime.fromisoformat(pred.scheduled_time.replace("Z", "+00:00")).replace(tzinfo=None)
                    if pred.enriched_at > sched:
                        continue
                except (ValueError, AttributeError):
                    pass
            try:
                er = EnrichedRunner(**json.loads(pred.enriched_json))
                fv = build_feature_vector(er)
                # Use position == 1 over row.winner (BUG-28) — the boolean can
                # be stale after re-seedings; position is authoritative.
                races_in_window.setdefault(row.race_id, []).append(
                    (fv, 1 if row.position == 1 else 0)
                )
                if row.race_id not in race_weight_by_id:
                    try:
                        race_date = date.fromisoformat(row.race_id[:10])
                        days_ago = (today - race_date).days
                    except Exception:
                        days_ago = 30
                    race_weight_by_id[row.race_id] = _math.exp(-days_ago / 30.0)
            except Exception:
                continue

        # Keep only races with exactly one winner and ≥2 runners — the same
        # validity check train_race_grouped applies internally; pre-filtering
        # gives accurate skip diagnostics.
        race_groups: list[list[tuple[list[float], int]]] = []
        race_sample_weights: list[float] = []
        for race_id, runners in races_in_window.items():
            if len(runners) < 2:
                continue
            if sum(l for _, l in runners) != 1:
                continue
            race_groups.append(runners)
            race_sample_weights.append(race_weight_by_id[race_id])

        if len(race_groups) < 50:
            window_results.append({
                "window_days": window,
                "training_races": len(race_groups),
                "skipped": True,
                "reason": "insufficient training races",
            })
            continue

        model = HorseModel()
        stats = await asyncio.to_thread(
            model.train_race_grouped, race_groups, sample_weights=race_sample_weights
        )

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
            # Position-based winner check (BUG-28) — matches the training-label
            # source so train and holdout judge "winner" the same way.
            actual_won = actual.position == 1
            if actual_won:
                win_picks += 1
            if actual.placed:
                place_picks += 1

            sp = actual.starting_price or 0
            implied = 1 / sp if sp > 1 else 0
            if (top_prob - implied) > 0.05 and sp > 0:
                value_bets += 1
                value_pnl += (sp - 1.0) if actual_won else -1.0

        win_rate = round(win_picks / total_races, 3) if total_races else 0
        place_rate = round(place_picks / total_races, 3) if total_races else 0
        roi = round(value_pnl / value_bets, 3) if value_bets else 0

        result = {
            "window_days": window,
            # Race-grouped training reports races (the unit of the softmax loss)
            # plus the total runners across those races, instead of the flat
            # example count the per-runner binary CE used to report.
            "training_races": len(race_groups),
            "training_runners": sum(len(r) for r in race_groups),
            "training_top1_hit_rate": stats.get("top1_hit_rate"),
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
    """Run by APScheduler every Sunday at 2am AEST — sweeps all three models."""
    log.info("[scheduler] Running weekly calibration sweep (win + place + exotic)")
    for label, fn in [
        ("win", _run_calibration_sweep),
        ("place", _run_place_calibration_sweep),
        ("exotic", _run_exotic_calibration_sweep),
    ]:
        try:
            result = await fn(holdout_days=14)
            if result.get("drift_flag"):
                log.warning("[calibrate/%s] DRIFT DETECTED: %s", label, result.get("drift_reason"))
            log.info("[calibrate/%s] Complete. Best window: %d days", label, result.get("best_window", 0))
        except Exception as e:
            log.exception("[calibrate/%s] Weekly calibration failed: %s", label, e)


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


# ── Place calibration sweep ───────────────────────────────────────────────────

async def _run_place_calibration_sweep(holdout_days: int = 14) -> dict:
    """
    Multi-window calibration for the place model (P(position ≤ 3)).
    Holdout metric: fraction of races where the top-ranked place pick actually placed.
    Saves best window's weights to place_model_weights.
    """
    import math as _math
    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(select(HistoricalResultRow))
        all_hr = hr_result.scalars().all()
        # History table only — written once pre-race, never overwritten by re-enrichments.
        # Cancelled filter added per BUG-33 (sister of BUG-29 in the win sweep) so a
        # scratched runner can't be the holdout rank-1 pick and silently drop the
        # race. Source="live" excludes prior validation-backtest rows.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )
        all_pred = pred_result.scalars().all()

    pred_by_key = {(p.race_id, p.horse_name): p for p in all_pred}

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
        training_data = []
        sw_list = []
        for row in all_hr:
            if row.race_id < train_cutoff or row.race_id >= holdout_cutoff:
                continue
            pred = pred_by_key.get((row.race_id, row.horse_name))
            if not pred:
                continue
            if pred.enriched_at and pred.scheduled_time:
                try:
                    sched = datetime.fromisoformat(pred.scheduled_time.replace("Z", "+00:00")).replace(tzinfo=None)
                    if pred.enriched_at > sched:
                        continue
                except (ValueError, AttributeError):
                    pass
            try:
                er = EnrichedRunner(**json.loads(pred.enriched_json))
                fv = build_feature_vector(er)
                training_data.append((fv, 1 if row.placed else 0))
                try:
                    days_ago = (today - date.fromisoformat(row.race_id[:10])).days
                except Exception:
                    days_ago = 30
                sw_list.append(_math.exp(-days_ago / 30.0))
            except Exception:
                continue

        if len(training_data) < 50:
            window_results.append({"window_days": window, "training_examples": len(training_data),
                                   "skipped": True, "reason": "insufficient training data"})
            continue

        model = PlaceModel()
        stats = await asyncio.to_thread(model.train, training_data, sample_weights=sw_list)

        total_races = place_hits = 0
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
            # BUG-41 fix: use the trained P(top-3) output (first tuple element),
            # not the win-model heuristic in the second element. Without this the
            # place calibration sweep optimises window selection against the
            # wrong objective.
            place_probs, _ = model.predict_field([fv for _, fv in runner_fvs])
            best_idx = place_probs.index(max(place_probs))
            top_runner = runner_fvs[best_idx][0]
            actual = holdout_results.get((race_id, top_runner.horse_name))
            if not actual:
                continue
            total_races += 1
            if actual.placed:
                place_hits += 1

        place_rate = round(place_hits / total_races, 3) if total_races else 0
        result = {
            "window_days": window,
            "training_examples": len(training_data),
            "training_log_loss": stats.get("log_loss"),
            "holdout_races": total_races,
            "place_rate": place_rate,
        }
        window_results.append(result)
        log.info("[place-calibrate] window=%d place=%.1f%%", window, place_rate * 100)

        if place_rate > best_score:
            best_score = place_rate
            best_window = window
            best_weights = stats["weights"]

    if not best_weights:
        return {"error": "no valid windows", "window_results": window_results}

    async with get_session() as session:
        await save_place_model_weights(session, best_weights)

    best_result = next((r for r in window_results if r.get("window_days") == best_window), {})
    drift_flag = False
    drift_reason = None

    async with get_session() as session:
        hist = await session.execute(
            select(CalibrationRow).order_by(CalibrationRow.ran_at.desc()).limit(4)
        )
        prev_runs = hist.scalars().all()

    if len(prev_runs) >= 4:
        prev_rates = [r.place_rate for r in prev_runs if r.place_rate is not None]
        if prev_rates:
            avg = sum(prev_rates) / len(prev_rates)
            cur = best_result.get("place_rate", 0)
            if avg - cur > _DRIFT_THRESHOLD:
                drift_flag = True
                drift_reason = f"Place rate dropped {avg - cur:.1%} below 4-run avg ({avg:.1%} → {cur:.1%})"

    log.info("[place-calibrate] Best window=%d days, place=%.1f%%, drift=%s",
             best_window, best_result.get("place_rate", 0) * 100, drift_flag)
    return {
        "best_window": best_window,
        "best_score": round(best_score, 3),
        "drift_flag": drift_flag,
        "drift_reason": drift_reason,
        "window_results": window_results,
    }


_place_calibration_status: dict = {"running": False, "done": False, "result": None, "error": None}


async def _run_place_calibration_task(holdout_days: int):
    global _place_calibration_status
    _place_calibration_status = {"running": True, "done": False, "result": None, "error": None,
                                  "started_at": datetime.utcnow().isoformat()}
    try:
        result = await _run_place_calibration_sweep(holdout_days=holdout_days)
        _place_calibration_status.update({"running": False, "done": True, "result": result,
                                           "finished_at": datetime.utcnow().isoformat()})
    except Exception as e:
        log.exception("[place-calibrate] Task failed: %s", e)
        _place_calibration_status.update({"running": False, "done": True, "error": str(e),
                                           "finished_at": datetime.utcnow().isoformat()})


@app.post("/api/admin/calibrate-place")
async def run_place_calibration(
    holdout_days: int = Query(14, ge=7, le=30),
    x_cron_secret: Optional[str] = Header(None),
):
    """Multi-window calibration sweep for the place model. Check /api/admin/calibrate-place/status."""
    _check_admin(x_cron_secret)
    if _place_calibration_status.get("running"):
        raise HTTPException(409, "Place calibration already running")
    asyncio.create_task(_run_place_calibration_task(holdout_days))
    return {"status": "started", "holdout_days": holdout_days,
            "message": "Check /api/admin/calibrate-place/status for progress"}


@app.get("/api/admin/calibrate-place/status")
async def place_calibration_status(x_cron_secret: Optional[str] = Header(None)):
    _check_admin(x_cron_secret)
    return _place_calibration_status


# ── Exotic calibration sweep ──────────────────────────────────────────────────

async def _run_exotic_calibration_sweep(holdout_days: int = 14) -> dict:
    """
    Multi-window calibration for the exotic model (trifecta/first four coverage).
    Holdout metric: trifecta box hit rate (top 3 picks == actual positions 1-2-3).
    Only uses races with field_size >= 7. Saves best weights to exotic_model_weights.
    """
    from horse_engine.models.database import ExoticBacktestRow
    import math as _math
    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.position.isnot(None))
        )
        all_hr = hr_result.scalars().all()
        # History table only — written once pre-race, never overwritten by re-enrichments.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow).where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
        )
        all_pred = pred_result.scalars().all()

    pred_by_key = {(p.race_id, p.horse_name): p for p in all_pred}

    # Group historical results by race for holdout
    holdout_race_results: dict[str, list[HistoricalResultRow]] = {}
    for r in all_hr:
        if r.race_id >= holdout_cutoff:
            holdout_race_results.setdefault(r.race_id, []).append(r)

    # Group predictions by race for holdout
    holdout_race_preds: dict[str, list] = {}
    for p in all_pred:
        if p.race_id >= holdout_cutoff:
            holdout_race_preds.setdefault(p.race_id, []).append(p)

    # Group training data by race
    all_race_results: dict[str, list[HistoricalResultRow]] = {}
    for r in all_hr:
        if r.race_id < holdout_cutoff:
            all_race_results.setdefault(r.race_id, []).append(r)

    window_results = []
    best_window = None
    best_score = float("-inf")
    best_weights = None

    for window in _EXOTIC_CANDIDATE_WINDOWS:
        train_cutoff = (today - timedelta(days=window)).isoformat()

        # Build race groups for train_exotic()
        race_groups = []
        for race_id, rows in all_race_results.items():
            if race_id < train_cutoff:
                continue
            if len(rows) < 7:
                continue
            group = []
            for row in rows:
                pred = pred_by_key.get((race_id, row.horse_name))
                if not pred:
                    continue
                if pred.enriched_at and pred.scheduled_time:
                    try:
                        sched = datetime.fromisoformat(pred.scheduled_time.replace("Z", "+00:00")).replace(tzinfo=None)
                        if pred.enriched_at > sched:
                            continue
                    except (ValueError, AttributeError):
                        pass
                try:
                    er = EnrichedRunner(**json.loads(pred.enriched_json))
                    fv = build_feature_vector(er)
                    label = 1 if row.placed else 0
                    group.append((fv, label, row.position))
                except Exception:
                    continue
            if len(group) >= 7:
                race_groups.append(group)

        if len(race_groups) < 20:
            window_results.append({"window_days": window, "training_races": len(race_groups),
                                   "skipped": True, "reason": "insufficient training races"})
            continue

        # Time-weight races: more recent = higher weight (via tri_lambda scaling)
        model = ExoticModel()
        stats = await asyncio.to_thread(model.train_exotic, race_groups)

        # Score holdout: trifecta box hit rate
        tri_hits = tri_races = ff_hits = ff_races = 0
        for race_id, result_rows in holdout_race_results.items():
            if len(result_rows) < 7:
                continue
            pred_rows = holdout_race_preds.get(race_id, [])
            runner_fvs = []
            for r in result_rows:
                pred = next((p for p in pred_rows if p.horse_name == r.horse_name), None)
                if not pred or not pred.enriched_json:
                    continue
                try:
                    er = EnrichedRunner(**json.loads(pred.enriched_json))
                    runner_fvs.append((r, build_feature_vector(er)))
                except Exception:
                    continue
            if len(runner_fvs) < 7:
                continue
            scores = [model.raw_score(fv) for _, fv in runner_fvs]
            ranked = sorted(range(len(runner_fvs)), key=lambda i: scores[i], reverse=True)
            actual_top3 = {runner_fvs[i][0].position for i in range(len(runner_fvs))
                           if runner_fvs[i][0].position in (1, 2, 3)}
            predicted_top3 = {runner_fvs[ranked[i]][0].position for i in range(3)
                              if runner_fvs[ranked[i]][0].position is not None}
            if len(actual_top3) == 3:
                tri_races += 1
                if predicted_top3 == actual_top3:
                    tri_hits += 1
            if len(result_rows) >= 8:
                actual_top4 = {runner_fvs[i][0].position for i in range(len(runner_fvs))
                               if runner_fvs[i][0].position in (1, 2, 3, 4)}
                predicted_top4 = {runner_fvs[ranked[i]][0].position for i in range(4)
                                  if runner_fvs[ranked[i]][0].position is not None}
                if len(actual_top4) == 4:
                    ff_races += 1
                    if predicted_top4 == actual_top4:
                        ff_hits += 1

        tri_rate = round(tri_hits / tri_races, 3) if tri_races else 0
        ff_rate = round(ff_hits / ff_races, 3) if ff_races else 0
        result = {
            "window_days": window,
            "training_races": len(race_groups),
            "training_log_loss": stats.get("log_loss"),
            "holdout_races": tri_races,
            "tri_box_hit_rate": tri_rate,
            "ff_box_hit_rate": ff_rate,
        }
        window_results.append(result)
        log.info("[exotic-calibrate] window=%d tri=%.1f%% ff=%.1f%%",
                 window, tri_rate * 100, ff_rate * 100)

        if tri_rate > best_score:
            best_score = tri_rate
            best_window = window
            best_weights = stats["weights"]

    if not best_weights:
        return {"error": "no valid windows", "window_results": window_results}

    async with get_session() as session:
        await save_exotic_model_weights(session, best_weights)

    best_result = next((r for r in window_results if r.get("window_days") == best_window), {})
    drift_flag = False
    drift_reason = None

    async with get_session() as session:
        hist = await session.execute(
            select(ExoticBacktestRow).order_by(ExoticBacktestRow.ran_at.desc()).limit(4)
        )
        prev_runs = hist.scalars().all()

    if len(prev_runs) >= 4:
        prev_rates = [r.best_holdout_box_hit_rate for r in prev_runs if r.best_holdout_box_hit_rate is not None]
        if prev_rates:
            avg = sum(prev_rates) / len(prev_rates)
            cur = best_result.get("tri_box_hit_rate", 0)
            if avg - cur > _DRIFT_THRESHOLD:
                drift_flag = True
                drift_reason = f"Trifecta hit rate dropped {avg - cur:.1%} below 4-run avg ({avg:.1%} → {cur:.1%})"

    async with get_session() as session:
        session.add(ExoticBacktestRow(
            ran_at=datetime.utcnow(),
            best_window=best_window,
            best_holdout_box_hit_rate=best_result.get("tri_box_hit_rate"),
            holdout_races=best_result.get("holdout_races"),
            holdout_days=holdout_days,
            results_json=json.dumps({
                "drift_flag": drift_flag,
                "drift_reason": drift_reason,
                "window_results": window_results,
            }),
        ))
        await session.commit()

    log.info("[exotic-calibrate] Best window=%d days, tri=%.1f%%, drift=%s",
             best_window, best_result.get("tri_box_hit_rate", 0) * 100, drift_flag)
    return {
        "best_window": best_window,
        "best_score": round(best_score, 3),
        "drift_flag": drift_flag,
        "drift_reason": drift_reason,
        "window_results": window_results,
    }


_exotic_calibration_status: dict = {"running": False, "done": False, "result": None, "error": None}


async def _run_exotic_calibration_task(holdout_days: int):
    global _exotic_calibration_status
    _exotic_calibration_status = {"running": True, "done": False, "result": None, "error": None,
                                   "started_at": datetime.utcnow().isoformat()}
    try:
        result = await _run_exotic_calibration_sweep(holdout_days=holdout_days)
        _exotic_calibration_status.update({"running": False, "done": True, "result": result,
                                            "finished_at": datetime.utcnow().isoformat()})
    except Exception as e:
        log.exception("[exotic-calibrate] Task failed: %s", e)
        _exotic_calibration_status.update({"running": False, "done": True, "error": str(e),
                                            "finished_at": datetime.utcnow().isoformat()})


@app.post("/api/admin/calibrate-exotic")
async def run_exotic_calibration(
    holdout_days: int = Query(14, ge=7, le=30),
    x_cron_secret: Optional[str] = Header(None),
):
    """Multi-window calibration sweep for the exotic model. Check /api/admin/calibrate-exotic/status."""
    _check_admin(x_cron_secret)
    if _exotic_calibration_status.get("running"):
        raise HTTPException(409, "Exotic calibration already running")
    asyncio.create_task(_run_exotic_calibration_task(holdout_days))
    return {"status": "started", "holdout_days": holdout_days,
            "message": "Check /api/admin/calibrate-exotic/status for progress"}


@app.get("/api/admin/calibrate-exotic/status")
async def exotic_calibration_status(x_cron_secret: Optional[str] = Header(None)):
    _check_admin(x_cron_secret)
    return _exotic_calibration_status


# ── Cron ──────────────────────────────────────────────────────────────────────

async def _load_venue_calibration() -> dict[str, float]:
    """Compute venue win-rate multipliers from last 60 days of historical results.

    Reads top picks from RunnerPredictionHistoryRow (immutable pre-race snapshot) per
    Ground Rule 1 — mutable can be overwritten by post-race re-enrichment for races
    that were never snapshotted, which would otherwise contaminate the calibration.
    """
    cutoff = (date.today() - timedelta(days=60)).isoformat()
    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()
        if not hr_rows:
            return {}
        race_ids = list({r.race_id for r in hr_rows})
        # Dedup-in-Python on latest enriched_at avoids the BUG-09 exact-timestamp
        # pitfall: races with duplicate snapshots keep the most recent pre-race row.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )
        top_picks: dict[str, RunnerPredictionHistoryRow] = {}
        for p in pred_result.scalars().all():
            if p.race_id not in top_picks:
                top_picks[p.race_id] = p

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}
    venue_stats: dict[str, dict] = {}
    for race_id, pick in top_picks.items():
        _, venue, _ = _parse_race_id(race_id)
        result = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
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
                async with get_session() as session:
                    await _inject_accumulated_stats(race, session)
                predictions, _ = await enrich_and_predict_race(race, model, venue_calibration=venue_cal, place_model=place_model, exotic_model=exotic_model)
                async with get_session() as session:
                    await save_race_predictions(
                        session,
                        race_id,
                        [_prediction_to_db_dict(p, race_id, race.scheduled_time, race=race) for p in predictions],
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
    date: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """Force re-enrich races for a given date (defaults to today). Fire-and-forget."""
    _check_admin(x_cron_secret)
    target = date or _today_aest().isoformat()

    async def _do_reenrich():
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)
        await _enrich_date(target, client, model, force=True)
        # Drop cached meeting list + every per-venue detail entry for this date
        # so the refreshed enrichment is immediately visible (BUG-37).
        _invalidate_meeting_caches(target)
        log.info("[reenrich] Completed re-enrich for %s", target)

    asyncio.create_task(_do_reenrich())
    return {"status": "reenrich_started", "date": target}


@app.post("/api/admin/restore-cancelled")
async def restore_cancelled(
    date: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Re-run cancellation check for a date, restoring any races that were
    falsely marked cancelled when Racing Australia was blocked. Fast — no re-enrichment.
    """
    _check_admin(x_cron_secret)
    target = date or _today_aest().isoformat()
    client = get_tab_client()
    await _cancel_abandoned_meetings(client, target)
    return {"status": "done", "date": target}


@app.post("/api/admin/force-restore")
async def force_restore(
    date: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Directly uncancels ALL cancelled predictions for a date without hitting the
    upstream meeting feed. Use when Racing Australia is blocked or returning
    stale data and `/api/admin/restore-cancelled` can't succeed.
    """
    from sqlalchemy import update as sa_update
    _check_admin(x_cron_secret)
    target = date or _today_aest().isoformat()
    async with get_session() as session:
        result = await session.execute(
            sa_update(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{target}_%"))
            .where(RunnerPredictionRow.cancelled.is_(True))
            .values(cancelled=False)
        )
        # Mirror into history so settled-race reads also see the restored state.
        hist_result = await session.execute(
            sa_update(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{target}_%"))
            .where(RunnerPredictionHistoryRow.cancelled.is_(True))
            .values(cancelled=False)
        )
        await session.commit()
        affected = result.rowcount
        hist_affected = hist_result.rowcount
    # Restore can affect every venue at the date — drop the list cache and every
    # per-venue cache entry for this date (BUG-30).
    _invalidate_meeting_caches(target)
    log.info("[force-restore] Uncancelled %d mutable / %d history row(s) for %s",
             affected, hist_affected, target)
    return {
        "status": "done",
        "date": target,
        "rows_restored": affected,
        "history_restored": hist_affected,
    }


@app.get("/api/admin/trifecta-model-comparison")
async def trifecta_model_comparison(
    holdout_days: int = Query(default=14, ge=7, le=30),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Compare win/place model (place_probability top-3) vs exotic model (exotic_model_rank top-3)
    on trifecta box hit rate — same holdout dataset used by calibrate-exotic.

    Uses RunnerPredictionRow + HistoricalResultRow (not history snapshot), so the
    sample size matches the exotic calibration's 530-race holdout.
    """
    _check_admin(x_cron_secret)
    today = _today_aest()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id >= holdout_cutoff)
            .where(HistoricalResultRow.position.isnot(None))
        )
        all_hr = hr_result.scalars().all()

        pred_result = await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id >= holdout_cutoff)
        )
        all_pred = pred_result.scalars().all()

    # Group results by race
    results_by_race: dict[str, list] = {}
    for r in all_hr:
        results_by_race.setdefault(r.race_id, []).append(r)

    # Group preds by race
    preds_by_race: dict[str, list] = {}
    for p in all_pred:
        preds_by_race.setdefault(p.race_id, []).append(p)

    wp_hits = ex_hits = total = 0
    race_rows = []

    for race_id, results in sorted(results_by_race.items()):
        if len(results) < 7:
            continue
        actual_top3 = {r.horse_name.strip().lower() for r in results if r.position in (1, 2, 3)}
        if len(actual_top3) != 3:
            continue

        preds = preds_by_race.get(race_id, [])
        if not preds:
            continue

        # Win/place: top-3 by place_probability
        wp_sorted = sorted(
            [p for p in preds if p.place_probability is not None],
            key=lambda p: p.place_probability, reverse=True
        )
        if len(wp_sorted) < 3:
            continue
        wp_top3 = {p.horse_name.strip().lower() for p in wp_sorted[:3]}

        # Exotic: top-3 by exotic_model_rank (lower = better); fall back to place_model_rank
        ex_candidates = [p for p in preds if p.exotic_model_rank is not None]
        if ex_candidates:
            ex_sorted = sorted(ex_candidates, key=lambda p: p.exotic_model_rank)
        else:
            ex_sorted = sorted(
                [p for p in preds if p.place_model_rank is not None],
                key=lambda p: p.place_model_rank
            )
        if len(ex_sorted) < 3:
            continue
        ex_top3 = {p.horse_name.strip().lower() for p in ex_sorted[:3]}

        total += 1
        wp_hit = wp_top3 == actual_top3
        ex_hit = ex_top3 == actual_top3
        if wp_hit:
            wp_hits += 1
        if ex_hit:
            ex_hits += 1

        race_rows.append({
            "race_id": race_id,
            "field_size": len(results),
            "wp_top3": sorted(wp_top3),
            "ex_top3": sorted(ex_top3),
            "actual_top3": sorted(actual_top3),
            "wp_hit": wp_hit,
            "ex_hit": ex_hit,
            "using_exotic_rank": len(ex_candidates) >= 3,
        })

    return {
        "holdout_days": holdout_days,
        "races_evaluated": total,
        "win_place_model": {
            "trifecta_box_hits": wp_hits,
            "trifecta_box_hit_rate": round(wp_hits / total, 4) if total else 0,
        },
        "exotic_model": {
            "trifecta_box_hits": ex_hits,
            "trifecta_box_hit_rate": round(ex_hits / total, 4) if total else 0,
            "races_using_exotic_rank": sum(1 for r in race_rows if r["using_exotic_rank"]),
            "races_using_fallback": sum(1 for r in race_rows if not r["using_exotic_rank"]),
        },
        "races": race_rows,
    }


@app.get("/api/admin/placement-model-comparison")
async def placement_model_comparison(
    days: int = Query(default=30, ge=1, le=90),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Compare win/place model (place_probability) vs exotic model (exotic_model_rank)
    at predicting actual placed finishers.

    For each settled race where we have both exotic_model_rank and place_probability
    in the pre-race history snapshot, we check how many of the actual placed runners
    appeared in each model's top-3 prediction.

    Returns per-race breakdown and aggregate recall scores.
    """
    _check_admin(x_cron_secret)

    cutoff = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        # Get all history rows from the window that have exotic_model_rank set
        history_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= cutoff)
            .where(RunnerPredictionHistoryRow.exotic_model_rank.isnot(None))
            .where(RunnerPredictionHistoryRow.place_probability.isnot(None))
        )
        history_rows = history_result.scalars().all()

        # Get all settled results for those races
        race_ids = list({r.race_id for r in history_rows})
        if not race_ids:
            return {"error": "no_data", "message": "No races with exotic_model_rank in history for this window"}

        results_result = await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.in_(race_ids))
        )
        result_rows = results_result.scalars().all()

    # Index results: race_id -> set of placed horse names
    placed_by_race: dict[str, set[str]] = {}
    for r in result_rows:
        if r.placed:
            placed_by_race.setdefault(r.race_id, set()).add(r.horse_name.strip().lower())

    # Only evaluate races that have settled results
    settled_race_ids = set(placed_by_race.keys())

    # Group history rows by race
    history_by_race: dict[str, list] = {}
    for h in history_rows:
        if h.race_id in settled_race_ids:
            history_by_race.setdefault(h.race_id, []).append(h)

    if not history_by_race:
        return {"error": "no_settled_data", "message": "No settled races found for comparison"}

    race_results = []
    win_place_hits = 0
    exotic_hits = 0
    win_place_total = 0
    exotic_total = 0

    for race_id, runners in sorted(history_by_race.items()):
        placed_names = placed_by_race.get(race_id, set())
        n_places = len(placed_names)
        if n_places == 0:
            continue

        # Win/place model: top-3 by place_probability descending
        wp_sorted = sorted(runners, key=lambda r: r.place_probability or 0, reverse=True)
        wp_top3 = {r.horse_name.strip().lower() for r in wp_sorted[:3]}

        # Exotic model: top-3 by exotic_model_rank ascending (rank 1 = best)
        ex_sorted = sorted(runners, key=lambda r: r.exotic_model_rank or 999)
        ex_top3 = {r.horse_name.strip().lower() for r in ex_sorted[:3]}

        wp_hit = len(wp_top3 & placed_names)
        ex_hit = len(ex_top3 & placed_names)

        win_place_hits += wp_hit
        exotic_hits += ex_hit
        win_place_total += n_places
        exotic_total += n_places

        race_results.append({
            "race_id": race_id,
            "field_size": len(runners),
            "n_placed": n_places,
            "wp_top3": sorted(wp_top3),
            "ex_top3": sorted(ex_top3),
            "actual_placed": sorted(placed_names),
            "wp_hits": wp_hit,
            "ex_hits": ex_hit,
            "wp_recall": round(wp_hit / n_places, 3),
            "ex_recall": round(ex_hit / n_places, 3),
            "agreement": len(wp_top3 & ex_top3),
        })

    n_races = len(race_results)
    wp_overall_recall = round(win_place_hits / win_place_total, 4) if win_place_total else 0
    ex_overall_recall = round(exotic_hits / exotic_total, 4) if exotic_total else 0

    # Races where each model beats the other
    wp_wins = sum(1 for r in race_results if r["wp_recall"] > r["ex_recall"])
    ex_wins = sum(1 for r in race_results if r["ex_recall"] > r["wp_recall"])
    ties = n_races - wp_wins - ex_wins

    return {
        "window_days": days,
        "races_evaluated": n_races,
        "summary": {
            "win_place_model": {
                "total_hits": win_place_hits,
                "total_placed": win_place_total,
                "recall": wp_overall_recall,
                "races_where_better": wp_wins,
            },
            "exotic_model": {
                "total_hits": exotic_hits,
                "total_placed": exotic_total,
                "recall": ex_overall_recall,
                "races_where_better": ex_wins,
            },
            "ties": ties,
        },
        "races": race_results,
    }


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _prediction_to_db_dict(pred, race_id: str, scheduled_time: str | None = None, race=None) -> dict:
    d = {
        "race_id": race_id,
        "enriched_at": datetime.utcnow(),
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
        "key_flags": json.dumps(pred.key_flags),
        "enriched_json": pred.enriched.model_dump_json(),
        "scheduled_time": scheduled_time or None,
        "class_change": int(pred.enriched.class_change) if pred.enriched.class_change is not None else None,
    }
    if race is not None:
        d.update({
            "venue": race.venue,
            "state": race.state,
            "race_number": race.race_number,
            "race_name": race.race_name,
            "distance": race.distance,
            "track_condition": race.track_condition,
            "field_size": len(race.runners),
            "prize_money": race.prize_money,
            "rail_position": race.rail_position,
        })
    return d


def _best_days_since(ra_days: int | None, hist_days: int | None) -> int | None:
    """Return the smaller (more recent) of the two days-since-last-run values.
    Ignores -1 (RA sentinel for no prior starts) and None."""
    candidates = [d for d in (ra_days, hist_days) if d is not None and d >= 0]
    return min(candidates) if candidates else ra_days


def _runner_response(row: RunnerPredictionRow, last10: dict | None = None) -> dict:
    enriched = {}
    if row.enriched_json:
        try:
            enriched = json.loads(row.enriched_json)
        except Exception:
            pass

    # Primary: RA-scraped form data stored at enrichment time.
    # Fallback: historical_results-derived stats for pre-June-2026 records
    # where enriched_json predates the wins_last_10 field.
    wins_last_10 = enriched.get("wins_last_10")
    places_last_10 = enriched.get("places_last_10")
    starts_last_10 = enriched.get("starts_last_10")
    if starts_last_10 is None and last10 is not None:
        wins_last_10 = last10.get("wins_last_10", 0)
        places_last_10 = last10.get("places_last_10", 0)
        starts_last_10 = last10.get("starts_last_10", 0)

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
        "key_flags": json.loads(row.key_flags or "[]"),
        "form_score": enriched.get("form_score"),
        "wins_last_10": wins_last_10,
        "places_last_10": places_last_10,
        "starts_last_10": starts_last_10,
        "distance_aptitude": enriched.get("distance_aptitude"),
        "sire_name": enriched.get("sire_name"),
        "pedigree_distance_match": enriched.get("pedigree_distance_match"),
        "pedigree_wet_score": enriched.get("pedigree_wet_score"),
        "speed_map_position": enriched.get("speed_map_position"),
        "is_steamed": enriched.get("is_steamed", False),
        "is_drifted": enriched.get("is_drifted", False),
        "trainer_overall_rate": enriched.get("trainer_overall_rate"),
        "jockey_overall_rate": enriched.get("jockey_overall_rate"),
        "days_since_last_run": _best_days_since(enriched.get("days_since_last_run"), last10.get("days_since_last_run_hist") if last10 else None),
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
