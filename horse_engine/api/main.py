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
  OddsPro          — multi-book live odds
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

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from horse_engine.api.database import get_session
from horse_engine.clients.factory import get_tab_client
from horse_engine.config import settings
from horse_engine.models.database import (
    BacktestResultRow,
    BacktestStateRow,
    BetRecommendationRow,
    CalibrationRow,
    ExoticBacktestRow,
    HistoricalResultRow,
    NightlyReviewRow,
    OddsSnapshotRow,
    QualityCheckRow,
    ResponseCacheRow,
    RunnerPredictionRow,
    RunnerPredictionHistoryRow,
    RacePredictionRow,
    WinCalibrationCurveRow,
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
from horse_engine.analysis.nightly_review import (
    generate_review as _generate_nightly_review,
)
from horse_engine.analysis.weekly_review import (
    generate_weekly_review as _generate_weekly_review,
)
from horse_engine.bets import (
    STRATEGY_GROUP as _STRATEGY_GROUP,
    STRATEGY_GROUP_LABELS as _STRATEGY_GROUP_LABELS,
    STRATEGY_REGISTRY as _BET_STRATEGIES,
    TAB_TRIFECTA_TAKEOUT as _TAB_TRIFECTA_TAKEOUT,
    compute_payout as _bet_compute_payout,
    estimate_printed_dividend as _harville_dividend,
    generate_recommendations as _build_bet_basket,
    harville_horse_top_n as _harville_horse_top_n,
    harville_top3_probability as _harville_top3_prob,
    is_hit as _bet_is_hit_typed,
    is_metro_venue as _is_metro_venue,
    is_trifecta_hit as _bet_is_hit,
)
from horse_engine.bookmakers import sportsbet_features as _sb
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


async def _rerank_race_after_scratch(session, race_id: str) -> bool:
    """Re-rank and renormalise probabilities over the surviving (uncancelled)
    runners of a race. Pure DB op — no ML re-inference, no upstream API calls.

    Without this step a cancelled rank-1 leaves model_rank=1 unfilled and the
    Lab queries that key off `model_rank == 1` silently drop the race. Called
    by _check_scratches_today whenever a runner is freshly cancelled.

    Mirrors changes into RunnerPredictionHistoryRow's latest enriched_at
    snapshot so settled-race views stay consistent too.
    """
    from sqlalchemy import update as sa_update

    survivors = (await session.execute(
        select(
            RunnerPredictionRow.id,
            RunnerPredictionRow.horse_name,
            RunnerPredictionRow.win_probability,
            RunnerPredictionRow.place_probability,
            RunnerPredictionRow.exotic_model_rank,
        )
        .where(RunnerPredictionRow.race_id == race_id)
        .where(
            RunnerPredictionRow.cancelled.is_(False)
            | RunnerPredictionRow.cancelled.is_(None)
        )
    )).fetchall()
    if not survivors:
        return False

    sw = sum((row.win_probability or 0) for row in survivors)
    sp = sum((row.place_probability or 0) for row in survivors)

    by_win = sorted(survivors, key=lambda r: -(r.win_probability or 0))
    by_place = sorted(survivors, key=lambda r: -(r.place_probability or 0))
    # Exotic rank is preserved relative-order — sort survivors by their existing
    # exotic_model_rank (None/0 sink to the back) and re-assign 1..N.
    by_exotic = sorted(
        survivors,
        key=lambda r: (r.exotic_model_rank if r.exotic_model_rank else 1_000_000),
    )

    win_rank: dict[int, int] = {r.id: i + 1 for i, r in enumerate(by_win)}
    place_rank: dict[int, int] = {r.id: i + 1 for i, r in enumerate(by_place)}
    exotic_rank: dict[int, int] = {r.id: i + 1 for i, r in enumerate(by_exotic)}

    for row in survivors:
        new_win = ((row.win_probability or 0) / sw) if sw > 0 else 0.0
        new_place = ((row.place_probability or 0) / sp) if sp > 0 else 0.0
        await session.execute(
            sa_update(RunnerPredictionRow)
            .where(RunnerPredictionRow.id == row.id)
            .values(
                win_probability=new_win,
                place_probability=new_place,
                model_rank=win_rank[row.id],
                place_model_rank=place_rank[row.id],
                exotic_model_rank=exotic_rank[row.id],
            )
        )
    # Null out ranks on cancelled rows. Acts as a "re-rank applied" marker so the
    # retroactive scan can identify races still needing re-ranking by looking for
    # cancelled rows that still have a non-null model_rank.
    await session.execute(
        sa_update(RunnerPredictionRow)
        .where(RunnerPredictionRow.race_id == race_id)
        .where(RunnerPredictionRow.cancelled.is_(True))
        .values(model_rank=None, place_model_rank=None, exotic_model_rank=None)
    )

    # Mirror into history's latest snapshot so settled-race code sees the
    # corrected ranking too. Only touches uncancelled history rows.
    latest_at = (await session.execute(
        select(func.max(RunnerPredictionHistoryRow.enriched_at))
        .where(RunnerPredictionHistoryRow.race_id == race_id)
    )).scalar()
    if latest_at is not None:
        hist_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.id,
                RunnerPredictionHistoryRow.horse_name,
            )
            .where(RunnerPredictionHistoryRow.race_id == race_id)
            .where(RunnerPredictionHistoryRow.enriched_at == latest_at)
            .where(
                RunnerPredictionHistoryRow.cancelled.is_(False)
                | RunnerPredictionHistoryRow.cancelled.is_(None)
            )
        )).fetchall()
        # Map survivor by horse_name → new ranks/probs
        by_name = {r.horse_name: r for r in survivors}
        for hist_row in hist_rows:
            surv = by_name.get(hist_row.horse_name)
            if surv is None:
                continue
            new_win = ((surv.win_probability or 0) / sw) if sw > 0 else 0.0
            new_place = ((surv.place_probability or 0) / sp) if sp > 0 else 0.0
            await session.execute(
                sa_update(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.id == hist_row.id)
                .values(
                    win_probability=new_win,
                    place_probability=new_place,
                    model_rank=win_rank[surv.id],
                    place_model_rank=place_rank[surv.id],
                    exotic_model_rank=exotic_rank[surv.id],
                )
            )

    return True


async def _check_scratches_today() -> int:
    """
    Lightweight scratch detection — no ML inference.
    Checks races starting within the next 10 hours so the full afternoon
    racing card is in scope from the first morning cron tick (was 4h,
    which missed scratchings on races jumping 4-10h out).
    Returns count of newly cancelled runners.
    """
    from sqlalchemy import update as sa_update
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    horizon = now_utc + timedelta(hours=10)
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
                            # Re-rank + renormalise over the surviving field so
                            # the Lab's rank-1 queries pick up the promoted horse.
                            try:
                                async with get_session() as rsession:
                                    if await _rerank_race_after_scratch(rsession, race_id):
                                        await rsession.commit()
                                        log.info("[scratch-check] %s: re-ranked survivors", race_id)
                            except Exception as re:
                                log.warning("[scratch-check] %s: rerank failed: %s", race_id, re)
            except Exception as e:
                log.debug("[scratch-check] Failed for %s: %s", slug, e)
    except Exception as e:
        log.exception("[scratch-check] Failed: %s", e)

    # Sync step: catch any mutable-cancelled runners whose history row predates this fix.
    # Runs every call so retroactive scratches (cancelled before this code was deployed) propagate.
    affected_race_ids: set[str] = set()
    try:
        async with get_session() as session:
            already_cancelled_mut = (await session.execute(
                select(RunnerPredictionRow.race_id, RunnerPredictionRow.horse_name)
                .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
                .where(RunnerPredictionRow.cancelled.is_(True))
            )).fetchall()
            if already_cancelled_mut:
                for race_id, horse_name in already_cancelled_mut:
                    affected_race_ids.add(race_id)
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

    # Retroactive re-rank: races flagged cancelled before this fix shipped never
    # had their ranks rebuilt. Detect "stale-ranked" races by looking for any
    # cancelled row that still has a model_rank set — the rerank helper clears
    # those, so a non-null rank on a cancelled row means we haven't run yet.
    try:
        async with get_session() as session:
            stale_race_ids = (await session.execute(
                select(RunnerPredictionRow.race_id).distinct()
                .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
                .where(RunnerPredictionRow.cancelled.is_(True))
                .where(RunnerPredictionRow.model_rank.is_not(None))
            )).scalars().all()
        for race_id in set(stale_race_ids) | affected_race_ids:
            try:
                async with get_session() as rsession:
                    if await _rerank_race_after_scratch(rsession, race_id):
                        await rsession.commit()
            except Exception as re:
                log.warning("[scratch-check] retro rerank %s failed: %s", race_id, re)
    except Exception as e:
        log.warning("[scratch-check] Retro rerank scan failed: %s", e)

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
    # Scratch horizon was 4h — too narrow. Races jumping in the afternoon
    # (15:00-17:00 AEST) when scratchings often appear in the morning
    # were sitting in 'no detection yet' state for hours after the
    # scratching landed. Widen to 10h so the entire day's racing card
    # comes under scratch surveillance from the first morning cron tick.
    scratch_horizon = now_utc + timedelta(hours=10)
    today = _today_aest().isoformat()
    date_sfx = f"-{today.replace('-', '')}"
    log.info("[pre-race] Combined enrich+scratch scan starting")

    try:
        client = get_tab_client()

        # (Previously cleared client._ra._meeting_cache here. That was
        # wiping the 30-min TTL every 15 min and forcing the first user
        # click on any venue after each sweep tick to pay a 25s RA
        # round-trip. The sweep's own per-meeting calls below now pass
        # force_fresh=True — bypass the cache read but STILL write the
        # fresh result back, so user endpoints hit warm cache while the
        # sweep still detects scratchings against fresh RA data.)

        async with get_session() as session:
            model = await _load_model(session)
            place_model = await _load_place_model(session)

        meetings = await client.get_meetings(today)
        if not meetings:
            log.warning("[pre-race] No meetings returned for %s", today)
            return

        # Build a prize-money lookup once. Used to decide which meetings
        # get the heavyweight enrich pass; scratch detection runs on ALL
        # meetings regardless (cheap, and country races still need
        # scratchings flagged so the Funk Me Up Sunday country card
        # doesn't show stale runners).
        PRIZE_MONEY_FLOOR = 30_000
        prize_by_venue: dict[str, int] = {}
        async with get_session() as session:
            from sqlalchemy import func as _func
            prize_rows = (await session.execute(
                select(RunnerPredictionRow.venue,
                       _func.max(RunnerPredictionRow.prize_money))
                .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
                .where(RunnerPredictionRow.prize_money.isnot(None))
                .group_by(RunnerPredictionRow.venue)
            )).fetchall()
        for venue, prize in prize_rows:
            if venue and prize:
                prize_by_venue[venue.strip().lower()] = int(prize)

        def _meeting_is_low_prize(m: dict) -> bool:
            venue = (m.get("venue") or "").strip().lower()
            pm = prize_by_venue.get(venue, 0)
            return bool(pm) and pm < PRIZE_MONEY_FLOOR

        # All meetings get fetched (cheap — cached) so scratchings are
        # caught even on country-only Sundays. Enrichment is the
        # expensive part; gate that separately further down.
        eligible_meetings = list(meetings)

        # Single get_meeting_races pass per meeting — shared by enrich, scratch, and cancel-check
        meeting_meta: dict[str, tuple] = {}   # vc -> (slug, venue_name, state)
        venue_raw_events: dict[str, list] = {}  # vc -> raw_events
        for m in eligible_meetings:
            slug = m.get("slug", "")
            if not slug:
                continue
            vc = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0]
            if _should_skip_venue(vc):
                continue  # blocklisted bush venue — no enrich, no scratch check
            meeting_meta[vc] = (slug, m.get("venue", vc), m.get("state", ""))
            try:
                # force_fresh=True: sweep needs current runner list to
                # catch scratchings, but the fresh result IS cached so
                # subsequent user-facing calls hit warm cache.
                venue_raw_events[vc] = await client.get_meeting_races(slug, force_fresh=True)
            except Exception as e:
                log.debug("[pre-race] get_meeting_races failed for %s: %s", slug, e)

        # Cancel-check using pre-fetched data — no extra API calls
        await _cancel_abandoned_meetings(client, today, meetings=meetings, venue_events=venue_raw_events)

        enriched_count = 0
        scratch_count = 0

        # Map vc → meeting dict so we can apply the prize-money gate per venue
        meeting_by_vc: dict[str, dict] = {}
        for m in eligible_meetings:
            slug = m.get("slug", "")
            if not slug:
                continue
            vc = slug[:-len(date_sfx)] if slug.endswith(date_sfx) else slug.split("-")[0]
            meeting_by_vc[vc] = m

        for vc, raw_events in venue_raw_events.items():
            slug, venue_name, state = meeting_meta[vc]
            # Gate enrichment by prize money — scratch detection still runs.
            # Saves the heavy enrich-and-predict call on sub-$30k meetings
            # that don't generate Lab bets anyway.
            is_low_prize = _meeting_is_low_prize(meeting_by_vc.get(vc, {}))
            for raw_event in raw_events:
                start_raw = raw_event.get("startTime")
                if not start_raw:
                    continue
                try:
                    jump = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                except ValueError:
                    continue

                in_enrich = now_utc <= jump <= enrich_horizon and not is_low_prize
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
                        # Build current-field set, treating runners marked
                        # 'SCRATCHED'/'WITHDRAWN' status as already removed
                        # (otherwise they'd remain in current_field and never
                        # be detected as a scratching).
                        current_field = set()
                        for sel in (full_event.get("selections") or []):
                            comp = sel.get("competitor") or {}
                            name = (comp.get("name") or "").strip()
                            if not name:
                                continue
                            status = (sel.get("status") or comp.get("status") or "").upper()
                            if status in ("SCRATCHED", "WITHDRAWN", "LATE_SCRATCHING", "LATESCRATCHING"):
                                continue
                            current_field.add(name)
                        if not current_field:
                            log.warning(
                                "[scratch-check] %s: empty current_field from TAB; skipping",
                                race_id,
                            )
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
        # Full-market odds for every AU thoroughbred runner today, used as a
        # fallback when the movers feed (which only includes horses whose
        # odds have moved recently) doesn't return a runner. Without this,
        # firm-priced favourites silently fall through to 0 in the DB.
        meeting_odds_by_track: dict[str, dict] = await op.get_meeting_odds(today)

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
            full_market_map = meeting_odds_by_track.get(op_track.lower(), {})
            if not odds_map and not full_market_map:
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
                        if op_runner:
                            raw = op_runner.get("currentBestOdds")
                            try:
                                val = float(raw) if raw else 0.0
                                if val > 1.0:
                                    new_odds[row.id] = val
                            except (TypeError, ValueError):
                                pass
                        else:
                            # Movers feed missed this runner — fall back to the
                            # full-market feed (no steam features but at least
                            # a current price).
                            fm_price = (
                                full_market_map.get((race_num, name_lower))
                                or full_market_map.get((race_num, norm_name))
                            )
                            if fm_price and fm_price > 1.0:
                                new_odds[row.id] = fm_price
                            continue
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
    Runs after parse_race() so it overrides the flat 10.0 defaults from RA.
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
        # Skip venues whose every race is already in historical_results.
        # Without this, every 30-min seed-cron tick re-fetches Results.aspx
        # for already-done venues — by mid-afternoon a Sat metro card
        # burns hundreds of redundant fetches and trips the proxy cap.
        if race_ids and race_ids.issubset(already_seeded_ra):
            continue
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
                        # tab_number is required by the bet-settlement path —
                        # without it, hit/miss can't be evaluated. Pull from
                        # the matched prediction row (same race + horse).
                        tab_number=getattr(matched, "tab_number", None) if matched else None,
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


async def _scheduled_morning_settle():
    """Early-morning catch-up: seed YESTERDAY's results then run settlement.
    The afternoon seed/settle crons don't start until 14:00 AEST, which
    leaves yesterday's late-evening races in pending state for half the
    next day. Running once at 08:00 fills that gap with ~5-15 RA fetches
    (results for yesterday's venues only — today's racing data isn't
    touched, so no double-cost vs the 14:00 sweep)."""
    yesterday = (_today_aest() - timedelta(days=1)).isoformat()
    try:
        n = await _seed_results_for_date(yesterday)
        log.info("[morning-settle] Seeded %d results for %s", n, yesterday)
    except Exception as e:
        log.exception("[morning-settle] Result seeding failed: %s", e)
    try:
        await _scheduled_settle_bets()
    except Exception as e:
        log.exception("[morning-settle] Settlement failed: %s", e)


async def _scheduled_exotic_retrain():
    """Run by APScheduler at 3am AEST — retrain exotic model after nightly calibration."""
    log.info("[scheduler] Running nightly exotic model retrain")
    try:
        from horse_engine.prediction.clean_features import (
            AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
        )
        async with get_session() as session:
            hr_result = await session.execute(select(HistoricalResultRow))
            hr_rows = hr_result.scalars().all()
            # source IN ('live', 'backfill') matches the win/place/exotic
            # calibration sweeps. Excludes 'validation' / 'backtest' sources
            # so backtest-fit signal doesn't leak into exotic weights.
            hist_result = await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
                .where(
                    RunnerPredictionHistoryRow.cancelled.is_(False)
                    | RunnerPredictionHistoryRow.cancelled.is_(None)
                )
                .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill")))
            )
            hist_rows = hist_result.scalars().all()

        # BUG-18-clean aggregate index — used by the clean-recompute path so
        # post-race enriched_at can't leak aggregate fields.
        index = AggregateIndex(hr_rows)

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
                fv = recompute_clean_feature_vector(row, index)
                if fv is None:
                    fv = fallback_feature_vector(row)
                if fv is None:
                    continue
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
            # Compute Sharp eligibility for this race, frozen at snapshot time.
            # Gates: (rank-1 prob ≥0.30 OR top-3 sum ≥0.60)
            #        AND rank-1 days_since_last_run ≤180
            # The flag is set ONLY on the rank-1 row (Sharp is a race-level
            # property scored via the rank-1 pick). Other rows leave is_sharp NULL.
            active = [rr for rr in runners if not rr.cancelled]
            active_sorted = sorted(active, key=lambda rr: rr.model_rank or 99)
            rank1 = active_sorted[0] if active_sorted else None
            top3_sum = sum((rr.win_probability or 0) for rr in active_sorted[:3])
            race_is_sharp = None
            if rank1 is not None:
                _high_conf = ((rank1.win_probability or 0) >= 0.30) or (top3_sum >= 0.60)
                _days_off = None
                if rank1.enriched_json:
                    try:
                        _e = json.loads(rank1.enriched_json)
                        _days_off = _e.get("days_since_last_run")
                    except Exception:
                        _days_off = None
                _layoff_ok = not (isinstance(_days_off, (int, float)) and _days_off > 180)
                race_is_sharp = bool(_high_conf and _layoff_ok)
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
                        # Only the rank-1 row carries the race-level Sharp flag.
                        is_sharp=race_is_sharp if r.model_rank == 1 else None,
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


# Module-level scheduler — assigned during lifespan() startup. Lets other
# code paths (e.g. bet-recommender) schedule per-race one-off jobs.
_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    global _scheduler
    scheduler = AsyncIOScheduler(timezone="Australia/Sydney")
    _scheduler = scheduler
    # Full-enrichment schedule chosen so everything's ready BEFORE first
    # race (typically 11:10 AEST on Sat metro days, 12:00 on weekdays):
    #   08:30 — early baseline (catches overnight market changes)
    #   10:30 — final pre-race refresh (40 min before 11:10 jump)
    # The 13:00 enrich is dropped — it was mid-racing and the per-15-min
    # pre-race-enrich-and-scratch cron already keeps individual race
    # data fresh as their jump windows open.
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=8,  minute=30, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=10, minute=30, timezone="Australia/Sydney"))
    # 11:30 AEST safety-net tick — recovers from days when the 8:30 + 10:30
    # runs were partially eaten by RA 503s. _enrich_date is idempotent
    # (force=True only on today; tomorrow/day-after use cached field), so
    # an extra run on a healthy morning costs ~24s of background work and
    # one set of Calendar.aspx hits, all post-stagger so RA-friendly. On a
    # day RA is flaky this is the cron tick that saves the afternoon.
    scheduler.add_job(_scheduled_enrich, CronTrigger(hour=11, minute=30, timezone="Australia/Sydney"))

    # Edge cache warm-up ticks at strategic times. The continuous prewarm
    # task (`_prewarm_edge_cache` below) refreshes every 60-90s once it
    # spins up, but it can lag right after a redeploy or if the Railway
    # container was idle. These targeted ticks force a fresh build of the
    # cache before users hit the page in significant numbers — first thing
    # in the morning, after each scheduled enrich completes (so the new
    # predictions land in the cache rather than the stale set), and at
    # the start of metro racing.
    async def _warm_edge():
        try:
            await get_edge_picks()
            log.info("[edge-warm] scheduled tick complete")
        except Exception as e:
            log.warning("[edge-warm] scheduled tick failed: %s", e)
        # Pre-warm /api/performance too — the Lounge fires this on every
        # page load for the date pills, and a cold call is ~5-13s on a
        # busy DB. Warming here keeps the Lounge snappy.
        try:
            await performance_summary(days=7, sharp=False)
            await performance_summary(days=7, sharp=True)
            log.info("[edge-warm] performance cache warmed")
        except Exception as e:
            log.warning("[edge-warm] performance warm failed: %s", e)
    scheduler.add_job(_warm_edge, CronTrigger(hour=6,  minute=30, timezone="Australia/Sydney"))  # pre-morning
    scheduler.add_job(_warm_edge, CronTrigger(hour=8,  minute=45, timezone="Australia/Sydney"))  # post 8:30 enrich
    scheduler.add_job(_warm_edge, CronTrigger(hour=10, minute=45, timezone="Australia/Sydney"))  # post 10:30 enrich
    scheduler.add_job(_warm_edge, CronTrigger(hour=11, minute=45, timezone="Australia/Sydney"))  # post 11:30 enrich
    scheduler.add_job(_warm_edge, CronTrigger(hour=12, minute=15, timezone="Australia/Sydney"))  # pre-metro
    scheduler.add_job(_scheduled_prerace_snapshot, CronTrigger(hour=9, minute=0, timezone="Australia/Sydney"))
    # Results seeding — every 30 min during racing hours. Previously only
    # fired at sparse hours (14/15/17/19/23), meaning a 16:00 race would
    # wait until 17:00 to be seeded. Settlement also self-seeds, but this
    # cadence keeps the historical_results table fresh for other readers
    # (edge/yesterday, dashboard, premium-perf, etc.).
    # Was 14-23: first races (Sat metro can start at 11:10) sat for 3h
    # with results published but not seeded. 11-23 catches those within
    # ~20-50 min of finishing. Extra 6 ticks/day × ~5 venues = ~30 RA
    # fetches, well within the proxy budget.
    scheduler.add_job(
        _scheduled_seed_results,
        CronTrigger(hour="11-23", minute="0,30", timezone="Australia/Sydney")
    )
    # 08:00 morning seed+settle for yesterday's results so any late-night
    # races that didn't seed by 23:30 are picked up before users open the
    # app in the morning.
    scheduler.add_job(
        _scheduled_morning_settle,
        CronTrigger(hour=8, minute=0, timezone="Australia/Sydney")
    )
    # Calibration was running daily but docstring says "weekly". Daily
    # was burning CPU on a model whose drift signal moves on a week+
    # timescale anyway. Sundays at 2am only.
    scheduler.add_job(_scheduled_calibrate,      CronTrigger(day_of_week="sun", hour=2,  minute=0, timezone="Australia/Sydney"))
    scheduler.add_job(_scheduled_exotic_retrain, CronTrigger(hour=3,  minute=0, timezone="Australia/Sydney"))
    # Isotonic output-space calibration — nightly rebuild from the last
    # 45 days of predictions vs actuals. Cheap (~1s + 1 DB write).
    # 03:30 keeps it after the exotic retrain (03:00) so the fit uses
    # the freshest model outputs available.
    scheduler.add_job(_scheduled_output_calibration,
                      CronTrigger(hour=3, minute=30, timezone="Australia/Sydney"))
    # Nightly data-integrity check — runs at 04:00 AEST after every
    # 2am–3am modeling job has settled. Persists to quality_checks
    # table; critical findings log at ERROR level for monitoring.
    scheduler.add_job(_scheduled_quality_check,
                      CronTrigger(hour=4, minute=0, timezone="Australia/Sydney"))
    # Racing-hours heal — clears duplicate-position rows + tab-number
    # gaps within a few hours of results landing, so frontend pages
    # render correctly during the day rather than waiting until the
    # 04:00 nightly. Runs at 10:00, 13:00, 16:00, 19:00 AEST.
    scheduler.add_job(_scheduled_racing_hours_heal,
                      CronTrigger(hour="10,13,16,19", minute=0, timezone="Australia/Sydney"))
    # Nightly review intentionally DISABLED (2026-06-30) — single-day
    # denominators were too small to surface actionable signals; weekly
    # carried all the value. _scheduled_nightly_review (and its analyser
    # in horse_engine/analysis/nightly_review.py) are kept on disk in
    # case we want it back — just re-add the scheduler.add_job line.
    # scheduler.add_job(_scheduled_nightly_review,
    #                   CronTrigger(hour=2, minute=30, timezone="Australia/Sydney"))
    # Weekly review — 03:00 AEST every Monday. Reviews the past 7 days
    # (Mon→Sun) and writes a NightlyReviewRow with source='weekly'.
    # Detectors use week-sized denominators (more credible signal than
    # the nightly's single-day window).
    scheduler.add_job(_scheduled_weekly_review,
                      CronTrigger(day_of_week="mon", hour=3, minute=0, timezone="Australia/Sydney"))
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
    # Paper-trading trifecta recommender — runs hourly during racing hours
    # to lock-in pre-race recommendations for the day's qualifying races.
    scheduler.add_job(
        _scheduled_generate_bets,
        CronTrigger(hour="6-21", minute=5, timezone="Australia/Sydney")
    )
    # Settlement sweep — runs every 30 min during the results window.
    # Pushed earlier (was 14-23) to cover Sat metro first race at 11:10.
    # Each settle is DB-only — no RA budget impact.
    scheduler.add_job(
        _scheduled_settle_bets,
        CronTrigger(hour="11-23", minute="10,40", timezone="Australia/Sydney")
    )
    scheduler.start()
    log.info("[scheduler] Cron jobs scheduled")

    # Enrich today on startup — but only if it hasn't run in the last
    # 30 min, to protect the RA proxy on heavy-deploy days (15+ deploys
    # in a session × full _scheduled_enrich each time = 100s of fetches).
    async def _startup_enrich_if_stale():
        from sqlalchemy import func as _func
        try:
            async with get_session() as session:
                last_enriched = (await session.execute(
                    select(_func.max(RunnerPredictionRow.enriched_at))
                )).scalar()
            if last_enriched and (datetime.utcnow() - last_enriched).total_seconds() < 1800:
                log.info("[startup-enrich] skipped — last enrichment %s ago",
                         (datetime.utcnow() - last_enriched))
                return
        except Exception as e:
            log.warning("[startup-enrich] staleness check failed: %s", e)
        await _scheduled_enrich()
    asyncio.create_task(_startup_enrich_if_stale())

    # Bet-gen + settlement + result-seed catch-up on startup. Without this,
    # any deploy that lands between cron ticks defers all of these to the
    # NEXT scheduled run — sometimes an hour away. On heavy-deploy days
    # the gaps compound and today's races can sit out of The Lab for the
    # whole afternoon. Sequenced + delayed so each waits on the previous
    # without dogpiling DB queries on container boot.
    async def _startup_bet_catchup():
        await asyncio.sleep(20)  # let enrichment land first
        try:
            await _scheduled_generate_bets()
        except Exception as e:
            log.warning("[startup] bet-gen catch-up failed: %s", e)
        try:
            await _scheduled_seed_results()
        except Exception as e:
            log.warning("[startup] result-seed catch-up failed: %s", e)
        try:
            await _scheduled_settle_bets()
        except Exception as e:
            log.warning("[startup] settle-bets catch-up failed: %s", e)
    asyncio.create_task(_startup_bet_catchup())

    # Hydrate /api/edge response cache from Postgres before any user can
    # hit the endpoint. Eliminates the 30-60s post-deploy cold-cache
    # window — first user gets last-known-good immediately while the
    # background prewarm refreshes it.
    global _edge_response_cache
    try:
        async with get_session() as session:
            row = (await session.execute(
                select(ResponseCacheRow).where(ResponseCacheRow.cache_key == "edge")
            )).scalar_one_or_none()
        if row and row.cache_version == _EDGE_CACHE_VERSION:
            body = json.loads(row.payload_json)
            # Backdate the timestamp by (TTL - 60s) so the prewarm task
            # refreshes very soon — we serve stale-but-fast immediately,
            # then real data lands within ~60s.
            backdated = datetime.utcnow() - timedelta(seconds=max(_EDGE_RESPONSE_TTL - 60, 0))
            _edge_response_cache = (backdated, body, _EDGE_CACHE_VERSION)
            log.info("[edge] hydrated cache from DB: %d picks, version %d",
                     len(body.get("picks", [])), row.cache_version)
    except Exception as e:
        log.warning("[edge] DB hydration skipped: %s", e)

    # Pre-warm the /api/edge response cache so users don't pay the 25-30s
    # cold-cache cost. Refresh every 4 min + 120s jitter (4-6 min). Cache
    # TTL is bumped to 7 min in parallel (_EDGE_RESPONSE_TTL=420) so the
    # worst-case 6-min prewarm interval still has 1 min safety margin
    # before the cache would go cold for a real user.
    async def _prewarm_edge_cache():
        """Keep the /api/edge response cache warm. Cache TTL is 7 min; we
        refresh every 4-6 min jittered so users never hit a cold-cache
        25-60s recompute. On failure, retry after 30s (not the full 4-6 min)
        so a transient hiccup doesn't leave the cache stale for 5+ minutes."""
        # Warm immediately on startup so the first user after a redeploy
        # doesn't hit a cold-cache timeout. 5s buffer for DB pool init.
        await asyncio.sleep(5)
        consecutive_failures = 0
        while True:
            try:
                body = await get_edge_picks()
                picks_count = len(body.get("picks", []))
                if consecutive_failures > 0:
                    log.info("[edge-prewarm] Recovered — cache warm with %d picks", picks_count)
                else:
                    log.info("[edge-prewarm] Cache warm with %d picks", picks_count)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                log.warning("[edge-prewarm] failed (consecutive=%d): %s", consecutive_failures, e)
            # On success: refresh every 60-90s (1-1.5 min) so finished-race
            # RESULTS surface on Hot Seat cards within ~3 min of the race
            # ending. Faster than the 120s TTL so users never hit a cold cache.
            # On failure: retry in 30s × consecutive failures (cap 5 min).
            if consecutive_failures == 0:
                await asyncio.sleep(60 + random.uniform(0, 30))
            else:
                retry_delay = min(30 * consecutive_failures, 300)
                await asyncio.sleep(retry_delay)
    asyncio.create_task(_prewarm_edge_cache())

    # Prewarm /api/edge/yesterday once on startup so the first user after
    # a redeploy doesn't pay the 10s on-demand-seed cost. Yesterday's
    # data is stable — no need for a refresh loop; one warming call
    # writes to the DB row that subsequent boots hydrate from instantly.
    async def _prewarm_yesterday():
        await asyncio.sleep(10)
        try:
            await get_edge_yesterday()
            log.info("[edge-yesterday-prewarm] cache warm")
        except Exception as e:
            log.warning("[edge-yesterday-prewarm] failed: %s", e)
    asyncio.create_task(_prewarm_yesterday())

    # Prewarm /api/meetings for the date strip range (-7..+2). After a
    # redeploy, every date click on the main page would otherwise pay
    # the 10-15s RA cold-fetch cost. Each date prewarm reads from the
    # ResponseCacheRow first (instant if it exists from a prior process),
    # so this only triggers RA traffic for genuinely new dates.
    async def _prewarm_meetings_strip():
        await asyncio.sleep(15)  # let edge prewarm + DB pool settle first
        try:
            today = _today_aest()
            dates = [(today + timedelta(days=i)).isoformat() for i in range(-7, 3)]
            for d in dates:
                try:
                    await list_meetings(d)
                except Exception as e:
                    log.debug("[meetings-prewarm] %s skipped: %s", d, e)
                await asyncio.sleep(2)  # gentle pace
            log.info("[meetings-prewarm] %d dates warm", len(dates))
        except Exception as e:
            log.warning("[meetings-prewarm] failed: %s", e)
    asyncio.create_task(_prewarm_meetings_strip())

    # Backfill last 3 days — catch up on any missed enrichments/results.
    # Throttled per-date: skip dates whose latest enriched_at is < 12h old.
    # Without this, every Railway redeploy (often several per day during
    # active work) would re-enrich the same 3 days, burning ~60 req/min on
    # the RA proxy for tens of minutes per deploy. The throttle reads
    # MAX(enriched_at) from RunnerPredictionRow — no new schema needed.
    async def _startup_backfill():
        client = get_tab_client()
        async with get_session() as session:
            model = await _load_model(session)
        from sqlalchemy import func as _func
        skip_if_within = timedelta(hours=12)
        for offset in (-3, -2, -1):
            seed_date = (_today_aest() + timedelta(days=offset)).isoformat()
            try:
                async with get_session() as session:
                    last_enriched = (await session.execute(
                        select(_func.max(RunnerPredictionRow.enriched_at))
                        .where(RunnerPredictionRow.race_id.like(f"{seed_date}_%"))
                    )).scalar()
                if last_enriched and (datetime.utcnow() - last_enriched) < skip_if_within:
                    age_h = round((datetime.utcnow() - last_enriched).total_seconds() / 3600, 1)
                    log.info("[startup] Skipping backfill for %s — last enriched %sh ago", seed_date, age_h)
                    continue
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


class BlocklistMiddleware(BaseHTTPMiddleware):
    """Drop requests from hard-blocked IPs at the very edge of the app, before
    any route handler runs. Returns 403 from a single global choke point so we
    don't have to wire _enforce_caller_rate into every endpoint.

    Uses the same XFF-leftmost rule as _caller_origin so the block applies to
    the real client, not Railway's edge proxy IP."""
    async def dispatch(self, request: StarletteRequest, call_next):
        xff = request.headers.get("x-forwarded-for", "")
        origin = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "")
        if origin in _HARD_BLOCKED_IPS:
            # Rate-limit our own logging — once per minute per origin
            now = _time.time()
            log_key = f"_mwblocklog:{origin}"
            last = _caller_hits.get(log_key, [0.0])[-1]
            if now - last >= 60.0:
                log.warning(
                    "[blocklist-mw] DROP %s %s — ua=%r",
                    request.method, request.url.path,
                    request.headers.get("user-agent", "?")[:80],
                )
                _caller_hits[log_key] = [now]
            from starlette.responses import JSONResponse
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
        return await call_next(request)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BlocklistMiddleware)


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


# ── Sportsbet-availability flag ──────────────────────────────────────────
# Venues that Sportsbet doesn't market — mostly tiny picnic / bush meetings
# where turnover is too thin for a corporate book. RA lists them anyway
# because they're stewards-approved, so our picks include them; this flag
# lets the frontend badge them so the user knows they can't back the pick
# on their usual book.
#
# List grows organically as venues are reported missing. Add lowercase
# hyphenated venue codes (matches how race_ids are constructed).
_NOT_ON_SPORTSBET_VENUES: frozenset = frozenset({
    "ilfracombe",       # QLD, Central West — picnic race club
    "esk",              # QLD, Somerset region
    "ingham",           # QLD, North Queensland
    "roma",             # QLD, Maranoa
    "wean-picnic",      # NSW/QLD picnic — Sportsbet doesn't cover
})


# Ingest-skip list — venues we don't want in our pipeline at all.
# Any venue whose enrichment/prediction data would just clutter The
# Lounge / Lab / Edge without being actionable (user can't back these
# picks on their book). Slugs match RA's meeting slug (before the
# date suffix). Currently mirrors _NOT_ON_SPORTSBET_VENUES; kept as
# a separate constant so we can differ later (e.g. flag on Sportsbet
# but not skip on TAB).
_SKIP_ENRICHMENT_VENUES: frozenset = frozenset(_NOT_ON_SPORTSBET_VENUES)


def _should_skip_venue(venue_code: str) -> bool:
    """Return True if a venue should be skipped by the enrich pipeline."""
    if not venue_code:
        return False
    return venue_code.lower() in _SKIP_ENRICHMENT_VENUES


def _is_sportsbet_available(venue_code: str, best_odds: Optional[float]) -> bool:
    """
    True when a Sportsbet market almost certainly exists for this
    pick, False when the venue is on the known-blocklist.

    Note: missing best_odds is NOT a signal here. Corporate books
    price races progressively (metros first, early-morning WA last)
    so a null-odds moment on a covered venue was producing false
    "Not on Sportsbet" chips (Morphettville, etc.). Rely on the
    blocklist only; missing odds are handled by other UX signals.
    """
    if not venue_code:
        return True
    return venue_code.lower() not in _NOT_ON_SPORTSBET_VENUES


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

# Cache full /api/edge response. /api/edge is the most expensive read in
# the app — 4 sequential rounds of asyncio.gather across upcoming days,
# each round potentially hitting RA for race times + live odds. Cold-cache
# costs ~30s when RA is blocked or breaker is oscillating. The breaker
# resets every 60s on first failure so a short TTL (60s) means every
# minute one user pays the slow cost.
#
# 120s (2 min) keeps finished-race RESULTS fresh on Hot Seat — the
# /api/edge response includes won/placed/actual_position annotations
# for finished picks, but a cache only refreshes when it expires.
# At 120s TTL + 60-90s prewarm interval, results land on cards within
# ~3 minutes of the race finishing (race ends → seed-on-demand fires
# on next compute → next compute happens within TTL → users see).
# Cost: more recomputes per hour, but each one is fast (~200ms after
# the gather settles) and the prewarmer absorbs all of it.
_EDGE_RESPONSE_TTL = 120
# Bump _EDGE_CACHE_VERSION whenever threshold or response shape changes so
# old cached responses are invalidated on deploy without a manual restart.
_EDGE_CACHE_VERSION = 11  # 2026-06-30: Edge trifecta now sources from Lab's trio_only (place-model top-3)
_edge_response_cache: tuple[datetime, dict, int] | None = None

# Cache full list_meetings response for 10 min (weather + RA calls are expensive)
_list_meetings_cache: dict[str, tuple[datetime, dict]] = {}  # date → (ts, response)


def _ra_breaker_open(client) -> bool:
    """Probe the composite client's RA breaker state. When open we skip
    external fetches in /api/edge and /api/meetings — they'd just hit the
    breaker and return empty after burning their timeout budget.
    Returns False if the client doesn't expose the breaker for any reason
    (we'd rather attempt and timeout than wrongly skip)."""
    try:
        return bool(client._ra._is_blocked())
    except Exception:
        return False

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
        # 5s timeout (was 20s). When RA is healthy this returns in ~500ms;
        # when blocked we want to fail fast and skip live odds rather than
        # block the whole /api/edge response for 20s per missing race.
        event = await asyncio.wait_for(client.get_race(slug, race_num), timeout=5)
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
        # 5s timeout (was 20s) — same reasoning as _fetch_live_odds.
        events = await asyncio.wait_for(client.get_meeting_races(slug), timeout=5)
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


_TWO_FUNK_TOP2_MIN = 55.0  # rank-1 + rank-2 win sum threshold (percentage points)
_TWO_FUNK_GAP_MIN = 10.0   # rank-2 vs rank-3 win gap threshold (pt)

# ── Feature → membership-tier registry ─────────────────────────────────
# Single source of truth for which feature belongs to which paid tier.
# Used by API responses to tag returned objects with their tier so the
# frontend (and the future gating layer) can paywall/hide consistently.
# Tiers escalate: edge < pro < labs. A user with "pro" gets everything
# tagged "edge" or "pro"; "labs" gets all three; the hidden "admin"
# tier (Russell-only) gets everything plus backtest/admin tooling.
FEATURE_TIERS: dict[str, str] = {
    # Edge tier — base predictions + edge picks
    "edge_picks": "edge",
    "field_expansion": "edge",
    # Pro tier — race-card analytics
    "hot_seat": "pro",
    "funk_me_up": "pro",
    "trifecta": "pro",       # edge race-card trifecta strip
    "first_four": "pro",     # edge race-card first-4 row
    "hedge": "pro",          # insurance-hedge calculator on edge cards
    # Funk-Me-Up play kinds — each play inherits its tier
    "play_lock": "pro",
    "play_bonus": "pro",
    "play_double": "pro",
    "play_banker": "pro",
    "play_quaddie": "pro",
    "play_combo_doubles": "pro",
    # Labs tier — premium analytics
    "lab": "labs",
    "two_funk": "labs",
    "lab_trifecta_box": "labs",
    "play_lab": "labs",
    "play_twofunk": "labs",
    # Legacy / retired play kinds — kept for historical (yesterday) renders
    "play_spine": "pro",
    "play_wave": "pro",
}


def feature_tier(feature: str) -> str:
    """Look up the membership tier that owns a feature. Defaults to 'edge'
    (i.e. base tier) if a feature is unmapped, so we never accidentally
    paywall something we haven't classified."""
    return FEATURE_TIERS.get(feature, "edge")


def _compute_two_funk(rank1_win_prob: float, hedge_runners: list, hedge_candidates: list[dict]) -> dict | None:
    """Detect the 'two clear standouts' pattern.

    Qualifies when:
      - rank-1 + rank-2 win probabilities sum to ≥ _TWO_FUNK_TOP2_MIN
      - rank-2 minus rank-3 win probability ≥ _TWO_FUNK_GAP_MIN

    hedge_runners is the rank 2-5 RunnerPredictionRow / HistoryRow list
    (sorted by model_rank). hedge_candidates is the same list pruned to
    horses with live win_odds available, used to enrich the partner with
    a tradeable odds figure.
    """
    if not hedge_runners or len(hedge_runners) < 2:
        return None
    rank2_row = next((hr for hr in hedge_runners if hr.model_rank == 2), None)
    rank3_row = next((hr for hr in hedge_runners if hr.model_rank == 3), None)
    if rank2_row is None or rank3_row is None:
        return None
    r1 = (rank1_win_prob or 0) * 100
    r2 = (rank2_row.win_probability or 0) * 100
    r3 = (rank3_row.win_probability or 0) * 100
    top2 = r1 + r2
    gap = r2 - r3
    qualified = (top2 >= _TWO_FUNK_TOP2_MIN) and (gap >= _TWO_FUNK_GAP_MIN)
    partner_odds = next(
        (h["win_odds"] for h in hedge_candidates if h["horse_name"] == rank2_row.horse_name),
        rank2_row.best_available_odds or 0,
    )
    return {
        "qualified": qualified,
        "top2_win_sum_pct": round(top2, 1),
        "rank2_rank3_gap_pct": round(gap, 1),
        "partner_horse_name": rank2_row.horse_name,
        "partner_tab_number": rank2_row.tab_number,
        "partner_win_pct": round(r2, 1),
        "partner_odds": partner_odds or None,
    }


@app.get("/api/edge")
async def get_edge_picks():
    """All picks for today + next 3 days with model win_probability >= 20%.

    Lowered from 29.5% to 20% on 2026-06-13 to support the Hot Seat view
    (mobile live-picks list for at-the-TAB use). Edge.html and Hot Seat
    both filter further client-side via display tiers / filter pills, so
    the API just returns the union and the UIs slice as needed."""
    global _edge_response_cache
    # Response cache: serve a fully-assembled response if it's <_EDGE_RESPONSE_TTL old.
    # First user pays the slow cost, everyone else for the TTL window is instant.
    if _edge_response_cache is not None:
        ts, body, version = _edge_response_cache
        if version == _EDGE_CACHE_VERSION and (datetime.utcnow() - ts).total_seconds() < _EDGE_RESPONSE_TTL:
            return body
    threshold = 0.20
    picks = []
    today = _today_aest()
    client = get_tab_client()
    # When RA's breaker is open, the per-venue/per-race external fetches in
    # asyncio.gather below would each burn their 5s timeout and return empty.
    # Skip them entirely — picks still surface from DB, just without live odds
    # overrides and scheduled-time enrichment.
    ra_blocked = _ra_breaker_open(client)

    # One-day horizon — was 4 days (today + next 3) which generated ~75%
    # wasted work because tomorrow's predictions don't exist yet in the
    # morning and day-after is empty. Drop to today only; users who want
    # tomorrow's picks can wait until tomorrow. The morning enrich still
    # processes today + next 2 days into the DB, so the data exists if a
    # future endpoint wants it.
    for i in range(1):
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

            # Batch-fetch the Lab's committed trio_only + core_top4 bets for
            # these races. The Edge trifecta strip mirrors what the Lab is
            # actually paper-trading so both pages tell the same story for
            # the same race — and races that don't pass the Lab's gates
            # (top-3 sum ≤55%, field ≤11) get no strip at all.
            #
            # Switched trifecta from core_top3 → trio_only on 2026-06-30.
            # 60-day comparison showed trio_only beats core_top3 on both
            # hit rate (11.8% vs 8.6%) and P&L (+$1386 vs +$1158) over
            # the same 144 / 163 settled bets. trio_only uses the PLACE
            # model's top-3 — structurally the right model for trifecta
            # since you need 3 horses to finish in placings, not 1 to win.
            #
            # First-four still uses core_top4. Quad_only had higher hit
            # rate (20.1% vs 17.8%) but lower $/box ($4.41 vs $5.01) —
            # we keep the better P&L variant.
            all_race_ids_for_bets = list(hist_race_ids | set(mut_race_ids_list))
            lab_top3_by_race: dict[str, BetRecommendationRow] = {}
            lab_top4_by_race: dict[str, BetRecommendationRow] = {}
            if all_race_ids_for_bets:
                lab_bet_rows = (await session.execute(
                    select(BetRecommendationRow)
                    .where(BetRecommendationRow.race_id.in_(all_race_ids_for_bets))
                    .where(BetRecommendationRow.strategy_label.in_(("trio_only", "core_top4")))
                )).scalars().all()
                # If multiple bets per (race, strategy) somehow exist, prefer the
                # most recently recommended one.
                for b in sorted(lab_bet_rows, key=lambda x: x.recommended_at or datetime.min):
                    if b.strategy_label == "trio_only":
                        lab_top3_by_race[b.race_id] = b
                    elif b.strategy_label == "core_top4":
                        lab_top4_by_race[b.race_id] = b

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

            # Full field per race — Hot Seat's expansion view shows every
            # runner with win/place %s + odds. ~10 rows/race × ~30 races =
            # ~300 rows extra; one batched query per source.
            field_runner_rows: list = []
            if hist_race_ids:
                field_runner_rows.extend((await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id.in_(hist_race_ids))
                )).scalars().all())
            if mut_race_ids_list:
                field_runner_rows.extend((await session.execute(
                    select(RunnerPredictionRow)
                    .where(RunnerPredictionRow.race_id.in_(mut_race_ids_list))
                )).scalars().all())

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

        # Full field per race for the Hot Seat expansion view
        field_map: dict[str, list[dict]] = {}
        for fr in field_runner_rows:
            field_map.setdefault(fr.race_id, []).append({
                "rank": fr.model_rank,
                "horse_name": fr.horse_name,
                "win_pct": round((fr.win_probability or 0) * 100, 1),
                "place_pct": round((fr.place_probability or 0) * 100, 1) if fr.place_probability else None,
                "odds": fr.best_available_odds,
                "scratched": bool(fr.cancelled),
            })
        for key in field_map:
            field_map[key].sort(key=lambda r: r["rank"] or 999)

        # Fetch scheduled times per unique meeting — we used to issue a
        # parallel _fetch_race_times call to RA's Acceptances.aspx for
        # every (venue × 4 dates), burning ~30 calls per /api/edge cache
        # miss. The runner rows already carry scheduled_time from the
        # last enrich; they update on the 15-min pre-race cron, which is
        # plenty fresh for jump-time display. So skip the external fetch
        # and let the existing `runner_row.scheduled_time` fallback below
        # do the work. Saves several hundred Acceptances/day.
        unique_venues = {_parse_race_id(r.race_id)[1] for r in rows}
        slug_map = {v: _meeting_slug(v, target_date) for v in unique_venues}
        race_times: dict[str, str | None] = {}
        live_odds_by_race: dict[str, dict[str, float]] = {}
        if not ra_blocked:
            # For races where rank-2/3 hedge candidates have 0 odds, fetch
            # live odds — but ONLY for races within the next 4 hours. Future
            # races (afternoon / next-day / day-after) get their odds
            # refreshed by the 15-min _scheduled_pre_race_enrich_and_scratch
            # cron once they enter the 2h window. Pre-this-change cold-cache
            # /api/edge fired ~30+ concurrent RA hits and waited on them all,
            # blowing wall time past 60s and violating the no-API-hammer rule.
            now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
            horizon = now_utc + timedelta(hours=4)
            def _imminent(row) -> bool:
                st = row.scheduled_time
                if not st:
                    return False
                try:
                    jump = datetime.fromisoformat(str(st).replace("Z", "+00:00"))
                except ValueError:
                    return False
                return now_utc <= jump <= horizon
            races_needing_live_odds = [
                r.race_id for r in rows
                if _imminent(r)
                and any((hr.best_available_odds or 0) <= 1.0 for hr in hedge_map.get(r.race_id, []))
            ]
            # Cap concurrency at 3 via semaphore so we never stampede the
            # proxy. asyncio.gather still preserves request ordering, but
            # only 3 fly at a time.
            _live_odds_sem = asyncio.Semaphore(3)
            async def _fetch_capped(rid: str):
                async with _live_odds_sem:
                    return await _fetch_live_odds(client, rid)
            live_odds_results = await asyncio.gather(
                *[_fetch_capped(rid) for rid in races_needing_live_odds]
            )
            live_odds_by_race = dict(zip(races_needing_live_odds, live_odds_results))

        for runner_row in rows:
            odds = runner_row.best_available_odds or hist_odds_override.get((runner_row.race_id, runner_row.horse_name), 0)
            model_pct = round(runner_row.win_probability * 100, 1)
            market_implied_pct = round((1 / odds) * 100, 1) if odds else None
            edge_pct = round(model_pct - market_implied_pct, 1) if market_implied_pct else None
            calibrated = next((r for t, r in _CALIBRATED_WIN_RATES if model_pct >= t), 66)
            hot = model_pct >= 45
            _, venue_code, race_num = _parse_race_id(runner_row.race_id)

            # Build trifecta legs by mirroring the Lab's committed paper bets.
            # 'core_top3' = 3-horse trifecta box (top-3 win model). 'core_top4'
            # = 4-horse first-four box. Races that don't pass the Lab's gating
            # criteria (top-3 sum ≤55, field ≤11) won't have these rows, which
            # means no strip on Edge — by design.
            lab_top3 = lab_top3_by_race.get(runner_row.race_id)
            lab_top4 = lab_top4_by_race.get(runner_row.race_id)
            # Build a per-race horse lookup so we can attach place_pct + tab
            # to each leg. Prefer the place model row (so place_pct matches
            # what the Lab is using), fall back to the win pick itself or
            # field_map (which carries every runner).
            tri_horse_by_name = {pr.horse_name: pr for pr in trifecta_map.get(runner_row.race_id, [])}
            field_for_race = field_map.get(runner_row.race_id, [])
            tri_field_by_name = {f["horse_name"]: f for f in field_for_race}

            def _leg_from_lab(name: str) -> Optional[dict]:
                pr = tri_horse_by_name.get(name)
                if pr is not None:
                    return {
                        "tab_number": pr.tab_number,
                        "horse_name": pr.horse_name,
                        "place_pct": round((pr.place_probability or 0) * 100, 1) if pr.place_probability else None,
                    }
                fr = tri_field_by_name.get(name)
                if fr is not None:
                    return {
                        "tab_number": fr.get("tab_number"),
                        "horse_name": name,
                        "place_pct": fr.get("place_pct"),
                    }
                # Last resort: leg with just the name + tab (from the Lab bet)
                return {"tab_number": None, "horse_name": name, "place_pct": None}

            tri_legs: list[dict] = []
            ff_legs: list[dict] = []
            if lab_top3:
                try:
                    names = json.loads(lab_top3.box_horse_names_json or "[]")
                    tabs = json.loads(lab_top3.box_horses_json or "[]")
                    for nm, tab in zip(names, tabs):
                        leg = _leg_from_lab(nm) or {"tab_number": tab, "horse_name": nm, "place_pct": None}
                        if leg.get("tab_number") is None and tab is not None:
                            leg["tab_number"] = tab
                        tri_legs.append(leg)
                except Exception:
                    tri_legs = []
            if lab_top4:
                try:
                    names = json.loads(lab_top4.box_horse_names_json or "[]")
                    tabs = json.loads(lab_top4.box_horses_json or "[]")
                    for nm, tab in zip(names, tabs):
                        leg = _leg_from_lab(nm) or {"tab_number": tab, "horse_name": nm, "place_pct": None}
                        if leg.get("tab_number") is None and tab is not None:
                            leg["tab_number"] = tab
                        ff_legs.append(leg)
                except Exception:
                    ff_legs = []
            # Combined-probability estimate: any-order trifecta hit rate. The
            # naive product-of-place-probs (the old formula) ignores that
            # filling one of the three slots raises the conditional chance
            # of the others. Use it as an approximation, no overround divisor.
            tri_probs = [l["place_pct"] for l in tri_legs if l.get("place_pct") is not None]
            tri_combined = round(tri_probs[0] * tri_probs[1] * tri_probs[2] / 10000, 1) if len(tri_probs) == 3 else None
            ff_probs = [l["place_pct"] for l in ff_legs if l.get("place_pct") is not None]
            ff_combined = round(ff_probs[0] * ff_probs[1] * ff_probs[2] * ff_probs[3] / 1000000, 1) if len(ff_probs) == 4 else None
            # Exotic alignment: compare exotic model's top-3 against win pick
            exotic_top3 = exotic_top3_map.get(runner_row.race_id, set())
            if not exotic_top3:
                exotic_alignment = "no_exotic"
            elif runner_row.horse_name in exotic_top3:
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
                "source": "lab",  # mirror of BetRecommendationRow trio_only + core_top4
                "tier": feature_tier("trifecta"),       # paywall hint — "pro"
                "first_four_tier": feature_tier("first_four"),
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
            if hedge:
                hedge["tier"] = feature_tier("hedge")  # paywall hint — "pro"
            two_funk = _compute_two_funk(runner_row.win_probability, hedge_runners, hedge_candidates)
            if two_funk:
                two_funk["tier"] = feature_tier("two_funk")  # paywall hint — "labs"
                # EV gate — recommend the Quinella only when the math is +EV.
                # Both partners short-priced (e.g. $2.25 + $2.50) gives a
                # ~$2.81 dividend that doesn't cover the loss rate.
                _l1 = odds if odds and odds > 1.0 else 0
                _l2 = two_funk.get("partner_odds") or 0
                if _l1 > 1.0 and _l2 > 1.0:
                    _quin_est = 0.5 * _l1 * _l2
                    _hit_prob = 2 * (runner_row.win_probability or 0) * ((two_funk.get("partner_win_pct") or 0) / 100)
                    _ev = _hit_prob * _quin_est * 10 - 10  # $10 flat stake
                    two_funk["quinella_est_dividend"] = round(_quin_est, 2)
                    two_funk["estimated_ev_dollars"] = round(_ev, 2)
                    two_funk["recommend"] = bool(two_funk.get("qualified")) and _ev > 0
                else:
                    # No live odds on one or both partners — can't compute EV
                    # accurately. Don't recommend; surface as informational only.
                    two_funk["estimated_ev_dollars"] = None
                    two_funk["recommend"] = False

            # Last-10 form (wins / placings / starts) for the horse-card display
            # — same source as _runner_response uses: enriched_json on the pred row.
            try:
                enriched_json_payload = json.loads(runner_row.enriched_json) if runner_row.enriched_json else {}
            except Exception:
                enriched_json_payload = {}
            wins_last_10 = enriched_json_payload.get("wins_last_10")
            places_last_10 = enriched_json_payload.get("places_last_10")
            starts_last_10 = enriched_json_payload.get("starts_last_10")
            days_since_last_run = enriched_json_payload.get("days_since_last_run")
            # Server-computed is_sharp flag — single source of truth for the
            # Sharp niche across all surfaces (Edge / Hot Seat / Lounge).
            # Gates: (model_pct ≥30 OR top-3 sum ≥60) AND days_off ≤180.
            field_top3 = (field_map.get(runner_row.race_id, []) or [])[:3]
            top3_sum_pct = sum(f.get("win_pct") or 0 for f in field_top3)
            _high_conf = (model_pct or 0) >= 30 or top3_sum_pct >= 60
            _layoff_ok = not (isinstance(days_since_last_run, (int, float)) and days_since_last_run > 180)
            is_sharp = bool(_high_conf and _layoff_ok)

            picks.append({
                "date": target_date,
                "race_id": runner_row.race_id,
                "venue": venue_code,
                "state": None,
                "race_number": race_num,
                "race_name": None,
                "distance": None,
                "track_condition": None,
                # Placeholder T00:00:00 values are written when the upstream
                # had a race name but no real start time — treat as missing
                # so the frontend doesn't render "12:00 AM" or run a bogus
                # countdown timer.
                "scheduled_time": (lambda s: None if (isinstance(s, str) and "T00:00:00" in s) else s)(
                    race_times.get(runner_row.race_id) or runner_row.scheduled_time
                ),
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
                "wins_last_10": wins_last_10,
                "places_last_10": places_last_10,
                "starts_last_10": starts_last_10,
                "days_since_last_run": days_since_last_run,
                "is_sharp": is_sharp,
                "top3_sum_pct": round(top3_sum_pct, 1),
                "trifecta": trifecta,
                "hedge": hedge,
                "two_funk": two_funk,
                "field": field_map.get(runner_row.race_id, []),
                "sportsbet_available": _is_sportsbet_available(venue_code, odds),
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
            # 2 Funk: annotate partner position + quinella hit so the card
            # can celebrate when both rank-1 and rank-2 fill the placings.
            tf = p.get("two_funk")
            if tf and tf.get("partner_horse_name"):
                partner_pos = db_positions.get((p["race_id"], tf["partner_horse_name"]))
                if partner_pos is not None:
                    tf["partner_actual_position"] = partner_pos
                if pos is not None and partner_pos is not None:
                    tf["hit"] = {pos, partner_pos} == {1, 2}

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

    # Attach hot-streak badges — the horse's most recent prior race
    # was a rank-1 or ≥25% pick that WON (hot) or PLACED (warm).
    # Backtest showed hot picks lift +18pp win rate, warm +11pp
    # (n=20+21 in the 90-day validation window). Small volume — about
    # 4-8 badges per week — but a genuine signal worth surfacing.
    streak_map = await _load_hot_streak_map(picks)
    for p in picks:
        streak = streak_map.get(p["race_id"])
        if streak:
            p["streak"] = streak

    body = {
        "generated_at": datetime.utcnow().isoformat(),
        "threshold_pct": int(threshold * 100),
        "tier": feature_tier("edge_picks"),       # surface tier — "edge"
        "picks": picks,
    }
    _edge_response_cache = (datetime.utcnow(), body, _EDGE_CACHE_VERSION)
    # Persist to Postgres so the next container redeploy can hydrate
    # this cache before any user request — kills the 30-60s post-deploy
    # cold-cache window.
    try:
        async with get_session() as session:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(ResponseCacheRow).values(
                cache_key="edge",
                payload_json=json.dumps(body),
                cache_version=_EDGE_CACHE_VERSION,
                updated_at=datetime.utcnow(),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["cache_key"],
                set_=dict(payload_json=stmt.excluded.payload_json,
                          cache_version=stmt.excluded.cache_version,
                          updated_at=stmt.excluded.updated_at),
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as e:
        log.debug("[edge] persistent-cache write skipped: %s", e)
    return body


_odds_refresh_last: datetime | None = None
_ODDS_REFRESH_COOLDOWN = 120  # seconds — prevents hammering RA/OddsPro

# Per-(endpoint, key) debounce. Any admin endpoint that triggers upstream
# fetches (Racing Australia, TAB, OddsPro) must register here so a wayward
# caller — Railway dashboard cron set to seconds, a stuck curl loop, an
# accidental retry storm — cannot drag us into a 403/WAF ban.
# Hard rule (feedback_no_api_hammer.md): this program can not HAMMER apis.
_admin_debounce: dict[tuple[str, str], datetime] = {}
_ADMIN_DEBOUNCE_SECONDS = 300  # 5 min — admin reenrich/restore shouldn't fire
                               # more often than this from any source.

def _should_debounce(endpoint: str, key: str) -> tuple[bool, float]:
    """Return (should_skip, age_seconds_of_last_call). Updates timestamp on
    accept so the next call within the window is debounced."""
    now = datetime.utcnow()
    k = (endpoint, key)
    last = _admin_debounce.get(k)
    if last is not None:
        age = (now - last).total_seconds()
        if age < _ADMIN_DEBOUNCE_SECONDS:
            return True, age
    _admin_debounce[k] = now
    return False, 0.0


# ── Per-caller rate limit (kill switch for runaway/bot callers) ──────────────
# Tracks recent request timestamps per origin IP. Any origin that exceeds
# the threshold gets 429'd until the window slides past.
#
# IMPORTANT: we identify the origin from the LEFTMOST X-Forwarded-For entry
# (Railway's edge proxy puts the real client IP there). All Railway traffic
# arrives via 100.64.x.x so request.client.host is useless for distinguishing
# callers — XFF is the only signal we can act on.
#
# Spoofing risk: a malicious caller can send their own XFF and look like
# multiple origins. We don't defend against that here. For our use case
# (curl loops + accidental hammering) XFF-based tracking is sufficient.
import time as _time
_caller_hits: dict[str, list[float]] = {}
_CALLER_RATE_LIMIT = 5       # max requests per origin per window
_CALLER_RATE_WINDOW = 60.0   # seconds

# Hard IP blocklist — short-circuit before rate-limit/auth so persistent
# bad actors stop adding log noise. Add IPs here only after confirming
# they aren't a legitimate user.
#
# Empty by default. Add an IP only after you've confirmed via geo-lookup
# that it ISN'T your own home connection or any of your users.
#
# Mistake history (2026-06-12): 202.172.97.92 was added thinking it was a
# bot — turned out to be the user's own Spintel home IP. Their laptop
# was running curl probes (probably an old `while true` script) and
# Claude Code's outbound traffic also went out through the same address.
# Always check `curl https://ipinfo.io/ip` from the user's machine first.
_HARD_BLOCKED_IPS: set[str] = set()

def _caller_origin(request: Request) -> str:
    """Resolve the upstream caller IP from the XFF chain. Leftmost entry
    is the real client; fall back to request.client.host (will be Railway
    internal) if XFF is missing."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _enforce_caller_rate(request: Request, endpoint: str) -> None:
    """403 if blocklisted; 429 if over rate limit. Called BEFORE auth so
    even unauthenticated scans can't outrun the limiter."""
    origin = _caller_origin(request)
    if origin in _HARD_BLOCKED_IPS:
        # Don't even log every hit — once per origin per minute is enough
        # to confirm the block is still firing.
        now = _time.time()
        last_log = _caller_hits.get(f"_blocklog:{origin}", [0.0])[-1]
        if now - last_log >= 60.0:
            log.warning("[blocklist] DROP %s on %s — ua=%r",
                        origin, endpoint, request.headers.get("user-agent", "?")[:80])
            _caller_hits[f"_blocklog:{origin}"] = [now]
        raise HTTPException(status_code=403, detail="Forbidden")
    now = _time.time()
    hits = _caller_hits.setdefault(origin, [])
    # Drop timestamps that fell out of the window
    hits[:] = [t for t in hits if now - t < _CALLER_RATE_WINDOW]
    if len(hits) >= _CALLER_RATE_LIMIT:
        ua = request.headers.get("user-agent", "?")[:80]
        retry = int(_CALLER_RATE_WINDOW - (now - hits[0])) + 1
        log.warning(
            "[rate-limit] BLOCKED %s on %s — %d hits / %ds — ua=%r",
            origin, endpoint, len(hits), int(_CALLER_RATE_WINDOW), ua,
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {_CALLER_RATE_LIMIT} requests per {int(_CALLER_RATE_WINDOW)}s per origin. Retry in {retry}s.",
            headers={"Retry-After": str(retry)},
        )
    hits.append(now)


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
async def refresh_edge_odds(request: Request, force: bool = False):
    """
    Fetch odds for all upcoming edge picks — today + next 3 days.
    Primary: OddsPro movers. Fallback: TAB API (all runners, no auth).
    Rate-limited to once per 2 minutes globally. Pass force=true to bypass.
    """
    _enforce_caller_rate(request, "edge-refresh-odds")
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
async def refresh_edge_results(request: Request):
    """
    Seed today's settled results on demand. Rate-limited globally with random jitter
    (100–130s) so TAB sees one semi-regular user, not a clock-perfect bot.
    Only runs during race hours 12pm–8pm AEST. Returns cached=True outside that window.
    """
    _enforce_caller_rate(request, "edge-refresh-results")
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


_yesterday_response_cache: dict[str, tuple[datetime, dict]] = {}
_YESTERDAY_CACHE_TTL = 1800  # 30 min — past dates are stable
_YESTERDAY_CACHE_TTL_TODAY = 60  # 1 min — today's results land throughout the day
_YESTERDAY_CACHE_VERSION = 3  # bumped to invalidate stale "all no_result" payloads from the 2026-07-10/11 RA outage


def _yesterday_cache_ttl(target_date: str) -> int:
    """Today's results stream in throughout the day; past dates are stable."""
    return _YESTERDAY_CACHE_TTL_TODAY if target_date == _today_aest().isoformat() else _YESTERDAY_CACHE_TTL


@app.get("/api/edge/yesterday")
async def get_edge_yesterday(for_date: Optional[str] = Query(None, alias="date")):
    """Qualifying picks with actual results and SP odds from Racing Australia.
    Accepts ?date=YYYY-MM-DD (defaults to yesterday)."""
    target_date = for_date or (_today_aest() - timedelta(days=1)).isoformat()
    ttl = _yesterday_cache_ttl(target_date)
    # In-memory cache. Past-date results are stable; today refreshes every minute.
    cached = _yesterday_response_cache.get(target_date)
    if cached is not None:
        ts, body = cached
        if (datetime.utcnow() - ts).total_seconds() < ttl:
            return body
    # Fall through to DB-persisted cache. Survives container redeploys so
    # the first user post-deploy doesn't pay the 10s on-demand-seed cost.
    cache_key = f"edge_yesterday:{target_date}"
    try:
        async with get_session() as session:
            row = (await session.execute(
                select(ResponseCacheRow).where(ResponseCacheRow.cache_key == cache_key)
            )).scalar_one_or_none()
        if row and row.cache_version == _YESTERDAY_CACHE_VERSION:
            age = (datetime.utcnow() - row.updated_at).total_seconds()
            if age < ttl:
                body = json.loads(row.payload_json)
                # Re-populate the in-memory tier so subsequent hits don't
                # round-trip to Postgres.
                _yesterday_response_cache[target_date] = (row.updated_at, body)
                return body
    except Exception as e:
        log.debug("[edge-yesterday] DB cache read skipped: %s", e)
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
            empty_body = {"date": target_date, "picks": [], "summary": None}
            _yesterday_response_cache[target_date] = (datetime.utcnow(), empty_body)
            return empty_body

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

    # Seed any missing results — wrapped in a 10s timeout so a slow upstream
    # can't block the whole page load. Anything not seeded in time will just
    # appear as 'no_result' on the response, which the frontend handles.
    all_race_ids = list({p.race_id for p in picks} | {pr.race_id for pr in yst_place_rows})
    try:
        await asyncio.wait_for(_seed_race_results_on_demand(all_race_ids), timeout=10)
    except asyncio.TimeoutError:
        log.warning("[edge/yesterday] seed timeout for %s (%d races) — proceeding with stored data", target_date, len(all_race_ids))
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
        # best_available_odds == 0 is a missing-data flag; treat as None.
        if sp == 0:
            sp = None
        winner = r.get("winner", False)
        position = r.get("position")
        # Race seeded but horse absent → scratched (HistoricalResultRow skips scratched runners)
        scratched = bool(p.race_id in seeded_race_ids and not r)
        # Race not seeded at all → result unavailable (RA had no data; don't show as Unplaced)
        no_result = bool(p.race_id not in seeded_race_ids and not r)
        model_pct = round(p.win_probability * 100, 1)
        payout = round(sp * stake, 2) if winner and sp else 0
        # Profit logic:
        #   • scratched/no_result → 0 (no bet placed)
        #   • winner WITH sp → payout − stake (real profit)
        #   • winner WITHOUT sp → 0 (result confirmed but dividend unknown;
        #     don't treat as a loss — the punter would've collected
        #     something; we just don't know what)
        #   • not winner → −stake
        if scratched or no_result:
            profit = 0
        elif winner:
            profit = round(payout - stake, 2) if sp else 0
        else:
            profit = -stake

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

        # Last-10 form from the pre-race enriched_json snapshot — same source
        # as the live /api/edge picks block.
        try:
            yst_enriched = json.loads(p.enriched_json) if p.enriched_json else {}
        except Exception:
            yst_enriched = {}
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
            "wins_last_10": yst_enriched.get("wins_last_10"),
            "places_last_10": yst_enriched.get("places_last_10"),
            "starts_last_10": yst_enriched.get("starts_last_10"),
            "trifecta": yst_trifecta,
        })

    active = [o for o in output if not o["scratched"] and not o["no_result"]]
    wins = [o for o in active if o["winner"]]
    placed_picks = [o for o in active if o["placed"] and not o["winner"]]
    total_staked = len(active) * stake
    total_returns = sum(o["payout"] for o in active)
    pnl = round(total_returns - total_staked, 2)

    response_body = {
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
    _yesterday_response_cache[target_date] = (datetime.utcnow(), response_body)
    # Persist so the next container redeploy hydrates this date instantly
    # instead of paying the ~10s on-demand-seed cost on the first hit.
    try:
        cache_key = f"edge_yesterday:{target_date}"
        async with get_session() as session:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(ResponseCacheRow).values(
                cache_key=cache_key,
                payload_json=json.dumps(response_body),
                cache_version=_YESTERDAY_CACHE_VERSION,
                updated_at=datetime.utcnow(),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["cache_key"],
                set_=dict(payload_json=stmt.excluded.payload_json,
                          cache_version=stmt.excluded.cache_version,
                          updated_at=stmt.excluded.updated_at),
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as e:
        log.debug("[edge-yesterday] DB cache write skipped: %s", e)
    return response_body


# ─── Bet recommender (paper-trading trifecta ledger) ──────────────────────
async def _generate_bets_for_race(race_id: str, *, regenerate: bool = False) -> int:
    """Create bet recommendations for a single race. No-op if rows already
    exist unless regenerate=True. Returns rows inserted.

    Prize-money gate: only generates bets for races with prize money
    ≥ $30k. Replaces the earlier metro-venue-only filter. The 30d
    backtest showed country races hit 40% (vs 44% metro) and were
    profitable, so a strict venue gate was leaving money on the table
    AND emptying The Lab on country-only days. Prize money better
    tracks dividend pool size: $30k+ races consistently clear the
    box-trifecta break-even on the available strategies."""
    PRIZE_MONEY_FLOOR = 30_000
    _date_str, venue_code, _race_num = _parse_race_id(race_id)
    async with get_session() as session:
        # Find existing strategy_labels for this race — additive insert by
        # default so newly-added strategies (e.g. wide_top5) get backfilled
        # on already-generated races without touching the existing rows.
        existing_labels: set[str] = set(
            (await session.execute(
                select(BetRecommendationRow.strategy_label)
                .where(BetRecommendationRow.race_id == race_id)
            )).scalars().all()
        )
        # Pull runners from the mutable table (pre-race) — falls back to
        # history if mutable was already cleared.
        rows = (await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id == race_id)
        )).scalars().all()
        if not rows:
            rows = (await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.race_id == race_id)
            )).scalars().all()
        if not rows:
            return 0
        # Prize-money gate (replaces metro-only filter). Use max across
        # rows — same value should be on every runner row of the race,
        # but max is robust to nulls / partial backfills. Skip races
        # whose prize money signals a low dividend pool.
        prize_money = max((getattr(r, "prize_money", None) or 0) for r in rows)
        if prize_money and prize_money < PRIZE_MONEY_FLOOR:
            return 0

        runners = [{
            "tab_number": getattr(r, "tab_number", None),
            "horse_name": r.horse_name,
            "win_probability": r.win_probability,
            "place_probability": r.place_probability,
            "model_rank": r.model_rank,
            "cancelled": bool(r.cancelled),
        } for r in rows]

        bets = _build_bet_basket(runners)
        if not bets:
            return 0
        if regenerate:
            await session.execute(
                __import__("sqlalchemy").delete(BetRecommendationRow)
                .where(BetRecommendationRow.race_id == race_id)
                .where(BetRecommendationRow.settled.is_(False))
            )
            existing_labels = set()  # cleared above
        # Additive insert: only add strategy_labels that don't already exist.
        # Lets us backfill new strategies (The Sweep on already-spread races)
        # without disturbing rows that may already be settled.
        new_bets = [b for b in bets if b["strategy_label"] not in existing_labels]
        if not new_bets:
            return 0
        for b in new_bets:
            session.add(BetRecommendationRow(
                race_id=race_id,
                strategy_label=b["strategy_label"],
                box_horses_json=json.dumps(b["box_horses"]),
                box_horse_names_json=json.dumps(b["box_horse_names"]),
                num_permutations=b["num_permutations"],
                stake_dollars=b["stake_dollars"],
            ))
        try:
            await session.commit()
        except Exception as e:
            await session.rollback()
            log.debug("[bets] insert raced (probably dup): %s", e)
            return 0
        inserted_count = len(new_bets)
        # Schedule a one-off settlement job for 5min + jitter past jump.
        # Survives across the (idempotent) 30-min bulk settle cron — if the
        # one-off misses (e.g. Railway redeploy drops in-memory jobs), the
        # cron picks up the unsettled rows on its next tick.
        sched_str = next((r.scheduled_time for r in rows
                          if getattr(r, "scheduled_time", None)), None)
        if sched_str and _scheduler is not None:
            try:
                sched_dt = datetime.fromisoformat(str(sched_str).replace("Z", "+00:00"))
                # 5 min cushion + 0-60s jitter so concurrent races don't
                # serialise on the same minute.
                fire_at = sched_dt + timedelta(minutes=5, seconds=random.uniform(0, 60))
                # Only schedule if it's in the future — past races settle on
                # next cron tick anyway.
                if fire_at > datetime.now(timezone.utc):
                    from apscheduler.triggers.date import DateTrigger
                    _scheduler.add_job(
                        _settle_one_race_with_seed,
                        DateTrigger(run_date=fire_at),
                        args=[race_id],
                        id=f"settle-{race_id}",
                        replace_existing=True,
                        misfire_grace_time=600,
                    )
                    log.info("[bets] scheduled per-race settlement for %s at %s",
                             race_id, fire_at.isoformat())
            except Exception as e:
                log.debug("[bets] per-race schedule failed for %s: %s", race_id, e)
        return inserted_count


# Lazy TABClient singleton used for dividend lookups only — the rest of
# the engine talks to RA via the composite client.
_tab_client_for_dividends = None


def _get_tab_client_for_dividends():
    global _tab_client_for_dividends
    if _tab_client_for_dividends is None:
        from horse_engine.clients.tab import TABClient
        _tab_client_for_dividends = TABClient()
    return _tab_client_for_dividends


async def _fetch_race_raw_from_tab(race_id: str) -> Optional[dict]:
    """Fetch raw TAB race detail (post-result) used for trifecta dividend
    extraction. TAB resolves meetings by slug; we map our race_id's venue
    code into the slug format TAB's client expects."""
    date_str, venue_code, race_num = _parse_race_id(race_id)
    if not (date_str and venue_code and race_num):
        return None
    tab = _get_tab_client_for_dividends()
    # TAB slug = venueSlug-YYYYMMDD. Our venue_code is the same slug shape
    # we use elsewhere ("nowra", "tamworth"), so this maps directly.
    slug = f"{venue_code}-{date_str.replace('-', '')}"
    try:
        return await tab.get_race(slug, int(race_num))
    except Exception as e:
        log.debug("[bets] tab.get_race failed for %s: %s", race_id, e)
    return None


def _extract_trifecta_from_tab_response(raw: dict) -> Optional[float]:
    """Best-effort trifecta dividend extraction. TAB's payload shape varies
    between endpoints; check a few likely keys."""
    if not isinstance(raw, dict):
        return None
    # Shape 1: top-level 'dividends' array of {poolName, price, ...}
    for d in raw.get("dividends") or []:
        if not isinstance(d, dict):
            continue
        name = (d.get("poolName") or d.get("name") or "").upper()
        if "TRIFECTA" in name and "FIRST" not in name:  # exclude 'First Four'
            for key in ("price", "amount", "dividend"):
                v = d.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
                if isinstance(v, str):
                    try:
                        return float(v.replace(",", "").replace("$", ""))
                    except ValueError:
                        pass
            # Some shapes nest the value under 'results' / 'finalDividend'
            for nested_key in ("finalDividend", "result", "results"):
                nv = d.get(nested_key)
                if isinstance(nv, (int, float)) and nv > 0:
                    return float(nv)
                if isinstance(nv, str):
                    try:
                        return float(nv.replace(",", "").replace("$", ""))
                    except ValueError:
                        pass
                if isinstance(nv, list) and nv:
                    for item in nv:
                        if isinstance(item, dict):
                            for k in ("price", "dividend", "amount"):
                                v = item.get(k)
                                if isinstance(v, (int, float)) and v > 0:
                                    return float(v)
    # Shape 2: 'results' object with named pools
    for pool_key in ("results", "pools", "exoticPools"):
        pools = raw.get(pool_key)
        if isinstance(pools, dict):
            for k, v in pools.items():
                if "TRIFECTA" in str(k).upper() and "FIRST" not in str(k).upper():
                    if isinstance(v, (int, float)) and v > 0:
                        return float(v)
                    if isinstance(v, dict):
                        for inner_k in ("dividend", "price", "amount"):
                            iv = v.get(inner_k)
                            if isinstance(iv, (int, float)) and iv > 0:
                                return float(iv)
    return None


async def _fetch_trifecta_dividend(race_id: str) -> Optional[float]:
    """Pull the trifecta dividend via TAB's race endpoint. RA's Results.aspx
    does not include exotic dividends so we go to TAB directly."""
    raw = await _fetch_race_raw_from_tab(race_id)
    if raw is None:
        return None
    return _extract_trifecta_from_tab_response(raw)


async def _settle_bets_for_race(race_id: str) -> int:
    """Settle every unsettled BetRecommendationRow for one race. Returns
    the number of rows updated. No-op if the race has no historical
    results or no trifecta dividend yet.

    If the race is >24h past its scheduled time and still has no top-3
    results in HistoricalResultRow (RA never published or meeting was
    abandoned), the open bets are *voided* rather than left pending.
    Same semantics as a scratched runner — bet doesn't count toward P&L
    or hit rate. 24h is well past RA's normal publish lag (minutes to
    hours), so anything still missing after a day isn't coming."""
    async with get_session() as session:
        result_rows = (await session.execute(
            select(HistoricalResultRow.tab_number, HistoricalResultRow.position,
                   HistoricalResultRow.horse_name)
            .where(HistoricalResultRow.race_id == race_id)
            .where(HistoricalResultRow.position.in_([1, 2, 3]))
        )).fetchall()
        if len(result_rows) < 3:
            # Look up scheduled_time — if the race jumped >24h ago and
            # still has <3 results, RA isn't going to publish them. Void
            # the open bets so the Lab ledger doesn't carry phantom
            # 'pending' rows indefinitely.
            sched_row = (await session.execute(
                select(func.max(RunnerPredictionRow.scheduled_time))
                .where(RunnerPredictionRow.race_id == race_id)
            )).scalar()
            from datetime import timezone as _tz
            stale = False
            try:
                if sched_row:
                    sched_dt = datetime.fromisoformat(str(sched_row).replace("Z", "+00:00"))
                    stale = (datetime.utcnow().replace(tzinfo=_tz.utc) - sched_dt).total_seconds() > 24 * 3600
            except (ValueError, TypeError):
                pass
            # Fallback: parse the YYYY-MM-DD prefix off race_id. Older
            # RunnerPredictionRows can have NULL scheduled_time but the
            # race_id always carries the racing date, and "raced two
            # calendar days ago with no results" is conclusive on its
            # own.
            if not stale:
                try:
                    date_prefix = race_id.split("_", 1)[0]
                    race_date = datetime.strptime(date_prefix, "%Y-%m-%d").date()
                    days_old = (_today_aest() - race_date).days
                    stale = days_old >= 2
                except (ValueError, IndexError):
                    pass
            if stale:
                from sqlalchemy import update as sa_update
                upd = (await session.execute(
                    sa_update(BetRecommendationRow)
                    .where(BetRecommendationRow.race_id == race_id)
                    .where(BetRecommendationRow.settled.is_(False))
                    .values(
                        settled=True,
                        voided=True,
                        is_hit=False,
                        settled_at=datetime.utcnow(),
                    )
                ))
                await session.commit()
                voided = upd.rowcount or 0
                if voided:
                    log.info("[bets] voided %d stale-no-result rows for %s", voided, race_id)
                return voided
            return 0

        # Some seed paths inserted duplicate rows for the same position
        # (e.g. a re-seed when name normalisation didn't match). Take the
        # first non-null tab per position so actual_top3 is exactly 3
        # tabs in finishing order — duplicates were causing false hits
        # because _bet_is_hit checks actual[:3] (which became [8,8,7]
        # for a duplicated [8,8,7,7,9,9] — a 2-horse set, not 3).
        top_by_pos: dict[int, int] = {}
        needs_lookup = False
        for tab, pos, name in result_rows:
            if pos in top_by_pos:
                continue  # keep first (dedup)
            if tab is not None:
                top_by_pos[pos] = tab
            else:
                needs_lookup = True
                top_by_pos[pos] = None  # placeholder
        if needs_lookup:
            tab_lookup: dict[str, int] = {}
            for src in (RunnerPredictionRow, RunnerPredictionHistoryRow):
                rows = (await session.execute(
                    select(src.horse_name, src.tab_number)
                    .where(src.race_id == race_id)
                    .where(src.tab_number.isnot(None))
                )).fetchall()
                for name, tab in rows:
                    tab_lookup[_normalize_horse(name)] = tab
            for tab, pos, name in result_rows:
                if top_by_pos.get(pos) is None and name:
                    t = tab_lookup.get(_normalize_horse(name))
                    if t is not None:
                        top_by_pos[pos] = t
        if any(top_by_pos.get(p) is None for p in (1, 2, 3)):
            # Results are present but we can't map top-3 finishers to
            # tab_numbers (RA's Results.aspx name doesn't normalize-match
            # anything in our RunnerPredictionRow — happens on ~10% of
            # races per day due to name spelling drift). If the race
            # jumped >24h ago, void the open bets rather than carrying
            # them as pending forever. Real TAB would refund an
            # unresolvable bet — same treatment.
            from sqlalchemy import update as sa_update
            sched_row = (await session.execute(
                select(func.max(RunnerPredictionRow.scheduled_time))
                .where(RunnerPredictionRow.race_id == race_id)
            )).scalar()
            from datetime import timezone as _tz
            stale = False
            try:
                if sched_row:
                    sched_dt = datetime.fromisoformat(str(sched_row).replace("Z", "+00:00"))
                    stale = (datetime.utcnow().replace(tzinfo=_tz.utc) - sched_dt).total_seconds() > 24 * 3600
            except (ValueError, TypeError):
                pass
            if not stale:
                try:
                    date_prefix = race_id.split("_", 1)[0]
                    race_date = datetime.strptime(date_prefix, "%Y-%m-%d").date()
                    stale = (_today_aest() - race_date).days >= 2
                except (ValueError, IndexError):
                    pass
            if stale:
                upd = (await session.execute(
                    sa_update(BetRecommendationRow)
                    .where(BetRecommendationRow.race_id == race_id)
                    .where(BetRecommendationRow.settled.is_(False))
                    .values(
                        settled=True,
                        voided=True,
                        is_hit=False,
                        settled_at=datetime.utcnow(),
                    )
                ))
                await session.commit()
                voided = upd.rowcount or 0
                if voided:
                    log.info("[bets] voided %d stale-name-drift rows for %s (top-3 tabs unresolvable)", voided, race_id)
                return voided
            return 0
        actual_top3 = [top_by_pos[1], top_by_pos[2], top_by_pos[3]]

        unsettled = (await session.execute(
            select(BetRecommendationRow)
            .where(BetRecommendationRow.race_id == race_id)
            .where(BetRecommendationRow.settled.is_(False))
        )).scalars().all()
        if not unsettled:
            return 0

    # Dividend may be None — TAB API no longer resolves and RA Results.aspx
    # doesn't carry exotic dividends. Settle anyway with hit/miss + winning
    # trifecta so the ledger shows actionable info; payout / P&L populate
    # later if/when a dividend source comes online.
    dividend = await _fetch_trifecta_dividend(race_id)
    dividend_estimated = False
    # Fallback: when TAB doesn't return a dividend, derive a Harville
    # estimate from the actual finishers' model win-probabilities. Same
    # math the Lab UI uses for the on-card 'estimated payout' badge —
    # accurate within ~10-20% of the printed TAB number on average.
    if dividend is None:
        async with get_session() as session:
            prob_rows = (await session.execute(
                select(RunnerPredictionHistoryRow.tab_number,
                       RunnerPredictionHistoryRow.win_probability)
                .where(RunnerPredictionHistoryRow.race_id == race_id)
                .where(RunnerPredictionHistoryRow.tab_number.in_(actual_top3))
                .where(RunnerPredictionHistoryRow.source == "live")
            )).fetchall()
        if not prob_rows:
            async with get_session() as session:
                prob_rows = (await session.execute(
                    select(RunnerPredictionRow.tab_number,
                           RunnerPredictionRow.win_probability)
                    .where(RunnerPredictionRow.race_id == race_id)
                    .where(RunnerPredictionRow.tab_number.in_(actual_top3))
                )).fetchall()
        # Pick the highest-confidence row per tab (history can have multiple)
        prob_by_tab: dict[int, float] = {}
        for tab, p in prob_rows:
            if p is None or p <= 0:
                continue
            if tab not in prob_by_tab or p > prob_by_tab[tab]:
                prob_by_tab[tab] = float(p)
        if all(t in prob_by_tab for t in actual_top3):
            p1, p2, p3 = (prob_by_tab[t] for t in actual_top3)
            ordered_prob = _harville_top3_prob(p1, p2, p3)
            if ordered_prob > 0:
                dividend = round((1.0 / ordered_prob) * (1 - _TAB_TRIFECTA_TAKEOUT), 2)
                dividend_estimated = True

    # Look up scratched-runner tab numbers for this race so we can void
    # any box containing one. A bet whose box includes a scratched horse
    # couldn't actually have hit — real TAB refunds the proportional
    # share; we mark the row voided and exclude it from hit-rate/ROI.
    async with get_session() as session:
        scratched_rows = (await session.execute(
            select(RunnerPredictionRow.tab_number)
            .where(RunnerPredictionRow.race_id == race_id)
            .where(RunnerPredictionRow.cancelled.is_(True))
            .where(RunnerPredictionRow.tab_number.isnot(None))
        )).fetchall()
        scratched_tabs: set[int] = {tab for (tab,) in scratched_rows}

    async with get_session() as session:
        updated = 0
        voided_count = 0
        for b in unsettled:
            row = await session.get(BetRecommendationRow, b.id)
            if row is None or row.settled:
                continue
            box = json.loads(row.box_horses_json or "[]")
            # When a box contains a scratched horse, real-world TAB
            # collapses the bet onto the remaining horses (not a refund
            # unless < 3 remain). Mirror that here: compute effective_box
            # = box minus scratched, then hit-check + payout on that.
            box_set = set(box)
            scratched_in_box = box_set & scratched_tabs
            effective_box = sorted(box_set - scratched_tabs)
            row.actual_top3_json = json.dumps(actual_top3)
            row.trifecta_dividend = dividend
            row.dividend_estimated = dividend_estimated
            if len(effective_box) < 3:
                # Box dead — full refund. Real bettor gets stake back.
                row.is_hit = False
                row.voided = True
                row.payout_dollars = 0.0
                row.pnl_dollars = 0.0
                voided_count += 1
            else:
                row.voided = False
                hit = _bet_is_hit(effective_box, actual_top3)
                row.is_hit = hit
                if dividend is not None:
                    # Payout is now based on the effective permutation
                    # count of the SHRUNK box, not the original. This
                    # approximates TAB's deductions for scratchings.
                    eff_perms = (len(effective_box) *
                                 (len(effective_box) - 1) *
                                 (len(effective_box) - 2))
                    payout, pnl = _bet_compute_payout(row.stake_dollars, eff_perms, dividend, hit)
                    row.payout_dollars = payout
                    row.pnl_dollars = pnl
            row.settled = True
            row.settled_at = datetime.utcnow()
            updated += 1
        await session.commit()
    if updated:
        log.info("[bets] settled %d bets for %s (div=%s, voided=%d, scratched_seen=%d)",
                 updated, race_id, f"${dividend:.2f}" if dividend else "unknown",
                 voided_count, len(scratched_tabs))
    return updated


async def _settle_one_race_with_seed(race_id: str) -> int:
    """Per-race settlement entry point — used by the one-off APScheduler
    job fired 5min + jitter after a race's scheduled jump. Seeds results
    for the date (idempotent, cheap on repeated calls) then settles the
    bets for the race."""
    date_str = race_id.split("_", 1)[0]
    try:
        await _seed_results_for_date(date_str)
    except Exception as e:
        log.debug("[bets] per-race seed failed for %s: %s", date_str, e)
    try:
        return await _settle_bets_for_race(race_id)
    except Exception as e:
        log.debug("[bets] per-race settle failed for %s: %s", race_id, e)
        return 0


async def _scheduled_settle_bets():
    """Sweep all unsettled bets whose races have historical results and a
    trifecta dividend. Pre-seeds results per pending date so settlement
    doesn't have to wait for the separate seed-results cron to fire."""
    from datetime import timezone as _tz
    now_utc = datetime.utcnow().replace(tzinfo=_tz.utc)
    try:
        async with get_session() as session:
            unsettled_rows = (await session.execute(
                select(BetRecommendationRow.race_id).distinct()
                .where(BetRecommendationRow.settled.is_(False))
            )).scalars().all()
        if not unsettled_rows:
            return

        # Only attempt to settle races that have actually jumped. Look up
        # scheduled_time so we don't waste RA calls seeding future races.
        async with get_session() as session:
            sched_map = dict((await session.execute(
                select(RunnerPredictionRow.race_id, func.max(RunnerPredictionRow.scheduled_time))
                .where(RunnerPredictionRow.race_id.in_(unsettled_rows))
                .group_by(RunnerPredictionRow.race_id)
            )).fetchall())

        def _has_jumped(race_id: str) -> bool:
            st = sched_map.get(race_id)
            if not st:
                return True  # no scheduled_time → assume past
            try:
                return datetime.fromisoformat(str(st).replace("Z", "+00:00")) <= now_utc
            except (ValueError, TypeError):
                return True

        pending_race_ids = [rid for rid in unsettled_rows if _has_jumped(rid)]
        if not pending_race_ids:
            return

        # Pre-seed: run _seed_results_for_date once per pending date.
        # _seed_results_for_date skips already-seeded races, so this is
        # bounded and idempotent. Without this, settlement would have to
        # wait for the separate seed-results cron to fire.
        dates = sorted({rid.split("_", 1)[0] for rid in pending_race_ids if "_" in rid})
        for d in dates:
            try:
                seeded = await _seed_results_for_date(d)
                if seeded:
                    log.info("[bets] pre-settlement seed for %s: %d", d, seeded)
            except Exception as e:
                log.debug("[bets] seed failed for %s: %s", d, e)

        total = 0
        for rid in pending_race_ids:
            try:
                total += await _settle_bets_for_race(rid)
            except Exception as e:
                log.debug("[bets] settle failed for %s: %s", rid, e)
        if total:
            log.info("[bets] settlement sweep: %d rows", total)
    except Exception as e:
        log.exception("[bets] scheduled settlement failed: %s", e)


async def _scheduled_generate_bets():
    """Hourly during racing hours — find upcoming races without bets and
    generate them. Runs after enrichment has populated runner predictions."""
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    horizon = now_utc + timedelta(hours=8)
    today = _today_aest().isoformat()
    try:
        async with get_session() as session:
            # Distinct race_ids for today + tomorrow that have prediction rows
            # but no bet rows yet.
            candidate_ids = (await session.execute(
                select(RunnerPredictionRow.race_id).distinct()
                .where(RunnerPredictionRow.race_id.like(f"{today}_%")
                       | RunnerPredictionRow.race_id.like(
                           f"{(_today_aest() + timedelta(days=1)).isoformat()}_%"))
            )).scalars().all()
            # Process every candidate — _generate_bets_for_race is now
            # additive (only inserts missing strategy_labels), so this also
            # backfills new strategies onto already-generated races.
            todo = list(candidate_ids)
        log.info("[bets] generating for %d races", len(todo))
        total = 0
        for rid in todo:
            try:
                total += await _generate_bets_for_race(rid)
            except Exception as e:
                log.debug("[bets] generate failed for %s: %s", rid, e)
        if total:
            log.info("[bets] inserted %d rows across %d races", total, len(todo))
    except Exception as e:
        log.exception("[bets] scheduled generation failed: %s", e)


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


# ── Funk Me Up — daily playbook combining model picks with Sportsbet features ──
# Five plays, each targeting a different bookmaker feature class. Designed
# around a $10 base outlay per play (~$40 cash exposure + bonus credit).
# See horse_engine/bookmakers/sportsbet_features.py for the static rule set.

_FUNK_BASE_STAKE = 10.0

# Metro-only venues for The Spine and The Wave (deeper markets, better
# liquidity, Top 4 reliably available, Quaddie pools larger).
_FUNK_METRO_VENUES = {
    "randwick", "royal-randwick", "rosehill", "rosehill-gardens",
    "canterbury-park", "warwick-farm",
    "caulfield", "caulfield-heath", "flemington", "moonee-valley",
    "sandown", "sandown-lakeside", "sandown-hillside", "the-valley",
    "doomben", "eagle-farm",
    "morphettville", "morphettville-parks",
    "ascot", "belmont",
}


def _market_implied_top4(odds: Optional[float]) -> Optional[float]:
    """Best-effort: invert TAB Top 4 odds to a market probability.
    Strip overround using Sportsbet's typical Top 4 over-round %."""
    if not odds or odds <= 1.0:
        return None
    return (1.0 / odds) * (1.0 - _sb.TOP4_OVERROUND_PCT)


# Bookmaker over-round on TAB Place markets (used to synthesise indicative
# place odds from model place_probability when actual market data isn't
# stored historically).
_PLACE_OVERROUND_PCT = 0.12

def _synth_place_odds(place_prob: float) -> float:
    """Indicative Place odds from a model place probability."""
    if place_prob <= 0:
        return 0.0
    implied = max(place_prob * (1 - _PLACE_OVERROUND_PCT), 0.05)
    return round(1.0 / implied, 2)


def _is_decisive_race(pick: dict, min_gap_pts: float = 5.0) -> bool:
    """Skip 'toss-up' races where rank-1 and rank-2 model_pct are within
    min_gap_pts of each other. Backtest (60d, 1344 races): rank-1 wins
    13% on ≤2pt gaps vs 24% on 10pt+ gaps. The 5pt threshold removes
    the worst of the trap zone."""
    field = pick.get("field") or []
    active = [f for f in field if not f.get("scratched")]
    if len(active) < 2:
        return False
    sorted_active = sorted(active, key=lambda f: -(f.get("win_pct") or 0))
    rank1 = sorted_active[0].get("win_pct") or 0
    rank2 = sorted_active[1].get("win_pct") or 0
    return (rank1 - rank2) >= min_gap_pts


def _build_spine(edge_picks: list[dict]) -> Optional[dict]:
    """The Spine: 4-leg cross-race Top 4 multi, boosted. Prefers metro
    venues; falls back to any race with field ≥8 if metro is empty
    (Sunday country cards, public-holiday wash-outs, etc.)."""
    def collect(metro_only: bool) -> dict:
        out: dict[str, dict] = {}
        for p in edge_picks:
            venue = (p.get("venue") or "").lower()
            if metro_only and venue not in _FUNK_METRO_VENUES:
                continue
            rid = p.get("race_id")
            if not rid or rid in out:
                continue
            field = p.get("field") or []
            active = [f for f in field if not f.get("scratched")]
            if len(active) < _sb.TOP4_MIN_FIELD_SIZE:
                continue
            win_probs = []
            target_idx = None
            for i, f in enumerate(active):
                wp = (f.get("win_pct") or 0) / 100.0
                win_probs.append(wp)
                if f.get("rank") == 1:
                    target_idx = i
            if target_idx is None or win_probs[target_idx] <= 0:
                continue
            others = win_probs[:target_idx] + win_probs[target_idx+1:]
            top4_prob = _harville_horse_top_n(win_probs[target_idx], others, 4)
            # Looser threshold on fallback so we still ship 4 legs
            min_top4 = 0.65 if metro_only else 0.60
            if top4_prob < min_top4:
                continue
            sb_top4_odds = round(1.0 / max(top4_prob - _sb.TOP4_OVERROUND_PCT * top4_prob, 0.05), 2)
            out[rid] = {
                "race_id": rid,
                "venue": p.get("venue"),
                "race_number": p.get("race_number"),
                "horse_name": active[target_idx]["horse_name"],
                "tab_number": active[target_idx].get("tab_number"),
                "scheduled_time": p.get("scheduled_time"),
                "model_top4_prob": round(top4_prob, 4),
                "sb_top4_odds_est": sb_top4_odds,
            }
        return out
    by_race = collect(metro_only=True)
    if len(by_race) < 4:
        by_race = collect(metro_only=False)
    if len(by_race) < 4:
        return None
    candidates = sorted(by_race.values(), key=lambda r: -r["model_top4_prob"])
    legs = candidates[:4]
    # Spread the 4 picks across jump time when possible
    legs = sorted(legs, key=lambda l: l.get("scheduled_time") or "")
    raw_multi = 1.0
    per_leg_strike = 1.0
    for l in legs:
        raw_multi *= l["sb_top4_odds_est"]
        per_leg_strike *= l["model_top4_prob"]
    raw_multi = round(raw_multi, 2)
    boosted = _sb.boosted_multi_odds(raw_multi, 4)
    return {
        "kind": "spine",
        "title": "The Spine",
        "subtitle": "4-leg Top 4 multi · cross-race · +20% Sportsbet boost",
        "confidence": "B",
        "legs": legs,
        "raw_multi_odds": raw_multi,
        "boosted_multi_odds": boosted,
        "model_hit_probability": round(per_leg_strike, 4),
        "stake_dollars": _FUNK_BASE_STAKE,
        "potential_return_dollars": round(boosted * _FUNK_BASE_STAKE, 2),
        "expected_value_dollars": round(per_leg_strike * boosted * _FUNK_BASE_STAKE - _FUNK_BASE_STAKE, 2),
    }


def _build_lock(edge_picks: list[dict]) -> Optional[dict]:
    """The Lock: single win bet on the day's strongest Premium pick.
    Defined by Edge page's Premium criteria (≥29.5% + ≥$3 + >5pt edge)
    plus an additional 'highest edge_pct' tiebreak."""
    candidates = []
    for p in edge_picks:
        odds = p.get("best_available_odds")
        edge = p.get("edge_pct")
        model_pct = p.get("model_pct") or 0
        if not odds or odds < 3.0:
            continue
        if model_pct < 29.5:
            continue
        if edge is None or edge <= 5:
            continue
        if not _is_decisive_race(p, 5.0):
            continue
        candidates.append(p)
    if not candidates:
        return None
    pick = max(candidates, key=lambda p: p.get("edge_pct") or 0)
    return {
        "kind": "lock",
        "title": "The Lock",
        "subtitle": "Single win · Premium-tier pick · best edge of the day",
        "confidence": "B",
        "race_id": pick["race_id"],
        "venue": pick.get("venue"),
        "race_number": pick.get("race_number"),
        "horse_name": pick["horse_name"],
        "tab_number": pick.get("tab_number"),
        "scheduled_time": pick.get("scheduled_time"),
        "model_pct": pick.get("model_pct"),
        "market_implied_pct": pick.get("market_implied_pct"),
        "edge_pct": pick.get("edge_pct"),
        "best_available_odds": pick.get("best_available_odds"),
        "model_hit_probability": round((pick.get("model_pct") or 0) / 100, 4),
        "stake_dollars": _FUNK_BASE_STAKE,
        "potential_return_dollars": round((pick.get("best_available_odds") or 0) * _FUNK_BASE_STAKE, 2),
        "expected_value_dollars": round(
            ((pick.get("model_pct") or 0) / 100) * (pick.get("best_available_odds") or 0) * _FUNK_BASE_STAKE - _FUNK_BASE_STAKE, 2
        ),
    }


def _build_double(edge_picks: list[dict]) -> Optional[dict]:
    """The Double: 2-leg place multi using the model's strongest place
    picks from two different races. Each leg requires ≥60% model place
    probability AND ≥5pt rank-gap to rank-2. 60d backtest: 27.6% hit,
    +71% ROI — the only multi format that survives the bookmaker margin."""
    candidates = []
    seen_races = set()
    for p in edge_picks:
        rid = p.get("race_id")
        if not rid or rid in seen_races:
            continue
        if not _is_decisive_race(p, 5.0):
            continue
        place_prob = (p.get("place_probability") or 0) / 100.0
        if place_prob < 0.60:
            continue
        seen_races.add(rid)
        place_odds = _synth_place_odds(place_prob)
        candidates.append({
            "race_id": rid,
            "horse_name": p["horse_name"],
            "venue": p.get("venue"),
            "race_number": p.get("race_number"),
            "scheduled_time": p.get("scheduled_time"),
            "tab_number": p.get("tab_number"),
            "place_prob": place_prob,
            "place_odds_est": place_odds,
        })
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda c: -c["place_prob"])
    legs = candidates[:2]
    legs.sort(key=lambda l: l.get("scheduled_time") or "")
    raw_multi = round(legs[0]["place_odds_est"] * legs[1]["place_odds_est"], 2)
    combined_p = legs[0]["place_prob"] * legs[1]["place_prob"]
    return {
        "kind": "double",
        "title": "The Double",
        "subtitle": "2-leg place multi · bread-and-butter low variance",
        "confidence": "A",
        "legs": legs,
        "raw_multi_odds": raw_multi,
        "model_hit_probability": round(combined_p, 4),
        "stake_dollars": _FUNK_BASE_STAKE,
        "potential_return_dollars": round(raw_multi * _FUNK_BASE_STAKE, 2),
        "expected_value_dollars": round(combined_p * raw_multi * _FUNK_BASE_STAKE - _FUNK_BASE_STAKE, 2),
    }


async def _build_early_quaddie(target_date: str) -> Optional[dict]:
    """The Early Quaddie: 4-leg WIN multi covering the early quaddie legs
    at one meeting.

    Queries the prediction tables directly (not /api/edge) so quaddies
    can form even when a single leg has a soft rank-1 pick below the
    Edge confidence threshold. Tries R1-R4 first (the canonical TAB
    early-quaddie nomination at most metro Saturday cards), then falls
    back to R2-R5 (used at some country/twilight cards).

    Across qualifying meetings, picks the one with the highest combined
    model probability. Confidence is 'C' — 4-leg WIN multis are high
    variance even with strong individual picks.
    """
    today_iso = _today_aest().isoformat()
    is_past = target_date < today_iso

    async with get_session() as session:
        # Pull rank-1 AND rank-2 picks so we can offer both the straight
        # quaddie (1 horse per leg) and the boxed variant (rank-1 +
        # rank-2 per leg = 16 perms, hits if either picked horse wins
        # each leg).
        if is_past:
            rows = (await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.race_id.like(f"{target_date}_%"))
                .where(RunnerPredictionHistoryRow.model_rank.in_([1, 2]))
                .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                       | RunnerPredictionHistoryRow.cancelled.is_(None))
                .where(RunnerPredictionHistoryRow.source == "live")
            )).scalars().all()
        else:
            rows = (await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id.like(f"{target_date}_%"))
                .where(RunnerPredictionRow.model_rank.in_([1, 2]))
                .where(RunnerPredictionRow.cancelled.is_(False)
                       | RunnerPredictionRow.cancelled.is_(None))
            )).scalars().all()

    if not rows:
        return None

    # Group by venue → race_number → {1: rank1_row, 2: rank2_row}
    by_venue: dict[str, dict[int, dict[int, object]]] = {}
    for r in rows:
        try:
            _, vc, rn = _parse_race_id(r.race_id)
        except (ValueError, AttributeError):
            continue
        if vc is None or rn is None or r.model_rank not in (1, 2):
            continue
        by_venue.setdefault(vc, {}).setdefault(rn, {})[r.model_rank] = r

    def _try_legs(leg_range: tuple[int, ...]) -> list[dict]:
        out = []
        for vc, races in by_venue.items():
            # Each race needs a rank-1 row. rank-2 is optional per-leg
            # (a leg without rank-2 in the box still works for the
            # straight version; the boxed combined_prob just uses
            # rank-1 only for that leg).
            if not all(rn in races and 1 in races[rn] for rn in leg_range):
                continue
            r1_legs = [races[rn][1] for rn in leg_range]
            r2_legs = [races[rn].get(2) for rn in leg_range]
            # Straight combined probability — product of rank-1 win probs.
            combined_straight = 1.0
            for leg in r1_legs:
                combined_straight *= (leg.win_probability or 0)
            # Boxed combined probability — product of (rank-1 + rank-2)
            # per leg. Treats win-prob as disjoint per leg (only one
            # horse can win each race), so the per-leg "any of our
            # picks wins" probability is the sum.
            combined_boxed = 1.0
            box_complete = True
            for r1, r2 in zip(r1_legs, r2_legs):
                p1 = r1.win_probability or 0
                p2 = (r2.win_probability or 0) if r2 is not None else 0
                if r2 is None:
                    box_complete = False
                combined_boxed *= (p1 + p2)
            # Straight multi odds = product of rank-1 best available.
            odds_product = 1.0
            any_missing_odds = False
            for leg in r1_legs:
                o = leg.best_available_odds
                if not o or o < 1.0:
                    any_missing_odds = True
                    break
                odds_product *= o
            out.append({
                "venue": vc,
                "leg_range": leg_range,
                "legs": r1_legs,
                "second_legs": r2_legs,
                "combined_prob": combined_straight,
                "combined_prob_boxed": combined_boxed if box_complete else None,
                "box_complete": box_complete,
                "odds_product": None if any_missing_odds else round(odds_product, 2),
            })
        return out

    # Prefer R1-R4 (canonical TAB early quaddie); fall back to R2-R5.
    candidates = _try_legs((1, 2, 3, 4))
    if not candidates:
        candidates = _try_legs((2, 3, 4, 5))
    if not candidates:
        return None

    # Pick the candidate with highest BOXED probability if it's available
    # (more achievable hit rate is the headline number), otherwise fall
    # back to highest straight combined probability.
    box_candidates = [c for c in candidates if c["combined_prob_boxed"] is not None]
    pool = box_candidates if box_candidates else candidates
    sort_key = "combined_prob_boxed" if box_candidates else "combined_prob"
    best = max(pool, key=lambda c: c[sort_key])
    r1_legs = best["legs"]
    r2_legs = best["second_legs"]
    leg_range = best["leg_range"]
    venue_code = best["venue"]
    venue_label = (getattr(r1_legs[0], "venue", None) or venue_code).replace("-", " ").title()
    raw_multi = best["odds_product"]
    combined_p = best["combined_prob"]
    combined_p_boxed = best["combined_prob_boxed"]
    range_label = f"R{leg_range[0]}-R{leg_range[-1]}"

    leg_dicts = []
    for r1, r2 in zip(r1_legs, r2_legs):
        leg = {
            "race_id": r1.race_id,
            "venue": getattr(r1, "venue", None) or venue_code,
            "race_number": getattr(r1, "race_number", None) or _parse_race_id(r1.race_id)[2],
            "horse_name": r1.horse_name,
            "tab_number": getattr(r1, "tab_number", None),
            "scheduled_time": getattr(r1, "scheduled_time", None),
            "model_pct": round((r1.win_probability or 0) * 100, 1),
            "best_available_odds": r1.best_available_odds,
        }
        if r2 is not None:
            leg["second_horse_name"] = r2.horse_name
            leg["second_tab_number"] = getattr(r2, "tab_number", None)
            leg["second_model_pct"] = round((r2.win_probability or 0) * 100, 1)
            leg["second_best_available_odds"] = r2.best_available_odds
        leg_dicts.append(leg)

    # Boxed quaddie = 2×2×2×2 = 16 perms. Standard $2/perm stake → $32.
    box_perms = 16 if best["box_complete"] else None
    boxed_stake = (box_perms * _FUNK_BASE_STAKE) if box_perms else None

    return {
        "kind": "quaddie",
        "title": "Early Quaddie",
        "subtitle": f"{venue_label} {range_label} · 4-leg win multi",
        "confidence": "C",
        "venue": venue_code,
        "legs": leg_dicts,
        # Straight (1 horse per leg, 1 perm)
        "raw_multi_odds": raw_multi,
        "model_hit_probability": round(combined_p, 6),
        "stake_dollars": _FUNK_BASE_STAKE,
        "potential_return_dollars": round(raw_multi * _FUNK_BASE_STAKE, 2) if raw_multi else None,
        "expected_value_dollars": round(
            combined_p * raw_multi * _FUNK_BASE_STAKE - _FUNK_BASE_STAKE, 2
        ) if raw_multi else None,
        # Boxed (2 horses per leg, 16 perms). Hit prob roughly 4× the
        # straight version on average — better hit rate for a higher
        # stake; the same fair return per perm.
        "boxed_available": best["box_complete"],
        "boxed_perms": box_perms,
        "boxed_stake_dollars": boxed_stake,
        "boxed_hit_probability": round(combined_p_boxed, 6) if combined_p_boxed else None,
    }


def _build_combo_doubles(edge_picks: list[dict]) -> Optional[dict]:
    """The Combo Doubles: 4 selections across different races → 6 doubles.

    Sportsbet-style "Combo Multi Doubles" bet — pick 4 horses across
    4 different races, get every 2-leg pair as a separate double.
    Requires only 2 of 4 to win for at least one double to pay; hits
    scale with how many legs land.

    Selection: top-4 rank-1 picks from /api/edge (already filtered to
    ≥29.5% confidence), one per race, sorted by model_pct descending
    then split-by-race. Requires all 4 to have live odds so we can
    compute fair pair odds and EV honestly. Sharp filter passthrough:
    if any leg has is_sharp=false, the play still forms — Sharp is a
    single-race notion, doesn't apply directly to a 4-race parlay.

    Cost/benefit: total stake = 6 × unit stake ($60 at $10 unit).
    Break-even on 2 wins requires combined winner odds > 6.0. On 3+
    wins the format is very forgiving (multiple pairs pay).
    """
    UNIT_STAKE = _FUNK_BASE_STAKE  # per-double stake
    # One rank-1 pick per race (edge_picks may be ordered arbitrarily).
    by_race: dict[str, dict] = {}
    for p in edge_picks:
        rid = p.get("race_id")
        if not rid or rid in by_race:
            continue
        if (p.get("model_pct") or 0) < 29.5:
            continue
        # Need live odds to price a double honestly.
        odds = p.get("best_available_odds") or 0
        if not odds or odds < 1.5:
            continue
        by_race[rid] = p

    # Rank picks by model confidence; take top 4.
    candidates = sorted(by_race.values(), key=lambda p: -(p.get("model_pct") or 0))
    if len(candidates) < 4:
        return None
    top4 = candidates[:4]
    # Sort selected legs by jump time so display order matches placing order.
    top4.sort(key=lambda p: p.get("scheduled_time") or "")

    # Per-leg probability (from model_pct) + fair odds.
    probs = [(p.get("model_pct") or 0) / 100.0 for p in top4]
    odds = [p.get("best_available_odds") or 0.0 for p in top4]

    # 6 doubles = every pair.
    from itertools import combinations
    pairs = []
    for i, j in combinations(range(4), 2):
        pair_p = probs[i] * probs[j]        # both legs win
        pair_return = odds[i] * odds[j] * UNIT_STAKE  # gross return if the pair hits
        pair_ev = pair_p * pair_return - UNIT_STAKE
        pairs.append({
            "leg_indices": [i, j],
            "leg1_horse": top4[i]["horse_name"],
            "leg2_horse": top4[j]["horse_name"],
            "pair_odds": round(odds[i] * odds[j], 2),
            "pair_prob": round(pair_p, 4),
            "gross_return": round(pair_return, 2),
            "ev_dollars": round(pair_ev, 2),
        })

    # Portfolio-level expected value: sum of per-pair EVs.
    total_stake = round(UNIT_STAKE * len(pairs), 2)
    portfolio_ev = round(sum(pr["ev_dollars"] for pr in pairs), 2)

    # Scenario probabilities via inclusion-exclusion on the 4 leg outcomes.
    # P(exactly k wins) — enumerate the 2^4 outcome space.
    scenario_counts = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for mask in range(16):
        wins = bin(mask).count("1")
        p_out = 1.0
        for k in range(4):
            hit = bool(mask & (1 << k))
            p_out *= probs[k] if hit else (1.0 - probs[k])
        scenario_counts[wins] += p_out
    # Cumulative "at least N wins" probabilities.
    p_at_least_1 = 1.0 - scenario_counts[0]
    p_at_least_2 = p_at_least_1 - scenario_counts[1]
    p_at_least_3 = p_at_least_2 - scenario_counts[2]
    p_all_4 = scenario_counts[4]

    legs = [{
        "race_id": p["race_id"],
        "venue": p.get("venue"),
        "race_number": p.get("race_number"),
        "horse_name": p["horse_name"],
        "tab_number": p.get("tab_number"),
        "scheduled_time": p.get("scheduled_time"),
        "model_pct": p.get("model_pct"),
        "best_available_odds": p.get("best_available_odds"),
    } for p in top4]

    return {
        "kind": "combo_doubles",
        "title": "Combo Doubles",
        "subtitle": f"4 picks × 6 doubles · {int(p_at_least_2 * 100)}% chance of a paying double",
        "confidence": "B",  # more forgiving than a 4-leg straight quaddie
        "unit_stake_dollars": UNIT_STAKE,
        "num_pairs": len(pairs),
        "legs": legs,
        "pairs": pairs,
        # Straight-across totals (all 6 doubles placed)
        "stake_dollars": total_stake,
        "expected_value_dollars": portfolio_ev,
        # Scenario probabilities
        "prob_at_least_1_win": round(p_at_least_1, 4),
        "prob_at_least_2_wins": round(p_at_least_2, 4),
        "prob_at_least_3_wins": round(p_at_least_3, 4),
        "prob_all_4_win": round(p_all_4, 4),
        # "model_hit_probability" for the hero-of-day ranker — use P(≥2 wins)
        # since that's when Combo Doubles actually cashes.
        "model_hit_probability": round(p_at_least_2, 4),
        # Sum of "if this pair hits" gross returns weighted by probability
        # — approximates expected gross return before subtracting stake
        "potential_return_dollars": round(
            sum(pr["pair_prob"] * pr["gross_return"] for pr in pairs), 2
        ),
    }


def _build_banker(edge_picks: list[dict]) -> Optional[dict]:
    """The Banker: single place bet on the day's strongest place pick.
    Place_prob ≥ 70% AND ≥5pt rank-gap. Expected hit ~65%, modest payout."""
    candidates = []
    for p in edge_picks:
        if not _is_decisive_race(p, 5.0):
            continue
        place_prob = (p.get("place_probability") or 0) / 100.0
        if place_prob < 0.70:
            continue
        place_odds = _synth_place_odds(place_prob)
        candidates.append({
            "race_id": p["race_id"],
            "horse_name": p["horse_name"],
            "venue": p.get("venue"),
            "race_number": p.get("race_number"),
            "scheduled_time": p.get("scheduled_time"),
            "tab_number": p.get("tab_number"),
            "place_prob": place_prob,
            "place_odds_est": place_odds,
            "model_pct": p.get("model_pct"),
        })
    if not candidates:
        return None
    pick = max(candidates, key=lambda c: c["place_prob"])
    return {
        "kind": "banker",
        "title": "The Banker",
        "subtitle": f"Place bet · model has {round(pick['place_prob']*100,0)}% top-3 confidence",
        "confidence": "A",
        "race_id": pick["race_id"],
        "venue": pick["venue"],
        "race_number": pick["race_number"],
        "horse_name": pick["horse_name"],
        "tab_number": pick["tab_number"],
        "scheduled_time": pick["scheduled_time"],
        "place_prob": pick["place_prob"],
        "place_odds_est": pick["place_odds_est"],
        "model_pct": pick["model_pct"],
        "model_hit_probability": round(pick["place_prob"], 4),
        "stake_dollars": _FUNK_BASE_STAKE,
        "potential_return_dollars": round(pick["place_odds_est"] * _FUNK_BASE_STAKE, 2),
        "expected_value_dollars": round(pick["place_prob"] * pick["place_odds_est"] * _FUNK_BASE_STAKE - _FUNK_BASE_STAKE, 2),
    }


def _build_wave(edge_picks: list[dict]) -> Optional[dict]:
    """The Wave: Quaddie box. Prefers metro venues for pool size; falls
    back to whichever venue has the most contiguous races on country-only
    days. 2 picks per leg × 4 legs = 16 perm box, $10 base stake."""
    def collect(metro_only: bool) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for p in edge_picks:
            venue = (p.get("venue") or "").lower()
            if metro_only and venue not in _FUNK_METRO_VENUES:
                continue
            out.setdefault(venue, []).append(p)
        return out
    by_venue = collect(metro_only=True)
    # Filter to venues with ≥4 races
    qualified = {v: ps for v, ps in by_venue.items()
                 if len({p["race_id"] for p in ps}) >= 4}
    if not qualified:
        by_venue = collect(metro_only=False)
        qualified = {v: ps for v, ps in by_venue.items()
                     if len({p["race_id"] for p in ps}) >= 4}
    by_venue = qualified
    # Need a venue with at least 4 distinct races
    selected_venue = None
    selected_races = []
    for venue, picks in by_venue.items():
        # Group picks by race_id
        races: dict[str, dict] = {}
        for p in picks:
            rid = p["race_id"]
            if rid in races:
                continue
            races[rid] = p
        if len(races) < 4:
            continue
        # Sort by race_number — we want consecutive races
        sorted_races = sorted(races.values(), key=lambda r: r.get("race_number") or 99)
        if len(sorted_races) < 4:
            continue
        # Prefer the 4 latest contiguous races (typical Quaddie slot)
        chosen = sorted_races[-4:]
        selected_venue = venue
        selected_races = chosen
        break
    if not selected_races:
        return None
    legs = []
    combined_prob = 1.0
    for r in selected_races:
        field = r.get("field") or []
        active = [f for f in field if not f.get("scratched")]
        sorted_active = sorted(active, key=lambda f: -(f.get("win_pct") or 0))[:2]
        if len(sorted_active) < 2:
            return None
        picks_for_leg = [
            {
                "horse_name": f["horse_name"],
                "win_pct": f.get("win_pct"),
                "tab_number": f.get("tab_number"),
            }
            for f in sorted_active
        ]
        # Probability at least one of these 2 picks wins this leg
        leg_p = sum((f.get("win_pct") or 0) / 100 for f in sorted_active)
        leg_p = min(leg_p, 1.0)
        combined_prob *= leg_p
        legs.append({
            "race_id": r["race_id"],
            "race_number": r.get("race_number"),
            "scheduled_time": r.get("scheduled_time"),
            "picks": picks_for_leg,
            "leg_hit_probability": round(leg_p, 4),
        })
    # 2 picks per leg × 4 legs = 16 combos. At $10/16 = ~$0.625 per combo,
    # which is below the $1 minimum bet — so flexi the box at base stake.
    perms = 16
    return {
        "kind": "wave",
        "title": "The Wave",
        "subtitle": f"Quaddie box · {selected_venue.replace('-', ' ').upper()} · 2 × 4 legs (16 perms)",
        "confidence": "C",
        "venue": selected_venue,
        "legs": legs,
        "perms": perms,
        "model_hit_probability": round(combined_prob, 4),
        "stake_dollars": _FUNK_BASE_STAKE,
        # Quaddie dividends are pari-mutuel; estimate using a typical metro
        # Quaddie dividend of ~$1,200 per $1 (varies enormously). The hit
        # probability x estimated dividend gives a rough EV — UI labels it
        # as 'estimated' so the user knows it's not a fixed-odds promise.
        "estimated_dividend_dollars": 1200.0,
        "potential_return_dollars": round((_FUNK_BASE_STAKE / perms) * 1200.0, 2),
        "expected_value_dollars": round(
            combined_prob * (_FUNK_BASE_STAKE / perms) * 1200.0 - _FUNK_BASE_STAKE, 2
        ),
    }


async def _build_lab_pick(today: str) -> Optional[dict]:
    """The Lab Pick: the highest-confidence Trio from today's Lab boxes,
    filtered by ≤55% top-3 sum AND ≤11 field (the live default).
    Returns the strongest Trio box of the day."""
    async with get_session() as session:
        bet_rows = (await session.execute(
            select(BetRecommendationRow)
            .where(BetRecommendationRow.race_id.like(f"{today}_%"))
            .where(BetRecommendationRow.strategy_label == "trio_only")
        )).scalars().all()
    if not bet_rows:
        return None
    # Score each candidate by the Lab's filter signal
    best = None
    best_score = -1.0
    for b in bet_rows:
        rid = b.race_id
        async with get_session() as session:
            prob_rows = (await session.execute(
                select(RunnerPredictionRow.win_probability, RunnerPredictionRow.model_rank,
                       RunnerPredictionRow.cancelled)
                .where(RunnerPredictionRow.race_id == rid)
            )).fetchall()
        active = [(p, r) for (p, r, c) in prob_rows if not c and p is not None and r is not None]
        if len(active) < 3:
            continue
        sorted_active = sorted(active, key=lambda x: x[1])
        top3_sum = sum(p for p, _ in sorted_active[:3]) * 100
        field_size = len(active)
        if top3_sum > 55 or field_size > 11:
            continue
        # Score: higher trio_hit prob (lower top3_sum = more open race
        # but the Trio strategy itself wants the model's top-3 to LAND).
        # Use the model's top-3 sum % directly — higher is better for Trio.
        # Tie-break by smaller field.
        score = (top3_sum, -field_size)
        if best is None or score > best_score:
            best = (b, top3_sum, field_size, sorted_active)
            best_score = score
    if best is None:
        return None
    bet, top3_sum, field_size, sorted_active = best
    box_horses = json.loads(bet.box_horses_json) if bet.box_horses_json else []
    box_names = json.loads(bet.box_horse_names_json) if bet.box_horse_names_json else []
    # Trio box hit probability — odds of the 3 model-top-3 horses filling
    # the 3 placings (any order).
    top3_probs = [p for p, _ in sorted_active[:3]]
    box_hit_prob = 0.0
    from itertools import permutations
    for perm in permutations(top3_probs):
        box_hit_prob += _harville_top3_prob(*perm)
    box_hit_prob = min(box_hit_prob, 1.0)
    perms = 6
    # Use the Lab's existing $200 trifecta dividend baseline (real TAB
    # dividends average $80-$300 on metro trifectas; this is conservative).
    est_dividend = 200.0
    return {
        "kind": "lab",
        "title": "The Lab Pick",
        "subtitle": f"Trio box · top-3 sum {top3_sum:.1f}% · {field_size}-horse field",
        "confidence": "A",
        "race_id": bet.race_id,
        "venue": (bet.race_id.split("_")[1] if "_" in bet.race_id else None),
        "race_number": int(bet.race_id.rsplit("_R", 1)[1]) if "_R" in bet.race_id else None,
        "box_horses": box_horses,
        "box_horse_names": box_names,
        "perms": perms,
        "top3_sum_pct": round(top3_sum, 1),
        "field_size": field_size,
        "model_hit_probability": round(box_hit_prob, 4),
        "stake_dollars": _FUNK_BASE_STAKE,
        "estimated_dividend_dollars": est_dividend,
        "potential_return_dollars": round((_FUNK_BASE_STAKE / perms) * est_dividend, 2),
        "expected_value_dollars": round(
            box_hit_prob * (_FUNK_BASE_STAKE / perms) * est_dividend - _FUNK_BASE_STAKE, 2
        ),
    }


def _build_bonus(edge_picks: list[dict]) -> Optional[dict]:
    """The Bonus: best $4-$10 horse with model overlay, sized for a $50
    bonus bet (stake not returned per Sportsbet's bonus rules)."""
    BONUS_STAKE = 50.0
    candidates = []
    for p in edge_picks:
        odds = p.get("best_available_odds")
        edge = p.get("edge_pct")
        model_pct = p.get("model_pct") or 0
        if not odds or odds < 4.0 or odds > 10.0:
            continue
        if model_pct < 20:
            continue
        if edge is None or edge <= 5:
            continue
        if not _is_decisive_race(p, 5.0):
            continue
        candidates.append(p)
    if not candidates:
        return None
    pick = max(candidates, key=lambda p: p.get("edge_pct") or 0)
    odds = pick["best_available_odds"]
    hit_p = (pick.get("model_pct") or 0) / 100
    profit_if_won = _sb.bonus_bet_return(BONUS_STAKE, odds)
    return {
        "kind": "bonus",
        "title": "The Bonus",
        "subtitle": f"${int(BONUS_STAKE)} bonus bet · ${odds:.2f} value pick",
        "confidence": "A",
        "race_id": pick["race_id"],
        "venue": pick.get("venue"),
        "race_number": pick.get("race_number"),
        "horse_name": pick["horse_name"],
        "tab_number": pick.get("tab_number"),
        "scheduled_time": pick.get("scheduled_time"),
        "model_pct": pick.get("model_pct"),
        "edge_pct": pick.get("edge_pct"),
        "best_available_odds": odds,
        "model_hit_probability": round(hit_p, 4),
        "stake_dollars": 0.0,  # cash exposure is zero; bonus credit
        "bonus_stake_dollars": BONUS_STAKE,
        "potential_return_dollars": profit_if_won,
        "expected_value_dollars": round(hit_p * profit_if_won, 2),
    }


def _build_two_funk(edge_picks: list[dict]) -> Optional[dict]:
    """2 Funk: Quinella suggestion for races where top-2 dominate.

    Pattern (validated by 60-day backtest):
      - top-2 win sum ≥ 55%
      - rank-2 vs rank-3 gap ≥ 10pt
    Of qualifying races, picks the one with the highest top-2 sum (the
    strongest signal). 60d backtest at this threshold: 14.3% quinella
    hit (vs 4.9% baseline, ~3× lift), 42.9% split-win hit.
    """
    candidates = []
    for p in edge_picks:
        tf = p.get("two_funk")
        if not tf or not tf.get("qualified"):
            continue
        if not p.get("horse_name") or not tf.get("partner_horse_name"):
            continue
        candidates.append(p)
    if not candidates:
        return None
    # Sort by combined signal strength: top-2 sum + gap
    pick = max(
        candidates,
        key=lambda p: (
            (p["two_funk"].get("top2_win_sum_pct") or 0)
            + (p["two_funk"].get("rank2_rank3_gap_pct") or 0)
        ),
    )
    tf = pick["two_funk"]
    leg1_odds = pick.get("best_available_odds") or 0
    leg2_odds = tf.get("partner_odds") or 0
    # Approximate quinella dividend on a $1 unit: AU pari-mutuel quinellas
    # for two well-priced horses pay roughly 0.5 × W1 × W2 (a standard
    # pool heuristic). Earlier formula used the geometric mean × 0.6
    # which undershot badly — e.g. 7.50/2.50 produced $2.60 (real
    # quinella expectation: ~$9–10), pushing EV negative on what is
    # actually a +EV pattern in the 60-day backtest. Treat as an
    # estimate only; the actual dividend is set by the pool at jump.
    quinella_est = None
    if leg1_odds > 1.0 and leg2_odds > 1.0:
        quinella_est = round(leg1_odds * leg2_odds * 0.5, 2)
    # Combined model probability that BOTH finish 1-2 in either order is
    # roughly 2 * P(r1) * P(r2) for two independent picks — an over-
    # estimate because they compete with each other, but useful as an
    # upper bound for display.
    p_r1 = (pick.get("model_pct") or 0) / 100
    p_r2 = (tf.get("partner_win_pct") or 0) / 100
    quin_prob = round(2 * p_r1 * p_r2, 4) if (p_r1 and p_r2) else None
    base_stake = _FUNK_BASE_STAKE
    pot_return = round(quinella_est * base_stake, 2) if quinella_est else None
    ev = round((quin_prob * pot_return - base_stake), 2) if (quin_prob and pot_return) else None
    # EV gate — don't recommend the Quinella when the math says no. Two
    # short-priced co-favourites (e.g. $2.25 + $2.50) qualify on win-side
    # concentration but yield a Quinella dividend too small to cover the
    # ~84%% loss rate. Suppress the play card entirely in that case.
    if ev is None or ev <= 0:
        return None
    return {
        "kind": "twofunk",
        "title": "2 Funk",
        "subtitle": "Quinella · two clear standouts (top-2 dominate)",
        "confidence": "B",
        "race_id": pick["race_id"],
        "venue": pick.get("venue"),
        "race_number": pick.get("race_number"),
        "scheduled_time": pick.get("scheduled_time"),
        "horse_name": pick["horse_name"],  # rank-1 (the primary leg)
        "tab_number": pick.get("tab_number"),
        "partner_horse_name": tf.get("partner_horse_name"),
        "partner_tab_number": tf.get("partner_tab_number"),
        "partner_odds": leg2_odds or None,
        "leg1_odds": leg1_odds or None,
        "model_pct": pick.get("model_pct"),
        "partner_win_pct": tf.get("partner_win_pct"),
        "top2_win_sum_pct": tf.get("top2_win_sum_pct"),
        "rank2_rank3_gap_pct": tf.get("rank2_rank3_gap_pct"),
        "model_hit_probability": quin_prob,
        "quinella_est_dividend": quinella_est,
        "stake_dollars": base_stake,
        "potential_return_dollars": pot_return,
        "expected_value_dollars": ev,
    }


async def _build_historical_funk_picks(target_date: str) -> list[dict]:
    """Rebuild a picks-list with the shape /api/edge would have produced
    for a given past date — used by the yesterday Funk Me Up view.
    Sources directly from RunnerPredictionHistoryRow + the latest
    enriched_at snapshot per horse."""
    async with get_session() as session:
        rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{target_date}_%"))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).scalars().all()
    # Dedup latest-per-(race, horse)
    seen: set = set()
    by_race: dict[str, list] = {}
    for r in rows:
        key = (r.race_id, _normalize_horse(r.horse_name))
        if key in seen:
            continue
        seen.add(key)
        by_race.setdefault(r.race_id, []).append(r)
    picks: list[dict] = []
    for rid, runners in by_race.items():
        runners.sort(key=lambda x: x.model_rank or 99)
        if not runners:
            continue
        rank1 = runners[0]
        _, venue, race_num = _parse_race_id(rid)
        field = []
        for r in runners:
            field.append({
                "rank": r.model_rank,
                "horse_name": r.horse_name,
                "tab_number": r.tab_number,
                "win_pct": (r.win_probability or 0) * 100,
                "place_pct": (r.place_probability or 0) * 100 if r.place_probability else 0,
                "odds": 0.0,
                "scratched": bool(r.cancelled),
            })
        picks.append({
            "race_id": rid,
            "horse_name": rank1.horse_name,
            "model_pct": round((rank1.win_probability or 0) * 100, 1),
            "venue": venue,
            "race_number": race_num,
            "scheduled_time": rank1.scheduled_time,
            "best_available_odds": rank1.best_available_odds or 0,
            "market_implied_pct": None,
            "edge_pct": rank1.overlay * 100 if rank1.overlay else None,
            "calibrated_win_rate": None,
            "field": field,
        })
    return picks


async def _resolve_play_outcome(play: dict, target_date: str) -> dict:
    """Look up actual race results for a Funk Me Up play and return an
    outcome dict: status (won/lost/partial), profit, details."""
    if not play:
        return {}
    kind = play.get("kind")
    async with get_session() as session:
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.tab_number, HistoricalResultRow.horse_name,
                   HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id.like(f"{target_date}_%"))
            .where(HistoricalResultRow.position.in_([1, 2, 3, 4]))
        )).fetchall()
    results: dict[str, dict[int, dict]] = {}
    for rid, pos, tab, name, sp in result_rows:
        if pos is None:
            continue
        results.setdefault(rid, {})[int(pos)] = {
            "tab": tab, "horse_name": name, "sp": sp,
        }

    def horse_pos(rid: str, horse_name: str) -> Optional[int]:
        slots = results.get(rid) or {}
        tgt = _normalize_horse(horse_name or "")
        for p, row in slots.items():
            if _normalize_horse(row["horse_name"]) == tgt:
                return p
        return None

    BASE = play.get("stake_dollars") or 10.0
    BONUS_STAKE = play.get("bonus_stake_dollars") or 0

    if kind in ("lock", "bonus"):
        rid = play["race_id"]
        pos = horse_pos(rid, play["horse_name"])
        won = pos == 1
        placed = pos in (1, 2, 3)
        sp = (results.get(rid) or {}).get(pos, {}).get("sp") if pos else None
        if kind == "lock":
            profit = (BASE * (sp - 1)) if won and sp else (-BASE if pos is not None else 0)
        else:
            profit = (BONUS_STAKE * (sp - 1)) if won and sp else 0
        return {
            "status": "won" if won else ("lost" if pos is not None else "no_result"),
            "won": won,
            "placed": placed,
            "actual_position": pos,
            "profit_dollars": round(profit, 2) if profit is not None else 0,
            "summary": (
                f"Won at ${sp:.2f}" if won and sp else
                f"Won (no SP)" if won else
                f"{pos}{'st' if pos==1 else 'nd' if pos==2 else 'rd' if pos==3 else 'th'}" if pos else
                "Results pending"
            ),
        }
    if kind == "spine":
        legs = play.get("legs") or []
        leg_hits = []
        for l in legs:
            pos = horse_pos(l["race_id"], l["horse_name"])
            leg_hits.append({
                "race_id": l["race_id"],
                "horse_name": l["horse_name"],
                "position": pos,
                "hit": pos is not None and pos <= 4,
            })
        all_hit = all(h["hit"] for h in leg_hits) if leg_hits else False
        any_pos_known = any(h["position"] is not None for h in leg_hits)
        boosted = play.get("boosted_multi_odds") or 0
        profit = BASE * boosted - BASE if all_hit else (-BASE if any_pos_known else 0)
        return {
            "status": "won" if all_hit else ("lost" if any_pos_known else "no_result"),
            "hit_count": sum(1 for h in leg_hits if h["hit"]),
            "leg_count": len(leg_hits),
            "leg_hits": leg_hits,
            "profit_dollars": round(profit, 2),
            "summary": (
                f"All 4 legs landed top-4 — multi paid ${BASE * boosted:.2f}"
                if all_hit else
                f"{sum(1 for h in leg_hits if h['hit'])}/{len(leg_hits)} legs hit"
                if any_pos_known else "Results pending"
            ),
        }
    if kind == "wave":
        legs = play.get("legs") or []
        leg_hits = []
        for l in legs:
            slots = results.get(l["race_id"]) or {}
            winner_row = slots.get(1)
            winner = winner_row["horse_name"] if winner_row else None
            our_picks_normed = {_normalize_horse(x["horse_name"]) for x in (l.get("picks") or [])}
            hit = (winner is not None) and (_normalize_horse(winner) in our_picks_normed)
            leg_hits.append({
                "race_id": l["race_id"],
                "winner": winner,
                "hit": hit,
            })
        all_hit = all(h["hit"] for h in leg_hits) if leg_hits else False
        any_known = any(h["winner"] is not None for h in leg_hits)
        # Pari-mutuel — use the play's own estimate
        dividend = play.get("estimated_dividend_dollars") or 1200
        perms = play.get("perms") or 16
        profit = (BASE / perms) * dividend - BASE if all_hit else (-BASE if any_known else 0)
        return {
            "status": "won" if all_hit else ("lost" if any_known else "no_result"),
            "hit_count": sum(1 for h in leg_hits if h["hit"]),
            "leg_count": len(leg_hits),
            "leg_hits": leg_hits,
            "profit_dollars": round(profit, 2),
            "summary": (
                f"All 4 winners came from our picks — Quaddie cashed (~${(BASE/perms)*dividend:.0f})"
                if all_hit else
                f"{sum(1 for h in leg_hits if h['hit'])}/{len(leg_hits)} legs hit"
                if any_known else "Results pending"
            ),
        }
    if kind == "double":
        legs = play.get("legs") or []
        leg_hits = []
        for l in legs:
            pos = horse_pos(l["race_id"], l["horse_name"])
            leg_hits.append({
                "race_id": l["race_id"],
                "horse_name": l["horse_name"],
                "position": pos,
                "hit": pos is not None and pos <= 3,
            })
        all_hit = all(h["hit"] for h in leg_hits) if leg_hits else False
        any_pos_known = any(h["position"] is not None for h in leg_hits)
        raw_multi = play.get("raw_multi_odds") or 0
        profit = BASE * raw_multi - BASE if all_hit else (-BASE if any_pos_known else 0)
        return {
            "status": "won" if all_hit else ("lost" if any_pos_known else "no_result"),
            "hit_count": sum(1 for h in leg_hits if h["hit"]),
            "leg_count": len(leg_hits),
            "leg_hits": leg_hits,
            "profit_dollars": round(profit, 2),
            "summary": (
                f"Both legs placed — Place Double paid ~${BASE * raw_multi:.2f}"
                if all_hit else
                f"{sum(1 for h in leg_hits if h['hit'])}/{len(leg_hits)} legs placed"
                if any_pos_known else "Results pending"
            ),
        }
    if kind == "banker":
        rid = play["race_id"]
        pos = horse_pos(rid, play["horse_name"])
        placed = pos is not None and pos <= 3
        odds = play.get("place_odds_est") or 0
        profit = BASE * odds - BASE if placed else (-BASE if pos is not None else 0)
        return {
            "status": "won" if placed else ("lost" if pos is not None else "no_result"),
            "actual_position": pos,
            "placed": placed,
            "profit_dollars": round(profit, 2),
            "summary": (
                f"Placed {pos}{'st' if pos==1 else 'nd' if pos==2 else 'rd' if pos==3 else 'th'} — paid ${BASE * odds:.2f}"
                if placed else
                f"Finished {pos}{'th' if pos and pos > 3 else ''}"
                if pos else "Results pending"
            ),
        }
    if kind == "twofunk":
        rid = play["race_id"]
        leg1 = horse_pos(rid, play.get("horse_name") or "")
        leg2 = horse_pos(rid, play.get("partner_horse_name") or "")
        # Quinella hits when both legs finish in positions 1-2 (either order)
        quinella_hit = leg1 in (1, 2) and leg2 in (1, 2) and leg1 != leg2
        any_known = leg1 is not None or leg2 is not None
        dividend = play.get("quinella_est_dividend") or 0
        profit = (BASE * dividend - BASE) if (quinella_hit and dividend) else (-BASE if any_known else 0)
        leg1_lbl = f"{leg1}{'st' if leg1==1 else 'nd' if leg1==2 else 'rd' if leg1==3 else 'th'}" if leg1 else "—"
        leg2_lbl = f"{leg2}{'st' if leg2==1 else 'nd' if leg2==2 else 'rd' if leg2==3 else 'th'}" if leg2 else "—"
        return {
            "status": "won" if quinella_hit else ("lost" if any_known else "no_result"),
            "leg1_position": leg1,
            "leg2_position": leg2,
            "profit_dollars": round(profit, 2),
            "summary": (
                f"Quinella hit — {play['horse_name']} {leg1_lbl}, {play['partner_horse_name']} {leg2_lbl}"
                if quinella_hit else
                f"{play['horse_name']} {leg1_lbl} / {play['partner_horse_name']} {leg2_lbl}"
                if any_known else "Results pending"
            ),
        }
    if kind == "lab":
        rid = play.get("race_id")
        if not rid:
            return {"status": "no_result", "summary": "Results pending"}
        slots = results.get(rid) or {}
        top3_names = []
        for p in (1, 2, 3):
            row = slots.get(p)
            if row:
                top3_names.append(_normalize_horse(row["horse_name"]))
        box = play.get("box_horse_names") or []
        box_normed = {_normalize_horse(n) for n in box}
        hit = len(top3_names) == 3 and set(top3_names) == box_normed
        any_known = len(top3_names) > 0
        dividend = play.get("estimated_dividend_dollars") or 200
        perms = play.get("perms") or 6
        profit = (BASE / perms) * dividend - BASE if hit else (-BASE if any_known else 0)
        # Did our box catch any of the placings (partial credit)?
        overlap = len(box_normed & set(top3_names))
        return {
            "status": "won" if hit else ("lost" if any_known else "no_result"),
            "overlap": overlap,
            "leg_count": 3,
            "profit_dollars": round(profit, 2),
            "summary": (
                f"Trio cashed — all 3 placings in our box (~${(BASE/perms)*dividend:.0f})"
                if hit else
                f"{overlap}/3 placings in our box"
                if any_known else "Results pending"
            ),
        }
    return {}


@app.get("/api/funk-me-up/today")
async def funk_me_up_today(date: Optional[str] = None):
    """Funk Me Up playbook for a given date.

    - date omitted or today  → live picks from /api/edge
    - date is tomorrow / N days forward → live picks filtered by date
    - date is yesterday / past → rebuilt from history with outcomes resolved

    Accepts ?date=YYYY-MM-DD. Defaults to today's AEST date."""
    today_iso = _today_aest().isoformat()
    target_date = date or today_iso
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")
    today_dt = _today_aest()
    is_past = target_dt < today_dt

    if is_past:
        picks_for_date = await _build_historical_funk_picks(target_date)
    else:
        edge = await get_edge_picks()
        prefix = f"{target_date}_"
        picks_for_date = [p for p in edge.get("picks", []) if (p.get("race_id") or "").startswith(prefix)]

    plays = []
    # 2026-06-23 rethink: dropped Spine (4-leg Top4 multi, -39% ROI) and
    # Wave (Quaddie, -100%) for high variance. Added Double (2-leg place
    # multi, +71% ROI) and Banker (single place) for low variance.
    # _build_bonus removed 2026-06-26 — user feedback: focus on real-money
    # plays only, don't condition cards on holding a Sportsbet bonus bet.
    # Function kept in code in case of a future opt-in toggle.
    for build_fn in (_build_lock, _build_double, _build_banker, _build_combo_doubles, _build_two_funk):
        try:
            play = build_fn(picks_for_date)
        except Exception as e:
            log.warning("[funk-me-up] %s failed for %s: %s", build_fn.__name__, target_date, e)
            play = None
        if play:
            plays.append(play)
    # Early Quaddie is async — queries the DB directly (not /api/edge)
    # so it can include rank-1 picks that fall below the Edge threshold.
    try:
        quaddie_play = await _build_early_quaddie(target_date)
        if quaddie_play:
            plays.append(quaddie_play)
    except Exception as e:
        log.warning("[funk-me-up] _build_early_quaddie failed for %s: %s", target_date, e)
    try:
        lab_play = await _build_lab_pick(target_date)
        if lab_play:
            plays.append(lab_play)
    except Exception as e:
        log.warning("[funk-me-up] lab pick failed for %s: %s", target_date, e)

    # Tag each play with its membership tier (matches the UI tier-marker).
    # Per-play kind is the authority — Lab Pick + 2 Funk are Labs, the rest
    # are Pro. Defaults to "pro" if a future kind is unmapped (Funk Me Up
    # itself is a Pro surface) so we never accidentally over-paywall.
    for p in plays:
        p["tier"] = feature_tier(f"play_{p.get('kind') or ''}") if FEATURE_TIERS.get(f"play_{p.get('kind') or ''}") else "pro"

    # Resolve outcomes for past dates
    if is_past:
        for p in plays:
            try:
                p["outcome"] = await _resolve_play_outcome(p, target_date)
            except Exception as e:
                log.debug("[funk-me-up] outcome resolution failed: %s", e)
                p["outcome"] = {"status": "no_result", "summary": "Results unavailable"}

    # Hero: highest expected_value_dollars across plays.
    hero = None
    if plays:
        hero = max(plays, key=lambda p: (
            p.get("expected_value_dollars") or 0,
            p.get("model_hit_probability") or 0
        ))

    total_cash_exposure = round(sum(p.get("stake_dollars") or 0 for p in plays), 2)
    bonus_exposure = round(sum(p.get("bonus_stake_dollars") or 0 for p in plays), 2)
    expected_profit = round(sum(p.get("expected_value_dollars") or 0 for p in plays), 2)
    actual_profit = None
    if is_past:
        actual_profit = round(
            sum((p.get("outcome", {}).get("profit_dollars") or 0) for p in plays), 2
        )

    return {
        "date": target_date,
        "is_past": is_past,
        "tier": feature_tier("funk_me_up"),      # surface tier — "pro"
        "base_stake": _FUNK_BASE_STAKE,
        "hero_kind": hero.get("kind") if hero else None,
        "plays": plays,
        "cash_exposure_dollars": total_cash_exposure,
        "bonus_exposure_dollars": bonus_exposure,
        "expected_profit_dollars": expected_profit,
        "actual_profit_dollars": actual_profit,
        "disclaimer": "Estimates use TAB odds; Sportsbet prices vary. Verify boost % and Top 4 prices in the Sportsbet betslip before placing.",
    }


_funk_backtest_cache: tuple[datetime, dict] | None = None
_FUNK_BACKTEST_TTL = 3600  # 1h — backtest moves once per day, no need to recompute often


@app.get("/api/funk-me-up/backtest")
async def funk_me_up_backtest(days: int = 7):
    """Replay each Funk Me Up play's selection logic against the last N
    days of historical predictions + results. Returns per-play hit count,
    ROI, and total P/L estimate so the UI can show a 7-day track record
    on each card."""
    global _funk_backtest_cache
    if _funk_backtest_cache is not None:
        ts, body = _funk_backtest_cache
        if (datetime.utcnow() - ts).total_seconds() < _FUNK_BACKTEST_TTL and body.get("days") == days:
            return body
    days = max(1, min(int(days), 30))
    today = _today_aest()
    target_dates = [(today - timedelta(days=i)).isoformat() for i in range(1, days + 1)]
    earliest = target_dates[-1]

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= f"{earliest}_")
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )).scalars().all()
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.tab_number, HistoricalResultRow.horse_name,
                   HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id >= f"{earliest}_")
        )).fetchall()
        lab_rows = (await session.execute(
            select(BetRecommendationRow)
            .where(BetRecommendationRow.race_id >= f"{earliest}_")
            .where(BetRecommendationRow.strategy_label == "trio_only")
            .where(BetRecommendationRow.settled.is_(True))
        )).scalars().all()

    # Bucket predictions by race; keep latest enriched_at per (race, horse)
    preds_by_race: dict[str, list] = {}
    seen: dict[tuple, datetime] = {}
    for p in pred_rows:
        key = (p.race_id, _normalize_horse(p.horse_name))
        if key in seen and (p.enriched_at or datetime.min) <= seen[key]:
            continue
        seen[key] = p.enriched_at or datetime.min
        preds_by_race.setdefault(p.race_id, []).append(p)

    # Results by race — keep position 1..4 with SP and tab
    result_by_race: dict[str, dict] = {}
    for rid, pos, tab, name, sp in result_rows:
        if pos is None:
            continue
        slot = result_by_race.setdefault(rid, {})
        slot[int(pos)] = {"tab": tab, "horse_name": name, "sp": sp}

    def race_date(rid: str) -> str:
        return rid.split("_", 1)[0]

    # Per-play accumulators
    stats = {k: {"days_evaluated": 0, "days_hit": 0,
                 "total_stake": 0.0, "total_pnl": 0.0}
             for k in ("spine", "lock", "wave", "lab", "bonus")}
    BASE = 10.0
    BONUS_STAKE = 50.0

    def _was_top_n(rid: str, horse_name: str, n: int) -> bool:
        slots = result_by_race.get(rid) or {}
        target = _normalize_horse(horse_name)
        for pos in range(1, n + 1):
            row = slots.get(pos)
            if row and _normalize_horse(row["horse_name"]) == target:
                return True
        return False

    def _winner_horse(rid: str) -> Optional[str]:
        slots = result_by_race.get(rid) or {}
        row = slots.get(1)
        return row["horse_name"] if row else None

    def _sp_for(rid: str, horse_name: str) -> Optional[float]:
        slots = result_by_race.get(rid) or {}
        target = _normalize_horse(horse_name)
        for pos in range(1, 5):
            row = slots.get(pos)
            if row and _normalize_horse(row["horse_name"]) == target:
                return row.get("sp")
        return None

    for date in target_dates:
        # Restrict to races on this date that have any result
        prefix = f"{date}_"
        race_ids_today = [rid for rid in preds_by_race
                          if rid.startswith(prefix) and rid in result_by_race]
        if not race_ids_today:
            continue

        # --- SPINE: 4-leg Top 4 multi ---
        spine_legs = []
        for rid in race_ids_today:
            runners = sorted(preds_by_race[rid], key=lambda p: p.model_rank or 99)
            if len(runners) < _sb.TOP4_MIN_FIELD_SIZE:
                continue
            rank1 = runners[0]
            if (rank1.win_probability or 0) <= 0:
                continue
            others = [r.win_probability for r in runners[1:] if r.win_probability]
            top4_prob = _harville_horse_top_n(rank1.win_probability, others, 4)
            if top4_prob < 0.60:
                continue
            sb_odds = round(1.0 / max(top4_prob - _sb.TOP4_OVERROUND_PCT * top4_prob, 0.05), 2)
            spine_legs.append({"race_id": rid, "horse_name": rank1.horse_name,
                               "sb_odds": sb_odds, "top4_prob": top4_prob})
        if len(spine_legs) >= 4:
            spine_legs.sort(key=lambda l: -l["top4_prob"])
            chosen = spine_legs[:4]
            raw_multi = 1.0
            for l in chosen:
                raw_multi *= l["sb_odds"]
            boosted = _sb.boosted_multi_odds(raw_multi, 4)
            all_hit = all(_was_top_n(l["race_id"], l["horse_name"], 4) for l in chosen)
            stats["spine"]["days_evaluated"] += 1
            stats["spine"]["total_stake"] += BASE
            if all_hit:
                stats["spine"]["days_hit"] += 1
                stats["spine"]["total_pnl"] += boosted * BASE - BASE
            else:
                stats["spine"]["total_pnl"] -= BASE

        # --- LOCK: best single Premium ---
        lock_candidates = []
        for rid in race_ids_today:
            runners = sorted(preds_by_race[rid], key=lambda p: p.model_rank or 99)
            if not runners:
                continue
            rank1 = runners[0]
            model_pct = (rank1.win_probability or 0) * 100
            if model_pct < 29.5:
                continue
            sp = _sp_for(rid, rank1.horse_name) or 0
            # Use winner's SP if our pick won; else use any active SP from race or skip
            slots = result_by_race.get(rid) or {}
            # Look for SP on rank-1 from any finishing pos; if no SP found, skip the race
            target_sp = None
            for p in (1, 2, 3, 4):
                row = slots.get(p)
                if row and _normalize_horse(row["horse_name"]) == _normalize_horse(rank1.horse_name):
                    target_sp = row.get("sp")
                    break
            if not target_sp or target_sp < 3.0:
                continue
            market_pct = (1.0 / target_sp) * 100 * 0.88
            edge = model_pct - market_pct
            if edge <= 5:
                continue
            lock_candidates.append({"race_id": rid, "horse_name": rank1.horse_name,
                                    "sp": target_sp, "edge": edge})
        if lock_candidates:
            best = max(lock_candidates, key=lambda c: c["edge"])
            won = _was_top_n(best["race_id"], best["horse_name"], 1)
            stats["lock"]["days_evaluated"] += 1
            stats["lock"]["total_stake"] += BASE
            if won:
                stats["lock"]["days_hit"] += 1
                stats["lock"]["total_pnl"] += BASE * best["sp"] - BASE
            else:
                stats["lock"]["total_pnl"] -= BASE

        # --- WAVE: Quaddie box at any venue with 4+ races ---
        by_venue: dict[str, list[str]] = {}
        for rid in race_ids_today:
            venue = rid.split("_")[1] if "_" in rid else None
            if not venue:
                continue
            by_venue.setdefault(venue, []).append(rid)
        wave_venue, wave_legs = None, None
        for venue, rids in by_venue.items():
            rids = sorted(rids, key=lambda r: int(r.rsplit("_R", 1)[1]) if "_R" in r else 0)
            if len(rids) < 4:
                continue
            wave_venue, wave_legs = venue, rids[-4:]
            break
        if wave_legs:
            all_legs_have_top2 = True
            wave_all_hit = True
            for rid in wave_legs:
                runners = sorted(preds_by_race[rid], key=lambda p: p.model_rank or 99)
                top2 = runners[:2]
                if len(top2) < 2:
                    all_legs_have_top2 = False
                    break
                winner = _winner_horse(rid)
                if winner is None:
                    all_legs_have_top2 = False
                    break
                tgt = _normalize_horse(winner)
                hit = any(_normalize_horse(r.horse_name) == tgt for r in top2)
                if not hit:
                    wave_all_hit = False
                    # Continue to confirm legs were valid; result still counted
            if all_legs_have_top2:
                stats["wave"]["days_evaluated"] += 1
                stats["wave"]["total_stake"] += BASE
                if wave_all_hit:
                    stats["wave"]["days_hit"] += 1
                    # Pari-mutuel approximation: assume $1,200 dividend on $1
                    # base, stake $10 / 16 perms = $0.625/perm. Use $1200 as
                    # baseline dividend, scaled by per-perm share.
                    stats["wave"]["total_pnl"] += (BASE / 16) * 1200.0 - BASE
                else:
                    stats["wave"]["total_pnl"] -= BASE

        # --- BONUS: $4-$10 overlay ---
        bonus_candidates = []
        for rid in race_ids_today:
            runners = sorted(preds_by_race[rid], key=lambda p: p.model_rank or 99)
            if not runners:
                continue
            rank1 = runners[0]
            model_pct = (rank1.win_probability or 0) * 100
            if model_pct < 20:
                continue
            sp = _sp_for(rid, rank1.horse_name)
            if not sp or sp < 4.0 or sp > 10.0:
                continue
            market_pct = (1.0 / sp) * 100 * 0.88
            edge = model_pct - market_pct
            if edge <= 5:
                continue
            bonus_candidates.append({"race_id": rid, "horse_name": rank1.horse_name,
                                     "sp": sp, "edge": edge})
        if bonus_candidates:
            best = max(bonus_candidates, key=lambda c: c["edge"])
            won = _was_top_n(best["race_id"], best["horse_name"], 1)
            stats["bonus"]["days_evaluated"] += 1
            # Bonus doesn't count as cash stake — track total_pnl as raw upside
            if won:
                stats["bonus"]["days_hit"] += 1
                stats["bonus"]["total_pnl"] += BONUS_STAKE * (best["sp"] - 1)
            # No cash loss on a missed bonus bet

    # --- LAB PICK: use existing BetRecommendationRow.is_hit on the trio_only ---
    lab_by_day: dict[str, list] = {}
    for b in lab_rows:
        d = race_date(b.race_id)
        if d not in target_dates:
            continue
        lab_by_day.setdefault(d, []).append(b)
    for d, bets in lab_by_day.items():
        # Pick best (smallest top3 sum from prediction history) — match live selector
        scored = []
        for b in bets:
            rid = b.race_id
            runners = sorted(preds_by_race.get(rid, []), key=lambda p: p.model_rank or 99)
            if len(runners) < 3:
                continue
            top3_sum = sum((r.win_probability or 0) for r in runners[:3]) * 100
            field = len([r for r in runners])
            if top3_sum > 55 or field > 11:
                continue
            scored.append((b, top3_sum))
        if not scored:
            continue
        chosen, _ = max(scored, key=lambda x: x[1])  # match live selector tiebreak
        stats["lab"]["days_evaluated"] += 1
        stats["lab"]["total_stake"] += BASE
        if chosen.is_hit:
            stats["lab"]["days_hit"] += 1
            # Use estimated $200 dividend × stake/6 perms
            stats["lab"]["total_pnl"] += (BASE / 6) * 200.0 - BASE
        else:
            stats["lab"]["total_pnl"] -= BASE

    out = {}
    for k, s in stats.items():
        hit_rate = round(s["days_hit"] / s["days_evaluated"] * 100, 1) if s["days_evaluated"] else 0
        roi = round(s["total_pnl"] / s["total_stake"] * 100, 1) if s["total_stake"] else None
        out[k] = {
            "days_evaluated": s["days_evaluated"],
            "days_hit": s["days_hit"],
            "hit_rate_pct": hit_rate,
            "total_stake": round(s["total_stake"], 2),
            "total_pnl": round(s["total_pnl"], 2),
            "roi_pct": roi,
        }
    body = {"days": days, "by_play": out}
    _funk_backtest_cache = (datetime.utcnow(), body)
    return body


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


# ─── Bet recommender endpoints (paper-trading) ────────────────────────────
def _row_to_bet_dict(b: BetRecommendationRow) -> dict:
    return {
        "id": b.id,
        "race_id": b.race_id,
        "strategy_label": b.strategy_label,
        "strategy_group": _STRATEGY_GROUP.get(b.strategy_label, "spread"),
        "box_horses": json.loads(b.box_horses_json) if b.box_horses_json else [],
        "box_horse_names": json.loads(b.box_horse_names_json) if b.box_horse_names_json else [],
        "num_permutations": b.num_permutations,
        "stake_dollars": b.stake_dollars,
        "recommended_at": b.recommended_at.isoformat() if b.recommended_at else None,
        "settled": bool(b.settled),
        "is_hit": b.is_hit,
        "actual_top3": json.loads(b.actual_top3_json) if b.actual_top3_json else None,
        "trifecta_dividend": b.trifecta_dividend,
        "dividend_estimated": bool(b.dividend_estimated),
        "payout_dollars": b.payout_dollars,
        "pnl_dollars": b.pnl_dollars,
        "settled_at": b.settled_at.isoformat() if b.settled_at else None,
    }


def _strategy_labels_for_group(group: str) -> Optional[list[str]]:
    """Return the list of strategy_label values that belong to the given
    group, or None if the group is 'all' / unset (no filter)."""
    if not group or group == "all":
        return None
    return [lbl for lbl, g in _STRATEGY_GROUP.items() if g == group]


@app.get("/api/bets/{race_id}")
async def get_bets_for_race(race_id: str, strategy: Optional[str] = None):
    """Paper-trading bet recommendations for a single race. Pass
    strategy=spread or strategy=sweep to filter."""
    labels = _strategy_labels_for_group(strategy)
    async with get_session() as session:
        q = (
            select(BetRecommendationRow)
            .where(BetRecommendationRow.race_id == race_id)
            .order_by(BetRecommendationRow.id)
        )
        if labels:
            q = q.where(BetRecommendationRow.strategy_label.in_(labels))
        rows = (await session.execute(q)).scalars().all()
    return {"race_id": race_id, "strategy": strategy or "all",
            "bets": [_row_to_bet_dict(b) for b in rows]}


@app.get("/api/bets")
async def list_bet_races(
    days: int = 7,
    strategy: Optional[str] = None,
    include_bets: bool = False,
):
    """Race-by-race paper-trading ledger. Pass strategy=spread or
    strategy=sweep to filter the per-race aggregation; default returns
    the combined view. With include_bets=true, each race carries an
    inline `bets_by_strategy` map so The Lab can render every strategy's
    boxes inside the race card."""
    days = max(1, min(int(days), 60))
    cutoff = datetime.utcnow() - timedelta(days=days)
    labels = _strategy_labels_for_group(strategy)
    async with get_session() as session:
        q = (
            select(BetRecommendationRow)
            .where(BetRecommendationRow.recommended_at >= cutoff)
            .order_by(BetRecommendationRow.recommended_at.desc())
        )
        if labels:
            q = q.where(BetRecommendationRow.strategy_label.in_(labels))
        rows = (await session.execute(q)).scalars().all()
        race_ids = list({r.race_id for r in rows})
        # Per-(race, tab) odds lookup — only fetched when include_bets is on,
        # used to annotate each box horse with its current live odds.
        odds_lookup: dict[tuple[str, int], float] = {}
        if include_bets and race_ids:
            odds_rows = (await session.execute(
                select(RunnerPredictionRow.race_id, RunnerPredictionRow.tab_number,
                       RunnerPredictionRow.best_available_odds)
                .where(RunnerPredictionRow.race_id.in_(race_ids))
                .where(RunnerPredictionRow.tab_number.isnot(None))
            )).fetchall()
            for rid, tab, odds in odds_rows:
                if odds and odds > 1.0:
                    odds_lookup[(rid, tab)] = odds
        # Top-3 results per race (for the winning-combination strip on
        # settled cards). Pull tab + horse name + position; group below.
        top3_map: dict[str, list[dict]] = {}
        if race_ids:
            top3_rows = (await session.execute(
                select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                       HistoricalResultRow.tab_number, HistoricalResultRow.horse_name)
                .where(HistoricalResultRow.race_id.in_(race_ids))
                .where(HistoricalResultRow.position.in_([1, 2, 3]))
            )).fetchall()
            for rid, pos, tab, name in top3_rows:
                top3_map.setdefault(rid, []).append({
                    "position": pos, "tab_number": tab, "horse_name": name,
                })
            # Backfill missing tab numbers per race from prediction tables.
            needs_backfill = {rid for rid, items in top3_map.items()
                              if any(it["tab_number"] is None for it in items)}
            if needs_backfill:
                for src in (RunnerPredictionRow, RunnerPredictionHistoryRow):
                    rows_with_tab = (await session.execute(
                        select(src.race_id, src.horse_name, src.tab_number)
                        .where(src.race_id.in_(needs_backfill))
                        .where(src.tab_number.isnot(None))
                    )).fetchall()
                    lookups: dict[tuple, int] = {}
                    for rid, name, tab in rows_with_tab:
                        lookups[(rid, _normalize_horse(name))] = tab
                    for rid in list(needs_backfill):
                        for it in top3_map.get(rid, []):
                            if it["tab_number"] is None:
                                t = lookups.get((rid, _normalize_horse(it["horse_name"])))
                                if t is not None:
                                    it["tab_number"] = t
            for rid in top3_map:
                top3_map[rid].sort(key=lambda x: x["position"])
        # Scheduled time per race — pulled from mutable first (newer odds-
        # refresh enrichments may have set it), falling back to history.
        sched_map: dict[str, str] = {}
        if race_ids:
            for raceid, st in (await session.execute(
                select(RunnerPredictionRow.race_id, func.max(RunnerPredictionRow.scheduled_time))
                .where(RunnerPredictionRow.race_id.in_(race_ids))
                .where(RunnerPredictionRow.scheduled_time.isnot(None))
                .group_by(RunnerPredictionRow.race_id)
            )).fetchall():
                if st:
                    sched_map[raceid] = st
            missing = [r for r in race_ids if r not in sched_map]
            if missing:
                for raceid, st in (await session.execute(
                    select(RunnerPredictionHistoryRow.race_id, func.max(RunnerPredictionHistoryRow.scheduled_time))
                    .where(RunnerPredictionHistoryRow.race_id.in_(missing))
                    .where(RunnerPredictionHistoryRow.scheduled_time.isnot(None))
                    .group_by(RunnerPredictionHistoryRow.race_id)
                )).fetchall():
                    if st:
                        sched_map[raceid] = st
    # Per-race feature extraction for the Lab's confidence filter +
    # downstream signal analysis. We compute several features per race
    # from the prediction snapshots:
    #   - rank1_win_pct  : the favourite's model probability
    #   - top3_sum_pct   : combined model prob of the model's top-3 horses
    #   - field_size     : number of active (non-cancelled) runners
    # Prefer history (frozen pre-race state); fall back to mutable predictions
    # for races that haven't been snapshotted yet.
    rank1_pct: dict[str, float] = {}
    top3_sum: dict[str, float] = {}
    field_size: dict[str, int] = {}
    if race_ids:
        from collections import defaultdict
        race_probs: dict[str, list[tuple[int, float]]] = defaultdict(list)
        async with get_session() as session:
            hist_rows = (await session.execute(
                select(RunnerPredictionHistoryRow.race_id,
                       RunnerPredictionHistoryRow.model_rank,
                       RunnerPredictionHistoryRow.win_probability,
                       RunnerPredictionHistoryRow.enriched_at)
                .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
                .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                       | RunnerPredictionHistoryRow.cancelled.is_(None))
                .where(RunnerPredictionHistoryRow.source == "live")
                .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
            )).fetchall()
        # Group by race; keep only the latest enriched_at snapshot per race
        # (rows are already DESC-sorted so first occurrence per race wins).
        seen_race = set()
        for rid, rank, p, _ in hist_rows:
            if rid in seen_race and rid in race_probs and len(race_probs[rid]) > 25:
                continue
            if rid not in seen_race:
                seen_race.add(rid)
                race_probs[rid] = []
            if rank is not None and p is not None:
                race_probs[rid].append((int(rank), float(p)))
        # For races with no history, pull from mutable.
        missing = [rid for rid in race_ids if rid not in race_probs]
        if missing:
            async with get_session() as session:
                live_rows = (await session.execute(
                    select(RunnerPredictionRow.race_id,
                           RunnerPredictionRow.model_rank,
                           RunnerPredictionRow.win_probability)
                    .where(RunnerPredictionRow.race_id.in_(missing))
                    .where(RunnerPredictionRow.cancelled.is_(False)
                           | RunnerPredictionRow.cancelled.is_(None))
                )).fetchall()
            for rid, rank, p in live_rows:
                if rank is not None and p is not None:
                    race_probs[rid].append((int(rank), float(p)))
        for rid, items in race_probs.items():
            items.sort()
            field_size[rid] = len(items)
            if items:
                rank1_pct[rid] = items[0][1] * 100
                top3 = [p for _, p in items[:3]]
                top3_sum[rid] = sum(top3) * 100

    by_race: dict[str, list[BetRecommendationRow]] = {}
    for r in rows:
        by_race.setdefault(r.race_id, []).append(r)
    races = []
    for race_id, bets in by_race.items():
        date, venue, race_num = _parse_race_id(race_id)
        total_stake = sum((b.stake_dollars or 0) for b in bets)
        total_payout = sum((b.payout_dollars or 0) for b in bets if b.settled and b.payout_dollars is not None)
        any_unsettled = any(not b.settled for b in bets)
        has_dividend = any(b.settled and b.trifecta_dividend is not None for b in bets)
        # P&L only computable when we actually have dividend data. Settled
        # races without a dividend show 'dividend pending'.
        pnl = round(total_payout - total_stake, 2) if (not any_unsettled and has_dividend) else None
        # Status: pending (race hasn't been processed), settled-no-div (hits
        # known but dividend missing), or settled (full P&L available).
        if any_unsettled:
            status = "pending"
        elif has_dividend:
            status = "settled"
        else:
            status = "settled_no_dividend"
        sched = sched_map.get(race_id)
        if isinstance(sched, str) and "T00:00:00" in sched:
            sched = None  # placeholder midnight = unknown
        any_estimated = any(getattr(b, "dividend_estimated", False) for b in bets if b.settled)
        race_entry = {
            "race_id": race_id,
            "date": date,
            "venue": venue,
            "race_number": race_num,
            "scheduled_time": sched,
            "num_bets": len(bets),
            "total_stake": round(total_stake, 2),
            "total_payout": round(total_payout, 2) if (not any_unsettled and has_dividend) else None,
            "pnl": pnl,
            "dividend_estimated": any_estimated,
            "rank1_win_pct": round(rank1_pct[race_id], 1) if race_id in rank1_pct else None,
            "top3_sum_pct": round(top3_sum[race_id], 1) if race_id in top3_sum else None,
            "field_size": field_size.get(race_id),
            "status": status,
            "hits": sum(1 for b in bets if b.is_hit),
            "top3": top3_map.get(race_id) or None,
        }
        if include_bets:
            # Group bets by strategy_group for compact per-card rendering.
            # Decorate each bet with per-horse odds + a Harville-estimated
            # payout for hit boxes (gives the user a $ figure before any
            # actual TAB dividend is manually entered).
            bets_by_group: dict[str, list[dict]] = {}
            for b in bets:
                g = _STRATEGY_GROUP.get(b.strategy_label, "spread")
                bd = _row_to_bet_dict(b)
                bd["box_horse_odds"] = [
                    odds_lookup.get((race_id, tab)) for tab in (bd["box_horses"] or [])
                ]
                # If hit and we have odds for the actual top-3, estimate
                # the dividend via Harville on the market-implied probs.
                bd["estimated_payout"] = None
                if bd.get("is_hit") and bd.get("actual_top3"):
                    actual_tabs = bd["actual_top3"]
                    if len(actual_tabs) >= 3 and bd.get("bet_type") != "first_four":
                        actual_odds = [odds_lookup.get((race_id, t)) for t in actual_tabs[:3]]
                        if all(o and o > 1.0 for o in actual_odds):
                            probs = [1.0 / o for o in actual_odds]
                            est_div = _harville_dividend(probs, "trifecta")
                            if est_div and bd.get("num_permutations"):
                                stake = bd.get("stake_dollars") or 0
                                bd["estimated_dividend"] = est_div
                                bd["estimated_payout"] = round(
                                    stake / bd["num_permutations"] * est_div, 2
                                )
                bets_by_group.setdefault(g, []).append(bd)
            race_entry["bets_by_strategy"] = bets_by_group
        races.append(race_entry)
    # Sort: currently-running races (jumped but within ~8 min) lead,
    # then truly upcoming sorted by soonest jump, then past sorted by
    # most-recent first. Parse scheduled_time as a tz-aware datetime —
    # string compare against UTC.isoformat() would break because AEST
    # offsets push '12:30+10:00' lexically above the current UTC clock.
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    running_threshold = now_utc - timedelta(minutes=8)

    def _jump_dt(r):
        st = r.get("scheduled_time")
        if not st:
            return None
        try:
            return datetime.fromisoformat(str(st).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    running, upcoming, past = [], [], []
    for r in races:
        dt = _jump_dt(r)
        if dt is None:
            past.append(r)
        elif dt > now_utc:
            upcoming.append(r)
        elif dt > running_threshold:
            running.append(r)
        else:
            past.append(r)
    running.sort(key=lambda r: _jump_dt(r) or now_utc, reverse=True)
    upcoming.sort(key=lambda r: _jump_dt(r) or now_utc)
    past.sort(key=lambda r: _jump_dt(r) or now_utc, reverse=True)
    return {
        "days": days,
        "tier": feature_tier("lab"),        # surface tier — "labs"
        "races": running + upcoming + past,
    }


@app.post("/api/admin/bets/generate/{race_id}")
async def admin_generate_bets(
    race_id: str,
    regenerate: bool = False,
    x_cron_secret: Optional[str] = Header(None),
):
    """Manual trigger to (re)generate bet recommendations for one race."""
    _check_admin(x_cron_secret)
    n = await _generate_bets_for_race(race_id, regenerate=regenerate)
    return {"race_id": race_id, "inserted": n, "regenerate": regenerate}


@app.post("/api/admin/bets/settle")
async def admin_settle_bets(x_cron_secret: Optional[str] = Header(None)):
    """Fire the settlement sweep in the background and return immediately.
    The bulk sweep can take >60s (one RA call per pending date for the
    pre-seed step), so doing it synchronously trips Railway's request
    timeout. The work still happens — just not under this HTTP request."""
    _check_admin(x_cron_secret)
    asyncio.create_task(_scheduled_settle_bets())
    return {"ok": True, "queued": True}


@app.get("/api/bet-insights/strategy-stats")
async def get_strategy_stats(days: int = 14):
    """Per-strategy-group aggregate stats. Used by The Lab's selector
    pills and the Dashboard hit-stats table. Returns races bet, race
    hit rate, total stake, total payout (where dividend known), P&L
    and ROI per group. Cap raised to 365 days so the dashboard table
    can ask for 'all-time' in a single call."""
    days = max(1, min(int(days), 365))
    cutoff = datetime.utcnow() - timedelta(days=days)
    async with get_session() as session:
        rows = (await session.execute(
            select(BetRecommendationRow)
            .where(BetRecommendationRow.recommended_at >= cutoff)
        )).scalars().all()
    by_group: dict[str, list] = {}
    for r in rows:
        g = _STRATEGY_GROUP.get(r.strategy_label, "spread")
        by_group.setdefault(g, []).append(r)
    out: list[dict] = []
    for g_key, bets in by_group.items():
        # Exclude voided rows (boxes that contained a scratched horse —
        # real TAB would refund these) from hit/ROI calculations. They're
        # still tracked separately as `voided_bets`.
        valid_bets = [b for b in bets if not getattr(b, "voided", False)]
        voided_bets = [b for b in bets if getattr(b, "voided", False)]
        valid_race_ids = {b.race_id for b in valid_bets}
        races_with_hit = {b.race_id for b in valid_bets if b.is_hit}
        total_stake = round(sum((b.stake_dollars or 0) for b in valid_bets), 2)
        settled_with_div = [b for b in valid_bets if b.trifecta_dividend is not None]
        total_payout = round(sum((b.payout_dollars or 0) for b in settled_with_div), 2)
        priced_stake = round(sum((b.stake_dollars or 0) for b in settled_with_div), 2)
        roi_pct = round((total_payout - priced_stake) / priced_stake * 100, 1) if priced_stake else None
        out.append({
            "strategy": g_key,
            "label": _STRATEGY_GROUP_LABELS.get(g_key, g_key),
            "races_bet": len(valid_race_ids),
            "races_hit": len(races_with_hit),
            "race_hit_rate_pct": round(len(races_with_hit) / len(valid_race_ids) * 100, 1) if valid_race_ids else 0,
            "total_stake_dollars": total_stake,
            "total_payout_dollars": total_payout if priced_stake else None,
            "pnl_dollars": round(total_payout - priced_stake, 2) if priced_stake else None,
            "roi_pct": roi_pct,
            "settled_with_dividend": len(settled_with_div),
            "voided_bets": len(voided_bets),
        })
    # Stable ordering: spread first then sweep, then whatever else.
    order = {"spread": 0, "sweep": 1}
    out.sort(key=lambda s: order.get(s["strategy"], 99))
    return {"days": days, "strategies": out}


@app.get("/api/bet-insights/today")
async def get_today_picks(limit: int = 5):
    """Top picks for today, filtered to upcoming (not-yet-jumped) races."""
    return await _get_picks_for_date(
        for_date=_today_aest().isoformat(),
        upcoming_only=True,
        limit=limit,
    )


@app.get("/api/bet-insights/tomorrow")
async def get_tomorrow_picks(limit: int = 5):
    """Score tomorrow's races by predicted hit rate from the 60-day backtest."""
    return await _get_picks_for_date(
        for_date=(_today_aest() + timedelta(days=1)).isoformat(),
        upcoming_only=False,
        limit=limit,
    )


async def _get_picks_for_date(for_date: str, upcoming_only: bool, limit: int):
    """Shared implementation: profile today's settled paper-trading
    results, then score the candidate races for `for_date` against the
    60-day backtest hit-rate buckets. If upcoming_only=True, drop races
    whose scheduled_time has already passed."""
    limit = max(1, min(int(limit), 12))
    today_str = _today_aest().isoformat()
    target_date_str = for_date

    # ── Profile today: avg win1 / top3-sum for hit vs miss races ──
    hit_w1, hit_t3, miss_w1, miss_t3 = [], [], [], []
    today_settled = 0
    today_hit_races = 0
    async with get_session() as session:
        bet_rows = (await session.execute(
            select(BetRecommendationRow.race_id, BetRecommendationRow.is_hit,
                   BetRecommendationRow.settled)
            .where(BetRecommendationRow.race_id.like(f"{today_str}_%"))
            .where(BetRecommendationRow.settled.is_(True))
        )).fetchall()
        by_race: dict[str, list[bool]] = {}
        for rid, hit, _ in bet_rows:
            by_race.setdefault(rid, []).append(bool(hit))
        race_ids = list(by_race.keys())
        today_settled = len(race_ids)
        today_hit_races = sum(1 for hits in by_race.values() if any(hits))
        # Pull rank-1..3 win prob per race (history first, mutable fallback).
        if race_ids:
            for src in (RunnerPredictionHistoryRow, RunnerPredictionRow):
                runner_rows = (await session.execute(
                    select(src.race_id, src.model_rank, src.win_probability)
                    .where(src.race_id.in_(race_ids))
                    .where(src.model_rank.in_([1, 2, 3]))
                )).fetchall()
                top_map: dict[str, dict[int, float]] = {}
                for rid, rank, win_p in runner_rows:
                    if rank and win_p is not None:
                        top_map.setdefault(rid, {})[rank] = (win_p or 0) * 100
                for rid in race_ids:
                    tops = top_map.get(rid, {})
                    if 1 not in tops:
                        continue
                    w1 = tops[1]
                    t3 = sum(tops.get(i, 0) for i in (1, 2, 3))
                    if any(by_race[rid]):
                        hit_w1.append(w1); hit_t3.append(t3)
                    else:
                        miss_w1.append(w1); miss_t3.append(t3)
                if hit_w1 or miss_w1:
                    break  # got data from this source, no fallback needed

    def _avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else None
    profile = {
        "races_settled": today_settled,
        "hit_races": today_hit_races,
        "miss_races": today_settled - today_hit_races,
        "hit_avg_rank1_win_pct": _avg(hit_w1),
        "hit_avg_top3_sum_pct": _avg(hit_t3),
        "miss_avg_rank1_win_pct": _avg(miss_w1),
        "miss_avg_top3_sum_pct": _avg(miss_t3),
    }

    # ── Score tomorrow's races against historical hit rates ──
    # Hit-rate lookups derived from /api/admin/bets/backtest-formula run on
    # 60 days of data. We score each race by averaging the historical hit
    # rate for its bucket in each of (rank-1 win, field size, top-3 sum).
    def _rank1_hit_rate(w: float) -> float:
        if w < 20: return 0.0  # below filter threshold
        if 20 <= w < 25: return 34.3
        if 25 <= w < 30: return 0.0  # trap zone — recommender skips
        if 30 <= w < 40: return 40.0
        return 45.5  # 40%+
    def _field_hit_rate(n: int) -> float:
        if n <= 7: return 48.7
        if n <= 9: return 32.5
        if n <= 11: return 24.7
        if n <= 13: return 34.3
        return 22.2
    def _top3_hit_rate(s: float) -> float:
        if s < 40: return 10.0
        if s < 50: return 26.2
        if s < 60: return 36.6
        if s < 70: return 50.0
        return 43.8

    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    candidates: list[dict] = []
    async with get_session() as session:
        runner_rows = (await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id.like(f"{target_date_str}_%"))
            .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
        )).scalars().all()
        sched_lookup: dict[str, str] = {}
        venue_lookup: dict[str, str] = {}
        race_runners: dict[str, list] = {}
        for r in runner_rows:
            race_runners.setdefault(r.race_id, []).append(r)
            if r.scheduled_time and r.race_id not in sched_lookup:
                sched_lookup[r.race_id] = r.scheduled_time
            if r.venue and r.race_id not in venue_lookup:
                venue_lookup[r.race_id] = r.venue

    for rid, runners in race_runners.items():
        # Metro-only filter to match the recommender — only score races
        # we'd actually generate bets for.
        _, venue_code, _ = _parse_race_id(rid)
        if venue_code and not _is_metro_venue(venue_code):
            continue
        runners = sorted([x for x in runners if x.model_rank], key=lambda x: x.model_rank)
        if len(runners) < 7 or len(runners) > 13:
            continue
        # Upcoming-only filter for the today view — drop already-jumped races.
        if upcoming_only:
            st = sched_lookup.get(rid)
            if not st:
                continue
            try:
                if datetime.fromisoformat(str(st).replace("Z", "+00:00")) <= now_utc:
                    continue
            except (ValueError, TypeError):
                continue
        w1 = (runners[0].win_probability or 0) * 100
        if w1 < 20 or (25 <= w1 < 30):
            continue  # below threshold or in the trap zone
        w2 = (runners[1].win_probability or 0) * 100
        w3 = (runners[2].win_probability or 0) * 100
        top3_sum = w1 + w2 + w3
        # Combined predicted hit-rate score = average of three bucket lookups
        score = round((_rank1_hit_rate(w1) + _field_hit_rate(len(runners)) + _top3_hit_rate(top3_sum)) / 3, 1)
        _, vc, race_num = _parse_race_id(rid)
        candidates.append({
            "race_id": rid,
            "venue": venue_lookup.get(rid, vc),
            "race_number": race_num,
            "scheduled_time": sched_lookup.get(rid),
            "field_size": len(runners),
            "rank1_win_pct": round(w1, 1),
            "rank2_win_pct": round(w2, 1),
            "top3_sum_pct": round(top3_sum, 1),
            "predicted_hit_rate_pct": score,
            "score": score,
        })
    candidates.sort(key=lambda c: -c["score"])
    return {
        "today_profile": profile,
        "tomorrow_candidates": candidates[:limit],
        "tomorrow_analysed": len(candidates),
        "scoring_rule": (
            "Skips field<7 or >13, rank-1<20% or in 25-30% trap zone. "
            "Predicted hit rate = avg of historical (60-day backtest) hit "
            "rates for the race's rank-1 / field-size / top-3-sum buckets."
        ),
    }


@app.get("/api/admin/bets/backtest-formula")
async def backtest_formula(
    days: int = 60,
    x_cron_secret: Optional[str] = Header(None),
):
    """Backtest the 5-box trifecta basket + the 'open races' filter rule
    against the last N days of settled racing.

    For every race in the window where we have BOTH a prediction history
    snapshot AND top-3 historical results: reconstruct the 5 boxes (same
    algorithm the live recommender uses), check which boxes hit, then
    bucket the races by rank-1 win% and field size to surface where
    the strategy actually performs.

    Returns hit rate by rank-1 win bucket, by field size, and the
    overall counts. No dividend data is required."""
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 120))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        # All settled races in window: must have HistoricalResultRow for
        # top-3 AND must have RunnerPredictionHistoryRow for prediction
        # reconstruction.
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.tab_number, HistoricalResultRow.horse_name)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2, 3]))
        )).fetchall()
        results_by_race: dict[str, list] = {}
        for rid, pos, tab, name in result_rows:
            results_by_race.setdefault(rid, []).append((pos, tab, name))

        complete_race_ids = [rid for rid, items in results_by_race.items() if len(items) >= 3]
        # Pull prediction snapshots for these races (history rows — pre-race).
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(complete_race_ids))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
        )).scalars().all() if complete_race_ids else []

    # Group by race_id
    preds_by_race: dict[str, list] = {}
    for p in pred_rows:
        preds_by_race.setdefault(p.race_id, []).append(p)

    # Build tab-number lookup for tab-less results.
    tab_lookup_per_race: dict[str, dict[str, int]] = {}
    for rid, pruns in preds_by_race.items():
        tab_lookup_per_race[rid] = {
            _normalize_horse(p.horse_name): p.tab_number
            for p in pruns if p.tab_number is not None
        }

    summary = {
        "total_eligible_races": 0,
        "races_with_any_hit": 0,
        "boxes_evaluated": 0,
        "boxes_hit": 0,
        "by_rank1_bucket": {},   # "15-20%": {races, hits, hit_rate}
        "by_field_size": {},
        "by_top3_sum_bucket": {},
    }

    rank1_buckets = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 100)]
    field_buckets = [(0, 8), (8, 10), (10, 12), (12, 14), (14, 99)]
    top3_buckets  = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 100)]

    def _bucket_label(rng: tuple, val: float) -> str:
        return f"{rng[0]}-{rng[1]}%"

    for rid, pruns in preds_by_race.items():
        if len(pruns) < 7:
            continue
        runners = [{
            "tab_number": p.tab_number,
            "horse_name": p.horse_name,
            "win_probability": p.win_probability,
            "place_probability": p.place_probability,
            "model_rank": p.model_rank,
            "cancelled": bool(p.cancelled),
        } for p in pruns]
        bets = _build_bet_basket(runners)
        if not bets:
            continue
        # Resolve actual top-3 tab numbers (fall back to name lookup).
        items = sorted(results_by_race[rid], key=lambda x: x[0])[:3]
        actual_top3: list = []
        tab_lookup = tab_lookup_per_race.get(rid, {})
        for pos, tab, name in items:
            t = tab
            if t is None:
                t = tab_lookup.get(_normalize_horse(name))
            actual_top3.append(t)
        if any(t is None for t in actual_top3):
            continue

        race_hit_any = False
        for b in bets:
            summary["boxes_evaluated"] += 1
            if _bet_is_hit(b["box_horses"], actual_top3):
                summary["boxes_hit"] += 1
                race_hit_any = True
        summary["total_eligible_races"] += 1
        if race_hit_any:
            summary["races_with_any_hit"] += 1

        # Bucket race for analysis
        runners_sorted = sorted(pruns, key=lambda p: p.model_rank or 99)
        w1 = (runners_sorted[0].win_probability or 0) * 100
        t3 = sum((runners_sorted[i].win_probability or 0) * 100 for i in range(min(3, len(runners_sorted))))
        for lo, hi in rank1_buckets:
            if lo <= w1 < hi:
                k = _bucket_label((lo, hi), w1)
                d = summary["by_rank1_bucket"].setdefault(k, {"races": 0, "hits": 0})
                d["races"] += 1
                if race_hit_any: d["hits"] += 1
                break
        for lo, hi in field_buckets:
            fs = len(pruns)
            if lo <= fs < hi:
                k = f"{lo}-{hi-1}"
                d = summary["by_field_size"].setdefault(k, {"races": 0, "hits": 0})
                d["races"] += 1
                if race_hit_any: d["hits"] += 1
                break
        for lo, hi in top3_buckets:
            if lo <= t3 < hi:
                k = _bucket_label((lo, hi), t3)
                d = summary["by_top3_sum_bucket"].setdefault(k, {"races": 0, "hits": 0})
                d["races"] += 1
                if race_hit_any: d["hits"] += 1
                break

    # Compute hit rates
    for grouping in ("by_rank1_bucket", "by_field_size", "by_top3_sum_bucket"):
        for k, d in summary[grouping].items():
            d["hit_rate_pct"] = round(d["hits"] / d["races"] * 100, 1) if d["races"] else 0
    summary["overall_hit_rate_pct"] = (
        round(summary["races_with_any_hit"] / summary["total_eligible_races"] * 100, 1)
        if summary["total_eligible_races"] else 0
    )
    summary["window_days"] = days
    return summary


@app.get("/api/admin/bets/place-decisiveness-backtest")
async def place_decisiveness_backtest(
    days: int = 60,
    top3_max: float = 55.0,
    field_max: int = 11,
    box_type: str = "trio",  # 'trio' (3-horse box) or 'quad' (4-horse box)
    x_cron_secret: Optional[str] = Header(None),
):
    """Does requiring a clear place gap between rank-3 and rank-4 lift
    Lab Sharp's hit rate?

    For each historical race that already passes Lab Sharp (top-3 ≤55,
    field ≤11), compute rank3_place_pct minus rank4_place_pct. Bucket
    by gap size and report trifecta-box hit rate + estimated ROI per
    bucket. Identifies the gap threshold that maximises hit rate while
    keeping a usable sample size."""
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    # Box-type config — box_size = horses in the box, gap_low/high = which
    # ranks define the "decisiveness gap" we're testing. Trio compares
    # rank-3 vs rank-4 (just outside the 3-horse box); Quad compares
    # rank-4 vs rank-5 (just outside the 4-horse box).
    bt = (box_type or "trio").lower()
    if bt not in ("trio", "quad"):
        raise HTTPException(400, "box_type must be 'trio' or 'quad'")
    box_size = 3 if bt == "trio" else 4
    gap_low = box_size  # rank-3 for trio, rank-4 for quad
    gap_high = box_size + 1  # rank-4 for trio, rank-5 for quad
    needed_ranks = list(range(1, gap_high + 1))

    async with get_session() as session:
        # Pull every rank 1..N+1 prediction so we can compute the gap.
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.model_rank,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.place_probability,
                RunnerPredictionHistoryRow.tab_number,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank.in_(needed_ranks))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        # Top-3 results to score trifecta box hits.
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.tab_number, HistoricalResultRow.horse_name)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2, 3]))
        )).fetchall()
        # Field-size count per race for the ≤11 cutoff
        from sqlalchemy import func as _func
        field_size_rows = (await session.execute(
            select(RunnerPredictionHistoryRow.race_id,
                   _func.count(RunnerPredictionHistoryRow.id))
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .group_by(RunnerPredictionHistoryRow.race_id)
        )).fetchall()

    field_sizes = {rid: int(n) for rid, n in field_size_rows}

    # Group predictions by race, keep latest per (race, horse)
    seen: set = set()
    per_race: dict[str, dict] = {}
    for rid, name, rank, wp, pp, tab, enr in pred_rows:
        key = (rid, _normalize_horse(name))
        if key in seen:
            continue
        seen.add(key)
        per_race.setdefault(rid, {})[int(rank)] = {
            "horse_name": name,
            "win_prob": float(wp or 0),
            "place_prob": float(pp or 0),
            "tab": tab,
        }

    # Results lookup
    results_by_race: dict[str, set] = {}
    for rid, pos, tab, name in result_rows:
        if pos is None:
            continue
        results_by_race.setdefault(rid, set()).add(_normalize_horse(name or ""))

    # Evaluate each race
    bucket_defs = [
        ("<0pt (rank-4 ahead)", -999, 0),
        ("0-1pt (very flat)", 0, 1),
        ("1-2pt (flat)", 1, 2),
        ("2-3pt (mild gap)", 2, 3),
        ("3-4pt (decisive)", 3, 4),
        ("4-6pt (clear)", 4, 6),
        ("6pt+ (dominant top-3)", 6, 999),
    ]
    buckets = {b[0]: {"n": 0, "hits": 0} for b in bucket_defs}
    all_eligible = 0
    all_hits = 0

    for rid, ranks in per_race.items():
        if not all(r in ranks for r in needed_ranks):
            continue
        # Apply baseline Lab Sharp filter
        top3_sum = sum(ranks[r]["win_prob"] for r in (1, 2, 3)) * 100
        fs = field_sizes.get(rid, 999)
        if top3_sum > top3_max or fs > field_max:
            continue
        # Need a result to score
        winners = results_by_race.get(rid)
        if not winners or len(winners) < 3:
            continue
        # Box hit:
        #   Trio: model's top-3 are EXACTLY the actual top-3
        #   Quad: all 3 actual placings are within the model's top-4
        box_normed = {_normalize_horse(ranks[r]["horse_name"]) for r in range(1, box_size + 1)}
        hit = winners.issubset(box_normed)
        # Place gap = rank gap_low - rank gap_high (i.e. last horse in box vs first horse out)
        gap = (ranks[gap_low]["place_prob"] - ranks[gap_high]["place_prob"]) * 100
        all_eligible += 1
        if hit:
            all_hits += 1
        for label, lo, hi in bucket_defs:
            if lo <= gap < hi:
                buckets[label]["n"] += 1
                if hit:
                    buckets[label]["hits"] += 1
                break

    overall_hit_pct = round(all_hits / all_eligible * 100, 1) if all_eligible else 0
    out_buckets = []
    for label, lo, hi in bucket_defs:
        d = buckets[label]
        n = d["n"]
        hit_pct = round(d["hits"]/n*100, 1) if n else 0
        out_buckets.append({
            "bucket": label,
            "n": n,
            "hits": d["hits"],
            "hit_rate_pct": hit_pct,
            "lift_pts": round(hit_pct - overall_hit_pct, 1),
        })

    # Cumulative impact: what if we required gap ≥ X for various X?
    thresholds = []
    for gap_min_pts in (0, 1, 2, 3, 4, 5, 6):
        kept_n = 0
        kept_hits = 0
        for rid, ranks in per_race.items():
            if not all(r in ranks for r in needed_ranks):
                continue
            top3_sum = sum(ranks[r]["win_prob"] for r in (1, 2, 3)) * 100
            fs = field_sizes.get(rid, 999)
            if top3_sum > top3_max or fs > field_max:
                continue
            winners = results_by_race.get(rid)
            if not winners or len(winners) < 3:
                continue
            gap = (ranks[gap_low]["place_prob"] - ranks[gap_high]["place_prob"]) * 100
            if gap < gap_min_pts:
                continue
            kept_n += 1
            box_normed = {_normalize_horse(ranks[r]["horse_name"]) for r in range(1, box_size + 1)}
            if winners.issubset(box_normed):
                kept_hits += 1
        coverage = round(kept_n / all_eligible * 100, 1) if all_eligible else 0
        hit_pct = round(kept_hits / kept_n * 100, 1) if kept_n else 0
        thresholds.append({
            "gap_min_pts": gap_min_pts,
            "races_kept": kept_n,
            "hits": kept_hits,
            "hit_rate_pct": hit_pct,
            "coverage_of_lab_sharp_pct": coverage,
            "lift_vs_baseline_pts": round(hit_pct - overall_hit_pct, 1),
        })

    return {
        "days": days,
        "box_type": bt,
        "box_size": box_size,
        "gap_definition": f"rank-{gap_low} place_pct minus rank-{gap_high} place_pct",
        "lab_sharp_filter": f"top3 ≤{top3_max}% AND field ≤{field_max}",
        "baseline_eligible_races": all_eligible,
        "baseline_hits": all_hits,
        "baseline_hit_rate_pct": overall_hit_pct,
        "by_gap_bucket": out_buckets,
        "cumulative_threshold_test": thresholds,
    }


# ─── Nightly review endpoints ─────────────────────────────────────────────

async def _load_applied_pattern_ids(session, before_date: str) -> list[str]:
    """All pattern_ids previously marked 'applied' — fed to the analyser so
    it can soft-suppress repeat suggestions."""
    rows = (await session.execute(
        select(NightlyReviewRow.suggestions_json)
        .where(NightlyReviewRow.review_date < before_date)
    )).fetchall()
    applied: set = set()
    for r in rows:
        try:
            for s in json.loads(r.suggestions_json or "[]"):
                if s.get("status") == "applied" and s.get("pattern_id"):
                    applied.add(s["pattern_id"])
        except Exception:
            continue
    return sorted(applied)


async def _run_nightly_review(review_date: date, source: str = "python") -> dict:
    """Generate + persist a review row. Idempotent on (review_date, source)."""
    async with get_session() as session:
        applied = await _load_applied_pattern_ids(session, review_date.isoformat())
        result = await _generate_nightly_review(session, review_date, applied_pattern_ids=applied)

        existing = (await session.execute(
            select(NightlyReviewRow)
            .where(NightlyReviewRow.review_date == review_date.isoformat())
            .where(NightlyReviewRow.source == source)
        )).scalar_one_or_none()

        # Preserve per-suggestion status across re-runs: if a prior run already
        # had this pattern_id in 'assessed'/'applied'/'dismissed', carry the
        # status + notes forward rather than resetting to pending.
        if existing:
            try:
                prior = {s["pattern_id"]: s for s in json.loads(existing.suggestions_json or "[]")}
            except Exception:
                prior = {}
            for s in result["suggestions"]:
                p = prior.get(s["pattern_id"])
                if p and p.get("status") in ("assessed", "applied", "dismissed"):
                    s["status"] = p["status"]
                    s["notes"] = p.get("notes", "")
                    s["applied_at"] = p.get("applied_at")
                    s["dismissed_reason"] = p.get("dismissed_reason")
            existing.summary_markdown = result["summary_markdown"]
            existing.suggestions_json = json.dumps(result["suggestions"])
            existing.headline_stats_json = json.dumps(result["headline_stats"])
            existing.updated_at = datetime.utcnow()
        else:
            session.add(NightlyReviewRow(
                review_date=review_date.isoformat(),
                source=source,
                summary_markdown=result["summary_markdown"],
                suggestions_json=json.dumps(result["suggestions"]),
                headline_stats_json=json.dumps(result["headline_stats"]),
            ))
        await session.commit()
    return result


async def _scheduled_nightly_review():
    """APScheduler job — review yesterday at 02:00 AEST after all racing settled."""
    yesterday = _today_aest() - timedelta(days=1)
    log.info("[scheduler] Running nightly review for %s", yesterday.isoformat())
    try:
        result = await _run_nightly_review(yesterday, source="python")
        log.info("[scheduler] Nightly review wrote %d suggestions", len(result["suggestions"]))
    except Exception as e:
        log.exception("[scheduler] Nightly review failed: %s", e)


async def _run_weekly_review(end_date: date) -> dict:
    """Generate + persist a weekly review row. Idempotent on review_date+source='weekly'."""
    async with get_session() as session:
        applied = await _load_applied_pattern_ids(session, end_date.isoformat())
        result = await _generate_weekly_review(session, end_date, applied_pattern_ids=applied)

        existing = (await session.execute(
            select(NightlyReviewRow)
            .where(NightlyReviewRow.review_date == end_date.isoformat())
            .where(NightlyReviewRow.source == "weekly")
        )).scalar_one_or_none()

        # Preserve per-suggestion triage state across re-runs (same as
        # nightly).
        if existing:
            try:
                prior = {s["pattern_id"]: s for s in json.loads(existing.suggestions_json or "[]")}
            except Exception:
                prior = {}
            for s in result["suggestions"]:
                p = prior.get(s["pattern_id"])
                if p and p.get("status") in ("assessed", "applied", "dismissed"):
                    s["status"] = p["status"]
                    s["notes"] = p.get("notes", "")
                    s["applied_at"] = p.get("applied_at")
                    s["dismissed_reason"] = p.get("dismissed_reason")
            existing.summary_markdown = result["summary_markdown"]
            existing.suggestions_json = json.dumps(result["suggestions"])
            existing.headline_stats_json = json.dumps(result["headline_stats"])
            existing.updated_at = datetime.utcnow()
        else:
            session.add(NightlyReviewRow(
                review_date=end_date.isoformat(),
                source="weekly",
                summary_markdown=result["summary_markdown"],
                suggestions_json=json.dumps(result["suggestions"]),
                headline_stats_json=json.dumps(result["headline_stats"]),
            ))
        await session.commit()
    return result


async def _scheduled_weekly_review():
    """APScheduler job — review the past 7 days every Monday at 03:00 AEST.
    end_date is set to yesterday (= last Sunday), so the review window is
    Mon→Sun ending the day before the cron fires."""
    end_date = _today_aest() - timedelta(days=1)
    log.info("[scheduler] Running weekly review for week ending %s", end_date.isoformat())
    try:
        result = await _run_weekly_review(end_date)
        log.info("[scheduler] Weekly review wrote %d suggestions", len(result["suggestions"]))
    except Exception as e:
        log.exception("[scheduler] Weekly review failed: %s", e)


@app.post("/api/admin/weekly-reviews/generate")
async def generate_weekly_review_endpoint(
    end_date: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """Run the weekly analyser now. `end_date` defaults to yesterday
    (= last Sunday if fired Monday morning). Idempotent — merging into
    an existing weekly row preserves prior triage state."""
    _check_admin(x_cron_secret)
    if end_date:
        _validate_date(end_date)
        target = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        target = _today_aest() - timedelta(days=1)
    result = await _run_weekly_review(target)
    return {
        "review_date": target.isoformat(),
        "source": "weekly",
        "summary_markdown": result["summary_markdown"],
        "headline_stats": result["headline_stats"],
        "suggestions_count": len(result["suggestions"]),
    }


@app.post("/api/admin/nightly-reviews/generate")
async def generate_nightly_review(
    date: Optional[str] = None,
    source: str = "python",
    x_cron_secret: Optional[str] = Header(None),
):
    """Run the analyser now for the given date (defaults to yesterday).
    Idempotent — re-running merges into the same row and preserves the
    status of any suggestions already triaged."""
    _check_admin(x_cron_secret)
    if source not in ("python", "claude", "weekly"):
        raise HTTPException(400, "source must be 'python', 'claude', or 'weekly'")
    if date:
        _validate_date(date)
        review_date = datetime.strptime(date, "%Y-%m-%d").date()
    else:
        review_date = _today_aest() - timedelta(days=1)
    result = await _run_nightly_review(review_date, source=source)
    return {
        "review_date": review_date.isoformat(),
        "source": source,
        "summary_markdown": result["summary_markdown"],
        "headline_stats": result["headline_stats"],
        "suggestions_count": len(result["suggestions"]),
    }


@app.get("/api/admin/nightly-reviews")
async def list_nightly_reviews(
    limit: int = 60,
    status_filter: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """List recent reviews (one row per date+source). If status_filter is
    set, only return reviews containing at least one suggestion with that
    status — handy for filtering 'show me everything pending'."""
    _check_admin(x_cron_secret)
    limit = max(1, min(int(limit), 365))
    async with get_session() as session:
        rows = (await session.execute(
            select(NightlyReviewRow)
            .order_by(NightlyReviewRow.review_date.desc(), NightlyReviewRow.source.asc())
            .limit(limit * 2)  # × 2 because we filter post-fetch
        )).scalars().all()

    out = []
    for r in rows:
        try:
            suggestions = json.loads(r.suggestions_json or "[]")
        except Exception:
            suggestions = []
        if status_filter and not any(s.get("status") == status_filter for s in suggestions):
            continue
        status_counts: dict[str, int] = {}
        for s in suggestions:
            status_counts[s.get("status", "pending")] = status_counts.get(s.get("status", "pending"), 0) + 1
        try:
            headline = json.loads(r.headline_stats_json or "{}")
        except Exception:
            headline = {}
        out.append({
            "review_date": r.review_date,
            "source": r.source,
            "suggestion_count": len(suggestions),
            "status_counts": status_counts,
            "headline_stats": headline,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        })
        if len(out) >= limit:
            break
    return {"reviews": out}


@app.get("/api/admin/nightly-reviews/{review_date}")
async def get_nightly_review(
    review_date: str,
    source: str = "python",
    x_cron_secret: Optional[str] = Header(None),
):
    """Full content of one review — markdown summary + every suggestion
    with its current status, notes, paste-prompt, etc."""
    _check_admin(x_cron_secret)
    _validate_date(review_date)
    if source not in ("python", "claude", "weekly"):
        raise HTTPException(400, "source must be 'python', 'claude', or 'weekly'")
    async with get_session() as session:
        row = (await session.execute(
            select(NightlyReviewRow)
            .where(NightlyReviewRow.review_date == review_date)
            .where(NightlyReviewRow.source == source)
        )).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "No review for that date+source")
    try:
        suggestions = json.loads(row.suggestions_json or "[]")
    except Exception:
        suggestions = []
    try:
        headline = json.loads(row.headline_stats_json or "{}")
    except Exception:
        headline = {}
    return {
        "review_date": row.review_date,
        "source": row.source,
        "summary_markdown": row.summary_markdown,
        "headline_stats": headline,
        "suggestions": suggestions,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


class _SuggestionPatch(BaseModel):
    status: Optional[str] = None  # 'pending' | 'assessed' | 'applied' | 'dismissed'
    notes: Optional[str] = None
    dismissed_reason: Optional[str] = None


@app.patch("/api/admin/nightly-reviews/{review_date}/suggestions/{suggestion_id}")
async def patch_nightly_suggestion(
    review_date: str,
    suggestion_id: str,
    patch: _SuggestionPatch,
    source: str = "python",
    x_cron_secret: Optional[str] = Header(None),
):
    """Update the status / notes of a single suggestion. Sets applied_at
    automatically when transitioning into 'applied'."""
    _check_admin(x_cron_secret)
    _validate_date(review_date)
    if source not in ("python", "claude", "weekly"):
        raise HTTPException(400, "source must be 'python', 'claude', or 'weekly'")
    if patch.status and patch.status not in ("pending", "assessed", "applied", "dismissed"):
        raise HTTPException(400, "invalid status")

    async with get_session() as session:
        row = (await session.execute(
            select(NightlyReviewRow)
            .where(NightlyReviewRow.review_date == review_date)
            .where(NightlyReviewRow.source == source)
        )).scalar_one_or_none()
        if not row:
            raise HTTPException(404, "review not found")
        try:
            suggestions = json.loads(row.suggestions_json or "[]")
        except Exception:
            suggestions = []
        target = None
        for s in suggestions:
            if s.get("id") == suggestion_id:
                target = s
                break
        if target is None:
            raise HTTPException(404, "suggestion not found in review")
        if patch.status is not None:
            target["status"] = patch.status
            if patch.status == "applied" and not target.get("applied_at"):
                target["applied_at"] = datetime.utcnow().isoformat()
            if patch.status != "applied":
                target["applied_at"] = None
        if patch.notes is not None:
            target["notes"] = patch.notes
        if patch.dismissed_reason is not None:
            target["dismissed_reason"] = patch.dismissed_reason
        row.suggestions_json = json.dumps(suggestions)
        row.updated_at = datetime.utcnow()
        await session.commit()
    return {"ok": True, "suggestion": target}


class _ClaudeSuggestion(BaseModel):
    pattern_id: str
    title: str
    severity: str = "medium"  # 'high' | 'medium' | 'low'
    rationale: str
    paste_prompt: str = ""
    code_pointer: str = ""
    evidence: list[dict] = []


class _ClaudeReviewSubmit(BaseModel):
    review_date: str           # YYYY-MM-DD
    summary_markdown: str
    headline_stats: dict = {}
    suggestions: list[_ClaudeSuggestion] = []


@app.post("/api/admin/nightly-reviews/submit")
async def submit_claude_nightly_review(
    payload: _ClaudeReviewSubmit,
    x_cron_secret: Optional[str] = Header(None),
):
    """Submission endpoint for the weekly Claude agent. The agent does its
    own analysis (against /api/admin/model-anomalies, /api/admin/bets/*
    etc.) and POSTs a structured review here. Each suggestion is assigned
    a stable id of `{pattern_id}__{review_date}` so the dashboard can
    dedupe across runs. Re-submitting for the same date merges and
    preserves status (same as the Python /generate endpoint)."""
    _check_admin(x_cron_secret)
    _validate_date(payload.review_date)
    for s in payload.suggestions:
        if s.severity not in ("high", "medium", "low"):
            raise HTTPException(400, f"bad severity '{s.severity}' on {s.pattern_id}")

    new_suggestions = [{
        "id": f"{s.pattern_id}__{payload.review_date}",
        "pattern_id": s.pattern_id,
        "title": s.title,
        "severity": s.severity,
        "rationale": s.rationale,
        "evidence": s.evidence,
        "paste_prompt": s.paste_prompt,
        "code_pointer": s.code_pointer,
        "metric_delta": {},
        "status": "pending",
        "notes": "",
        "applied_at": None,
        "dismissed_reason": None,
    } for s in payload.suggestions]

    async with get_session() as session:
        existing = (await session.execute(
            select(NightlyReviewRow)
            .where(NightlyReviewRow.review_date == payload.review_date)
            .where(NightlyReviewRow.source == "claude")
        )).scalar_one_or_none()

        if existing:
            try:
                prior = {s["pattern_id"]: s for s in json.loads(existing.suggestions_json or "[]")}
            except Exception:
                prior = {}
            for s in new_suggestions:
                p = prior.get(s["pattern_id"])
                if p and p.get("status") in ("assessed", "applied", "dismissed"):
                    s["status"] = p["status"]
                    s["notes"] = p.get("notes", "")
                    s["applied_at"] = p.get("applied_at")
                    s["dismissed_reason"] = p.get("dismissed_reason")
            existing.summary_markdown = payload.summary_markdown
            existing.suggestions_json = json.dumps(new_suggestions)
            existing.headline_stats_json = json.dumps(payload.headline_stats)
            existing.updated_at = datetime.utcnow()
        else:
            session.add(NightlyReviewRow(
                review_date=payload.review_date,
                source="claude",
                summary_markdown=payload.summary_markdown,
                suggestions_json=json.dumps(new_suggestions),
                headline_stats_json=json.dumps(payload.headline_stats),
            ))
        await session.commit()
    return {"ok": True, "review_date": payload.review_date, "source": "claude",
            "suggestions_persisted": len(new_suggestions)}


@app.get("/api/admin/nightly-reviews/applied/history")
async def applied_suggestions_history(
    limit: int = 100,
    pattern_id: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """Every suggestion ever marked 'applied' — fed to the weekly Claude
    agent so it can see what's already shipped. Optional pattern_id filter."""
    _check_admin(x_cron_secret)
    limit = max(1, min(int(limit), 500))
    async with get_session() as session:
        rows = (await session.execute(
            select(NightlyReviewRow)
            .order_by(NightlyReviewRow.review_date.desc())
            .limit(365)
        )).scalars().all()

    applied: list[dict] = []
    for r in rows:
        try:
            suggestions = json.loads(r.suggestions_json or "[]")
        except Exception:
            continue
        for s in suggestions:
            if s.get("status") != "applied":
                continue
            if pattern_id and s.get("pattern_id") != pattern_id:
                continue
            applied.append({
                "review_date": r.review_date,
                "source": r.source,
                "pattern_id": s.get("pattern_id"),
                "title": s.get("title"),
                "applied_at": s.get("applied_at"),
                "notes": s.get("notes", ""),
                "code_pointer": s.get("code_pointer", ""),
            })
            if len(applied) >= limit:
                break
        if len(applied) >= limit:
            break
    return {"applied": applied}


# ─── /Nightly review endpoints ────────────────────────────────────────────


@app.get("/api/admin/probe-tab-race")
async def probe_tab_race(race_id: str, x_cron_secret: Optional[str] = Header(None)):
    """Dump raw TAB response for a race — checks whether TAB's
    finishingPosition (or a similar field) is populated after the race
    settles. Prep work for a potential TAB-based results seeder as a
    backup / replacement for the RA HTML scrape."""
    _check_admin(x_cron_secret)
    parts = race_id.split("_")
    if len(parts) < 3:
        raise HTTPException(400, "bad race_id")
    race_date, venue_code = parts[0], parts[1]
    try:
        race_num = int(parts[2].replace("R", ""))
    except ValueError:
        raise HTTPException(400, "bad race number")
    client = get_tab_client()
    slug = _meeting_slug(venue_code, race_date)
    raw = await client.get_race(slug, race_num)
    if not raw:
        return {"race_id": race_id, "raw": None, "note": "TAB returned no event"}
    top_level = {k: v for k, v in raw.items() if k != "runners" and not isinstance(v, list)}
    runners = raw.get("runners") or []
    trimmed_runners = []
    for r in runners:
        if not isinstance(r, dict):
            continue
        # Keep every field on the runner — helps spot position under a
        # non-obvious name (e.g. inside "results", "meta", "outcome").
        trimmed = {}
        for k, v in r.items():
            if isinstance(v, dict) or k == "prices":
                trimmed[k] = v
            elif not isinstance(v, list):
                trimmed[k] = v
        trimmed_runners.append(trimmed)
    return {
        "race_id": race_id,
        "slug": slug,
        "top_level_keys": sorted(top_level.keys()),
        "top_level_values": top_level,
        "runner_keys_sample": sorted(runners[0].keys()) if runners else [],
        "runner_count": len(runners),
        "first_3_runners": trimmed_runners[:3],
    }


@app.post("/api/admin/backfill-is-sharp")
async def admin_backfill_is_sharp(x_cron_secret: Optional[str] = Header(None)):
    """Manual trigger for the is_sharp backfill on rank-1 rows where the
    flag is NULL. Uses the ORIGINAL Sharp gate (rank1_pct ≥30 OR top3_sum
    ≥60), NOT the days_off variant — so past dates lock in with their
    contemporary Sharp definition and can't be retroactively shifted by
    future gate refinements.

    Returns the row count updated for each stage. Idempotent — a re-run
    is a no-op once every rank-1 row has is_sharp set to True or False."""
    _check_admin(x_cron_secret)
    from sqlalchemy import text as _sa_text
    stages = [
        ("stage_1_win_prob_ge_30", """
            UPDATE runner_prediction_history
            SET is_sharp = TRUE
            WHERE model_rank = 1
              AND is_sharp IS NULL
              AND win_probability >= 0.30
        """),
        ("stage_2_top3_sum_ge_60", """
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
        """),
        ("stage_3_remaining_false", """
            UPDATE runner_prediction_history
            SET is_sharp = FALSE
            WHERE model_rank = 1 AND is_sharp IS NULL
        """),
    ]
    results = {}
    for label, stmt in stages:
        try:
            async with get_session() as session:
                res = await session.execute(_sa_text(stmt))
                await session.commit()
                results[label] = res.rowcount
        except Exception as e:
            results[label] = f"error: {e}"
    # Bust perf cache so the numbers show up immediately
    _PERF_CACHE.clear()
    return {"ok": True, "rows_updated": results, "perf_cache": "cleared"}


@app.get("/api/admin/debug-bet-generator")
async def debug_bet_generator(
    race_id: str,
    x_cron_secret: Optional[str] = Header(None),
):
    """Mirror `_generate_bets_for_race` step by step and return what
    each gate sees. Lets us tell exactly why bet generation produces
    zero rows for a race."""
    _check_admin(x_cron_secret)
    async with get_session() as session:
        rows = (await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id == race_id)
        )).scalars().all()
        source = "mutable"
        if not rows:
            rows = (await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.race_id == race_id)
            )).scalars().all()
            source = "history"

    if not rows:
        return {"race_id": race_id, "source": None, "row_count": 0,
                "note": "no rows in either mutable or history"}

    prize_money = max((getattr(r, "prize_money", None) or 0) for r in rows)

    runners = [{
        "tab_number": getattr(r, "tab_number", None),
        "horse_name": r.horse_name,
        "win_probability": r.win_probability,
        "place_probability": r.place_probability,
        "model_rank": r.model_rank,
        "cancelled": bool(r.cancelled),
    } for r in rows]

    active = [r for r in runners if not r.get("cancelled")]
    active_with_tab_and_rank = [
        r for r in active
        if r.get("model_rank") is not None and r.get("tab_number") is not None
    ]
    bets = _build_bet_basket(runners) or []

    return {
        "race_id": race_id,
        "source": source,
        "row_count": len(rows),
        "prize_money_max": prize_money,
        "active_count": len(active),
        "active_with_tab_and_rank": len(active_with_tab_and_rank),
        "top1_win_prob": active_with_tab_and_rank[0]["win_probability"] if active_with_tab_and_rank else None,
        "top1_win_pct": (active_with_tab_and_rank[0]["win_probability"] * 100) if active_with_tab_and_rank else None,
        "bet_basket_count": len(bets),
        "bet_basket_labels": [b["strategy_label"] for b in bets],
        "first_3_runners": runners[:3],
        "all_runners": runners,
    }


@app.get("/api/admin/inspect-historical-results")
async def admin_inspect_historical_results(
    race_id: str,
    x_cron_secret: Optional[str] = Header(None),
):
    """Dump every HistoricalResultRow for one race_id. Debug aid."""
    _check_admin(x_cron_secret)
    async with get_session() as session:
        rows = (await session.execute(
            select(HistoricalResultRow.position, HistoricalResultRow.horse_name,
                   HistoricalResultRow.tab_number, HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id == race_id)
        )).fetchall()
    return {
        "race_id": race_id,
        "row_count": len(rows),
        "rows": [{"position": r.position, "horse": r.horse_name,
                  "tab": r.tab_number, "sp": float(r.starting_price or 0)} for r in rows],
        "top3_count": sum(1 for r in rows if r.position in (1, 2, 3)),
    }


@app.post("/api/admin/bets/void-stale-noresult")
async def admin_void_stale_noresult(
    days_back: int = 2,
    x_cron_secret: Optional[str] = Header(None),
):
    """One-shot void for any unsettled bet whose race is >=days_back days
    old AND has <3 top-finishers in historical_results. Returns the
    voided count synchronously — unlike /api/admin/bets/settle which
    fires asyncio.create_task and swallows errors at log.debug level."""
    _check_admin(x_cron_secret)
    days_back = max(1, min(int(days_back), 30))
    cutoff_date = (_today_aest() - timedelta(days=days_back)).isoformat()
    voided_per_race: dict[str, int] = {}
    examined = 0
    errors = []
    async with get_session() as session:
        # Distinct unsettled race_ids whose calendar date is <= cutoff.
        unsettled_race_ids = (await session.execute(
            select(BetRecommendationRow.race_id).distinct()
            .where(BetRecommendationRow.settled.is_(False))
            .where(BetRecommendationRow.race_id <= f"{cutoff_date}_￿")
        )).scalars().all()
        examined = len(unsettled_race_ids)

        for rid in unsettled_race_ids:
            try:
                top3 = (await session.execute(
                    select(HistoricalResultRow.position)
                    .where(HistoricalResultRow.race_id == rid)
                    .where(HistoricalResultRow.position.in_([1, 2, 3]))
                )).fetchall()
                if len(top3) >= 3:
                    continue  # has results — normal settle should handle
                from sqlalchemy import update as sa_update
                upd = await session.execute(
                    sa_update(BetRecommendationRow)
                    .where(BetRecommendationRow.race_id == rid)
                    .where(BetRecommendationRow.settled.is_(False))
                    .values(
                        settled=True,
                        voided=True,
                        is_hit=False,
                        settled_at=datetime.utcnow(),
                    )
                )
                if upd.rowcount:
                    voided_per_race[rid] = upd.rowcount
            except Exception as e:
                errors.append(f"{rid}: {type(e).__name__}: {e}")
        await session.commit()
    return {
        "examined_race_ids": examined,
        "voided_race_ids": len(voided_per_race),
        "voided_rows_total": sum(voided_per_race.values()),
        "voided_per_race": voided_per_race,
        "errors": errors,
    }


@app.get("/api/admin/horse-history")
async def admin_horse_history(
    name: str,
    days: int = 90,
    x_cron_secret: Optional[str] = Header(None),
):
    """Every prediction we made for one horse over the last N days, joined
    with the actual result. Country-code-suffix-insensitive name match
    (so 'DAKOTA LEE' matches 'DAKOTA LEE (NZ)'). Use when checking how
    accurate the model has been on a specific runner."""
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()
    norm_target = _normalize_horse(name)

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.place_probability,
                RunnerPredictionHistoryRow.model_rank,
                RunnerPredictionHistoryRow.market_rank,
                RunnerPredictionHistoryRow.enriched_at,
                RunnerPredictionHistoryRow.scheduled_time,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.race_id.desc(),
                      RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()

        # Filter by normalised name in Python — country-code suffixes vary.
        matched = [r for r in pred_rows if _normalize_horse(r.horse_name) == norm_target]
        # Dedupe to latest snapshot per race.
        seen: set = set()
        latest = []
        for r in matched:
            if r.race_id in seen:
                continue
            seen.add(r.race_id)
            latest.append(r)
        if not latest:
            return {"horse_name": name, "days": days, "races": [], "summary": None}

        race_ids = [r.race_id for r in latest]
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
                   HistoricalResultRow.position, HistoricalResultRow.starting_price,
                   HistoricalResultRow.field_size)
            .where(HistoricalResultRow.race_id.in_(race_ids))
        )).fetchall()

    results_by_race: dict[str, dict] = {}
    for r in result_rows:
        if _normalize_horse(r.horse_name or "") == norm_target:
            results_by_race[r.race_id] = {
                "position": r.position,
                "starting_price": float(r.starting_price or 0),
                "field_size": r.field_size,
            }

    out_races = []
    rank1_picks = 0
    rank1_wins = 0
    rank1_placed = 0
    rank1_with_result = 0
    any_wins = 0
    any_placed = 0
    any_with_result = 0
    for r in latest:
        actual = results_by_race.get(r.race_id, {})
        pos = actual.get("position")
        actually_won = pos == 1
        actually_placed = bool(pos and pos <= 3)
        out_races.append({
            "race_id": r.race_id,
            "model_pct": round((r.win_probability or 0) * 100, 1),
            "model_rank": r.model_rank,
            "market_rank": r.market_rank,
            "position": pos,
            "starting_price": actual.get("starting_price") or None,
            "field_size": actual.get("field_size"),
            "won": actually_won,
            "placed": actually_placed,
            "scheduled_time": r.scheduled_time,
        })
        if pos is not None and pos > 0:
            any_with_result += 1
            if actually_won:
                any_wins += 1
            if actually_placed:
                any_placed += 1
            if r.model_rank == 1:
                rank1_with_result += 1
                if actually_won:
                    rank1_wins += 1
                if actually_placed:
                    rank1_placed += 1
        if r.model_rank == 1:
            rank1_picks += 1

    return {
        "horse_name": name,
        "days": days,
        "races": out_races,
        "summary": {
            "total_races_predicted": len(out_races),
            "rank1_picks": rank1_picks,
            "rank1_with_result": rank1_with_result,
            "rank1_wins": rank1_wins,
            "rank1_placed": rank1_placed,
            "rank1_win_pct": round(rank1_wins / rank1_with_result * 100, 1) if rank1_with_result else None,
            "rank1_place_pct": round(rank1_placed / rank1_with_result * 100, 1) if rank1_with_result else None,
            "any_with_result": any_with_result,
            "any_wins": any_wins,
            "any_placed": any_placed,
        },
    }


@app.get("/api/admin/model-anomalies")
async def model_anomalies(
    days: int = 90,
    min_model_pct: float = 25.0,
    min_sp: float = 8.0,
    max_position: int = 5,
    limit: int = 50,
    include_backfill: bool = False,
    x_cron_secret: Optional[str] = Header(None),
):
    """Find races where:
      - model gave the rank-1 horse a high win% (≥ min_model_pct)
      - market priced the same horse long (SP ≥ min_sp — market disagreed)
      - horse finished poorly (position > max_position OR scratched-by-DNF)

    These are the worst-case 'model very wrong' picks. For each we surface
    the diagnostic features from the enriched snapshot so failure modes
    can be classified (long layoff, market-rank gap, weak form proxy, etc.).
    """
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.place_probability,
                RunnerPredictionHistoryRow.market_rank,
                RunnerPredictionHistoryRow.tab_number,
                RunnerPredictionHistoryRow.scheduled_time,
                RunnerPredictionHistoryRow.enriched_json,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.win_probability >= min_model_pct / 100.0)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill"))
                   if include_backfill else RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
                   HistoricalResultRow.position, HistoricalResultRow.starting_price,
                   HistoricalResultRow.field_size)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
        )).fetchall()

    # Latest snapshot per (race, horse) — picks list is reverse-chron-ordered
    seen: set = set()
    latest_picks = []
    for row in pred_rows:
        key = (row.race_id, _normalize_horse(row.horse_name))
        if key in seen:
            continue
        seen.add(key)
        latest_picks.append(row)

    # Index results: (race_id, horse_name_normed) → (position, sp, field_size)
    results_by_horse: dict[tuple[str, str], tuple] = {}
    for r in result_rows:
        results_by_horse[(r.race_id, _normalize_horse(r.horse_name or ""))] = (
            r.position, float(r.starting_price or 0), r.field_size,
        )

    anomalies: list[dict] = []
    for p in latest_picks:
        key = (p.race_id, _normalize_horse(p.horse_name))
        result = results_by_horse.get(key)
        if not result:
            continue
        pos, sp, field_size = result
        if pos is None or pos <= 0:
            continue  # didn't finish — DNF/scratched after enrich
        if sp < min_sp:
            continue  # market backed it too — not an anomaly
        if pos <= max_position:
            continue  # actually ran well enough — not an anomaly
        try:
            enriched = json.loads(p.enriched_json) if p.enriched_json else {}
        except Exception:
            enriched = {}
        model_pct = (p.win_probability or 0) * 100
        market_implied_pct = (1.0 / sp) * 100 if sp > 1 else None
        edge_pct = round(model_pct - market_implied_pct, 1) if market_implied_pct else None
        market_rank = p.market_rank
        days_off = enriched.get("days_since_last_run")
        form_score = enriched.get("form_score")
        # Failure-mode tags — heuristic; multiple may fire per anomaly
        tags = []
        if isinstance(days_off, (int, float)) and days_off > 180:
            tags.append("long_layoff")
        if isinstance(days_off, (int, float)) and days_off > 365:
            tags.append("year_plus_spell")
        if market_rank and market_rank >= 5:
            tags.append("market_disagreed")
        if isinstance(form_score, (int, float)) and form_score >= 0.75 and isinstance(days_off, (int, float)) and days_off > 120:
            tags.append("stale_form_signal")
        if enriched.get("class_change") == 1 and isinstance(days_off, (int, float)) and days_off > 120:
            tags.append("class_drop_post_spell")
        if (enriched.get("runs_this_prep") or 0) == 0:
            tags.append("first_up_resuming")
        if (enriched.get("starts_last_10") or 0) <= 2:
            tags.append("thin_record")
        if not tags:
            tags.append("no_obvious_cause")
        anomalies.append({
            "race_id": p.race_id,
            "horse_name": p.horse_name,
            "model_pct": round(model_pct, 1),
            "starting_price": sp,
            "market_implied_pct": round(market_implied_pct, 1) if market_implied_pct else None,
            "edge_pct": edge_pct,
            "model_rank": 1,
            "market_rank": market_rank,
            "actual_position": pos,
            "field_size": field_size,
            "form_score": form_score,
            "days_since_last_run": days_off,
            "class_change": enriched.get("class_change"),
            "runs_this_prep": enriched.get("runs_this_prep"),
            "wins_last_10": enriched.get("wins_last_10"),
            "starts_last_10": enriched.get("starts_last_10"),
            "speed_map_position": enriched.get("speed_map_position"),
            "trainer_overall_rate": enriched.get("trainer_overall_rate"),
            "jockey_overall_rate": enriched.get("jockey_overall_rate"),
            "win_rate_distance": enriched.get("win_rate_distance"),
            "wet_track_record": enriched.get("wet_track_record"),
            "key_flags": enriched.get("key_flags"),
            "scheduled_time": p.scheduled_time,
            "tags": tags,
        })

    # Sort by worst-edge first (biggest model-market disagreement)
    anomalies.sort(key=lambda a: -(a.get("edge_pct") or 0))

    # Tag frequency summary so we can see dominant failure modes
    tag_counts: dict[str, int] = {}
    for a in anomalies:
        for t in a["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    return {
        "window_days": days,
        "filters": {
            "min_model_pct": min_model_pct,
            "min_sp": min_sp,
            "max_position": max_position,
        },
        "total_anomalies": len(anomalies),
        "failure_mode_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
        "anomalies": anomalies[:limit],
    }


@app.get("/api/admin/bets/going-calibration")
async def going_calibration(
    days: int = 90,
    include_backfill: bool = True,
    x_cron_secret: Optional[str] = Header(None),
):
    """Per-going calibration test for rank-1 picks. Bins by model_pct
    and reports actual win rate per bin per going category. If 'soft'
    horses with 30%% model_pct actually win 15%% of the time, the model
    is over-confident on soft tracks by ~2x and we should apply a
    going-specific multiplier.

    Returns a per-bin × per-going table plus a recommended multiplier
    per going (= overall actual / actual on this going).
    """
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.enriched_json,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill"))
                   if include_backfill else RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
                   HistoricalResultRow.position, HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position == 1)
        )).fetchall()

    seen: set = set()
    picks: list[dict] = []
    for r in pred_rows:
        key = (r.race_id, _normalize_horse(r.horse_name))
        if key in seen:
            continue
        seen.add(key)
        try:
            enr = json.loads(r.enriched_json) if r.enriched_json else {}
        except Exception:
            enr = {}
        picks.append({
            "race_id": r.race_id,
            "horse_name": _normalize_horse(r.horse_name),
            "win_prob": float(r.win_probability or 0),
            "enr": enr,
        })

    winners: dict[str, str] = {}
    sp_by_race: dict[str, float] = {}
    for r in result_rows:
        winners[r.race_id] = _normalize_horse(r.horse_name or "")
        sp_by_race[r.race_id] = float(r.starting_price or 0)

    scoreable: list[dict] = []
    for p in picks:
        winner_norm = winners.get(p["race_id"])
        if winner_norm is None:
            continue
        p["won"] = (p["horse_name"] == winner_norm)
        p["sp"] = sp_by_race.get(p["race_id"], 0)
        scoreable.append(p)

    if not scoreable:
        return {"days": days, "rank1_picks": 0}

    bin_defs = [
        ("<20%",  0.0,  0.20),
        ("20-30%", 0.20, 0.30),
        ("30-40%", 0.30, 0.40),
        (">=40%",  0.40, 1.01),
    ]
    going_buckets = ("good", "soft", "heavy", "firm")

    def _bin_for(prob: float) -> Optional[str]:
        for label, lo, hi in bin_defs:
            if lo <= prob < hi:
                return label
        return None

    # rows[going][bin] = {n, wins}
    rows: dict[str, dict[str, dict]] = {g: {b[0]: {"n": 0, "wins": 0, "sp_won_sum": 0.0}
                                              for b in bin_defs} for g in going_buckets}
    overall: dict[str, dict] = {b[0]: {"n": 0, "wins": 0} for b in bin_defs}
    overall_total = {"n": 0, "wins": 0}

    for p in scoreable:
        going = (p["enr"].get("track_condition_category") or "").lower()
        if going not in going_buckets:
            continue
        b = _bin_for(p["win_prob"])
        if b is None:
            continue
        rows[going][b]["n"] += 1
        if p["won"]:
            rows[going][b]["wins"] += 1
            rows[going][b]["sp_won_sum"] += p["sp"]
        overall[b]["n"] += 1
        if p["won"]:
            overall[b]["wins"] += 1
        overall_total["n"] += 1
        if p["won"]:
            overall_total["wins"] += 1

    # Per-going recommended multiplier — overall win % on good / actual win % on this going
    # ratio'd so that applying it brings the displayed model_pct in line.
    overall_good_win = (rows["good"][b[0]] for b in bin_defs)
    good_total_n = sum(rows["good"][b[0]]["n"] for b in bin_defs)
    good_total_wins = sum(rows["good"][b[0]]["wins"] for b in bin_defs)
    good_win_rate = good_total_wins / good_total_n if good_total_n else None

    recommended_multiplier = {}
    for g in going_buckets:
        n = sum(rows[g][b[0]]["n"] for b in bin_defs)
        w = sum(rows[g][b[0]]["wins"] for b in bin_defs)
        if n < 5 or good_win_rate is None:
            recommended_multiplier[g] = {"sample": n, "multiplier": None,
                                          "note": "thin sample"}
            continue
        actual = w / n
        # Multiplier brings the model's expected output in line with actual.
        # If model expects 27%% good baseline and soft horses actually win 17%%,
        # multiplier = 17/27 = 0.63.
        mult = round(actual / good_win_rate, 3) if good_win_rate else None
        recommended_multiplier[g] = {
            "sample": n,
            "actual_win_pct": round(actual * 100, 1),
            "good_baseline_win_pct": round(good_win_rate * 100, 1),
            "multiplier": mult,
        }

    out_rows = []
    for g in going_buckets:
        bin_rows = []
        for b_label, _, _ in bin_defs:
            d = rows[g][b_label]
            if d["n"] == 0:
                continue
            bin_rows.append({
                "model_pct_bin": b_label,
                "n": d["n"],
                "wins": d["wins"],
                "actual_win_pct": round(d["wins"]/d["n"]*100, 1),
                "avg_winner_sp": round(d["sp_won_sum"]/d["wins"], 2) if d["wins"] else None,
            })
        if bin_rows:
            out_rows.append({"going": g, "bins": bin_rows})

    overall_bins = []
    for b_label, _, _ in bin_defs:
        d = overall[b_label]
        if d["n"] == 0:
            continue
        overall_bins.append({
            "model_pct_bin": b_label,
            "n": d["n"],
            "wins": d["wins"],
            "actual_win_pct": round(d["wins"]/d["n"]*100, 1),
        })

    return {
        "days": days,
        "include_backfill": include_backfill,
        "rank1_picks_with_going": overall_total["n"],
        "overall_by_bin": overall_bins,
        "by_going_and_bin": out_rows,
        "recommended_multiplier_per_going": recommended_multiplier,
    }


@app.get("/api/admin/bets/lab-winner-vs-loser-features")
async def lab_winner_vs_loser_features(
    days: int = 14,
    include_races: bool = True,
    include_backfill: bool = False,
    x_cron_secret: Optional[str] = Header(None),
):
    """Lab-Sharp equivalent of /winner-vs-loser-features.

    Scope: races that pass the Lab Sharp gate (top-3 sum ≤55% AND
    field_size ≤11) over the last N days. Outcome: did our predicted
    top-3 box contain the actual top-3 finishers (in any order)?

    For each per-race numerical feature, reports mean / median between
    hits and misses with `loser_minus_winner` = misses_mean − hits_mean.
    For each categorical feature, reports hit rate per category. Large
    abs_lift values are candidates for a new Lab Sharp gate (e.g.,
    `lab_sharp_exclude_<slice>`).

    Per the 'keep it sharp' rule, the analyser only surfaces TIGHTENING
    candidates — slices to exclude from Lab Sharp. The recommendation
    layer can be wired to refuse loosening suggestions.
    """
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        # Pull top-3 predictions per race (rank 1-3, cancelled excluded)
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.model_rank,
                RunnerPredictionHistoryRow.market_rank,
                RunnerPredictionHistoryRow.enriched_json,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank.in_([1, 2, 3]))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill"))
                   if include_backfill else RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        # Field-size lookup — distinct uncancelled runners per race
        field_rows = (await session.execute(
            select(RunnerPredictionHistoryRow.race_id, RunnerPredictionHistoryRow.horse_name)
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill"))
                   if include_backfill else RunnerPredictionHistoryRow.source == "live")
        )).fetchall()
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
                   HistoricalResultRow.position, HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2, 3]))
        )).fetchall()

    # Field size per race (distinct horses)
    field_size_by_race: dict[str, set] = {}
    for r in field_rows:
        field_size_by_race.setdefault(r.race_id, set()).add(_normalize_horse(r.horse_name))
    field_size_by_race = {k: len(v) for k, v in field_size_by_race.items()}

    # Latest snapshot per (race, horse, rank). Build top-3 box per race.
    seen: set = set()
    top3_by_race: dict[str, list[dict]] = {}
    for r in pred_rows:
        key = (r.race_id, _normalize_horse(r.horse_name))
        if key in seen:
            continue
        seen.add(key)
        try:
            enr = json.loads(r.enriched_json) if r.enriched_json else {}
        except Exception:
            enr = {}
        top3_by_race.setdefault(r.race_id, []).append({
            "horse_norm": _normalize_horse(r.horse_name),
            "horse_name": r.horse_name,
            "win_prob": float(r.win_probability or 0),
            "model_rank": r.model_rank,
            "market_rank": r.market_rank,
            "enr": enr,
        })
    # Sort by rank, keep first 3
    for rid, runners in top3_by_race.items():
        runners.sort(key=lambda x: x["model_rank"] or 99)
        top3_by_race[rid] = runners[:3]

    # Actual top-3 finishers per race
    actual_top3: dict[str, set] = {}
    for r in result_rows:
        if r.position and r.position in (1, 2, 3):
            actual_top3.setdefault(r.race_id, set()).add(
                _normalize_horse(r.horse_name or "")
            )

    # Build per-race records
    races: list[dict] = []
    for rid, top3 in top3_by_race.items():
        if len(top3) < 3:
            continue
        field_size = field_size_by_race.get(rid, 0)
        top3_sum = sum(r["win_prob"] for r in top3) * 100
        # Lab Sharp gate: top-3 ≤55% AND field ≤11
        if top3_sum > 55 or field_size > 11:
            continue
        actual = actual_top3.get(rid)
        if not actual or len(actual) < 3:
            continue
        predicted = {r["horse_norm"] for r in top3}
        # Trifecta box HIT: all 3 actual top-3 are in our predicted top-3.
        # (For a 3-horse box, that means our top-3 == actual top-3 as sets.)
        hit = actual.issubset(predicted)
        rank1, rank2, rank3 = top3[0], top3[1], top3[2]
        # Per-race features
        days_off = [r["enr"].get("days_since_last_run") for r in top3]
        days_off_valid = [d for d in days_off if isinstance(d, (int, float))]
        form = [r["enr"].get("form_score") for r in top3]
        form_valid = [f for f in form if isinstance(f, (int, float))]
        starts10 = [r["enr"].get("starts_last_10") for r in top3]
        starts10_valid = [s for s in starts10 if isinstance(s, (int, float))]
        market_aligned_count = sum(
            1 for r in top3 if isinstance(r.get("market_rank"), int) and r["market_rank"] <= 3
        )
        rank1_pct = rank1["win_prob"] * 100
        rank2_pct = rank2["win_prob"] * 100
        rank3_pct = rank3["win_prob"] * 100
        rank1_rank2_gap = rank1_pct - rank2_pct
        rank2_rank3_gap = rank2_pct - rank3_pct
        # Race-level context (from rank-1's enriched_json)
        enr1 = rank1["enr"]
        races.append({
            "race_id": rid,
            "hit": hit,
            "field_size": field_size,
            "top3_sum_pct": round(top3_sum, 2),
            "rank1_pct": round(rank1_pct, 2),
            "rank2_pct": round(rank2_pct, 2),
            "rank3_pct": round(rank3_pct, 2),
            "rank1_rank2_gap_pp": round(rank1_rank2_gap, 2),
            "rank2_rank3_gap_pp": round(rank2_rank3_gap, 2),
            "avg_days_off_top3": round(sum(days_off_valid)/len(days_off_valid), 1) if days_off_valid else None,
            "max_days_off_top3": max(days_off_valid) if days_off_valid else None,
            "avg_form_top3": round(sum(form_valid)/len(form_valid), 3) if form_valid else None,
            "avg_starts10_top3": round(sum(starts10_valid)/len(starts10_valid), 1) if starts10_valid else None,
            "market_aligned_top3_count": market_aligned_count,
            "track_condition_category": enr1.get("track_condition_category"),
            "distance_aptitude": enr1.get("distance_aptitude"),
            "distance": enr1.get("distance") or enr1.get("race_distance"),
        })

    if not races:
        return {"days": days, "lab_eligible_races": 0,
                "note": "no settled Lab-Sharp eligible races in window"}

    hits = [r for r in races if r["hit"]]
    misses = [r for r in races if not r["hit"]]
    n = len(races)

    # Numerical breakdown
    NUMERICAL_FEATURES = [
        "field_size", "top3_sum_pct", "rank1_pct", "rank2_pct", "rank3_pct",
        "rank1_rank2_gap_pp", "rank2_rank3_gap_pp",
        "avg_days_off_top3", "max_days_off_top3",
        "avg_form_top3", "avg_starts10_top3",
        "market_aligned_top3_count", "distance",
    ]
    CATEGORICAL_FEATURES = ["track_condition_category", "distance_aptitude"]

    def _vals(cohort: list[dict], feat: str) -> list[float]:
        return [r[feat] for r in cohort if isinstance(r.get(feat), (int, float))]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "median": None}
        s = sorted(vals)
        m = sum(s) / len(s)
        median = s[len(s)//2] if len(s) % 2 else (s[len(s)//2-1] + s[len(s)//2]) / 2
        return {"n": len(vals), "mean": round(m, 4), "median": round(median, 4)}

    feature_breakdown = []
    for feat in NUMERICAL_FEATURES:
        hit_stats = _stats(_vals(hits, feat))
        miss_stats = _stats(_vals(misses, feat))
        if hit_stats["mean"] is None or miss_stats["mean"] is None:
            continue
        lift = round(miss_stats["mean"] - hit_stats["mean"], 4)
        feature_breakdown.append({
            "feature": feat,
            "hit": hit_stats,
            "miss": miss_stats,
            "miss_minus_hit": lift,
            "abs_lift": abs(lift),
        })
    feature_breakdown.sort(key=lambda f: -f["abs_lift"])

    categorical_breakdown = []
    for feat in CATEGORICAL_FEATURES:
        counts: dict[str, dict] = {}
        for r in races:
            v = r.get(feat)
            if v is None:
                continue
            key = str(v)
            d = counts.setdefault(key, {"n": 0, "hits": 0})
            d["n"] += 1
            if r["hit"]:
                d["hits"] += 1
        cats = []
        for cat, d in counts.items():
            if d["n"] < 3:
                continue
            cats.append({
                "category": cat,
                "n": d["n"],
                "hit_pct": round(d["hits"] / d["n"] * 100, 1),
            })
        cats.sort(key=lambda c: -c["hit_pct"])
        if cats:
            categorical_breakdown.append({"feature": feat, "categories": cats})

    # Tightening-only candidate refinements: only suggest excluding the
    # *worse* side of the split (i.e., the side closer to the miss mean).
    refinements = []
    for f in feature_breakdown[:8]:
        if f["hit"]["n"] < 5 or f["miss"]["n"] < 5:
            continue
        direction = "lower" if f["miss_minus_hit"] > 0 else "higher"
        worse_mean = f["miss"]["mean"]
        refinements.append({
            "feature": f["feature"],
            "observation": (
                f"Trifecta-hit races average {f['hit']['mean']} ({f['hit']['n']} samples). "
                f"Miss races average {f['miss']['mean']} ({f['miss']['n']} samples). "
                f"Hits tend to be {direction} on this feature."
            ),
            "candidate_gate": (
                f"Consider excluding Lab Sharp races where {f['feature']} is closer to "
                f"the miss mean ({worse_mean}). Pattern_id: "
                f"`lab_sharp_exclude_{f['feature']}_drift`."
            ),
        })

    response = {
        "days": days,
        "include_backfill": include_backfill,
        "lab_eligible_races": n,
        "hits": len(hits),
        "misses": len(misses),
        "hit_pct": round(len(hits) / n * 100, 1) if n else 0,
        "numerical_features_ranked": feature_breakdown,
        "categorical_features": categorical_breakdown,
        "candidate_refinements": refinements[:5],
    }
    if include_races:
        response["per_race_detail"] = races
    return response


@app.get("/api/admin/bets/winner-vs-loser-features")
async def winner_vs_loser_features(
    days: int = 1,
    include_picks: bool = True,
    include_backfill: bool = False,
    x_cron_secret: Optional[str] = Header(None),
):
    """Deep-dive comparison of rank-1 picks: winners vs losers across all
    enriched features. For each numerical feature, reports mean / median
    in each cohort and a 'lift' = loser_mean - winner_mean. Features with
    big absolute lifts are candidates for model refinement (the model
    isn't already weighting them enough to discriminate).

    For each categorical feature reports win rate per category.

    include_picks=true also returns the per-pick detail (race_id, won,
    placed, key features). Useful for the per-race walkthrough on a short
    window (yesterday). Turn off for week+ windows to keep payload small.
    """
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.market_rank,
                RunnerPredictionHistoryRow.best_available_odds,
                RunnerPredictionHistoryRow.enriched_json,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill"))
                   if include_backfill else RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
                   HistoricalResultRow.position, HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
        )).fetchall()

    # Latest snapshot per (race, horse)
    seen: set = set()
    picks: list[dict] = []
    for r in pred_rows:
        key = (r.race_id, _normalize_horse(r.horse_name))
        if key in seen:
            continue
        seen.add(key)
        try:
            enr = json.loads(r.enriched_json) if r.enriched_json else {}
        except Exception:
            enr = {}
        picks.append({
            "race_id": r.race_id,
            "horse_name": _normalize_horse(r.horse_name),
            "display_name": r.horse_name,
            "win_prob": float(r.win_probability or 0),
            "odds": float(r.best_available_odds or 0),
            "market_rank": r.market_rank,
            "enr": enr,
        })

    # Positions per (race, horse) — keep both rank-1 result and the actual top-3 horses
    pos_by_key: dict[tuple[str, str], tuple[int, float]] = {}
    for r in result_rows:
        if r.position is None:
            continue
        pos_by_key[(r.race_id, _normalize_horse(r.horse_name or ""))] = (
            int(r.position), float(r.starting_price or 0)
        )

    # Annotate each pick with outcome
    scoreable: list[dict] = []
    for p in picks:
        result = pos_by_key.get((p["race_id"], p["horse_name"]))
        if not result:
            continue
        pos, sp = result
        p["actual_position"] = pos
        p["sp"] = sp
        p["won"] = pos == 1
        p["placed"] = pos <= 3
        scoreable.append(p)

    if not scoreable:
        return {"days": days, "settled_rank1_picks": 0, "note": "no settled picks in window"}

    winners = [p for p in scoreable if p["won"]]
    placers = [p for p in scoreable if p["placed"] and not p["won"]]
    losers = [p for p in scoreable if not p["placed"]]
    n = len(scoreable)

    # Numerical features to compare. These are the enriched_json keys the
    # model sees (subset of FEATURE_NAMES_BASE plus useful derived signals).
    NUMERICAL_FEATURES = [
        "form_score", "days_since_last_run", "runs_this_prep", "starts_last_10",
        "wins_last_10", "places_last_10", "win_rate_distance", "trainer_overall_rate",
        "jockey_overall_rate", "trainer_track_rate", "trainer_distance_rate",
        "jockey_track_rate", "wet_track_record", "pedigree_distance_match",
        "pedigree_wet_score", "dosage_index", "weight", "barrier",
        "win_rate_career", "place_rate_career", "win_rate_track", "career_starts",
        "class_change", "best_finish_distance", "avg_beaten_margin_last5",
    ]
    CATEGORICAL_FEATURES = [
        "speed_map_position", "distance_aptitude", "track_condition_category",
        "is_steamed", "is_drifted",
    ]
    # Derived signals not in raw enriched_json
    def _derive(p: dict) -> dict:
        d = {
            "win_prob_pct": p["win_prob"] * 100,
            "odds": p["odds"],
            "market_rank": p["market_rank"] or 0,
            "model_pct_edge": (p["win_prob"] * 100) - ((100 / p["odds"]) if p["odds"] > 1 else 0),
        }
        return d
    DERIVED_FEATURES = ["win_prob_pct", "odds", "market_rank", "model_pct_edge"]

    def _vals(cohort: list[dict], feat: str) -> list[float]:
        if feat in DERIVED_FEATURES:
            return [_derive(p)[feat] for p in cohort if _derive(p).get(feat) is not None]
        return [p["enr"].get(feat) for p in cohort
                if isinstance(p["enr"].get(feat), (int, float))]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "median": None}
        s = sorted(vals)
        m = sum(s) / len(s)
        median = s[len(s)//2] if len(s) % 2 else (s[len(s)//2-1] + s[len(s)//2]) / 2
        return {"n": len(vals), "mean": round(m, 4), "median": round(median, 4)}

    feature_breakdown: list[dict] = []
    for feat in NUMERICAL_FEATURES + DERIVED_FEATURES:
        w_stats = _stats(_vals(winners, feat))
        l_stats = _stats(_vals(losers, feat))
        if w_stats["mean"] is None or l_stats["mean"] is None:
            continue
        lift = round(l_stats["mean"] - w_stats["mean"], 4)
        feature_breakdown.append({
            "feature": feat,
            "winner": w_stats,
            "loser": l_stats,
            "loser_minus_winner": lift,
            "abs_lift": abs(lift),
        })
    feature_breakdown.sort(key=lambda f: -f["abs_lift"])

    # Categorical breakdown: for each category, win rate
    categorical_breakdown: list[dict] = []
    for feat in CATEGORICAL_FEATURES:
        counts: dict[str, dict] = {}
        for p in scoreable:
            v = p["enr"].get(feat)
            if v is None:
                continue
            key = str(v)
            d = counts.setdefault(key, {"n": 0, "wins": 0, "placed": 0})
            d["n"] += 1
            if p["won"]:
                d["wins"] += 1
            if p["placed"]:
                d["placed"] += 1
        cats = []
        for cat, d in counts.items():
            if d["n"] < 3:
                continue
            cats.append({
                "category": cat,
                "n": d["n"],
                "win_pct": round(d["wins"] / d["n"] * 100, 1),
                "place_pct": round(d["placed"] / d["n"] * 100, 1),
            })
        cats.sort(key=lambda c: -c["win_pct"])
        if cats:
            categorical_breakdown.append({"feature": feat, "categories": cats})

    # Suggested refinements — features with biggest absolute lift AND
    # decent sample on both sides. Note: these are descriptive observations,
    # not validated until backtested.
    refinements: list[dict] = []
    for f in feature_breakdown[:10]:
        if f["winner"]["n"] < 5 or f["loser"]["n"] < 5:
            continue
        direction = "lower" if f["loser_minus_winner"] > 0 else "higher"
        refinements.append({
            "feature": f["feature"],
            "observation": (
                f"Winners average {f['winner']['mean']} ({f['winner']['n']} samples). "
                f"Losers average {f['loser']['mean']} ({f['loser']['n']} samples). "
                f"Winners tend to be {direction} on this feature."
            ),
            "candidate_filter": f"Consider de-emphasising rank-1 picks where {f['feature']} is "
                                f"closer to the loser mean ({f['loser']['mean']}).",
        })

    response = {
        "days": days,
        "include_backfill": include_backfill,
        "settled_rank1_picks": n,
        "winners": len(winners),
        "placed_no_win": len(placers),
        "losers": len(losers),
        "win_pct": round(len(winners) / n * 100, 1) if n else 0,
        "place_pct": round((len(winners) + len(placers)) / n * 100, 1) if n else 0,
        "numerical_features_ranked": feature_breakdown,
        "categorical_features": categorical_breakdown,
        "candidate_refinements": refinements[:5],
    }
    if include_picks:
        response["per_pick_detail"] = [{
            "race_id": p["race_id"],
            "horse": p["display_name"],
            "model_win_pct": round(p["win_prob"] * 100, 1),
            "odds": p["odds"],
            "actual_position": p["actual_position"],
            "won": p["won"],
            "placed": p["placed"],
            "sp": p["sp"],
            "form_score": p["enr"].get("form_score"),
            "days_since_last_run": p["enr"].get("days_since_last_run"),
            "starts_last_10": p["enr"].get("starts_last_10"),
            "speed_map_position": p["enr"].get("speed_map_position"),
            "win_rate_distance": p["enr"].get("win_rate_distance"),
            "trainer_overall_rate": p["enr"].get("trainer_overall_rate"),
            "jockey_overall_rate": p["enr"].get("jockey_overall_rate"),
        } for p in scoreable]
    return response


@app.get("/api/admin/bets/signal-scan")
async def signal_scan(
    days: int = 60,
    stake: float = 10.0,
    min_sample: int = 30,
    include_backfill: bool = False,
    x_cron_secret: Optional[str] = Header(None),
):
    """Sweep a battery of candidate filter patterns against the rank-1 history
    table and rank them by hit-rate lift + ROI vs baseline. Designed to surface
    untapped signals — combinations of features that, applied as filters on top
    of the existing model, deliver a meaningful edge.

    Each signal: a name + description + a predicate over the rank-1 pick's
    enriched features. We compute races, wins, win_pct, avg winner SP, P&L at
    flat stake, ROI%, and the lift over the unfiltered baseline.
    """
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.place_probability,
                RunnerPredictionHistoryRow.market_rank,
                RunnerPredictionHistoryRow.best_available_odds,
                RunnerPredictionHistoryRow.overlay,
                RunnerPredictionHistoryRow.enriched_json,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill"))
                   if include_backfill else RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
                   HistoricalResultRow.position, HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position == 1)
        )).fetchall()

    # Latest snapshot per (race, horse)
    seen: set = set()
    picks: list[dict] = []
    for r in pred_rows:
        key = (r.race_id, _normalize_horse(r.horse_name))
        if key in seen:
            continue
        seen.add(key)
        try:
            enr = json.loads(r.enriched_json) if r.enriched_json else {}
        except Exception:
            enr = {}
        picks.append({
            "race_id": r.race_id,
            "horse_name": _normalize_horse(r.horse_name),
            "win_prob": float(r.win_probability or 0),
            "place_prob": float(r.place_probability or 0),
            "market_rank": int(r.market_rank) if r.market_rank else None,
            "odds": float(r.best_available_odds or 0),
            "overlay": float(r.overlay or 0),
            "enr": enr,
        })

    # Winners per race
    winners: dict[str, tuple[str, float]] = {}
    for r in result_rows:
        winners[r.race_id] = (_normalize_horse(r.horse_name or ""), float(r.starting_price or 0))

    # Annotate each pick with outcome
    scoreable: list[dict] = []
    for p in picks:
        w = winners.get(p["race_id"])
        if not w:
            continue
        winner_name, w_sp = w
        p["won"] = (p["horse_name"] == winner_name)
        p["winner_sp"] = w_sp
        scoreable.append(p)

    def _score(pool: list[dict]) -> dict:
        n = len(pool)
        wins = sum(1 for p in pool if p.get("won"))
        sp_sample = [p for p in pool if (p.get("winner_sp") or 0) > 1.0]
        pl_races = len(sp_sample)
        pl = 0.0
        for p in sp_sample:
            pl += (p["winner_sp"] * stake - stake) if p.get("won") else -stake
        avg_sp = (sum(p["winner_sp"] for p in sp_sample) / pl_races) if pl_races else None
        return {
            "races": n,
            "wins": wins,
            "win_pct": round(wins / n * 100, 1) if n else 0,
            "avg_winner_sp": round(avg_sp, 2) if avg_sp else None,
            "pl_dollars": round(pl, 2),
            "pl_sample": pl_races,
            "roi_pct": round(pl / (pl_races * stake) * 100, 1) if pl_races else None,
        }

    base = _score(scoreable)

    # Signal predicates — each is a function(pick) → bool. enr is the
    # parsed enriched_json with all per-runner features.
    def _gt(v, t): return v is not None and v >= t
    def _eq(v, t): return v is not None and v == t

    signals: list[tuple[str, str, callable]] = [
        ("Sharp (current)",
         "model_pct ≥ 30 AND edge ≥ 0",
         lambda p: p["win_prob"] * 100 >= 30 and ((100 - (100 / p["odds"] if p["odds"] > 1 else 0)) if p["odds"] > 1 else None) is not None
                   and (p["win_prob"] * 100) >= ((100 / p["odds"]) if p["odds"] > 1 else 0)),
        ("Confident (no edge gate)",
         "model_pct ≥ 30 — just the confidence half",
         lambda p: p["win_prob"] * 100 >= 30),
        ("Dominant favourite",
         "rank-1 win_prob ≥ 40%",
         lambda p: p["win_prob"] * 100 >= 40),
        ("Premium overlay",
         "model_pct ≥ 30 AND odds ≥ $3 AND edge ≥ 5pt",
         lambda p: p["win_prob"] * 100 >= 30 and p["odds"] >= 3.0
                   and (p["win_prob"] * 100 - (100 / p["odds"] if p["odds"] > 1 else 0)) >= 5),
        ("Value overlay (mid-price)",
         "edge ≥ 5pt AND $4 ≤ odds ≤ $10",
         lambda p: p["odds"] >= 4.0 and p["odds"] <= 10.0
                   and (p["win_prob"] * 100 - (100 / p["odds"] if p["odds"] > 1 else 0)) >= 5),
        ("Market consensus",
         "model_rank = 1 AND market_rank = 1 AND model_pct ≥ 25",
         lambda p: p["win_prob"] * 100 >= 25 and p.get("market_rank") == 1),
        ("Model contrarian",
         "market_rank ≥ 4 AND edge ≥ 10pt (model finds value market missed)",
         lambda p: (p.get("market_rank") or 0) >= 4
                   and (p["win_prob"] * 100 - (100 / p["odds"] if p["odds"] > 1 else 0)) >= 10),
        ("Hot trainer + confident",
         "trainer_overall_rate ≥ 25% AND model_pct ≥ 25",
         lambda p: _gt(p["enr"].get("trainer_overall_rate"), 25) and p["win_prob"] * 100 >= 25),
        ("Hot jockey + confident",
         "jockey_overall_rate ≥ 20% AND model_pct ≥ 25",
         lambda p: _gt(p["enr"].get("jockey_overall_rate"), 20) and p["win_prob"] * 100 >= 25),
        ("Class drop in form",
         "class_change = -1 AND form_score ≥ 0.75 AND model_pct ≥ 25",
         lambda p: _eq(p["enr"].get("class_change"), -1) and _gt(p["enr"].get("form_score"), 0.75)
                   and p["win_prob"] * 100 >= 25),
        ("Distance specialist",
         "win_rate_distance ≥ 0.30 AND starts_last_10 ≥ 3 AND model_pct ≥ 25",
         lambda p: _gt(p["enr"].get("win_rate_distance"), 0.30)
                   and _gt(p["enr"].get("starts_last_10"), 3)
                   and p["win_prob"] * 100 >= 25),
        ("Fresh horse (7-21d)",
         "days_since_last_run 7-21d AND model_pct ≥ 30",
         lambda p: 7 <= (p["enr"].get("days_since_last_run") or 0) <= 21
                   and p["win_prob"] * 100 >= 30),
        ("Speed-map leader",
         "speed_map_position = 'leader' AND model_pct ≥ 25",
         lambda p: p["enr"].get("speed_map_position") == "leader"
                   and p["win_prob"] * 100 >= 25),
        ("Outstanding recent form",
         "form_score ≥ 0.85 AND starts_last_10 ≥ 3 AND model_pct ≥ 25",
         lambda p: _gt(p["enr"].get("form_score"), 0.85)
                   and _gt(p["enr"].get("starts_last_10"), 3)
                   and p["win_prob"] * 100 >= 25),
        ("Open race fav",
         "model_pct ≥ 30 AND top-2 horses sum < 50% (race is open)",
         lambda p: p["win_prob"] * 100 >= 30
                   # No top2 sum here — approximate via place_prob being similar to win_prob (implies open)
                   and (p["place_prob"] * 100) <= (p["win_prob"] * 100 + 5)),
        ("Concentrated favourite",
         "model_pct ≥ 40 AND place_prob ≥ 30% (model dominates)",
         lambda p: p["win_prob"] * 100 >= 40 and p["place_prob"] * 100 >= 30),
        ("Premium price + confidence",
         "model_pct ≥ 30 AND $5 ≤ odds ≤ $15",
         lambda p: p["win_prob"] * 100 >= 30 and 5.0 <= p["odds"] <= 15.0),
        ("Steamed (market backing)",
         "is_steamed = true AND model_pct ≥ 25",
         lambda p: p["enr"].get("is_steamed") is True and p["win_prob"] * 100 >= 25),
        ("Drifted (market fading)",
         "is_drifted = true AND model_pct ≥ 25 — for filtering OUT",
         lambda p: p["enr"].get("is_drifted") is True and p["win_prob"] * 100 >= 25),
        ("Long layoff resumer",
         "days_since_last_run ≥ 180 AND model_pct ≥ 25 — for filtering OUT",
         lambda p: _gt(p["enr"].get("days_since_last_run"), 180) and p["win_prob"] * 100 >= 25),
        ("Wet track specialist",
         "track in (soft, heavy) AND wet_track_record ≥ 0.5 AND model_pct ≥ 25",
         lambda p: p["enr"].get("track_condition_category") in ("soft", "heavy")
                   and _gt(p["enr"].get("wet_track_record"), 0.5)
                   and p["win_prob"] * 100 >= 25),
    ]

    out_signals = []
    for name, desc, pred in signals:
        try:
            pool = [p for p in scoreable if pred(p)]
        except Exception as e:
            log.warning(f"signal-scan '{name}' failed: {e}")
            continue
        s = _score(pool)
        s["name"] = name
        s["description"] = desc
        s["win_lift_pts"] = round(s["win_pct"] - base["win_pct"], 1)
        s["roi_lift_pts"] = round((s["roi_pct"] or 0) - (base["roi_pct"] or 0), 1) if s["roi_pct"] is not None else None
        s["thin_sample"] = s["races"] < min_sample
        out_signals.append(s)

    # Sort: ROI desc among samples ≥ min_sample, then by win_lift
    def _sort_key(s):
        if s["thin_sample"]:
            return (1, 0, 0)
        return (0, -(s["roi_pct"] or -999), -(s["win_lift_pts"] or -999))
    out_signals.sort(key=_sort_key)

    return {
        "days": days,
        "stake_per_bet": stake,
        "min_sample": min_sample,
        "baseline_rank1": base,
        "signals_ranked_by_roi": out_signals,
    }


@app.get("/api/admin/bets/top2-gap-backtest")
async def top2_gap_backtest(
    days: int = 60,
    stake: float = 10.0,
    sharp_only: bool = False,
    x_cron_secret: Optional[str] = Header(None),
):
    """Backtest the Hot Seat 'Top-2 gap' filter — does picking the model's
    rank-1 horse in races where it has a big lead over rank-2 actually
    result in higher win rate and/or positive ROI vs the baseline?

    For each race in the window with rank-1 + rank-2 + a result:
      - gap = (rank1 win prob) - (rank2 win prob) in percentage points
      - Bucket by gap size (<2pt, 2-5pt, 5-10pt, 10-15pt, 15-20pt, 20pt+)
      - Per bucket: win rate of rank-1, avg SP, P&L at flat stake, ROI%

    sharp_only mirrors the Hot Seat default — also require Sharp niche:
    model_pct ≥ 30 OR top-3 sum ≥ 60.
    """
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.model_rank,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank.in_([1, 2, 3]))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.horse_name, HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position == 1)
        )).fetchall()

    # Latest snapshot per (race, horse)
    seen: set = set()
    per_race: dict[str, dict] = {}
    for rid, name, rank, wp, enr in pred_rows:
        key = (rid, _normalize_horse(name))
        if key in seen:
            continue
        seen.add(key)
        per_race.setdefault(rid, {})[int(rank)] = {
            "horse_name": _normalize_horse(name),
            "win_prob": float(wp or 0),
        }

    # Winner per race
    winners: dict[str, tuple[str, float]] = {}
    for rid, pos, name, sp in result_rows:
        if pos == 1 and rid not in winners:
            winners[rid] = (_normalize_horse(name or ""), float(sp or 0))

    bucket_defs = [
        ("<2pt (toss-up)",   -999.0,  2.0),
        ("2-5pt",              2.0,   5.0),
        ("5-10pt",             5.0,  10.0),
        ("10-15pt",           10.0,  15.0),
        ("15-20pt",           15.0,  20.0),
        ("20pt+ (dominant)",  20.0, 999.0),
    ]
    buckets = {b[0]: {"races": 0, "wins": 0, "sp_sum": 0.0, "pl": 0.0, "pl_races": 0}
               for b in bucket_defs}
    baseline = {"races": 0, "wins": 0, "sp_sum": 0.0, "pl": 0.0, "pl_races": 0}

    for rid, ranks in per_race.items():
        if not all(r in ranks for r in (1, 2)):
            continue
        winner = winners.get(rid)
        if not winner:
            continue
        winner_name, w_sp = winner
        # Optional Sharp gate (matches Hot Seat default)
        if sharp_only:
            top3_sum = sum((ranks.get(r, {}).get("win_prob") or 0) for r in (1, 2, 3)) * 100
            r1_pct = ranks[1]["win_prob"] * 100
            if not (r1_pct >= 30 or top3_sum >= 60):
                continue
        gap = (ranks[1]["win_prob"] - ranks[2]["win_prob"]) * 100
        won = (ranks[1]["horse_name"] == winner_name)
        # Baseline
        baseline["races"] += 1
        if won: baseline["wins"] += 1
        if w_sp > 1.0:
            baseline["sp_sum"] += w_sp
            baseline["pl_races"] += 1
            baseline["pl"] += (w_sp * stake - stake) if won else -stake
        # Bucket
        for label, lo, hi in bucket_defs:
            if lo <= gap < hi:
                b = buckets[label]
                b["races"] += 1
                if won: b["wins"] += 1
                if w_sp > 1.0:
                    b["sp_sum"] += w_sp
                    b["pl_races"] += 1
                    b["pl"] += (w_sp * stake - stake) if won else -stake
                break

    def _summarise(d: dict) -> dict:
        n = d["races"]
        win_pct = round(d["wins"] / n * 100, 1) if n else 0
        avg_sp = round(d["sp_sum"] / d["pl_races"], 2) if d["pl_races"] else None
        roi_pct = None
        if d["pl_races"] > 0:
            cost = d["pl_races"] * stake
            roi_pct = round(d["pl"] / cost * 100, 1)
        return {
            "races": n,
            "wins": d["wins"],
            "win_pct": win_pct,
            "avg_winner_sp": avg_sp,
            "pl_dollars": round(d["pl"], 2),
            "pl_sample": d["pl_races"],
            "roi_pct": roi_pct,
        }

    base_sum = _summarise(baseline)
    return {
        "days": days,
        "stake_per_bet": stake,
        "sharp_only": sharp_only,
        "baseline": base_sum,
        "by_gap_bucket": [
            {"bucket": label, **_summarise(buckets[label]),
             "win_lift_pts": round(_summarise(buckets[label])["win_pct"] - base_sum["win_pct"], 1)}
            for label, _, _ in bucket_defs
        ],
    }


@app.get("/api/admin/bets/clear-pair-backtest")
async def clear_pair_backtest(
    days: int = 60,
    top2_min: float = 50.0,
    gap_min: float = 8.0,
    stake: float = 10.0,
    place_top: int = 3,  # double-sharp variant: require both partners' place_model_rank ≤ this
    x_cron_secret: Optional[str] = Header(None),
):
    """Backtest the 'clear pair' pattern: two horses way out in front
    (top-2 win sum ≥ top2_min %), big cliff to rank-3 (rank2_win - rank3_win
    ≥ gap_min pt).

    For each qualifying race in the last N days:
      - quinella hit = rank-1 AND rank-2 finished 1st-2nd (either order)
      - split-win hit = rank-1 OR rank-2 won outright
      - estimated split-win ROI = $stake on rank-1 + $stake on rank-2 at SP

    Compared against an unfiltered baseline of all races in the window so
    we can see whether the pattern actually lifts these outcomes.
    """
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.model_rank,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.place_model_rank,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank.in_([1, 2, 3]))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.horse_name, HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2]))
        )).fetchall()

    # Group predictions: latest snapshot per (race, horse)
    seen: set = set()
    per_race: dict[str, dict] = {}
    for rid, name, rank, wp, prk, enr in pred_rows:
        key = (rid, _normalize_horse(name))
        if key in seen:
            continue
        seen.add(key)
        per_race.setdefault(rid, {})[int(rank)] = {
            "horse_name": _normalize_horse(name),
            "win_prob": float(wp or 0),
            "place_model_rank": int(prk) if prk is not None else None,
        }

    # Results: race_id → {position: (horse_normalised, sp)}
    results_by_race: dict[str, dict] = {}
    for rid, pos, name, sp in result_rows:
        if pos is None:
            continue
        results_by_race.setdefault(rid, {})[int(pos)] = (_normalize_horse(name or ""), float(sp or 0))

    def _score_race(ranks: dict, winners_map: dict, stake_each: float) -> dict | None:
        if not all(r in ranks for r in (1, 2)):
            return None
        if not all(p in winners_map for p in (1, 2)):
            return None
        r1 = ranks[1]["horse_name"]
        r2 = ranks[2]["horse_name"]
        winner, w_sp = winners_map[1]
        runnerup, _ = winners_map[2]
        finishers = {winner, runnerup}
        quinella_hit = (r1 in finishers) and (r2 in finishers)
        win_hit_r1 = (r1 == winner)
        win_hit_r2 = (r2 == winner)
        split_win_hit = win_hit_r1 or win_hit_r2
        # Split-win ROI: stake on r1 + stake on r2 at SP. Only the winner's leg pays.
        # If the SP is missing we skip ROI on that race.
        payout = 0.0
        sp_available = False
        for rank_name in (r1, r2):
            if rank_name == winner and w_sp > 1.0:
                payout = w_sp * stake_each
                sp_available = True
                break
        cost = 2 * stake_each
        roi_pl = (payout - cost) if sp_available or not split_win_hit else None
        return {
            "quinella_hit": quinella_hit,
            "split_win_hit": split_win_hit,
            "split_win_pl_dollars": roi_pl if (sp_available or not split_win_hit) else None,
            "cost_dollars": cost,
        }

    # Baseline: every race with rank-1 + rank-2 + a result
    baseline = {"races": 0, "quin_hits": 0, "split_hits": 0, "pl_sum": 0.0, "pl_races": 0}
    qualified = {"races": 0, "quin_hits": 0, "split_hits": 0, "pl_sum": 0.0, "pl_races": 0}
    # Double-sharp: same gate PLUS both partners' place_model_rank ≤ place_top.
    # Tests whether requiring win + place model agreement boosts hit rate.
    double_sharp = {"races": 0, "quin_hits": 0, "split_hits": 0, "pl_sum": 0.0, "pl_races": 0}
    bucket_defs = [
        ("8-10pt", 8.0, 10.0),
        ("10-12pt", 10.0, 12.0),
        ("12-15pt", 12.0, 15.0),
        ("15-20pt", 15.0, 20.0),
        ("20pt+", 20.0, 999.0),
    ]
    buckets = {b[0]: {"races": 0, "quin_hits": 0, "split_hits": 0, "pl_sum": 0.0, "pl_races": 0} for b in bucket_defs}

    for rid, ranks in per_race.items():
        winners_map = results_by_race.get(rid)
        if not winners_map:
            continue
        scored = _score_race(ranks, winners_map, stake)
        if scored is None:
            continue
        # Baseline counts every race we can score.
        baseline["races"] += 1
        if scored["quinella_hit"]: baseline["quin_hits"] += 1
        if scored["split_win_hit"]: baseline["split_hits"] += 1
        if scored["split_win_pl_dollars"] is not None:
            baseline["pl_sum"] += scored["split_win_pl_dollars"]
            baseline["pl_races"] += 1
        # Filter check
        if not all(r in ranks for r in (1, 2, 3)):
            continue
        top2_sum = (ranks[1]["win_prob"] + ranks[2]["win_prob"]) * 100
        gap23 = (ranks[2]["win_prob"] - ranks[3]["win_prob"]) * 100
        if top2_sum < top2_min or gap23 < gap_min:
            continue
        qualified["races"] += 1
        if scored["quinella_hit"]: qualified["quin_hits"] += 1
        if scored["split_win_hit"]: qualified["split_hits"] += 1
        if scored["split_win_pl_dollars"] is not None:
            qualified["pl_sum"] += scored["split_win_pl_dollars"]
            qualified["pl_races"] += 1
        # Double-sharp variant — both partners agree on place too
        r1_plc = ranks[1].get("place_model_rank")
        r2_plc = ranks[2].get("place_model_rank")
        if (r1_plc is not None and r2_plc is not None
                and r1_plc <= place_top and r2_plc <= place_top):
            double_sharp["races"] += 1
            if scored["quinella_hit"]: double_sharp["quin_hits"] += 1
            if scored["split_win_hit"]: double_sharp["split_hits"] += 1
            if scored["split_win_pl_dollars"] is not None:
                double_sharp["pl_sum"] += scored["split_win_pl_dollars"]
                double_sharp["pl_races"] += 1
        # Bucket by gap size
        for label, lo, hi in bucket_defs:
            if lo <= gap23 < hi:
                b = buckets[label]
                b["races"] += 1
                if scored["quinella_hit"]: b["quin_hits"] += 1
                if scored["split_win_hit"]: b["split_hits"] += 1
                if scored["split_win_pl_dollars"] is not None:
                    b["pl_sum"] += scored["split_win_pl_dollars"]
                    b["pl_races"] += 1
                break

    def _summarise(d: dict) -> dict:
        n = d["races"]
        roi_pct = None
        if d["pl_races"] > 0:
            cost = d["pl_races"] * 2 * stake
            roi_pct = round(d["pl_sum"] / cost * 100, 1)
        return {
            "races": n,
            "quinella_hit_pct": round(d["quin_hits"] / n * 100, 1) if n else 0,
            "split_win_hit_pct": round(d["split_hits"] / n * 100, 1) if n else 0,
            "split_win_pl_dollars": round(d["pl_sum"], 2),
            "split_win_pl_races": d["pl_races"],
            "split_win_roi_pct": roi_pct,
        }

    out_buckets = [
        {"bucket": label, **_summarise(buckets[label])}
        for label, _, _ in bucket_defs
    ]

    return {
        "days": days,
        "filter": f"top-2 win sum ≥ {top2_min}% AND rank2-rank3 gap ≥ {gap_min}pt",
        "double_sharp_filter": f"(above) PLUS both partners' place_model_rank ≤ {place_top}",
        "stake_per_horse": stake,
        "baseline_all_races": _summarise(baseline),
        "qualified": _summarise(qualified),
        "qualified_double_sharp": _summarise(double_sharp),
        "by_gap_bucket": out_buckets,
        "lift_pts": {
            "quinella_qualified_vs_baseline": round(
                (_summarise(qualified)["quinella_hit_pct"] or 0) - (_summarise(baseline)["quinella_hit_pct"] or 0),
                1,
            ),
            "quinella_double_sharp_vs_qualified": round(
                (_summarise(double_sharp)["quinella_hit_pct"] or 0) - (_summarise(qualified)["quinella_hit_pct"] or 0),
                1,
            ),
            "split_win_qualified_vs_baseline": round(
                (_summarise(qualified)["split_win_hit_pct"] or 0) - (_summarise(baseline)["split_win_hit_pct"] or 0),
                1,
            ),
            "split_win_double_sharp_vs_qualified": round(
                (_summarise(double_sharp)["split_win_hit_pct"] or 0) - (_summarise(qualified)["split_win_hit_pct"] or 0),
                1,
            ),
        },
    }


@app.get("/api/admin/bets/all-tactics-filtered-backtest")
async def all_tactics_filtered_backtest(
    days: int = 90,
    top3_max: float = 45.0,
    field_max: int = 11,
    x_cron_secret: Optional[str] = Header(None),
):
    """Retroactive backtest of EVERY strategy in STRATEGY_REGISTRY against
    historical races, filtered by top-3 sum % ceiling + field-size
    ceiling. For each strategy returns races evaluated, races hit (any
    of its boxes caught the placings), and box-level hit count.

    'Race-hit' = at least one box generated by the strategy matched the
    top-3 finishers. 'Box-hit' counts every individual hitting box (some
    strategies generate multiple boxes per race so they get more attempts).
    """
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.tab_number, HistoricalResultRow.horse_name)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2, 3]))
        )).fetchall()
        results_by_race: dict[str, list] = {}
        for rid, pos, tab, name in result_rows:
            results_by_race.setdefault(rid, []).append((pos, tab, name))
        complete = [rid for rid, items in results_by_race.items() if len(items) >= 3]
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(complete))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
        )).scalars().all() if complete else []

    preds_by_race: dict[str, list] = {}
    for p in pred_rows:
        preds_by_race.setdefault(p.race_id, []).append(p)
    tab_per_race: dict[str, dict[str, int]] = {}
    for rid, pruns in preds_by_race.items():
        tab_per_race[rid] = {
            _normalize_horse(p.horse_name): p.tab_number
            for p in pruns if p.tab_number is not None
        }

    # Pre-resolve actual top-3 + features per race that survives the filter
    eligible: dict[str, dict] = {}
    universe_count = 0
    for rid, pruns in preds_by_race.items():
        if len(pruns) < 3 or rid not in results_by_race:
            continue
        items = sorted(results_by_race[rid], key=lambda x: x[0])[:3]
        actual: list = []
        for pos, tab, name in items:
            t = tab if tab is not None else tab_per_race.get(rid, {}).get(_normalize_horse(name))
            actual.append(t)
        if any(t is None for t in actual):
            continue
        universe_count += 1
        sorted_pruns = sorted(pruns, key=lambda p: p.model_rank or 99)
        top3_sum = sum((sorted_pruns[i].win_probability or 0) for i in range(min(3, len(sorted_pruns)))) * 100
        field_size = len(pruns)
        if top3_sum > top3_max or field_size > field_max:
            continue
        eligible[rid] = {
            "pruns": pruns,
            "actual_top3": actual,
            "top3_sum_pct": top3_sum,
            "field_size": field_size,
        }

    # Use the production bet basket — produces every box (Trio, Quad,
    # Stack×5, Spread×5, Net, Sweep) that the live recommender emits.
    from collections import defaultdict
    by_label = defaultdict(lambda: {"races_with_bet": 0, "races_hit": 0,
                                      "boxes": 0, "boxes_hit": 0})
    by_group = defaultdict(lambda: {"races_with_bet": 0, "races_hit": 0,
                                      "boxes": 0, "boxes_hit": 0})
    races_with_any_box = 0
    for rid, info in eligible.items():
        runners = [{
            "tab_number": p.tab_number,
            "horse_name": p.horse_name,
            "win_probability": p.win_probability,
            "place_probability": p.place_probability,
            "model_rank": p.model_rank,
            "cancelled": bool(p.cancelled),
        } for p in info["pruns"]]
        try:
            bets = _build_bet_basket(runners) or []
        except Exception:
            continue
        if not bets:
            continue
        races_with_any_box += 1
        # Each bet has a strategy_label — group its hit status
        seen_labels: set = set()
        seen_groups: set = set()
        for b in bets:
            label = b.get("strategy_label")
            if not label:
                continue
            group = _STRATEGY_GROUP.get(label, "?")
            hit = _bet_is_hit(b["box_horses"], info["actual_top3"])
            by_label[label]["boxes"] += 1
            by_group[group]["boxes"] += 1
            if hit:
                by_label[label]["boxes_hit"] += 1
                by_group[group]["boxes_hit"] += 1
            if label not in seen_labels:
                by_label[label]["races_with_bet"] += 1
                seen_labels.add(label)
                if hit:
                    by_label[label]["races_hit"] += 1
            elif hit and by_label[label]["races_with_bet"] == 1:
                # already counted but mark hit (single box per label per race)
                pass
            if group not in seen_groups:
                by_group[group]["races_with_bet"] += 1
                seen_groups.add(group)
        # group race-hit increments (one hit per group counts the race)
        for grp in seen_groups:
            any_grp_hit = any(
                _bet_is_hit(b["box_horses"], info["actual_top3"])
                for b in bets
                if _STRATEGY_GROUP.get(b.get("strategy_label"), "?") == grp
            )
            if any_grp_hit:
                by_group[grp]["races_hit"] += 1

    def fmt(d: dict, key: str) -> list:
        out = []
        for k, s in d.items():
            out.append({
                "label" if key == "label" else "group": k,
                "races_with_bet": s["races_with_bet"],
                "races_hit": s["races_hit"],
                "race_hit_rate_pct": round(s["races_hit"]/s["races_with_bet"]*100, 1) if s["races_with_bet"] else 0,
                "boxes_evaluated": s["boxes"],
                "boxes_hit": s["boxes_hit"],
                "box_hit_rate_pct": round(s["boxes_hit"]/s["boxes"]*100, 2) if s["boxes"] else 0,
            })
        out.sort(key=lambda x: -x["race_hit_rate_pct"])
        return out

    strategy_results = fmt(by_label, "label")
    group_results = fmt(by_group, "group")
    return {
        "window_days": days,
        "top3_sum_max_pct": top3_max,
        "field_max": field_max,
        "universe_races": universe_count,
        "filter_matched_races": len(eligible),
        "races_with_any_box": races_with_any_box,
        "coverage_pct": round(len(eligible)/universe_count*100, 1) if universe_count else 0,
        "by_group": group_results,
        "by_strategy": strategy_results,
    }


@app.get("/api/admin/bets/trio-filtered-backtest")
async def trio_filtered_backtest(
    days: int = 90,
    top3_max: float = 55.0,
    field_max: int = 11,
    x_cron_secret: Optional[str] = Header(None),
):
    """Retroactive backtest of the Trio strategy (3-horse box of model's
    top-3 by win probability) on historical races, filtered by
    top-3 sum % ceiling and field-size ceiling. Returns hit rate and
    a small daily-rollup for sanity checking."""
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.tab_number, HistoricalResultRow.horse_name)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2, 3]))
        )).fetchall()
        results_by_race: dict[str, list] = {}
        for rid, pos, tab, name in result_rows:
            results_by_race.setdefault(rid, []).append((pos, tab, name))
        complete = [rid for rid, items in results_by_race.items() if len(items) >= 3]
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(complete))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
        )).scalars().all() if complete else []

    preds_by_race: dict[str, list] = {}
    for p in pred_rows:
        preds_by_race.setdefault(p.race_id, []).append(p)

    tab_per_race: dict[str, dict[str, int]] = {}
    for rid, pruns in preds_by_race.items():
        tab_per_race[rid] = {
            _normalize_horse(p.horse_name): p.tab_number
            for p in pruns if p.tab_number is not None
        }

    universe_count = 0
    filter_count = 0
    hit_count = 0
    by_day: dict[str, dict] = {}
    for rid, pruns in preds_by_race.items():
        if len(pruns) < 3:
            continue
        if rid not in results_by_race:
            continue
        # Resolve actual top-3 tabs
        items = sorted(results_by_race[rid], key=lambda x: x[0])[:3]
        actual_top3: list = []
        for pos, tab, name in items:
            t = tab if tab is not None else tab_per_race.get(rid, {}).get(_normalize_horse(name))
            actual_top3.append(t)
        if any(t is None for t in actual_top3):
            continue
        universe_count += 1
        # Compute features
        sorted_pruns = sorted(pruns, key=lambda p: p.model_rank or 99)
        field_size = len(pruns)
        top3_sum = sum((sorted_pruns[i].win_probability or 0) for i in range(min(3, len(sorted_pruns)))) * 100
        if top3_sum > top3_max or field_size > field_max:
            continue
        # Build Trio box (top-3 ranked horses' tab numbers)
        trio_tabs = [p.tab_number for p in sorted_pruns[:3]]
        if any(t is None for t in trio_tabs):
            continue
        filter_count += 1
        hit = set(trio_tabs) == set(actual_top3)
        if hit:
            hit_count += 1
        date_str = rid.split("_", 1)[0]
        d = by_day.setdefault(date_str, {"races": 0, "hits": 0})
        d["races"] += 1
        if hit:
            d["hits"] += 1

    overall = {
        "window_days": days,
        "top3_sum_max_pct": top3_max,
        "field_max": field_max,
        "universe_races": universe_count,
        "filter_matched_races": filter_count,
        "trio_hits": hit_count,
        "hit_rate_pct": round(hit_count / filter_count * 100, 1) if filter_count else 0,
        "coverage_pct": round(filter_count / universe_count * 100, 1) if universe_count else 0,
    }
    daily = [
        {"date": d, "races": v["races"], "hits": v["hits"],
         "hit_rate_pct": round(v["hits"]/v["races"]*100, 1) if v["races"] else 0}
        for d, v in sorted(by_day.items())
    ]
    return {"overall": overall, "by_day": daily}


@app.get("/api/admin/bets/strategy-shootout")
async def strategy_shootout(
    days: int = 60,
    exclude_dow: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """Backtest every strategy in bets.STRATEGY_REGISTRY side-by-side
    against the last N days of settled racing. Returns per-strategy
    race-coverage, hit rate, total boxes generated, effective
    cost-per-hit, and weekend/weekday/day-of-week breakdown.

    exclude_dow: comma-separated list of weekday names to skip, e.g.
    'Tue,Fri'. Useful for testing whether dropping the worst days
    lifts the overall hit rate.
    """
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 365))
    dow_name_to_int = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    excluded_dows: set[int] = set()
    if exclude_dow:
        for token in exclude_dow.split(","):
            t = token.strip().title()
            if t in dow_name_to_int:
                excluded_dows.add(dow_name_to_int[t])
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    # Pull positions 1-4 so we can evaluate first-four boxes too.
    async with get_session() as session:
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.tab_number, HistoricalResultRow.horse_name)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2, 3, 4]))
        )).fetchall()
        results_by_race: dict[str, list] = {}
        for rid, pos, tab, name in result_rows:
            results_by_race.setdefault(rid, []).append((pos, tab, name))
        complete = [rid for rid, items in results_by_race.items() if len(items) >= 3]

        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(complete))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
        )).scalars().all() if complete else []

    preds_by_race: dict[str, list] = {}
    for p in pred_rows:
        preds_by_race.setdefault(p.race_id, []).append(p)

    # Resolve actual top-4 tab numbers (fall back to name lookup). Top-3
    # is a slice of this — first-four hit checks use the full 4.
    tab_per_race: dict[str, dict[str, int]] = {}
    for rid, pruns in preds_by_race.items():
        tab_per_race[rid] = {
            _normalize_horse(p.horse_name): p.tab_number
            for p in pruns if p.tab_number is not None
        }
    resolved_top4: dict[str, list[int]] = {}
    for rid in complete:
        items = sorted(results_by_race[rid], key=lambda x: x[0])[:4]
        tabs = []
        for pos, tab, name in items:
            t = tab if tab is not None else tab_per_race.get(rid, {}).get(_normalize_horse(name))
            tabs.append(t)
        # Trifecta strategies only need top-3, first-four needs top-4.
        # Skip races where top-3 isn't fully resolved (a hole there
        # invalidates both bet types).
        if len(tabs) >= 3 and all(t is not None for t in tabs[:3]):
            resolved_top4[rid] = tabs

    total_universe = len(resolved_top4)
    resolved_top3 = {rid: tabs[:3] for rid, tabs in resolved_top4.items()}

    strategies = list(_BET_STRATEGIES.items())
    strategy_results: list[dict] = []
    for label, gen in strategies:
        races_evaluated = 0
        races_hit = 0
        total_boxes = 0
        total_perms = 0
        boxes_hit = 0
        # Day-of-week buckets (0=Mon..6=Sun) + weekend split for the
        # weekend-vs-weekday hypothesis test.
        dow_counts = {i: {"races": 0, "hits": 0} for i in range(7)}
        weekend = {"races": 0, "hits": 0}
        weekday = {"races": 0, "hits": 0}
        for rid, top3 in resolved_top3.items():
            pruns = preds_by_race.get(rid, [])
            if len(pruns) < 7:
                continue
            # Apply day-of-week exclusion early so excluded days don't
            # inflate the universe or appear in the per-dow breakdown.
            if excluded_dows:
                try:
                    if datetime.fromisoformat(rid.split("_", 1)[0]).date().weekday() in excluded_dows:
                        continue
                except (ValueError, KeyError):
                    pass
            runners = [{
                "tab_number": p.tab_number,
                "horse_name": p.horse_name,
                "win_probability": p.win_probability,
                "place_probability": p.place_probability,
                "model_rank": p.model_rank,
                "place_model_rank": p.place_model_rank,
                "exotic_model_rank": p.exotic_model_rank,
                "cancelled": bool(p.cancelled),
            } for p in pruns]
            try:
                bets = gen(runners)
            except Exception:
                continue
            if not bets:
                continue
            races_evaluated += 1
            race_hit = False
            for b in bets:
                total_boxes += 1
                total_perms += b.get("num_permutations") or 0
                bt = b.get("bet_type", "trifecta")
                actual = (resolved_top4.get(rid) or []) if bt == "first_four" else top3
                if bt == "first_four" and (len(actual) < 4 or any(t is None for t in actual)):
                    continue
                # Dispatch by structure: standout (banker_tabs),
                # required-in-top3 set, or plain box.
                if b.get("banker_tabs"):
                    from horse_engine.bets import is_standout_hit
                    box_hit = is_standout_hit(b, actual)
                elif b.get("required_in_top3"):
                    required = set(b["required_in_top3"])
                    box_set = set(b["box_horses"])
                    box_hit = (required.issubset(set(top3)) and
                               all(t in box_set for t in top3))
                elif bt == "first_four":
                    box_hit = _bet_is_hit_typed(b["box_horses"], actual, "first_four")
                else:
                    box_hit = _bet_is_hit(b["box_horses"], top3)
                if box_hit:
                    boxes_hit += 1
                    race_hit = True
            if race_hit:
                races_hit += 1
            # Day-of-week classification (Mon=0..Sun=6).
            try:
                race_date = datetime.fromisoformat(rid.split("_", 1)[0]).date()
                dow = race_date.weekday()
                dow_counts[dow]["races"] += 1
                if race_hit:
                    dow_counts[dow]["hits"] += 1
                if dow >= 5:
                    weekend["races"] += 1
                    if race_hit:
                        weekend["hits"] += 1
                else:
                    weekday["races"] += 1
                    if race_hit:
                        weekday["hits"] += 1
            except (ValueError, KeyError):
                pass
        cost_per_race = round(total_perms * 2 / races_evaluated / max(1, total_boxes // races_evaluated), 2) if races_evaluated else 0
        # Total paper stake across the window = total_perms * (stake/perms)
        # but for flexi boxes the user-set stake is per BOX not per perm.
        # Approximate total stake as boxes × $2.
        total_stake = total_boxes * DEFAULT_STAKE if False else total_boxes * 2.0
        cost_per_hit = round(total_stake / boxes_hit, 2) if boxes_hit else None
        cost_per_race_hit = round(total_stake / races_hit, 2) if races_hit else None
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        by_dow = {
            dow_names[i]: {
                "races": dow_counts[i]["races"],
                "hits": dow_counts[i]["hits"],
                "hit_rate_pct": (round(dow_counts[i]["hits"] / dow_counts[i]["races"] * 100, 1)
                                 if dow_counts[i]["races"] else 0),
            }
            for i in range(7)
        }
        wkend_hit = round(weekend["hits"] / weekend["races"] * 100, 1) if weekend["races"] else 0
        wkday_hit = round(weekday["hits"] / weekday["races"] * 100, 1) if weekday["races"] else 0
        strategy_results.append({
            "strategy": label,
            "races_evaluated": races_evaluated,
            "weekend_races": weekend["races"],
            "weekend_hits": weekend["hits"],
            "weekend_hit_rate_pct": wkend_hit,
            "weekday_races": weekday["races"],
            "weekday_hits": weekday["hits"],
            "weekday_hit_rate_pct": wkday_hit,
            "by_dow": by_dow,
            "coverage_pct": round(races_evaluated / total_universe * 100, 1) if total_universe else 0,
            "races_hit": races_hit,
            "race_hit_rate_pct": round(races_hit / races_evaluated * 100, 1) if races_evaluated else 0,
            "total_boxes": total_boxes,
            "boxes_hit": boxes_hit,
            "box_hit_rate_pct": round(boxes_hit / total_boxes * 100, 1) if total_boxes else 0,
            "avg_boxes_per_race": round(total_boxes / races_evaluated, 2) if races_evaluated else 0,
            "total_stake_dollars": round(total_stake, 2),
            "stake_per_race_hit": cost_per_race_hit,
        })
    # Sort: most race hits per dollar staked first (effective hit rate / cost).
    strategy_results.sort(key=lambda s: (s["stake_per_race_hit"] or 1e9))
    return {
        "window_days": days,
        "races_in_universe": total_universe,
        "strategies": strategy_results,
        "note": "stake_per_race_hit = total_stake / races_hit (lower = better). Use ranking column.",
    }


class DividendEntry(BaseModel):
    trifecta: float


@app.post("/api/admin/bets/dividend/{race_id}")
async def admin_set_dividend(
    race_id: str,
    body: DividendEntry,
    x_cron_secret: Optional[str] = Header(None),
):
    """Manually attach a trifecta dividend to a race's bet rows. Used as a
    stopgap until an automated dividend source comes online. Operates on
    any row in the race regardless of settled status — sets the dividend,
    recomputes payout + P&L from is_hit, flips settled=true.

    Body: {"trifecta": 234.50}
    """
    _check_admin(x_cron_secret)
    if body.trifecta is None or body.trifecta <= 0:
        raise HTTPException(400, "trifecta must be > 0")

    async with get_session() as session:
        rows = (await session.execute(
            select(BetRecommendationRow)
            .where(BetRecommendationRow.race_id == race_id)
        )).scalars().all()
        if not rows:
            raise HTTPException(404, f"No bets found for {race_id}")
        # If is_hit hasn't been populated yet, settle from the existing
        # HistoricalResultRow first so we don't have to ask the caller to
        # do two steps.
        needs_settle = any(r.is_hit is None for r in rows)
        if needs_settle:
            try:
                await _settle_bets_for_race(race_id)
                rows = (await session.execute(
                    select(BetRecommendationRow)
                    .where(BetRecommendationRow.race_id == race_id)
                )).scalars().all()
            except Exception as e:
                log.debug("[bets] pre-settle for dividend failed: %s", e)

        updated = 0
        total_pnl = 0.0
        hits = 0
        for r in rows:
            row = await session.get(BetRecommendationRow, r.id)
            if row.is_hit is None:
                continue  # still missing top-3 results
            payout, pnl = _bet_compute_payout(
                row.stake_dollars, row.num_permutations, body.trifecta, bool(row.is_hit)
            )
            row.trifecta_dividend = body.trifecta
            row.payout_dollars = payout
            row.pnl_dollars = pnl
            if not row.settled:
                row.settled = True
                row.settled_at = datetime.utcnow()
            if row.is_hit:
                hits += 1
            total_pnl += pnl
            updated += 1
        await session.commit()
    return {
        "race_id": race_id,
        "trifecta": body.trifecta,
        "rows_updated": updated,
        "hits": hits,
        "total_pnl_dollars": round(total_pnl, 2),
    }


@app.get("/api/admin/bets/needing-dividend")
async def admin_needing_dividend(
    days: int = 3,
    x_cron_secret: Optional[str] = Header(None),
):
    """List recent races with settled bets but no dividend attached —
    the work queue for manual dividend entry. Returns one row per race
    with venue, race #, scheduled_time, hit count, and links to make
    entry quick."""
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 14))
    cutoff = datetime.utcnow() - timedelta(days=days)
    async with get_session() as session:
        rows = (await session.execute(
            select(BetRecommendationRow)
            .where(BetRecommendationRow.recommended_at >= cutoff)
            .where(BetRecommendationRow.settled.is_(True))
            .where(BetRecommendationRow.trifecta_dividend.is_(None))
            .where(BetRecommendationRow.is_hit.is_(True))
            .order_by(BetRecommendationRow.race_id)
        )).scalars().all()
        # Pull scheduled_time per race
        race_ids = list({r.race_id for r in rows})
        sched: dict[str, str] = {}
        if race_ids:
            for rid, st in (await session.execute(
                select(RunnerPredictionRow.race_id, func.max(RunnerPredictionRow.scheduled_time))
                .where(RunnerPredictionRow.race_id.in_(race_ids))
                .group_by(RunnerPredictionRow.race_id)
            )).fetchall():
                if st:
                    sched[rid] = st
    by_race: dict[str, int] = {}
    for r in rows:
        by_race[r.race_id] = by_race.get(r.race_id, 0) + 1
    items = []
    for rid, hit_count in by_race.items():
        _, vc, race_num = _parse_race_id(rid)
        items.append({
            "race_id": rid,
            "venue": vc,
            "race_number": race_num,
            "scheduled_time": sched.get(rid),
            "hit_bets": hit_count,
        })
    items.sort(key=lambda x: x.get("scheduled_time") or "", reverse=True)
    return {"days": days, "races_needing_dividend": items}


@app.post("/api/admin/bets/detect-scratchings")
async def admin_detect_scratchings(
    days: int = 2,
    x_cron_secret: Optional[str] = Header(None),
):
    """Scan recent bet races for horses that don't have a finishing
    position in HistoricalResultRow — they were scratched (or DNF).
    Flag them as cancelled in RunnerPredictionRow and re-settle each
    affected race so the resulting bet rows get marked voided.

    Critical when the proxy capped out and the scheduled scratch-
    detection cron missed updates: corrects retroactive stats in one
    sweep without manual per-race entry."""
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 30))
    cutoff = datetime.utcnow() - timedelta(days=days)
    async with get_session() as session:
        race_ids = (await session.execute(
            select(BetRecommendationRow.race_id).distinct()
            .where(BetRecommendationRow.recommended_at >= cutoff)
        )).scalars().all()

    flagged: list[dict] = []
    re_settled = 0
    for rid in race_ids:
        async with get_session() as session:
            pred_horses = (await session.execute(
                select(RunnerPredictionRow.id, RunnerPredictionRow.tab_number,
                       RunnerPredictionRow.horse_name, RunnerPredictionRow.cancelled)
                .where(RunnerPredictionRow.race_id == rid)
            )).fetchall()
            finished_names = (await session.execute(
                select(HistoricalResultRow.horse_name)
                .where(HistoricalResultRow.race_id == rid)
            )).scalars().all()
        finished_norm: set[str] = {_normalize_horse(n) for n in finished_names if n}
        if not finished_norm:
            continue  # race hasn't been seeded yet — skip
        to_flag: list[tuple] = []  # (row_id, tab, name)
        for pid, tab, name, was_cancelled in pred_horses:
            if was_cancelled:
                continue
            if _normalize_horse(name or "") in finished_norm:
                continue
            to_flag.append((pid, tab, name))
        if not to_flag:
            continue
        async with get_session() as session:
            for pid, tab, name in to_flag:
                row = await session.get(RunnerPredictionRow, pid)
                if row is not None and not row.cancelled:
                    row.cancelled = True
            await session.commit()
        flagged.append({
            "race_id": rid,
            "scratched": [{"tab_number": tab, "horse_name": name} for _, tab, name in to_flag],
        })
        # Force re-settle so existing bet rows pick up voided flags.
        try:
            async with get_session() as session:
                # Clear the settled flag for bet rows in this race so the
                # settle function reprocesses them.
                from sqlalchemy import update as _sa_update
                await session.execute(
                    _sa_update(BetRecommendationRow)
                    .where(BetRecommendationRow.race_id == rid)
                    .values(settled=False)
                )
                await session.commit()
            n = await _settle_bets_for_race(rid)
            re_settled += n
        except Exception as e:
            log.debug("[scratch-detect] re-settle failed for %s: %s", rid, e)
    return {
        "days": days,
        "races_examined": len(race_ids),
        "races_with_scratchings": len(flagged),
        "rows_re_settled": re_settled,
        "details": flagged,
    }


@app.post("/api/admin/meetings/cache-bust")
async def admin_bust_meetings_cache(
    race_date: str,
    x_cron_secret: Optional[str] = Header(None),
):
    """Drop the per-date meetings response cache so the next /api/meetings
    call refetches from RA. Useful when an earlier proxy cap-out poisoned
    the cache with an empty response and we don't want to wait 10 min."""
    _check_admin(x_cron_secret)
    _validate_date(race_date)
    _invalidate_meeting_caches(race_date)
    return {"ok": True, "date": race_date}


@app.get("/api/admin/cancelled-today")
async def admin_cancelled_today(x_cron_secret: Optional[str] = Header(None)):
    """List every cancelled runner for today across the mutable predictions
    table. Used to verify which scratchings the sweep has picked up vs which
    are still missing from our upstream feed."""
    _check_admin(x_cron_secret)
    today = _today_aest().isoformat()
    async with get_session() as session:
        rows = (await session.execute(
            select(
                RunnerPredictionRow.race_id,
                RunnerPredictionRow.horse_name,
                RunnerPredictionRow.model_rank,
                RunnerPredictionRow.win_probability,
            )
            .where(RunnerPredictionRow.race_id.like(f"{today}_%"))
            .where(RunnerPredictionRow.cancelled.is_(True))
            .order_by(RunnerPredictionRow.race_id, RunnerPredictionRow.horse_name)
        )).fetchall()
    return {
        "today": today,
        "count": len(rows),
        "cancelled": [
            {
                "race_id": r.race_id,
                "horse_name": r.horse_name,
                "stale_model_rank": r.model_rank,
                "win_probability": r.win_probability,
            }
            for r in rows
        ],
    }


@app.post("/api/admin/enrich-now")
async def admin_enrich_now(x_cron_secret: Optional[str] = Header(None)):
    """Manually trigger the full morning-enrichment pass (today + next 2 days).
    Use this after a proxy outage or to backfill races that the 8:30am cron
    couldn't reach. Runs in the background so the request returns immediately;
    the actual enrich takes 5-10 minutes for a full day."""
    _check_admin(x_cron_secret)
    asyncio.create_task(_scheduled_enrich())
    return {"ok": True, "queued": True, "note": "Enrich running in background — takes 5-10 min for a full day"}


@app.post("/api/admin/reenrich-race/{race_id}")
async def admin_reenrich_race(race_id: str, x_cron_secret: Optional[str] = Header(None)):
    """
    Re-run the enrich + predict pipeline for a single race and overwrite
    its RunnerPredictionRow / RunnerPredictionHistoryRow rows. Bypasses
    the already-enriched skip and the jump-time-window guard, so it works
    on races that have already run (useful for verifying model/pipeline
    fixes against a specific case without triggering a full-day sweep).

    Loads all calibrations (venue, output-isotonic) so the resulting rows
    reflect the latest pipeline exactly as production would emit them.
    """
    _check_admin(x_cron_secret)
    race_date, venue_code, race_num = _parse_race_id(race_id)
    if not race_date or not venue_code or not race_num:
        raise HTTPException(400, f"malformed race_id: {race_id}")

    client = get_tab_client()
    slug = _meeting_slug(venue_code, race_date)

    stage = "get_meetings"
    try:
        # Meeting details (venue name, state) so parse_race can populate them.
        meetings = await client.get_meetings(race_date)
        m = next((mm for mm in meetings if mm.get("slug") == slug), None)
        venue_name = (m or {}).get("venue", venue_code)
        state = (m or {}).get("state", "")

        stage = "get_meeting_races"
        # Pull via get_meeting_races first — the composite client's get_race
        # needs _slug_to_key populated, which get_meetings alone doesn't
        # always do (especially on a warm-cache path). get_meeting_races
        # calls _fetch_meeting → populates _slug_to_key + returns races.
        raw_races = await client.get_meeting_races(slug, force_fresh=True)
        if not raw_races:
            raise HTTPException(404, f"No races returned for meeting {slug}")

        stage = "get_race"
        full_event = await client.get_race(slug, race_num)
        if not full_event:
            raise HTTPException(
                404,
                f"Race {race_num} not found in meeting {slug} — meeting has {len(raw_races)} race(s)",
            )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("[reenrich-race] %s failed at stage=%s: %s", race_id, stage, e)
        raise HTTPException(500, f"reenrich failed at stage '{stage}': {type(e).__name__}: {e}")

    # Wrap the pipeline stages in try/except and surface the failing
    # stage in the 500 message. Otherwise a raw "Internal Server Error"
    # tells us nothing about which step (parse / enrich / save) broke.
    stage = "parse_race"
    try:
        race = await client.parse_race(full_event, race_date, venue_name, state)
        stage = "load_models"
        async with get_session() as session:
            model = await _load_model(session)
            place_model = await _load_place_model(session)
            exotic_model = await _load_exotic_model(session)
            stage = "inject_accumulated_stats"
            await _inject_accumulated_stats(race, session)
        stage = "load_venue_calibration"
        venue_cal = await _load_venue_calibration()
        stage = "load_output_calibration"
        output_cal = await _load_output_calibration_curve()
        stage = "enrich_and_predict"
        predictions, _ = await enrich_and_predict_race(
            race, model,
            venue_calibration=venue_cal,
            place_model=place_model,
            exotic_model=exotic_model,
            output_calibration_curve=output_cal,
        )
        stage = "save_race_predictions"
        async with get_session() as session:
            await save_race_predictions(
                session,
                race_id,
                [_prediction_to_db_dict(p, race_id, race.scheduled_time, race=race)
                 for p in predictions],
            )
    except Exception as e:
        log.exception("[reenrich-race] %s failed at stage=%s: %s", race_id, stage, e)
        raise HTTPException(500, f"reenrich failed at stage '{stage}': {type(e).__name__}: {e}")

    global _edge_response_cache
    _edge_response_cache = None

    return {
        "ok": True,
        "race_id": race_id,
        "runners_written": len(predictions),
        "top_pick": predictions[0].runner.horse_name if predictions else None,
        "top_pick_win_prob": round(predictions[0].win_prob, 4) if predictions else None,
        "top_pick_place_prob": round(predictions[0].place_prob, 4) if predictions else None,
    }


@app.post("/api/admin/scratch-sweep-now")
async def admin_scratch_sweep_now(x_cron_secret: Optional[str] = Header(None)):
    """Run the today-scratch detection immediately and bust the edge cache so
    cancellations surface on the Edge page within seconds (vs waiting for
    the next 15-min cron tick)."""
    _check_admin(x_cron_secret)
    global _edge_response_cache
    cancelled = await _check_scratches_today()
    _edge_response_cache = None
    return {"ok": True, "newly_cancelled": cancelled}


async def _auto_heal_findings(target: str, findings: list[dict]) -> list[dict]:
    """
    Apply safe, idempotent fixes for a subset of quality-check findings.
    Returns a list of heal actions with outcome:
      { check, action, affected, outcome: "healed" | "skipped" | "failed", detail }
    Only heals items we can derive from existing data — never
    fabricates. Skips anything requiring re-enrichment from RA.
    """
    from sqlalchemy import update as sa_update, delete as sa_delete, func as _func

    actions: list[dict] = []
    global _edge_response_cache

    for f in findings:
        check = f.get("check")

        if check == "top3_missing_tab_number":
            # Backfill via horse_name → tab_number lookup from prediction rows.
            # Same logic as /api/admin/patch-tab-numbers. Idempotent.
            examples = f.get("example") or []
            patched = 0
            async with get_session() as session:
                for rid in examples:
                    # Build lookup table from predictions with tab
                    tab_lookup: dict[str, int] = {}
                    for src in (RunnerPredictionRow, RunnerPredictionHistoryRow):
                        rows = (await session.execute(
                            select(src.horse_name, src.tab_number)
                            .where(src.race_id == rid)
                            .where(src.tab_number.isnot(None))
                        )).fetchall()
                        for name, tab in rows:
                            tab_lookup[_normalize_horse(name)] = tab
                    # Update HistoricalResultRow rows lacking a tab
                    result_rows = (await session.execute(
                        select(HistoricalResultRow)
                        .where(HistoricalResultRow.race_id == rid)
                        .where(HistoricalResultRow.tab_number.is_(None))
                    )).scalars().all()
                    for r in result_rows:
                        t = tab_lookup.get(_normalize_horse(r.horse_name))
                        if t is not None:
                            await session.execute(
                                sa_update(HistoricalResultRow)
                                .where(HistoricalResultRow.id == r.id)
                                .values(tab_number=t)
                            )
                            patched += 1
                await session.commit()
            actions.append({"check": check, "action": "patch_tab_numbers",
                            "affected": patched, "outcome": "healed" if patched else "skipped"})

        elif check == "duplicate_runner_rows":
            # Delete duplicate RunnerPredictionRow rows with null tab_number
            # when another row for the same race+horse has a tab.
            examples = f.get("example") or []
            deleted = 0
            async with get_session() as session:
                for entry in examples:
                    rid = entry.get("race_id")
                    hname = entry.get("horse_name")
                    if not rid or not hname:
                        continue
                    # Any row with tab_number?
                    has_tab = (await session.execute(
                        select(RunnerPredictionRow.id)
                        .where(RunnerPredictionRow.race_id == rid)
                        .where(RunnerPredictionRow.horse_name == hname)
                        .where(RunnerPredictionRow.tab_number.isnot(None))
                        .limit(1)
                    )).scalar_one_or_none()
                    if has_tab is None:
                        continue  # can't safely dedupe
                    res = await session.execute(
                        sa_delete(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == rid)
                        .where(RunnerPredictionRow.horse_name == hname)
                        .where(RunnerPredictionRow.tab_number.is_(None))
                    )
                    deleted += res.rowcount or 0
                await session.commit()
            actions.append({"check": check, "action": "delete_null_tab_duplicates",
                            "affected": deleted, "outcome": "healed" if deleted else "skipped"})

        elif check == "stale_pending_bets":
            # Fire _settle_bets_for_race on each — it either settles or
            # voids-stale (the 24h+ path already exists).
            race_ids = f.get("example_race_ids") or []
            settled = 0
            for rid in race_ids:
                try:
                    n = await _settle_bets_for_race(rid)
                    settled += n or 0
                except Exception as e:
                    log.debug("[auto-heal] settle failed %s: %s", rid, e)
            actions.append({"check": check, "action": "settle_or_void",
                            "affected": settled, "outcome": "healed" if settled else "skipped"})

        elif check == "is_sharp_missing":
            # Recompute is_sharp on rank-1 history rows that have it null.
            # Uses the original gate: rank1_pct>=30 OR top3_sum>=60.
            # /api/admin/backfill-is-sharp does this globally; here scope
            # to the target date only.
            updated = 0
            async with get_session() as session:
                # Fetch rank-1 rows with null is_sharp for the date
                to_update = (await session.execute(
                    select(RunnerPredictionHistoryRow)
                    .where(RunnerPredictionHistoryRow.race_id.like(f"{target}_%"))
                    .where(RunnerPredictionHistoryRow.model_rank == 1)
                    .where(RunnerPredictionHistoryRow.is_sharp.is_(None))
                    .where(RunnerPredictionHistoryRow.source == "live")
                )).scalars().all()
                for r in to_update:
                    # Get top-3 sum for the race
                    top3_rows = (await session.execute(
                        select(RunnerPredictionHistoryRow.win_probability)
                        .where(RunnerPredictionHistoryRow.race_id == r.race_id)
                        .where(RunnerPredictionHistoryRow.model_rank.in_([1, 2, 3]))
                        .where(RunnerPredictionHistoryRow.source == "live")
                    )).scalars().all()
                    top3_sum_pct = sum(p or 0 for p in top3_rows) * 100
                    rank1_pct = (r.win_probability or 0) * 100
                    is_sharp = rank1_pct >= 30 or top3_sum_pct >= 60
                    await session.execute(
                        sa_update(RunnerPredictionHistoryRow)
                        .where(RunnerPredictionHistoryRow.id == r.id)
                        .values(is_sharp=is_sharp)
                    )
                    updated += 1
                await session.commit()
            actions.append({"check": check, "action": "recompute_is_sharp",
                            "affected": updated, "outcome": "healed" if updated else "skipped"})

        elif check == "duplicate_positions_in_result":
            # Delete duplicate HistoricalResultRow rows for the same race+position,
            # keeping the one with tab_number populated (or earliest id if tied).
            race_ids = f.get("example_race_ids") or []
            deleted = 0
            async with get_session() as session:
                for rid in race_ids:
                    rows = (await session.execute(
                        select(HistoricalResultRow)
                        .where(HistoricalResultRow.race_id == rid)
                        .order_by(HistoricalResultRow.position, HistoricalResultRow.id)
                    )).scalars().all()
                    seen_pos: dict[int, int] = {}  # position -> kept row id
                    for r in rows:
                        if r.position is None or r.position <= 0:
                            continue
                        if r.position in seen_pos:
                            # Duplicate — prefer the one with tab_number
                            kept_id = seen_pos[r.position]
                            kept = next((x for x in rows if x.id == kept_id), None)
                            if kept and kept.tab_number is None and r.tab_number is not None:
                                # Swap: delete kept, keep this one
                                await session.execute(
                                    sa_delete(HistoricalResultRow)
                                    .where(HistoricalResultRow.id == kept_id)
                                )
                                seen_pos[r.position] = r.id
                            else:
                                await session.execute(
                                    sa_delete(HistoricalResultRow)
                                    .where(HistoricalResultRow.id == r.id)
                                )
                            deleted += 1
                        else:
                            seen_pos[r.position] = r.id
                await session.commit()
            actions.append({"check": check, "action": "dedupe_positions",
                            "affected": deleted, "outcome": "healed" if deleted else "skipped"})

        elif check == "scratched_mutable_history_mismatch":
            # Sync cancelled flag between mutable and history for the mismatched runners.
            entries = f.get("mismatches") or []
            synced = 0
            async with get_session() as session:
                for e in entries:
                    rid = e.get("race_id")
                    hname = e.get("horse_name")
                    if not rid or not hname:
                        continue
                    # Union: if EITHER has cancelled=true, both should
                    await session.execute(
                        sa_update(RunnerPredictionRow)
                        .where(RunnerPredictionRow.race_id == rid)
                        .where(RunnerPredictionRow.horse_name == hname)
                        .values(cancelled=True)
                    )
                    await session.execute(
                        sa_update(RunnerPredictionHistoryRow)
                        .where(RunnerPredictionHistoryRow.race_id == rid)
                        .where(RunnerPredictionHistoryRow.horse_name == hname)
                        .values(cancelled=True)
                    )
                    synced += 1
                await session.commit()
            actions.append({"check": check, "action": "sync_cancelled_flag",
                            "affected": synced, "outcome": "healed" if synced else "skipped"})

        elif check == "stale_response_cache":
            _edge_response_cache = None
            actions.append({"check": check, "action": "bust_edge_cache",
                            "affected": 1, "outcome": "healed"})

    return actions


async def _run_quality_check(target: str) -> dict:
    """
    Core quality-check logic — shared between the admin endpoint and
    the nightly cron. Returns the report dict.
    """

    critical: list[dict] = []
    warning: list[dict] = []
    info: list[dict] = []

    # Compute how old the target date is up-front so downstream checks
    # can use stale_hours without a forward reference.
    from datetime import datetime as _dt_early
    _now_early = _dt_early.utcnow().replace(tzinfo=timezone.utc)
    try:
        _target_dt_early = _dt_early.strptime(target, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        stale_hours = (_now_early - _target_dt_early).total_seconds() / 3600
    except Exception:
        stale_hours = 0

    async with get_session() as session:
        # 1. Meetings/venues covered by predictions (mutable + history)
        mut_race_ids = set((await session.execute(
            select(RunnerPredictionRow.race_id).distinct()
            .where(RunnerPredictionRow.race_id.like(f"{target}_%"))
        )).scalars().all())
        hist_race_ids = set((await session.execute(
            select(RunnerPredictionHistoryRow.race_id).distinct()
            .where(RunnerPredictionHistoryRow.race_id.like(f"{target}_%"))
        )).scalars().all())

        # 2. HistoricalResultRow rows
        hr_result_rows = (await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.like(f"{target}_%"))
        )).scalars().all()
        hr_race_ids = {r.race_id for r in hr_result_rows}
        hr_races_with_position = {r.race_id for r in hr_result_rows if r.position and r.position > 0}
        hr_races_with_tab = {r.race_id for r in hr_result_rows if r.tab_number is not None}

        # 3. Predictions with no matching historical result
        settled_pred = mut_race_ids & hr_race_ids
        pending_pred = mut_race_ids - hr_race_ids

        # 4. Rank-1 pick rows for calibration + is_sharp coverage
        rank1_hist = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{target}_%"))
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )).scalars().all()

    # ── Race count integrity ──────────────────────────────────────
    n_mut = len(mut_race_ids)
    n_hist = len(hist_race_ids)
    n_settled = len(settled_pred)
    n_pending = len(pending_pred)
    n_history_only = len(hist_race_ids - mut_race_ids)

    info.append({
        "check": "race_counts",
        "predictions_mutable": n_mut,
        "predictions_history": n_hist,
        "results_seeded": len(hr_race_ids),
        "predicted_and_settled": n_settled,
        "predicted_but_pending_result": n_pending,
        "history_only_no_mutable": n_history_only,
    })

    if n_history_only > 0:
        warning.append({
            "check": "history_without_mutable",
            "n_races": n_history_only,
            "reason": "Race snapshotted at 9am but mutable row wiped after; user may see stale data",
        })

    # ── Coverage vs "races that actually ran" (best proxy: results seeded) ──
    # Break down by venue and filter out blocklisted venues (they're
    # intentionally not enriched — flagging them as "missed" is a false
    # positive we noticed on the first run.).
    missing_by_venue: dict[str, int] = {}
    for rid in hr_race_ids - mut_race_ids:
        try:
            venue = rid.split("_")[1]
            if venue.lower() in _SKIP_ENRICHMENT_VENUES:
                continue  # intentional skip, not a coverage miss
            missing_by_venue[venue] = missing_by_venue.get(venue, 0) + 1
        except IndexError:
            continue
    non_blocklisted_gap = sum(missing_by_venue.values())
    if non_blocklisted_gap > 0:
        warning.append({
            "check": "results_without_predictions",
            "n_races": non_blocklisted_gap,
            "venues_affected": sorted(
                [{"venue": v, "unpredicted_races": n} for v, n in missing_by_venue.items()],
                key=lambda x: -x["unpredicted_races"],
            )[:10],
            "reason": "HistoricalResultRow has races we never enriched — venue tile will under-count wins/place figures because those races don't exist in our prediction data (blocklisted venues excluded)",
        })

    coverage_pct = round(n_settled / len(hr_race_ids) * 100, 1) if hr_race_ids else 0
    info.append({
        "check": "prediction_coverage",
        "coverage_pct": coverage_pct,
        "expected_min": 85.0,
        "notes": f"day card marks complete=True when >=85%",
    })

    # ── Missing tab_number on top-3 finishers ─────────────────────
    seeded_races_missing_tab: list[str] = []
    for rid in hr_race_ids:
        top3 = [r for r in hr_result_rows if r.race_id == rid and r.position in (1, 2, 3)]
        if len(top3) == 3 and any(r.tab_number is None for r in top3):
            seeded_races_missing_tab.append(rid)
    if seeded_races_missing_tab:
        critical.append({
            "check": "top3_missing_tab_number",
            "n_races": len(seeded_races_missing_tab),
            "example": seeded_races_missing_tab[:20],  # all-in for heal
            "reason": "Bet settlement can't map positions to bet boxes; stays 'pending' forever",
        })

    # ── Duplicate RunnerPredictionRow rows (same race + horse) ────
    async with get_session() as session:
        dup_rows = (await session.execute(
            select(
                RunnerPredictionRow.race_id,
                RunnerPredictionRow.horse_name,
                func.count(RunnerPredictionRow.id).label("n"),
            )
            .where(RunnerPredictionRow.race_id.like(f"{target}_%"))
            .group_by(RunnerPredictionRow.race_id, RunnerPredictionRow.horse_name)
            .having(func.count(RunnerPredictionRow.id) > 1)
        )).all()
    if dup_rows:
        warning.append({
            "check": "duplicate_runner_rows",
            "n_pairs": len(dup_rows),
            "example": [{"race_id": r[0], "horse_name": r[1], "count": r[2]}
                        for r in dup_rows[:20]],
            "reason": "Same horse appears multiple times for the same race; matched_pred lookup can grab the wrong row (null tab_number bug on OCCULT class)",
        })

    # ── Duplicate positions in HistoricalResultRow ────────────────
    dup_pos_races: list[str] = []
    for rid in hr_race_ids:
        seen = set()
        for r in hr_result_rows:
            if r.race_id != rid or not r.position or r.position <= 0:
                continue
            if r.position in seen:
                dup_pos_races.append(rid)
                break
            seen.add(r.position)
    if dup_pos_races:
        critical.append({
            "check": "duplicate_positions_in_result",
            "n_races": len(dup_pos_races),
            "example_race_ids": dup_pos_races[:20],
            "reason": "Same finishing position twice per race; actual_top3 becomes wrong → false bet hits",
        })

    # ── Bet ledger: unsettled but race jumped > 24h ago ───────────
    from datetime import datetime as _dt
    _now = _dt.utcnow().replace(tzinfo=timezone.utc)
    stale_bet_races: list[str] = []
    async with get_session() as session:
        unsettled = (await session.execute(
            select(BetRecommendationRow.race_id).distinct()
            .where(BetRecommendationRow.race_id.like(f"{target}_%"))
            .where(BetRecommendationRow.settled.is_(False))
        )).scalars().all()
        for rid in unsettled:
            sched = (await session.execute(
                select(func.max(RunnerPredictionRow.scheduled_time))
                .where(RunnerPredictionRow.race_id == rid)
            )).scalar()
            try:
                if sched:
                    jump = _dt.fromisoformat(str(sched).replace("Z", "+00:00"))
                    if (_now - jump).total_seconds() > 24 * 3600:
                        stale_bet_races.append(rid)
                elif stale_hours >= 24:
                    stale_bet_races.append(rid)  # no sched_time but date is old
            except Exception:
                pass
    if stale_bet_races:
        critical.append({
            "check": "stale_pending_bets",
            "n_races": len(stale_bet_races),
            "example_race_ids": stale_bet_races[:20],
            "reason": "Bets still marked unsettled >24h after jump — settle sweep failed or race never seeded",
        })

    # ── Scratched flag mismatch: mutable vs history ───────────────
    async with get_session() as session:
        mut_cancelled = (await session.execute(
            select(RunnerPredictionRow.race_id, RunnerPredictionRow.horse_name)
            .where(RunnerPredictionRow.race_id.like(f"{target}_%"))
            .where(RunnerPredictionRow.cancelled.is_(True))
        )).all()
        hist_cancelled = (await session.execute(
            select(RunnerPredictionHistoryRow.race_id, RunnerPredictionHistoryRow.horse_name)
            .where(RunnerPredictionHistoryRow.race_id.like(f"{target}_%"))
            .where(RunnerPredictionHistoryRow.cancelled.is_(True))
        )).all()
    mut_set = {(rid, hn) for rid, hn in mut_cancelled}
    hist_set = {(rid, hn) for rid, hn in hist_cancelled}
    mismatches = list(mut_set ^ hist_set)  # symmetric difference
    if mismatches:
        warning.append({
            "check": "scratched_mutable_history_mismatch",
            "n_runners": len(mismatches),
            "mismatches": [{"race_id": r, "horse_name": h} for r, h in mismatches[:20]],
            "reason": "cancelled=true set on one table but not the other — downstream can double-count or miss the scratch",
        })

    # ── Place-prob sanity (per horse: P(top-3) >= P(top-1)) ──────
    place_violations = 0
    place_broken_races = set()
    sum_place_off_races: list[str] = []
    async with get_session() as session:
        active_pred = (await session.execute(
            select(
                RunnerPredictionRow.race_id,
                RunnerPredictionRow.horse_name,
                RunnerPredictionRow.win_probability,
                RunnerPredictionRow.place_probability,
            )
            .where(RunnerPredictionRow.race_id.like(f"{target}_%"))
            .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
        )).all()
    by_race: dict[str, list] = {}
    for rid, hn, wp, pp in active_pred:
        by_race.setdefault(rid, []).append((hn, wp or 0, pp or 0))
    for rid, rows in by_race.items():
        for hn, wp, pp in rows:
            if wp > 0.05 and pp < wp:  # meaningful only when both non-trivial
                place_violations += 1
                place_broken_races.add(rid)
        sum_p = sum(r[2] for r in rows)
        n = len(rows)
        expected = 3.0 if n >= 8 else 2.0 if n >= 5 else 1.0
        if abs(sum_p - expected) > expected * 0.5 and n >= 5:  # off by >50%
            sum_place_off_races.append(rid)
    if place_violations > 0:
        critical.append({
            "check": "place_prob_less_than_win_prob",
            "n_violations": place_violations,
            "n_races": len(place_broken_races),
            "example_race_ids": list(place_broken_races)[:20],
            "reason": "Structurally impossible: P(top-3) < P(top-1); place model broken (2a587db class)",
        })
    if sum_place_off_races:
        warning.append({
            "check": "place_prob_sum_off_from_places_count",
            "n_races": len(sum_place_off_races),
            "example_race_ids": sum_place_off_races[:20],
            "reason": "Field-wide place-prob sum should ≈ places count (3 for 8+ runners); off by >50% means place model output is unnormalized",
        })

    # ── URL health: internal (our site) + external (data sources) ─
    # Runs a small GET pass against each URL, flags non-200.
    # 5s per probe, done in parallel.
    import httpx as _httpx_probe

    _FRONTEND_HOST = "https://horse-racing-predictions.funkyiq.com"
    internal_urls = [
        f"{_FRONTEND_HOST}/",
        f"{_FRONTEND_HOST}/edge.html",
        f"{_FRONTEND_HOST}/index.html",
        f"{_FRONTEND_HOST}/hotseat.html",
        f"{_FRONTEND_HOST}/funkmeup.html",
        f"{_FRONTEND_HOST}/bets.html",
        f"{_FRONTEND_HOST}/dashboard.html",
        "https://web-production-dec62.up.railway.app/api/health",
        "https://web-production-dec62.up.railway.app/api/edge",
    ]
    from horse_engine.clients.racing_australia import _ra_date as _ra_date_fn
    external_urls = [
        # RA direct (residential IP should always work — verifies RA isn't down)
        "https://www.racingaustralia.horse/FreeFields/Calendar.aspx?State=NSW",
        "https://www.racingaustralia.horse/FreeFields/Calendar.aspx?State=VIC",
        f"https://www.racingaustralia.horse/FreeFields/Results.aspx?"
        f"Key={_ra_date_fn(target)},NSW,Royal%20Randwick",
        # Our RA proxy droplet — health check for the actual path production uses
        "https://170-64-130-33.sslip.io/proxy/FreeFields/Calendar.aspx?State=NSW",
    ]

    async def _probe(client: "_httpx_probe.AsyncClient", url: str) -> dict:
        try:
            r = await client.get(url, timeout=5.0, follow_redirects=True)
            return {"url": url, "status": r.status_code,
                    "ok": 200 <= r.status_code < 400}
        except Exception as e:
            return {"url": url, "status": 0, "ok": False,
                    "error": f"{type(e).__name__}: {str(e)[:100]}"}

    try:
        async with _httpx_probe.AsyncClient() as client:
            probes = await asyncio.gather(
                *[_probe(client, u) for u in internal_urls + external_urls]
            )
        broken_internal = [p for p in probes[:len(internal_urls)] if not p["ok"]]
        broken_external = [p for p in probes[len(internal_urls):] if not p["ok"]]
        if broken_internal:
            critical.append({
                "check": "internal_urls_down",
                "n_urls": len(broken_internal),
                "urls": broken_internal,
                "reason": "One or more of our own pages/endpoints returned non-200 — users may be seeing errors",
            })
        if broken_external:
            # Distinguish "RA is down" (rare) from "our proxy droplet is down" (recurring).
            # sslip.io hostname is our droplet; anything else is real RA.
            proxy_broken = any("sslip.io" in (u.get("url") or "") for u in broken_external)
            ra_broken = any("racingaustralia.horse" in (u.get("url") or "") for u in broken_external)
            if proxy_broken and not ra_broken:
                critical.append({
                    "check": "ra_proxy_droplet_down",
                    "urls": [u for u in broken_external if "sslip.io" in (u.get("url") or "")],
                    "reason": "The DigitalOcean RA proxy is unreachable but RA itself is up. Self-heal (ra-proxy-healthcheck.timer) should have restarted it; if you see this in the morning report, the droplet is either powered off, out of memory globally, or the self-heal cron failed to install.",
                    "remediation": "SSH to 170.64.130.33, run: systemctl status ra-proxy-healthcheck.timer; tail -50 /var/log/ra-proxy-healthcheck.log",
                })
            else:
                critical.append({
                    "check": "external_urls_down",
                    "n_urls": len(broken_external),
                    "urls": broken_external,
                    "reason": "Upstream data source returning non-200 — seed/enrich pipeline will silently starve",
                })
        info.append({
            "check": "url_health",
            "internal_ok": len(internal_urls) - len(broken_internal),
            "internal_total": len(internal_urls),
            "external_ok": len(external_urls) - len(broken_external),
            "external_total": len(external_urls),
        })
    except Exception as e:
        warning.append({
            "check": "url_health_probe_failed",
            "reason": f"Couldn't run URL health checks: {type(e).__name__}: {str(e)[:120]}",
        })

    # ── Pending picks older than 24h — RA never published or we lost the race ──
    if stale_hours > 24 and n_pending > 0:
        critical.append({
            "check": "stale_pending_seeds",
            "n_races": n_pending,
            "hours_stale": round(stale_hours, 1),
            "example_race_ids": sorted(pending_pred)[:5],
            "reason": "Races predicted but never seeded — RA didn't publish OR seed pipeline broken",
        })

    # ── is_sharp coverage on rank-1 rows ──────────────────────────
    n_rank1 = len(rank1_hist)
    n_is_sharp_set = sum(1 for r in rank1_hist if r.is_sharp is not None)
    if n_rank1 > 0 and n_is_sharp_set < n_rank1:
        warning.append({
            "check": "is_sharp_missing",
            "n_missing": n_rank1 - n_is_sharp_set,
            "n_total_rank1": n_rank1,
            "reason": "Sharp flag not populated; sharp-mode metrics will under-count",
        })

    # ── Sportsbet-availability consistency per venue ──────────────
    # Historical rows don't carry sportsbet_available (computed at
    # response time), so just check the blocklist is applied
    # consistently — any predicted venue in the blocklist implies
    # enrichment leaked through the skip guard.
    predicted_venues = {rid.split("_")[1] for rid in mut_race_ids | hist_race_ids
                        if len(rid.split("_")) >= 3}
    leaked_venues = predicted_venues & _SKIP_ENRICHMENT_VENUES
    if leaked_venues:
        warning.append({
            "check": "blocklisted_venue_leaked_through_enrich",
            "venues": sorted(leaked_venues),
            "reason": "Predictions exist for venues we're supposed to skip — enrichment gate missed",
        })

    # ── Win-rate sanity: baseline should be 20-35% for rank-1 ──
    if n_rank1 > 20:
        rank1_with_result = [r for r in rank1_hist
                             if any(hr.race_id == r.race_id and hr.position and hr.position > 0
                                    for hr in hr_result_rows)]
        rank1_wins = 0
        for r in rank1_with_result:
            winner_row = next((hr for hr in hr_result_rows
                              if hr.race_id == r.race_id and hr.position == 1), None)
            if winner_row and _normalize_horse(winner_row.horse_name) == _normalize_horse(r.horse_name):
                rank1_wins += 1
        if rank1_with_result:
            win_rate = rank1_wins / len(rank1_with_result)
            info.append({
                "check": "rank1_win_rate",
                "wins": rank1_wins,
                "settled_races": len(rank1_with_result),
                "win_rate_pct": round(win_rate * 100, 1),
                "expected_band": "20-35%",
            })
            if win_rate < 0.10 and len(rank1_with_result) >= 20:
                critical.append({
                    "check": "win_rate_collapse",
                    "win_rate_pct": round(win_rate * 100, 1),
                    "n": len(rank1_with_result),
                    "reason": "Sub-10% win rate suggests systemic pipeline breakage or bad data",
                })

    return {
        "date": target,
        "hours_since_start_of_day_utc": round(stale_hours, 1),
        "critical": critical,
        "warning": warning,
        "info": info,
        "summary": {
            "critical_count": len(critical),
            "warning_count": len(warning),
            "info_count": len(info),
        },
    }


@app.get("/api/admin/quality-check")
async def admin_quality_check(
    date: Optional[str] = None,
    heal: bool = Query(False, description="If true, apply auto-heal after the initial check and re-verify"),
    x_cron_secret: Optional[str] = Header(None),
):
    """On-demand data-integrity report for a given day.

    heal=true triggers the same auto-heal engine the nightly cron uses:
    initial check → apply safe/idempotent fixes → re-run to verify.
    Returns {initial, heal_actions, final, summary}.
    """
    _check_admin(x_cron_secret)
    if date:
        _validate_date(date)
        target = date
    else:
        target = (_today_aest() - timedelta(days=1)).isoformat()
    try:
        initial = await _run_quality_check(target)
    except Exception as e:
        import traceback
        tb = traceback.format_exc().splitlines()[-8:]
        raise HTTPException(500, f"{type(e).__name__}: {e} | tb: {' | '.join(tb)}")

    if not heal:
        return initial

    healable = (initial.get("critical") or []) + (initial.get("warning") or [])
    try:
        heal_actions = await _auto_heal_findings(target, healable)
    except Exception as e:
        heal_actions = [{"check": "_engine", "action": "abort",
                         "outcome": "failed", "detail": str(e)[:200]}]
    try:
        final = await _run_quality_check(target)
    except Exception:
        final = initial

    return {
        "date": target,
        "initial": initial,
        "heal_actions": heal_actions,
        "final": final,
        "summary": {
            "initial_critical": len(initial.get("critical") or []),
            "initial_warning": len(initial.get("warning") or []),
            "final_critical": len(final.get("critical") or []),
            "final_warning": len(final.get("warning") or []),
            "heals_applied": sum(1 for a in heal_actions if a.get("outcome") == "healed"),
            "heals_skipped": sum(1 for a in heal_actions if a.get("outcome") == "skipped"),
        },
    }


@app.get("/api/admin/quality-check/history")
async def admin_quality_check_history(
    limit: int = 30,
    x_cron_secret: Optional[str] = Header(None),
):
    """Recent quality-check results from the nightly cron. Newest first."""
    _check_admin(x_cron_secret)
    limit = max(1, min(int(limit), 90))
    async with get_session() as session:
        rows = (await session.execute(
            select(QualityCheckRow)
            .order_by(QualityCheckRow.ran_at.desc())
            .limit(limit)
        )).scalars().all()
    return {
        "rows": [
            {
                "check_date": r.check_date,
                "ran_at": r.ran_at.isoformat() if r.ran_at else None,
                "critical_count": r.critical_count,
                "warning_count": r.warning_count,
                "info_count": r.info_count,
            }
            for r in rows
        ],
    }


@app.get("/api/admin/quality-check/history/{check_date}")
async def admin_quality_check_history_detail(
    check_date: str,
    x_cron_secret: Optional[str] = Header(None),
):
    """Full report for a specific date from the nightly cron history."""
    _check_admin(x_cron_secret)
    _validate_date(check_date)
    async with get_session() as session:
        row = (await session.execute(
            select(QualityCheckRow)
            .where(QualityCheckRow.check_date == check_date)
            .order_by(QualityCheckRow.ran_at.desc())
        )).scalars().first()
    if not row:
        raise HTTPException(404, f"No quality-check row for {check_date}")
    return json.loads(row.report_json or "{}")


async def _scheduled_racing_hours_heal():
    """
    Racing-hours cron — runs the quality check + auto-heal against
    TODAY every 3h during racing hours (10:00, 13:00, 16:00, 19:00
    AEST). Catches duplicate-position seeds and null-tab issues
    within a couple of hours of the offending race jumping, so
    results render on frontend pages without waiting for the next
    04:00 quality-check cron.

    Only heals — doesn't persist a full quality_checks row like the
    nightly cron does (that runs against yesterday to give a stable
    day-boundary metric).
    """
    target = _today_aest().isoformat()
    try:
        initial = await _run_quality_check(target)
        healable = (initial.get("critical") or []) + (initial.get("warning") or [])
        actions = await _auto_heal_findings(target, healable)
        applied = sum(1 for a in actions if a.get("outcome") == "healed")
        if applied:
            log.info("[racing-hours-heal %s] applied %d heals: %s",
                     target, applied,
                     ", ".join(f"{a['check']}:{a['affected']}" for a in actions
                               if a.get("outcome") == "healed"))
        # Bust the edge cache so cleaned-up results surface immediately.
        global _edge_response_cache
        _edge_response_cache = None
    except Exception as e:
        log.exception("[racing-hours-heal] %s failed: %s", target, e)


async def _scheduled_quality_check():
    """
    Nightly cron entry — runs the quality check against yesterday,
    applies safe auto-heals, re-runs to verify, and persists the
    combined report. Logs at severity-matching levels for monitoring.
    """
    target = (_today_aest() - timedelta(days=1)).isoformat()
    try:
        initial = await _run_quality_check(target)
    except Exception as e:
        log.exception("[cron] quality-check initial pass failed for %s: %s", target, e)
        return

    # Apply auto-heal for both critical and warning findings — anything
    # in the heal engine's whitelist gets fixed idempotently.
    healable = (initial.get("critical") or []) + (initial.get("warning") or [])
    try:
        heal_actions = await _auto_heal_findings(target, healable)
    except Exception as e:
        log.exception("[cron] auto-heal failed for %s: %s", target, e)
        heal_actions = [{"check": "_engine", "action": "abort",
                         "outcome": "failed", "detail": str(e)[:200]}]

    # Re-run to see what remains after heals.
    try:
        final = await _run_quality_check(target)
    except Exception as e:
        log.exception("[cron] quality-check verify pass failed for %s: %s", target, e)
        final = initial  # fall back — better to persist something

    combined = {
        "date": target,
        "initial": initial,
        "heal_actions": heal_actions,
        "final": final,
        "summary": {
            "initial_critical": len(initial.get("critical") or []),
            "initial_warning": len(initial.get("warning") or []),
            "final_critical": len(final.get("critical") or []),
            "final_warning": len(final.get("warning") or []),
            "heals_applied": sum(1 for a in heal_actions if a.get("outcome") == "healed"),
            "heals_skipped": sum(1 for a in heal_actions if a.get("outcome") == "skipped"),
        },
    }

    async with get_session() as session:
        session.add(QualityCheckRow(
            check_date=target,
            critical_count=len(final.get("critical") or []),
            warning_count=len(final.get("warning") or []),
            info_count=len(final.get("info") or []),
            report_json=json.dumps(combined),
        ))
        await session.commit()

    if final.get("critical"):
        for f in final["critical"]:
            log.error("[quality-check %s] CRITICAL still present after heal: %s: %s",
                      target, f.get("check"), f)
    elif final.get("warning"):
        for f in final["warning"]:
            log.warning("[quality-check %s] WARNING remaining: %s: %s",
                        target, f.get("check"), f)
    else:
        log.info("[quality-check %s] clean after heals (%d applied)",
                 target, combined["summary"]["heals_applied"])


@app.get("/api/admin/morning-report")
async def admin_morning_report(x_cron_secret: Optional[str] = Header(None)):
    """
    Morning summary — latest nightly quality-check run, formatted for
    a quick skim. Read this each morning:
      - date + when the check ran
      - what was auto-healed (fix action + count)
      - what critical/warning findings REMAIN after heal (need manual)
      - health snapshot (URL probes, coverage %, rank-1 win rate)
    """
    _check_admin(x_cron_secret)
    async with get_session() as session:
        row = (await session.execute(
            select(QualityCheckRow)
            .order_by(QualityCheckRow.ran_at.desc())
            .limit(1)
        )).scalars().first()
    if not row:
        return {"status": "no_data", "note": "Nightly quality-check hasn't run yet. Try /api/admin/quality-check?date=yesterday to see it live."}

    payload = json.loads(row.report_json or "{}")
    initial = payload.get("initial") or {}
    final = payload.get("final") or {}
    heals = payload.get("heal_actions") or []

    healed = [h for h in heals if h.get("outcome") == "healed"]
    skipped = [h for h in heals if h.get("outcome") == "skipped"]

    # Extract key metrics from info findings
    def _pull(bucket_list: list[dict], check_name: str) -> Optional[dict]:
        return next((f for f in bucket_list if f.get("check") == check_name), None)

    info = final.get("info") or []
    coverage = _pull(info, "prediction_coverage")
    win_rate = _pull(info, "rank1_win_rate")
    url_health = _pull(info, "url_health")

    return {
        "date": payload.get("date"),
        "ran_at": row.ran_at.isoformat() if row.ran_at else None,
        "verdict": (
            "all_clear" if not final.get("critical") and not final.get("warning")
            else "warnings_only" if not final.get("critical")
            else "critical"
        ),
        "auto_healed": [
            {"issue": h.get("check"), "action": h.get("action"),
             "affected": h.get("affected")}
            for h in healed
        ],
        "still_needs_attention": {
            "critical": final.get("critical") or [],
            "warning": final.get("warning") or [],
        },
        "health_snapshot": {
            "prediction_coverage_pct": (coverage or {}).get("coverage_pct"),
            "rank1_win_rate_pct": (win_rate or {}).get("win_rate_pct"),
            "url_health_internal": f"{(url_health or {}).get('internal_ok', '?')}/{(url_health or {}).get('internal_total', '?')}",
            "url_health_external": f"{(url_health or {}).get('external_ok', '?')}/{(url_health or {}).get('external_total', '?')}",
        },
        "summary_counts": payload.get("summary") or {},
    }


@app.post("/api/admin/purge-blocklisted-venues")
async def purge_blocklisted_venues(
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Delete existing rows for venues in _SKIP_ENRICHMENT_VENUES so
    they stop appearing on Edge / Lounge / Lab immediately, rather
    than lingering until they naturally age out. Idempotent — deletes
    only rows whose race_id contains a blocklisted venue slug.

    Wipes: RunnerPredictionRow, RunnerPredictionHistoryRow,
    BetRecommendationRow. Leaves HistoricalResultRow alone (settled
    results are model-training data — no harm keeping them).
    """
    _check_admin(x_cron_secret)
    from sqlalchemy import delete as sa_delete
    from sqlalchemy import or_

    if not _SKIP_ENRICHMENT_VENUES:
        return {"ok": True, "purged": 0, "venues": []}

    conditions_mut = or_(*(
        RunnerPredictionRow.race_id.like(f"%_{v}_%")
        for v in _SKIP_ENRICHMENT_VENUES
    ))
    conditions_hist = or_(*(
        RunnerPredictionHistoryRow.race_id.like(f"%_{v}_%")
        for v in _SKIP_ENRICHMENT_VENUES
    ))
    conditions_bet = or_(*(
        BetRecommendationRow.race_id.like(f"%_{v}_%")
        for v in _SKIP_ENRICHMENT_VENUES
    ))

    async with get_session() as session:
        mut = await session.execute(sa_delete(RunnerPredictionRow).where(conditions_mut))
        hist = await session.execute(sa_delete(RunnerPredictionHistoryRow).where(conditions_hist))
        bets = await session.execute(sa_delete(BetRecommendationRow).where(conditions_bet))
        await session.commit()

    global _edge_response_cache
    _edge_response_cache = None

    return {
        "ok": True,
        "venues": sorted(_SKIP_ENRICHMENT_VENUES),
        "deleted": {
            "runner_predictions": mut.rowcount or 0,
            "runner_prediction_history": hist.rowcount or 0,
            "bet_recommendations": bets.rowcount or 0,
        },
    }


@app.post("/api/admin/scratch-runner")
async def admin_scratch_runner(
    race_id: str = Query(...),
    horse_name: str = Query(...),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Manually mark a single runner as scratched. Used when a bookmaker
    (Sportsbet/TAB) has scratched the horse but RA Acceptances.aspx
    hasn't updated yet, so the automatic sweep can't detect it.

    Sets cancelled=True on both the mutable and history rows for
    (race_id, horse_name), busts the edge cache, and invalidates the
    per-venue meeting cache so the change surfaces immediately.

    Idempotent — a re-run against an already-cancelled runner is a
    no-op.
    """
    _check_admin(x_cron_secret)
    from sqlalchemy import update as sa_update
    global _edge_response_cache

    target = _normalize_horse(horse_name)
    async with get_session() as session:
        mut_rows = (await session.execute(
            select(RunnerPredictionRow.horse_name)
            .where(RunnerPredictionRow.race_id == race_id)
        )).scalars().all()
        matched_mut = [n for n in mut_rows if _normalize_horse(n) == target]
        hist_rows = (await session.execute(
            select(RunnerPredictionHistoryRow.horse_name)
            .where(RunnerPredictionHistoryRow.race_id == race_id)
        )).scalars().all()
        matched_hist = [n for n in hist_rows if _normalize_horse(n) == target]

        if not matched_mut and not matched_hist:
            raise HTTPException(404, f"no runner matching '{horse_name}' in {race_id}")

        mut_result = await session.execute(
            sa_update(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id == race_id)
            .where(RunnerPredictionRow.horse_name.in_(matched_mut or [""]))
            .values(cancelled=True)
        ) if matched_mut else None
        hist_result = await session.execute(
            sa_update(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id == race_id)
            .where(RunnerPredictionHistoryRow.horse_name.in_(matched_hist or [""]))
            .values(cancelled=True)
        ) if matched_hist else None
        await session.commit()

    _edge_response_cache = None
    try:
        _, vc, _ = _parse_race_id(race_id)
        _invalidate_meeting_caches(race_id[:10], vc)
    except Exception:
        pass

    # Regenerate Lab bets for the race so the shrunk field re-picks
    # a new basket (e.g. rank-1 scratch → new rank-1 leads the box).
    # Only regenerate for races that haven't jumped — post-jump reruns
    # would create bet rows for a race that's already run, which the
    # settlement path would then have to handle as edge cases.
    regenerated = 0
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    async with get_session() as session:
        sched = (await session.execute(
            select(func.max(RunnerPredictionRow.scheduled_time))
            .where(RunnerPredictionRow.race_id == race_id)
        )).scalar()
    race_in_future = True
    if sched:
        try:
            race_in_future = datetime.fromisoformat(str(sched).replace("Z", "+00:00")) > now_utc
        except (ValueError, TypeError):
            race_in_future = True
    if race_in_future:
        try:
            regenerated = await _generate_bets_for_race(race_id, regenerate=True)
        except Exception as e:
            log.warning("[scratch-runner] bet regen failed for %s: %s", race_id, e)

    return {
        "ok": True,
        "race_id": race_id,
        "matched_horse": matched_mut[0] if matched_mut else (matched_hist[0] if matched_hist else None),
        "mutable_updated": mut_result.rowcount if mut_result else 0,
        "history_updated": hist_result.rowcount if hist_result else 0,
        "bets_regenerated": regenerated,
        "regen_skipped_reason": None if race_in_future else "race already jumped",
    }


@app.post("/api/admin/bets/generate-all")
async def admin_generate_all(x_cron_secret: Optional[str] = Header(None)):
    """Fire the hourly bet-generation sweep immediately. Additive — only
    inserts strategy_labels that don't already exist per race, so it's
    safe to call repeatedly and lets newly-added strategies backfill
    onto already-generated races without waiting for the next cron tick."""
    _check_admin(x_cron_secret)
    asyncio.create_task(_scheduled_generate_bets())
    return {"ok": True, "queued": True}


@app.get("/api/admin/bets/portfolio-sim")
async def portfolio_simulation(
    days: int = 270,
    avg_dividend: float = 100.0,
    x_cron_secret: Optional[str] = Header(None),
):
    """Simulate fixed portfolios (mixes of box bets per race) across the
    backtest window. For each race, build each portfolio's boxes,
    check against actual top-3, compute payout at a constant assumed
    dividend, and report hit rate + ROI per portfolio.
    """
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    # Portfolios are defined by RANK (1=model's top horse). Translated to
    # tab numbers per race using the prediction history.
    portfolios = {
        "trio_only ($2/race)": [
            [1,2,3],
        ],
        "trio_x3 ($6/race)": [
            [1,2,3], [1,2,4], [1,2,5],
        ],
        "trio_x5 ($10/race)": [
            [1,2,3], [1,2,4], [1,2,5], [1,3,4], [2,3,4],
        ],
        "trio_x5_value ($10/race)": [
            # Includes value runner 5 for outsider scenarios
            [1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5],
        ],
        "trio_x10 ($20/race)": [
            [1,2,3], [1,2,4], [1,2,5], [1,2,6],
            [1,3,4], [1,3,5], [2,3,4], [2,3,5],
            [1,4,5], [3,4,5],
        ],
        "spread_5box ($10/race)": [
            [1,2,3], [1,2,3,4], [1,2,5], [1,2,6], [2,3,4,5],
        ],
        "trio_plus_value ($6/race)": [
            [1,2,3], [1,2,5], [1,2,6],
        ],
        "trio_plus_quad ($4/race)": [
            [1,2,3], [1,2,3,4],
        ],
        "ten_box_mega ($20/race)": [
            [1,2,3], [1,2,5], [1,2,6], [1,2,4], [2,3,4],
            [1,2,3,4], [1,2,3,5], [2,3,4,5],
            [1,2,3,4,5], [1,2,3,4,5,6],
        ],
        "net_only ($2/race)": [
            [1,2,3,4,5,6],
        ],
    }
    stake_per_box = 2.0

    # Pull all races with top-3 + prediction history (same as shootout)
    async with get_session() as session:
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.tab_number, HistoricalResultRow.horse_name)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2, 3]))
        )).fetchall()
        results_by_race: dict[str, list] = {}
        for rid, pos, tab, name in result_rows:
            results_by_race.setdefault(rid, []).append((pos, tab, name))
        complete = [rid for rid, items in results_by_race.items() if len(items) >= 3]
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(complete))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
        )).scalars().all() if complete else []

    preds_by_race: dict[str, list] = {}
    for p in pred_rows:
        preds_by_race.setdefault(p.race_id, []).append(p)

    # Resolve top-3 tabs
    tab_per_race: dict[str, dict[str, int]] = {}
    for rid, pruns in preds_by_race.items():
        tab_per_race[rid] = {_normalize_horse(p.horse_name): p.tab_number
                             for p in pruns if p.tab_number is not None}
    resolved_top3: dict[str, list[int]] = {}
    for rid in complete:
        items = sorted(results_by_race[rid], key=lambda x: x[0])[:3]
        tabs = []
        for pos, tab, name in items:
            t = tab if tab is not None else tab_per_race.get(rid, {}).get(_normalize_horse(name))
            tabs.append(t)
        if all(t is not None for t in tabs):
            resolved_top3[rid] = tabs

    # Run each portfolio across the universe
    results: list[dict] = []
    for portfolio_name, rank_boxes in portfolios.items():
        races_evaluated = 0
        races_hit = 0
        boxes_evaluated = 0
        boxes_hit = 0
        total_stake = 0.0
        total_payout = 0.0
        for rid, actual_top3 in resolved_top3.items():
            pruns = preds_by_race.get(rid, [])
            if len(pruns) < 7:
                continue
            # Sort by model_rank to get tabs by rank
            sorted_p = sorted(
                [p for p in pruns if p.model_rank and p.tab_number],
                key=lambda p: p.model_rank,
            )
            if len(sorted_p) < 6:
                continue  # portfolio includes ranks up to 6
            rank_to_tab = {i + 1: sorted_p[i].tab_number for i in range(len(sorted_p))}
            races_evaluated += 1
            race_hit_any = False
            for rank_box in rank_boxes:
                tabs = [rank_to_tab.get(r) for r in rank_box]
                if any(t is None for t in tabs):
                    continue
                boxes_evaluated += 1
                total_stake += stake_per_box
                if _bet_is_hit(tabs, actual_top3):
                    boxes_hit += 1
                    race_hit_any = True
                    n = len(rank_box)
                    perms = n * (n - 1) * (n - 2)
                    payout = (stake_per_box / perms) * avg_dividend
                    total_payout += payout
            if race_hit_any:
                races_hit += 1
        pnl = round(total_payout - total_stake, 2)
        roi = round(pnl / total_stake * 100, 1) if total_stake else 0
        results.append({
            "portfolio": portfolio_name,
            "boxes_per_race": len(rank_boxes),
            "races_evaluated": races_evaluated,
            "races_hit": races_hit,
            "race_hit_rate_pct": round(races_hit / races_evaluated * 100, 1) if races_evaluated else 0,
            "total_stake": round(total_stake, 2),
            "total_payout": round(total_payout, 2),
            "pnl": pnl,
            "roi_pct": roi,
            "boxes_hit_rate_pct": round(boxes_hit / boxes_evaluated * 100, 1) if boxes_evaluated else 0,
        })
    results.sort(key=lambda r: -r["roi_pct"])
    return {
        "window_days": days,
        "avg_dividend_assumed": avg_dividend,
        "portfolios": results,
    }


@app.get("/api/admin/predictions/funkmeup-rethink")
async def funkmeup_rethink_analysis(
    days: int = 60,
    x_cron_secret: Optional[str] = Header(None),
):
    """Answers two questions for the Funk Me Up redesign:
    1. Does the win rate on rank-1 drop when rank-1 and rank-2 are close
       in model probability (model is 'indecisive')?
    2. What's the realistic ROI on a 3-4 leg PLACE multi using the model's
       top picks at typical TAB place odds?
    """
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        # All rank 1-3 history rows for races in window
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.model_rank,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.place_probability,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank.in_([1, 2, 3]))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        # Result rows — winner + placings 1-3
        result_rows = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.position,
                   HistoricalResultRow.horse_name, HistoricalResultRow.starting_price)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position.in_([1, 2, 3]))
        )).fetchall()

    # Build per-race {rank: (name, win_p, place_p)} keeping latest per horse
    per_race: dict[str, dict] = {}
    seen: set = set()
    for rid, name, rank, wp, pp, enr in pred_rows:
        key = (rid, _normalize_horse(name))
        if key in seen:
            continue
        seen.add(key)
        per_race.setdefault(rid, {})[int(rank)] = {
            "horse_name": name,
            "win_prob": float(wp or 0),
            "place_prob": float(pp or 0),
        }
    # Results lookup
    results: dict[str, dict] = {}
    for rid, pos, name, sp in result_rows:
        if pos is None:
            continue
        results.setdefault(rid, {})[int(pos)] = {
            "horse_name": name, "sp": sp,
        }

    # Q1: rank-1 win rate by gap-to-rank-2
    gap_buckets = [
        ("≤2pt gap (toss-up)", 0, 2),
        ("2-5pt gap (lean)", 2, 5),
        ("5-10pt gap (clear)", 5, 10),
        ("10pt+ gap (dominant)", 10, 999),
    ]
    by_gap = {b[0]: {"races": 0, "wins": 0, "places": 0} for b in gap_buckets}
    overall_n = 0
    overall_wins = 0
    overall_places = 0
    for rid, ranks in per_race.items():
        if 1 not in ranks or 2 not in ranks:
            continue
        if rid not in results:
            continue
        winner_norm = _normalize_horse((results[rid].get(1) or {}).get("horse_name") or "")
        placings_norm = {
            _normalize_horse((results[rid].get(p) or {}).get("horse_name") or "")
            for p in (1, 2, 3)
        }
        if not winner_norm:
            continue
        rank1 = ranks[1]
        rank2 = ranks[2]
        gap_pts = (rank1["win_prob"] - rank2["win_prob"]) * 100
        won = _normalize_horse(rank1["horse_name"]) == winner_norm
        placed = _normalize_horse(rank1["horse_name"]) in placings_norm
        overall_n += 1
        if won: overall_wins += 1
        if placed: overall_places += 1
        for label, lo, hi in gap_buckets:
            if lo <= gap_pts < hi:
                by_gap[label]["races"] += 1
                if won: by_gap[label]["wins"] += 1
                if placed: by_gap[label]["places"] += 1
                break

    q1_out = []
    for label, lo, hi in gap_buckets:
        d = by_gap[label]
        n = d["races"]
        q1_out.append({
            "bucket": label,
            "n": n,
            "rank1_win_pct": round(d["wins"]/n*100, 1) if n else 0,
            "rank1_place_pct": round(d["places"]/n*100, 1) if n else 0,
        })

    # Q2: 3 / 4-leg PLACE multi using top-1 picks per race (favourite-style)
    # Approximate TAB place odds as 1 / (place_prob * (1 - book_margin))
    # where book_margin ~ 0.12 (Place markets carry ~12-15% over-round).
    # Use the model's place_probability as proxy for actual place odds since
    # we don't store TAB place prices historically.
    BOOK_MARGIN = 0.12

    def synth_place_odds(place_prob: float) -> float:
        if place_prob <= 0:
            return 0
        # Implied odds assuming the market over-round is shared evenly
        implied = max(place_prob * (1 - BOOK_MARGIN), 0.05)
        return round(1.0 / implied, 2)

    # Build per-race rank-1 pick's place_prob + did it place
    race_picks = []
    for rid, ranks in per_race.items():
        if 1 not in ranks or rid not in results:
            continue
        rank1 = ranks[1]
        if rank1["place_prob"] <= 0:
            continue
        winner_norm = _normalize_horse((results[rid].get(1) or {}).get("horse_name") or "")
        placings_norm = {
            _normalize_horse((results[rid].get(p) or {}).get("horse_name") or "")
            for p in (1, 2, 3)
        }
        placed = _normalize_horse(rank1["horse_name"]) in placings_norm
        date_str = rid.split("_", 1)[0]
        race_picks.append({
            "race_id": rid,
            "date": date_str,
            "place_prob": rank1["place_prob"],
            "place_odds_est": synth_place_odds(rank1["place_prob"]),
            "placed": placed,
        })

    # Group by date, simulate N-leg place multis using the day's strongest
    # place picks (sorted by place_prob desc).
    def simulate_place_multis(legs: int):
        from collections import defaultdict
        by_date: dict[str, list] = defaultdict(list)
        for rp in race_picks:
            by_date[rp["date"]].append(rp)
        days_total = 0
        multis_hit = 0
        total_stake = 0.0
        total_return = 0.0
        for d, picks in by_date.items():
            picks_sorted = sorted(picks, key=lambda x: -x["place_prob"])
            if len(picks_sorted) < legs:
                continue
            chosen = picks_sorted[:legs]
            multi_odds = 1.0
            for c in chosen:
                multi_odds *= c["place_odds_est"]
            stake = 10.0
            days_total += 1
            total_stake += stake
            if all(c["placed"] for c in chosen):
                multis_hit += 1
                total_return += stake * multi_odds
        pnl = total_return - total_stake
        return {
            "legs": legs,
            "days_evaluated": days_total,
            "multis_hit": multis_hit,
            "hit_rate_pct": round(multis_hit/days_total*100, 1) if days_total else 0,
            "total_stake": round(total_stake, 2),
            "total_return": round(total_return, 2),
            "pnl": round(pnl, 2),
            "roi_pct": round(pnl/total_stake*100, 1) if total_stake else 0,
        }

    q2_out = [simulate_place_multis(n) for n in (2, 3, 4, 5)]

    return {
        "days": days,
        "overall_rank1_races": overall_n,
        "overall_rank1_win_pct": round(overall_wins/overall_n*100, 1) if overall_n else 0,
        "overall_rank1_place_pct": round(overall_places/overall_n*100, 1) if overall_n else 0,
        "q1_decisiveness": q1_out,
        "q2_place_multi": q2_out,
        "q2_caveats": [
            "Place odds estimated from model place_probability with 12% book over-round.",
            "Actual TAB place odds may differ by ±10%.",
            "Assumes favourite is bet at each leg (max model place_prob per race).",
            "All metro + country races included.",
        ],
    }


@app.get("/api/admin/predictions/niche-analysis")
async def predictions_niche_analysis(
    days: int = 60,
    min_sample: int = 30,
    x_cron_secret: Optional[str] = Header(None),
):
    """Where is the model actually strong? Buckets the rank-1 win rate
    over the last N days by metro/country, day-of-week, field size,
    distance band, track condition, prize money, model confidence band,
    state, and combinations of these. Returns each band with sample
    count and win rate, plus the overall baseline so the user can see
    the gap above the average."""
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff_date = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        # Rank-1 predictions only — that's the model's actual call.
        pred_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
                RunnerPredictionHistoryRow.distance,
                RunnerPredictionHistoryRow.track_condition,
                RunnerPredictionHistoryRow.prize_money,
                RunnerPredictionHistoryRow.state,
                RunnerPredictionHistoryRow.field_size,
                RunnerPredictionHistoryRow.enriched_at,
            )
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()
        # Top-3 sum: pull rank 1-3 win_prob per race for the concentration feature.
        top3_rows = (await session.execute(
            select(RunnerPredictionHistoryRow.race_id,
                   RunnerPredictionHistoryRow.model_rank,
                   RunnerPredictionHistoryRow.win_probability)
            .where(RunnerPredictionHistoryRow.race_id >= f"{cutoff_date}_")
            .where(RunnerPredictionHistoryRow.model_rank.in_([1, 2, 3]))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )).fetchall()
        # Winners
        winners = (await session.execute(
            select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name)
            .where(HistoricalResultRow.race_id >= f"{cutoff_date}_")
            .where(HistoricalResultRow.position == 1)
        )).fetchall()

    # Dedup predictions to one row per race (latest enriched first thanks to ORDER BY)
    rank1_by_race: dict[str, dict] = {}
    for rid, name, p, dist, tc, pm, state, fs, enr in pred_rows:
        if rid in rank1_by_race:
            continue
        rank1_by_race[rid] = {
            "race_id": rid,
            "horse_name": name,
            "model_pct": (p or 0) * 100,
            "distance": dist,
            "track_condition": tc,
            "prize_money": pm,
            "state": state,
            "field_size": fs,
        }
    # Top-3 sum per race
    top3_by_race: dict[str, dict[int, float]] = {}
    for rid, rank, p in top3_rows:
        if p is None or rank is None:
            continue
        top3_by_race.setdefault(rid, {})[int(rank)] = float(p) * 100
    # Winners lookup
    winners_by_race: dict[str, str] = {}
    for rid, name in winners:
        winners_by_race[rid] = _normalize_horse(name or "")

    # Build evaluation set
    evals: list[dict] = []
    for rid, info in rank1_by_race.items():
        if rid not in winners_by_race:
            continue  # no result yet
        won = _normalize_horse(info["horse_name"]) == winners_by_race[rid]
        date_str, venue, _ = _parse_race_id(rid)
        dow = "?"
        if date_str:
            try:
                dow = datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")
            except Exception:
                pass
        top3 = top3_by_race.get(rid, {})
        top3_sum = sum(top3.get(r, 0) for r in (1, 2, 3))
        info_full = dict(info)
        info_full.update({
            "won": won,
            "dow": dow,
            "venue": venue,
            "metro": (venue or "").lower() in _FUNK_METRO_VENUES,
            "top3_sum_pct": top3_sum,
        })
        evals.append(info_full)

    total_n = len(evals)
    total_wins = sum(1 for e in evals if e["won"])
    overall_rate = round(total_wins / total_n * 100, 1) if total_n else 0

    def _bucket(name: str, key_fn, bands=None):
        """Return list of {bucket, n, wins, win_rate_pct, lift_pct}.
        bands lets you collapse continuous values into named ranges."""
        groups: dict[str, list] = {}
        for e in evals:
            k = key_fn(e)
            if k is None:
                continue
            groups.setdefault(k, []).append(e)
        out = []
        for k, items in groups.items():
            n = len(items)
            wins = sum(1 for x in items if x["won"])
            rate = round(wins / n * 100, 1) if n else 0
            out.append({
                "bucket": k,
                "n": n,
                "wins": wins,
                "win_rate_pct": rate,
                "lift_pct": round(rate - overall_rate, 1),
            })
        out.sort(key=lambda r: -r["win_rate_pct"])
        return out

    def _band_dist(d):
        if not d: return None
        if d <= 1200: return "sprint (≤1200m)"
        if d <= 1600: return "mile (1300-1600m)"
        if d <= 2000: return "middle (1700-2000m)"
        return "staying (2100m+)"

    def _band_field(f):
        if not f: return None
        if f <= 8: return "small (≤8)"
        if f <= 11: return "medium (9-11)"
        if f <= 14: return "large (12-14)"
        return "huge (15+)"

    def _band_model(p):
        if p is None: return None
        if p < 25: return "weak rank-1 (<25%)"
        if p < 30: return "mod rank-1 (25-29%)"
        if p < 35: return "strong rank-1 (30-34%)"
        if p < 40: return "very strong (35-39%)"
        return "dominant (≥40%)"

    def _band_top3(p):
        if not p: return None
        if p < 45: return "open (<45%)"
        if p < 55: return "moderate (45-54%)"
        if p < 65: return "concentrated (55-64%)"
        return "heavy fav (≥65%)"

    def _band_prize(p):
        if not p: return "unknown"
        if p >= 80000: return "metro ($80k+)"
        if p >= 30000: return "feature ($30-79k)"
        if p >= 15000: return "country ($15-29k)"
        return "picnic (<$15k)"

    def _band_tc(t):
        if not t: return None
        t = str(t).lower()
        if "good" in t: return "Good"
        if "soft" in t: return "Soft"
        if "heavy" in t: return "Heavy"
        if "synth" in t or "tape" in t or "poly" in t: return "Synthetic"
        return "Other"

    result = {
        "days": days,
        "overall_races": total_n,
        "overall_wins": total_wins,
        "overall_win_rate_pct": overall_rate,
        "min_sample": min_sample,
        "buckets": {
            "metro_vs_country": _bucket("metro", lambda e: "Metro" if e["metro"] else "Country/Provincial"),
            "day_of_week": _bucket("dow", lambda e: e["dow"] if e["dow"] != "?" else None),
            "field_size": _bucket("field", lambda e: _band_field(e["field_size"])),
            "distance": _bucket("dist", lambda e: _band_dist(e["distance"])),
            "track_condition": _bucket("tc", lambda e: _band_tc(e["track_condition"])),
            "prize_money": _bucket("prize", lambda e: _band_prize(e["prize_money"])),
            "state": _bucket("state", lambda e: e.get("state") if e.get("state") else None),
            "model_confidence": _bucket("mod", lambda e: _band_model(e["model_pct"])),
            "top3_sum": _bucket("t3", lambda e: _band_top3(e["top3_sum_pct"])),
        },
    }
    # Filter each bucket to only bands with sample ≥ min_sample for the
    # "honest" view (small samples are too noisy to act on).
    result["honest_buckets"] = {
        name: [b for b in band if b["n"] >= min_sample]
        for name, band in result["buckets"].items()
    }
    return result


@app.get("/api/admin/bets/feature-analysis")
async def admin_feature_analysis(
    days: int = 30,
    x_cron_secret: Optional[str] = Header(None),
):
    """Per-race feature → trifecta hit-rate breakdown over the last N days.
    Helps answer: which races are the box bets most likely to hit?

    For every settled race we compute:
      - rank1_win_pct  : favourite's model probability
      - top3_sum_pct   : combined prob of model's top-3 horses
      - field_size     : number of active runners
      - metro_country  : metro vs country meeting
      - day_of_week    : Sat/Sun vs weekday

    For each feature we bucket the races and report hit rate (% of races
    where any bet box caught the trifecta) and average estimated ROI.
    Designed to surface which signal actually predicts box hits.
    """
    _check_admin(x_cron_secret)
    days = max(7, min(int(days), 365))
    cutoff = datetime.utcnow() - timedelta(days=days)
    from collections import defaultdict

    async with get_session() as session:
        bet_race_ids = list({
            rid for (rid,) in (await session.execute(
                select(BetRecommendationRow.race_id)
                .where(BetRecommendationRow.recommended_at >= cutoff)
                .where(BetRecommendationRow.settled.is_(True))
            )).fetchall()
        })
        if not bet_race_ids:
            return {"races_examined": 0, "buckets": {}}

        # Bets per race for hit + P&L aggregation
        bet_rows = (await session.execute(
            select(BetRecommendationRow.race_id,
                   BetRecommendationRow.is_hit,
                   BetRecommendationRow.stake_dollars,
                   BetRecommendationRow.payout_dollars,
                   BetRecommendationRow.voided)
            .where(BetRecommendationRow.race_id.in_(bet_race_ids))
            .where(BetRecommendationRow.settled.is_(True))
        )).fetchall()

        # Per-race aggregations
        race_hits: dict[str, int] = defaultdict(int)
        race_stake: dict[str, float] = defaultdict(float)
        race_payout: dict[str, float] = defaultdict(float)
        for rid, hit, stake, payout, voided in bet_rows:
            if voided:
                continue
            if hit:
                race_hits[rid] += 1
            race_stake[rid] += stake or 0
            race_payout[rid] += payout or 0

        # Per-race features from history (latest snapshot wins)
        feat_rows = (await session.execute(
            select(RunnerPredictionHistoryRow.race_id,
                   RunnerPredictionHistoryRow.model_rank,
                   RunnerPredictionHistoryRow.win_probability)
            .where(RunnerPredictionHistoryRow.race_id.in_(bet_race_ids))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False)
                   | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).fetchall()

    race_probs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for rid, rank, p in feat_rows:
        if rank is None or p is None:
            continue
        race_probs[rid].append((int(rank), float(p)))
    for rid in race_probs:
        race_probs[rid].sort()

    races: list[dict] = []
    METRO = {"randwick", "royal-randwick", "rosehill", "rosehill-gardens",
             "canterbury-park", "warwick-farm", "caulfield", "caulfield-heath",
             "flemington", "moonee-valley", "sandown", "sandown-lakeside",
             "sandown-hillside", "the-valley", "doomben", "eagle-farm",
             "morphettville", "morphettville-parks", "ascot", "belmont"}
    for rid in bet_race_ids:
        items = race_probs.get(rid) or []
        if not items:
            continue
        rank1 = items[0][1] * 100
        top3 = sum(p for _, p in items[:3]) * 100
        date_str, venue, _ = _parse_race_id(rid)
        dow = "?"
        if date_str:
            try:
                dow = datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")
            except Exception:
                pass
        races.append({
            "race_id": rid,
            "rank1": rank1,
            "top3_sum": top3,
            "field": len(items),
            "metro": (venue or "").lower() in METRO,
            "dow": dow,
            "hit": race_hits[rid] > 0,
            "stake": race_stake[rid],
            "payout": race_payout[rid],
        })

    def _bucketize(name: str, buckets: list[tuple]):
        out = []
        for lo, hi, label in buckets:
            sub = [r for r in races
                   if r[name] is not None and lo <= r[name] < hi]
            if not sub:
                continue
            hit_pct = round(sum(1 for r in sub if r["hit"]) / len(sub) * 100, 1)
            stake = sum(r["stake"] for r in sub)
            payout = sum(r["payout"] for r in sub)
            roi = round((payout - stake) / stake * 100, 1) if stake else 0
            out.append({"bucket": label, "n": len(sub),
                        "hit_pct": hit_pct, "roi_pct": roi,
                        "pnl": round(payout - stake, 2)})
        return out

    rank1_buckets = _bucketize("rank1", [(0, 20, "<20"), (20, 25, "20-25"),
        (25, 30, "25-30"), (30, 35, "30-35"), (35, 40, "35-40"), (40, 100, "40+")])
    top3_buckets = _bucketize("top3_sum", [(0, 40, "<40"), (40, 50, "40-50"),
        (50, 60, "50-60"), (60, 70, "60-70"), (70, 80, "70-80"), (80, 100, "80+")])
    field_buckets = _bucketize("field", [(0, 8, "<8"), (8, 10, "8-9"),
        (10, 12, "10-11"), (12, 14, "12-13"), (14, 99, "14+")])
    dow_buckets = []
    for dow in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        sub = [r for r in races if r["dow"] == dow]
        if not sub:
            continue
        hit_pct = round(sum(1 for r in sub if r["hit"]) / len(sub) * 100, 1)
        stake = sum(r["stake"] for r in sub)
        payout = sum(r["payout"] for r in sub)
        roi = round((payout - stake) / stake * 100, 1) if stake else 0
        dow_buckets.append({"bucket": dow, "n": len(sub),
            "hit_pct": hit_pct, "roi_pct": roi, "pnl": round(payout - stake, 2)})
    metro_buckets = []
    for is_metro, label in [(True, "metro"), (False, "country")]:
        sub = [r for r in races if r["metro"] == is_metro]
        if not sub:
            continue
        hit_pct = round(sum(1 for r in sub if r["hit"]) / len(sub) * 100, 1)
        stake = sum(r["stake"] for r in sub)
        payout = sum(r["payout"] for r in sub)
        roi = round((payout - stake) / stake * 100, 1) if stake else 0
        metro_buckets.append({"bucket": label, "n": len(sub),
            "hit_pct": hit_pct, "roi_pct": roi, "pnl": round(payout - stake, 2)})

    return {
        "races_examined": len(races),
        "days": days,
        "rank1_win_pct": rank1_buckets,
        "top3_sum_pct": top3_buckets,
        "field_size": field_buckets,
        "day_of_week": dow_buckets,
        "metro_country": metro_buckets,
    }


@app.post("/api/admin/bets/resettle-all")
async def admin_resettle_all(days: int = 7, x_cron_secret: Optional[str] = Header(None)):
    """Force-resettle every bet row in the window — clears the settled
    flag then runs settlement again. Use to retroactively apply the
    dedup-actual-top3 fix to rows already on the wrong side of it."""
    _check_admin(x_cron_secret)
    days = max(1, min(int(days), 30))
    cutoff = datetime.utcnow() - timedelta(days=days)
    async with get_session() as session:
        from sqlalchemy import update as _sa_update
        race_ids = (await session.execute(
            select(BetRecommendationRow.race_id).distinct()
            .where(BetRecommendationRow.recommended_at >= cutoff)
        )).scalars().all()
        # Clear settled flag on all rows in the window.
        await session.execute(
            _sa_update(BetRecommendationRow)
            .where(BetRecommendationRow.recommended_at >= cutoff)
            .values(settled=False)
        )
        await session.commit()
    settled = 0
    for rid in race_ids:
        try:
            settled += await _settle_bets_for_race(rid)
        except Exception as e:
            log.debug("[bets] resettle failed for %s: %s", rid, e)
    return {"ok": True, "races_examined": len(race_ids), "rows_resettled": settled}


@app.post("/api/admin/bets/settle-only")
async def admin_settle_only(x_cron_secret: Optional[str] = Header(None)):
    """Settle bets using whatever results are already in the DB — no
    re-seeding. Fast (~1s), useful when results are confirmed populated
    and you just need the bet rows flipped to 'settled'."""
    _check_admin(x_cron_secret)
    async with get_session() as session:
        race_ids = (await session.execute(
            select(BetRecommendationRow.race_id).distinct()
            .where(BetRecommendationRow.settled.is_(False))
        )).scalars().all()
    total = 0
    for rid in race_ids:
        try:
            total += await _settle_bets_for_race(rid)
        except Exception as e:
            log.debug("[bets] settle-only failed for %s: %s", rid, e)
    return {"ok": True, "settled": total, "races_examined": len(race_ids)}


@app.get("/api/admin/bets/debug-dividend/{race_id}")
async def admin_debug_dividend(race_id: str, x_cron_secret: Optional[str] = Header(None)):
    """Inspect TAB's raw payload + the extracted trifecta dividend for a race.
    Use to verify the dividend-extraction logic against real TAB data."""
    _check_admin(x_cron_secret)
    raw = await _fetch_race_raw_from_tab(race_id)
    if raw is None:
        return {"race_id": race_id, "fetched": False, "trifecta": None}
    extracted = _extract_trifecta_from_tab_response(raw)
    return {
        "race_id": race_id,
        "fetched": True,
        "trifecta": extracted,
        "raw_keys": list(raw.keys())[:30] if isinstance(raw, dict) else None,
        "raw_sample": {k: (str(v)[:200] if not isinstance(v, (dict, list)) else
                            (v[:3] if isinstance(v, list) else dict(list(v.items())[:5])))
                       for k, v in (raw.items() if isinstance(raw, dict) else [])}
    }


_track_record_cache: tuple[datetime, dict] | None = None
_TRACK_RECORD_TTL = 600  # 10 min — tier rates barely move between settlements

@app.get("/api/track-record")
async def get_track_record():
    """Public endpoint — tier win rates from the unified all-time backtest +
    live dataset (the same source /api/admin/backtest/analysis uses).

    Previously this was a 30-day live-only window which produced tiny per-tier
    samples (e.g. 6 picks in the Hot tier, 1 win, 17% — pure noise). Switched
    to all-time on 2026-06-13. Now scales with stored history (3,000+ picks
    in the unified set) so the tier numbers are statistically meaningful."""
    global _track_record_cache
    if _track_record_cache is not None:
        ts, body = _track_record_cache
        if (datetime.utcnow() - ts).total_seconds() < _TRACK_RECORD_TTL:
            return body
    async with get_session() as session:
        # Retroactive backtest rows (built via the offline backtest pipeline)
        bt_rows = (await session.execute(
            select(BacktestResultRow).where(BacktestResultRow.source == "backtest")
        )).scalars().all()

        # Live: history rank-1 joined with historical_results — dedup on
        # latest enriched_at per race.
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

    # Build unified row list (same shape as /api/admin/backtest/analysis)
    unified = []
    for r in bt_rows:
        if r.win_probability is not None:
            unified.append({"win_prob": r.win_probability, "winner": bool(r.winner)})
    for r in live_rows:
        hr = hr_map.get((r.race_id, r.horse_name))
        if hr and r.win_probability is not None:
            unified.append({"win_prob": r.win_probability, "winner": hr.position == 1})

    tiers = [
        {"badge": "hot",      "min": 0.45, "max": 1.01, "conf_min": 45, "conf_max": None},
        {"badge": "high",     "min": 0.35, "max": 0.45, "conf_min": 35, "conf_max": 45},
        {"badge": "standard", "min": 0.30, "max": 0.35, "conf_min": 30, "conf_max": 35},
    ]
    output = []
    for tier in tiers:
        picks = [r for r in unified if tier["min"] <= r["win_prob"] < tier["max"]]
        wins  = sum(1 for r in picks if r["winner"])
        win_pct = round(wins / len(picks) * 100) if picks else 0
        output.append({
            "badge":    tier["badge"],
            "win_pct":  win_pct,
            "races":    len(picks),
            "conf_min": tier["conf_min"],
            "conf_max": tier["conf_max"],
        })
    body = {
        "tiers": output,
        "total_races": len(unified),
        "generated_at": datetime.utcnow().isoformat(),
    }
    _track_record_cache = (datetime.utcnow(), body)
    return body


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
    # Persistent DB cache — survives Railway redeploys. Read here BEFORE
    # hitting RA so a freshly-booted container serves date pills in <500ms
    # instead of the full 10-15s RA cold-fetch.
    try:
        async with get_session() as session:
            row = (await session.execute(
                select(ResponseCacheRow)
                .where(ResponseCacheRow.cache_key == f"meetings:{race_date}")
            )).scalar_one_or_none()
        if row:
            # 6h staleness budget — older than that we re-fetch from RA to
            # pick up new meetings / scratchings published since the cache
            # was written.
            age = (datetime.utcnow() - row.updated_at).total_seconds()
            if age < 21600:
                body = json.loads(row.payload_json)
                _list_meetings_cache[race_date] = (datetime.utcnow(), body)
                return body
    except Exception as e:
        log.debug("[list_meetings] DB cache read skipped: %s", e)
    client = get_tab_client()
    # PAST DATES: skip RA entirely. Every meeting that ran that day is
    # already in the DB (via RacePredictionRow / RunnerPredictionRow /
    # HistoricalResultRow). Hitting RA for past dates was contributing to
    # the ~30s past-day page-load — same class of bug as the meeting-
    # detail + live-odds fixes shipped earlier today.
    is_past_date = race_date < _today_aest().isoformat()
    if is_past_date:
        meetings = []
    elif _ra_breaker_open(client):
        log.info("[list_meetings] RA breaker open — using DB-only path for %s", race_date)
        meetings = []
    else:
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

    # Classify each meeting by tier — max prize_money across the day's
    # races. >= $80k = metro; <$30k = country; in between = provincial.
    # Falls back to METRO_VENUES hardcoded set when prize_money missing.
    if items:
        prize_prefix = f"{race_date}_"
        async with get_session() as session:
            prize_rows = (await session.execute(
                select(RunnerPredictionRow.race_id, func.max(RunnerPredictionRow.prize_money))
                .where(RunnerPredictionRow.race_id.like(f"{prize_prefix}%"))
                .group_by(RunnerPredictionRow.race_id)
            )).fetchall()
        max_prize_by_venue: dict[str, int] = {}
        for rid, prize in prize_rows:
            if not prize:
                continue
            _, vc, _ = _parse_race_id(rid)
            if vc:
                max_prize_by_venue[vc] = max(max_prize_by_venue.get(vc, 0), int(prize))
        for it in items:
            vc = it.get("venue_code") or ""
            mx = max_prize_by_venue.get(vc, 0)
            if mx >= 80000:
                it["tier"] = "metro"
            elif mx >= 30000:
                it["tier"] = "provincial"
            elif mx > 0:
                it["tier"] = "country"
            else:
                # No prize data — fall back to the hardcoded metro set.
                from horse_engine.bets import is_metro_venue
                it["tier"] = "metro" if is_metro_venue(vc) else "country"
            it["max_prize_money"] = mx or None

    result = {"date": race_date, "meetings": items}
    _list_meetings_cache[race_date] = (datetime.utcnow(), result)
    # Persist to Postgres so the next container redeploy hydrates this
    # date immediately on startup. Without this, every date in the
    # main-page strip is cold after a deploy → 10-15s per click.
    try:
        async with get_session() as session:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            cache_key = f"meetings:{race_date}"
            stmt = pg_insert(ResponseCacheRow).values(
                cache_key=cache_key,
                payload_json=json.dumps(result),
                cache_version=1,
                updated_at=datetime.utcnow(),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["cache_key"],
                set_=dict(payload_json=stmt.excluded.payload_json,
                          cache_version=stmt.excluded.cache_version,
                          updated_at=stmt.excluded.updated_at),
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as e:
        log.debug("[list_meetings] DB cache write skipped: %s", e)
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
    # Adds: live status for open/closed races + any races not yet enriched.
    # For PAST dates every race is already settled, race times + names are
    # already in the DB, and RA has nothing new to add — skip the 25s TAB
    # call entirely. The Lounge fires this endpoint per meeting on every
    # date click, so past-date pages were paying 25s × N meetings of pure
    # latency for zero information gain.
    ra_times: dict[str, str] = {}  # race_id → startTime, for DB back-fill
    is_past_date = race_date < _today_aest().isoformat()
    if not is_past_date:
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
        # Rank-2 win prob — used to compute the decisiveness indicator
        # (>5pt gap = model has a clear favourite, not a toss-up).
        rank2_win_probs: dict[str, Optional[float]] = {race_id: None for race_id in race_ids}

        if completed_ids:
            hist_tp_result = await session.execute(
                select(RunnerPredictionHistoryRow)
                .where(RunnerPredictionHistoryRow.race_id.in_(completed_ids))
                .where(RunnerPredictionHistoryRow.model_rank.in_([1, 2]))
                .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
                .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
            )
            seen_rank: dict[tuple[str, int], bool] = {}
            for p in hist_tp_result.scalars().all():
                key = (p.race_id, p.model_rank)
                if key in seen_rank:
                    continue
                seen_rank[key] = True
                if p.model_rank == 1:
                    top_picks[p.race_id] = p.horse_name
                    top_win_probs[p.race_id] = p.win_probability
                    top_place_probs[p.race_id] = p.place_probability
                elif p.model_rank == 2:
                    rank2_win_probs[p.race_id] = p.win_probability

        upcoming_ids = [rid for rid in race_ids if rid not in completed_ids]
        if upcoming_ids:
            tp_result = await session.execute(
                select(RunnerPredictionRow)
                .where(RunnerPredictionRow.race_id.in_(upcoming_ids))
                .where(RunnerPredictionRow.model_rank.in_([1, 2]))
                .where(RunnerPredictionRow.cancelled.is_(False) | RunnerPredictionRow.cancelled.is_(None))
            )
            for p in tp_result.scalars().all():
                if p.model_rank == 1:
                    top_picks[p.race_id] = p.horse_name
                    top_win_probs[p.race_id] = p.win_probability
                    top_place_probs[p.race_id] = p.place_probability
                elif p.model_rank == 2:
                    rank2_win_probs[p.race_id] = p.win_probability

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
            "rank2_win_probability": rank2_win_probs.get(rid),
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


# Per-race response cache for live-odds. Every race-detail page reload
# previously triggered an RA call — and the frontend polls every ~15s
# during race viewing. With a 30s cache window, repeated viewers and
# polls share the same call. Pre-race: odds shift on a timescale of
# minutes, so 30s of staleness is invisible. Settled: data is fixed,
# longer cache would also be fine but 30s keeps the path uniform.
_live_odds_cache: dict[str, tuple[datetime, dict]] = {}
_LIVE_ODDS_TTL = 30

@app.get("/api/races/{race_id}/live-odds")
async def live_odds(race_id: str):
    """
    Re-fetch current tote odds from Racing Australia for a race and compute updated overlays.
    Fast (~1s) — does not regenerate model predictions, just refreshes market data.
    """
    # Response cache — see _live_odds_cache above for rationale.
    cached = _live_odds_cache.get(race_id)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < _LIVE_ODDS_TTL:
        return cached[1]

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
    client = get_tab_client()
    # Skip when RA breaker is open — saves the 5s timeout pain. Page falls
    # back to DB-only positions, which is correct for settled races and
    # acceptable degradation for unsettled.
    # Also skip entirely for settled past races — SP + positions are
    # already in HistoricalResultRow, so the TAB call is pure latency for
    # zero new data. Cuts live-odds endpoint from ~5s to ~50ms for past
    # dates. Only call TAB when the race is still open (no DB results).
    if not db_settled and not _ra_breaker_open(client):
        try:
            slug = _meeting_slug(venue_code, race_date)
            # Timeout 15s → 5s. Was burning 15s/race-page-load when RA
            # degraded; the result bar would either not render or arrive
            # painfully late. 5s is enough headroom for a healthy RA.
            raw_event = await asyncio.wait_for(client.get_race(slug, race_num), timeout=5)
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
    # Use _normalize_horse() for comparisons — matches /api/meetings/.../{venue}'s
    # logic. Raw string equality failed when RA returned 'Autumn Gem' and we
    # stored 'AUTUMN GEM' — result-badge wrongly said the top pick missed
    # while the RAG dot on the race pill correctly said 'model correct'.
    norm_pick = _normalize_horse(top_model_pick) if top_model_pick else None
    model_correct = (
        _normalize_horse(winner_name) == norm_pick
        if winner_name and norm_pick else None
    )
    norm_placed = {_normalize_horse(h) for h in placed_names}
    model_placed = (norm_pick in norm_placed) if (norm_placed and norm_pick) else None

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
            # Normalize before comparing — same fix as model_correct above.
            "is_top_pick": (norm_pick is not None and _normalize_horse(horse) == norm_pick),
        })

    runners_odds.sort(key=lambda x: x["model_win_prob"], reverse=True)

    body = {
        "race_id": race_id,
        "fetched_at": datetime.utcnow().isoformat(),
        "settled": settled,
        "winner": winner_name,
        "model_correct": model_correct,
        "model_placed": model_placed,
        "runners": runners_odds,
    }
    _live_odds_cache[race_id] = (datetime.utcnow(), body)
    return body


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
        output_cal = await _load_output_calibration_curve()
        predictions, _ = await enrich_and_predict_race(
            race, model,
            venue_calibration=venue_cal,
            output_calibration_curve=output_cal,
        )

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
            AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
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
                        # Fallback uses safe pydantic construction (filters out
                        # None values that would crash the strict EnrichedRunner
                        # validation on pre-2026-06-11 enriched_json).
                        fv = fallback_feature_vector(row)
                        if fv is None:
                            continue  # both clean + fallback failed; drop row
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
            AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
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
                    fv = fallback_feature_vector(row)
                    if fv is None:
                        continue
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
    _check_admin(x_cron_secret)
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
    reranked = False
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
    # Re-rank + renormalise so the Lab's rank-1 queries pick up the promoted horse.
    # Always fires (even on a no-op cancel) so idempotent retries can recover from
    # an earlier cancel that landed before the rerank code was deployed.
    try:
        async with get_session() as rsession:
            if await _rerank_race_after_scratch(rsession, race_id):
                await rsession.commit()
                reranked = True
    except Exception as re:
        log.warning("[cancel-runner] %s rerank failed: %s", race_id, re)
    # Clear the per-venue meeting cache so the scratch is immediately visible.
    # The helper also drops the list cache for this date — strictly unnecessary
    # for a single-runner scratch (the venue list rarely changes) but the cost
    # of an extra RA fetch on the next list_meetings call is trivial.
    date_part, venue_part, _ = _parse_race_id(race_id)
    _invalidate_meeting_caches(date_part, venue_part)
    # Bust the edge response cache so /api/edge (and downstream consumers like
    # /api/funk-me-up/today) drop the cancelled horse on the next fetch.
    global _edge_response_cache
    _edge_response_cache = None
    return {
        "updated": result.rowcount,
        "history_updated": hist_result.rowcount,
        "reranked": reranked,
        "race_id": race_id,
        "horse_name": horse_name,
    }


@app.get("/api/admin/debug-odds")
async def debug_odds(venue: str = "", date: str = "", x_cron_secret: Optional[str] = Header(None)):
    """Probe OddsPro + TAB for a venue. Returns raw odds data for diagnosis."""
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

    return result


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
    # Find anchors/tags that look like race containers so we can figure
    # out the new HTML structure. Grep out race headers, tables, and any
    # id/class that mentions "race", "results", "position", "runner".
    import re as _re
    signals = {
        "race_h_tags": _re.findall(r'<h\d[^>]*>[^<]*[Rr]ace[^<]*</h\d>', html)[:8],
        "class_race_*": sorted(set(_re.findall(r'class="[^"]*race[^"]*"', html)))[:15],
        "id_race_*": sorted(set(_re.findall(r'id="[^"]*[Rr]ace[^"]*"', html)))[:15],
        "class_results_*": sorted(set(_re.findall(r'class="[^"]*[Rr]esult[^"]*"', html)))[:15],
        "class_finish_*": sorted(set(_re.findall(r'class="[^"]*[Ff]inish[^"]*"', html)))[:15],
        "table_classes": sorted(set(_re.findall(r'<table[^>]*class="([^"]+)"', html)))[:15],
    }
    # Search for "Race 3" text (Gatton R3) to see how the page marks it
    r3_pos = html.find("Race 3")
    r3_context = html[max(0, r3_pos-100):r3_pos+800] if r3_pos > 0 else ""
    return {
        "url": url,
        "ra_key": ra_key,
        "html_len": html_len,
        "html_snippet": snippet,
        "races_found": races_found,
        "sample": sample,
        "structure_signals": signals,
        "race3_context": r3_context,
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
async def probe_ra_results(
    race_date: str = "",
    venue_filter: str = "",
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Diagnostic: for each venue on race_date, show stored venue/state, keys tried, and
    whether RA returned results. Helps diagnose why seed-ra-results gets 0.
    `venue_filter` is a case-insensitive substring filter to keep probing fast (the
    full sweep does 7×venues RA fetches and times out under busy load).
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
    vf = (venue_filter or "").lower()
    for (venue_name, state), race_ids in venue_state.items():
        if vf and vf not in venue_name.lower():
            continue
        tried = []
        found_key = None
        for prefix in prefixes:
            ra_key = f"{ra_date_str},{state},{prefix}{venue_name}"
            url = f"{_BASE_RA}/FreeFields/Results.aspx?Key={quote(ra_key, safe='')}"
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


@app.get("/api/admin/calibration-filter-trace")
async def calibration_filter_trace(
    holdout_days: int = Query(14, ge=7, le=30),
    window_days: int = Query(270, ge=30, le=365),
    x_cron_secret: Optional[str] = Header(None),
):
    """Walk the EXACT win-calibration filter pipeline, counting how many rows
    survive at each step. Lets us see whether rows are being dropped at
    pred-lookup, time-guard, or feature-vector construction.
    """
    _check_admin(x_cron_secret)
    from horse_engine.prediction.clean_features import (
        AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
    )
    today = date.today()
    train_cutoff = (today - timedelta(days=window_days)).isoformat()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        all_hr = (await session.execute(select(HistoricalResultRow))).scalars().all()
        all_pred = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )).scalars().all()

    pred_by_key = {(p.race_id, p.horse_name): p for p in all_pred}
    index = AggregateIndex(all_hr)

    # Per-step counters
    total = 0
    in_window = 0
    pred_found = 0
    pred_missing = 0
    time_guard_passed = 0
    time_guard_filtered = 0
    recompute_ok = 0
    fallback_used = 0
    fallback_ok = 0
    final_added = 0
    exception_caught = 0

    races_in_window_count = 0
    races_in_window: dict = {}

    # Sample a few miss reasons for diagnostic
    pred_missing_samples = []
    time_guard_samples = []
    exception_samples = []

    for row in all_hr:
        total += 1
        if row.race_id < train_cutoff or row.race_id >= holdout_cutoff:
            continue
        in_window += 1
        pred = pred_by_key.get((row.race_id, row.horse_name))
        if not pred:
            pred_missing += 1
            if len(pred_missing_samples) < 5:
                # Show alternative pred horse_name candidates for this race
                alt = [p.horse_name for p in all_pred if p.race_id == row.race_id][:5]
                pred_missing_samples.append({
                    "race_id": row.race_id,
                    "hr_horse_name": row.horse_name,
                    "pred_candidates_in_race": alt,
                })
            continue
        pred_found += 1
        if pred.enriched_at and pred.scheduled_time:
            try:
                sched = datetime.fromisoformat(pred.scheduled_time.replace("Z", "+00:00")).replace(tzinfo=None)
                if pred.enriched_at > sched:
                    time_guard_filtered += 1
                    if len(time_guard_samples) < 5:
                        time_guard_samples.append({
                            "race_id": row.race_id,
                            "horse_name": row.horse_name,
                            "enriched_at": str(pred.enriched_at),
                            "scheduled_time": str(pred.scheduled_time),
                        })
                    continue
            except (ValueError, AttributeError):
                pass
        time_guard_passed += 1
        try:
            fv = recompute_clean_feature_vector(pred, index)
            if fv is None:
                fallback_used += 1
                fv = fallback_feature_vector(pred)
                if fv is None:
                    continue
                fallback_ok += 1
            else:
                recompute_ok += 1
            final_added += 1
            races_in_window.setdefault(row.race_id, []).append((fv, 1 if row.position == 1 else 0))
        except Exception as e:
            exception_caught += 1
            if len(exception_samples) < 5:
                exception_samples.append({
                    "race_id": row.race_id,
                    "horse_name": row.horse_name,
                    "error": f"{type(e).__name__}: {e}",
                })

    races_with_winner = 0
    valid_race_groups = 0
    for race_id, runners in races_in_window.items():
        if len(runners) >= 2 and sum(l for _, l in runners) == 1:
            valid_race_groups += 1
        if any(l == 1 for _, l in runners):
            races_with_winner += 1

    return {
        "params": {
            "window_days": window_days,
            "holdout_days": holdout_days,
            "train_cutoff": train_cutoff,
            "holdout_cutoff": holdout_cutoff,
            "today": today.isoformat(),
        },
        "data_volume": {
            "all_hr_rows": len(all_hr),
            "all_pred_rows": len(all_pred),
            "pred_by_key_unique": len(pred_by_key),
        },
        "filter_funnel": {
            "total_hr_rows_seen": total,
            "in_window": in_window,
            "pred_found": pred_found,
            "pred_missing": pred_missing,
            "time_guard_passed": time_guard_passed,
            "time_guard_filtered": time_guard_filtered,
            "recompute_ok": recompute_ok,
            "fallback_used": fallback_used,
            "fallback_ok": fallback_ok,
            "exception_caught": exception_caught,
            "final_added_to_race_groups": final_added,
        },
        "race_group_stats": {
            "unique_races_in_window": len(races_in_window),
            "races_with_at_least_one_winner": races_with_winner,
            "valid_race_groups_after_train_race_grouped_check": valid_race_groups,
            "need_at_least_50": valid_race_groups >= 50,
        },
        "miss_samples": {
            "pred_missing": pred_missing_samples,
            "time_guard_filtered": time_guard_samples,
            "exceptions": exception_samples,
        },
    }


@app.get("/api/admin/recompute-debug")
async def recompute_debug(
    sample: int = Query(5, ge=1, le=20),
    x_cron_secret: Optional[str] = Header(None),
):
    """Process `sample` random history rows through the full
    recompute + fallback chain and report exactly what happens for each row —
    including the actual exception message when something fails.

    This is the diagnostic the calibration sweep needed but didn't have.
    """
    _check_admin(x_cron_secret)
    import random as _random
    import traceback as _tb
    from horse_engine.prediction.clean_features import (
        AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
        safe_enriched_runner, _patch_clean_aggregates,
    )

    async with get_session() as session:
        candidates = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )).scalars().all()
        hr_rows = (await session.execute(select(HistoricalResultRow))).scalars().all()

    if not candidates:
        return {"error": "no candidate rows"}

    chosen = _random.sample(candidates, min(sample, len(candidates)))
    index = AggregateIndex(hr_rows)

    results = []
    for row in chosen:
        item: dict = {
            "race_id": row.race_id,
            "horse_name": row.horse_name,
            "enriched_at": row.enriched_at.isoformat() if row.enriched_at else None,
        }
        # Step 1: raw construction — what the OLD code did
        try:
            raw_er = EnrichedRunner(**json.loads(row.enriched_json))
            item["raw_construct"] = "ok"
            try:
                build_feature_vector(raw_er)
                item["raw_build_fv"] = "ok"
            except Exception as e:
                item["raw_build_fv"] = f"failed: {type(e).__name__}: {e}"
        except Exception as e:
            item["raw_construct"] = f"failed: {type(e).__name__}: {e}"

        # Step 2: safe construction (current fallback path)
        try:
            safe_er = safe_enriched_runner(json.loads(row.enriched_json))
            item["safe_construct"] = "ok" if safe_er is not None else "returned None"
            if safe_er is not None:
                try:
                    build_feature_vector(safe_er)
                    item["safe_build_fv"] = "ok"
                except Exception as e:
                    item["safe_build_fv"] = f"failed: {type(e).__name__}: {e}"
        except Exception as e:
            item["safe_construct"] = f"raised: {type(e).__name__}: {e}"

        # Step 3: clean recompute path
        try:
            fv = recompute_clean_feature_vector(row, index)
            item["recompute"] = "ok" if fv is not None else "returned None"
        except Exception as e:
            item["recompute"] = f"raised: {type(e).__name__}: {e}"

        # Step 4: fallback path
        try:
            fv = fallback_feature_vector(row)
            item["fallback"] = "ok" if fv is not None else "returned None"
        except Exception as e:
            item["fallback"] = f"raised: {type(e).__name__}: {e}"

        results.append(item)

    # Aggregate summary
    summary = {
        "raw_construct_ok": sum(1 for r in results if r.get("raw_construct") == "ok"),
        "safe_construct_ok": sum(1 for r in results if r.get("safe_construct") == "ok"),
        "recompute_ok": sum(1 for r in results if r.get("recompute") == "ok"),
        "fallback_ok": sum(1 for r in results if r.get("fallback") == "ok"),
        "total": len(results),
    }
    return {"summary": summary, "rows": results}


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
async def seed_results(race_date: str, request: Request, x_cron_secret: Optional[str] = Header(None)):
    """Fetch race results from Racing Australia for a past date and store as training data."""
    _enforce_caller_rate(request, "seed-results")
    _check_admin(x_cron_secret)
    skip, age = _should_debounce("seed-results", race_date)
    if skip:
        return {
            "status": "debounced",
            "date": race_date,
            "seconds_since_last_call": round(age, 1),
            "cooldown_seconds": _ADMIN_DEBOUNCE_SECONDS,
        }
    seeded = await _seed_results_for_date(race_date)
    return {"status": "seeded", "results": seeded}


@app.post("/api/admin/seed-ra-results/{race_date}")
async def seed_ra_results(
    race_date: str,
    request: Request,
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
    _enforce_caller_rate(request, "seed-ra-results")
    _check_admin(x_cron_secret)
    skip, age = _should_debounce("seed-ra-results", race_date)
    if skip and not force:
        return {
            "status": "debounced",
            "date": race_date,
            "seconds_since_last_call": round(age, 1),
            "cooldown_seconds": _ADMIN_DEBOUNCE_SECONDS,
        }
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
        # Per-venue try/except — one venue's parse failure (e.g. Geelong's
        # 31KB abandoned-meeting page on 2026-07-10) MUST NOT crash the
        # whole seed sweep. Was 500-ing the endpoint for every date that
        # had any pathological RA page.
        try:
            ra_key, results = await ra.find_results(race_date, state, venue_name)
        except Exception as e:
            log.warning("[seed-ra-results] %s/%s find_results raised: %s",
                        venue_name, state, e)
            detail.append({"venue": venue_name, "state": state,
                           "races_found": 0, "error": str(e)[:200]})
            continue
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

                    # Match by race + normalized horse name. When duplicates
                    # exist (post-race re-enrichment can leave two rows
                    # for the same horse — one with tab_number, one
                    # without), prefer the one WITH tab_number so
                    # HistoricalResultRow.tab_number populates for the
                    # settle path. Falling back on mutable-then-history-
                    # then-nothing.
                    norm_name = _normalize_horse(name_lower)
                    _matches = [p for p in pred_rows if p.race_id == race_id
                                and _normalize_horse(p.horse_name) == norm_name]
                    matched_pred = next(
                        (p for p in _matches if p.tab_number is not None),
                        _matches[0] if _matches else None,
                    )
                    # Fallback tab_number: if the matched mutable row has
                    # no tab, look up in history (source="live" only).
                    tab_from_hist = None
                    if matched_pred is None or matched_pred.tab_number is None:
                        matched_hist_tab = hist_by_key.get((race_id, norm_name))
                        if matched_hist_tab and matched_hist_tab.tab_number is not None:
                            tab_from_hist = matched_hist_tab.tab_number

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
                        # Without tab_number, _settle_bets_for_race can't
                        # map finishing positions to bet-box tab numbers
                        # and leaves the race in 'pending' forever.
                        # Priority: matched_pred.tab_number → history-row
                        # fallback → None.
                        tab_number=(
                            getattr(matched_pred, "tab_number", None)
                            if matched_pred and matched_pred.tab_number is not None
                            else tab_from_hist
                        ),
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


@app.post("/api/admin/patch-tab-numbers/{race_date}")
async def patch_tab_numbers(
    race_date: str,
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Backfill NULL HistoricalResultRow.tab_number for every race on
    race_date by matching horse_name against RunnerPredictionRow /
    RunnerPredictionHistoryRow. Fixes rows seeded by the admin path
    before the tab_number-population patch (2026-07-11) — settlement
    can't map finishing positions without tab numbers, so those bets
    stay pending forever.

    Reports per-race how many rows got patched and which horses
    couldn't be resolved.
    """
    _check_admin(x_cron_secret)
    from sqlalchemy import update as sa_update

    detail: list[dict] = []
    total_patched = 0
    async with get_session() as session:
        result_rows = (await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id.like(f"{race_date}_%"))
            .where(HistoricalResultRow.tab_number.is_(None))
        )).scalars().all()

        by_race: dict[str, list] = {}
        for r in result_rows:
            by_race.setdefault(r.race_id, []).append(r)

        for race_id, rows in by_race.items():
            tab_lookup: dict[str, int] = {}
            for src in (RunnerPredictionRow, RunnerPredictionHistoryRow):
                pred = (await session.execute(
                    select(src.horse_name, src.tab_number)
                    .where(src.race_id == race_id)
                    .where(src.tab_number.isnot(None))
                )).fetchall()
                for name, tab in pred:
                    tab_lookup[_normalize_horse(name)] = tab

            patched = 0
            unresolved: list[str] = []
            for r in rows:
                t = tab_lookup.get(_normalize_horse(r.horse_name))
                if t is not None:
                    await session.execute(
                        sa_update(HistoricalResultRow)
                        .where(HistoricalResultRow.id == r.id)
                        .values(tab_number=t)
                    )
                    patched += 1
                else:
                    unresolved.append(r.horse_name)
            if patched:
                await session.commit()
                total_patched += patched
            detail.append({
                "race_id": race_id,
                "patched": patched,
                "unresolved": unresolved,
                "lookup_size": len(tab_lookup),
            })

    return {
        "ok": True,
        "race_date": race_date,
        "total_patched": total_patched,
        "races_processed": len(detail),
        "detail": [d for d in detail if d["patched"] or d["unresolved"]],
    }


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
async def backfill_status(x_cron_secret: Optional[str] = Header(None)):
    """Current backfill progress."""
    _check_admin(x_cron_secret)
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
async def db_backfill_status(x_cron_secret: Optional[str] = Header(None)):
    """Current DB backfill progress."""
    _check_admin(x_cron_secret)
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

@app.get("/api/admin/backtest-reliability")
async def backtest_reliability(
    days: int = Query(90, ge=14, le=180),
    lookback: int = Query(180, ge=30, le=365),
    min_prior_picks: int = Query(3, ge=1, le=10),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Backtest the proposed horse-reliability "trust badge" signal.

    For every rank-1 pick over `days` where the race has a settled
    winner, look back at that same horse's prior rank-1 picks (in the
    `lookback` window BEFORE the current race, model_pct ≥ 25%) and
    compute a badge as-of-race-day. Then measure the current-race
    outcome per badge tier and compare against the overall baseline.

    If badged horses hit meaningfully more than unbadged same-model_pct
    horses, the badge is predictive. If lift is <2pp, it's decoration.

    Uses history snapshots + immutable results only — no leakage.
    """
    _check_admin(x_cron_secret)
    horizon_end = date.today()
    horizon_start = horizon_end - timedelta(days=days)
    prior_cutoff = horizon_start - timedelta(days=lookback)

    async with get_session() as session:
        # Pull every rank-1 live-source snapshot across the outer window
        # (evaluation) plus the prior window (needed to build as-of stats).
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= prior_cutoff.isoformat())
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .where(RunnerPredictionHistoryRow.win_probability >= 0.25)
        )).scalars().all()

        race_ids = list({p.race_id for p in pred_rows})
        hr_rows = (await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id.in_(race_ids))
        )).scalars().all()

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}

    # Group picks per horse chronologically so we can walk them and build
    # as-of-race-day priors without contamination.
    picks_by_horse: dict[str, list] = {}
    for p in pred_rows:
        key = _normalize_horse(p.horse_name)
        picks_by_horse.setdefault(key, []).append(p)
    for key in picks_by_horse:
        picks_by_horse[key].sort(key=lambda p: p.race_id)  # race_id starts with date

    horizon_start_iso = horizon_start.isoformat()

    def _classify(n: int, wins: int, placed: int) -> str:
        if n < min_prior_picks:
            return "unbadged"
        wr = wins / n
        pr = placed / n
        if wr >= 0.66:
            return "locked_in"
        if wr >= 0.50 or pr >= 0.75:
            return "reliable"
        if n >= 4 and wr <= 0.20 and pr <= 0.40:
            return "fades"
        return "unbadged"

    # Bucket: {badge: {"n_picks": int, "wins": int, "placed": int}}
    buckets = {b: {"n": 0, "wins": 0, "placed": 0} for b in
               ("locked_in", "reliable", "fades", "unbadged")}
    total = {"n": 0, "wins": 0, "placed": 0}

    for key, picks in picks_by_horse.items():
        # Walk prior picks — record hits — then for picks in the eval
        # horizon, classify AS-OF that pick (using prior stats only).
        prior_n = prior_wins = prior_placed = 0
        # Cap prior window at `lookback` days
        recent_hits: list[tuple[str, int, int]] = []  # (race_date, won, placed)
        for p in picks:
            race_date = p.race_id[:10]
            # Trim priors older than lookback from current pick's date
            cutoff_iso = (date.fromisoformat(race_date) - timedelta(days=lookback)).isoformat()
            while recent_hits and recent_hits[0][0] < cutoff_iso:
                _, w, pl = recent_hits.pop(0)
                prior_wins -= w
                prior_placed -= pl
                prior_n -= 1

            # Only count as an "evaluation" pick if it's in the horizon
            if race_date >= horizon_start_iso:
                r = result_by_key.get((p.race_id, key))
                if r and r.position and r.position > 0:
                    badge = _classify(prior_n, prior_wins, prior_placed)
                    buckets[badge]["n"] += 1
                    total["n"] += 1
                    if r.winner:
                        buckets[badge]["wins"] += 1
                        total["wins"] += 1
                    if r.placed:
                        buckets[badge]["placed"] += 1
                        total["placed"] += 1

            # Then update prior stats to include this pick (only if the
            # race actually ran — scratchings don't count either way).
            r = result_by_key.get((p.race_id, key))
            if r and r.position and r.position > 0:
                w = 1 if r.winner else 0
                pl = 1 if r.placed else 0
                recent_hits.append((race_date, w, pl))
                prior_wins += w
                prior_placed += pl
                prior_n += 1

    def _stats(b):
        n = b["n"]
        return {
            "n": n,
            "wins": b["wins"],
            "placed": b["placed"],
            "win_rate": round(b["wins"] / n, 4) if n else None,
            "place_rate": round(b["placed"] / n, 4) if n else None,
        }

    baseline_win = total["wins"] / total["n"] if total["n"] else 0.0
    baseline_place = total["placed"] / total["n"] if total["n"] else 0.0

    per_bucket = {}
    for b, agg in buckets.items():
        s = _stats(agg)
        s["win_lift_vs_baseline"] = round(s["win_rate"] - baseline_win, 4) if s["win_rate"] is not None else None
        s["place_lift_vs_baseline"] = round(s["place_rate"] - baseline_place, 4) if s["place_rate"] is not None else None
        per_bucket[b] = s

    return {
        "params": {
            "days": days,
            "lookback": lookback,
            "min_prior_picks": min_prior_picks,
            "evaluation_window": [horizon_start.isoformat(), horizon_end.isoformat()],
        },
        "baseline": {
            "n": total["n"],
            "win_rate": round(baseline_win, 4),
            "place_rate": round(baseline_place, 4),
        },
        "per_badge": per_bucket,
    }


@app.get("/api/admin/backtest-trainer-reliability")
async def backtest_trainer_reliability(
    days: int = Query(90, ge=14, le=180),
    lookback: int = Query(180, ge=30, le=365),
    min_prior_picks: int = Query(5, ge=2, le=30),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Backtest a per-TRAINER reliability signal.

    For every rank-1 pick in the eval window, look at that TRAINER's
    prior rank-1 hit rate over the lookback window and bucket:
      elite:   ≥40% win rate over min_prior_picks
      solid:   ≥30% win rate over min_prior_picks
      average: 20-30% win rate
      weak:    <20% win rate over min_prior_picks
      unrated: fewer than min_prior_picks prior picks

    If elite/solid trainers lift meaningfully above baseline (say +3pp),
    the badge is real and worth building at the trainer level rather
    than the horse level (which just failed the backtest at 0 badged
    picks out of 424).
    """
    _check_admin(x_cron_secret)
    horizon_end = date.today()
    horizon_start = horizon_end - timedelta(days=days)
    prior_cutoff = horizon_start - timedelta(days=lookback)

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= prior_cutoff.isoformat())
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .where(RunnerPredictionHistoryRow.trainer.isnot(None))
        )).scalars().all()

        race_ids = list({p.race_id for p in pred_rows})
        hr_rows = (await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id.in_(race_ids))
        )).scalars().all()

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}

    # Group picks by normalized trainer name so common spelling
    # variants ("Ciaron Maher & D Eustace" vs "Ciaron Maher") don't
    # split a trainer's track record.
    def _norm_trainer(t: str) -> str:
        return (t or "").strip().lower()

    picks_by_trainer: dict[str, list] = {}
    for p in pred_rows:
        picks_by_trainer.setdefault(_norm_trainer(p.trainer), []).append(p)
    for k in picks_by_trainer:
        picks_by_trainer[k].sort(key=lambda p: p.race_id)

    horizon_start_iso = horizon_start.isoformat()

    def _classify(n: int, wins: int) -> str:
        if n < min_prior_picks:
            return "unrated"
        wr = wins / n
        if wr >= 0.40:
            return "elite"
        if wr >= 0.30:
            return "solid"
        if wr >= 0.20:
            return "average"
        return "weak"

    buckets = {b: {"n": 0, "wins": 0, "placed": 0} for b in
               ("elite", "solid", "average", "weak", "unrated")}
    total = {"n": 0, "wins": 0, "placed": 0}

    for trainer_key, picks in picks_by_trainer.items():
        if not trainer_key:
            continue
        prior_n = prior_wins = 0
        recent_hits: list[tuple[str, int]] = []
        for p in picks:
            race_date = p.race_id[:10]
            cutoff_iso = (date.fromisoformat(race_date) - timedelta(days=lookback)).isoformat()
            while recent_hits and recent_hits[0][0] < cutoff_iso:
                _, w = recent_hits.pop(0)
                prior_wins -= w
                prior_n -= 1

            if race_date >= horizon_start_iso:
                r = result_by_key.get((p.race_id, _normalize_horse(p.horse_name)))
                if r and r.position and r.position > 0:
                    badge = _classify(prior_n, prior_wins)
                    buckets[badge]["n"] += 1
                    total["n"] += 1
                    if r.winner:
                        buckets[badge]["wins"] += 1
                        total["wins"] += 1
                    if r.placed:
                        buckets[badge]["placed"] += 1
                        total["placed"] += 1

            r = result_by_key.get((p.race_id, _normalize_horse(p.horse_name)))
            if r and r.position and r.position > 0:
                w = 1 if r.winner else 0
                recent_hits.append((race_date, w))
                prior_wins += w
                prior_n += 1

    def _stats(b, baseline_win, baseline_place):
        n = b["n"]
        wr = b["wins"] / n if n else None
        pr = b["placed"] / n if n else None
        return {
            "n": n,
            "wins": b["wins"],
            "placed": b["placed"],
            "win_rate": round(wr, 4) if wr is not None else None,
            "place_rate": round(pr, 4) if pr is not None else None,
            "win_lift_vs_baseline": round(wr - baseline_win, 4) if wr is not None else None,
            "place_lift_vs_baseline": round(pr - baseline_place, 4) if pr is not None else None,
        }

    baseline_win = total["wins"] / total["n"] if total["n"] else 0.0
    baseline_place = total["placed"] / total["n"] if total["n"] else 0.0
    per_bucket = {b: _stats(agg, baseline_win, baseline_place) for b, agg in buckets.items()}

    return {
        "params": {"days": days, "lookback": lookback, "min_prior_picks": min_prior_picks,
                   "evaluation_window": [horizon_start.isoformat(), horizon_end.isoformat()]},
        "baseline": {"n": total["n"], "win_rate": round(baseline_win, 4),
                     "place_rate": round(baseline_place, 4)},
        "per_bucket": per_bucket,
    }


@app.get("/api/admin/backtest-last-start-alignment")
async def backtest_last_start_alignment(
    days: int = Query(90, ge=14, le=180),
    lookback: int = Query(90, ge=14, le=180),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Backtest a per-horse LAST-START-ALIGNMENT signal.

    For every rank-1 pick in the eval window, find that horse's most
    recent PRIOR race (any race with a settled result) within lookback
    and classify:
      hot:      prior race, model_pct ≥ 25%, WON
      warm:     prior race, model_pct ≥ 25%, PLACED (but didn't win)
      cold:     prior race, model_pct ≥ 25%, position ≥ 5 (badly beaten)
      neutral:  prior race, model_pct ≥ 25%, position 4 (close miss)
      off_radar: prior race, model_pct < 25% (we didn't pick them then)
      no_prior:  horse's first race in our history within lookback

    Every rank-1 pick has SOME classification (data drought unlikely).
    Then measure current-pick outcome per bucket.
    """
    _check_admin(x_cron_secret)
    horizon_end = date.today()
    horizon_start = horizon_end - timedelta(days=days)
    prior_cutoff = horizon_start - timedelta(days=lookback)

    async with get_session() as session:
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= prior_cutoff.isoformat())
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )).scalars().all()

        race_ids = list({p.race_id for p in pred_rows})
        hr_rows = (await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id.in_(race_ids))
        )).scalars().all()

    result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}

    # Group EVERY pick per horse (all ranks, not just 1), so we can
    # find the horse's most recent prior race regardless of what we
    # thought of them last time.
    all_picks_by_horse: dict[str, list] = {}
    for p in pred_rows:
        key = _normalize_horse(p.horse_name)
        all_picks_by_horse.setdefault(key, []).append(p)
    for key in all_picks_by_horse:
        all_picks_by_horse[key].sort(key=lambda p: p.race_id)

    horizon_start_iso = horizon_start.isoformat()

    def _classify(prior_pick, prior_result) -> str:
        if not prior_pick or not prior_result or not prior_result.position or prior_result.position <= 0:
            return "no_prior"
        prior_pct = (prior_pick.win_probability or 0) * 100
        if prior_pct < 25.0:
            return "off_radar"
        pos = prior_result.position
        if prior_result.winner:
            return "hot"
        if prior_result.placed:
            return "warm"
        if pos == 4:
            return "neutral"
        if pos >= 5:
            return "cold"
        return "neutral"

    buckets = {b: {"n": 0, "wins": 0, "placed": 0} for b in
               ("hot", "warm", "neutral", "cold", "off_radar", "no_prior")}
    total = {"n": 0, "wins": 0, "placed": 0}

    for key, picks in all_picks_by_horse.items():
        for i, p in enumerate(picks):
            race_date = p.race_id[:10]
            if race_date < horizon_start_iso:
                continue
            if p.model_rank != 1:
                continue  # only score rank-1 picks
            cur_result = result_by_key.get((p.race_id, key))
            if not cur_result or not cur_result.position or cur_result.position <= 0:
                continue

            # Find most recent prior pick within lookback window
            prior_pick = None
            prior_result = None
            cutoff_iso = (date.fromisoformat(race_date) - timedelta(days=lookback)).isoformat()
            for j in range(i - 1, -1, -1):
                cand = picks[j]
                cand_date = cand.race_id[:10]
                if cand_date < cutoff_iso:
                    break
                r = result_by_key.get((cand.race_id, key))
                if r and r.position and r.position > 0:
                    prior_pick = cand
                    prior_result = r
                    break

            badge = _classify(prior_pick, prior_result)
            buckets[badge]["n"] += 1
            total["n"] += 1
            if cur_result.winner:
                buckets[badge]["wins"] += 1
                total["wins"] += 1
            if cur_result.placed:
                buckets[badge]["placed"] += 1
                total["placed"] += 1

    baseline_win = total["wins"] / total["n"] if total["n"] else 0.0
    baseline_place = total["placed"] / total["n"] if total["n"] else 0.0

    def _stats(b):
        n = b["n"]
        wr = b["wins"] / n if n else None
        pr = b["placed"] / n if n else None
        return {
            "n": n,
            "wins": b["wins"],
            "placed": b["placed"],
            "win_rate": round(wr, 4) if wr is not None else None,
            "place_rate": round(pr, 4) if pr is not None else None,
            "win_lift_vs_baseline": round(wr - baseline_win, 4) if wr is not None else None,
            "place_lift_vs_baseline": round(pr - baseline_place, 4) if pr is not None else None,
        }

    return {
        "params": {"days": days, "lookback": lookback,
                   "evaluation_window": [horizon_start.isoformat(), horizon_end.isoformat()]},
        "baseline": {"n": total["n"], "win_rate": round(baseline_win, 4),
                     "place_rate": round(baseline_place, 4)},
        "per_bucket": {b: _stats(agg) for b, agg in buckets.items()},
    }


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
            if actual.position == 1:
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


@app.get("/api/admin/backtest/exotic-feature-ablation")
async def exotic_feature_ablation(
    holdout_days: int = Query(14, ge=7, le=30),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Feature ablation for the EXOTIC model. For each feature, zero it out
    across holdout races and measure the change in trifecta box hit rate
    (top-3 ranked runners == actual positions 1-2-3 in any order).

    Mirrors /api/admin/backtest/feature-ablation but optimises for box
    coverage rather than top-1 win rate. Positive delta = removing the
    feature improves trifecta box hit rate (harmful feature). Field size
    must be >= 7 (consistent with exotic model training).
    """
    _check_admin(x_cron_secret)
    from horse_engine.prediction.features import FEATURE_NAMES
    from horse_engine.prediction.clean_features import (
        AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
    )

    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        exotic_weights = await load_exotic_model_weights(session)
        all_hr = (await session.execute(select(HistoricalResultRow))).scalars().all()
        # Match exotic calibration sweep filters so the ablation measures
        # the same distribution the model trained on.
        all_pred = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill")))
        )).scalars().all()

    model = ExoticModel.from_weights_dict(exotic_weights) if exotic_weights else ExoticModel()
    index = AggregateIndex(all_hr)
    pred_by_key = {(p.race_id, p.horse_name): p for p in all_pred}

    # Build holdout per race: result rows + feature vectors per runner.
    # Same clean-recompute chain as the calibration sweep.
    holdout_race_results: dict[str, list[HistoricalResultRow]] = {}
    for r in all_hr:
        if r.race_id >= holdout_cutoff and r.position is not None:
            holdout_race_results.setdefault(r.race_id, []).append(r)

    holdout_fvs: dict[str, list[tuple]] = {}  # race_id → [(result_row, feature_vector)]
    for race_id, result_rows in holdout_race_results.items():
        if len(result_rows) < 7:  # exotic model only trained on field_size >= 7
            continue
        runner_fvs = []
        for r in result_rows:
            pred = pred_by_key.get((race_id, r.horse_name))
            if not pred:
                continue
            fv = recompute_clean_feature_vector(pred, index)
            if fv is None:
                fv = fallback_feature_vector(pred)
            if fv is None:
                continue
            runner_fvs.append((r, fv))
        if len(runner_fvs) >= 7:
            holdout_fvs[race_id] = runner_fvs

    def _tri_box_hit_rate(fv_sets):
        tri_hits = tri_races = 0
        for race_id, runner_fvs in fv_sets.items():
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
        return round(tri_hits / tri_races * 100, 2) if tri_races else 0.0, tri_races

    baseline_rate, total_races = _tri_box_hit_rate(holdout_fvs)

    ablation = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        zeroed = {}
        for race_id, runner_fvs in holdout_fvs.items():
            zeroed[race_id] = [
                (r, [v if i != feat_idx else 0.0 for i, v in enumerate(fv)])
                for r, fv in runner_fvs
            ]
        ablated_rate, _ = _tri_box_hit_rate(zeroed)
        delta = round(ablated_rate - baseline_rate, 2)
        ablation.append({
            "feature": feat_name,
            "weight": round(model.weights[feat_idx] if feat_idx < len(model.weights) else 0.0, 4),
            "baseline_tri_box_rate": baseline_rate,
            "ablated_tri_box_rate": ablated_rate,
            "delta": delta,
            "verdict": "valuable" if delta < -0.5 else ("noisy/harmful" if delta > 0.5 else "neutral"),
        })

    ablation.sort(key=lambda x: x["delta"])
    return {
        "holdout_races": total_races,
        "holdout_days": holdout_days,
        "baseline_tri_box_rate_pct": baseline_rate,
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


_PERF_CACHE: dict[tuple[int, bool], tuple[datetime, dict]] = {}
_PERF_CACHE_TTL_TODAY = 60        # 1 min — today's data still moves
_PERF_CACHE_TTL_PAST = 30 * 60    # 30 min — past dates are stable
_PERF_CACHE_MAX_ENTRIES = 16


@app.get("/api/performance")
async def performance_summary(
    days: int = Query(5, ge=1, le=365),
    sharp: bool = Query(False),
):
    """
    Per-day performance strip for the last N days.
    Shows top-pick win rate, place rate, and value P&L per day.
    No auth required — displayed publicly on the frontend.

    When sharp=true, filters to the high-confidence niche (rank-1
    model_pct ≥ 30 OR top-3 sum ≥ 60) — the band where the model
    historically hits 30-35% win rate vs the ~17% overall baseline.

    Cached in-memory: 60s for windows that include today, 30 min for
    pure past-date windows. The Lounge fires this on every page load so
    caching takes the cold response from ~5s to ~5ms.
    """
    cache_key = (int(days), bool(sharp))
    # Today is always in the window since we look back N days from today,
    # so the TTL is the "today" tier. (If you ever add an explicit
    # `end_date` param, branch the TTL on whether end_date >= today.)
    ttl = _PERF_CACHE_TTL_TODAY
    cached = _PERF_CACHE.get(cache_key)
    if cached is not None:
        ts, body = cached
        if (datetime.utcnow() - ts).total_seconds() < ttl:
            return body
    cutoff = (_today_aest() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )
        hr_rows = hr_result.scalars().all()

        if not hr_rows:
            return {"days": days, "summary": [], "overall_win_rate": None, "sharp": sharp}

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

        # Sharp filter — read the snapshot-time `is_sharp` flag straight
        # off the rank-1 row. The flag is frozen at race time so future
        # gate refinements don't retroactively rewrite historical Sharp
        # counts. See RunnerPredictionHistoryRow.is_sharp.
        if sharp and top_picks:
            top_picks = {rid: p for rid, p in top_picks.items() if p.is_sharp is True}

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
        actual = result_by_key.get((race_id, _normalize_horse(pick.horse_name)))
        # Scratched-rank-1 races aren't model misses — the horse didn't run.
        # Drop them from both numerator and denominator so win-rate reflects
        # the slice the model actually competed in. Matches the nightly
        # review's "rank1_races_with_result" denominator.
        if not actual or not actual.position or actual.position <= 0:
            continue
        d = by_date.setdefault(race_date, {
            "races": 0, "wins": 0, "places": 0, "value_pnl": 0.0, "value_bets": 0,
            "tier_premium": 0, "tier_hot": 0, "tier_high": 0, "tier_strong": 0,
        })
        d["races"] += 1
        act_won = _normalize_horse(pick.horse_name) == _normalize_horse(winner)
        act_placed = bool(actual.position and actual.position <= 3) or act_won
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

    # "Sharp meetings" metric — rank-1 wins across meetings that had at
    # least one Sharp-qualifying race that day. Matches what a user sees
    # when they click through Sharp mode on the Lounge (all races at
    # each Sharp-qualifying meeting), rather than the narrower "only
    # Sharp races" cut. E.g., Ballarat Tuesday had 1 Sharp race but the
    # model went 4/6 across the whole card — this exposes that.
    if not sharp:
        # Find (date, venue_code) pairs that had at least one is_sharp=TRUE
        # rank-1 pick. Uses the top_picks we already loaded.
        sharp_venues_by_date: dict[str, set] = {}
        for rid, pick in top_picks.items():
            if pick.is_sharp is True:
                d = rid[:10]
                try:
                    venue_code = rid.split("_", 2)[1]
                except IndexError:
                    continue
                sharp_venues_by_date.setdefault(d, set()).add(venue_code)
        # Now walk the full rank-1 set and count wins at Sharp meetings only.
        for row in summary:
            d = row["date"]
            sharp_venues = sharp_venues_by_date.get(d, set())
            if not sharp_venues:
                row["sharp_meeting_races"] = 0
                row["sharp_meeting_wins"] = 0
                row["sharp_meeting_win_rate"] = None
                row["sharp_meeting_place_rate"] = None
                continue
            sm_races = 0
            sm_wins = 0
            sm_placed = 0
            for rid, pick in top_picks.items():
                if not rid.startswith(f"{d}_"):
                    continue
                try:
                    venue_code = rid.split("_", 2)[1]
                except IndexError:
                    continue
                if venue_code not in sharp_venues:
                    continue
                winner = winners.get(rid)
                if not winner:
                    continue
                actual = result_by_key.get((rid, _normalize_horse(pick.horse_name)))
                if not actual or not actual.position or actual.position <= 0:
                    continue
                sm_races += 1
                if _normalize_horse(pick.horse_name) == _normalize_horse(winner):
                    sm_wins += 1
                if actual.position <= 3:
                    sm_placed += 1
            row["sharp_meeting_races"] = sm_races
            row["sharp_meeting_wins"] = sm_wins
            row["sharp_meeting_win_rate"] = round(sm_wins / sm_races, 3) if sm_races else None
            row["sharp_meeting_place_rate"] = round(sm_placed / sm_races, 3) if sm_races else None

    total_races = sum(d["races"] for d in by_date.values())
    total_wins = sum(d["wins"] for d in by_date.values())
    body = {
        "days": days,
        "sharp": sharp,
        "overall_win_rate": round(total_wins / total_races, 3) if total_races else None,
        "overall_races": total_races,
        "summary": summary,
    }
    # Cache the response. Trim to MAX_ENTRIES so we don't grow unbounded
    # across (days, sharp) variations.
    _PERF_CACHE[cache_key] = (datetime.utcnow(), body)
    if len(_PERF_CACHE) > _PERF_CACHE_MAX_ENTRIES:
        # drop the oldest entry by timestamp
        oldest_key = min(_PERF_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _PERF_CACHE.pop(oldest_key, None)
    return body


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
    P&L analysis for Premium picks. Filter:
      win model's top-1 pick AND place_model_rank >= 2

    Why this filter (2026-06-12 redefinition):
    The previous value-betting filter (model_pct>=30 AND SP>=3 AND overlay>5%)
    optimised for ROI/value and produced only ~4 picks per 30 days — too small
    to meaningfully report on.

    The win-place ensemble diagnostic showed that when the win model's top-1
    pick is ALSO the place model's #1 (a 'consensus' pick), it tends to be a
    well-known favourite and wins at only 16.8%. When the place model dissents
    (ranks it 2+), the win model has found a fundamentals-driven pick that wins
    at 24.5% on 165 picks per 30 days. That's the new Premium definition.

    Coverage is ~32% of all top-1 picks. Win rate +5pp over the unfiltered
    baseline. See [[feedback_win_rate_primary]] for the rationale on optimising
    win % over ROI.

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
        if (pick.place_model_rank or 0) >= 2:
            pnl = (sp - 1.0) if actual.position == 1 else -1.0
            picks.append({
                "date": race_id[:10],
                "race_id": race_id,
                "horse_name": pick.horse_name,
                "model_pct": model_pct,
                "sp": sp,
                "overlay_pct": round(overlay * 100, 1),
                "winner": actual.position == 1,
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
        if (pick.place_model_rank or 0) >= 2:
            bets += 1
            if actual.position == 1:
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


_premium_monthly_cache: tuple[datetime, dict] | None = None
_PREMIUM_MONTHLY_TTL = 600  # 10 min — monthly P&L only changes on settlement

@app.get("/api/performance/premium/monthly")
async def premium_performance_monthly():
    """Public monthly breakdown of Premium pick P&L for last 6 months inc MTD (no auth required)."""
    global _premium_monthly_cache
    if _premium_monthly_cache is not None:
        ts, body = _premium_monthly_cache
        if (datetime.utcnow() - ts).total_seconds() < _PREMIUM_MONTHLY_TTL:
            return body
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
        if (pick.place_model_rank or 0) >= 2:
            month = race_id[:7]  # YYYY-MM
            if month not in monthly:
                monthly[month] = {"bets": 0, "wins": 0, "pnl": 0.0}
            monthly[month]["bets"] += 1
            if actual.position == 1:
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

    body = {"months": months_out}
    _premium_monthly_cache = (datetime.utcnow(), body)
    return body


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
        if (pick.place_model_rank or 0) >= 2:
            day = race_id[:10]
            if day not in daily:
                daily[day] = {"bets": 0, "wins": 0, "pnl": 0.0}
            daily[day]["bets"] += 1
            if actual.position == 1:
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

    Uses the BUG-18-clean recompute path so candidate-window scoring sees the
    same feature distribution that production inference now uses.

    Iterates predictions directly (mirroring retrain_model's pattern) rather
    than walking HistoricalResultRow and trying to look up matching predictions.
    The HR table currently has ~136K rows but only ~15K of them have a matching
    RunnerPredictionHistoryRow — most are for older races that were never
    enriched. The old "iterate HR" design hit 99.8% pred_missing and dropped
    every window; iterating predictions directly only processes rows that
    actually have features to train on. (2026-06-11 calibration sweep rewrite.)
    """
    from horse_engine.prediction.clean_features import (
        AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
    )
    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        all_hr = (await session.execute(select(HistoricalResultRow))).scalars().all()
        # Calibration deliberately includes source IN ('live', 'backfill') here.
        # 'source="live"' alone produces only ~15K rows heavily skewed to the
        # last few weeks, and the 14-day holdout eats most of them — every
        # window dropped to <50 training races. Including 'backfill' rows
        # (backfilled history from older races) unlocks the full 9-month
        # training horizon. The 'backfill' rows can contain BUG-18-contaminated
        # aggregates BUT recompute_clean_feature_vector rebuilds those fields
        # from the date-safe AggregateIndex on the fly, so contamination is
        # neutralised at training time. 'validation' and 'backtest' sources
        # are still excluded — those rows come from validation/backtest jobs
        # and would leak backtest-fit signal into the model.
        all_pred = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill")))
        )).scalars().all()

    # BUG-18-clean aggregate index, built once over full HR pool.
    index = AggregateIndex(all_hr)

    # Build winners per race + per-(race,horse) result lookup. Used for labels
    # in training and for holdout-pick scoring.
    winners: dict[str, str] = {}  # race_id -> normalized winner horse name
    hr_by_key: dict[tuple, HistoricalResultRow] = {}
    for r in all_hr:
        hr_by_key[(r.race_id, r.horse_name)] = r
        if r.position == 1:
            winners[r.race_id] = _normalize_horse(r.horse_name)

    # Holdout = predictions in the most recent N days, grouped by race.
    holdout_races: dict[str, list] = {}
    for p in all_pred:
        if p.race_id >= holdout_cutoff:
            holdout_races.setdefault(p.race_id, []).append(p)

    window_results = []
    best_window = None
    best_score = float("-inf")
    best_weights = None

    import math as _math
    for window in _CANDIDATE_WINDOWS:
        train_cutoff = (today - timedelta(days=window)).isoformat()

        # Group training predictions by race_id. Only include races we have a
        # winner for (so labels can be assigned), within the window, that pass
        # the pre-race time guard.
        races_in_window: dict[str, list] = {}
        for pred in all_pred:
            if pred.race_id < train_cutoff or pred.race_id >= holdout_cutoff:
                continue
            if pred.race_id not in winners:
                continue  # no result yet, can't label
            # No pre-race time guard: recompute_clean_feature_vector uses the
            # BUG-18-clean AggregateIndex which only reads HR results strictly
            # before each row's race_date, so post-race enriched_at can't leak
            # this-race results into the feature vector. Adding the
            # enriched_at > scheduled_time check dropped 99% of backfilled
            # rows (their enriched_at is the backfill time, not the original
            # pre-race time) — taking us down to 28 races per window.
            # retrain_model uses the same recompute chain without the guard
            # and trains cleanly.
            races_in_window.setdefault(pred.race_id, []).append(pred)

        # Build per-race (fv, label) groups using the clean recompute chain.
        race_groups: list[list[tuple[list[float], int]]] = []
        race_sample_weights: list[float] = []
        for race_id, preds in races_in_window.items():
            winner_name = winners[race_id]
            race: list[tuple] = []
            for pred in preds:
                try:
                    fv = recompute_clean_feature_vector(pred, index)
                    if fv is None:
                        fv = fallback_feature_vector(pred)
                        if fv is None:
                            continue
                    label = 1 if _normalize_horse(pred.horse_name) == winner_name else 0
                    race.append((fv, label))
                except Exception:
                    continue
            # train_race_grouped requires ≥2 runners and exactly one winner.
            if len(race) < 2 or sum(l for _, l in race) != 1:
                continue
            race_groups.append(race)
            try:
                race_date = date.fromisoformat(race_id[:10])
                days_ago = (today - race_date).days
            except Exception:
                days_ago = 30
            race_sample_weights.append(_math.exp(-days_ago / 30.0))

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
                    # BUG-18 clean recompute for holdout scoring too, with
                    # safe-construction fallback (handles old null-fielded
                    # enriched_json).
                    fv = recompute_clean_feature_vector(r, index)
                    if fv is None:
                        fv = fallback_feature_vector(r)
                        if fv is None:
                            continue
                    runner_fvs.append((r, fv))
                except Exception:
                    continue
            if not runner_fvs:
                continue

            win_probs, _ = model.predict_field([fv for _, fv in runner_fvs])
            best_idx = win_probs.index(max(win_probs))
            top_runner = runner_fvs[best_idx][0]
            top_prob = win_probs[best_idx]

            actual = hr_by_key.get((race_id, top_runner.horse_name))
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

    # NOTE: deliberately NOT saving best_weights to production. Calibration
    # is a MEASUREMENT tool, not a deployment tool. Saving here used to
    # overwrite production weights nightly with a fresh-window retrain
    # that almost always scored lower than the all-data production model
    # (e.g. 21% vs 37% on 2026-06-13). To update production weights, fire
    # POST /api/retrain?days=0 explicitly.

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

    Uses the BUG-18-clean recompute path so candidate-window scoring sees the
    same feature distribution that production inference now uses (mirrors the
    win-sweep wiring).
    """
    from horse_engine.prediction.clean_features import (
        AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
    )
    import math as _math
    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        all_hr = (await session.execute(select(HistoricalResultRow))).scalars().all()
        # See _run_calibration_sweep — same source IN ('live', 'backfill')
        # widening so the place sweep also has access to backfilled history.
        all_pred = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill")))
        )).scalars().all()

    index = AggregateIndex(all_hr)

    # Per-(race, horse) result lookup — used to label both training (placed?)
    # and holdout (did the top pick place?) examples. Same rewrite pattern as
    # the win sweep above: iterate predictions directly instead of iterating
    # the 136K-row HR table and missing 99.8% on the join.
    hr_by_key: dict[tuple, HistoricalResultRow] = {
        (r.race_id, r.horse_name): r for r in all_hr
    }

    # Holdout = predictions in the most recent N days, grouped by race.
    holdout_races: dict[str, list] = {}
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
        for pred in all_pred:
            if pred.race_id < train_cutoff or pred.race_id >= holdout_cutoff:
                continue
            actual = hr_by_key.get((pred.race_id, pred.horse_name))
            if actual is None:
                continue  # no result for this horse
            # No pre-race time guard here (same reason as win sweep): the clean
            # recompute path is already date-safe via AggregateIndex.
            try:
                fv = recompute_clean_feature_vector(pred, index)
                if fv is None:
                    fv = fallback_feature_vector(pred)
                    if fv is None:
                        continue
                training_data.append((fv, 1 if actual.placed else 0))
                try:
                    days_ago = (today - date.fromisoformat(pred.race_id[:10])).days
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
                    # BUG-18 clean recompute for holdout scoring, with safe
                    # construction fallback for old null-fielded rows.
                    fv = recompute_clean_feature_vector(r, index)
                    if fv is None:
                        fv = fallback_feature_vector(r)
                        if fv is None:
                            continue
                    runner_fvs.append((r, fv))
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
            actual = hr_by_key.get((race_id, top_runner.horse_name))
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

    # Measurement only — see _run_calibration_sweep for full rationale.
    # To update production place weights, fire POST /api/admin/retrain-place.

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
    from horse_engine.prediction.clean_features import (
        AggregateIndex, recompute_clean_feature_vector, fallback_feature_vector,
    )
    import math as _math
    today = date.today()
    holdout_cutoff = (today - timedelta(days=holdout_days)).isoformat()

    async with get_session() as session:
        hr_result = await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.position.isnot(None))
        )
        all_hr = hr_result.scalars().all()
        # Match the win + place sweeps: source IN ('live', 'backfill') unlocks
        # the full 9-month training horizon. cancelled NULL/false stays.
        # 'validation' / 'backtest' sources are still excluded — they'd leak
        # backtest-fit signal into exotic weights.
        pred_result = await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.enriched_json.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source.in_(("live", "backfill")))
        )
        all_pred = pred_result.scalars().all()

    pred_by_key = {(p.race_id, p.horse_name): p for p in all_pred}

    # BUG-18-clean aggregate index, built once over the full HR pool. Reused
    # by recompute_clean_feature_vector below to rebuild aggregate fields
    # (trainer/jockey rates etc) using only HR rows strictly before each
    # training row's race_date.
    index = AggregateIndex(all_hr)

    def _build_fv(pred):
        """Clean-recompute first, then fall back to raw construction. Same
        chain the win + place sweeps use — neutralises BUG-18 contamination
        and survives null fields in older enriched_json."""
        fv = recompute_clean_feature_vector(pred, index)
        if fv is None:
            fv = fallback_feature_vector(pred)
        return fv

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
                # No pre-race time guard (same reason as win + place sweeps):
                # recompute_clean_feature_vector uses the date-safe
                # AggregateIndex, so post-race enriched_at can't leak this
                # race's results into the aggregate fields. The old guard
                # dropped 99% of backfilled rows.
                try:
                    fv = _build_fv(pred)
                    if fv is None:
                        continue
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
                # Same clean-recompute chain as training so the holdout score
                # is on the same feature distribution the model trained on.
                try:
                    fv = _build_fv(pred)
                    if fv is None:
                        continue
                    runner_fvs.append((r, fv))
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

    # Measurement only — see _run_calibration_sweep for full rationale.
    # To update production exotic weights, fire POST /api/admin/retrain-exotic
    # (or wait for the 03:00 AEST _scheduled_exotic_retrain cron, which trains
    # on all data the right way).

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


# ── Hot-streak badges ──────────────────────────────────────────────────────
async def _load_hot_streak_map(picks: list[dict]) -> dict[str, dict]:
    """
    For each pick, find the horse's most recent prior race (within 90d)
    and classify:
      hot:  model_pct ≥ 25% AND won
      warm: model_pct ≥ 25% AND placed (didn't win)
    No badge otherwise.

    Batched: 2 DB queries total across the whole picks list.

    Returns {current_race_id → streak_info} — only entries that qualify.
    Backtest (2026-07-11): hot picks lift +18pp win vs baseline, warm
    +11pp. See /api/admin/backtest-last-start-alignment.
    """
    if not picks:
        return {}

    horse_keys = {_normalize_horse(p["horse_name"]) for p in picks if p.get("horse_name")}
    if not horse_keys:
        return {}

    # Exact-name-set for the SQL filter. Horses in RunnerPredictionHistoryRow
    # are stored under a stable canonical spelling per race — same as the
    # value we display on /api/edge — so an exact IN(...) match catches
    # every prior prediction for the horses we care about. Without this
    # filter we scanned every history row from the last 90d, timing out
    # /api/edge at ~90s.
    pick_names = {p["horse_name"] for p in picks if p.get("horse_name")}

    cutoff = (date.today() - timedelta(days=90)).isoformat()
    async with get_session() as session:
        prior_rows = (await session.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
            )
            .where(RunnerPredictionHistoryRow.race_id >= cutoff)
            .where(RunnerPredictionHistoryRow.horse_name.in_(pick_names))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
        )).all()

    # Bucket by normalized horse. We keep only horses we actually care
    # about (in the current picks list) to bound memory on cold-cache.
    horse_history: dict[str, list] = {}
    for rid, hname, pct in prior_rows:
        key = _normalize_horse(hname)
        if key in horse_keys:
            horse_history.setdefault(key, []).append((rid, pct))

    all_prior_race_ids: set[str] = set()
    for rows in horse_history.values():
        for rid, _ in rows:
            all_prior_race_ids.add(rid)
    if not all_prior_race_ids:
        return {}

    async with get_session() as session:
        result_rows = (await session.execute(
            select(
                HistoricalResultRow.race_id,
                HistoricalResultRow.horse_name,
                HistoricalResultRow.position,
                HistoricalResultRow.winner,
                HistoricalResultRow.placed,
            )
            .where(HistoricalResultRow.race_id.in_(all_prior_race_ids))
        )).all()
    result_by_key: dict[tuple[str, str], tuple] = {
        (rid, _normalize_horse(hname)): (pos, won, plc)
        for rid, hname, pos, won, plc in result_rows
    }

    streak_map: dict[str, dict] = {}
    for p in picks:
        cur_rid = p["race_id"]
        cur_key = _normalize_horse(p.get("horse_name") or "")
        if not cur_key:
            continue
        history = horse_history.get(cur_key, [])
        # Most recent prior race (strictly before cur_rid by lexicographic
        # compare — race_ids start with YYYY-MM-DD so it's chronological).
        sorted_hist = sorted(history, key=lambda h: h[0], reverse=True)
        for prior_rid, prior_pct in sorted_hist:
            if prior_rid >= cur_rid:
                continue
            r = result_by_key.get((prior_rid, cur_key))
            if not r:
                continue  # unseeded prior — skip to next-most-recent
            pos, won, placed = r
            if not pos or pos <= 0:
                continue  # scratched — skip
            prior_pct_num = (prior_pct or 0) * 100
            if prior_pct_num < 25.0:
                # Off-radar last out — no badge, but stop here (this WAS
                # their most recent race).
                break
            if won:
                badge = "hot"
            elif placed:
                badge = "warm"
            else:
                break  # cold/neutral — no badge
            streak_map[cur_rid] = {
                "badge": badge,
                "last_race_id": prior_rid,
                "last_race_date": prior_rid[:10],
                "last_race_position": int(pos),
                "last_race_pct": round(prior_pct_num, 1),
            }
            break
    return streak_map


# ── Output calibration (isotonic curve on model_pct → win rate) ────────────
async def _load_output_calibration_curve() -> Optional[list[tuple[float, float]]]:
    """Load the persisted isotonic calibration curve. None = identity."""
    try:
        async with get_session() as session:
            row = (await session.execute(
                select(WinCalibrationCurveRow).order_by(WinCalibrationCurveRow.id.desc())
            )).scalars().first()
        if not row or not row.curve_json:
            return None
        raw = json.loads(row.curve_json)
        return [(float(x), float(y)) for x, y in raw]
    except Exception as e:
        log.debug("[output_calibration] load skipped: %s", e)
        return None


async def _compute_output_calibration_curve(days: int = 45) -> dict:
    """
    Rebuild the isotonic calibration curve from the last `days` of
    predictions vs actuals and persist it. Called nightly + on demand
    via the admin endpoint. Uses history snapshots (Ground Rule 1) so
    post-race re-enrichment can't contaminate the fit.

    Returns diagnostic dict with sample counts + curve.
    """
    from horse_engine.prediction.output_calibration import compute_calibration_curve

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    async with get_session() as session:
        hr_rows = (await session.execute(
            select(HistoricalResultRow).where(HistoricalResultRow.race_id >= cutoff)
        )).scalars().all()
        if not hr_rows:
            return {"ok": False, "reason": "no historical results in window"}
        race_ids = list({r.race_id for r in hr_rows})
        result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}

        # Every-runner slice — calibrate across the whole distribution,
        # not just rank-1. Top-pick-only would over-fit the tail and
        # under-serve the mid-band where most value bets sit.
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id.in_(race_ids))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .where(RunnerPredictionHistoryRow.win_probability.isnot(None))
        )).scalars().all()

    samples: list[tuple[float, int]] = []
    for p in pred_rows:
        r = result_by_key.get((p.race_id, _normalize_horse(p.horse_name)))
        # Skip runners whose race has no winner recorded yet (unsettled)
        # or where the horse was scratched (position 0 / null).
        if not r or r.position is None or r.position <= 0:
            continue
        model_pct = (p.win_probability or 0.0) * 100.0
        won = 1 if r.winner else 0
        samples.append((model_pct, won))

    curve = compute_calibration_curve(samples)
    if curve is None:
        return {
            "ok": False,
            "sample_size": len(samples),
            "reason": f"insufficient samples (need ~500, got {len(samples)})",
        }

    async with get_session() as session:
        existing = (await session.execute(
            select(WinCalibrationCurveRow).order_by(WinCalibrationCurveRow.id.desc())
        )).scalars().first()
        payload = json.dumps([[round(x, 2), round(y, 4)] for x, y in curve])
        if existing:
            existing.curve_json = payload
            existing.sample_days = days
            existing.sample_size = len(samples)
            existing.updated_at = datetime.utcnow()
        else:
            session.add(WinCalibrationCurveRow(
                curve_json=payload,
                sample_days=days,
                sample_size=len(samples),
            ))
        await session.commit()

    log.info("[output_calibration] rebuilt from %d samples over %d days: %d breakpoints",
             len(samples), days, len(curve))
    return {
        "ok": True,
        "sample_size": len(samples),
        "sample_days": days,
        "breakpoints": len(curve),
        "curve": curve,
    }


async def _scheduled_output_calibration():
    """Nightly job: rebuild the isotonic calibration curve."""
    try:
        result = await _compute_output_calibration_curve(days=45)
        log.info("[cron] output_calibration: %s", result.get("reason") or f"ok n={result.get('sample_size')}")
    except Exception as e:
        log.exception("[cron] output_calibration failed: %s", e)


@app.post("/api/admin/calibrate-output")
async def admin_calibrate_output(
    days: int = Query(45, ge=7, le=120),
    x_cron_secret: Optional[str] = Header(None),
):
    """Manually rebuild the output-calibration curve. Reports the fit."""
    _check_admin(x_cron_secret)
    return await _compute_output_calibration_curve(days=days)


@app.get("/api/admin/calibrate-output/status")
async def admin_calibrate_output_status(x_cron_secret: Optional[str] = Header(None)):
    """Inspect the persisted curve + how old it is."""
    _check_admin(x_cron_secret)
    async with get_session() as session:
        row = (await session.execute(
            select(WinCalibrationCurveRow).order_by(WinCalibrationCurveRow.id.desc())
        )).scalars().first()
    if not row:
        return {"exists": False}
    return {
        "exists": True,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "sample_days": row.sample_days,
        "sample_size": row.sample_size,
        "curve": json.loads(row.curve_json or "[]"),
    }


async def _enrich_date(race_date: str, client, model, force: bool = False, place_model: PlaceModel | None = None, exotic_model: ExoticModel | None = None) -> list[dict]:
    """Enrich all meetings for a single date. Returns summary list."""
    venue_cal = await _load_venue_calibration()
    output_cal = await _load_output_calibration_curve()
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
        # Blocklisted venues (bush / picnic tracks corporate books don't
        # cover) — skip enrichment entirely so they never generate picks,
        # bets, or hot-streak history. See _SKIP_ENRICHMENT_VENUES.
        if _should_skip_venue(venue_code):
            log.info("[enrich] skipping blocked venue: %s/%s", venue_code, race_date)
            continue
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
                predictions, _ = await enrich_and_predict_race(
                    race, model,
                    venue_calibration=venue_cal,
                    place_model=place_model,
                    exotic_model=exotic_model,
                    output_calibration_curve=output_cal,
                )
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
    request: Request,
    date: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """Force re-enrich races for a given date (defaults to today). Fire-and-forget."""
    _enforce_caller_rate(request, "reenrich")
    _check_admin(x_cron_secret)
    target = date or _today_aest().isoformat()

    # Caller fingerprint — helps identify a runaway external loop. UA, Referer
    # and X-Forwarded-For are usually enough to pin down which Railway service
    # (or external script) is hitting us.
    h = request.headers
    caller_fp = (
        f"ua={h.get('user-agent','?')[:80]!r} "
        f"ref={h.get('referer','?')!r} "
        f"xff={h.get('x-forwarded-for','?')!r} "
        f"client={request.client.host if request.client else '?'}"
    )

    # Debounce — per (endpoint, date). Defends against external schedulers
    # (Railway dashboard cron, stuck curl loops) firing this every few seconds
    # and hammering Racing Australia into 403/WAF bans.
    skip, age = _should_debounce("reenrich", target)
    if skip:
        log.info("[reenrich] DEBOUNCED %s — last call %.0fs ago — %s", target, age, caller_fp)
        return {
            "status": "debounced",
            "date": target,
            "seconds_since_last_call": round(age, 1),
            "cooldown_seconds": _ADMIN_DEBOUNCE_SECONDS,
        }

    log.info("[reenrich] ACCEPTED %s — %s", target, caller_fp)

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
    request: Request,
    date: Optional[str] = None,
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Re-run cancellation check for a date, restoring any races that were
    falsely marked cancelled when Racing Australia was blocked. Fast — no re-enrichment.
    """
    _enforce_caller_rate(request, "restore-cancelled")
    _check_admin(x_cron_secret)
    target = date or _today_aest().isoformat()
    skip, age = _should_debounce("restore-cancelled", target)
    if skip:
        log.info("[restore-cancelled] DEBOUNCED %s — last call %.0fs ago", target, age)
        return {
            "status": "debounced",
            "date": target,
            "seconds_since_last_call": round(age, 1),
            "cooldown_seconds": _ADMIN_DEBOUNCE_SECONDS,
        }
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


@app.get("/api/admin/scratched-trifecta-edge")
async def scratched_trifecta_edge(
    days: int = Query(default=30, ge=7, le=180),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Measures whether the model's POST-SCRATCHING top-3 trifecta box hits
    at a higher rate than the model's ORIGINAL (pre-scratching) top-3.

    For each settled race in the last `days` days where 1+ of the original
    top-3 picks (from the history snapshot) was later cancelled in mutable,
    we compute:
      - Original (history snapshot) trifecta box hit rate
      - Post-scratching (mutable at race time) trifecta box hit rate

    A meaningful lift on the second number (vs the first AND vs the unfiltered
    baseline) would confirm the pattern the user spotted on
    2026-06-13_newcastle_R8: BELLEVUE/ALL MACHIAVELLIAN/OAKFIELD NEPTUNE was
    the original trifecta, BELLEVUE + ALL MACHIAVELLIAN got scratched, and
    the new top-3 (HIDDEN STAR/OAKFIELD NEPTUNE/COSY CORNERS) box-hit.

    Trifecta box definition matches the edge page:
        leg 1 = model_rank=1 (the win pick)
        legs 2,3 = top 2 place_model_rank picks excluding the win pick
    """
    _check_admin(x_cron_secret)
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    async with get_session() as session:
        # All settled races in the window
        hr_rows = (await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id >= cutoff)
            .where(HistoricalResultRow.position.isnot(None))
        )).scalars().all()
        # History snapshot — top-3 picks per race (pre-cancellation state)
        hist_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.race_id >= cutoff)
            .where(RunnerPredictionHistoryRow.source == "live")
            .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
        )).scalars().all()
        # Mutable — what the model thinks NOW (after any re-enrichments/cancellations)
        mut_rows = (await session.execute(
            select(RunnerPredictionRow)
            .where(RunnerPredictionRow.race_id >= cutoff)
        )).scalars().all()

    # Group results by race_id → {position: horse_name (normalised)}
    actual_top3: dict[str, set] = {}
    for r in hr_rows:
        if r.position in (1, 2, 3):
            actual_top3.setdefault(r.race_id, set()).add(_normalize_horse(r.horse_name))

    def _build_top3(rows: list, race_id: str, include_cancelled: bool) -> tuple[set, int]:
        """Return (top-3 normalised horse names, count cancelled in those 3)."""
        race_runners = [r for r in rows if r.race_id == race_id]
        if not race_runners:
            return set(), 0
        # Win pick = model_rank=1
        win_pick = next((r for r in race_runners if r.model_rank == 1), None)
        if not win_pick:
            return set(), 0
        # Place legs = top 2 place_model_rank not equal to win pick
        place_candidates = sorted(
            [r for r in race_runners
             if r.place_model_rank is not None and r.horse_name != win_pick.horse_name],
            key=lambda r: r.place_model_rank
        )[:2]
        top3 = [win_pick] + place_candidates
        if len(top3) < 3:
            return set(), 0
        cancelled_n = sum(1 for r in top3 if r.cancelled) if not include_cancelled else 0
        names = {_normalize_horse(r.horse_name) for r in top3}
        return names, cancelled_n

    # Walk every settled race
    hist_by_race_top3: dict[str, set] = {}
    races_with_scratched_orig = []
    for rid in actual_top3.keys():
        hist_top3, _ = _build_top3(hist_rows, rid, include_cancelled=True)
        if len(hist_top3) < 3:
            continue
        hist_by_race_top3[rid] = hist_top3
        # Count how many of those 3 are cancelled in MUTABLE (the late-scratching signal)
        scratched_n = 0
        for nm in hist_top3:
            mut = next(
                (m for m in mut_rows
                 if m.race_id == rid and _normalize_horse(m.horse_name) == nm),
                None,
            )
            if mut and mut.cancelled:
                scratched_n += 1
        if scratched_n > 0:
            races_with_scratched_orig.append((rid, scratched_n))

    # Baseline: trifecta box hit rate on ALL settled races (history snapshot)
    base_total = base_hits = 0
    for rid, hist_t3 in hist_by_race_top3.items():
        actual_t3 = actual_top3.get(rid) or set()
        if len(actual_t3) < 3:
            continue
        base_total += 1
        if hist_t3 == actual_t3:
            base_hits += 1

    # Affected races (1+ of original top-3 was scratched): compare history vs mutable top-3
    orig_hits = mut_hits = affected_total = 0
    by_scratch_count: dict[int, dict] = {}  # {n_scratched: {races, orig_hits, mut_hits}}
    for rid, scratched_n in races_with_scratched_orig:
        actual_t3 = actual_top3.get(rid) or set()
        if len(actual_t3) < 3:
            continue
        orig_t3 = hist_by_race_top3.get(rid) or set()
        mut_t3, _ = _build_top3(mut_rows, rid, include_cancelled=True)
        if len(orig_t3) < 3 or len(mut_t3) < 3:
            continue
        affected_total += 1
        orig_hit = orig_t3 == actual_t3
        mut_hit = mut_t3 == actual_t3
        if orig_hit:
            orig_hits += 1
        if mut_hit:
            mut_hits += 1
        b = by_scratch_count.setdefault(scratched_n, {"races": 0, "orig_hits": 0, "mut_hits": 0})
        b["races"] += 1
        if orig_hit:
            b["orig_hits"] += 1
        if mut_hit:
            b["mut_hits"] += 1

    return {
        "window_days": days,
        "cutoff_from": cutoff,
        "baseline": {
            "description": "All settled races — trifecta box hit rate using the original history snapshot's top-3",
            "races": base_total,
            "hits": base_hits,
            "hit_pct": round(base_hits / base_total * 100, 1) if base_total else None,
        },
        "affected_races": {
            "description": "Races where 1+ of the original (history) top-3 was later cancelled in mutable",
            "races": affected_total,
            "original_top3_hit_pct": round(orig_hits / affected_total * 100, 1) if affected_total else None,
            "post_scratching_top3_hit_pct": round(mut_hits / affected_total * 100, 1) if affected_total else None,
            "lift_pp": round((mut_hits - orig_hits) / affected_total * 100, 1) if affected_total else None,
        },
        "by_scratch_count": [
            {
                "scratched_from_original_top3": n,
                "races": b["races"],
                "original_hit_pct": round(b["orig_hits"] / b["races"] * 100, 1) if b["races"] else None,
                "post_scratching_hit_pct": round(b["mut_hits"] / b["races"] * 100, 1) if b["races"] else None,
            }
            for n, b in sorted(by_scratch_count.items())
        ],
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


@app.get("/api/admin/win-place-ensemble")
async def win_place_ensemble(
    days: int = Query(default=14, ge=7, le=60),
    x_cron_secret: Optional[str] = Header(None),
):
    """
    Tier-3 ensemble diagnostic: does filtering the win model's top-1 pick by
    'place model also ranks this horse top-3' improve win rate?

    For each settled race in the last `days` days where we have both win and
    place rankings, bucket the win model's top-1 pick by the place model's
    rank-of-that-pick and report win rate per bucket. If the bucket-1 (place
    model agrees, rank 1-3) win rate is materially higher than the unfiltered
    baseline, the ensemble filter is a clear premium-tier qualifier.
    """
    _check_admin(x_cron_secret)
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    async with get_session() as session:
        # Top-1 win picks per race, with their place_model_rank
        pred_rows = (await session.execute(
            select(RunnerPredictionHistoryRow)
            .where(RunnerPredictionHistoryRow.model_rank == 1)
            .where(RunnerPredictionHistoryRow.place_model_rank.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
            .where(RunnerPredictionHistoryRow.source == "live")
            .where(RunnerPredictionHistoryRow.race_id >= cutoff)
        )).scalars().all()
        # Dedup per race — pick the most recently enriched pre-race snapshot
        pred_rows.sort(key=lambda r: r.enriched_at, reverse=True)
        seen = set()
        top_picks: list = []
        for p in pred_rows:
            if p.race_id in seen:
                continue
            seen.add(p.race_id)
            top_picks.append(p)
        # Results lookup
        hr_rows = (await session.execute(
            select(HistoricalResultRow)
            .where(HistoricalResultRow.race_id >= cutoff)
        )).scalars().all()
        result_by_key = {(r.race_id, _normalize_horse(r.horse_name)): r for r in hr_rows}

    # Bucket by place model's rank-of-the-winpick
    buckets: dict[int, dict] = {}
    for p in top_picks:
        actual = result_by_key.get((p.race_id, _normalize_horse(p.horse_name)))
        if actual is None:
            continue
        rank = int(p.place_model_rank)
        b = buckets.setdefault(rank, {"picks": 0, "wins": 0, "places": 0})
        b["picks"] += 1
        if actual.position == 1:
            b["wins"] += 1
        if actual.position and actual.position <= 3:
            b["places"] += 1

    # Build report
    out_buckets = []
    total_picks = total_wins = total_places = 0
    for rank in sorted(buckets.keys()):
        b = buckets[rank]
        total_picks += b["picks"]
        total_wins += b["wins"]
        total_places += b["places"]
        out_buckets.append({
            "place_model_rank": rank,
            "picks": b["picks"],
            "wins": b["wins"],
            "win_pct": round(b["wins"] / b["picks"] * 100, 1) if b["picks"] else None,
            "places": b["places"],
            "place_pct": round(b["places"] / b["picks"] * 100, 1) if b["picks"] else None,
        })

    # Aggregate consensus buckets (rank 1-3 = place model also likes it)
    consensus_picks = sum(b["picks"] for k, b in buckets.items() if k <= 3)
    consensus_wins = sum(b["wins"] for k, b in buckets.items() if k <= 3)
    consensus_places = sum(b["places"] for k, b in buckets.items() if k <= 3)
    dissent_picks = sum(b["picks"] for k, b in buckets.items() if k > 3)
    dissent_wins = sum(b["wins"] for k, b in buckets.items() if k > 3)
    dissent_places = sum(b["places"] for k, b in buckets.items() if k > 3)

    return {
        "holdout_days": days,
        "cutoff_from": cutoff,
        "total_picks": total_picks,
        "baseline_win_pct": round(total_wins / total_picks * 100, 1) if total_picks else None,
        "baseline_place_pct": round(total_places / total_picks * 100, 1) if total_picks else None,
        "consensus_filter": {
            "definition": "win model's top-1 AND place model's rank <= 3",
            "picks": consensus_picks,
            "wins": consensus_wins,
            "places": consensus_places,
            "win_pct": round(consensus_wins / consensus_picks * 100, 1) if consensus_picks else None,
            "place_pct": round(consensus_places / consensus_picks * 100, 1) if consensus_picks else None,
            "coverage": round(consensus_picks / total_picks * 100, 1) if total_picks else None,
        },
        "dissent_filter": {
            "definition": "win model's top-1 BUT place model's rank > 3",
            "picks": dissent_picks,
            "wins": dissent_wins,
            "win_pct": round(dissent_wins / dissent_picks * 100, 1) if dissent_picks else None,
        },
        "by_place_rank": out_buckets,
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
