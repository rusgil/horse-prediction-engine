"""
Composite racing client: Racing Australia (race cards + results) + OddsPro (odds).

Optional enrichment layers (all activated by Betfair credentials):
  BetfairClient       — REST catalogue for runner metadata (sire, dam, age, sex)
  BetfairStreamClient — live streaming odds; replaces REST polling and provides
                        steam_60 / steam_30 / late_money / drift_flag / odds_velocity

Betfair layers activate only when BETFAIR_APP_KEY / BETFAIR_USERNAME /
BETFAIR_PASSWORD are set in the environment.

Data stack: Racing Australia + OddsPro + Betfair (REST + stream).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date

from horse_engine.clients.racing_australia import RacingAustraliaClient
from horse_engine.clients.oddspro import OddsProClient
from horse_engine.models.race import Race

log = logging.getLogger(__name__)


async def _empty_dict() -> dict:
    return {}


async def _none() -> None:
    return None


def _normalize(name: str) -> str:
    """Lowercase + strip country codes for name matching."""
    return re.sub(r'\s*\([^)]+\)', '', name).strip().lower()


class CompositeClient:
    def __init__(self) -> None:
        self._ra = RacingAustraliaClient()
        self._odds = OddsProClient()
        self._bf = None       # BetfairClient (REST) — lazy init
        self._stream = None   # BetfairStreamClient — lazy init + started once

    def _get_betfair(self):
        if self._bf is None:
            try:
                from horse_engine.config import settings
                # Both credentials AND the kill switch must be true. Default is
                # disabled — see feedback_no_api_hammer.md. The stream client
                # was hitting identitysso.betfair.com.au every 30s with failed
                # auth before this guard.
                if (
                    settings.betfair_enabled
                    and settings.betfair_app_key
                    and settings.betfair_username
                    and settings.betfair_password
                ):
                    from horse_engine.clients.betfair import BetfairClient
                    self._bf = BetfairClient()
                    log.info("BetfairClient activated for odds + metadata enrichment")
            except Exception as e:
                log.debug("BetfairClient init skipped: %s", e)
        return self._bf

    async def _get_stream(self):
        """Return BetfairStreamClient, starting it once if enabled + credentials are present."""
        if self._stream is None:
            try:
                from horse_engine.config import settings
                if (
                    settings.betfair_enabled
                    and settings.betfair_app_key
                    and settings.betfair_username
                    and settings.betfair_password
                ):
                    from horse_engine.clients.betfair_stream import BetfairStreamClient
                    self._stream = BetfairStreamClient(
                        app_key=settings.betfair_app_key,
                        username=settings.betfair_username,
                        password=settings.betfair_password,
                    )
                    await self._stream.start()
                    log.info("BetfairStreamClient started — live AU odds active")
            except Exception as e:
                log.debug("BetfairStreamClient init skipped: %s", e)
        return self._stream

    # ── Discovery (delegated to RA) ───────────────────────────────────────────

    async def get_meetings(self, race_date: str | None = None) -> list[dict]:
        return await self._ra.get_meetings(race_date)

    async def get_meeting_by_slug(self, slug: str) -> dict | None:
        return await self._ra.get_meeting_by_slug(slug)

    async def get_meeting_races(self, slug: str) -> list[dict]:
        return await self._ra.get_meeting_races(slug)

    # ── Race card + odds merge ────────────────────────────────────────────────

    async def get_race(self, slug: str, race_number: int) -> dict | None:
        raw = await self._ra.get_race(slug, race_number)
        if not raw:
            return None

        meeting = raw.get("_meeting") or {}
        venue = meeting.get("venue", "")
        ra_key = meeting.get("id", "")
        race_date = meeting.get("date") or meeting.get("meetingDateLocal") or date.today().isoformat()

        # Kick off all parallel fetches — stream client is started lazily here
        bf = self._get_betfair()
        stream = await self._get_stream()
        tracks = await self._odds.get_tracks(race_date)
        op_track = self._odds.find_matching_track(venue, tracks)
        if not op_track:
            log.debug("OddsPro: no track match for '%s' on %s", venue, race_date)

        odds_map, ra_results, bf_race = await asyncio.gather(
            self._odds.get_track_odds(op_track) if op_track else _empty_dict(),
            self._ra.get_results(ra_key) if ra_key else _empty_dict(),
            bf.get_race(slug, race_number) if bf else _none(),
        )

        # Build Betfair REST selection lookup keyed by normalized horse name
        bf_by_name: dict[str, dict] = {}
        bf_market_id: str = ""
        if bf_race:
            bf_market_id = bf_race.get("id", "")
            for bf_sel in (bf_race.get("selections") or []):
                bf_name = _normalize((bf_sel.get("competitor") or {}).get("name", ""))
                if bf_name:
                    bf_by_name[bf_name] = bf_sel

        race_results: dict[str, dict] = (ra_results.get(race_number) or {}).get("runners", {})

        for sel in raw.get("selections", []):
            name_raw = (sel.get("competitor") or {}).get("name", "")
            name = name_raw.lower()
            norm = _normalize(name_raw)
            op = odds_map.get((race_number, name), {})
            ra = race_results.get(name, {})
            bf_sel = bf_by_name.get(norm) or bf_by_name.get(name)

            # ── Odds priority: stream LTP > OddsPro > Betfair REST ──────────
            stream_features: dict = {}
            if stream and bf_market_id:
                stream_features = stream.get_odds_features(bf_market_id, name_raw)
                if stream_features.get("current_ltp"):
                    sel["topToteWin"] = stream_features["current_ltp"]
                    # steam/drift signals stored for parse_race to pick up
                    sel["_stream"] = stream_features

            if not sel.get("topToteWin"):
                if op:
                    sel["topToteWin"] = op.get("currentBestOdds")
                    sel["_odds_opening"] = op.get("firstPrice")
                elif bf_sel and bf_sel.get("topToteWin"):
                    sel["topToteWin"] = bf_sel["topToteWin"]

            # Betfair metadata: sire, dam, age, sex, colour
            if bf_sel:
                comp = sel.setdefault("competitor", {})
                bf_comp = bf_sel.get("competitor") or {}
                for field in ("sire", "dam", "age", "sex", "colour"):
                    if bf_comp.get(field) and not comp.get(field):
                        comp[field] = bf_comp[field]

            if ra:
                sel["_finishing_position"] = ra.get("position")
                sel["_margin"] = ra.get("margin", 0)
                if ra.get("sp") and not sel.get("topToteWin"):
                    sel["topToteWin"] = ra["sp"]

        runners_out = []
        for sel in raw.get("selections", []):
            if (sel.get("status") or "").upper() == "SCRATCHED":
                continue
            name = (sel.get("competitor") or {}).get("name", "")
            op = odds_map.get((race_number, name.lower()), {})
            ra = race_results.get(name.lower(), {})
            best = op.get("currentBestOdds") or sel.get("topToteWin") or ra.get("sp")
            runners_out.append({
                "runnerName": name,
                "finishingPosition": ra.get("position"),
                "margin": ra.get("margin", 0),
                "scratched": False,
                "prices": [{"priceType": "Win", "winPrice": best}] if best else [],
            })
        raw["runners"] = runners_out

        return raw

    # ── Parsing ───────────────────────────────────────────────────────────────

    async def parse_race(self, raw_event: dict, race_date: str, venue: str, state: str) -> Race:
        race = await self._ra.parse_race(raw_event, race_date, venue, state)

        sel_map = {
            _normalize((sel.get("competitor") or {}).get("name", "")): sel
            for sel in (raw_event.get("selections") or [])
        }
        for runner in race.runners:
            sel = sel_map.get(_normalize(runner.horse_name))
            if not sel:
                continue
            if sel.get("_odds_opening"):
                runner.odds_opening = sel["_odds_opening"]
            # Apply live stream signals if available
            sf = sel.get("_stream") or {}
            if sf:
                if sf.get("current_ltp"):
                    runner.best_available_odds = sf["current_ltp"]
                    runner.tote_win_odds = sf["current_ltp"]
                runner.odds_movement = sf.get("steam_60", 0.0)
                runner.is_steamed = sf.get("steam_60", 0.0) > 0.5
                runner.is_drifted = sf.get("drift_flag", 0.0) > 0.0
                runner.steam_60 = sf.get("steam_60", 0.0)
                runner.steam_30 = sf.get("steam_30", 0.0)
                runner.late_money = sf.get("late_money", 0.0)
                runner.drift_flag = sf.get("drift_flag", 0.0)
                runner.odds_velocity = sf.get("odds_velocity", 0.0)

        return race
