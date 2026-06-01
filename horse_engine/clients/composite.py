"""
Composite racing client: Racing Australia (race cards + results) + OddsPro (odds).

Optional: BetfairClient enriches runner metadata (sire, dam, age, sex) and
provides exchange odds as a fallback when OddsPro has no price.
Betfair is activated only when BETFAIR_APP_KEY / BETFAIR_USERNAME / BETFAIR_PASSWORD
are set in the environment.
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
        self._bf = None  # BetfairClient — lazy init only if credentials set

    def _get_betfair(self):
        if self._bf is None:
            try:
                from horse_engine.config import settings
                if settings.betfair_app_key and settings.betfair_username and settings.betfair_password:
                    from horse_engine.clients.betfair import BetfairClient
                    self._bf = BetfairClient()
                    log.info("BetfairClient activated for odds + metadata enrichment")
            except Exception as e:
                log.debug("BetfairClient init skipped: %s", e)
        return self._bf

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

        # Find OddsPro track and start Betfair fetch in parallel
        bf = self._get_betfair()
        tracks = await self._odds.get_tracks(race_date)
        op_track = self._odds.find_matching_track(venue, tracks)
        if not op_track:
            log.debug("OddsPro: no track match for '%s' on %s", venue, race_date)

        odds_map, ra_results, bf_race = await asyncio.gather(
            self._odds.get_track_odds(op_track) if op_track else _empty_dict(),
            self._ra.get_results(ra_key) if ra_key else _empty_dict(),
            bf.get_race(slug, race_number) if bf else _none(),
        )

        # Build Betfair selection lookup keyed by normalized horse name
        bf_by_name: dict[str, dict] = {}
        if bf_race:
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

            # Odds: OddsPro first, Betfair exchange as fallback
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
            if sel:
                runner.odds_opening = sel.get("_odds_opening")

        return race
