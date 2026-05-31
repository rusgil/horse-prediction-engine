"""
Composite racing client: Racing Australia (race cards) + OddsPro (odds & results).

Racing Australia provides: barriers, weights, jockeys, trainers, form strings.
OddsPro provides: multi-book best odds, opening price (steam/drift), finishing positions.
"""
from __future__ import annotations

import logging
from datetime import date

from horse_engine.clients.racing_australia import RacingAustraliaClient
from horse_engine.clients.oddspro import OddsProClient
from horse_engine.models.race import Race

log = logging.getLogger(__name__)


class CompositeClient:
    def __init__(self) -> None:
        self._ra = RacingAustraliaClient()
        self._odds = OddsProClient()

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
        race_date = meeting.get("date") or meeting.get("meetingDateLocal") or date.today().isoformat()

        # Find OddsPro track that matches this RA venue
        tracks = await self._odds.get_tracks(race_date)
        op_track = self._odds.find_matching_track(venue, tracks)

        odds_map: dict[tuple, dict] = {}
        if op_track:
            odds_map = await self._odds.get_track_odds(op_track)
        else:
            log.debug("OddsPro: no track match for '%s' on %s", venue, race_date)

        # Enrich RA selections with OddsPro odds data
        for sel in raw.get("selections", []):
            name = (sel.get("competitor") or {}).get("name", "").lower()
            op = odds_map.get((race_number, name))
            if op:
                sel["topToteWin"] = op.get("currentBestOdds")
                sel["_odds_opening"] = op.get("firstPrice")
                sel["_finishing_position"] = op.get("finishingPosition")

        # Build runners list in TAB-compatible format for result seeding
        runners_out = []
        for sel in raw.get("selections", []):
            if (sel.get("status") or "").upper() == "SCRATCHED":
                continue
            name = (sel.get("competitor") or {}).get("name", "")
            op = odds_map.get((race_number, name.lower()), {})
            best = op.get("currentBestOdds")
            runners_out.append({
                "runnerName": name,
                "finishingPosition": op.get("finishingPosition"),
                "margin": 0,
                "scratched": False,
                "prices": [{"priceType": "Win", "winPrice": best}] if best else [],
            })
        raw["runners"] = runners_out

        return raw

    # ── Parsing ───────────────────────────────────────────────────────────────

    async def parse_race(self, raw_event: dict, race_date: str, venue: str, state: str) -> Race:
        race = await self._ra.parse_race(raw_event, race_date, venue, state)

        # Patch odds_opening onto runners from OddsPro data embedded in selections
        sel_map = {
            (sel.get("competitor") or {}).get("name", "").lower(): sel
            for sel in (raw_event.get("selections") or [])
        }
        for runner in race.runners:
            sel = sel_map.get(runner.horse_name.lower())
            if sel:
                runner.odds_opening = sel.get("_odds_opening")

        return race
