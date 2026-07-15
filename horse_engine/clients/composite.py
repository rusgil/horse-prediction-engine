"""
Composite racing client: Racing Australia (race cards + results) + OddsPro (odds).

Data stack: Racing Australia + OddsPro.

Previously included Betfair REST + WebSocket stream layers for live LTP odds
and steam/drift signals. Betfair access was WAF/account-blocked for a long
period, and a 2026-06-12 feature ablation showed every Betfair-derived
feature (steam_60, steam_30, drift_flag, odds_velocity, late_money,
odds_movement_norm) was either net-zero or net-harmful to the win model.
Removed entirely 2026-06-13 — RA + OddsPro fully cover the model's
training inputs.

The steam_60 / steam_30 / drift_flag / odds_velocity / late_money fields on
EnrichedRunner are still in the schema but always default to 0.0. Stored
weights stay 41-dim, no migration needed.
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


def _normalize(name: str) -> str:
    """Lowercase + strip country codes for name matching."""
    return re.sub(r'\s*\([^)]+\)', '', name).strip().lower()


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
        ra_key = meeting.get("id", "")
        race_date = meeting.get("date") or meeting.get("meetingDateLocal") or date.today().isoformat()

        tracks = await self._odds.get_tracks(race_date)
        op_track = self._odds.find_matching_track(venue, tracks)
        if not op_track:
            log.debug("OddsPro: no track match for '%s' on %s", venue, race_date)

        # Load BOTH the movers dict (for opening prices / steam signals, when
        # we bring those back) AND the full-market meeting dict. The movers
        # endpoint only returns runners whose price has moved recently, so
        # relying on it alone left firm-priced favourites (and every non-mover)
        # with sel.topToteWin=None. Market features then defaulted every
        # horse's market_implied_prob to 1/N, blinding the model to the market
        # signal. Root cause of the 2026-07-15/16 flat-distribution incident.
        # Full-market fetch is one API call for the whole country, cached
        # for 2 minutes inside OddsProClient — cheap.
        odds_map, meeting_odds, ra_results = await asyncio.gather(
            self._odds.get_track_odds(op_track) if op_track else _empty_dict(),
            self._odds.get_meeting_odds(race_date) if op_track else _empty_dict(),
            self._ra.get_results(ra_key) if ra_key else _empty_dict(),
        )
        full_market = meeting_odds.get((op_track or "").lower(), {}) if op_track else {}

        race_results: dict[str, dict] = (ra_results.get(race_number) or {}).get("runners", {})

        # ── Odds priority for each runner:
        #    OddsPro movers.currentBestOdds → OddsPro full-market → RA SP
        # Movers keeps whatever opening-price signal we later need; full-market
        # is the fallback that ensures firm favourites and stable prices land.
        for sel in raw.get("selections", []):
            name_raw = (sel.get("competitor") or {}).get("name", "")
            name = name_raw.lower()
            name_norm = _normalize(name_raw)
            op = odds_map.get((race_number, name), {})
            ra = race_results.get(name, {})

            if not sel.get("topToteWin"):
                if op and op.get("currentBestOdds"):
                    sel["topToteWin"] = op.get("currentBestOdds")
                    sel["_odds_opening"] = op.get("firstPrice")
                else:
                    fm_price = full_market.get((race_number, name)) or full_market.get((race_number, name_norm))
                    if fm_price and fm_price > 1.0:
                        sel["topToteWin"] = fm_price

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

        return race
