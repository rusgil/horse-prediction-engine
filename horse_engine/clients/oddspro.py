"""
OddsPro external API client.

Provides multi-bookmaker odds, steam/drift detection, and finishing positions
for Australian domestic thoroughbred racing. No authentication required.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx

log = logging.getLogger(__name__)

_BASE = "https://oddspro.com.au/api/external"
_MEETINGS_BASE = "https://oddspro.com.au/api/meetings"
_AU_STATES = {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"}


class OddsProClient:
    def __init__(self) -> None:
        # date → (ts, [track_name, ...])
        self._tracks_cache: dict[str, tuple[datetime, list[str]]] = {}
        # track → (ts, {(race_num, runner_name_lower): runner_dict})
        self._track_cache: dict[str, tuple[datetime, dict]] = {}
        # date → (ts, {track_lower: {(race_num, runner_lower): best_price}})
        self._meetings_cache: dict[str, tuple[datetime, dict]] = {}
        self._sem: asyncio.Semaphore | None = None

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(4)
        return self._sem

    async def _get(self, url: str) -> dict:
        async with self._get_sem():
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()

    async def get_tracks(self, race_date: str) -> list[str]:
        """Return AU domestic thoroughbred track names racing on race_date."""
        cached = self._tracks_cache.get(race_date)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 300:
            return cached[1]
        try:
            data = await self._get(f"{_BASE}/tracks?location=domestic&code=T&date={race_date}")
            tracks = data.get("tracks", [])
            self._tracks_cache[race_date] = (datetime.utcnow(), tracks)
            return tracks
        except Exception as e:
            log.debug("OddsPro tracks failed for %s: %s", race_date, e)
            return []

    async def get_track_odds(self, track: str) -> dict[tuple, dict]:
        """
        Return {(race_num, runner_name_lower): runner_dict} for all runners at track.
        Cached 90 seconds — one call covers all races at the venue.
        """
        cached = self._track_cache.get(track)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 90:
            return cached[1]
        try:
            data = await self._get(f"{_BASE}/movers/track/{track}")
            result: dict[tuple, dict] = {}
            for r in data.get("data", []):
                rn = r.get("raceNumber")
                name = (r.get("runnerName") or "").lower()
                if rn and name:
                    result[(rn, name)] = r
            self._track_cache[track] = (datetime.utcnow(), result)
            log.debug("OddsPro: %d runners loaded for %s", len(result), track)
            return result
        except Exception as e:
            log.debug("OddsPro track odds failed for %s: %s", track, e)
            return {}

    async def get_meeting_odds(self, date: str) -> dict[str, dict[tuple[int, str], float]]:
        """
        Fetch full fixed-win odds for all AU domestic thoroughbred meetings on date.
        Returns {track_lower: {(race_num, runner_name_lower): best_fixed_win_price}}
        One API call covers all venues — much better than movers/track per venue.
        Cached 2 minutes.
        """
        cached = self._meetings_cache.get(date)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 120:
            return cached[1]
        try:
            data = await self._get(f"{_MEETINGS_BASE}?date={date}")
            if not isinstance(data, list):
                data = []
        except Exception as e:
            log.debug("OddsPro meetings failed for %s: %s", date, e)
            return {}

        result: dict[str, dict[tuple[int, str], float]] = {}
        for meeting in data:
            if meeting.get("location") not in _AU_STATES or meeting.get("type") != "T":
                continue
            track_lower = (meeting.get("track") or "").lower()
            if not track_lower:
                continue
            track_odds: dict[tuple[int, str], float] = {}
            for race in meeting.get("races", []):
                race_num = race.get("number")
                if not race_num:
                    continue
                for runner in race.get("runners", []):
                    if (runner.get("status") or "").upper() == "SCRATCHED":
                        continue
                    name_lower = (runner.get("name") or "").lower().strip()
                    if not name_lower:
                        continue
                    markets = runner.get("bookmakerMarkets", [])
                    prices = [
                        bm["fixedWin"]["price"]
                        for bm in markets
                        if bm.get("fixedWin") and (bm["fixedWin"].get("price") or 0) > 1.0
                    ]
                    if prices:
                        track_odds[(race_num, name_lower)] = max(prices)
            result[track_lower] = track_odds
            log.debug("OddsPro meetings: %s %d runners", track_lower, len(track_odds))

        self._meetings_cache[date] = (datetime.utcnow(), result)
        log.info("OddsPro meetings: %d AU tracks on %s", len(result), date)
        return result

    async def get_results(self, date: str) -> dict[str, dict[int, dict]]:
        """Finished-race results per AU domestic thoroughbred track for `date`.

        Returns {track_lower: {race_num: {"finishers": [...], "ran": [...]}}}.
        - "finishers": [{"number","name","position","sp"}] — the placed runners
          from the OddsPro `results` field (finishing order by runner number,
          each slot a list for dead-heats: [[8],[1],[9],[4]] = 1st #8 … 4th #4).
          NOTE: OddsPro only publishes the top placings (~top 4), NOT the full
          field, so runners that finished 5th+ are absent here.
        - "ran": [{"number","name","sp"}] — every NON-scratched runner in the
          field. Callers use this to mark predicted runners that ran but did
          not place as losses (unplaced), so win/place denominators aren't
          silently inflated by dropping 5th+ finishers (2026-07-23 fix).

        NON-RA: unaffected by the Racing Australia WAF. Only status=='finished'
        races with a populated `results` array are returned.
        """
        try:
            data = await self._get(f"{_MEETINGS_BASE}?date={date}")
            if not isinstance(data, list):
                data = []
        except Exception as e:
            log.debug("OddsPro results failed for %s: %s", date, e)
            return {}

        out: dict[str, dict[int, dict]] = {}
        for meeting in data:
            if meeting.get("location") not in _AU_STATES or meeting.get("type") != "T":
                continue
            track_lower = (meeting.get("track") or "").lower().strip()
            if not track_lower:
                continue
            for race in meeting.get("races", []):
                # Key on `results` populated, not `status=="finished"`. OddsPro
                # frequently leaves status='scheduled' even after a race has run
                # and results are published (observed 2026-07-23 — Wyong R2/R3,
                # Warrnambool R1/R2 all had results but status='scheduled', so
                # nothing settled all morning). The `results` array is the
                # definitive signal a race has been called.
                order = race.get("results")
                race_num = race.get("number")
                if not order or not isinstance(order, list) or not race_num:
                    continue
                # runner number -> {name, sp}; and the ran (non-scratched) field
                info: dict[int, dict] = {}
                ran: list[dict] = []
                for runner in race.get("runners", []):
                    num = runner.get("number")
                    name = (runner.get("name") or "").strip()
                    if not num or not name:
                        continue
                    prices = [
                        bm["fixedWin"]["price"]
                        for bm in runner.get("bookmakerMarkets", [])
                        if bm.get("fixedWin") and (bm["fixedWin"].get("price") or 0) > 1.0
                    ]
                    sp = max(prices) if prices else None
                    info[num] = {"name": name, "sp": sp}
                    if (runner.get("status") or "").upper() != "SCRATCHED":
                        ran.append({"number": num, "name": name, "sp": sp})
                finishers: list[dict] = []
                for pos_idx, slot in enumerate(order, start=1):
                    nums = slot if isinstance(slot, list) else [slot]
                    for num in nums:
                        ri = info.get(num, {})
                        if not ri.get("name"):
                            continue
                        finishers.append({
                            "number": num,
                            "name": ri["name"],
                            "position": pos_idx,
                            "sp": ri.get("sp"),
                        })
                if finishers:
                    out.setdefault(track_lower, {})[race_num] = {"finishers": finishers, "ran": ran}
        return out

    async def get_scratchings(self, date: str) -> dict[str, dict[int, set[str]]]:
        """Scratched runners per AU domestic thoroughbred track/race for `date`.

        Returns {track_lower: {race_num: {runner_name_lower, ...}}}. Uses the
        runner `status` field (=='SCRATCHED'), which OddsPro populates LIVE and
        pre-race throughout the day — so scratch detection runs without touching
        Racing Australia. This is a DIRECT flag, unlike the RA path which has to
        infer scratchings from a runner's absence from the acceptance field.
        """
        try:
            data = await self._get(f"{_MEETINGS_BASE}?date={date}")
            if not isinstance(data, list):
                data = []
        except Exception as e:
            log.debug("OddsPro scratchings failed for %s: %s", date, e)
            return {}

        out: dict[str, dict[int, set[str]]] = {}
        for meeting in data:
            if meeting.get("location") not in _AU_STATES or meeting.get("type") != "T":
                continue
            track_lower = (meeting.get("track") or "").lower().strip()
            if not track_lower:
                continue
            for race in meeting.get("races", []):
                race_num = race.get("number")
                if not race_num:
                    continue
                scratched = {
                    (runner.get("name") or "").lower().strip()
                    for runner in race.get("runners", [])
                    if (runner.get("status") or "").upper() == "SCRATCHED"
                    and (runner.get("name") or "").strip()
                }
                if scratched:
                    out.setdefault(track_lower, {})[race_num] = scratched
        return out

    def find_matching_track(self, venue: str, tracks: list[str]) -> Optional[str]:
        """
        Match an RA venue name to an OddsPro track name.
        e.g. 'Sandown Lakeside' → 'Sandown', 'eagle-farm' → 'Eagle Farm'
        """
        # Normalize hyphens to spaces on both sides — our venue codes use
        # 'eagle-farm' while OddsPro shows 'Eagle Farm'. Without this normalize,
        # multi-word tracks silently failed and best_available_odds stayed 0
        # (Balaklava-diagnostic 2026-07-15: eagle-farm and warwick-farm both
        # unmatched even though OddsPro had them).
        def norm(s: str) -> str:
            return s.lower().replace('-', ' ').strip()

        # Historical venue aliases (see sportsbet_schedule._NORM_ALIASES):
        # old race_ids keep the RA-era code, the feed names the town.
        _ALIASES = {"pioneer park": "alice springs"}
        venue_norm = norm(venue)
        venue_norm = _ALIASES.get(venue_norm, venue_norm)
        for t in tracks:
            if norm(t) == venue_norm:
                return t
        # OddsPro name is a prefix/substring of RA venue (most common mismatch)
        for t in tracks:
            if norm(t) in venue_norm:
                return t
        # RA venue is contained in OddsPro name
        for t in tracks:
            if venue_norm in norm(t):
                return t
        return None
