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


class OddsProClient:
    def __init__(self) -> None:
        # date → (ts, [track_name, ...])
        self._tracks_cache: dict[str, tuple[datetime, list[str]]] = {}
        # track → (ts, {(race_num, runner_name_lower): runner_dict})
        self._track_cache: dict[str, tuple[datetime, dict]] = {}
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

    def find_matching_track(self, venue: str, tracks: list[str]) -> Optional[str]:
        """
        Match an RA venue name to an OddsPro track name.
        e.g. 'Sandown Lakeside' → 'Sandown'
        """
        venue_lower = venue.lower()
        for t in tracks:
            if t.lower() == venue_lower:
                return t
        # OddsPro name is a prefix/substring of RA venue (most common mismatch)
        for t in tracks:
            if t.lower() in venue_lower:
                return t
        # RA venue is contained in OddsPro name
        for t in tracks:
            if venue_lower in t.lower():
                return t
        return None
