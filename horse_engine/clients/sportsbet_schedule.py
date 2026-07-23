"""Sportsbet racing-schedule allowlist.

Each day we fetch the meetings Sportsbet actually books and use that as an
allowlist BEFORE enriching from Racing Australia — so we only pull/enrich AU
thoroughbred meetings punters can bet on, and never have to hand-maintain a
blocklist of obscure country tracks.

FAIL-OPEN by design: on any error, empty response, or a Sportsbet outage the
fetch returns None and callers must then keep ALL Racing Australia meetings
(never drop everything because one upstream hiccuped).

Endpoint (public, no auth):
  GET /apigw/sportsbook-racing/Sportsbook/Racing/AllRacing/{YYYY-MM-DD}
AU thoroughbred = section raceType=="horse" AND meeting regionName=="Australia"
AND isInternational is false. The region filter is essential — Sportsbet lists
international tracks that collide by name (e.g. a USA "Canterbury Park").
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

_URL = "https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/AllRacing/{date}"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_TTL_SECONDS = 2 * 3600  # re-fetch at most every 2h; enrich crons run a few times/day
_cache: dict[str, tuple[datetime, frozenset[str]]] = {}


def _norm(s: str) -> str:
    """Collapse a track name to letters+digits for fuzzy matching:
    'Sandown Hillside' → 'sandownhillside', 'eagle-farm' → 'eaglefarm'."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


async def get_sportsbet_au_meetings(date: str) -> frozenset[str] | None:
    """Normalised AU thoroughbred track names Sportsbet books on `date`.

    Returns None on any failure OR an empty result — callers treat None as
    "don't filter" (fail-open). Cached for _TTL_SECONDS."""
    cached = _cache.get(date)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < _TTL_SECONDS:
        return cached[1]
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=15.0,
        ) as client:
            resp = await client.get(_URL.format(date=date))
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("[sportsbet-schedule] fetch failed for %s: %s — allowlist disabled", date, e)
        return None

    names: set[str] = set()
    try:
        for dt in data.get("dates", []):
            for sec in dt.get("sections", []):
                if sec.get("raceType") != "horse":
                    continue
                for m in sec.get("meetings", []):
                    if m.get("regionName") == "Australia" and not m.get("isInternational"):
                        n = _norm(m.get("name"))
                        if n:
                            names.add(n)
    except Exception as e:
        log.warning("[sportsbet-schedule] parse failed for %s: %s", date, e)
        return None

    if not names:
        log.warning("[sportsbet-schedule] no AU thoroughbred meetings parsed for %s — allowlist disabled", date)
        return None

    frozen = frozenset(names)
    _cache[date] = (datetime.utcnow(), frozen)
    log.info("[sportsbet-schedule] %s: %d AU thoroughbred meetings on Sportsbet", date, len(frozen))
    return frozen


def venue_on_sportsbet(venue: str, allowlist: frozenset[str] | None) -> bool:
    """Does `venue` (an RA venue name or hyphenated code) appear in the Sportsbet
    allowlist? Fuzzy: exact normalised match, or one is a substring of the other
    ('Sandown Hillside' ↔ 'sandown'). When allowlist is None/empty, returns True
    (fail-open — no allowlist means don't filter)."""
    if not allowlist:
        return True
    v = _norm(venue)
    if not v:
        return True
    if v in allowlist:
        return True
    for a in allowlist:
        if a and (a in v or v in a):
            return True
    return False
