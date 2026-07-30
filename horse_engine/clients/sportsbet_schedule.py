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
_TTL_SECONDS = 900  # re-fetch at most every 15 min (results update through the day)
_raw_cache: dict[str, tuple[datetime, dict]] = {}   # date -> (ts, raw AllRacing json)


# Historical venue aliases: RA-era venue codes whose bookmaker feed name
# differs (RA "Pioneer Park" = feed "Alice Springs"). New enrichments use the
# aliased name at the RA client, but HISTORICAL race_ids keep the old code —
# aliasing here lets result seeding/matching heal old dates too.
_NORM_ALIASES = {"pioneerpark": "alicesprings"}


def _norm(s: str) -> str:
    """Collapse a track name to letters+digits for fuzzy matching:
    'Sandown Hillside' → 'sandownhillside', 'eagle-farm' → 'eaglefarm'."""
    n = re.sub(r"[^a-z0-9]", "", (s or "").lower())
    return _NORM_ALIASES.get(n, n)


async def _fetch_allracing(date: str) -> dict | None:
    """Fetch (and 15-min cache) the raw Sportsbet AllRacing payload for `date`.
    One call feeds both the meeting allowlist and the results backup. None on
    failure.

    SPORTSBET_PROXY_URL (2026-07-27): Sportsbet started 403-ing every
    non-AU-residential source (~2026-07-20) — Railway, Hetzner, and generic
    residential exits all blocked; only Australian residential IPs pass.
    Webshare supports country targeting via the username: swap '-rotate' for
    '-AU-rotate' in the residential proxy URL and set it as
    SPORTSBET_PROXY_URL on Railway. Unset → direct (fine if SB ever unblocks
    datacenter IPs; fail-open behaviour unchanged either way)."""
    cached = _raw_cache.get(date)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < _TTL_SECONDS:
        return cached[1]
    import os
    _proxy = os.getenv("SPORTSBET_PROXY_URL") or None
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=25.0 if _proxy else 15.0,
            proxy=_proxy,
        ) as client:
            resp = await client.get(_URL.format(date=date))
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("[sportsbet-schedule] fetch failed for %s: %s", date, e)
        return None
    if not isinstance(data, dict):
        return None
    _raw_cache[date] = (datetime.utcnow(), data)
    return data


def _iter_au_horse_meetings(data: dict):
    """Yield AU thoroughbred meeting dicts from an AllRacing payload."""
    for dt in data.get("dates", []):
        for sec in dt.get("sections", []):
            if sec.get("raceType") != "horse":
                continue
            for m in sec.get("meetings", []):
                if m.get("regionName") == "Australia" and not m.get("isInternational"):
                    yield m


async def get_sportsbet_au_meetings(date: str) -> frozenset[str] | None:
    """Normalised AU thoroughbred track names Sportsbet books on `date`.

    Returns None on any failure OR an empty result — callers treat None as
    "don't filter" (fail-open)."""
    data = await _fetch_allracing(date)
    if data is None:
        return None
    names: set[str] = set()
    try:
        for m in _iter_au_horse_meetings(data):
            n = _norm(m.get("name"))
            if n:
                names.add(n)
    except Exception as e:
        log.warning("[sportsbet-schedule] allowlist parse failed for %s: %s", date, e)
        return None
    if not names:
        log.warning("[sportsbet-schedule] no AU thoroughbred meetings parsed for %s — allowlist disabled", date)
        return None
    frozen = frozenset(names)
    log.info("[sportsbet-schedule] %s: %d AU thoroughbred meetings on Sportsbet", date, len(frozen))
    return frozen


async def get_sportsbet_results(date: str) -> dict[str, dict[int, list[int]]] | None:
    """Finishing order (top ~3) per AU thoroughbred track/race for `date`.

    Returns {track_lower: {race_num: [tab_num_1st, tab_num_2nd, tab_num_3rd]}}
    for resulted races (statusCode 'R'), parsed from each event's `result`
    field ("8,1,9" = 1st #8, 2nd #1, 3rd #9). Runner NUMBERS — callers match to
    their own field by tab_number. Backup to OddsPro (no RA, not WAF-blocked).
    None on failure."""
    data = await _fetch_allracing(date)
    if data is None:
        return None
    out: dict[str, dict[int, list[int]]] = {}
    try:
        for m in _iter_au_horse_meetings(data):
            track = _norm(m.get("name"))
            if not track:
                continue
            for e in m.get("events", []):
                if e.get("statusCode") != "R":
                    continue
                rnum = e.get("raceNumber")
                raw = (e.get("result") or "").strip()
                if not rnum or not raw:
                    continue
                order: list[int] = []
                for part in raw.split(","):
                    part = part.strip()
                    if part.isdigit():
                        order.append(int(part))
                if order:
                    out.setdefault(track, {})[rnum] = order
    except Exception as e:
        log.warning("[sportsbet-schedule] results parse failed for %s: %s", date, e)
        return None
    return out or None


async def get_sportsbet_race_times(date: str) -> dict[str, dict[int, str]] | None:
    """Race start times per AU thoroughbred track/race for `date`, from Sportsbet.

    Returns {track_lower: {race_num: start_iso_utc}}. Sportsbet AllRacing events
    carry `startTime` (unix seconds); we surface it as an ISO-8601 UTC string so
    the snapshot / edge scheduling never needs RA's Calendar/Acceptances. Reuses
    the cached AllRacing payload. None on failure."""
    data = await _fetch_allracing(date)
    if data is None:
        return None
    out: dict[str, dict[int, str]] = {}
    try:
        for m in _iter_au_horse_meetings(data):
            track = _norm(m.get("name"))
            if not track:
                continue
            for e in m.get("events", []):
                rnum = e.get("raceNumber")
                st = e.get("startTime")
                if not rnum or not st:
                    continue
                try:
                    iso = datetime.utcfromtimestamp(int(st)).isoformat() + "Z"
                except (ValueError, OSError, OverflowError):
                    continue
                out.setdefault(track, {})[rnum] = iso
    except Exception as e:
        log.warning("[sportsbet-schedule] race-times parse failed for %s: %s", date, e)
        return None
    return out or None


_COND_TTL_SECONDS = 1800  # racecard trackStatus re-fetched at most every 30 min
_cond_cache: dict[str, tuple[datetime, dict]] = {}   # date -> (ts, {norm_track: "Soft 5"})

_RACECARD_URL = "https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/Events/{event_id}/Racecard"


def _clean_track_status(raw: str) -> str | None:
    """'Soft (5)' → 'Soft 5', 'Good' → 'Good'. None for junk."""
    m = re.match(r"\s*(Firm|Good|Soft|Heavy|Synthetic)\s*\(?(\d*)\)?", raw or "", re.I)
    if not m:
        return None
    return f"{m.group(1).title()} {m.group(2)}".strip()


async def get_sportsbet_track_conditions(date: str) -> dict[str, str] | None:
    """Live going per AU meeting from Sportsbet racecards: {norm_track: 'Soft 5'}.

    RA's program pages carry no live going (our parser hardcodes Good 4), so
    Sportsbet's per-event `trackStatus` is the race-day truth. ONE racecard
    request per meeting (trackStatus is meeting-level), sequential, results
    cached 30 min — never a hot loop (see HARD RULE: no API hammering).
    Fail-open: None on outage; per-meeting failures just skip that meeting.
    """
    cached = _cond_cache.get(date)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < _COND_TTL_SECONDS:
        return cached[1]
    data = await _fetch_allracing(date)
    if data is None:
        return None
    import os
    _proxy = os.getenv("SPORTSBET_PROXY_URL") or None
    out: dict[str, str] = {}
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=25.0 if _proxy else 15.0,
            proxy=_proxy,
        ) as client:
            for m in _iter_au_horse_meetings(data):
                events = m.get("events") or []
                if not events:
                    continue
                # prefer a not-yet-resulted race so the status is current
                ev = next((e for e in events if not e.get("result")), events[-1])
                try:
                    resp = await client.get(_RACECARD_URL.format(event_id=ev.get("id")))
                    resp.raise_for_status()
                    cond = _clean_track_status((resp.json() or {}).get("trackStatus") or "")
                except Exception as e:  # one bad card never kills the sweep
                    log.info("[sportsbet-conditions] %s racecard failed: %s", m.get("name"), e)
                    continue
                if cond:
                    out[_norm(m.get("name") or "")] = cond
    except Exception as e:
        log.warning("[sportsbet-conditions] sweep failed for %s: %s", date, e)
        return None
    log.info("[sportsbet-conditions] %s: %d meetings -> %s", date, len(out), out)
    if not out:
        return None
    _cond_cache[date] = (datetime.utcnow(), out)
    return out


_PLACE_TTL_SECONDS = 600  # racecard place prices re-fetched at most every 10 min per race
_place_cache: dict[tuple, tuple[datetime, dict]] = {}   # (date, norm_track, race_no) -> (ts, {norm_horse: place_price})


async def get_sportsbet_place_prices(date: str, track: str, race_number: int) -> dict[str, float] | None:
    """Fixed PLACE prices per runner from the Sportsbet racecard.

    {norm_horse: place_price}. Pre-race place odds are recorded into
    OddsSnapshotRow (OddsPro is fixed-win only, so place_odds was always
    NULL). One racecard request per race, 10-min cache, fail-open — callers
    are throttled to the pre-jump window so this stays within the no-hammer
    rule.
    """
    key = (date, _norm(track), int(race_number))
    cached = _place_cache.get(key)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < _PLACE_TTL_SECONDS:
        return cached[1]
    data = await _fetch_allracing(date)
    if data is None:
        return None
    ev_id = None
    for m in _iter_au_horse_meetings(data):
        if _norm(m.get("name") or "") != _norm(track):
            continue
        for e in m.get("events") or []:
            if e.get("raceNumber") == int(race_number):
                ev_id = e.get("id")
                break
    if not ev_id:
        return None
    import os
    _proxy = os.getenv("SPORTSBET_PROXY_URL") or None
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=25.0 if _proxy else 15.0,
            proxy=_proxy,
        ) as client:
            resp = await client.get(_RACECARD_URL.format(event_id=ev_id))
            resp.raise_for_status()
            card = resp.json() or {}
    except Exception as e:
        log.info("[sportsbet-place] racecard %s R%s failed: %s", track, race_number, e)
        return None
    out: dict[str, float] = {}
    for mk in card.get("markets") or []:
        if "win or place" not in (mk.get("name") or "").lower():
            continue
        for sel in mk.get("selections") or []:
            best = None
            for p in sel.get("prices") or []:
                pp = p.get("placePrice")
                if pp and pp > 1.0:
                    best = float(pp)   # last entry wins ('L' = current fixed)
            if best and sel.get("name"):
                out[_norm(sel["name"])] = best
        break
    if not out:
        return None
    _place_cache[key] = (datetime.utcnow(), out)
    return out


async def get_sportsbet_exotic_dividends(date: str, track: str, race_number: int) -> dict | None:
    """Settled exotic tote dividends from the Sportsbet racecard:
    {quinella, exacta, trifecta, first_four} (floats, only keys present).

    SB's post-race racecard carries `exoticResults` — the one AU source we
    already have that publishes quinella dividends for country/provincial/NT
    meetings (RA free results and TAB do not). One racecard request per race.
    None if the race isn't found or no exotics resulted yet.
    """
    data = await _fetch_allracing(date)
    if data is None:
        return None
    ev_id = None
    for m in _iter_au_horse_meetings(data):
        if _norm(m.get("name") or "") != _norm(track):
            continue
        for e in m.get("events") or []:
            if e.get("raceNumber") == int(race_number):
                ev_id = e.get("id")
                break
    if not ev_id:
        return None
    import os
    _proxy = os.getenv("SPORTSBET_PROXY_URL") or None
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=25.0 if _proxy else 15.0,
            proxy=_proxy,
        ) as client:
            resp = await client.get(_RACECARD_URL.format(event_id=ev_id))
            resp.raise_for_status()
            card = resp.json() or {}
    except Exception as e:
        log.info("[sportsbet-exotics] %s R%s failed: %s", track, race_number, e)
        return None
    _map = {"quinella": "quinella", "exacta": "exacta", "trifecta": "trifecta",
            "first 4": "first_four", "first four": "first_four"}
    out: dict[str, float] = {}
    for r in card.get("exoticResults") or []:
        key = _map.get((r.get("name") or "").strip().lower())
        div = r.get("dividend")
        if key and div is not None:
            try:
                out[key] = float(str(div).replace("$", "").replace(",", ""))
            except ValueError:
                pass
    return out or None


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
