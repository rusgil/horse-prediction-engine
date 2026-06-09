"""
Betfair Exchange data client for AU horse racing.

Fetches meeting and race-card data for today and tomorrow only.
Uses the Betfair Exchange REST API (non-cert username/password login).

Required environment variables:
  BETFAIR_APP_KEY  — developer app key from the Betfair developer portal
  BETFAIR_USERNAME — Betfair account username (email)
  BETFAIR_PASSWORD — Betfair account password
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from horse_engine.models.race import (
    JockeyStats, PedigreeProfile, Race, Runner, TrainerStats,
)
from horse_engine.pedigree.sire_profiles import SIRE_PROFILES

log = logging.getLogger(__name__)

_LOGIN_URL = "https://identitysso.betfair.com.au/api/login"
_API_BASE = "https://api.betfair.com.au/exchange/betting/rest/v1.0/"
_HORSE_RACING_TYPE_ID = "7"

_AU_STATES = {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"}

# Betfair event timezone → AU state
_TZ_STATE: dict[str, str] = {
    "Australia/Melbourne": "VIC",
    "Australia/Victoria": "VIC",
    "Australia/Sydney": "NSW",
    "Australia/ACT": "ACT",
    "Australia/Canberra": "ACT",
    "Australia/Brisbane": "QLD",
    "Australia/Queensland": "QLD",
    "Australia/Adelaide": "SA",
    "Australia/South": "SA",
    "Australia/Perth": "WA",
    "Australia/West": "WA",
    "Australia/Darwin": "NT",
    "Australia/North": "NT",
    "Australia/Hobart": "TAS",
    "Australia/Currie": "TAS",
    "Australia/Broken_Hill": "NSW",
}

# Betfair SEX_TYPE → single-char code used in our model
_SEX_MAP = {
    "Gelding": "G", "Mare": "M", "Filly": "F",
    "Colt": "C", "Horse": "H", "Rig": "G", "Stallion": "H",
}

# Month name suffix pattern in event names like "Werribee 29th May"
_DATE_SUFFIX_RE = re.compile(
    r'\s+\d+\w*\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
    r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
    r'Nov(?:ember)?|Dec(?:ember)?)$',
    re.IGNORECASE,
)
_RACE_NAME_RE = re.compile(r'[Rr](?:ace\s*)?\s*(\d+)(?:\D+?(\d{3,5})\s*m)?')


def _slugify(name: str) -> str:
    clean = re.sub(r'\s*\([^)]+\)', '', name).strip()
    return re.sub(r'[^a-z0-9]+', '-', clean.lower()).strip('-')


def _make_meeting_slug(venue_name: str, race_date: str) -> str:
    return f"{_slugify(venue_name)}-{race_date.replace('-', '')}"


def _extract_venue(event: dict) -> str:
    """Return clean venue name from a Betfair event object."""
    venue = (event.get("venue") or "").strip()
    if venue:
        return venue
    name = event.get("name", "")
    name = re.sub(r'\s*\([^)]+\)', '', name)   # strip "(Vic)" etc
    name = _DATE_SUFFIX_RE.sub('', name)
    return name.strip()


def _parse_market_name(market_name: str) -> tuple[int | None, int]:
    """'R1 1400m' → (1, 1400). Returns (None, 0) if unparseable."""
    m = _RACE_NAME_RE.match(market_name)
    if m:
        dist = int(m.group(2)) if m.group(2) else 0
        return int(m.group(1)), dist
    return None, 0


def _parse_weight(meta: dict) -> float:
    raw = meta.get("WEIGHT_VALUE") or ""
    try:
        kg = float(raw)
    except (ValueError, TypeError):
        return 0.0
    units = (meta.get("WEIGHT_UNITS") or "kg").lower()
    if units in ("lb", "lbs"):
        kg = kg / 2.2046
    return round(kg, 1)


def _build_selections(runners: list[dict], book: dict) -> list[dict]:
    """
    Merge Betfair runner catalogue metadata with live market book odds
    into the internal selection format expected by parse_race / _parse_runner.
    """
    odds_by_sel: dict[int, float | None] = {}
    scratched_ids: set[int] = set()
    for br in (book.get("runners") or []):
        sel_id = br.get("selectionId")
        if br.get("status") == "REMOVED":
            scratched_ids.add(sel_id)
            continue
        backs = (br.get("ex") or {}).get("availableToBack") or []
        odds_by_sel[sel_id] = float(backs[0]["price"]) if backs else None

    result = []
    for runner in runners:
        sel_id = runner.get("selectionId")
        if sel_id in scratched_ids:
            continue
        meta = runner.get("metadata") or {}
        tote_win = odds_by_sel.get(sel_id)

        result.append({
            "id": str(sel_id or ""),
            "competitorNumber": int(runner.get("sortPriority") or 0),
            "barrierNumber": int(meta.get("STALL_DRAW") or 0),
            "weight": _parse_weight(meta),
            "weightUnit": "kg",
            "jockeyWeight": float(meta.get("JOCKEY_CLAIM") or 0),
            "gearChanges": "",
            "topToteWin": tote_win,
            "topTotePlace": None,
            "startingPrice": None,
            "flucs": {},
            "selectionResult": None,
            "officialMargin": None,
            "officialTime": None,
            "status": "",
            "competitor": {
                "id": str(sel_id or ""),
                "name": runner.get("runnerName", "Unknown"),
                "slug": _slugify(runner.get("runnerName", "")),
                "sire": meta.get("SIRE_NAME") or "",
                "dam": meta.get("DAM_NAME") or "",
                "country": meta.get("BRED") or "AUS",
                "age": int(meta.get("AGE") or 0),
                "colour": meta.get("COLOUR_TYPE") or "",
                "sex": _SEX_MAP.get(meta.get("SEX_TYPE") or "", ""),
                "stats": {},
                "forms": [],
            },
            "jockey": {
                "id": "",
                "name": meta.get("JOCKEY_NAME") or "",
                "slug": _slugify(meta.get("JOCKEY_NAME") or ""),
                "apprentice": bool(meta.get("JOCKEY_CLAIM")),
            },
            "trainer": {
                "id": "",
                "name": meta.get("TRAINER_NAME") or "",
                "slug": _slugify(meta.get("TRAINER_NAME") or ""),
            },
            "lastRun": None,
        })
    return result


class BetfairClient:
    def __init__(self) -> None:
        from horse_engine.config import settings
        self._app_key: str = getattr(settings, "betfair_app_key", "")
        self._username: str = getattr(settings, "betfair_username", "")
        self._password: str = getattr(settings, "betfair_password", "")
        self._session_token: str | None = None
        # date → (ts, list[market_catalogue])
        self._catalogue_cache: dict[str, tuple[datetime, list]] = {}
        # slug → (ts, meeting_dict)
        self._meeting_cache: dict[str, tuple[datetime, dict]] = {}
        # marketId → (ts, market_book)
        self._book_cache: dict[str, tuple[datetime, dict]] = {}

    # ── Auth ─────────────────────────────────────────────────────────────

    async def _login(self) -> bool:
        if not (self._app_key and self._username and self._password):
            log.warning(
                "Betfair credentials not configured — set BETFAIR_APP_KEY, "
                "BETFAIR_USERNAME, BETFAIR_PASSWORD"
            )
            return False
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    _LOGIN_URL,
                    data={"username": self._username, "password": self._password},
                    headers={
                        "X-Application": self._app_key,
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )
                resp.raise_for_status()
                body = resp.json()
                if body.get("status") == "SUCCESS":
                    self._session_token = body["token"]
                    log.info("Betfair login successful")
                    return True
                log.warning("Betfair login failed: %s", body.get("error", "unknown"))
                return False
        except Exception as e:
            log.warning("Betfair login error: %s", e)
            return False

    async def _ensure_auth(self) -> bool:
        if not self._session_token:
            return await self._login()
        return True

    # ── API ──────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=10))
    async def _api(self, operation: str, body: dict) -> Any:
        if not await self._ensure_auth():
            return None
        url = f"{_API_BASE}{operation}/"
        headers = {
            "X-Application": self._app_key,
            "X-Authentication": self._session_token or "",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code == 401:
                self._session_token = None
                if not await self._login():
                    return None
                headers["X-Authentication"] = self._session_token or ""
                resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()

    # ── Catalogue loading ─────────────────────────────────────────────────

    async def _load_catalogue(self, race_date: str) -> list[dict]:
        """Fetch all AU thoroughbred WIN markets for race_date (AEST)."""
        cached = self._catalogue_cache.get(race_date)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 600:
            return cached[1]

        # AEST is UTC+10 — express the date window with timezone offset so
        # Betfair's UTC filter captures the correct local race day.
        from_dt = f"{race_date}T00:00:00+10:00"
        to_dt = f"{race_date}T23:59:59+10:00"

        result = await self._api("listMarketCatalogue", {
            "filter": {
                "eventTypeIds": [_HORSE_RACING_TYPE_ID],
                "marketCountries": ["AU"],
                "marketTypeCodes": ["WIN"],
                "marketStartTime": {"from": from_dt, "to": to_dt},
                "bspOnly": False,
            },
            "marketProjection": ["EVENT", "RUNNER_METADATA", "MARKET_START_TIME"],
            "sort": "FIRST_TO_START",
            "maxResults": 1000,
        })

        markets: list[dict] = result if isinstance(result, list) else []
        self._catalogue_cache[race_date] = (datetime.utcnow(), markets)
        log.info("Loaded %d Betfair WIN markets for %s", len(markets), race_date)
        return markets

    def _build_meeting_index(self, markets: list[dict], race_date: str) -> dict[str, dict]:
        """Group markets by event id into meeting dicts keyed by slug."""
        by_event: dict[str, dict] = {}
        for mkt in markets:
            event = mkt.get("event") or {}
            event_id = str(event.get("id") or "")
            if not event_id:
                continue
            if event_id not in by_event:
                tz = event.get("timezone") or ""
                state = _TZ_STATE.get(tz, "NSW")
                venue_name = _extract_venue(event)
                slug = _make_meeting_slug(venue_name, race_date)
                by_event[event_id] = {
                    "id": event_id,
                    "name": venue_name,
                    "slug": slug,
                    "venue": venue_name,
                    "state": state,
                    "rail_position": "",
                    "date": race_date,
                    # Standard keys used by main.py
                    "railPosition": "",
                    "meetingDateLocal": race_date,
                    "markets": [],
                }
            by_event[event_id]["markets"].append(mkt)
        return by_event

    # ── Public API ────────────────────────────────────────────────────────

    async def get_meetings(self, race_date: str | None = None) -> list[dict]:
        """Return AU thoroughbred meetings for race_date."""
        d = race_date or date.today().isoformat()
        markets = await self._load_catalogue(d)
        by_event = self._build_meeting_index(markets, d)

        result = []
        for m in by_event.values():
            if m["state"] not in _AU_STATES:
                continue
            self._meeting_cache[m["slug"]] = (datetime.utcnow(), m)
            result.append({
                "id": m["id"],
                "name": m["name"],
                "slug": m["slug"],
                "venue": m["venue"],
                "state": m["state"],
                "rail_position": "",
                "date": d,
            })

        log.info("Found %d AU meetings on %s (Betfair)", len(result), d)
        return result

    async def get_meeting_by_slug(self, slug: str) -> dict | None:
        """Return meeting metadata for a slug. Primes cache if needed."""
        cached = self._meeting_cache.get(slug)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 600:
            return cached[1]

        # Derive race_date from slug suffix (last 8 digits = YYYYMMDD)
        m = re.search(r'-(\d{8})$', slug)
        if m:
            raw = m.group(1)
            race_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
            await self.get_meetings(race_date)
            cached = self._meeting_cache.get(slug)
            if cached:
                return cached[1]
        return None

    async def get_meeting_races(self, slug: str) -> list[dict]:
        """Return lightweight race list for a meeting (no runner details)."""
        meeting = await self.get_meeting_by_slug(slug)
        if not meeting:
            return []

        events = []
        sorted_mkts = sorted(
            meeting.get("markets", []),
            key=lambda mk: mk.get("marketStartTime") or "",
        )
        for i, mkt in enumerate(sorted_mkts, start=1):
            race_num, distance = _parse_market_name(mkt.get("marketName") or "")
            if race_num is None:
                race_num = i
            events.append({
                "id": mkt.get("marketId", ""),
                "eventNumber": race_num,
                "name": mkt.get("marketName", ""),
                "distance": distance,
                "startTime": mkt.get("marketStartTime", ""),
                "status": "Open",
                "raceType": "R",
                "eventClass": "",
                "trackCondition": {"overall": "Good", "rating": "4"},
            })
        return events

    async def get_race(self, slug: str, race_number: int) -> dict | None:
        """Return full event dict (with selections + live odds) for one race."""
        meeting = await self.get_meeting_by_slug(slug)
        if not meeting:
            return None

        target_mkt: dict | None = None
        sorted_mkts = sorted(
            meeting.get("markets", []),
            key=lambda mk: mk.get("marketStartTime") or "",
        )
        for i, mkt in enumerate(sorted_mkts, start=1):
            rn, dist = _parse_market_name(mkt.get("marketName") or "")
            if rn is None:
                rn = i
            if rn == race_number:
                target_mkt = mkt
                break

        if not target_mkt:
            log.warning("Race %d not found in meeting %s (Betfair)", race_number, slug)
            return None

        market_id = target_mkt.get("marketId") or ""
        book = await self._fetch_market_book(market_id)
        rn, dist = _parse_market_name(target_mkt.get("marketName") or "")
        if rn is None:
            rn = race_number

        return {
            "id": market_id,
            "eventNumber": rn,
            "name": target_mkt.get("marketName", ""),
            "distance": dist,
            "startTime": target_mkt.get("marketStartTime", ""),
            "status": "Open",
            "raceType": "R",
            "eventClass": "",
            "trackCondition": {"overall": "Good", "rating": "4"},
            "selections": _build_selections(target_mkt.get("runners") or [], book),
            "_meeting": meeting,
        }

    async def _fetch_market_book(self, market_id: str) -> dict:
        if not market_id:
            return {}
        cached = self._book_cache.get(market_id)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 60:
            return cached[1]
        try:
            result = await self._api("listMarketBook", {
                "marketIds": [market_id],
                "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
            })
            if isinstance(result, list) and result:
                book = result[0]
                self._book_cache[market_id] = (datetime.utcnow(), book)
                return book
        except Exception as e:
            log.debug("listMarketBook failed for %s: %s", market_id, e)
        return {}

    # ── Parsing ───────────────────────────────────────────────────────────

    async def parse_race(self, raw_event: dict, race_date: str, venue: str, state: str) -> Race:
        meeting = raw_event.get("_meeting") or {}
        race_num = raw_event.get("eventNumber", 0)
        meeting_slug = meeting.get("slug") or ""
        date_compact = race_date.replace('-', '')
        venue_slug = (
            meeting_slug.replace(f"-{date_compact}", "")
            if f"-{date_compact}" in meeting_slug
            else meeting_slug
        )

        tc = raw_event.get("trackCondition") or {}
        track_condition = f"{tc.get('overall', 'Good')} {tc.get('rating', '4')}".strip()

        return Race(
            race_id=f"{race_date}_{venue_slug}_R{race_num}",
            date=race_date,
            venue=venue,
            state=state,
            race_number=race_num,
            race_name=raw_event.get("name", ""),
            race_class=raw_event.get("eventClass", ""),
            distance=int(raw_event.get("distance") or 0),
            track_condition=track_condition,
            rail_position=meeting.get("railPosition", ""),
            prize_money=0,
            scheduled_time=raw_event.get("startTime", ""),
            race_type="R",
            runners=self._parse_runners(raw_event.get("selections") or [], track_condition),
        )

    def _parse_runners(self, selections: list[dict], track_condition: str = "Good 4") -> list[Runner]:
        runners = []
        for sel in selections:
            if (sel.get("status") or "").upper() == "SCRATCHED":
                continue
            try:
                r = self._parse_runner(sel, track_condition)
                if r:
                    runners.append(r)
            except Exception as e:
                log.debug("Runner parse error: %s", e)
        return runners

    def _parse_runner(self, sel: dict, track_condition: str = "Good 4") -> Runner | None:
        comp = sel.get("competitor") or {}
        jock = sel.get("jockey") or {}
        trnr = sel.get("trainer") or {}

        horse_name = comp.get("name", "Unknown")
        if not horse_name or horse_name == "Unknown":
            return None

        sire = comp.get("sire") or ""
        profile = SIRE_PROFILES.get(sire, {})
        pedigree = PedigreeProfile(
            sire=sire,
            dam=comp.get("dam") or "",
            dam_sire="",
            distance_aptitude=profile.get("aptitude", "mile"),
            distance_min=int(profile.get("dist_min", 1000)),
            distance_max=int(profile.get("dist_max", 2400)),
            wet_track_score=float(profile.get("wet", 5)),
            first_up_score=float(profile.get("first_up", 5)),
            second_up_score=5.0,
            on_pace_tendency=float(profile.get("on_pace", 5)),
            stamina_index=float(profile.get("stamina", 5)),
            brilliance_index=float(profile.get("brilliance", 5)),
        )

        j_name = jock.get("name") or ""
        t_name = trnr.get("name") or ""
        tote_win = sel.get("topToteWin")

        return Runner(
            barrier=int(sel.get("barrierNumber") or 0),
            tab_number=int(sel.get("competitorNumber") or 0),
            horse_name=horse_name,
            age=int(comp.get("age") or 0),
            sex=comp.get("sex") or "",
            colour=comp.get("colour") or "",
            weight=float(sel.get("weight") or 0),
            jockey=j_name,
            trainer=t_name,
            country=comp.get("country") or "AUS",
            career_starts=0,
            career_wins=0,
            career_places=0,
            pedigree=pedigree,
            tote_win_odds=float(tote_win) if tote_win else None,
            best_available_odds=float(tote_win) if tote_win else None,
            jockey_stats=JockeyStats(
                name=j_name,
                win_rate_overall=10.0,
                win_rate_track=10.0,
                win_rate_distance=10.0,
                win_rate_barrier_low=10.0,
                win_rate_barrier_mid=10.0,
                win_rate_barrier_wide=10.0,
                wins_today=0,
                prizemoney_season=0,
                wins_season=0,
                trainer_jockey_combo_rate=10.0,
            ) if j_name else None,
            trainer_stats=TrainerStats(
                name=t_name,
                win_rate_overall=10.0,
                win_rate_track=10.0,
                win_rate_distance=10.0,
                win_rate_first_up=10.0,
                win_rate_second_up=10.0,
                win_rate_wet=10.0,
                prizemoney_season=0,
                runners_season=0,
                wins_season=0,
            ) if t_name else None,
        )
