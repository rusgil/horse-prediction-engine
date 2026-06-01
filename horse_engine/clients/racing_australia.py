"""
Racing Australia free fields client.

Scrapes https://www.racingaustralia.horse/FreeFields/ for:
  - Meeting discovery across all AU states (Calendar.aspx per state)
  - Full race cards with runners (Acceptances.aspx per meeting)

No odds — integrate TAB scraper separately for prices.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from horse_engine.models.race import (
    FormStart, JockeyStats, PedigreeProfile, Race, Runner, TrainerStats,
)
from horse_engine.pedigree.sire_profiles import SIRE_PROFILES

log = logging.getLogger(__name__)

_BASE = "https://www.racingaustralia.horse/FreeFields"
_AU_STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Timezone offset per state for constructing ISO start times (non-DST / winter)
_STATE_TZ = {
    "NSW": "+10:00", "VIC": "+10:00", "TAS": "+10:00", "ACT": "+10:00",
    "QLD": "+10:00", "SA": "+09:30", "WA": "+08:00", "NT": "+09:30",
}

# Strip bookmaker/sponsor prefixes from venue names so slugs match Punters format
_SPONSOR_RE = re.compile(
    r"^(sportsbet|bet365|tab|ubet|ladbrokes|neds|palmerbet|pointsbet|"
    r"betfair|bluebet|boombet|playup|elitebet|topbetta|racing\.com|"
    r"betstar|centrebet|william\s+hill|draftstars)\s+",
    re.IGNORECASE,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ra_date(iso_date: str) -> str:
    """'2026-05-31' → '2026May31' (Racing Australia calendar key format)."""
    y, m, d = iso_date.split("-")
    return f"{y}{_MONTH_NAMES[int(m)-1]}{int(d):02d}"


_TAPETA_RE = re.compile(r"\s+Tapeta\b", re.IGNORECASE)

def _clean_venue(raw: str) -> str:
    """Strip sponsor prefix and track-surface qualifiers for consistent slugs.
    'Sportsbet Sandown Lakeside' → 'Sandown Lakeside'
    'Devonport Tapeta Synthetic' → 'Devonport Synthetic'
    """
    name = _SPONSOR_RE.sub("", raw).strip()
    name = _TAPETA_RE.sub("", name).strip()
    return name


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _make_slug(venue: str, race_date: str) -> str:
    return f"{_slugify(_clean_venue(venue))}-{race_date.replace('-', '')}"


def _parse_weight(raw: str) -> float:
    m = re.search(r"([\d.]+)\s*kg", raw, re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def _parse_jockey(raw: str) -> tuple[str, float]:
    """'Ms Grace Palmer(a3/52kg)' → ('Ms Grace Palmer', 3.0)"""
    m = re.match(r"^(.+?)\(a(\d+)/[\d.]+kg\)$", raw.strip())
    if m:
        return m.group(1).strip(), float(m.group(2))
    return raw.strip(), 0.0


def _parse_race_header(text: str) -> tuple[int, str, int, str]:
    """
    'Race 3 - 1:40PM RIVERVIEW HOTEL MAIDEN HANDICAP (1556 METRES)'
    → (3, '1:40PM', 1556, 'RIVERVIEW HOTEL MAIDEN HANDICAP')
    """
    m = re.match(
        r"Race\s+(\d+)\s*-\s*([\d:]+[AP]M)\s+(.+?)\s*\((\d+)\s*METRES\)",
        text, re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), m.group(2), int(m.group(4)), m.group(3).strip()
    # Fallback: at least extract race number
    m2 = re.match(r"Race\s+(\d+)", text, re.IGNORECASE)
    return (int(m2.group(1)) if m2 else 0), "", 0, text


def _parse_prize_money(text: str) -> int:
    """Extract 'Of $30,000.' → 30000"""
    m = re.search(r"Of\s+\$([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else 0


def _parse_race_class(text: str) -> str:
    """Extract class from prize description: 'Maiden, Handicap' → 'Maiden Handicap'"""
    # After the prize/jockey welfare lines, class info follows
    m = re.search(r"Jockey Welfare Fund.*?\.([^.]+?)(?:,\s*(?:Set|Handicap|Weight)|BOBS|Track|$)", text)
    if m:
        return m.group(1).strip()
    return ""


def _build_start_time(race_date: str, time_str: str, state: str) -> str:
    """'2026-05-31' + '12:20PM' + 'NSW' → '2026-05-31T12:20:00+10:00'"""
    if not time_str:
        return ""
    try:
        t = datetime.strptime(time_str.strip().upper(), "%I:%M%p")
        tz = _STATE_TZ.get(state, "+10:00")
        return f"{race_date}T{t.strftime('%H:%M')}:00{tz}"
    except ValueError:
        return ""


def _parse_form_string(form: str) -> list[FormStart]:
    """'3088251677' → up to 10 simplified FormStart entries (oldest→newest)."""
    starts = []
    for ch in reversed(form.replace("x", "").replace("X", "")):
        try:
            pos = int(ch) if ch != "0" else 10
            starts.append(FormStart(
                date="", track="", distance=0, track_condition="Good",
                barrier=0, weight=0.0, jockey="", position=pos,
                finishers=max(pos + 3, 8), beaten_margin=0.0,
                race_class="", prize_money=0,
            ))
        except ValueError:
            continue
        if len(starts) >= 10:
            break
    return list(reversed(starts))


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_acceptances_page(html: str, ra_key: str, race_date: str, state: str) -> dict:
    """
    Parse an Acceptances.aspx page into a structured meeting dict with all
    races and their selections.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")

    races: list[dict] = []
    current_race: Optional[dict] = None
    in_header = False

    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        text = cells[0].get_text(strip=True)

        # Race header row — one cell spanning full width
        if len(cells) == 1 and re.match(r"Race\s+\d+", text, re.IGNORECASE):
            race_num, time_str, distance, race_name = _parse_race_header(text)
            if race_num:
                current_race = {
                    "eventNumber": race_num,
                    "name": race_name,
                    "distance": distance,
                    "startTime": _build_start_time(race_date, time_str, state),
                    "status": "Open",
                    "raceType": "R",
                    "eventClass": "",
                    "prize_money": 0,
                    "trackCondition": {"overall": "Good", "rating": "4"},
                    "selections": [],
                }
                races.append(current_race)
                in_header = False
            continue

        # Prize money / class description row
        if current_race and len(cells) == 1 and text.startswith("Of $"):
            current_race["prize_money"] = _parse_prize_money(text)
            current_race["eventClass"] = _parse_race_class(text)
            continue

        # Column header row (No, Last 10, Horse, ...)
        if text in ("No", "Race") or (len(cells) >= 3 and cells[2].get_text(strip=True) == "Horse"):
            in_header = False
            continue

        # Runner row — expect at least 7 cells: No, Last10, Horse, Trainer, Jockey, Barrier, Weight
        if current_race and len(cells) >= 7:
            values = [c.get_text(strip=True) for c in cells]
            tab_num_raw, form_str, horse, trainer, jockey_raw, barrier_raw, weight_raw = values[:7]

            # Validate tab number is numeric
            if not re.match(r"^\d+$", tab_num_raw):
                continue

            jockey_name, jockey_claim = _parse_jockey(jockey_raw)
            weight = _parse_weight(weight_raw)
            try:
                barrier = int(barrier_raw)
            except ValueError:
                barrier = 0

            current_race["selections"].append({
                "competitorNumber": int(tab_num_raw),
                "barrierNumber": barrier,
                "weight": weight,
                "weightUnit": "kg",
                "jockeyWeight": weight - jockey_claim if jockey_claim else weight,
                "jockeyWeightClaim": jockey_claim,
                "gearChanges": "",
                "topToteWin": None,
                "topTotePlace": None,
                "startingPrice": None,
                "flucs": {},
                "selectionResult": None,
                "officialMargin": None,
                "officialTime": None,
                "status": "",
                "form_string": form_str,
                "competitor": {
                    "id": "",
                    "name": horse,
                    "slug": _slugify(horse),
                    "sire": "", "dam": "",
                    "country": "AUS",
                    "age": 0, "colour": "", "sex": "",
                    "stats": {}, "forms": [],
                },
                "jockey": {
                    "id": "", "name": jockey_name,
                    "slug": _slugify(jockey_name),
                    "apprentice": bool(jockey_claim),
                },
                "trainer": {
                    "id": "", "name": trainer,
                    "slug": _slugify(trainer),
                },
                "lastRun": None,
            })

    # Parse venue from RA key: "2026May31,NSW,Sportsbet Sandown Lakeside"
    parts = ra_key.split(",", 2)
    raw_venue = parts[2] if len(parts) > 2 else ""
    venue = _clean_venue(raw_venue)
    slug = _make_slug(raw_venue, race_date)

    return {
        "id": ra_key,
        "name": venue,
        "slug": slug,
        "venue": venue,
        "state": state,
        "rail_position": "",
        "railPosition": "",
        "meetingDateLocal": race_date,
        "date": race_date,
        "races": races,
    }


# ── Results parser ────────────────────────────────────────────────────────────

def _parse_results_page(html: str) -> dict[int, dict]:
    """
    Parse a Results.aspx page.
    Returns {race_number: {'track_condition': str, 'runners': {name_lower: {'position', 'margin', 'sp'}}}}
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    results: dict[int, dict] = {}
    current_race_num: Optional[int] = None

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue
        first_text = rows[0].get_text(strip=True)

        # Race header: "Race 3 - 1:55PM ..."
        m = re.match(r"Race\s+(\d+)\s*-", first_text, re.IGNORECASE)
        if m:
            current_race_num = int(m.group(1))
            results[current_race_num] = {"track_condition": "", "runners": {}}
            continue

        # Race details: "Of $50,000 ... Track Condition:Heavy 8 ..."
        if first_text.startswith("Of $") and current_race_num is not None:
            tc = re.search(r"Track Condition:\s*([A-Za-z]+\s*\d*)", first_text)
            if tc:
                results[current_race_num]["track_condition"] = tc.group(1).strip()
            continue

        # Results table: header row has "Finish" and "Horse"
        header = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])]
        if "Finish" not in header or "Horse" not in header or current_race_num is None:
            continue

        fi = header.index("Finish")
        hi = header.index("Horse")
        mi = header.index("Margin") if "Margin" in header else None
        si = header.index("Starting Price") if "Starting Price" in header else None

        for row in rows[1:]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) <= hi:
                continue
            horse = cells[hi]
            if not horse:
                continue

            finish = cells[fi] if fi < len(cells) else ""
            try:
                position = int(finish) if finish.isdigit() else None
            except (ValueError, AttributeError):
                position = None

            margin = 0.0
            if mi is not None and mi < len(cells):
                m2 = re.search(r"([\d.]+)", cells[mi])
                if m2:
                    margin = float(m2.group(1))

            sp = None
            if si is not None and si < len(cells):
                sp_raw = cells[si].replace("F", "").replace("$", "").strip()
                try:
                    sp = float(sp_raw) if sp_raw else None
                except ValueError:
                    pass

            results[current_race_num]["runners"][horse.lower()] = {
                "position": position,
                "margin": margin,
                "sp": sp,
            }

    return results


# ── Client ────────────────────────────────────────────────────────────────────

class RacingAustraliaClient:
    def __init__(self) -> None:
        # "date:state" → (ts, list of {ra_key, venue, slug, state})
        self._calendar_cache: dict[str, tuple[datetime, list]] = {}
        # ra_key → (ts, parsed meeting dict)
        self._meeting_cache: dict[str, tuple[datetime, dict]] = {}
        # ra_key → (ts, parsed results dict)
        self._results_cache: dict[str, tuple[datetime, dict]] = {}
        # slug → ra_key
        self._slug_to_key: dict[str, str] = {}
        # rate-limit: one request at a time
        self._sem: asyncio.Semaphore | None = None

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(3)
        return self._sem

    async def _get(self, url: str) -> str:
        async with self._get_sem():
            await asyncio.sleep(0.3)
            async with httpx.AsyncClient(headers=_HEADERS, timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text

    # ── Calendar ──────────────────────────────────────────────────────────────

    async def _fetch_state_calendar(self, state: str, race_date: str) -> list[dict]:
        cache_key = f"{race_date}:{state}"
        cached = self._calendar_cache.get(cache_key)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 600:
            return cached[1]

        ra_date = _ra_date(race_date)
        try:
            html = await self._get(f"{_BASE}/Calendar.aspx?State={state}")
        except Exception as e:
            log.debug("Calendar fetch failed for %s: %s", state, e)
            return []

        soup = BeautifulSoup(html, "html.parser")
        meetings = []
        for link in soup.find_all("a", href=re.compile("Acceptances")):
            href = link.get("href", "")
            # Extract key from href like /FreeFields/Acceptances.aspx?Key=2026May31%2CNSW%2CVenue
            m = re.search(r"Key=([^&\"]+)", href)
            if not m:
                continue
            from urllib.parse import unquote
            ra_key = unquote(m.group(1))
            if not ra_key.startswith(ra_date):
                continue
            parts = ra_key.split(",", 2)
            if len(parts) < 3:
                continue
            raw_venue = parts[2]
            # Skip trial/trackwork/jumpout meetings — no race cards or odds available
            if re.search(r"\b(Trial|Trail|Trials|TRL|Jumpout|Jump\s*Out)\b", raw_venue, re.IGNORECASE):
                continue
            venue = _clean_venue(raw_venue)
            slug = _make_slug(raw_venue, race_date)
            self._slug_to_key[slug] = ra_key
            meetings.append({
                "id": ra_key,
                "name": venue,
                "slug": slug,
                "venue": venue,
                "state": state,
                "rail_position": "",
                "date": race_date,
            })

        self._calendar_cache[cache_key] = (datetime.utcnow(), meetings)
        return meetings

    # ── Acceptances ───────────────────────────────────────────────────────────

    async def _fetch_meeting(self, ra_key: str, race_date: str, state: str) -> dict | None:
        cached = self._meeting_cache.get(ra_key)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 300:
            return cached[1]
        from urllib.parse import quote
        url = f"{_BASE}/Acceptances.aspx?Key={quote(ra_key, safe='')}"
        try:
            html = await self._get(url)
        except Exception as e:
            log.warning("Acceptances fetch failed for %s: %s", ra_key, e)
            return None
        parsed = _parse_acceptances_page(html, ra_key, race_date, state)
        self._meeting_cache[ra_key] = (datetime.utcnow(), parsed)
        return parsed

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_meetings(self, race_date: str | None = None) -> list[dict]:
        d = race_date or date.today().isoformat()
        tasks = [self._fetch_state_calendar(s, d) for s in _AU_STATES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        meetings: list[dict] = []
        seen_slugs: set[str] = set()
        for r in results:
            if isinstance(r, list):
                for m in r:
                    slug = m.get("slug", "")
                    if slug and slug in seen_slugs:
                        continue
                    seen_slugs.add(slug)
                    meetings.append(m)
        log.info("Found %d AU meetings on %s (Racing Australia)", len(meetings), d)
        return meetings

    async def get_meeting_by_slug(self, slug: str) -> dict | None:
        ra_key = self._slug_to_key.get(slug)
        if not ra_key:
            # Try to derive date + prime the cache
            m = re.search(r"-(\d{8})$", slug)
            if m:
                raw = m.group(1)
                race_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
                await self.get_meetings(race_date)
                ra_key = self._slug_to_key.get(slug)
        if not ra_key:
            return None
        parts = ra_key.split(",", 2)
        state = parts[1] if len(parts) > 1 else "NSW"
        race_date = self._ra_key_to_date(ra_key)
        meeting = await self._fetch_meeting(ra_key, race_date, state)
        if meeting:
            return {
                "id": meeting["id"],
                "name": meeting["name"],
                "slug": meeting["slug"],
                "railPosition": "",
                "meetingDateLocal": meeting["meetingDateLocal"],
                "venue": {"name": meeting["venue"], "state": meeting["state"]},
            }
        return None

    async def get_meeting_races(self, slug: str) -> list[dict]:
        ra_key = self._slug_to_key.get(slug)
        if not ra_key:
            await self.get_meeting_by_slug(slug)
            ra_key = self._slug_to_key.get(slug)
        if not ra_key:
            return []
        parts = ra_key.split(",", 2)
        state = parts[1] if len(parts) > 1 else "NSW"
        race_date = self._ra_key_to_date(ra_key)
        meeting = await self._fetch_meeting(ra_key, race_date, state)
        if not meeting:
            return []
        return [
            {
                "id": f"{ra_key}_R{r['eventNumber']}",
                "eventNumber": r["eventNumber"],
                "name": r["name"],
                "distance": r["distance"],
                "startTime": r["startTime"],
                "status": r["status"],
                "raceType": "R",
                "eventClass": r["eventClass"],
                "trackCondition": r["trackCondition"],
            }
            for r in meeting["races"]
        ]

    async def get_race(self, slug: str, race_number: int) -> dict | None:
        ra_key = self._slug_to_key.get(slug)
        if not ra_key:
            await self.get_meeting_by_slug(slug)
            ra_key = self._slug_to_key.get(slug)
        if not ra_key:
            return None
        parts = ra_key.split(",", 2)
        state = parts[1] if len(parts) > 1 else "NSW"
        race_date = self._ra_key_to_date(ra_key)
        meeting = await self._fetch_meeting(ra_key, race_date, state)
        if not meeting:
            return None
        race = next((r for r in meeting["races"] if r["eventNumber"] == race_number), None)
        if not race:
            log.warning("Race %d not found in meeting %s", race_number, slug)
            return None
        return {**race, "_meeting": meeting}

    async def get_results(self, ra_key: str) -> dict[int, dict]:
        """
        Fetch race results for a meeting.
        Returns {race_num: {'track_condition': str, 'runners': {name_lower: {'position', 'margin', 'sp'}}}}
        Cached 5 minutes — results don't change once published.
        """
        cached = self._results_cache.get(ra_key)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 300:
            return cached[1]
        from urllib.parse import quote
        url = f"{_BASE}/Results.aspx?Key={quote(ra_key, safe='')}"
        try:
            html = await self._get(url)
        except Exception as e:
            log.warning("RA results fetch failed for %s: %s", ra_key, e)
            return {}
        parsed = _parse_results_page(html)
        self._results_cache[ra_key] = (datetime.utcnow(), parsed)
        total_runners = sum(len(r["runners"]) for r in parsed.values())
        log.debug("RA results: %d races, %d runners for %s", len(parsed), total_runners, ra_key)
        return parsed

    async def parse_race(self, raw_event: dict, race_date: str, venue: str, state: str) -> Race:
        meeting = raw_event.get("_meeting") or {}
        race_num = raw_event.get("eventNumber", 0)
        slug = meeting.get("slug") or ""
        date_compact = race_date.replace("-", "")
        venue_slug = (
            slug.replace(f"-{date_compact}", "")
            if f"-{date_compact}" in slug else slug
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
            prize_money=int(raw_event.get("prize_money") or 0),
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

        horse_name = comp.get("name", "")
        if not horse_name:
            return None

        sire = comp.get("sire") or ""
        profile = SIRE_PROFILES.get(sire, {})
        pedigree = PedigreeProfile(
            sire=sire, dam=comp.get("dam") or "", dam_sire="",
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

        last_10 = _parse_form_string(sel.get("form_string") or "")
        j_name = jock.get("name") or ""
        t_name = trnr.get("name") or ""

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
            career_starts=0, career_wins=0, career_places=0,
            last_10_starts=last_10,
            pedigree=pedigree,
            tote_win_odds=sel.get("topToteWin"),
            best_available_odds=sel.get("topToteWin"),
            jockey_stats=JockeyStats(
                name=j_name, win_rate_overall=10.0, win_rate_track=10.0,
                win_rate_distance=10.0, win_rate_barrier_low=10.0,
                win_rate_barrier_mid=10.0, win_rate_barrier_wide=10.0,
                wins_today=0, prizemoney_season=0, wins_season=0,
                trainer_jockey_combo_rate=10.0,
            ) if j_name else None,
            trainer_stats=TrainerStats(
                name=t_name, win_rate_overall=10.0, win_rate_track=10.0,
                win_rate_distance=10.0, win_rate_first_up=10.0,
                win_rate_second_up=10.0, win_rate_wet=10.0,
                prizemoney_season=0, runners_season=0, wins_season=0,
            ) if t_name else None,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ra_key_to_date(ra_key: str) -> str:
        """'2026May31,NSW,Venue' → '2026-05-31'"""
        part = ra_key.split(",")[0]  # '2026May31'
        m = re.match(r"(\d{4})([A-Za-z]+)(\d{2})", part)
        if m:
            year, mon_str, day = m.group(1), m.group(2), m.group(3)
            try:
                month = _MONTH_NAMES.index(mon_str[:3].capitalize()) + 1
                return f"{year}-{month:02d}-{int(day):02d}"
            except ValueError:
                pass
        return date.today().isoformat()
