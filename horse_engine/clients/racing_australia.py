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
import os
import random
import re
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from horse_engine.models.race import (
    FormStart, JockeyStats, PedigreeProfile, Race, Runner, TrainerStats,
)
from horse_engine.pedigree.sire_profiles import SIRE_PROFILES

log = logging.getLogger(__name__)

_BASE = "https://www.racingaustralia.horse/FreeFields"
_IF_BASE = "https://www.racingaustralia.horse/InteractiveForm"

# When RA_PROXY_URL is set, all upstream requests are routed through that proxy
# instead of hitting racingaustralia.horse directly. The proxy is a tiny
# FastAPI app on a DigitalOcean droplet (see droplet-proxy/ in repo root) —
# its IP isn't on RA's WAF blocklist, so this bypasses the block. Auth via
# X-Proxy-Secret header, which the proxy validates before forwarding.
#
# Both env vars must be set to enable; if RA_PROXY_URL is empty the client
# behaves exactly as before (direct to RA).
_RA_PROXY_URL = os.environ.get("RA_PROXY_URL", "").rstrip("/")
_RA_PROXY_SECRET = os.environ.get("RA_PROXY_SECRET", "")
_RA_PROXY_ACTIVE = bool(_RA_PROXY_URL and _RA_PROXY_SECRET)
if _RA_PROXY_ACTIVE:
    log.info("[RA] Proxy enabled — routing through %s", _RA_PROXY_URL)


def _proxied(url: str) -> str:
    """Rewrite an RA URL to go through the configured proxy.
    https://www.racingaustralia.horse/FreeFields/Calendar.aspx?State=NSW
       -> {RA_PROXY_URL}/proxy/FreeFields/Calendar.aspx?State=NSW
    No-op when the proxy isn't configured."""
    if not _RA_PROXY_ACTIVE:
        return url
    prefix = "https://www.racingaustralia.horse/"
    if url.startswith(prefix):
        return f"{_RA_PROXY_URL}/proxy/{url[len(prefix):]}"
    return url
_AU_STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Timezone offset per state for constructing ISO start times (non-DST / winter)
_STATE_TZ = {
    "NSW": "+10:00", "VIC": "+10:00", "TAS": "+10:00", "ACT": "+10:00",
    "QLD": "+10:00", "SA": "+09:30", "WA": "+08:00", "NT": "+09:30",
}

# Strip bookmaker/sponsor prefixes from venue names so slugs match our internal format
_SPONSOR_RE = re.compile(
    r"^(sportsbet|bet365|tab|ubet|ladbrokes|neds|palmerbet|pointsbet|"
    r"betfair|bluebet|boombet|playup|elitebet|topbetta|racing\.com|"
    r"betstar|centrebet|william\s+hill|draftstars)\s+",
    re.IGNORECASE,
)

# Pool of recent, plausible browser UAs. Rotated per-request so we look like
# a mix of organic visitors rather than one bot. Refresh occasionally as
# version numbers age (Cloudflare-style WAFs flag stale UAs).
_USER_AGENTS = [
    # macOS Chrome
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Windows Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # macOS Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    # Windows Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    # macOS Firefox
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:128.0) Gecko/20100101 Firefox/128.0",
    # Windows Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
]

def _build_headers(referer: str | None = None) -> dict[str, str]:
    """Realistic browser header set with rotating UA. Including a Referer
    where one makes sense (inner-page requests) makes the traffic look like
    a normal user clicking through the site, not a bot scraping URLs."""
    ua = random.choice(_USER_AGENTS)
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "max-age=0",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin" if referer else "none",
        "Sec-Fetch-User": "?1",
    }
    if referer:
        headers["Referer"] = referer
    return headers


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


# ── InteractiveForm parsers ───────────────────────────────────────────────────

def _going_category(track_condition: str) -> str:
    tc = track_condition.lower()
    if "heavy" in tc:
        return "heavy"
    if "soft" in tc or "yield" in tc:
        return "soft"
    if "firm" in tc:
        return "firm"
    return "good"


def _parse_stat(text: str, label: str) -> tuple[int, int, int]:
    """Parse 'Label: N:W-P-T' → (starts, wins, places). Returns (0,0,0) if not found."""
    m = re.search(rf"{re.escape(label)}:\s*(\d+):(\d+)-(\d+)-(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return 0, 0, 0


def _parse_horse_form_page(html: str) -> dict:
    """
    Parse HorseFullForm.aspx into a dict of career stats and recent run history.
    The 'Track' and 'Dist' stats are context-specific to the race entry (raceentry param).
    """
    soup = BeautifulSoup(html, "html.parser")
    result: dict = {}

    # Extract sire/dam/age/sex/colour from the header td in horse-search-details.
    # Format: "HORSENAME 3yo Bay Colt D.O.B: 29-Oct-2022 by SIRE from DAM View..."
    for table in soup.find_all("table", class_="horse-search-details"):
        # Find the td that contains the pedigree/identity block (may not be the first td)
        header_td = next(
            (td for td in table.find_all("td") if " by " in td.get_text(" ")),
            None,
        )
        if header_td:
            header = header_td.get_text(" ", strip=True)
            age_m = re.search(r"(\d+)yo", header)
            if age_m:
                result["age"] = int(age_m.group(1))
            sex_m = re.search(r"\b(Colt|Filly|Gelding|Mare|Horse|Rig|Stallion)\b", header)
            if sex_m:
                result["sex"] = sex_m.group(1)
            colour_m = re.search(r"(\d+yo)\s+([\w\s/]+?)\s+(Colt|Filly|Gelding|Mare|Horse|Rig|Stallion)", header)
            if colour_m:
                result["colour"] = colour_m.group(2).strip()
            sire_m = re.search(r"\bby\s+([A-Z][A-Z0-9 '()\-]+?)\s+from\b", header)
            if sire_m:
                result["sire"] = sire_m.group(1).strip()
            dam_m = re.search(r"\bfrom\s+([A-Z][A-Z0-9 '()\-]+?)(?:\s+View|\s*$)", header)
            if dam_m:
                result["dam"] = dam_m.group(1).strip()
        break

    # Find the Career row in the horse-search-details table
    for table in soup.find_all("table", class_="horse-search-details"):
        for row in table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            if "Career" not in th.get_text():
                continue

            text = td.get_text(" ", strip=True)

            # Summary: "25-1:5:5" → starts-wins:places:thirds
            m = re.search(r"Summary:\s*(\d+)-(\d+):(\d+):(\d+)", text)
            if m:
                result["career_starts"] = int(m.group(1))
                result["career_wins"] = int(m.group(2))
                result["career_places"] = int(m.group(3))

            # Track / Dist / Track/Dist — context-aware via raceentry
            for label, key in [
                ("Track/Dist", "track_dist"),
                ("Track", "track"),
                ("Dist", "dist"),
            ]:
                s, w, p = _parse_stat(text, label)
                result[f"{key}_starts"] = s
                result[f"{key}_wins"] = w
                result[f"{key}_places"] = p

            # Preparation stats
            for label, key in [("1st Up", "first_up"), ("2nd Up", "second_up")]:
                s, w, _ = _parse_stat(text, label)
                result[f"{key}_starts"] = s
                result[f"{key}_wins"] = w

            # Going categories
            for cond in ["Firm", "Good", "Soft", "Heavy"]:
                s, w, _ = _parse_stat(text, cond)
                result[f"{cond.lower()}_starts"] = s
                result[f"{cond.lower()}_wins"] = w
            break

    # Parse recent run history from interactive-race-fields table
    form_starts: list[FormStart] = []
    table = soup.find("table", class_="interactive-race-fields")
    if table:
        for row in table.find_all("tr"):
            if "OddRow" not in (row.get("class") or []) and "EvenRow" not in (row.get("class") or []):
                continue
            pos_td = row.find("td", class_="Pos")
            remain_td = row.find("td", class_="remain")
            if not pos_td or not remain_td:
                continue

            pos_text = pos_td.get_text(strip=True)
            remain_text = remain_td.get_text(" ", strip=True)

            pos_m = re.match(r"(\d+)(?:st|nd|rd|th)\s+of\s+(\d+)", pos_text)
            if not pos_m:
                continue
            pos = int(pos_m.group(1))
            finishers = int(pos_m.group(2))

            # Date: "26Sep25" → "2025-09-26"
            run_date = ""
            date_m = re.search(r"\b(\d{1,2})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{2})\b", remain_text)
            if date_m:
                day, mon, yr = int(date_m.group(1)), date_m.group(2), date_m.group(3)
                month_num = _MONTH_NAMES.index(mon) + 1
                run_date = f"20{yr}-{month_num:02d}-{day:02d}"

            # Track code: first word(s) before date
            track = ""
            track_m = re.match(r"^([A-Z][A-Z\s]+?)\s+\d{1,2}[A-Z]", remain_text)
            if track_m:
                track = track_m.group(1).strip()

            # Distance
            distance = 0
            dist_m = re.search(r"\b(\d{3,4})m\b", remain_text)
            if dist_m:
                distance = int(dist_m.group(1))

            # Going (e.g. "Good4", "Soft5", "Heavy8")
            going = "Good"
            going_m = re.search(r"\b(Firm|Good|Soft|Heavy|Synthetic)(\d?)\b", remain_text)
            if going_m:
                going = f"{going_m.group(1)} {going_m.group(2)}".strip()

            # Beaten margin — winner has 0.0, others have NL pattern like "9.51L,"
            margin = 0.0
            if pos > 1:
                margin_m = re.search(r"[\s,]([\d.]+)L,", remain_text)
                if margin_m:
                    margin = float(margin_m.group(1))
                else:
                    margin = 5.0  # unknown, assume beaten

            # Weight (first kg value — should be this horse's weight)
            weight = 0.0
            weight_m = re.search(r"([\d.]+)kg", remain_text)
            if weight_m:
                weight = float(weight_m.group(1))

            # Barrier
            barrier = 0
            barrier_m = re.search(r"Barrier\s+(\d+)", remain_text)
            if barrier_m:
                barrier = int(barrier_m.group(1))

            # Race class
            race_class = ""
            class_m = re.search(
                r"\b(MDN|Maiden|CL\d|Class\s*\d|BM\d+|G[1-3]|Listed|Highway|CTRY|F&M|2YO|3YO)\b",
                remain_text, re.IGNORECASE,
            )
            if class_m:
                race_class = class_m.group(1).upper()

            # Prize
            prize = 0
            prize_m = re.search(r"\$([\d,]+)\b", remain_text)
            if prize_m:
                prize = int(prize_m.group(1).replace(",", ""))

            # Jockey
            jockey_link = remain_td.find("a", href=re.compile(r"JockeyLastRuns"))
            jockey = jockey_link.get_text(strip=True) if jockey_link else ""

            form_starts.append(FormStart(
                date=run_date,
                track=track,
                distance=distance,
                track_condition=going,
                barrier=barrier,
                weight=weight,
                jockey=jockey,
                position=pos,
                finishers=finishers,
                beaten_margin=margin,
                race_class=race_class,
                prize_money=prize,
            ))

    result["form_starts"] = form_starts
    return result


def _parse_person_form_page(html: str) -> dict:
    """
    Parse JockeyLastRuns.aspx or TrainerLastRuns.aspx.
    Returns overall win rate, wet track win rate, and run count.
    """
    soup = BeautifulSoup(html, "html.parser")

    total = wins = wet_total = wet_wins = 0

    table = soup.find("table", class_="interactive-race-fields")
    if table:
        for row in table.find_all("tr"):
            if "OddRow" not in (row.get("class") or []) and "EvenRow" not in (row.get("class") or []):
                continue
            pos_td = row.find("td", class_="Pos")
            remain_td = row.find("td", class_="remain")
            if not pos_td or not remain_td:
                continue

            pos_m = re.match(r"(\d+)(?:st|nd|rd|th)\s+of\s+(\d+)", pos_td.get_text(strip=True))
            if not pos_m:
                continue

            pos = int(pos_m.group(1))
            total += 1
            if pos == 1:
                wins += 1

            remain_text = remain_td.get_text(" ", strip=True)
            going_m = re.search(r"\b(Soft|Heavy)\d?\b", remain_text, re.IGNORECASE)
            if going_m:
                wet_total += 1
                if pos == 1:
                    wet_wins += 1

    win_rate = round(wins / total * 100, 2) if total > 0 else 10.0
    wet_rate = round(wet_wins / wet_total * 100, 2) if wet_total > 0 else win_rate

    return {"win_rate": win_rate, "wet_rate": wet_rate, "total_runs": total}


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

            # Extract InteractiveForm codes from href attributes
            horsecode = raceentry = jockeycode = trainercode = ""
            horse_link = cells[2].find("a", href=True)
            if horse_link:
                href = horse_link.get("href", "")
                hm = re.search(r"horsecode=([^&\"]+)", href)
                rem = re.search(r"raceentry=([^&\"]+)", href)
                horsecode = hm.group(1) if hm else ""
                raceentry = rem.group(1) if rem else ""
            trainer_link = cells[3].find("a", href=True)
            if trainer_link:
                tm = re.search(r"trainercode=([^&\"]+)", trainer_link.get("href", ""))
                trainercode = tm.group(1) if tm else ""
            jockey_link = cells[4].find("a", href=True)
            if jockey_link:
                jm = re.search(r"jockeycode=([^&\"]+)", jockey_link.get("href", ""))
                jockeycode = jm.group(1) if jm else ""

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
                "horsecode": horsecode,
                "raceentry": raceentry,
                "jockeycode": jockeycode,
                "trainercode": trainercode,
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

_DIVIDEND_LABELS = {
    "trifecta": "trifecta",
    "exacta": "exacta",
    "quinella": "quinella",
    "first four": "first_four",
    "first 4": "first_four",
}


def _parse_dividends_from_text(text: str) -> dict[str, float]:
    """Extract exotic dividends from a 'Race Dividends' table-cell text blob.

    RA's Results.aspx lays out dividends in many shapes — most commonly
    "Trifecta 10-11-2 $324.40" on a single line, sometimes split across
    cells. Greedy regex over the whole block, label-prefixed dollar amount.
    Returns: {trifecta: float, exacta: float, ...} for any labels found.
    """
    out: dict[str, float] = {}
    if not text:
        return out
    # Capture "<label> ... $<amount>" with the amount possibly comma-grouped.
    for label, key in _DIVIDEND_LABELS.items():
        m = re.search(
            rf"{re.escape(label)}\b[^\$]{{0,80}}\$([\d,]+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if m:
            try:
                out[key] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return out


def _parse_results_page(html: str) -> dict[int, dict]:
    """
    Parse a Results.aspx page.
    Returns {race_number: {'track_condition': str, 'dividends': dict,
             'runners': {name_lower: {'position', 'margin', 'sp'}}}}
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
            results[current_race_num] = {"track_condition": "", "dividends": {}, "runners": {}}
            continue

        # Dividend block — contains text like 'Trifecta 10-11-2 $324.40'.
        # Match any cell whose text starts with one of the dividend labels.
        if current_race_num is not None:
            full_text = table.get_text(" ", strip=True)
            divs = _parse_dividends_from_text(full_text)
            if divs:
                results[current_race_num].setdefault("dividends", {}).update(divs)

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
        # RA's results table has a "No." column with the saddle number =
        # tab_number. Previously ignored, which meant seed_ra_results had
        # to derive tab_number by name-matching against our own prediction
        # rows. Any finisher that wasn't in our pre-race field (late
        # scratchings, missed enrichment) got tab_number=NULL and left
        # bet settlement stuck 'pending' forever (health-check finding
        # top3_missing_tab_number). Extract it here so RA is the primary
        # source of truth for tab_number.
        ti = header.index("No.") if "No." in header else None

        for row in rows[1:]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) <= hi:
                continue
            horse = cells[hi]
            if not horse:
                continue

            finish = cells[fi] if fi < len(cells) else ""
            try:
                fm = re.match(r"^(\d+)", finish.strip())
                position = int(fm.group(1)) if fm else None
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

            tab_number = None
            if ti is not None and ti < len(cells):
                tab_raw = cells[ti].strip()
                if tab_raw:
                    # RA marks emergency runners with an alpha suffix on the
                    # saddle number ('11e', '16e' — the 'e' = emergency, they
                    # got the start after a scratching). Strip the suffix so
                    # the tab_number stored matches what the TAB API/bet
                    # settlement uses (they only carry the integer part).
                    m_tab = re.match(r"(\d+)", tab_raw)
                    if m_tab:
                        try:
                            tab_number = int(m_tab.group(1))
                        except ValueError:
                            pass

            results[current_race_num]["runners"][horse.lower()] = {
                "position": position,
                "margin": margin,
                "sp": sp,
                "tab_number": tab_number,
            }

    return results


# Persistent calendar cache helpers — keyed by (race_date, state). Loaded
# on cold start so Railway redeploys don't trigger a fresh 32-Calendar
# fanout. Bumped from 1h → 6h on 2026-07-21 to cut Calendar.aspx daily
# volume by 6x — RA publishes meetings days ahead, so intra-day churn is
# rare; a mid-day new meeting shows up within one refresh window at the
# next 6h tick or on the first user cache-miss for that state.
_PERSIST_TTL_SECONDS = 21600


async def _load_calendar_from_db(race_date: str, state: str):
    """Return (meetings, slug_to_key_kvs) if a fresh cache row exists,
    else None. Imports DB models lazily to avoid a circular import."""
    try:
        from horse_engine.models.database import RaCalendarCacheRow
        from horse_engine.api.database import get_session
        from sqlalchemy import select as _select
        import json as _json
        async with get_session() as session:
            row = (await session.execute(
                _select(RaCalendarCacheRow)
                .where(RaCalendarCacheRow.race_date == race_date)
                .where(RaCalendarCacheRow.state == state)
            )).scalars().first()
        if row is None:
            return None
        age = (datetime.utcnow() - row.fetched_at).total_seconds()
        if age > _PERSIST_TTL_SECONDS:
            return None
        meetings = _json.loads(row.meetings_json or "[]")
        slug_kvs = _json.loads(row.slug_to_key_json or "{}")
        return meetings, slug_kvs
    except Exception as e:
        log.debug("calendar DB load failed for %s/%s: %s", race_date, state, e)
        return None


async def _persist_calendar_to_db(race_date: str, state: str, meetings: list, slug_kvs: dict):
    """Upsert the (race_date, state) cache row. Best-effort — never raises
    upward."""
    if not meetings:
        return
    try:
        from horse_engine.models.database import RaCalendarCacheRow
        from horse_engine.api.database import get_session
        from sqlalchemy import select as _select
        import json as _json
        async with get_session() as session:
            row = (await session.execute(
                _select(RaCalendarCacheRow)
                .where(RaCalendarCacheRow.race_date == race_date)
                .where(RaCalendarCacheRow.state == state)
            )).scalars().first()
            if row is None:
                session.add(RaCalendarCacheRow(
                    race_date=race_date, state=state,
                    meetings_json=_json.dumps(meetings),
                    slug_to_key_json=_json.dumps(slug_kvs),
                ))
            else:
                row.meetings_json = _json.dumps(meetings)
                row.slug_to_key_json = _json.dumps(slug_kvs)
                row.fetched_at = datetime.utcnow()
            await session.commit()
    except Exception as e:
        log.debug("calendar DB persist failed for %s/%s: %s", race_date, state, e)


# ── Client ────────────────────────────────────────────────────────────────────

class RacingAustraliaClient:
    def __init__(self) -> None:
        # "date:state" → (ts, list of {ra_key, venue, slug, state})
        self._calendar_cache: dict[str, tuple[datetime, list]] = {}
        # Per-cache-key locks — when multiple concurrent callers miss the
        # in-memory + DB cache, only one of them issues the RA fetch.
        # Without this the cache layer was being stampeded — production
        # logs showed 11k Calendar.aspx hits/24h vs the 768 the TTL
        # should have allowed (14× over).
        self._calendar_locks: dict[str, asyncio.Lock] = {}
        self._results_locks: dict[str, asyncio.Lock] = {}
        # ra_key → (ts, parsed meeting dict)
        self._meeting_cache: dict[str, tuple[datetime, dict]] = {}
        # ra_key → (ts, parsed results dict)
        self._results_cache: dict[str, tuple[datetime, dict]] = {}
        # slug → ra_key
        self._slug_to_key: dict[str, str] = {}
        # horsecode → (ts, parsed form dict)  — 1 hour TTL
        self._horse_form_cache: dict[str, tuple[datetime, dict]] = {}
        # jockeycode → (ts, parsed form dict)
        self._jockey_form_cache: dict[str, tuple[datetime, dict]] = {}
        # trainercode → (ts, parsed form dict)
        self._trainer_form_cache: dict[str, tuple[datetime, dict]] = {}
        # rate-limit
        self._sem: asyncio.Semaphore | None = None
        # Circuit breaker: when RA returns 403 we back off the whole client
        # for a cooldown window so we don't grind the WAF block deeper.
        # Hard rule: this program can not HAMMER apis.
        self._blocked_until: datetime | None = None
        self._block_count: int = 0  # tracks consecutive 403s for backoff growth
        # DB-hydration flag for the venue-key cache. On first find_results
        # call, we do ONE bulk read from ra_venue_key_cache to populate
        # _slug_to_key. All subsequent lookups hit RAM only. Prevents the
        # per-request DB session opens that crashed asyncpg pool on
        # 2026-07-18. Successful new resolutions write back via a fire-
        # and-forget task (see _persist_venue_key_bg).
        self._db_hydrated: bool = False
        self._hydrate_lock = asyncio.Lock()

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            # Single-flight requests. RA is small enough that we don't need
            # parallelism, and parallel calls were how we tripped the WAF.
            self._sem = asyncio.Semaphore(1)
        return self._sem

    def _is_blocked(self) -> bool:
        if self._blocked_until is None:
            return False
        if datetime.utcnow() >= self._blocked_until:
            self._blocked_until = None
            self._block_count = 0
            return False
        return True

    def _trip_breaker(self) -> None:
        """Called on 403. Exponential backoff: 60s, 5min, 15min, 60min cap."""
        self._block_count += 1
        backoff = min(60 * (5 ** (self._block_count - 1)), 3600)
        from datetime import timedelta as _td
        self._blocked_until = datetime.utcnow() + _td(seconds=backoff)
        log.warning(
            "[RA] WAF block detected (403). Circuit breaker tripped — "
            "no requests for %ds (consecutive=%d).",
            backoff, self._block_count,
        )

    async def _request(self, url: str, *, referer: str | None = None, timeout: float = 20.0) -> str:
        """Shared GET path. Honours the breaker, jitters delay, rotates UA,
        and trips the breaker on 403 instead of grinding the WAF deeper.

        Double-check pattern: the pre-semaphore check catches callers made
        AFTER the breaker was tripped. A burst arrival (asyncio.gather of
        30+ form fetches) can pass the first check together, queue at the
        semaphore, and then request #1 trips the breaker on its 403 while
        requests #2..30 are still queued but past the check — they'd all
        fire anyway, giving RA another 30 free 403s to fingerprint on.
        Re-checking inside the semaphore short-circuits every queued
        request the moment #1 has tripped, so a WAF-block burst goes from
        30+ wasted hits to at most 1.
        """
        if self._is_blocked():
            raise httpx.HTTPStatusError(
                f"RA circuit breaker open until {self._blocked_until.isoformat()}",
                request=httpx.Request("GET", url),
                response=httpx.Response(503),
            )
        async with self._get_sem():
            # Re-check the breaker AFTER queuing behind the semaphore. Cuts
            # the 30-hit 403 burst down to at most 1 hit per burst.
            if self._is_blocked():
                raise httpx.HTTPStatusError(
                    f"RA circuit breaker open until {self._blocked_until.isoformat()}",
                    request=httpx.Request("GET", url),
                    response=httpx.Response(503),
                )
            # 0.6–1.2s jittered pause between requests. Was 0.3s — that was too
            # tight; combined with Semaphore(3) it produced ~10 req/s sustained.
            await asyncio.sleep(0.6 + random.random() * 0.6)
            headers = _build_headers(referer=referer)
            fetch_url = _proxied(url)
            if _RA_PROXY_ACTIVE and fetch_url != url:
                # Proxy auth header (validated by the droplet proxy before
                # forwarding to RA). Doesn't replace UA/Referer — those still
                # get applied at the proxy layer when it talks to RA.
                headers["X-Proxy-Secret"] = _RA_PROXY_SECRET
                if referer:
                    headers["X-Proxy-Referer"] = referer
            async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(fetch_url)
                if resp.status_code == 403:
                    self._trip_breaker()
                    resp.raise_for_status()
                resp.raise_for_status()
                # Successful response → reset breaker counter (block_count was
                # for *consecutive* 403s).
                self._block_count = 0
                return resp.text

    async def _get(self, url: str) -> str:
        return await self._request(url, timeout=20.0)

    async def _get_form(self, url: str) -> str:
        # Form pages are inner pages — supply a Referer so the traffic looks
        # like a normal navigation from the calendar/meeting page.
        return await self._request(url, referer=f"{_BASE}/Calendar.aspx", timeout=10.0)

    # ── InteractiveForm fetchers ──────────────────────────────────────────────

    async def _fetch_horse_form(self, horsecode: str, raceentry: str) -> dict:
        if not horsecode:
            return {}
        cached = self._horse_form_cache.get(horsecode)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 21600:
            return cached[1]
        url = f"{_IF_BASE}/HorseFullForm.aspx?horsecode={horsecode}&src=horseform&raceentry={raceentry}"
        try:
            html = await self._get_form(url)
            data = _parse_horse_form_page(html)
        except Exception as e:
            log.debug("Horse form fetch failed %s: %s", horsecode, e)
            data = {}
        self._horse_form_cache[horsecode] = (datetime.utcnow(), data)
        return data

    async def _fetch_person_form(self, code: str, kind: str) -> dict:
        if not code:
            return {}
        cache = self._jockey_form_cache if kind == "jockey" else self._trainer_form_cache
        cached = cache.get(code)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 21600:
            return cached[1]
        if kind == "jockey":
            url = f"{_IF_BASE}/JockeyLastRuns.aspx?jockeycode={code}"
        else:
            url = f"{_IF_BASE}/TrainerLastRuns.aspx?trainercode={code}"
        try:
            html = await self._get_form(url)
            data = _parse_person_form_page(html)
        except Exception as e:
            log.debug("Person form fetch failed %s %s: %s", kind, code, e)
            data = {}
        cache[code] = (datetime.utcnow(), data)
        return data

    # NOTE (2026-07-18): the DB-persistence variant of _fetch_horse_form /
    # _fetch_person_form (commit 2982a72) was reverted here because opening
    # a fresh get_session() per fetch exhausted the asyncpg pool during
    # _batch_fetch_runner_forms bursts (30+ concurrent form fetches × 2
    # sessions each = 60 concurrent connects, TimeoutError). Re-add later
    # via a single batched session opened at the _batch_fetch level so we
    # do 1 session for N codes instead of 2N sessions.

    async def _batch_fetch_runner_forms(self, selections: list[dict]) -> dict:
        """Fetch horse/jockey/trainer forms for all selections in parallel."""
        task_keys: list[str] = []
        coros: list = []
        seen: set[str] = set()

        for sel in selections:
            horsecode = sel.get("horsecode", "")
            raceentry = sel.get("raceentry", "")
            jockeycode = sel.get("jockeycode", "")
            trainercode = sel.get("trainercode", "")

            key = f"h:{horsecode}"
            if horsecode and key not in seen:
                seen.add(key)
                task_keys.append(key)
                coros.append(self._fetch_horse_form(horsecode, raceentry))

            key = f"j:{jockeycode}"
            if jockeycode and key not in seen:
                seen.add(key)
                task_keys.append(key)
                coros.append(self._fetch_person_form(jockeycode, "jockey"))

            key = f"t:{trainercode}"
            if trainercode and key not in seen:
                seen.add(key)
                task_keys.append(key)
                coros.append(self._fetch_person_form(trainercode, "trainer"))

        if not coros:
            return {}

        results = await asyncio.gather(*coros, return_exceptions=True)
        form_map: dict[str, dict] = {}
        for key, result in zip(task_keys, results):
            form_map[key] = result if isinstance(result, dict) else {}

        log.debug("Batch form fetch: %d requests for %d selections", len(coros), len(selections))
        return form_map

    # ── Calendar ──────────────────────────────────────────────────────────────

    async def _fetch_state_calendar(self, state: str, race_date: str) -> list[dict]:
        cache_key = f"{race_date}:{state}"
        # Fast-path cache check (no lock needed for reads).
        cached = self._calendar_cache.get(cache_key)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 21600:
            return cached[1]

        # Thundering-herd guard: hold a per-cache-key lock around the
        # miss-path (DB read + RA fetch + cache write). When N concurrent
        # callers miss the cache, only one issues the RA hit; the others
        # wait for the lock and pick up the freshly-cached value.
        lock = self._calendar_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            # Re-check in-memory cache — another caller may have populated it.
            cached = self._calendar_cache.get(cache_key)
            if cached and (datetime.utcnow() - cached[0]).total_seconds() < 21600:
                return cached[1]

            # Persistent DB cache — survives Railway redeploys.
            db_hit = await _load_calendar_from_db(race_date, state)
            if db_hit is not None:
                meetings, slug_kvs = db_hit
                self._calendar_cache[cache_key] = (datetime.utcnow(), meetings)
                self._slug_to_key.update(slug_kvs)
                return meetings

            # Cache miss + no DB row — fall through to the actual RA fetch,
            # still inside the lock so concurrent callers don't all hit RA.
            ra_date = _ra_date(race_date)
            # Retry-with-backoff on 503 (RA transient overload). Without this
            # a single 503 burst during the 8:30am cron silently kills the
            # whole day's enrichment for that state. Three attempts: 2s, 5s,
            # 15s waits. Other status codes (e.g. 403 trip the breaker, 404
            # is a real "no such page") aren't retried.
            html = None
            for attempt, wait in enumerate((2, 5, 15), start=1):
                try:
                    html = await self._get(f"{_BASE}/Calendar.aspx?State={state}")
                    break
                except httpx.HTTPStatusError as e:
                    sc = e.response.status_code if e.response is not None else None
                    if sc == 503 and attempt < 3:
                        log.warning(
                            "Calendar 503 for %s (attempt %d/3) — sleeping %ds then retrying",
                            state, attempt, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    log.warning("Calendar fetch failed for %s: %s", state, e)
                    # Cache the failure for 5 minutes so we don't death-spiral
                    # the proxy when it's returning 503 (cap hit). Empty list
                    # is a valid 'no meetings today' so we serve that until
                    # the cap window rolls.
                    self._calendar_cache[cache_key] = (datetime.utcnow() - timedelta(seconds=3300), [])
                    return []
                except Exception as e:
                    log.warning("Calendar fetch failed for %s: %s", state, e)
                    self._calendar_cache[cache_key] = (datetime.utcnow() - timedelta(seconds=3300), [])
                    return []
            if html is None:
                # Exhausted retries on 503
                log.warning("Calendar fetch failed for %s after 3 attempts (503)", state)
                self._calendar_cache[cache_key] = (datetime.utcnow() - timedelta(seconds=3300), [])
                return []

            soup = BeautifulSoup(html, "html.parser")
            meetings = []
            from urllib.parse import unquote
            for link in soup.find_all("a", href=re.compile(r"(Acceptances|Results\.aspx)")):
                href = link.get("href", "")
                m = re.search(r"Key=([^&\"]+)", href)
                if not m:
                    continue
                ra_key = unquote(m.group(1))
                if not ra_key.startswith(ra_date):
                    continue
                parts = ra_key.split(",", 2)
                if len(parts) < 3:
                    continue
                raw_venue = parts[2]
                if re.search(r"\b(Trial|Trail|Trials|TRL|Jumpout|Jump\s*Out)\b", raw_venue, re.IGNORECASE):
                    continue
                venue = _clean_venue(raw_venue)
                slug = _make_slug(raw_venue, race_date)
                self._slug_to_key[slug] = ra_key
                self._slug_to_key[f"{race_date}:{state}:{venue}"] = ra_key
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
            date_compact = race_date.replace("-", "")
            slug_subset = {
                k: v for k, v in self._slug_to_key.items()
                if k.endswith(f"-{date_compact}") or k.startswith(f"{race_date}:{state}:")
            }
            await _persist_calendar_to_db(race_date, state, meetings, slug_subset)
            return meetings

    # ── Acceptances ───────────────────────────────────────────────────────────

    async def _fetch_meeting(self, ra_key: str, race_date: str, state: str, force_fresh: bool = False) -> dict | None:
        cached = self._meeting_cache.get(ra_key)
        # 30-min TTL. Was 15min, which on a heavy-deploy day burned the
        # droplet's 5000/24h cap by 21:30 UTC. Acceptances change rarely
        # in the morning (scratchings land late) so a 30-min TTL halves
        # our Acceptances volume without meaningful freshness loss; the
        # scratch-detection cron still catches scratchings within one
        # 15-min cron tick + at most one stale window.
        # force_fresh=True bypasses the cache read but STILL writes the
        # fresh result back, so downstream user endpoints hit the warm
        # cache immediately after. Used by the scratch sweep — needs
        # fresh RA data to detect scratchings but shouldn't cold-strip
        # the cache for the /api/meetings/{date}/{venue} path (which
        # was paying 25s per venue click after every 15-min sweep tick).
        if not force_fresh and cached and (datetime.utcnow() - cached[0]).total_seconds() < 1800:
            return cached[1]
        from urllib.parse import quote
        url = f"{_BASE}/Acceptances.aspx?Key={quote(ra_key, safe='')}"
        try:
            html = await self._get(url)
        except Exception as e:
            log.warning("Acceptances fetch failed for %s: %s", ra_key, e)
            # Cache None for 5 min so a 503-storm doesn't dogpile RA.
            # 30-min TTL minus 25 min = 5 min remaining.
            self._meeting_cache[ra_key] = (datetime.utcnow() - timedelta(seconds=1500), None)
            return None
        parsed = _parse_acceptances_page(html, ra_key, race_date, state)
        self._meeting_cache[ra_key] = (datetime.utcnow(), parsed)
        return parsed

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_meetings(self, race_date: str | None = None) -> list[dict]:
        d = race_date or date.today().isoformat()
        # Sequential per-state fetch with 3s stagger. Was parallel via
        # asyncio.gather — fired 8 Calendar.aspx requests within ~1s,
        # which is what triggered RA's 503-throttling during the 8:30am
        # cron on 2026-06-25. Staggering trades ~24s of cron runtime for
        # massively higher reliability. Cache hits and DB hits return
        # instantly inside _fetch_state_calendar, so most days the stagger
        # only matters on a true cold-cache enrich.
        meetings: list[dict] = []
        seen_slugs: set[str] = set()
        for i, s in enumerate(_AU_STATES):
            try:
                r = await self._fetch_state_calendar(s, d)
                if isinstance(r, list):
                    for m in r:
                        slug = m.get("slug", "")
                        if slug and slug in seen_slugs:
                            continue
                        seen_slugs.add(slug)
                        meetings.append(m)
            except Exception as e:
                log.warning("get_meetings: %s failed: %s", s, e)
            # Stagger between cold-cache RA hits — skip sleep on the last
            # state to avoid wasted wait, and skip on cache hits (the
            # underlying _request already has a 0.6-1.2s jitter, but that
            # only delays in-flight requests, not the spacing between them).
            if i < len(_AU_STATES) - 1:
                await asyncio.sleep(3)
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

    async def get_meeting_races(self, slug: str, force_fresh: bool = False) -> list[dict]:
        ra_key = self._slug_to_key.get(slug)
        if not ra_key:
            await self.get_meeting_by_slug(slug)
            ra_key = self._slug_to_key.get(slug)
        if not ra_key:
            return []
        parts = ra_key.split(",", 2)
        state = parts[1] if len(parts) > 1 else "NSW"
        race_date = self._ra_key_to_date(ra_key)
        meeting = await self._fetch_meeting(ra_key, race_date, state, force_fresh=force_fresh)
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
        Cached 6 hours — once a race is published, the result is final.
        Was 30 min; on a Sat metro day with 26 seed-cron ticks × 30+
        venues that meant 600+ Results.aspx fetches/day just for seeding.
        6h means at most 4 fetches per meeting per day, total under 200.
        Per-key lock prevents thundering herd from concurrent settlers.
        """
        cached = self._results_cache.get(ra_key)
        if cached and (datetime.utcnow() - cached[0]).total_seconds() < 21600:
            return cached[1]
        lock = self._results_locks.setdefault(ra_key, asyncio.Lock())
        async with lock:
            # Re-check after acquiring — another caller may have populated.
            cached = self._results_cache.get(ra_key)
            if cached and (datetime.utcnow() - cached[0]).total_seconds() < 21600:
                return cached[1]
            from urllib.parse import quote
            # _BASE already ends in "/FreeFields" — do NOT prepend a second
            # /FreeFields/. A previous edit (2026-07-10) added one under the
            # false belief that RA had moved the endpoint. The site never
            # moved; the double-nested path just 404s. Verified via curl:
            #   /FreeFields/Results.aspx        → 200 (~28KB)
            #   /FreeFields/FreeFields/Results  → 404
            url = f"{_BASE}/Results.aspx?Key={quote(ra_key, safe='')}"
            try:
                html = await self._get(url)
            except Exception as e:
                log.warning("RA results fetch failed for %s: %s", ra_key, e)
                # Cache the failure for 5 min — backdate the TTL so that
                # the proxy 503-loop doesn't dogpile RA while the cap clears.
                self._results_cache[ra_key] = (datetime.utcnow() - timedelta(seconds=21300), {})
                return {}
            parsed = _parse_results_page(html)
            self._results_cache[ra_key] = (datetime.utcnow(), parsed)
            total_runners = sum(len(r["runners"]) for r in parsed.values())
            log.debug("RA results: %d races, %d runners for %s", len(parsed), total_runners, ra_key)
        return parsed

    _SPONSOR_PREFIXES = ["TAB ", "Sportsbet ", "Ladbrokes ", "Palmerbet ", "Neds ", "Ubet "]

    async def _ensure_hydrated(self) -> None:
        """Populate _slug_to_key from the DB once per process lifetime.

        Structure chosen after 2026-07-18 postmortem:
          - ONE bulk read at first use — cheap on a table of ~10k rows/yr
            (venue-key mapping is (date, state, venue) tuple, ~30 venues/day)
          - Lock-protected so concurrent first callers don't stampede
          - Failure is non-fatal — hot path falls back to sponsor fanout

        Contrast the previous per-request DB read (commit 7c1fce9) which
        opened a fresh session inside every find_results and, under seed-
        cron's parallel-venue fanout, exhausted the asyncpg pool.
        """
        if self._db_hydrated:
            return
        async with self._hydrate_lock:
            if self._db_hydrated:  # re-check inside lock
                return
            try:
                from horse_engine.models.database import (
                    get_session, RAVenueKeyCacheRow,
                )
                from sqlalchemy import select as _select
                async with get_session() as _s:
                    rows = (await _s.execute(_select(RAVenueKeyCacheRow))).scalars().all()
                for r in rows:
                    ck = f"{r.race_date}:{r.state}:{r.clean_venue}"
                    self._slug_to_key[ck] = r.ra_key
                log.info("[RA] Hydrated %d venue keys from DB", len(rows))
            except Exception as e:
                log.warning("[RA] DB hydrate skipped: %s", e)
            finally:
                # Mark hydrated even on failure — one failed attempt is
                # enough; retrying on every call would defeat the point.
                self._db_hydrated = True

    def _schedule_persist_venue_key(
        self, race_date: str, state: str, clean_venue: str, ra_key: str
    ) -> None:
        """Fire-and-forget write to ra_venue_key_cache. Uses asyncio.create_task
        so the DB round-trip does NOT block the caller (find_results returns
        immediately with results from RA). Writes are rare — only when a new
        (date, state, venue) resolves — so the task queue never piles up.
        """
        async def _bg():
            try:
                from horse_engine.models.database import (
                    get_session, save_ra_venue_key,
                )
                async with get_session() as _s:
                    await save_ra_venue_key(_s, race_date, state, clean_venue, ra_key)
            except Exception as e:
                log.debug("[RA] bg venue-key persist skipped: %s", e)
        try:
            asyncio.create_task(_bg())
        except Exception:
            # No running loop — quietly skip. Should never happen in the
            # async server context but guard is cheap.
            pass

    async def find_results(self, race_date: str, state: str, clean_venue: str) -> tuple[str, dict[int, dict]]:
        """
        Try to find RA results for a venue whose stored name has had the sponsor prefix stripped.

        Lookup order:
          1. DB-hydrated RAM cache (_slug_to_key), populated once per process
             from ra_venue_key_cache. Survives Railway redeploys.
          2. RAM cache populated by Calendar.aspx during this process's
             enrichment cycle (same dict, different source).
          3. Plain unprefixed key.
          4. Sponsor-prefix fanout.

        Successful resolutions from 3+4 are written back to the DB via a
        fire-and-forget task so a future container's step-1 hydrate picks
        them up.
        """
        await self._ensure_hydrated()
        ra_date_str = _ra_date(race_date)

        # Layers 1 + 2 — DB-hydrated + Calendar.aspx RAM cache share the dict.
        cache_key = f"{race_date}:{state}:{clean_venue}"
        if cache_key in self._slug_to_key:
            ra_key = self._slug_to_key[cache_key]
            results = await self.get_results(ra_key)
            if results:
                return ra_key, results

        # 3. Try cleaned name directly.
        base_key = f"{ra_date_str},{state},{clean_venue}"
        results = await self.get_results(base_key)
        if results:
            self._slug_to_key[cache_key] = base_key
            self._schedule_persist_venue_key(race_date, state, clean_venue, base_key)
            return base_key, results

        # 4. Sponsor-prefix fanout — last resort.
        for prefix in self._SPONSOR_PREFIXES:
            ra_key = f"{ra_date_str},{state},{prefix}{clean_venue}"
            results = await self.get_results(ra_key)
            if results:
                log.info("RA results found with prefix '%s' for %s/%s", prefix, state, clean_venue)
                self._slug_to_key[cache_key] = ra_key
                self._schedule_persist_venue_key(race_date, state, clean_venue, ra_key)
                return ra_key, results

        log.warning("RA results not found for %s/%s/%s (tried %d key variants)",
                    race_date, state, clean_venue, 2 + len(self._SPONSOR_PREFIXES))
        return "", {}

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

        selections = raw_event.get("selections") or []
        form_map = await self._batch_fetch_runner_forms(selections)

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
            runners=self._parse_runners(selections, track_condition, form_map),
        )

    def _parse_runners(
        self,
        selections: list[dict],
        track_condition: str = "Good 4",
        form_map: dict | None = None,
    ) -> list[Runner]:
        runners = []
        for sel in selections:
            if (sel.get("status") or "").upper() == "SCRATCHED":
                continue
            try:
                horsecode = sel.get("horsecode", "")
                jockeycode = sel.get("jockeycode", "")
                trainercode = sel.get("trainercode", "")
                horse_form = (form_map or {}).get(f"h:{horsecode}", {})
                jockey_form = (form_map or {}).get(f"j:{jockeycode}", {})
                trainer_form = (form_map or {}).get(f"t:{trainercode}", {})
                r = self._parse_runner(sel, track_condition, horse_form, jockey_form, trainer_form)
                if r:
                    runners.append(r)
            except Exception as e:
                log.debug("Runner parse error: %s", e)
        return runners

    def _parse_runner(
        self,
        sel: dict,
        track_condition: str = "Good 4",
        horse_form: dict | None = None,
        jockey_form: dict | None = None,
        trainer_form: dict | None = None,
    ) -> Runner | None:
        comp = sel.get("competitor") or {}
        jock = sel.get("jockey") or {}
        trnr = sel.get("trainer") or {}

        horse_name = comp.get("name", "")
        if not horse_name:
            return None

        hf_pre = horse_form or {}
        # Backfill pedigree/identity fields from the form page if acceptances page left them empty
        if not comp.get("sire") and hf_pre.get("sire"):
            comp["sire"] = hf_pre["sire"]
        if not comp.get("dam") and hf_pre.get("dam"):
            comp["dam"] = hf_pre["dam"]
        if not comp.get("age") and hf_pre.get("age"):
            comp["age"] = hf_pre["age"]
        if not comp.get("sex") and hf_pre.get("sex"):
            comp["sex"] = hf_pre["sex"]
        if not comp.get("colour") and hf_pre.get("colour"):
            comp["colour"] = hf_pre["colour"]

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

        hf = hf_pre
        jf = jockey_form or {}
        tf = trainer_form or {}

        # Use scraped form history if available, else fall back to form string
        form_starts_raw: list[FormStart] | None = hf.get("form_starts")
        last_10: list[FormStart] = form_starts_raw[:10] if form_starts_raw else _parse_form_string(
            sel.get("form_string") or ""
        )

        # Map today's going to a condition category for condition_starts lookup
        cond = _going_category(track_condition)
        condition_starts = hf.get(f"{cond}_starts", 0)
        condition_wins = hf.get(f"{cond}_wins", 0)

        j_name = jock.get("name") or ""
        t_name = trnr.get("name") or ""

        jockey_rate = jf.get("win_rate", 10.0)
        trainer_rate = tf.get("win_rate", 10.0)
        trainer_wet_rate = tf.get("wet_rate", trainer_rate)

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
            career_starts=hf.get("career_starts", 0),
            career_wins=hf.get("career_wins", 0),
            career_places=hf.get("career_places", 0),
            track_starts=hf.get("track_starts", 0),
            track_wins=hf.get("track_wins", 0),
            distance_starts=hf.get("dist_starts", 0),
            distance_wins=hf.get("dist_wins", 0),
            track_distance_starts=hf.get("track_dist_starts", 0),
            track_distance_wins=hf.get("track_dist_wins", 0),
            condition_starts=condition_starts,
            condition_wins=condition_wins,
            first_up_starts=hf.get("first_up_starts", 0),
            first_up_wins=hf.get("first_up_wins", 0),
            second_up_starts=hf.get("second_up_starts", 0),
            second_up_wins=hf.get("second_up_wins", 0),
            last_10_starts=last_10,
            pedigree=pedigree,
            tote_win_odds=sel.get("topToteWin"),
            best_available_odds=sel.get("topToteWin"),
            jockey_stats=JockeyStats(
                name=j_name,
                win_rate_overall=jockey_rate,
                win_rate_track=jockey_rate,
                win_rate_distance=jockey_rate,
                win_rate_barrier_low=jockey_rate,
                win_rate_barrier_mid=jockey_rate,
                win_rate_barrier_wide=jockey_rate,
                wins_today=0,
                prizemoney_season=0,
                wins_season=0,
                trainer_jockey_combo_rate=jockey_rate,
            ) if j_name else None,
            trainer_stats=TrainerStats(
                name=t_name,
                win_rate_overall=trainer_rate,
                win_rate_track=trainer_rate,
                win_rate_distance=trainer_rate,
                win_rate_first_up=trainer_rate,
                win_rate_second_up=trainer_rate,
                win_rate_wet=trainer_wet_rate,
                prizemoney_season=0,
                runners_season=0,
                wins_season=0,
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
