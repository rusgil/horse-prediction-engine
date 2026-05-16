"""
Punters.com.au data client.

Uses the punters.com.au public GraphQL API (no real auth required)
and the NUXT SSR page to discover today's Australian meetings.

GraphQL endpoint: https://api.punters.com.au/racing
Auth: Authorization: Bearer none
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from horse_engine.models.race import (
    FormStart, PedigreeProfile, Race, Runner,
)
from horse_engine.pedigree.sire_profiles import SIRE_PROFILES

log = logging.getLogger(__name__)

_AU_STATES = {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"}

_GRAPHQL = "https://api.punters.com.au/racing"
_FORM_GUIDE_INDEX = "https://www.punters.com.au/form-guide/horses/"
_PUNTERS_HOME = "https://www.punters.com.au/"

_HEADERS_GQL = {
    "Authorization": "Bearer none",
    "Content-Type": "application/json",
    "Origin": "https://www.punters.com.au",
    "Referer": "https://www.punters.com.au/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_HEADERS_HTML = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}

# Full race card query — one network call per meeting
_MEETING_FULL_QUERY = """
{
  meeting(id: $MEETING_ID) {
    id
    name
    slug
    meetingDateLocal
    railPosition
    venue { name state }
    events {
      id
      eventNumber
      name
      distance
      startTime
      endTime
      status
      raceType
      eventClass
      trackCondition { overall rating }
      selections {
        id
        competitorNumber
        barrierNumber
        weight
        weightUnit
        jockeyWeight
        gearChanges
        topToteWin
        topTotePlace
        startingPrice
        selectionResult
        officialMargin
        officialTime
        status
        competitor {
          id
          name
          slug
          sire
          dam
          country
          age
          colour
          sex
          forms {
            finishPosition
            isDistance
            isTrack
            meetingDate
            meetingName
            weight
            jockey { name }
            prizeMoney
            winnerName
          }
        }
        jockey { id name slug apprentice }
        trainer { id name slug }
        lastRun {
          finishPosition
          weight
          eventDistance
          trackCondition
          meetingDate
          meetingName
        }
      }
    }
  }
}
"""

_MEETINGS_BY_DATE_QUERY = """
{
  meetings(sport: HorseRacing, startDate: "$DATE", endDate: "$DATE") {
    id
    name
    slug
    meetingDateLocal
    railPosition
    venue { name state }
  }
}
"""

_MEETING_SLUG_QUERY = """
{
  meeting(slug: "$SLUG", sport: HorseRacing) {
    id
    name
    slug
    meetingDateLocal
    railPosition
    venue { name state }
  }
}
"""


def _today() -> str:
    return date.today().isoformat()


class PuntersClient:
    def __init__(self):
        self._cookie_jar: dict[str, str] = {}

    # ── Internal helpers ────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _gql(self, query: str) -> dict:
        async with httpx.AsyncClient(headers=_HEADERS_GQL, timeout=20.0) as client:
            resp = await client.post(_GRAPHQL, json={"query": query})
            resp.raise_for_status()
            return resp.json()

    async def _ensure_cookies(self, client: httpx.AsyncClient) -> None:
        """Hit homepage to establish session cookies (required for NUXT data)."""
        if not self._cookie_jar:
            r = await client.get(_PUNTERS_HOME, follow_redirects=True)
            self._cookie_jar = dict(r.cookies)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _get_html(self, url: str) -> str:
        async with httpx.AsyncClient(
            headers=_HEADERS_HTML, cookies=self._cookie_jar, timeout=25.0
        ) as client:
            await self._ensure_cookies(client)
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.text

    # ── NUXT data decoder ────────────────────────────────────────────────

    @staticmethod
    def _decode_nuxt(html: str) -> list:
        m = re.search(r'<script[^>]+id="__NUXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if not m:
            return []
        return json.loads(m.group(1))

    @staticmethod
    def _resolve(D: list, ref: Any, depth: int = 0) -> Any:
        if depth > 12:
            return None
        if isinstance(ref, int) and 0 <= ref < len(D):
            val = D[ref]
            if isinstance(val, dict):
                return {k: PuntersClient._resolve(D, v, depth + 1) for k, v in val.items() if k != "__typename"}
            elif isinstance(val, list):
                return [PuntersClient._resolve(D, i, depth + 1) for i in val]
            return val
        return ref

    # ── Public API ───────────────────────────────────────────────────────

    async def get_meetings(self, race_date: str | None = None) -> list[dict]:
        """
        Return list of Australian thoroughbred meetings for race_date via GraphQL.
        Each dict: {id, name, slug, venue, state, rail_position, date}
        """
        d = race_date or _today()
        query = _MEETINGS_BY_DATE_QUERY.replace("$DATE", d)
        try:
            data = await self._gql(query)
        except Exception as e:
            log.warning("get_meetings GraphQL failed: %s", e)
            return []

        all_meetings = (data.get("data") or {}).get("meetings") or []
        meetings = []
        for m in all_meetings:
            venue = m.get("venue") or {}
            state = venue.get("state", "")
            if state not in _AU_STATES:
                continue
            slug = m.get("slug", "")
            if slug.endswith("-bt"):  # exclude barrier trials
                continue
            meetings.append({
                "id": str(m.get("id") or ""),
                "name": m.get("name", ""),
                "slug": slug,
                "venue": m.get("name", ""),
                "state": state,
                "rail_position": m.get("railPosition") or "",
                "date": d,
            })

        log.info("Found %d Australian meetings on %s", len(meetings), d)
        return meetings

    async def get_meeting_by_slug(self, slug: str) -> dict | None:
        """Resolve a meeting slug to its numeric ID and basic metadata."""
        query = _MEETING_SLUG_QUERY.replace("$SLUG", slug)
        try:
            data = await self._gql(query)
            return data.get("data", {}).get("meeting")
        except Exception as e:
            log.warning("get_meeting_by_slug failed for %s: %s", slug, e)
            return None

    async def get_meeting_races(self, slug: str) -> list[dict]:
        """Return all race events for a meeting slug (lightweight — no runners)."""
        meeting = await self.get_meeting_by_slug(slug)
        if not meeting:
            return []
        # Fetch full detail by id
        full = await self._fetch_meeting_full(meeting["id"])
        if not full:
            return []
        return full.get("events", [])

    async def _fetch_meeting_full(self, meeting_id: str) -> dict | None:
        query = _MEETING_FULL_QUERY.replace("$MEETING_ID", f'"{meeting_id}"')
        try:
            data = await self._gql(query)
            return data.get("data", {}).get("meeting")
        except Exception as e:
            log.warning("_fetch_meeting_full failed for id=%s: %s", meeting_id, e)
            return None

    async def get_race(self, slug: str, race_number: int) -> dict | None:
        """
        Return the raw event dict for a specific race number within a meeting.
        Includes full selection data (jockey, trainer, weight, pedigree, odds).
        """
        meeting = await self.get_meeting_by_slug(slug)
        if not meeting:
            return None
        full = await self._fetch_meeting_full(meeting["id"])
        if not full:
            return None
        event = next((e for e in full.get("events", []) if e.get("eventNumber") == race_number), None)
        if not event:
            log.warning("Race %d not found in meeting %s", race_number, slug)
        # Attach meeting-level data needed for parse_race
        event["_meeting"] = full
        return event

    # ── Parsing ──────────────────────────────────────────────────────────

    def parse_race(self, raw_event: dict, race_date: str, venue: str, state: str) -> Race:
        meeting = raw_event.get("_meeting", {})
        race_num = raw_event.get("eventNumber", 0)
        meeting_slug = meeting.get("slug") or ""
        # meeting slug is like "werribee-20260514" — strip the date suffix
        date_suffix = f"-{race_date.replace('-', '')}"
        venue_slug = meeting_slug.replace(date_suffix, "") if date_suffix in meeting_slug else meeting_slug

        tc = raw_event.get("trackCondition") or {}
        tc_overall = tc.get("overall", "Good")
        tc_rating = tc.get("rating", "4")
        track_condition = f"{tc_overall} {tc_rating}".strip()

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
            runners=self._parse_runners(raw_event.get("selections", [])),
        )

    def _parse_runners(self, selections: list[dict]) -> list[Runner]:
        runners = []
        for sel in selections:
            if (sel.get("status") or "").upper() == "SCRATCHED":
                continue
            if sel.get("selectionResult") == -1:
                continue
            try:
                runner = self._parse_runner(sel)
                if runner:
                    runners.append(runner)
            except Exception as e:
                log.debug("Runner parse error: %s", e)
        return runners

    def _parse_runner(self, sel: dict) -> Runner | None:
        comp = sel.get("competitor") or {}
        jock = sel.get("jockey") or {}
        trnr = sel.get("trainer") or {}

        horse_name = comp.get("name", "Unknown")
        if not horse_name or horse_name == "Unknown":
            return None

        sire = comp.get("sire") or ""
        dam = comp.get("dam") or ""

        # Pedigree enrichment from sire database
        profile = SIRE_PROFILES.get(sire, {})
        pedigree = PedigreeProfile(
            sire=sire,
            dam=dam,
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

        tote_win = sel.get("topToteWin")
        tote_place = sel.get("topTotePlace")
        sp = sel.get("startingPrice")

        best_odds = None
        if tote_win:
            best_odds = float(tote_win)
        elif sp:
            best_odds = float(sp)

        last_10_starts = self._parse_forms(
            comp.get("forms") or [],
            sel.get("lastRun"),
        )

        return Runner(
            barrier=int(sel.get("barrierNumber") or 0),
            tab_number=int(sel.get("competitorNumber") or 0),
            horse_name=horse_name,
            age=int(comp.get("age") or 0),
            sex=comp.get("sex") or "",
            colour=comp.get("colour") or "",
            weight=float(sel.get("weight") or 0),
            jockey=jock.get("name") or "",
            trainer=trnr.get("name") or "",
            country=comp.get("country") or "AUS",
            career_starts=0,
            career_wins=0,
            career_places=0,
            last_10_starts=last_10_starts,
            pedigree=pedigree,
            tote_win_odds=float(tote_win) if tote_win else None,
            tote_place_odds=float(tote_place) if tote_place else None,
            fixed_win_odds=float(sp) if sp else None,
            best_available_odds=best_odds,
        )

    @staticmethod
    def _parse_forms(forms: list[dict], last_run: dict | None) -> list[FormStart]:
        """
        Build up to 10 FormStart entries from the competitor's full forms history.
        forms gives career depth; last_run supplements the most recent start with
        exact distance and track condition (not available in forms).
        """
        # Index lastRun by date for enrichment
        lr_date = ""
        lr_distance = 0
        lr_condition = "Good"
        if last_run:
            lr_date = (last_run.get("meetingDate") or "")[:10]
            lr_distance = int(last_run.get("eventDistance") or 0)
            lr_condition = last_run.get("trackCondition") or "Good"

        # Sort forms by date descending, take last 10
        valid = []
        for f in forms:
            date_raw = (f.get("meetingDate") or "")[:10]
            pos = f.get("finishPosition")
            if not pos or int(pos) <= 0 or not date_raw:
                continue
            valid.append((date_raw, f))
        valid.sort(key=lambda x: x[0], reverse=True)

        starts = []
        for date_str, f in valid[:10]:
            try:
                pos = int(f.get("finishPosition") or 0)
                # Supplement most recent start with lastRun distance/condition
                distance = lr_distance if date_str == lr_date else 0
                condition = lr_condition if date_str == lr_date else "Good"
                starts.append(FormStart(
                    date=date_str,
                    track=f.get("meetingName") or "",
                    distance=distance,
                    track_condition=condition,
                    barrier=0,
                    weight=float(f.get("weight") or 0),
                    jockey=(f.get("jockey") or {}).get("name", ""),
                    position=pos,
                    finishers=max(pos, 10),
                    beaten_margin=0.0,
                    race_class="",
                    prize_money=int(f.get("prizeMoney") or 0),
                ))
            except Exception:
                continue

        # Fall back to lastRun alone if forms was empty
        if not starts and last_run:
            try:
                pos = int(last_run.get("finishPosition") or 0)
                if pos > 0:
                    starts.append(FormStart(
                        date=lr_date,
                        track=last_run.get("meetingName") or "",
                        distance=lr_distance,
                        track_condition=lr_condition,
                        barrier=0,
                        weight=float(last_run.get("weight") or 0),
                        jockey="",
                        position=pos,
                        finishers=max(pos, 10),
                        beaten_margin=0.0,
                        race_class="",
                        prize_money=0,
                    ))
            except Exception:
                pass

        return starts
