"""
TAB Australia public API client.

All endpoints are unauthenticated (read-only public data).
Base: https://api.tab.com.au/v1/tab-info-service
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from horse_engine.config import settings
from horse_engine.models.race import (
    FormStart, JockeyStats, Meeting, PedigreeProfile, Race, Runner, TrainerStats
)

log = logging.getLogger(__name__)

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; FunkyIQ/1.0)",
}
TIMEOUT = 20.0


def _today() -> str:
    return date.today().isoformat()


class TABClient:
    def __init__(self, jurisdiction: str | None = None):
        self.base = settings.tab_base_url
        self.jurisdiction = jurisdiction or settings.tab_jurisdiction

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base}{path}"
        p = {"jurisdiction": self.jurisdiction, **(params or {})}
        async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT) as client:
            resp = await client.get(url, params=p)
            resp.raise_for_status()
            return resp.json()

    async def get_meetings(self, race_date: str | None = None) -> list[dict]:
        """Return list of thoroughbred meetings for the given date."""
        d = race_date or _today()
        data = await self._get(f"/racing/dates/{d}/meetings")
        meetings = data.get("meetings", [])
        # Filter to thoroughbreds only (raceType == "R")
        return [m for m in meetings if m.get("meetingCode") and m.get("raceType") == "R"]

    async def get_race(self, race_date: str, venue_code: str, race_number: int) -> dict | None:
        """Return full race card including runners, form, and current odds."""
        d = race_date or _today()
        try:
            data = await self._get(
                f"/racing/dates/{d}/meetings/{venue_code}/R/races/{race_number}"
            )
            return data
        except httpx.HTTPStatusError as e:
            log.warning("TAB race fetch failed %s/%s R%s: %s", venue_code, d, race_number, e)
            return None

    async def get_meeting_races(self, race_date: str, venue_code: str) -> list[dict]:
        """Return all races at a meeting."""
        d = race_date or _today()
        try:
            data = await self._get(f"/racing/dates/{d}/meetings/{venue_code}/R/races")
            return data.get("races", [])
        except httpx.HTTPStatusError:
            return []

    # ── Parsing helpers ────────────────────────────────────────────────────

    def parse_meeting(self, raw: dict, race_date: str) -> Meeting:
        venue = raw.get("venueName", "Unknown")
        venue_code = raw.get("meetingCode", "")
        state = raw.get("location", {}).get("state", "")
        track_condition = raw.get("trackCondition", {}).get("name", "Good 4")
        return Meeting(
            date=race_date,
            venue=venue,
            state=state,
            track_condition=track_condition,
        )

    def parse_race(self, raw: dict, race_date: str, venue: str, state: str) -> Race:
        race_num = raw.get("raceNumber", 0)
        venue_code = raw.get("meetingCode", venue)
        return Race(
            race_id=f"{race_date}_{venue_code}_R{race_num}",
            date=race_date,
            venue=venue,
            state=state,
            race_number=race_num,
            race_name=raw.get("raceName", ""),
            race_class=raw.get("raceClassConditions", ""),
            distance=raw.get("raceDistance", 0),
            track_condition=raw.get("trackCondition", {}).get("name", "Good 4"),
            rail_position=raw.get("railPosition", ""),
            prize_money=int(raw.get("prizeMoney", 0) or 0),
            scheduled_time=raw.get("raceStartTime", ""),
            race_type="R",
            runners=self._parse_runners(raw.get("runners", [])),
        )

    def _parse_runners(self, raw_runners: list[dict]) -> list[Runner]:
        runners = []
        for r in raw_runners:
            if r.get("scratched"):
                continue
            try:
                runner = self._parse_runner(r)
                runners.append(runner)
            except Exception as e:
                log.debug("Skip runner parse error: %s", e)
        return runners

    def _parse_runner(self, r: dict) -> Runner:
        # TAB form is an array of last-10 starts
        form_raw = r.get("form", {}).get("recentStarts", [])
        last_10 = [self._parse_form_start(s) for s in form_raw[:10] if s]

        trainer_raw = r.get("trainer", {})
        jockey_raw = r.get("jockey", {})

        # Trainer stats embedded in TAB response
        trainer_stats = TrainerStats(
            name=trainer_raw.get("fullName", "Unknown"),
            win_rate_overall=float(trainer_raw.get("winPercentage", 0) or 0),
            win_rate_track=float(trainer_raw.get("trackWinPercentage", 0) or 0),
            win_rate_distance=float(trainer_raw.get("distanceWinPercentage", 0) or 0),
            win_rate_first_up=float(trainer_raw.get("firstUpWinPercentage", 0) or 0),
            win_rate_second_up=float(trainer_raw.get("secondUpWinPercentage", 0) or 0),
            win_rate_wet=float(trainer_raw.get("wetWinPercentage", 0) or 0),
            prizemoney_season=int(trainer_raw.get("seasonPrizeMoney", 0) or 0),
            runners_season=int(trainer_raw.get("seasonRunners", 0) or 0),
            wins_season=int(trainer_raw.get("seasonWins", 0) or 0),
        ) if trainer_raw else None

        jockey_stats = JockeyStats(
            name=jockey_raw.get("fullName", "Unknown"),
            win_rate_overall=float(jockey_raw.get("winPercentage", 0) or 0),
            win_rate_track=float(jockey_raw.get("trackWinPercentage", 0) or 0),
            win_rate_distance=float(jockey_raw.get("distanceWinPercentage", 0) or 0),
            win_rate_barrier_low=float(jockey_raw.get("barrierLowWinPct", 0) or 0),
            win_rate_barrier_mid=float(jockey_raw.get("barrierMidWinPct", 0) or 0),
            win_rate_barrier_wide=float(jockey_raw.get("barrierWideWinPct", 0) or 0),
            wins_today=int(jockey_raw.get("winsToday", 0) or 0),
            prizemoney_season=int(jockey_raw.get("seasonPrizeMoney", 0) or 0),
            wins_season=int(jockey_raw.get("seasonWins", 0) or 0),
            trainer_jockey_combo_rate=float(jockey_raw.get("trainerComboWinPct", 0) or 0),
        ) if jockey_raw else None

        # Pedigree — TAB provides sire/dam names
        pedigree_raw = r.get("pedigree", {})
        pedigree = PedigreeProfile(
            sire=pedigree_raw.get("sire", "Unknown"),
            dam=pedigree_raw.get("dam", "Unknown"),
            dam_sire=pedigree_raw.get("damSire", "Unknown"),
            distance_aptitude="mile",   # will be enriched by pedigree module
            distance_min=1000,
            distance_max=2400,
            wet_track_score=5.0,
            first_up_score=5.0,
            second_up_score=5.0,
            on_pace_tendency=5.0,
            stamina_index=5.0,
            brilliance_index=5.0,
        ) if pedigree_raw else None

        # Odds
        prices = r.get("prices", [])
        fixed_win = None
        tote_win = None
        for p in prices:
            if p.get("priceType") == "FixedWin":
                fixed_win = float(p.get("winPrice", 0) or 0) or None
            elif p.get("priceType") == "Win":
                tote_win = float(p.get("winPrice", 0) or 0) or None

        gear = r.get("gear", {})
        gear_changes: list[str] = []
        if gear.get("blinkers") == "First Time":
            gear_changes.append("blinkers_on")
        if gear.get("tongueTie") == "First Time":
            gear_changes.append("tongue_tie_on")
        if gear.get("blinkers") == "Off":
            gear_changes.append("blinkers_off")

        career = r.get("form", {}).get("overall", {})

        return Runner(
            barrier=int(r.get("barrier", 0) or 0),
            tab_number=int(r.get("runnerNumber", 0) or 0),
            horse_name=r.get("runnerName", "Unknown"),
            age=int(r.get("age", 0) or 0),
            sex=r.get("sex", ""),
            colour=r.get("colour", ""),
            weight=float(r.get("handicapWeight", 0) or 0),
            jockey=jockey_raw.get("fullName", "Unknown"),
            trainer=trainer_raw.get("fullName", "Unknown"),
            country=r.get("country", "AUS"),
            career_starts=int(career.get("starts", 0) or 0),
            career_wins=int(career.get("wins", 0) or 0),
            career_places=int(career.get("places", 0) or 0),
            last_10_starts=last_10,
            trainer_stats=trainer_stats,
            jockey_stats=jockey_stats,
            pedigree=pedigree,
            fixed_win_odds=fixed_win,
            tote_win_odds=tote_win,
            best_available_odds=fixed_win or tote_win,
        )

    def _parse_form_start(self, s: dict) -> FormStart:
        return FormStart(
            date=s.get("startDate", ""),
            track=s.get("track", ""),
            distance=int(s.get("distance", 0) or 0),
            track_condition=s.get("trackCondition", "Good 4"),
            barrier=int(s.get("barrier", 0) or 0),
            weight=float(s.get("weightCarried", 0) or 0),
            jockey=s.get("jockyName", ""),
            position=int(s.get("finishingPosition", 99) or 99),
            finishers=int(s.get("noOfStarters", 1) or 1),
            beaten_margin=float(s.get("margin", 0) or 0),
            race_class=s.get("raceClass", ""),
            prize_money=int(s.get("prizeMoney", 0) or 0),
            starting_price=float(s.get("startingPrice", 0) or 0) or None,
            sectional_last600=float(s.get("last600", 0) or 0) or None,
            in_running=s.get("inRunning"),
            comment=s.get("runnerComment"),
        )
