"""
Composite racing client: Racing Australia (race cards + results) + OddsPro (odds).

Optional: BetfairClient enriches runner metadata (sire, dam, age, sex) and
provides exchange odds as a fallback when OddsPro has no price.
Betfair is activated only when BETFAIR_APP_KEY / BETFAIR_USERNAME / BETFAIR_PASSWORD
are set in the environment.
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


async def _none() -> None:
    return None


def _normalize(name: str) -> str:
    """Lowercase + strip country codes for name matching."""
    return re.sub(r'\s*\([^)]+\)', '', name).strip().lower()


class CompositeClient:
    def __init__(self) -> None:
        self._ra = RacingAustraliaClient()
        self._odds = OddsProClient()
        self._bf = None      # BetfairClient — lazy init only if credentials set
        self._punters = None  # PuntersClient — lazy init only if proxy URL set

    def _get_betfair(self):
        if self._bf is None:
            try:
                from horse_engine.config import settings
                if settings.betfair_app_key and settings.betfair_username and settings.betfair_password:
                    from horse_engine.clients.betfair import BetfairClient
                    self._bf = BetfairClient()
                    log.info("BetfairClient activated for odds + metadata enrichment")
            except Exception as e:
                log.debug("BetfairClient init skipped: %s", e)
        return self._bf

    def _get_punters(self):
        if self._punters is None:
            try:
                from horse_engine.config import settings
                if settings.punters_proxy_url:
                    from horse_engine.clients.punters import PuntersClient
                    self._punters = PuntersClient()
                    log.info("PuntersClient activated via proxy: %s", settings.punters_proxy_url)
            except Exception as e:
                log.debug("PuntersClient init skipped: %s", e)
        return self._punters

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

        # Find OddsPro track and start Betfair + Punters fetches in parallel
        bf = self._get_betfair()
        punters = self._get_punters()
        tracks = await self._odds.get_tracks(race_date)
        op_track = self._odds.find_matching_track(venue, tracks)
        if not op_track:
            log.debug("OddsPro: no track match for '%s' on %s", venue, race_date)

        odds_map, ra_results, bf_race, punters_race = await asyncio.gather(
            self._odds.get_track_odds(op_track) if op_track else _empty_dict(),
            self._ra.get_results(ra_key) if ra_key else _empty_dict(),
            bf.get_race(slug, race_number) if bf else _none(),
            punters.get_race(slug, race_number) if punters else _none(),
        )

        # Build Betfair selection lookup keyed by normalized horse name
        bf_by_name: dict[str, dict] = {}
        if bf_race:
            for bf_sel in (bf_race.get("selections") or []):
                bf_name = _normalize((bf_sel.get("competitor") or {}).get("name", ""))
                if bf_name:
                    bf_by_name[bf_name] = bf_sel

        race_results: dict[str, dict] = (ra_results.get(race_number) or {}).get("runners", {})

        for sel in raw.get("selections", []):
            name_raw = (sel.get("competitor") or {}).get("name", "")
            name = name_raw.lower()
            norm = _normalize(name_raw)
            op = odds_map.get((race_number, name), {})
            ra = race_results.get(name, {})
            bf_sel = bf_by_name.get(norm) or bf_by_name.get(name)

            # Odds: OddsPro first, Betfair exchange as fallback
            if op:
                sel["topToteWin"] = op.get("currentBestOdds")
                sel["_odds_opening"] = op.get("firstPrice")
            elif bf_sel and bf_sel.get("topToteWin"):
                sel["topToteWin"] = bf_sel["topToteWin"]

            # Betfair metadata: sire, dam, age, sex, colour
            if bf_sel:
                comp = sel.setdefault("competitor", {})
                bf_comp = bf_sel.get("competitor") or {}
                for field in ("sire", "dam", "age", "sex", "colour"):
                    if bf_comp.get(field) and not comp.get(field):
                        comp[field] = bf_comp[field]

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

        # Store Punters selections for parse_race enrichment
        if punters_race:
            raw["_punters_sels"] = punters_race.get("selections") or []

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
            if sel:
                runner.odds_opening = sel.get("_odds_opening")

        punters_sels = raw_event.get("_punters_sels") or []
        if punters_sels:
            punters = self._get_punters()
            if punters:
                await self._enrich_from_punters(race, punters_sels, punters)

        return race

    async def _enrich_from_punters(self, race: Race, punters_sels: list[dict], punters_client) -> None:
        from horse_engine.clients.punters import PuntersClient, _parse_stat

        j_stats, t_stats = await punters_client._fetch_all_people_stats(punters_sels)

        p_by_name: dict[str, dict] = {}
        for sel in punters_sels:
            name = _normalize((sel.get("competitor") or {}).get("name", ""))
            if name:
                p_by_name[name] = sel

        for runner in race.runners:
            sel = p_by_name.get(_normalize(runner.horse_name))
            if not sel:
                continue

            comp = sel.get("competitor") or {}
            jock = sel.get("jockey") or {}
            trnr = sel.get("trainer") or {}
            raw_stats = comp.get("stats") or {}

            runner.career_starts = int(raw_stats.get("totalRuns") or 0)
            runner.career_wins = int(raw_stats.get("wins") or 0)
            runner.career_places = (
                runner.career_wins
                + int(raw_stats.get("seconds") or 0)
                + int(raw_stats.get("thirds") or 0)
            )

            tr_s, tr_w, _, _ = _parse_stat(raw_stats.get("track") or "")
            di_s, di_w, _, _ = _parse_stat(raw_stats.get("distance") or "")
            td_s, td_w, _, _ = _parse_stat(raw_stats.get("trackDistance") or "")
            fu_s, fu_w, _, _ = _parse_stat(raw_stats.get("firstUp") or "")
            su_s, su_w, _, _ = _parse_stat(raw_stats.get("secondUp") or "")

            runner.track_starts = tr_s
            runner.track_wins = tr_w
            runner.distance_starts = di_s
            runner.distance_wins = di_w
            runner.track_distance_starts = td_s
            runner.track_distance_wins = td_w
            runner.first_up_starts = fu_s
            runner.first_up_wins = fu_w
            runner.second_up_starts = su_s
            runner.second_up_wins = su_w

            last_10 = PuntersClient._parse_forms(comp.get("forms") or [], sel.get("lastRun"))
            if last_10:
                runner.last_10_starts = last_10

            j_slug = jock.get("slug") or ""
            j_raw = j_stats.get(j_slug) or {}
            if j_raw:
                j_win = float(j_raw.get("winPercentage") or 10)
                if runner.jockey_stats:
                    runner.jockey_stats.win_rate_overall = j_win
                    runner.jockey_stats.win_rate_track = j_win
                    runner.jockey_stats.win_rate_distance = j_win
                    runner.jockey_stats.win_rate_barrier_low = j_win
                    runner.jockey_stats.win_rate_barrier_mid = j_win
                    runner.jockey_stats.win_rate_barrier_wide = j_win
                    runner.jockey_stats.wins_season = int(j_raw.get("wins") or 0)
                    runner.jockey_stats.trainer_jockey_combo_rate = j_win

            t_slug = trnr.get("slug") or ""
            t_raw = t_stats.get(t_slug) or {}
            if t_raw:
                t_win = float(t_raw.get("winPercentage") or 10)
                if runner.trainer_stats:
                    runner.trainer_stats.win_rate_overall = t_win
                    runner.trainer_stats.win_rate_track = t_win
                    runner.trainer_stats.win_rate_distance = t_win
                    runner.trainer_stats.win_rate_first_up = t_win
                    runner.trainer_stats.win_rate_second_up = t_win
                    runner.trainer_stats.win_rate_wet = t_win
                    runner.trainer_stats.wins_season = int(t_raw.get("wins") or 0)

            log.debug("Punters enriched: %s (career %d/%d, j=%s %.1f%%, t=%s %.1f%%)",
                      runner.horse_name, runner.career_wins, runner.career_starts,
                      jock.get("name", ""), float(j_raw.get("winPercentage") or 0),
                      trnr.get("name", ""), float(t_raw.get("winPercentage") or 0))
