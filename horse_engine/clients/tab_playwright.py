"""
TAB Australia data client via Playwright headless browser.

Visits tab.com.au like a real browser and intercepts the API calls
it makes to api.beta.tab.com.au — bypassing the IP whitelist entirely.

Resources (images, CSS, fonts, media) are aborted to minimise memory use
on Railway. Data requests pass through normally so the Angular app boots
and fires its API calls, which we capture and return.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from horse_engine.config import settings
from horse_engine.models.race import (
    FormStart, JockeyStats, Meeting, PedigreeProfile, Race, Runner, TrainerStats,
)

log = logging.getLogger(__name__)

TAB_BASE = "https://www.tab.com.au"
API_HOST = "api.beta.tab.com.au"

# Resource types to block — saves ~60% memory and speeds up page loads
BLOCKED_TYPES = {"image", "stylesheet", "font", "media", "ping", "other"}

LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-software-rasterizer",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-default-apps",
    "--mute-audio",
    "--no-first-run",
    "--no-zygote",
    "--single-process",          # critical for Railway's container environment
]


class PlaywrightTABClient:
    """
    Async TAB client that drives a headless Chromium to intercept
    TAB's own internal API responses.
    """

    def __init__(self, jurisdiction: str | None = None):
        self.jurisdiction = (jurisdiction or settings.tab_jurisdiction).upper()

    async def get_meetings(self, race_date: str) -> list[dict]:
        """Return raw meeting dicts for thoroughbred races on race_date."""
        url = f"{TAB_BASE}/racing/{race_date}/meetings/thoroughbred"
        captured: dict[str, Any] = {}

        async def _on_response(response):
            if (
                API_HOST in response.url
                and "tab-info-service/racing/dates" in response.url
                and "/meetings" in response.url
                and "/races/" not in response.url
            ):
                try:
                    data = await response.json()
                    captured["meetings"] = data.get("meetings", [])
                    log.debug("Captured meetings response: %d meetings", len(captured["meetings"]))
                except Exception as e:
                    log.debug("Could not parse meetings response: %s", e)

        await self._browse(url, _on_response)
        meetings = captured.get("meetings", [])
        return [m for m in meetings if m.get("raceType") == "R"]

    async def get_race(self, race_date: str, venue_code: str, race_number: int) -> dict | None:
        """Return raw race dict including runners, form, and current odds."""
        url = (
            f"{TAB_BASE}/racing/{race_date}/meetings/thoroughbred"
            f"/{venue_code}/races/{race_number}"
        )
        captured: dict[str, Any] = {}

        async def _on_response(response):
            rurl = response.url
            if (
                API_HOST in rurl
                and "tab-info-service/racing/dates" in rurl
                and f"/{venue_code}/" in rurl.upper()
                and f"/races/{race_number}" in rurl
            ):
                try:
                    data = await response.json()
                    captured["race"] = data
                    log.debug("Captured race response for R%s", race_number)
                except Exception as e:
                    log.debug("Could not parse race response: %s", e)

        await self._browse(url, _on_response, timeout=45_000)
        return captured.get("race")

    async def get_meeting_races(self, race_date: str, venue_code: str) -> list[dict]:
        """Return list of race summaries for all races at a meeting."""
        url = (
            f"{TAB_BASE}/racing/{race_date}/meetings/thoroughbred/{venue_code}"
        )
        captured: list = []

        async def _on_response(response):
            rurl = response.url
            if (
                API_HOST in rurl
                and "tab-info-service/racing/dates" in rurl
                and f"/{venue_code}/" in rurl.upper()
                and "/races" in rurl
                and not any(f"/races/{n}" in rurl for n in range(1, 20))
            ):
                try:
                    data = await response.json()
                    races = data.get("races", [])
                    if races:
                        captured.extend(races)
                        log.debug("Captured %d races for %s", len(races), venue_code)
                except Exception as e:
                    log.debug("Could not parse races response: %s", e)

        await self._browse(url, _on_response)
        return captured

    # ── Core browser driver ────────────────────────────────────────────────

    async def _browse(self, url: str, on_response, timeout: int = 30_000) -> None:
        """
        Launch headless Chromium, navigate to url, wait for network idle,
        fire on_response for every captured response. Cleans up on exit.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            log.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
            return

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=LAUNCH_ARGS,
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
            )

            page = await context.new_page()

            # Block heavyweight resources
            async def _route(route):
                if route.request.resource_type in BLOCKED_TYPES:
                    await route.abort()
                else:
                    await route.continue_()

            await page.route("**/*", _route)
            page.on("response", on_response)

            log.info("Playwright navigating to %s", url)
            try:
                await page.goto(url, wait_until="networkidle", timeout=timeout)
            except Exception as e:
                log.warning("Page navigation timed out or errored (%s) — data may be partial", e)

            await browser.close()

    # ── Parsing (reuse same logic as HTTP client) ─────────────────────────

    def parse_meeting(self, raw: dict, race_date: str) -> Meeting:
        return Meeting(
            date=race_date,
            venue=raw.get("venueName", "Unknown"),
            state=raw.get("location", {}).get("state", ""),
            track_condition=raw.get("trackCondition", {}).get("name", "Good 4"),
        )

    def parse_race(self, raw: dict, race_date: str, venue: str, state: str) -> Race:
        from horse_engine.clients.tab import TABClient
        return TABClient().parse_race(raw, race_date, venue, state)
