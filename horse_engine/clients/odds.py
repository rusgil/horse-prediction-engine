"""
Multi-bookmaker odds scraper.

Sources:
  - TAB (primary, via API)
  - Sportsbet (scraped)
  - Ladbrokes (scraped)
  - Betfair (exchange, best available)

Returns best available odds and movement signals for each runner.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "Accept": "application/json, text/html",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
TIMEOUT = 15.0


async def fetch_sportsbet_odds(horse_name: str, race_id: str) -> Optional[float]:
    """
    Scrape Sportsbet for a horse's current fixed win odds.
    Sportsbet uses a public endpoint pattern that can be queried.
    """
    # Sportsbet racing API endpoint (public, no auth)
    try:
        url = "https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/Events/Next/5"
        async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            events = resp.json()
            for event in events.get("data", {}).get("events", []):
                for comp in event.get("competitors", []):
                    if comp.get("name", "").lower().strip() == horse_name.lower().strip():
                        prices = comp.get("prices", [])
                        if prices:
                            return float(prices[0].get("winPrice", 0) or 0) or None
    except Exception as e:
        log.debug("Sportsbet scrape error for %s: %s", horse_name, e)
    return None


async def fetch_ladbrokes_odds(horse_name: str, venue: str, race_number: int) -> Optional[float]:
    """
    Fetch Ladbrokes odds via their public racing API.
    """
    try:
        url = f"https://www.ladbrokes.com.au/racing/api/v2/race/{venue.lower()}-race-{race_number}"
        async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
            for runner in data.get("runners", []):
                if runner.get("name", "").lower().strip() == horse_name.lower().strip():
                    return float(runner.get("fixedWinOdds", 0) or 0) or None
    except Exception as e:
        log.debug("Ladbrokes scrape error for %s: %s", horse_name, e)
    return None


def best_odds(
    tab: Optional[float],
    sportsbet: Optional[float],
    ladbrokes: Optional[float],
) -> tuple[float, str]:
    """Return (best_price, bookmaker_name)."""
    options = [
        (tab or 0, "TAB"),
        (sportsbet or 0, "Sportsbet"),
        (ladbrokes or 0, "Ladbrokes"),
    ]
    best = max(options, key=lambda x: x[0])
    if best[0] == 0:
        return 0.0, "N/A"
    return best


def compute_odds_movement(opening: Optional[float], current: Optional[float]) -> float:
    """
    Positive = shortened (money in = steam).
    Negative = drifted (money out).
    """
    if not opening or not current or opening == 0:
        return 0.0
    return round(opening - current, 2)


def implied_probability(odds: float) -> float:
    """Convert decimal odds to raw implied probability."""
    if not odds or odds <= 1.0:
        return 0.0
    return round(1.0 / odds, 4)


def overround_free_probability(all_odds: list[float]) -> list[float]:
    """
    Remove bookmaker margin to get true implied probabilities.
    All odds should be from the same book.
    """
    if not all_odds:
        return []
    raw = [implied_probability(o) for o in all_odds]
    total = sum(raw)
    if total == 0:
        return raw
    return [round(p / total, 4) for p in raw]
