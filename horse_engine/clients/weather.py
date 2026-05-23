"""Weather client using Open-Meteo (free, no API key required)."""
from __future__ import annotations

import logging
import time

import httpx
from sqlalchemy import select

log = logging.getLogger(__name__)

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CACHE_TTL = 3600  # weather cached for 1 hour per venue+date

_weather_cache: dict[str, dict] = {}
_cache_expiry: dict[str, float] = {}

_WMO: dict[int, tuple[str, str]] = {
    0:  ("Clear", "☀️"),
    1:  ("Mainly clear", "🌤️"),
    2:  ("Partly cloudy", "⛅"),
    3:  ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Foggy", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Drizzle", "🌦️"),
    55: ("Heavy drizzle", "🌦️"),
    61: ("Light rain", "🌧️"),
    63: ("Rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    71: ("Light snow", "❄️"),
    73: ("Snow", "❄️"),
    75: ("Heavy snow", "❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Showers", "🌧️"),
    81: ("Showers", "🌧️"),
    82: ("Heavy showers", "🌧️"),
    85: ("Snow showers", "❄️"),
    86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm", "⛈️"),
    99: ("Thunderstorm", "⛈️"),
}


_STATE_FULL = {
    "NSW": "New South Wales", "VIC": "Victoria", "QLD": "Queensland",
    "SA": "South Australia", "WA": "Western Australia", "TAS": "Tasmania",
    "NT": "Northern Territory", "ACT": "Australian Capital Territory",
}


async def _geocode(venue: str, state: str) -> tuple[float, float] | None:
    state_full = _STATE_FULL.get(state.upper(), state)
    for query in [venue, f"{venue} {state_full}"]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(_GEO_URL, params={
                    "name": query, "count": 10, "language": "en", "format": "json",
                })
                r.raise_for_status()
                results = r.json().get("results", [])
                au = [x for x in results if x.get("country_code") == "AU"]
                if au:
                    return au[0]["latitude"], au[0]["longitude"]
        except Exception as e:
            log.warning("Geocoding failed for %s: %s", query, e)
    return None


async def _get_coords(venue: str, state: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a venue, using DB cache."""
    from horse_engine.models.database import SessionLocal, VenueCoordinateRow

    key = f"{venue}|{state}"
    async with SessionLocal() as session:
        row = (await session.execute(
            select(VenueCoordinateRow).where(VenueCoordinateRow.venue_key == key)
        )).scalar_one_or_none()
        if row:
            return row.latitude, row.longitude

    coords = await _geocode(venue, state)
    if coords:
        lat, lon = coords
        async with SessionLocal() as session:
            session.add(VenueCoordinateRow(venue_key=key, latitude=lat, longitude=lon))
            try:
                await session.commit()
            except Exception:
                pass  # ignore race-condition duplicate writes
    return coords


async def get_weather_for_venue(venue: str, state: str, race_date: str) -> dict | None:
    """Fetch daily forecast weather for a venue on a given date."""
    if not venue:
        return None

    cache_key = f"{venue}|{state}|{race_date}"
    now = time.time()
    if cache_key in _weather_cache and _cache_expiry.get(cache_key, 0) > now:
        return _weather_cache[cache_key]

    coords = await _get_coords(venue, state)
    if not coords:
        return None
    lat, lon = coords

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(_FORECAST_URL, params={
                "latitude": lat,
                "longitude": lon,
                "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
                "timezone": "Australia/Sydney",
                "start_date": race_date,
                "end_date": race_date,
            })
            r.raise_for_status()
            daily = r.json().get("daily", {})

        def _first(key: str):
            vals = daily.get(key) or []
            return vals[0] if vals else None

        code = _first("weathercode") or 0
        desc, icon = _WMO.get(int(code), ("Unknown", "🌡️"))
        result = {
            "icon": icon,
            "condition": desc,
            "temp_max": round(_first("temperature_2m_max"), 1) if _first("temperature_2m_max") is not None else None,
            "temp_min": round(_first("temperature_2m_min"), 1) if _first("temperature_2m_min") is not None else None,
            "wind_kmh": round(_first("windspeed_10m_max"), 0) if _first("windspeed_10m_max") is not None else None,
            "rain_mm": round(_first("precipitation_sum"), 1) if _first("precipitation_sum") is not None else None,
        }
        _weather_cache[cache_key] = result
        _cache_expiry[cache_key] = now + _CACHE_TTL
        return result
    except Exception as e:
        log.warning("Weather fetch failed for %s on %s: %s", venue, race_date, e)
        return None
