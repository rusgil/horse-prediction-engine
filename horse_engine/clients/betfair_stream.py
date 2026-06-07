"""
Betfair Exchange Streaming API client for live AU horse racing odds.

Maintains a persistent SSL connection to stream-api.betfair.com, subscribes to
all Australian thoroughbred WIN markets for the current day, and provides:
  - Current LTP (last traded price) per runner
  - Steam / drift features computed from the full LTP history

Required env vars (same as BetfairClient):
  BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

_STREAM_HOST = "stream-api.betfair.com"
_STREAM_PORT = 443
_LOGIN_URL = "https://identitysso.betfair.com.au/api/login"
_MAX_LTP_HISTORY = 2000   # price ticks kept per runner
_RECONNECT_DELAY = 30     # seconds before reconnect attempt


def _strip_barrier(name: str) -> str:
    """'1. Dark Fox' → 'dark fox' — strip leading 'N. ' prefix."""
    return re.sub(r'^\d+\.\s*', '', name).lower().strip()


class BetfairStreamClient:
    """
    Background asyncio task that streams live odds for AU horse racing.

    Usage:
        client = BetfairStreamClient(app_key, username, password)
        await client.start()          # fires background task, returns immediately
        ...
        features = client.get_odds_features(market_id, "Dark Fox")
        # {"current_ltp": 4.2, "steam_60": 1.8, "steam_30": 0.9, ...}
        await client.stop()
    """

    def __init__(self, app_key: str, username: str, password: str) -> None:
        self._app_key = app_key
        self._username = username
        self._password = password
        self._session_token: str | None = None

        # market_id → runner_id → deque of (timestamp_ms, ltp)
        self._ltp: dict[str, dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=_MAX_LTP_HISTORY))
        )
        # market_id → runner_id → display name
        self._runner_names: dict[str, dict[int, str]] = defaultdict(dict)
        # market_id → normalised_name → runner_id  (two keys per runner: raw + stripped)
        self._name_to_id: dict[str, dict[str, int]] = defaultdict(dict)
        # market_id → status string (OPEN / SUSPENDED / CLOSED)
        self._market_status: dict[str, str] = {}
        # market_id → runner_id → "WINNER" | "LOSER" | "REMOVED"
        self._market_results: dict[str, dict[int, str]] = defaultdict(dict)

        self._task: asyncio.Task | None = None
        self._running = False
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever(), name="betfair-stream")
        log.info("Betfair stream task started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("Betfair stream stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Public query API ──────────────────────────────────────────────────────

    def get_current_ltp(self, market_id: str, runner_name: str) -> float | None:
        rid = self._resolve_runner(market_id, runner_name)
        if rid is None:
            return None
        hist = self._ltp.get(market_id, {}).get(rid)
        return hist[-1][1] if hist else None

    def get_odds_features(self, market_id: str, runner_name: str) -> dict[str, Any]:
        """
        Return steam/drift features for a runner.

        Keys: current_ltp, steam_60, steam_30, late_money, drift_flag, odds_velocity.
        Returns {} if no data yet.
        """
        rid = self._resolve_runner(market_id, runner_name)
        if rid is None:
            return {}

        hist = list(self._ltp.get(market_id, {}).get(rid, []))
        if not hist:
            return {}

        now_ms = datetime.now(timezone.utc).timestamp() * 1000
        current_ltp = hist[-1][1]

        def price_at_offset(minutes_ago: float) -> float | None:
            target = now_ms - minutes_ago * 60_000
            # Walk backward to find last price at or before target
            for ts, ltp in reversed(hist):
                if ts <= target:
                    return ltp
            return None  # all data is more recent than the offset

        p60 = price_at_offset(60)
        p30 = price_at_offset(30)
        p15 = price_at_offset(15)
        first_price = hist[0][1]

        # Positive = price shortened (money in)
        steam_60 = round(p60 - current_ltp, 3) if p60 is not None else 0.0
        steam_30 = round(p30 - current_ltp, 3) if p30 is not None else 0.0
        late_money = round(p15 - current_ltp, 3) if p15 is not None else 0.0

        # Drifted: opened shorter than 5.0 and is now 20%+ longer
        drift_flag = 1.0 if (first_price < 5.0 and current_ltp > first_price * 1.2) else 0.0

        # Velocity: average odds change per minute over the last 60 min
        odds_velocity = round((p60 - current_ltp) / 60.0, 4) if p60 is not None else 0.0

        return {
            "current_ltp": current_ltp,
            "steam_60": steam_60,
            "steam_30": steam_30,
            "late_money": late_money,
            "drift_flag": drift_flag,
            "odds_velocity": odds_velocity,
        }

    def get_market_winner(self, market_id: str) -> str | None:
        """Return the horse name of the WINNER once the market is settled, else None."""
        for rid, status in self._market_results.get(market_id, {}).items():
            if status == "WINNER":
                return self._runner_names.get(market_id, {}).get(rid)
        return None

    def get_market_placers(self, market_id: str) -> list[str]:
        """Return names of runners marked WINNER in the PLACE market (top 3)."""
        return [
            self._runner_names[market_id][rid]
            for rid, status in self._market_results.get(market_id, {}).items()
            if status == "WINNER" and rid in self._runner_names.get(market_id, {})
        ]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _resolve_runner(self, market_id: str, runner_name: str) -> int | None:
        nm = self._name_to_id.get(market_id, {})
        # Try exact lowercased name first, then barrier-stripped version
        key1 = runner_name.lower().strip()
        key2 = _strip_barrier(runner_name)
        return nm.get(key1) or nm.get(key2)

    async def _run_forever(self) -> None:
        while self._running:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._connected = False
                log.warning("Stream disconnected (%s), reconnecting in %ds", exc, _RECONNECT_DELAY)
                await asyncio.sleep(_RECONNECT_DELAY)
                # Re-authenticate before reconnect
                await self._login()

    async def _login(self) -> bool:
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
                    log.info("Betfair stream: login successful")
                    return True
                log.warning("Betfair stream login failed: %s", body.get("error"))
                return False
        except Exception as exc:
            log.warning("Betfair stream login error: %s", exc)
            return False

    async def _connect_and_stream(self) -> None:
        if not self._session_token:
            if not await self._login():
                raise RuntimeError("Cannot authenticate with Betfair")

        ctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(_STREAM_HOST, _STREAM_PORT, ssl=ctx)
        log.info("Betfair stream: TCP+SSL connected to %s:%d", _STREAM_HOST, _STREAM_PORT)

        try:
            # Receive server connection message
            conn_line = await asyncio.wait_for(reader.readline(), timeout=15.0)
            log.debug("Stream connection msg: %s", conn_line[:120])

            # Authenticate
            await self._send(writer, {
                "op": "authentication",
                "id": 1,
                "appKey": self._app_key,
                "session": self._session_token,
            })
            auth_line = await asyncio.wait_for(reader.readline(), timeout=15.0)
            auth = json.loads(auth_line)
            if auth.get("statusCode") != "SUCCESS":
                log.error("Stream auth failed: %s", auth)
                self._session_token = None  # force re-login on next attempt
                raise RuntimeError(f"Stream auth failed: {auth.get('errorCode')}")
            log.info("Betfair stream: authenticated")

            # Subscribe to AU thoroughbred WIN markets
            await self._send(writer, {
                "op": "marketSubscription",
                "id": 2,
                "marketFilter": {
                    "countries": ["AU"],
                    "eventTypeIds": ["7"],   # horse racing
                    "marketTypes": ["WIN"],
                },
                "marketDataFilter": {
                    "fields": ["EX_LTP", "EX_MARKET_DEF"],
                },
                "heartbeatMs": 5000,
            })
            log.info("Betfair stream: subscribed to AU WIN markets")
            self._connected = True

            # Process incoming messages indefinitely
            while self._running:
                line = await asyncio.wait_for(reader.readline(), timeout=65.0)
                if not line:
                    raise ConnectionError("Stream EOF")
                self._process_message(json.loads(line))

        finally:
            writer.close()
            self._connected = False

    async def _send(self, writer: asyncio.StreamWriter, msg: dict) -> None:
        writer.write((json.dumps(msg) + "\r\n").encode())
        await writer.drain()

    def _process_message(self, msg: dict) -> None:
        op = msg.get("op")
        if op == "connection":
            return  # already handled in connect loop
        if op == "status":
            return  # auth/subscription status
        if op != "mcm":
            return  # heartbeat or unknown

        pt: int = msg.get("pt", 0)  # publish timestamp ms

        for mc in msg.get("mc", []):
            market_id: str | None = mc.get("id")
            if not market_id:
                continue

            # Market definition (initial image or update)
            md = mc.get("marketDefinition")
            if md:
                self._market_status[market_id] = md.get("status", "")
                for runner in md.get("runners", []):
                    rid: int = runner.get("id", 0)
                    name: str = runner.get("name", "")
                    if not rid or not name:
                        continue

                    # Store name and build lookup keys
                    self._runner_names[market_id][rid] = name
                    nm = self._name_to_id[market_id]
                    nm[name.lower().strip()] = rid
                    nm[_strip_barrier(name)] = rid

                    # Record settlement status
                    status = runner.get("status", "")
                    if status in ("WINNER", "LOSER", "REMOVED"):
                        self._market_results[market_id][rid] = status

            # Runner changes (LTP updates)
            for rc in mc.get("rc", []):
                rid = rc.get("id", 0)
                ltp = rc.get("ltp")
                if rid and ltp and pt:
                    self._ltp[market_id][rid].append((pt, float(ltp)))
