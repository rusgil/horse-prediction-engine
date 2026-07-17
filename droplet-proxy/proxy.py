"""
Minimal HTTPS proxy for Racing Australia.

Runs on any VPS with a public IP (DigitalOcean historically; Vultr/Linode
Sydney or Hetzner Singapore work with a different-ASN benefit). The Railway
backend's RacingAustraliaClient points its base URL at this proxy instead of
racingaustralia.horse directly. Result: RA sees the VPS's IP, not Railway's
WAF-blocked one.

Design:
  - One catch-all GET route at /proxy/{path}.
  - Forwards to https://www.racingaustralia.horse/{path} (preserving query string).
  - Returns the upstream response body + status code verbatim.
  - Auth: caller must send X-Proxy-Secret matching PROXY_SECRET env var.
  - Rate-limited at 1 req/sec per origin to prevent the proxy itself becoming a
    hammer (the hard rule about no API hammering still applies).
  - Uses curl_cffi (libcurl-impersonate) so the outbound TLS handshake AND
    HTTP/2 frame ordering matches a real Chrome browser. Plain httpx has a
    distinctive JA3 fingerprint that WAFs like Cloudflare / Akamai use to
    flag automated traffic — swapping to curl_cffi fixed a hard block that
    hit us on 2026-07-17 despite low request volume (~3253/day) and browser-
    like headers.

Deploy:
  See README.md in this directory. tl;dr - Caddy in front for free TLS,
  systemd unit to run the FastAPI app, ufw locking down to 22/80/443.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Optional

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError
from fastapi import FastAPI, HTTPException, Request, Response


PROXY_SECRET = os.environ.get("PROXY_SECRET", "")
if not PROXY_SECRET:
    raise SystemExit("PROXY_SECRET env var must be set")

UPSTREAM_BASE = "https://www.racingaustralia.horse"

# Impersonation profile — curl_cffi routes each request through a libcurl
# build that matches this browser's exact TLS ClientHello, HTTP/2 SETTINGS
# frame ordering, and header order. Keep the UA string in sync with the
# impersonate profile so the two signals agree (a Chrome131 TLS handshake
# with a Firefox UA is itself a fingerprint).
_IMPERSONATE_PROFILE = "chrome131"
_UA_STRING = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Single-flight + delay between requests - the proxy IS our single client to RA,
# so it must not hammer. asyncio.Lock + jittered sleep between each request.
_request_lock = asyncio.Lock()
_last_request_at = 0.0
_MIN_INTERVAL = 0.7  # seconds; with jitter, ~0.7-1.2s between RA calls

# Daily cap - rolling 24h. Belt-and-braces against runaway callers.
# Env-tunable via RA_PROXY_DAILY_CAP so a retrain / big enrichment day
# doesn't need a code change; default 10000 covers steady-state + a
# nightly retrain surge with headroom.
_DAILY_CAP = int(os.environ.get("RA_PROXY_DAILY_CAP", "10000"))
_daily_count = 0
_daily_window_start = 0.0  # set on first request

# Track RA 403s through the proxy. If RA blocks us, this jumps and the
# CRITICAL log lines surface in journalctl -u ra-proxy.
_recent_403_count = 0

app = FastAPI(title="ra-proxy")


def _headers(referer: Optional[str]) -> dict:
    # These headers get merged with what curl_cffi's impersonation profile
    # emits at the HTTP/2 layer, so the final header set matches a real
    # Chrome navigation. Keeping the UA aligned with _IMPERSONATE_PROFILE
    # is critical — a mismatched UA (JA3 says Chrome, UA says Firefox) is
    # itself a bot signature.
    headers = {
        "User-Agent": _UA_STRING,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/admin/cap-status")
async def cap_status(request: Request):
    """Report daily counter state. Gated by x-proxy-secret."""
    if request.headers.get("x-proxy-secret", "") != PROXY_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    now = time.monotonic()
    window_age_s = (now - _daily_window_start) if _daily_window_start else 0.0
    return {
        "daily_count": _daily_count,
        "daily_cap": _DAILY_CAP,
        "window_age_seconds": window_age_s,
        "window_remaining_seconds": max(0.0, 86400 - window_age_s) if _daily_window_start else 0.0,
        "recent_403_count": _recent_403_count,
    }


@app.post("/admin/reset-cap")
async def reset_cap(request: Request):
    """Zero the rolling-24h counter without restarting the service.
    For the rare case where a legit workload bumps the cap and we
    need to keep going. Gated by x-proxy-secret."""
    global _daily_count, _daily_window_start, _recent_403_count
    if request.headers.get("x-proxy-secret", "") != PROXY_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    prev_count = _daily_count
    _daily_count = 0
    _recent_403_count = 0
    _daily_window_start = time.monotonic()
    import logging as _l
    _l.getLogger("ra-proxy").warning(
        "Daily counter reset via /admin/reset-cap (was %d/%d)", prev_count, _DAILY_CAP
    )
    return {"reset": True, "previous_count": prev_count, "daily_cap": _DAILY_CAP}


@app.get("/proxy/{path:path}")
async def proxy(path: str, request: Request):
    """Forward GET to {UPSTREAM_BASE}/{path}?{query} and return upstream
    body + status verbatim. Caller must send X-Proxy-Secret."""
    global _last_request_at, _daily_count, _daily_window_start, _recent_403_count

    # Auth - fail closed.
    secret = request.headers.get("x-proxy-secret", "")
    if secret != PROXY_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Daily cap - rolling 24h window. Hard 503 when exceeded so a runaway
    # caller can't drain the budget overnight without anyone noticing.
    now = time.monotonic()
    if _daily_window_start == 0.0 or (now - _daily_window_start) > 86400:
        _daily_count = 0
        _recent_403_count = 0
        _daily_window_start = now
    if _daily_count >= _DAILY_CAP:
        import logging as _l
        _l.getLogger("ra-proxy").warning(
            "Daily cap reached (%d requests in current 24h window) - refusing further calls",
            _daily_count,
        )
        raise HTTPException(status_code=503, detail="Daily request cap reached")

    # Build upstream URL preserving query string.
    qs = request.url.query
    upstream_url = f"{UPSTREAM_BASE}/{path}"
    if qs:
        upstream_url = f"{upstream_url}?{qs}"

    referer = request.headers.get("x-proxy-referer") or None

    # Single-flight with min interval - proxy must not become the new hammer.
    async with _request_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_INTERVAL:
            await asyncio.sleep(_MIN_INTERVAL - elapsed + random.random() * 0.5)
        # curl_cffi AsyncSession with impersonate= handles the TLS
        # ClientHello + HTTP/2 SETTINGS ordering that WAFs fingerprint on.
        # Same call shape as httpx (resp.status_code, resp.content, resp.headers).
        try:
            async with AsyncSession(impersonate=_IMPERSONATE_PROFILE) as client:
                resp = await client.get(
                    upstream_url,
                    headers=_headers(referer),
                    timeout=20.0,
                    allow_redirects=True,
                )
        except RequestsError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
        _last_request_at = time.monotonic()
        _daily_count += 1

    # CRITICAL: RA returned 403 to the proxy. Source IP may be WAF-flagged.
    # Log loudly so it surfaces in `journalctl -u ra-proxy`.
    if resp.status_code == 403:
        _recent_403_count += 1
        import logging as _l
        _l.getLogger("ra-proxy").critical(
            "RA returned 403 (count=%d/cap=%d daily window) - droplet IP may be WAF-blocked. url=%s",
            _recent_403_count, _daily_count, upstream_url[:200],
        )

    # Return upstream body unchanged. Filter hop-by-hop headers so downstream
    # callers (httpx / anything else) don't get confused by re-encoded content.
    # curl_cffi returns bytes on .content, matching httpx's shape.
    drop = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    headers_out = {k: v for k, v in resp.headers.items() if k.lower() not in drop}
    return Response(content=resp.content, status_code=resp.status_code, headers=headers_out)
