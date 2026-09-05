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

# Optional residential/mobile upstream proxy. RA WAF-blocks datacenter ASNs
# (DigitalOcean, then Hetzner nbg1 + hel1 all 403'd), so when this is set the
# outbound curl_cffi request to RA is tunnelled through a residential exit IP
# and RA never sees this box's datacenter IP. Format:
#   http://user:pass@host:port   (or socks5h://user:pass@host:port)
# curl_cffi still performs the impersonated TLS handshake to RA through the
# CONNECT tunnel, so the fingerprint layer is preserved. Empty = direct.
RESIDENTIAL_PROXY_URL = os.environ.get("RESIDENTIAL_PROXY_URL", "").strip()
_PROXIES = (
    {"http": RESIDENTIAL_PROXY_URL, "https": RESIDENTIAL_PROXY_URL}
    if RESIDENTIAL_PROXY_URL
    else None
)

# Impersonation profile — curl_cffi routes each request through a libcurl
# build that matches this browser's exact TLS ClientHello, HTTP/2 SETTINGS
# frame ordering, and header order. Keep the UA string in sync with the
# impersonate profile so the two signals agree (a Chrome124 TLS handshake
# with a Firefox UA is itself a fingerprint).
# chrome124 is the highest available in curl_cffi 0.7.x — chrome131 needs
# 0.8.x which isn't yet on PyPI (verified 2026-07-18 against 0.7.4).
_IMPERSONATE_PROFILE = "chrome124"
_UA_STRING = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Sticky residential session (2026-07-24). Webshare rotates its residential
# pool PER CONNECTION, not per request — a kept-alive connection holds ONE
# exit IP (verified: 3 requests over one connection all returned the same IP).
# So we keep ONE persistent curl_cffi session and reuse it for every request:
# once we've locked onto an IP RA accepts, every fetch rides that same good IP,
# and a full enrich completes instead of re-rolling the block lottery on every
# request. On a 403 (that IP just got blocked) we DROP the session to force a
# fresh connection → a new residential IP, then retry. Rotate only on failure.
_STICKY_ROTATE_RETRIES = int(os.environ.get("RA_PROXY_ROTATE_RETRIES", "4"))
_STICKY_ROTATE_BACKOFF = 3.0   # seconds between IP rotations
_sticky_session: Optional[AsyncSession] = None


async def _get_session() -> AsyncSession:
    """The persistent session (one kept-alive connection = one residential IP)."""
    global _sticky_session
    if _sticky_session is None:
        _sticky_session = AsyncSession(impersonate=_IMPERSONATE_PROFILE, proxies=_PROXIES)
    return _sticky_session


async def _rotate_session() -> None:
    """Drop the current connection so the next request opens a fresh one —
    webshare hands a NEW residential exit IP on the new connection."""
    global _sticky_session
    s, _sticky_session = _sticky_session, None
    if s is not None:
        try:
            await s.close()
        except Exception:
            pass

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
# Track soft-blocks: a 200 with an empty/decoy Calendar page (WAF silent block).
# These carry no 403, so without detection the exit IP never rotates.
_recent_softblock_count = 0


def _looks_soft_blocked(path: str, resp) -> bool:
    """True if this is a Calendar 200 that carries NO meeting links — i.e. a
    silent WAF soft-block, not a genuine empty racing day. A real RA calendar
    page always lists Acceptances/Results links for the surrounding week (even
    when today is empty), so 'zero links' can only mean a decoy/interstitial."""
    if resp is None or resp.status_code != 200:
        return False
    if "Calendar.aspx" not in path:
        return False
    try:
        body = resp.content or b""
    except Exception:
        return False
    if len(body) < 200:                       # a real calendar is tens of KB
        return True
    return (b"Acceptances" not in body) and (b"Results.aspx" not in body)

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
    # residential flag is a bool only — never echo the proxy URL (it carries
    # user:pass credentials).
    return {"status": "ok", "residential": bool(_PROXIES)}


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
        "recent_softblock_count": _recent_softblock_count,
    }


@app.post("/admin/reset-cap")
async def reset_cap(request: Request):
    """Zero the rolling-24h counter without restarting the service.
    For the rare case where a legit workload bumps the cap and we
    need to keep going. Gated by x-proxy-secret."""
    global _daily_count, _daily_window_start, _recent_403_count, _recent_softblock_count
    if request.headers.get("x-proxy-secret", "") != PROXY_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    prev_count = _daily_count
    _daily_count = 0
    _recent_403_count = 0
    _recent_softblock_count = 0
    _daily_window_start = time.monotonic()
    import logging as _l
    _l.getLogger("ra-proxy").warning(
        "Daily counter reset via /admin/reset-cap (was %d/%d)", prev_count, _DAILY_CAP
    )
    return {"reset": True, "previous_count": prev_count, "daily_cap": _DAILY_CAP}


@app.post("/admin/rotate")
async def admin_rotate(request: Request):
    """Force a residential exit-IP rotation on demand — drops the sticky
    session so the next RA request opens a fresh connection (new webshare exit).
    Use when the current IP is soft-blocked but hasn't 403'd. Gated by
    x-proxy-secret."""
    if request.headers.get("x-proxy-secret", "") != PROXY_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    await _rotate_session()
    import logging as _l
    _l.getLogger("ra-proxy").warning("residential session rotated via /admin/rotate")
    return {"rotated": True}


@app.get("/proxy/{path:path}")
async def proxy(path: str, request: Request):
    """Forward GET to {UPSTREAM_BASE}/{path}?{query} and return upstream
    body + status verbatim. Caller must send X-Proxy-Secret."""
    global _last_request_at, _daily_count, _daily_window_start, _recent_403_count, _recent_softblock_count

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
        # Reuse the persistent (sticky) session so every request rides the same
        # residential exit IP. On a 403 the IP is blocked → rotate to a fresh
        # connection/IP and retry; only after all rotations fail do we surface
        # the 403. The impersonate= profile handles the TLS ClientHello + HTTP/2
        # SETTINGS ordering that WAFs fingerprint on.
        import logging as _l
        _log = _l.getLogger("ra-proxy")
        resp = None
        last_err = None
        for attempt in range(1 + _STICKY_ROTATE_RETRIES):
            if attempt > 0:
                await _rotate_session()                    # force a new exit IP
                await asyncio.sleep(_STICKY_ROTATE_BACKOFF)
                _log.warning("rotating residential IP (attempt %d/%d) for %s",
                             attempt, _STICKY_ROTATE_RETRIES, upstream_url[:120])
            session = await _get_session()
            try:
                resp = await session.get(
                    upstream_url,
                    headers=_headers(referer),
                    timeout=40.0,
                    allow_redirects=True,
                )
            except RequestsError as e:
                last_err = e
                await _rotate_session()                    # drop the bad connection
                resp = None
                continue
            if resp.status_code == 403:
                # This exit IP just got blocked — rotate to a new one and retry.
                continue
            if _looks_soft_blocked(path, resp):
                # Silent block: a 200 with no meeting links on a Calendar page.
                # Treat exactly like a 403 — rotate to a fresh exit IP and retry,
                # otherwise we'd forward a phantom "no meetings" that the backend
                # caches as an empty racing day (the partial-card incident).
                _recent_softblock_count += 1
                _log.warning(
                    "soft-block (200, no meeting links) on %s — rotating IP (attempt %d/%d)",
                    upstream_url[:120], attempt, _STICKY_ROTATE_RETRIES,
                )
                continue
            break                                          # non-403 → done
        _last_request_at = time.monotonic()
        _daily_count += 1
        if resp is None:
            raise HTTPException(status_code=502, detail=f"Upstream error: {last_err}")

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
