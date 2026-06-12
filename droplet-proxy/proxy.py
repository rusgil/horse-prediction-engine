"""
Minimal HTTPS proxy for Racing Australia.

Runs on a DigitalOcean droplet (or any VPS with a public IP). The Railway
backend's RacingAustraliaClient points its base URL at this proxy instead of
racingaustralia.horse directly. Result: RA sees the droplet's IP, not Railway's
WAF-blocked one.

Design:
  - One catch-all GET route at /proxy/{path}.
  - Forwards to https://www.racingaustralia.horse/{path} (preserving query string).
  - Returns the upstream response body + status code verbatim.
  - Auth: caller must send X-Proxy-Secret matching PROXY_SECRET env var.
  - Rate-limited at 1 req/sec per origin to prevent the proxy itself becoming a
    hammer (the hard rule about no API hammering still applies).
  - Rotates UA + browser-realistic headers, same as the in-app RA client.

Deploy:
  See README.md in this directory. tl;dr — Caddy in front for free TLS,
  systemd unit to run the FastAPI app, ufw locking down to 22/80/443.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response


PROXY_SECRET = os.environ.get("PROXY_SECRET", "")
if not PROXY_SECRET:
    raise SystemExit("PROXY_SECRET env var must be set")

UPSTREAM_BASE = "https://www.racingaustralia.horse"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
]

# Single-flight + delay between requests — the proxy IS our single client to RA,
# so it must not hammer. asyncio.Lock + jittered sleep between each request.
_request_lock = asyncio.Lock()
_last_request_at = 0.0
_MIN_INTERVAL = 0.7  # seconds; with jitter, ~0.7-1.2s between RA calls

app = FastAPI(title="ra-proxy")


def _headers(referer: Optional[str]) -> dict:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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


@app.get("/proxy/{path:path}")
async def proxy(path: str, request: Request):
    """Forward GET to {UPSTREAM_BASE}/{path}?{query} and return upstream
    body + status verbatim. Caller must send X-Proxy-Secret."""
    global _last_request_at

    # Auth — fail closed.
    secret = request.headers.get("x-proxy-secret", "")
    if secret != PROXY_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Build upstream URL preserving query string.
    qs = request.url.query
    upstream_url = f"{UPSTREAM_BASE}/{path}"
    if qs:
        upstream_url = f"{upstream_url}?{qs}"

    referer = request.headers.get("x-proxy-referer") or None

    # Single-flight with min interval — proxy must not become the new hammer.
    async with _request_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_INTERVAL:
            await asyncio.sleep(_MIN_INTERVAL - elapsed + random.random() * 0.5)
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            try:
                resp = await client.get(upstream_url, headers=_headers(referer))
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
        _last_request_at = time.monotonic()

    # Return upstream body unchanged. Filter hop-by-hop headers so httpx
    # downstream doesn't get confused.
    drop = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    headers_out = {k: v for k, v in resp.headers.items() if k.lower() not in drop}
    return Response(content=resp.content, status_code=resp.status_code, headers=headers_out)
