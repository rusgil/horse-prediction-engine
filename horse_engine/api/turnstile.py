"""Cloudflare Turnstile verification.

Turnstile is Cloudflare's captcha alternative — invisible for real
users, blocks scripted abuse. The frontend widget produces a token;
this module verifies it against Cloudflare's siteverify endpoint.

Behavior when unconfigured:
  - If TURNSTILE_SECRET_KEY is unset, verify() returns True unconditionally
    and endpoints continue accepting requests without the token.
  - This lets us ship the code before the Cloudflare account exists
    and turn the check on later via env var, no redeploy of frontend
    beyond the site-key drop.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from horse_engine.config import settings

log = logging.getLogger(__name__)

_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def is_enabled() -> bool:
    """True when both site + secret are configured. Frontend uses the
    site-key half to decide whether to render the widget; the guard
    below uses the secret-key half to decide whether to verify."""
    if settings.turnstile_disabled:
        return False
    return bool(settings.turnstile_secret_key and settings.turnstile_site_key)


async def verify(token: Optional[str], remote_ip: Optional[str] = None) -> bool:
    """Return True if the token verifies, or if Turnstile isn't
    configured (fail-open for the unconfigured case). False if the
    token is missing/invalid/expired when Turnstile IS configured.

    Cloudflare's siteverify accepts multipart/form-urlencoded; we
    match that with httpx's `data=` kwarg. `remote_ip` is optional
    but recommended — Cloudflare uses it as an anti-replay signal.
    """
    if settings.turnstile_disabled:
        # Explicit kill-switch — accept every request (rate limits still apply).
        return True
    if not settings.turnstile_secret_key:
        # Unconfigured — treat as pass-through so local dev + pre-config
        # deploys don't 403 every request.
        return True
    if not token:
        return False
    payload = {
        "secret": settings.turnstile_secret_key,
        "response": token,
    }
    if remote_ip:
        payload["remoteip"] = remote_ip
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_VERIFY_URL, data=payload)
        if resp.status_code // 100 != 2:
            log.warning("[turnstile] siteverify non-2xx: %d %s",
                        resp.status_code, resp.text[:120])
            return False
        body = resp.json() or {}
        success = bool(body.get("success"))
        if not success:
            # Log error-codes for debugging (invalid-input, expired, etc.).
            # See https://developers.cloudflare.com/turnstile/get-started/server-side-validation/
            log.info("[turnstile] verify FAILED — errors=%s hostname=%s",
                     body.get("error-codes"), body.get("hostname"))
        return success
    except Exception as e:
        # Cloudflare outage or network blip — fail closed to protect
        # against the endpoint being open when it thinks it's guarded.
        log.warning("[turnstile] verify raised: %s", e)
        return False
