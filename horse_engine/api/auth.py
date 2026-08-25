"""Auth utilities — magic links, sessions, cookies, dependency injection.

Design summary:
  - User enters email → we generate a random 32-byte token, store its
    SHA-256 hash in magic_links with 15-min expiry, email the raw token
    as part of a verify URL.
  - User clicks link → GET /api/auth/verify?t=<raw_token>. We hash it,
    look up the row, mark used_at, then create a SessionRow with a fresh
    random 32-byte session token. Return HttpOnly cookie with the raw
    session token.
  - Every subsequent request reads the cookie, hashes it, looks up the
    session, and hands the associated UserRow to the endpoint.

Nothing in this file talks to Stripe or the subscription table. That
lives in subscription-checking dependencies elsewhere. Auth here just
identifies the user; subscription status gates paid endpoints.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Cookie, HTTPException, Request, Response
from sqlalchemy import select, delete, func
from sqlalchemy.exc import IntegrityError

from horse_engine.config import settings
from horse_engine.models.database import (
    MagicLinkRow, SessionRow, UserRow,
)
# get_session lives in api.database, NOT models.database. First deploy of
# this module failed startup with ImportError until this was split.
from horse_engine.api.database import get_session

log = logging.getLogger(__name__)


# Public constants — used by main.py to set cookies and craft URLs.
COOKIE_NAME = "fiq_session"
MAGIC_LINK_TTL = timedelta(minutes=15)
SESSION_TTL = timedelta(days=7)


# ── Token helpers ──────────────────────────────────────────────────────

def generate_token(nbytes: int = 32) -> str:
    """URL-safe 32-byte token — ~256 bits of entropy. Used for both
    magic-link tokens and session cookies. Distinct token spaces (the
    two never collide because they're stored in different tables)."""
    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:
    """SHA-256 hex digest. Stored in DB; the raw token is only ever
    handed to the user (via email or cookie). A DB leak cannot be
    replayed as valid tokens."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ── Magic-link flow ────────────────────────────────────────────────────

async def issue_magic_link(
    email: str,
    intent: str = "login",
    invite_token_hash: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    referral_source: Optional[str] = None,
) -> str:
    """Create a magic-link row, return the RAW token (caller emails it
    inside a verify URL). Never stores the raw token — only the hash.

    intent = 'login' for existing accounts, 'signup' when the flow
    should redirect to onboarding on verify. Signup profile fields
    (first/last name, referral source) ride along and are applied to the
    account on creation.
    """
    email_norm = email.strip().lower()
    token = generate_token()
    now = datetime.utcnow()
    async with get_session() as session:
        session.add(MagicLinkRow(
            email=email_norm,
            token_hash=hash_token(token),
            intent=intent,
            invite_token_hash=invite_token_hash,
            first_name=(first_name or None),
            last_name=(last_name or None),
            referral_source=(referral_source or None),
            created_at=now,
            expires_at=now + MAGIC_LINK_TTL,
        ))
        await session.commit()
    return token


async def consume_magic_link(token: str) -> Optional[MagicLinkRow]:
    """Look up + validate a magic-link token. Returns the row while
    still within its 15-min TTL, regardless of how many times it's
    been hit. None on any failure.

    Multi-use within TTL: email-scanner services (Gmail Safe Browsing,
    Outlook link protection, corporate proxies) hit magic-link URLs
    zero to many times BEFORE the user's real click, over intervals
    that can stretch minutes. A once-consumed token would strand the
    user. We treat the token as valid throughout its expires_at
    window instead — first-write on used_at stamps for observability
    but never rejects a subsequent hit.

    Security: token entropy is 256 bits (32 random URL-safe bytes) and
    is only ever transmitted over TLS. TTL is 15 min. A stolen token
    is a) implausible to obtain, b) dead almost immediately anyway.
    Multi-use within TTL doesn't materially widen the exploit surface.

    Fail closed: never signal *why* validation failed, to avoid oracle
    attacks that distinguish 'expired' from 'wrong token' etc.
    """
    if not token:
        return None
    now = datetime.utcnow()
    h = hash_token(token)
    async with get_session() as session:
        row = (await session.execute(
            select(MagicLinkRow).where(MagicLinkRow.token_hash == h).limit(1)
        )).scalars().first()
        if row is None:
            return None
        if row.expires_at < now:
            return None
        # Stamp used_at on first consume for observability. Subsequent
        # hits within TTL are allowed without re-stamping (so used_at
        # records the crawler's fetch time, not the user's).
        if row.used_at is None:
            row.used_at = now
            await session.commit()
        return row


# ── Session flow ──────────────────────────────────────────────────────

async def create_session(user_id: int, request: Optional[Request] = None) -> str:
    """Insert a SessionRow, return the RAW cookie value (caller sets it
    into a Set-Cookie header). last_seen_at is initialised to now; it
    gets touched on each authed request via touch_session().
    """
    token = generate_token()
    now = datetime.utcnow()
    ua = request.headers.get("user-agent", "")[:512] if request is not None else None
    ip = None
    if request is not None:
        # Prefer leftmost forwarded IP (Railway sits behind a proxy).
        xff = request.headers.get("x-forwarded-for", "")
        ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else None)
    async with get_session() as session:
        session.add(SessionRow(
            user_id=user_id,
            cookie_hash=hash_token(token),
            created_at=now,
            expires_at=now + SESSION_TTL,
            last_seen_at=now,
            user_agent=ua,
            ip_address=ip,
        ))
        await session.commit()
    return token


async def revoke_session(cookie_token: str) -> None:
    """Delete the row matching this cookie. Idempotent — a missing row
    is fine (logout after already-expired session is a no-op).
    """
    if not cookie_token:
        return
    h = hash_token(cookie_token)
    async with get_session() as session:
        await session.execute(delete(SessionRow).where(SessionRow.cookie_hash == h))
        await session.commit()


async def revoke_all_sessions_for_user(user_id: int) -> int:
    """Kill every session for a user — 'log me out everywhere'. Also
    used by admin 'force-logout this user' actions. Returns count."""
    async with get_session() as session:
        result = await session.execute(delete(SessionRow).where(SessionRow.user_id == user_id))
        await session.commit()
        return result.rowcount or 0


async def touch_session(cookie_hash: str) -> None:
    """Update last_seen_at. Fire-and-forget — a failure here doesn't
    block the request. Called from the auth dependency on every
    protected route."""
    try:
        async with get_session() as session:
            row = (await session.execute(
                select(SessionRow).where(SessionRow.cookie_hash == cookie_hash).limit(1)
            )).scalars().first()
            if row is not None:
                row.last_seen_at = datetime.utcnow()
                await session.commit()
    except Exception as e:
        log.debug("[auth] touch_session skipped: %s", e)


# ── User lookup + creation ─────────────────────────────────────────────

async def get_or_create_user(
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    referral_source: Optional[str] = None,
) -> UserRow:
    """Return the user with this email, creating a fresh row if none
    exists. Called after magic-link verification — at which point we
    know the user controls this email address.

    Newly created users have role='member', no seat allocation, no
    invited_by attribution (those get set by the invite consumption
    flow, not here). Signup profile fields (name + referral) are applied
    only on creation; existing users keep whatever they already had.
    """
    email_norm = email.strip().lower()
    fn = (first_name or "").strip() or None
    ln = (last_name or "").strip() or None
    ref = (referral_source or "").strip() or None
    display = " ".join(p for p in (fn, ln) if p) or None
    async with get_session() as session:
        row = (await session.execute(
            select(UserRow).where(UserRow.email == email_norm).limit(1)
        )).scalars().first()
        if row is not None:
            return row
        # New account: assign the next sequential member number now (founding
        # = the first 100). member_number is UNIQUE, so on the rare concurrent
        # collision we roll back and retry with a fresh max.
        for _ in range(5):
            max_num = (await session.execute(
                select(func.max(UserRow.member_number))
            )).scalar() or 0
            num = int(max_num) + 1
            row = UserRow(
                email=email_norm, role="member", member_number=num,
                founding=(num <= 100), created_at=datetime.utcnow(),
                first_name=fn, last_name=ln, referral_source=ref, name=display,
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                # Either the number was taken concurrently, or this email was
                # just created by a parallel request — re-check for the email.
                existing = (await session.execute(
                    select(UserRow).where(UserRow.email == email_norm).limit(1)
                )).scalars().first()
                if existing is not None:
                    return existing
                continue
            await session.refresh(row)
            return row
        # Exhausted retries (should never happen at our volume) — surface the
        # user if they now exist, else re-raise on a final attempt.
        existing = (await session.execute(
            select(UserRow).where(UserRow.email == email_norm).limit(1)
        )).scalars().first()
        if existing is not None:
            return existing
        raise RuntimeError("could not allocate member_number for new user")


async def get_user_by_cookie(cookie_token: Optional[str]) -> Optional[UserRow]:
    """Cookie → user, or None. Doesn't touch last_seen_at (caller can
    call touch_session with the same hash if it needs to).
    """
    if not cookie_token:
        return None
    h = hash_token(cookie_token)
    now = datetime.utcnow()
    async with get_session() as session:
        session_row = (await session.execute(
            select(SessionRow).where(SessionRow.cookie_hash == h).limit(1)
        )).scalars().first()
        if session_row is None or session_row.expires_at < now:
            return None
        user_row = (await session.execute(
            select(UserRow).where(UserRow.id == session_row.user_id).limit(1)
        )).scalars().first()
        return user_row


# ── FastAPI dependency helpers ─────────────────────────────────────────
# These are used with Depends() on protected endpoints. Callers do:
#   @app.get("/api/thing")
#   async def thing(user: UserRow = Depends(current_user)):
#       ...
# Two flavours: strict (raises 401) and lax (returns None).

async def current_user_optional(
    fiq_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> Optional[UserRow]:
    """Returns the user or None. Never raises. Use on endpoints that
    behave differently for anon vs authed but don't require auth."""
    if not fiq_session:
        return None
    user = await get_user_by_cookie(fiq_session)
    if user is not None:
        # Fire-and-forget touch so last_seen_at reflects activity.
        import asyncio
        asyncio.create_task(touch_session(hash_token(fiq_session)))
    return user


async def current_user(
    fiq_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> UserRow:
    """Strict — 401 if unauthenticated. Use on any protected endpoint
    that shouldn't be reached by anon users.

    The parameter list intentionally contains only Cookie/Header/Query
    typed params — never a bare `user: Optional[UserRow]` default. Any
    typed-with-a-non-Pydantic-class-and-a-default parameter is treated
    by FastAPI as a query/body field, which crashes app startup with
    "Invalid args for response field" because UserRow is a SQLAlchemy
    DeclarativeBase, not a Pydantic model. Do not add a `user=None`
    param to any Depends() function.
    """
    user = await current_user_optional(fiq_session)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def current_admin(
    fiq_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> UserRow:
    """Strict — 401 if unauthenticated, 403 if not admin. See
    current_user() docstring for why there's no `user=None` param."""
    user = await current_user_optional(fiq_session)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# ── Cookie header helper ───────────────────────────────────────────────

def set_session_cookie(response: Response, cookie_token: str) -> None:
    """Attach the session cookie with the right security flags.
    HttpOnly = JS can't read it (XSS mitigation).
    Secure = HTTPS only (Railway is behind TLS).
    SameSite=None = REQUIRED because the frontend (funkyiq.com on Vercel)
    fetches the API cross-origin (Railway). SameSite=Lax would not send
    the cookie on cross-origin fetch/XHR — only on top-level navigation.
    The CSRF risk of SameSite=None is mitigated by the 256-bit random
    session token (unguessable) and the fact that mutating endpoints
    all require a JSON body or explicit method, which browsers can't
    forge with a simple form POST.
    Path=/ = valid for all routes.
    Max-Age matches the DB expiry.
    """
    response.set_cookie(
        key=COOKIE_NAME,
        value=cookie_token,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        # Shared across *.funkyiq.com in prod (COOKIE_DOMAIN=.funkyiq.com) so
        # the product pages' same-origin /api/* rewrites carry the session and
        # the paywall can recognise members. Blank ⇒ host-only (dev/preview).
        domain=settings.cookie_domain or None,
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/", domain=settings.cookie_domain or None)
