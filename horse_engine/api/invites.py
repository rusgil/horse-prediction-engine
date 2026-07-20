"""Invite + waitlist helpers.

Design notes:
  - Raw invite codes are 16 chars of url-safe entropy (~96 bits). Short
    enough to be pasted in Slack / SMS, long enough that guessing is
    infeasible even without rate limiting.
  - The DB stores SHA-256(code); a leak of the invites table doesn't
    yield usable invites. Same pattern as magic_links / sessions.
  - `consume_invite` is the atomic step. Under Postgres we could rely
    on `UPDATE ... WHERE consumed_at IS NULL RETURNING id` for
    single-round-trip check-and-set, but the code path also has to
    return the row to the caller (they need issued_by_user_id for
    lineage). The pattern here is: SELECT with FOR UPDATE, verify
    still-valid, then UPDATE — under a single transaction so a
    concurrent verify can't consume the same invite twice.

Nothing in this module talks to email / Stripe. Consumers wire those
into main.py endpoints.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Literal

from sqlalchemy import func, select, update

from horse_engine.api.database import get_session
from horse_engine.models.database import InviteRow, UserRow, WaitlistRow

log = logging.getLogger(__name__)


# Default invite lifespan. Long enough that a friend can sit on the
# email for a couple of weeks; short enough that abandoned codes don't
# pile up in the table forever. Bulk admin mints can override.
INVITE_TTL = timedelta(days=30)

# Raw code length in bytes. 12 bytes → 16 url-safe chars → ~96 bits
# entropy, which is fine for a code that a human might read aloud.
_CODE_BYTES = 12


def hash_code(code: str) -> str:
    """SHA-256 hex of the raw invite code. Exposed because the verify
    path needs to store the hash on MagicLinkRow.invite_token_hash at
    request-code time, then look up + consume by hash at verify time."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


# Legacy shim so this module's own callers keep working while I migrate
# the public name. Not exported; keep helpers below using `hash_code`.
_hash_code = hash_code


def generate_code() -> str:
    """URL-safe alphanumeric-ish code (16 chars). Passes cleanly through
    query strings without encoding."""
    return secrets.token_urlsafe(_CODE_BYTES)


# ── Mint ────────────────────────────────────────────────────────────

async def mint_invite(
    issued_by_user_id: Optional[int],
    invited_email: Optional[str] = None,
    ttl: timedelta = INVITE_TTL,
) -> tuple[str, InviteRow]:
    """Create an invite row and return (raw_code, row).

    issued_by_user_id is None for admin bulk mints that don't attribute
    to a specific admin — the caller is expected to check admin auth
    before invoking. invited_email is normalised lowercase; None means
    'anyone can redeem this code'.

    Caller is responsible for decrementing the issuer's
    `invites_remaining` counter separately (we don't do it here because
    admin bulk mints skip the counter entirely).
    """
    code = generate_code()
    now = datetime.utcnow()
    row = InviteRow(
        code_hash=_hash_code(code),
        issued_to_email=(invited_email.strip().lower() if invited_email else None),
        issued_by_user_id=issued_by_user_id,
        created_at=now,
        expires_at=now + ttl,
    )
    async with get_session() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return code, row


# ── Resolve (read-only lookup) ──────────────────────────────────────

InviteStatus = Literal["valid", "used", "expired", "revoked", "unknown"]


async def resolve_invite(code: str) -> tuple[Optional[InviteRow], InviteStatus]:
    """Look up an invite by raw code. Never mutates. Used by:
      - GET /api/invites/{code} — landing page probe
      - POST /api/auth/request-code — pre-issue check so we return a
        useful error instead of a magic link the user can't use.

    Returns 'unknown' rather than 'expired'/'used' when the code doesn't
    exist — we don't want to give a probe an oracle for distinguishing
    'never existed' from 'was valid, now expired'.
    """
    if not code:
        return None, "unknown"
    h = _hash_code(code)
    async with get_session() as session:
        row = (await session.execute(
            select(InviteRow).where(InviteRow.code_hash == h).limit(1)
        )).scalars().first()
    if row is None:
        return None, "unknown"
    if row.revoked_at is not None:
        return row, "revoked"
    if row.consumed_at is not None:
        return row, "used"
    if row.expires_at < datetime.utcnow():
        return row, "expired"
    return row, "valid"


# ── Consume (atomic single-shot) ────────────────────────────────────

async def consume_invite_by_hash(code_hash: str, consumer_user_id: int) -> Optional[InviteRow]:
    """Same as consume_invite(), but the caller already has the hash.
    Used by the verify path, which only has the invite hash stored on
    MagicLinkRow (never the raw code)."""
    if not code_hash:
        return None
    now = datetime.utcnow()
    async with get_session() as session:
        result = await session.execute(
            update(InviteRow)
            .where(
                InviteRow.code_hash == code_hash,
                InviteRow.consumed_at.is_(None),
                InviteRow.revoked_at.is_(None),
                InviteRow.expires_at > now,
            )
            .values(consumed_at=now, consumed_by_user_id=consumer_user_id)
            .execution_options(synchronize_session=False)
        )
        if not (result.rowcount or 0):
            await session.commit()
            return None
        await session.commit()
        row = (await session.execute(
            select(InviteRow).where(InviteRow.code_hash == code_hash).limit(1)
        )).scalars().first()
        return row


async def consume_invite(code: str, consumer_user_id: int) -> Optional[InviteRow]:
    """Atomically mark an invite as consumed by consumer_user_id. Returns
    the mutated row on success, None if the invite is already
    used/expired/revoked/unknown (caller can still create the user; the
    lineage attribution just won't happen).

    Implementation: a conditional UPDATE that only fires when the row
    is still eligible. If rowcount is 0, someone else got there first
    (or the code was invalid) — no need for a follow-up SELECT to
    diagnose. If the caller wants a reason for logging, use
    resolve_invite() beforehand.
    """
    if not code:
        return None
    return await consume_invite_by_hash(hash_code(code), consumer_user_id)


# ── Issuer-side actions ─────────────────────────────────────────────

async def decrement_invites_remaining(user_id: int) -> bool:
    """Atomic decrement. Returns True on success, False when the user's
    counter is already 0 (caller should surface 'no invites left' to
    the UI without minting).

    Uses a conditional UPDATE so two concurrent invite-create requests
    from the same user can't dip below 0.
    """
    async with get_session() as session:
        result = await session.execute(
            update(UserRow)
            .where(UserRow.id == user_id, UserRow.invites_remaining > 0)
            .values(invites_remaining=UserRow.invites_remaining - 1)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        return bool(result.rowcount or 0)


async def list_invites_by_issuer(user_id: int, limit: int = 50) -> list[InviteRow]:
    """Newest first. Used by the dashboard 'my invites' panel so the
    user can see who they invited, whether it was accepted, and whether
    it's still live."""
    async with get_session() as session:
        rows = (await session.execute(
            select(InviteRow)
            .where(InviteRow.issued_by_user_id == user_id)
            .order_by(InviteRow.created_at.desc())
            .limit(limit)
        )).scalars().all()
    return list(rows)


async def revoke_invite(invite_id: int, requester_user_id: int, is_admin: bool) -> bool:
    """Idempotent revoke. Members can only revoke their own unconsumed
    invites; admins can revoke any. Returns True if the row transitioned
    to revoked, False if the requester isn't allowed or the invite is
    already used/revoked (no state change).
    """
    now = datetime.utcnow()
    async with get_session() as session:
        stmt = update(InviteRow).where(
            InviteRow.id == invite_id,
            InviteRow.consumed_at.is_(None),
            InviteRow.revoked_at.is_(None),
        )
        if not is_admin:
            stmt = stmt.where(InviteRow.issued_by_user_id == requester_user_id)
        result = await session.execute(
            stmt.values(revoked_at=now)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        return bool(result.rowcount or 0)


# ── Waitlist ────────────────────────────────────────────────────────

async def add_to_waitlist(
    email: str,
    source: Optional[str] = None,
    notes: Optional[str] = None,
) -> bool:
    """Idempotent insert. Returns True if a new row was created, False
    if the email was already on the list (in which case we don't
    overwrite source/notes — an admin may have annotated them).

    We swallow the unique-violation instead of raising because from the
    user's POV 'you're on the list' is the correct response whether
    it's their first submission or their fifth.
    """
    email_norm = email.strip().lower()
    if not email_norm or "@" not in email_norm:
        return False
    async with get_session() as session:
        existing = (await session.execute(
            select(WaitlistRow.id).where(WaitlistRow.email == email_norm).limit(1)
        )).scalar()
        if existing is not None:
            return False
        session.add(WaitlistRow(
            email=email_norm,
            source=source,
            notes=notes,
            created_at=datetime.utcnow(),
        ))
        try:
            await session.commit()
            return True
        except Exception as e:
            # Race with a concurrent insert on the same email.
            await session.rollback()
            log.debug("[invites] waitlist insert race for %s: %s", email_norm, e)
            return False


async def list_waitlist(limit: int = 200) -> list[WaitlistRow]:
    """Newest first. Used by the admin dashboard."""
    async with get_session() as session:
        rows = (await session.execute(
            select(WaitlistRow)
            .order_by(WaitlistRow.created_at.desc())
            .limit(limit)
        )).scalars().all()
    return list(rows)


# ── Cap enforcement ─────────────────────────────────────────────────

async def count_seats_taken() -> int:
    """Number of UserRows currently occupying a seat against member_cap.
    A 'seat' is any user with seat_active=True — trial or paid alike per
    the Phase 2 design decision ("seat is used during trial").

    Called from the verify endpoint before creating a new user: if
    this count is already >= member_cap, the incoming user goes on
    the waitlist instead of into users.
    """
    async with get_session() as session:
        n = (await session.execute(
            select(func.count()).select_from(UserRow).where(UserRow.seat_active.is_(True))
        )).scalar()
    return int(n or 0)
