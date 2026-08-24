"""Paid-access domain core — provider-agnostic.

The rest of the app should import only two things from here:
  - has_active_access(user) -> bool   — THE paywall check
  - grant_access(...)                  — extend a user's paid window + ledger it

Nothing in this module knows the word "Paddle" (or Stripe, or any other
provider). Billing adapters live in horse_engine/api/billing/*; each one
verifies its own webhook then calls grant_access(). Swapping providers in
6 months never touches this file. See UserRow.access_until + AccessGrantRow.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select

from horse_engine.models.database import UserRow, AccessGrantRow
# get_session lives in api.database, NOT models.database (see auth.py note).
from horse_engine.api.database import get_session

log = logging.getLogger(__name__)

# Roles that always see full content regardless of payment — staff/testing.
_ALWAYS_ACCESS_ROLES = frozenset({"admin", "power_user"})


def has_active_access(user: Optional[UserRow], *, now: Optional[datetime] = None) -> bool:
    """True if this user may see full (un-paywalled) picks right now.

    Deliberately tiny and reads ONLY UserRow — no provider state, no
    network, no query. A logged-out user (None) is never a member.
    """
    if user is None:
        return False
    if user.role in _ALWAYS_ACCESS_ROLES:
        return True
    # Pre-existing power-user comp hook: any non-empty plan override grants
    # access (used to exercise production flows without a real payment).
    if getattr(user, "test_plan_override", None):
        return True
    au = user.access_until
    if au is None:
        return False
    return au > (now or datetime.utcnow())


async def grant_access(
    *,
    user_id: int,
    days: int,
    provider: str,
    external_txn_id: str,
    amount: Optional[float] = None,
    currency: Optional[str] = None,
) -> Optional[datetime]:
    """Extend a user's paid window by `days` and record it in the ledger.

    Idempotent on `external_txn_id`: if a grant for that transaction
    already exists we no-op and return the current access_until — so a
    webhook retry (or a double-fired event) can never double-grant.

    Stacking: access_until = max(now, current access_until) + days, so
    buying again while still active adds to the tail instead of resetting.

    Returns the new access_until (or the existing value on a duplicate),
    or None if the user_id doesn't resolve.
    """
    now = datetime.utcnow()
    async with get_session() as session:
        # Idempotency guard — external_txn_id is UNIQUE in the schema.
        existing = (await session.execute(
            select(AccessGrantRow).where(
                AccessGrantRow.external_txn_id == external_txn_id
            ).limit(1)
        )).scalars().first()
        if existing is not None:
            log.info("grant_access: duplicate txn %s ignored", external_txn_id)
            user = (await session.execute(
                select(UserRow).where(UserRow.id == user_id).limit(1)
            )).scalars().first()
            return user.access_until if user else existing.access_until_after

        user = (await session.execute(
            select(UserRow).where(UserRow.id == user_id).limit(1)
        )).scalars().first()
        if user is None:
            log.warning("grant_access: no user id=%s for txn %s", user_id, external_txn_id)
            return None

        # Stack from the later of (now, current tail).
        base = user.access_until if (user.access_until and user.access_until > now) else now
        new_until = base + timedelta(days=days)
        user.access_until = new_until
        # NOTE: intentionally NOT touching seat_active / member_number here.
        # Those belong to the founding-cap flow; the paywall gate reads only
        # access_until. Seat/founding bookkeeping gets wired in Stage 2 when
        # real payments land, to avoid interacting with the cap logic now.

        session.add(AccessGrantRow(
            user_id=user_id,
            provider=provider,
            external_txn_id=external_txn_id,
            days_granted=days,
            amount=amount,
            currency=currency,
            access_until_after=new_until,
            created_at=now,
        ))
        await session.commit()
        log.info(
            "grant_access: user=%s +%dd via %s txn=%s -> access_until=%s",
            user_id, days, provider, external_txn_id, new_until.isoformat(),
        )
        return new_until
