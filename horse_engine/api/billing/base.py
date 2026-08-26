"""Billing provider boundary — the strippable seam.

Everything provider-specific lives behind this interface. The rest of the
app only ever touches:
  - get_provider()                     (in __init__.py)
  - provider.create_checkout(...)      -> a redirect URL
  - provider.verify_and_parse(...)     -> a GrantIntent or None
  - access.grant_access(...)           (the domain core, provider-neutral)

Swapping Creem → Stripe → … in the future is: add a sibling module that
implements BillingProvider, register it in get_provider(). No caller changes,
no schema changes (AccessGrantRow is already provider-tagged).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, runtime_checkable


@dataclass
class GrantIntent:
    """A verified, provider-neutral 'grant this user paid access' instruction,
    produced from a validated webhook. Consumed by access.grant_access()."""
    user_id: int
    days: int
    external_txn_id: str          # provider txn/order id — UNIQUE, gives idempotency
    amount: Optional[float] = None    # decimal major units (e.g. 10.00)
    currency: Optional[str] = None    # 'AUD'


@runtime_checkable
class BillingProvider(Protocol):
    name: str

    async def create_checkout(
        self, *, user_id: int, email: Optional[str], success_url: str,
        price_id: str, days: int, plan: str, mode: str = "payment",
        discount_coupon: Optional[str] = None,
    ) -> str:
        """Create a hosted checkout for the given plan and return the redirect
        URL. `mode` is 'payment' (one-off, e.g. the 5-day pass) or 'subscription'
        (recurring, e.g. monthly/annual). `days` and `plan` are stamped into the
        payment metadata (and the subscription metadata, for renewals) so the
        webhook grants the right access length. Raises on failure."""
        ...

    def verify_and_parse(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> Optional[GrantIntent]:
        """Verify the webhook signature against raw_body, and if it's a
        successful-payment event, return a GrantIntent. Returns None for a
        valid-but-non-grant event (so the caller 200s it). Raises only when the
        signature is INVALID (caller returns 400)."""
        ...
