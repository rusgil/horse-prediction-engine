"""Billing package — provider dispatch.

get_provider() returns the active BillingProvider based on
settings.billing_provider. Adding a provider = implement BillingProvider in a
sibling module and add one line here. Nothing else in the app names a provider.
"""
from __future__ import annotations

from horse_engine.config import settings
from horse_engine.api.billing.base import BillingProvider, GrantIntent

_PROVIDERS = {}


def get_provider() -> BillingProvider:
    name = (settings.billing_provider or "creem").lower()
    prov = _PROVIDERS.get(name)
    if prov is not None:
        return prov
    if name == "creem":
        from horse_engine.api.billing.creem import CreemProvider
        prov = CreemProvider()
    elif name == "stripe":
        from horse_engine.api.billing.stripe import StripeProvider
        prov = StripeProvider()
    else:
        raise RuntimeError(f"Unknown billing provider: {name!r}")
    _PROVIDERS[name] = prov
    return prov


__all__ = ["get_provider", "BillingProvider", "GrantIntent"]
