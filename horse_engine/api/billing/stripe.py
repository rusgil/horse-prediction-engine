"""Stripe billing adapter (direct processor; you are merchant of record).

Implements BillingProvider for Stripe:
  - create_checkout → POST /v1/checkout/sessions (Bearer secret key, FORM-encoded
    — Stripe is not JSON), mode=payment, one-time price, metadata.user_id +
    client_reference_id; returns the hosted session URL.
  - verify_and_parse → verify the Stripe-Signature header (scheme
    "t=<ts>,v1=<hmac>", where the signed payload is "<t>.<raw_body>" and the mac
    is HMAC-SHA256 with the webhook signing secret); map checkout.session.completed
    → GrantIntent.

Test vs live is by KEY PREFIX (sk_test_ / sk_live_ / sk_), same api.stripe.com
base — no env switch needed. Stripe supports AUD, so the Price can be A$9.90.
Replay is covered by grant_access idempotency (external_txn_id = payment_intent),
so we don't enforce a timestamp tolerance and can't false-reject on clock skew.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Mapping, Optional

import httpx

from horse_engine.config import settings
from horse_engine.api.billing.base import BillingProvider, GrantIntent

log = logging.getLogger(__name__)
_API = "https://api.stripe.com/v1"


class StripeProvider(BillingProvider):
    name = "stripe"

    async def create_checkout(
        self, *, user_id: int, email: Optional[str], success_url: str
    ) -> str:
        if not settings.stripe_secret_key or not settings.billing_price_id:
            raise RuntimeError("Stripe not configured (STRIPE_SECRET_KEY / BILLING_PRICE_ID)")
        data = {
            "mode": "payment",
            "line_items[0][price]": settings.billing_price_id,
            "line_items[0][quantity]": "1",
            "success_url": success_url,
            "cancel_url": success_url,
            "client_reference_id": str(user_id),
            "metadata[user_id]": str(user_id),
        }
        # Managed Payments — Stripe is merchant of record + handles tax (needs the
        # preview API version header below and a Product with an eligible tax_code).
        if settings.stripe_managed_payments:
            data["managed_payments[enabled]"] = "true"
        if email:
            data["customer_email"] = email
        headers = {"Authorization": f"Bearer {settings.stripe_secret_key}"}
        if settings.stripe_api_version:
            headers["Stripe-Version"] = settings.stripe_api_version
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{_API}/checkout/sessions",
                headers=headers,
                data=data,  # Stripe wants application/x-www-form-urlencoded
            )
            resp.raise_for_status()
            body = resp.json()
        url = body.get("url")
        if not url:
            raise RuntimeError(f"Stripe checkout returned no url: {body}")
        return url

    def verify_and_parse(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> Optional[GrantIntent]:
        secret = settings.stripe_webhook_secret
        if not secret:
            raise RuntimeError("Stripe webhook secret not configured")
        sig_header = None
        for k, v in headers.items():
            if k.lower() == "stripe-signature":
                sig_header = v
                break
        if not sig_header:
            raise ValueError("missing Stripe-Signature")
        parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
        t, v1 = parts.get("t"), parts.get("v1")
        if not t or not v1:
            raise ValueError("malformed Stripe-Signature")
        signed = f"{t}.".encode() + raw_body
        expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, v1):
            raise ValueError("Stripe signature mismatch")

        evt = json.loads(raw_body)
        if evt.get("type") != "checkout.session.completed":
            log.info("stripe webhook: ignoring event %s", evt.get("type"))
            return None
        obj = (evt.get("data") or {}).get("object") or {}
        if obj.get("payment_status") not in (None, "paid"):
            log.info("stripe webhook: session not paid (%s)", obj.get("payment_status"))
            return None
        meta = obj.get("metadata") or {}
        raw_uid = meta.get("user_id") or obj.get("client_reference_id")
        try:
            user_id = int(raw_uid)
        except (TypeError, ValueError):
            log.warning("stripe webhook: no usable user_id (meta=%s, ref=%s)",
                        meta, obj.get("client_reference_id"))
            return None
        # payment_intent is the stable per-payment id → idempotency key.
        txn = obj.get("payment_intent") or obj.get("id")
        if not txn:
            log.warning("stripe webhook: no payment_intent/session id for idempotency")
            return None
        amt = obj.get("amount_total")
        amount = (amt / 100.0) if isinstance(amt, (int, float)) else None
        return GrantIntent(
            user_id=user_id,
            days=settings.billing_pass_days,
            external_txn_id=str(txn),
            amount=amount,
            currency=(obj.get("currency") or "").upper() or None,
        )
