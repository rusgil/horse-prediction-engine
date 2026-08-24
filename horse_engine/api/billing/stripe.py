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
        self, *, user_id: int, email: Optional[str], success_url: str,
        price_id: str, days: int, plan: str,
    ) -> str:
        if not settings.stripe_secret_key or not price_id:
            raise RuntimeError("Stripe not configured (STRIPE_SECRET_KEY / price id)")
        data = {
            "mode": mode,   # 'payment' (5-day one-off) | 'subscription' (monthly/annual)
            "line_items[0][price]": price_id,
            "line_items[0][quantity]": "1",
            "success_url": success_url,
            "cancel_url": success_url,
            "client_reference_id": str(user_id),
            "metadata[user_id]": str(user_id),
            "metadata[days]": str(days),   # webhook grants exactly this many
            "metadata[plan]": str(plan),
        }
        if mode == "subscription":
            # Stamp the subscription itself so RENEWAL invoices can map back to
            # the user + grant length (checkout metadata only covers the 1st pay).
            data["subscription_data[metadata][user_id]"] = str(user_id)
            data["subscription_data[metadata][days]"] = str(days)
            data["subscription_data[metadata][plan]"] = str(plan)
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
        etype = evt.get("type")
        obj = (evt.get("data") or {}).get("object") or {}

        # First payment of ANY plan (one-off 5-day OR the first cycle of a sub).
        if etype == "checkout.session.completed":
            if obj.get("payment_status") not in (None, "paid"):
                log.info("stripe webhook: session not paid (%s)", obj.get("payment_status"))
                return None
            meta = obj.get("metadata") or {}
            return self._grant(
                meta.get("user_id") or obj.get("client_reference_id"),
                meta.get("days"),
                obj.get("payment_intent") or obj.get("id"),
                obj.get("amount_total"), obj.get("currency"))

        # Subscription RENEWALS (monthly/annual). Only the recurring cycles —
        # the first cycle is already granted by checkout.session.completed, so
        # skip subscription_create to avoid a double-grant. The user/days ride
        # on the subscription metadata we stamped at checkout.
        if etype in ("invoice.payment_succeeded", "invoice.paid"):
            if obj.get("billing_reason") not in ("subscription_cycle", "subscription_update"):
                return None
            meta = self._invoice_metadata(obj)
            return self._grant(
                meta.get("user_id"), meta.get("days"),
                obj.get("id"),  # invoice id — one per cycle, idempotency key
                obj.get("amount_paid") or obj.get("total"), obj.get("currency"))

        log.info("stripe webhook: ignoring event %s", etype)
        return None

    @staticmethod
    def _invoice_metadata(inv: dict) -> dict:
        """Best-effort pull of the subscription metadata off a renewal invoice —
        its location shifts across Stripe API shapes, so check the known spots.
        NOTE: verify against a live renewal event on the preview API."""
        for m in (
            (inv.get("subscription_details") or {}).get("metadata"),
            ((inv.get("parent") or {}).get("subscription_details") or {}).get("metadata"),
        ):
            if m:
                return m
        for ln in (inv.get("lines") or {}).get("data") or []:
            if ln.get("metadata"):
                return ln["metadata"]
        return {}

    @staticmethod
    def _grant(uid, days_str, txn, amount_minor, currency) -> Optional[GrantIntent]:
        try:
            user_id = int(uid)
        except (TypeError, ValueError):
            log.warning("stripe webhook: no usable user_id (%s)", uid)
            return None
        if not txn:
            log.warning("stripe webhook: no id for idempotency")
            return None
        try:
            days = int(days_str)
        except (TypeError, ValueError):
            days = settings.billing_pass_days
        amount = (amount_minor / 100.0) if isinstance(amount_minor, (int, float)) else None
        return GrantIntent(
            user_id=user_id, days=days, external_txn_id=str(txn),
            amount=amount, currency=(currency or "").upper() or None)
