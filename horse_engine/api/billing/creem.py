"""Creem billing adapter (merchant of record).

Implements BillingProvider for Creem:
  - create_checkout → POST {base}/v1/checkouts (x-api-key), returns checkout_url
  - verify_and_parse → verify creem-signature (HMAC-SHA256 hex of raw body),
    map a 'checkout.completed' event to a GrantIntent

Docs verified 2026-08-24:
  base    prod  https://api.creem.io/v1     test  https://test-api.creem.io/v1
  auth    header  x-api-key: <creem_ | creem_test_ key>
  webhook header  creem-signature = hex(HMAC_SHA256(raw_body, webhook_secret))
  payload eventType, object.id (ch_), object.order.id (ord_),
          object.order.amount (minor units), object.order.currency,
          object.customer.email/id, object.request_id, object.metadata
We pass metadata.user_id (+ request_id) at checkout so the webhook maps the
payment back to OUR user — never relying on Creem's customer id as the key.
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


def _base_url() -> str:
    # .strip() — a stray space in BILLING_ENV must never silently flip
    # test↔production (observed " test" with a leading space, 2026-08-24).
    env = (settings.billing_env or "").strip().lower()
    test = env.startswith("test") or env in ("sandbox", "dev", "")
    return "https://test-api.creem.io/v1" if test else "https://api.creem.io/v1"


class CreemProvider(BillingProvider):
    name = "creem"

    async def create_checkout(
        self, *, user_id: int, email: Optional[str], success_url: str
    ) -> str:
        if not settings.creem_api_key or not settings.billing_price_id:
            raise RuntimeError("Creem not configured (CREEM_API_KEY / BILLING_PRICE_ID)")
        payload: dict = {
            "product_id": settings.billing_price_id,
            "request_id": str(user_id),          # echoed back in the webhook
            "metadata": {"user_id": str(user_id)},
            "success_url": success_url,
        }
        if email:
            payload["customer"] = {"email": email}
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{_base_url()}/checkouts",
                headers={"x-api-key": settings.creem_api_key,
                         "content-type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        url = data.get("checkout_url") or data.get("url")
        if not url:
            raise RuntimeError(f"Creem checkout returned no url: {data}")
        return url

    def verify_and_parse(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> Optional[GrantIntent]:
        secret = settings.creem_webhook_secret
        if not secret:
            raise RuntimeError("Creem webhook secret not configured")
        # Case-insensitive header lookup (Starlette lower-cases, but be safe).
        sig = None
        for k, v in headers.items():
            if k.lower() == "creem-signature":
                sig = v
                break
        computed = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        if not sig or not hmac.compare_digest(computed, sig):
            raise ValueError("creem-signature mismatch")

        evt = json.loads(raw_body)
        etype = evt.get("eventType") or evt.get("type")
        obj = evt.get("object") or {}
        # Only the one-off pass purchase grants access. Subscription/refund/
        # dispute events are acknowledged (200) but not acted on here.
        if etype != "checkout.completed":
            log.info("creem webhook: ignoring event %s", etype)
            return None

        meta = obj.get("metadata") or {}
        raw_uid = meta.get("user_id") or obj.get("request_id")
        try:
            user_id = int(raw_uid)
        except (TypeError, ValueError):
            log.warning("creem webhook: no usable user_id (meta=%s request_id=%s)",
                        meta, obj.get("request_id"))
            return None

        order = obj.get("order") or {}
        txn = order.get("id") or obj.get("id")
        if not txn:
            log.warning("creem webhook: no order/checkout id to key idempotency on")
            return None
        amt = order.get("amount")
        amount = (amt / 100.0) if isinstance(amt, (int, float)) else None
        return GrantIntent(
            user_id=user_id,
            days=settings.billing_pass_days,
            external_txn_id=str(txn),
            amount=amount,
            currency=order.get("currency"),
        )
