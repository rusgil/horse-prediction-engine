"""Transactional email delivery via Resend.

Kept intentionally small — Resend's REST API only needs one endpoint.
If we ever swap providers (SES, Postmark), only this file changes; the
callers just call send_magic_link() etc.

All functions return True on delivery-accepted, False otherwise. They
never raise on delivery failure — the auth flow degrades gracefully
(user can retry request-code) rather than 500ing on a Resend outage.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from horse_engine.config import settings

log = logging.getLogger(__name__)

_RESEND_URL = "https://api.resend.com/emails"


async def _send(
    to_email: str,
    subject: str,
    html_body: str,
    text_body: Optional[str] = None,
    bcc: Optional[str] = None,
) -> bool:
    """Low-level Resend API call. Called by the higher-level send_* helpers.

    Fails closed to False on any error. Callers should log the outcome
    but continue — never crash the caller on a mail failure. Resend
    accepts HTML + optional plain-text alt; text is a good idea for
    spam-filter scoring.

    `bcc` supports a single blind-carbon-copy recipient. Used to shadow
    invite emails to the admin so you can see every invite that goes
    out without opening the DB — see send_invite_email() below.
    """
    if not settings.resend_api_key:
        log.warning("[mailer] RESEND_API_KEY unset — cannot send to %s", to_email)
        return False
    payload = {
        "from": settings.sender_email,
        "to": [to_email],
        "subject": subject,
        "html": html_body,
    }
    if bcc:
        # Resend accepts a list here even for a single BCC recipient.
        payload["bcc"] = [bcc]
    if text_body:
        payload["text"] = text_body
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _RESEND_URL,
                headers={
                    "Authorization": f"Bearer {settings.resend_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if resp.status_code // 100 == 2:
            log.info("[mailer] sent '%s' to %s (resend_id=%s)",
                     subject[:40], to_email, (resp.json() or {}).get("id"))
            return True
        log.warning("[mailer] Resend rejected send to %s: %d %s",
                    to_email, resp.status_code, resp.text[:200])
        return False
    except Exception as e:
        log.warning("[mailer] send to %s failed: %s", to_email, e)
        return False


async def send_invite_email(
    to_email: str,
    invite_url: str,
    inviter_email: Optional[str] = None,
) -> bool:
    """Send an invite link to a friend. Called from POST /api/invites/create
    when the caller opts in to `send_email=true`. `inviter_email` is used
    to personalise the subject / greeting when known.

    Same fail-closed semantics as send_magic_link — a Resend outage
    returns False, doesn't crash the endpoint. The invite row is still
    in the DB either way, so the caller can retry or copy the URL by
    hand.
    """
    inviter_label = inviter_email or "A Funky IQ member"
    subject = f"{inviter_label} invited you to Funky IQ"
    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:0 auto;padding:32px 24px;color:#111;">
      <h1 style="font-size:22px;font-weight:700;margin:0 0 20px">You're invited to Funky IQ</h1>
      <p style="font-size:15px;line-height:1.6;color:#333">
        {inviter_label} thinks you'd get something out of Funky IQ — a
        model-driven Australian horse-racing picks service. The site is
        invite-only right now, and this link claims a seat for you.
      </p>
      <p style="margin:28px 0"><a href="{invite_url}" style="display:inline-block;background:#22c55e;color:#07090f;text-decoration:none;padding:12px 22px;border-radius:6px;font-weight:700;letter-spacing:0.02em">Claim your seat</a></p>
      <p style="font-size:13px;color:#666;line-height:1.6">Or copy this URL into your browser:<br><span style="word-break:break-all;color:#333">{invite_url}</span></p>
      <p style="font-size:12px;color:#999;margin-top:32px;padding-top:20px;border-top:1px solid #eee">This invite expires in 30 days and can only be used once. If you weren't expecting it, you can ignore this email.</p>
    </div>
    """
    text = f"""You're invited to Funky IQ

{inviter_label} thinks you'd get something out of Funky IQ — a
model-driven Australian horse-racing picks service. The site is
invite-only right now, and this link claims a seat for you.

Claim your seat: {invite_url}

This invite expires in 30 days and can only be used once.
"""
    # BCC the invite to the first admin so every invite that goes out
    # is visible from the admin's inbox — no need to check the DB or
    # the /members admin console to see who was invited when.
    bcc = (settings.first_admin_email or "").strip() or None
    return await _send(to_email, subject, html, text, bcc=bcc)


async def send_magic_link(to_email: str, verify_url: str, intent: str = "login") -> bool:
    """Send the login/verification email. `verify_url` should be the full
    https URL a user clicks to complete authentication. intent shapes
    the subject line (login vs finish-signup) but the email body is the
    same shape either way.
    """
    is_signup = intent == "signup"
    subject = "Finish setting up your Funky IQ account" if is_signup else "Your Funky IQ login link"
    heading = "One more step" if is_signup else "Sign in to Funky IQ"
    cta_text = "Finish signup" if is_signup else "Log in"
    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:0 auto;padding:32px 24px;color:#111;">
      <h1 style="font-size:22px;font-weight:700;margin:0 0 20px">{heading}</h1>
      <p style="font-size:15px;line-height:1.6;color:#333">Click the button below to {cta_text.lower()}. This link expires in 15 minutes and can only be used once.</p>
      <p style="margin:28px 0"><a href="{verify_url}" style="display:inline-block;background:#111;color:#fff;text-decoration:none;padding:12px 20px;border-radius:6px;font-weight:600">{cta_text}</a></p>
      <p style="font-size:13px;color:#666;line-height:1.6">Or copy this URL into your browser:<br><span style="word-break:break-all;color:#333">{verify_url}</span></p>
      <p style="font-size:12px;color:#999;margin-top:32px;padding-top:20px;border-top:1px solid #eee">If you didn't request this email, you can safely ignore it. No account changes have been made.</p>
    </div>
    """
    text = f"""{heading}

{cta_text}: {verify_url}

This link expires in 15 minutes and can only be used once.

If you didn't request this, you can safely ignore this email.
"""
    return await _send(to_email, subject, html, text)
