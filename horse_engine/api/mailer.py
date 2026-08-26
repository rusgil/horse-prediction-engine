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


async def _log_email(email: str, kind: str, subject: str, ok: bool, provider_id: Optional[str]) -> None:
    """Best-effort record of a send for the admin customer view. Never raises
    into the mail path."""
    try:
        from datetime import datetime
        from horse_engine.api.database import get_session
        from horse_engine.models.database import EmailLogRow
        async with get_session() as s:
            s.add(EmailLogRow(
                email=(email or "").strip().lower(), kind=kind,
                subject=(subject or "")[:250], ok=bool(ok),
                provider_id=provider_id, sent_at=datetime.utcnow(),
            ))
            await s.commit()
    except Exception as e:
        log.debug("[mailer] email_log write failed: %s", e)


async def _send(
    to_email: str,
    subject: str,
    html_body: str,
    text_body: Optional[str] = None,
    bcc: Optional[str] = None,
    kind: str = "other",
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
        await _log_email(to_email, kind, subject, False, None)
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
            pid = (resp.json() or {}).get("id")
            log.info("[mailer] sent '%s' to %s (resend_id=%s)", subject[:40], to_email, pid)
            await _log_email(to_email, kind, subject, True, pid)
            return True
        log.warning("[mailer] Resend rejected send to %s: %d %s",
                    to_email, resp.status_code, resp.text[:200])
        await _log_email(to_email, kind, subject, False, None)
        return False
    except Exception as e:
        log.warning("[mailer] send to %s failed: %s", to_email, e)
        await _log_email(to_email, kind, subject, False, None)
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
    return await _send(to_email, subject, html, text, bcc=bcc, kind="invite")


async def send_magic_link(to_email: str, verify_url: str, intent: str = "login") -> bool:
    """Send the login/verification email. `verify_url` should be the full
    https URL a user clicks to complete authentication.

    intent used to shape the subject line ('login' vs 'signup') with a
    "Finish setting up your account" variant for the first click. A
    recipient (Anna, 2026-07-21) read that as "I need to complete a
    profile" and disengaged — there is no profile to complete. Both
    intents now use the same neutral "Sign in" copy; the only visible
    difference between first-click and subsequent clicks is which page
    they land on post-verify (backend routes signup → /account so the
    new member sees their invite widget).
    """
    subject = "Sign in to Funky IQ"
    heading = "Sign in to Funky IQ"
    cta_text = "Sign in"
    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:0 auto;padding:32px 24px;color:#111;">
      <h1 style="font-size:22px;font-weight:700;margin:0 0 20px">{heading}</h1>
      <p style="font-size:15px;line-height:1.6;color:#333">Click the button below to sign in. This link expires in 15 minutes and can only be used once.</p>
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
    return await _send(to_email, subject, html, text, kind="magic_link")

async def send_expiry_reminder(to_email: str, first_name: Optional[str],
                               expires_at, plans_url: str) -> bool:
    """Remind a 5-day-pass member that their pass is about to expire, with a
    link to the plans page to renew. `expires_at` is a naive-UTC datetime."""
    name = (first_name or "").strip()
    greeting = f"Hi {name}," if name else "Hi there,"
    try:
        from zoneinfo import ZoneInfo
        from datetime import timezone
        exp = expires_at.replace(tzinfo=timezone.utc).astimezone(
            ZoneInfo("Australia/Sydney")).strftime("%A %d %b, %-I:%M %p AEST")
    except Exception:
        exp = expires_at.strftime("%d %b %Y") if expires_at else "soon"
    subject = "Your FunkyIQ 5-Day Pass is about to expire"
    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:0 auto;padding:32px 24px;color:#111;">
      <h1 style="font-size:22px;font-weight:700;margin:0 0 16px">Your 5-Day Pass expires soon</h1>
      <p style="font-size:15px;line-height:1.6;color:#333">{greeting}</p>
      <p style="font-size:15px;line-height:1.6;color:#333">Your FunkyIQ 5-Day Pass ends <b>{exp}</b>. Keep your full access — including the Lounge, Hot Seat and Listings — by renewing, or step up to Monthly or Annual for the Edge and the daily Playbook.</p>
      <p style="margin:28px 0"><a href="{plans_url}" style="display:inline-block;background:#22c55e;color:#062b13;text-decoration:none;padding:12px 22px;border-radius:8px;font-weight:700;letter-spacing:0.02em">See plans &amp; renew</a></p>
      <p style="font-size:13px;color:#666;line-height:1.6">Or copy this URL into your browser:<br><span style="word-break:break-all;color:#333">{plans_url}</span></p>
      <p style="font-size:12px;color:#999;margin-top:32px;padding-top:20px;border-top:1px solid #eee">You're receiving this because you hold an active FunkyIQ pass.</p>
    </div>
    """
    text = f"""Your FunkyIQ 5-Day Pass expires soon

{greeting}

Your 5-Day Pass ends {exp}. Renew or upgrade to keep your access:
{plans_url}
"""
    return await _send(to_email, subject, html, text, kind="expiry_reminder")
