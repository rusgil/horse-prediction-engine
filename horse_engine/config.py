from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    cron_secret: str = ""
    database_url: str = "sqlite+aiosqlite:///./horse_predictions.db"  # overridden by DATABASE_URL env var on Railway

    tab_base_url: str = "https://api.tab.com.au/v1/tab-info-service"
    tab_jurisdiction: str = "NSW"  # default jurisdiction for single-jurisdiction calls

    # Membership + auth (added 2026-07-20). Set via Railway env vars:
    #   RESEND_API_KEY       — key from resend.com dashboard (starts 're_')
    #   SENDER_EMAIL         — 'from' address for magic links, dunning, etc.
    #                          Must be at a verified Resend domain.
    #   APP_BASE_URL         — public URL used in outbound magic-link emails.
    #                          e.g. 'https://funkyiq.com'.
    #   FIRST_ADMIN_EMAIL    — this email gets role=admin on init_db bootstrap.
    #                          Change and redeploy is a no-op — bootstrap
    #                          runs once, thereafter the user row is normal.
    #   MEMBER_CAP           — total active-seat cap. First 100 = founding.
    #                          Raise this later via env var, no code change.
    resend_api_key: str = ""
    sender_email: str = "no-reply@funkyiq.com"
    # Cloudflare Turnstile (human check on /login, /waitlist, /invite).
    # Both blank → guard no-ops silently, endpoints still accept requests.
    # Once both are set (via Railway env vars) the guard activates:
    #   TURNSTILE_SITE_KEY   — public, returned by /api/config/public
    #                          so the frontend can render the widget.
    #   TURNSTILE_SECRET_KEY — private, used to verify the token
    #                          server-side against Cloudflare's siteverify.
    turnstile_site_key: str = ""
    turnstile_secret_key: str = ""
    # funkyiq.com is the parent brand. Horse-prediction lives on a
    # subdomain so future products (NRL predictions, etc.) get their
    # own subdomain under the same brand.
    app_base_url: str = "https://horse-racing-predictions.funkyiq.com"
    # Public API URL used in magic-link emails so the click lands as a
    # TOP-LEVEL navigation on the API host (Set-Cookie always works),
    # then the backend redirects the browser back to app_base_url. This
    # is the OAuth-style pattern — sidesteps iOS Safari's ITP handling
    # of Set-Cookie inside cross-origin fetch responses, which was the
    # root cause of the mobile "You're in → back to login" loop.
    api_base_url: str = "https://api.funkyiq.com"
    first_admin_email: str = "rusgil@gmail.com"
    member_cap: int = 1000

    # ── Billing (freemium 5-day pass, 2026-08-24) ─────────────────────
    # Provider-agnostic surface (Creem is merchant-of-record — handles
    # AU/global tax). Secrets are read server-side only, in the billing
    # adapter, and NEVER serialised to a client. Set via Railway env vars:
    #   BILLING_PROVIDER      — 'creem' (default) | 'stripe' | ...
    #   BILLING_ENV           — 'test' | 'production'  (drives Creem base URL)
    #   BILLING_PRICE_ID      — the $10 / 5-day product id (Creem product_id)
    #   CREEM_API_KEY         — server secret (x-api-key); creem_ / creem_test_
    #   CREEM_WEBHOOK_SECRET  — HMAC-SHA256 secret for creem-signature verify
    # Blank api_key/price_id ⇒ billing.enabled=false (Unlock routes to
    # /login instead of a checkout) — so this ships inert until keys land.
    # Master kill-switch for the freemium gate. Default ON (gating live).
    # Flip to false via PAYWALL_ENABLED=false on Railway to instantly open
    # the whole site back up — no code change / redeploy needed.
    paywall_enabled: bool = True
    billing_provider: str = "creem"
    billing_env: str = "test"
    # client_token stays for SDK-based providers (Stripe/Paddle); Creem
    # uses a server-created redirect checkout, so it needs no public token.
    billing_client_token: str = ""
    billing_price_id: str = ""
    # Creem charges USD/EUR only — no AUD. Price in USD; the button shows a
    # ~AUD approximation via billing_price_note (display only; the real charge
    # is the USD amount, converted by the buyer's bank). US$6.99 ≈ A$9.90.
    billing_price_amount: float = 6.99
    billing_currency: str = "USD"
    billing_price_note: str = "~A$9.90"
    billing_pass_days: int = 5
    # Creem (merchant of record).
    creem_api_key: str = ""
    creem_webhook_secret: str = ""
    # Session-cookie Domain. Default "" = host-only (works on localhost +
    # *.vercel.app previews). In prod set COOKIE_DOMAIN=.funkyiq.com so the
    # session cookie is shared across the apex + api + product subdomains —
    # REQUIRED for the paywall to recognise a logged-in member on the
    # product pages, which call /api/* same-origin (Vercel rewrite → Railway)
    # and otherwise wouldn't carry the api.funkyiq.com host-only cookie.
    cookie_domain: str = ""

    # Benter blend (2026-07-22): final ranking score is
    #   alpha·log(p_model) + beta·log(p_market)  →  softmax within race,
    # applied mass-preservingly (see engine._apply_benter_blend). Both 0.0
    # = disabled (current shrinkage-only behaviour). Enable by setting
    # BENTER_ALPHA / BENTER_BETA on Railway — a deliberate human release
    # action per [[feedback_model_release_process]]. Evidence 2026-07-22:
    # holdout top-1 21.8% (model) vs 35.0% (SP market) vs 35.1% (blend);
    # live-market LODO fit alpha=0.14 beta=0.53 (n=85, thin — refit as
    # clean-market data accumulates).
    benter_alpha: float = 0.0
    benter_beta: float = 0.0

    # R1-3 segment calibration (2026-08-25, Crucible-confirmed). The model
    # UNDER-rates early-card rank-1 picks: 180d scan (n=2918) predicted 21.4%
    # vs actual 30.2%, OOS-confirmed (train 1.63x, test 1.22x). When set >1.0,
    # the final rank-1 on races 1-3 is scaled by this factor (capped at 0.90).
    # Calibration only — does NOT change which horse is picked (rank order
    # preserved), so it lifts the displayed %, tier and Sharp-eligibility of
    # early-race top picks to match reality; it does not change win rate. Set
    # R1_3_SEGMENT_CALIB=1.22 (the conservative OOS-test ratio) on Railway to
    # enable. Default off — a deliberate human release action.
    r1_3_segment_calib: float = 0.0

    # +EV "value bet" gate (2026-08-25, Crucible-confirmed). The bet set that
    # survived out-of-sample walk-forward at positive ROI (180d flat SP): HOT
    # (rank-1 >= 46%, +28% ROI, OOS train +33% / test +23%) OR a HIGH pick
    # (36-45%) the MARKET DOUBTS (market_rank >= 3). Combined = ~+19% ROI,
    # OOS-stable. This is the profitable EDGE tier, distinct from the Sharp
    # QUALITY gate. When true, picks carry value_bet=true so the product can
    # surface / filter to the +EV set. Off by default — a human release action.
    value_bet_gate: bool = False

    # Market-defer guardrail (2026-08-19): on a marginal win-rank disagreement,
    # defer our rank-1 to the MARKET favourite. Fires only when the market
    # favourite is NOT our rank-1 AND our rank-1's win_prob leads the market
    # favourite's by LESS than this many percentage points — i.e. a thin call
    # the current model tends to get wrong by over-fading the market fav.
    # Crucible walk-forward (guardrail-backtest, days=60): at 3pp the OOS
    # test-split top-1 win rate lifted +1.3pp (28.9%→30.3%) with train intact
    # (+0.3pp); 5pp began hurting train (−0.4pp), so the band stays tight.
    #   0.0 = off (DEFAULT — enable as a deliberate release per
    #         [[feedback_model_release_process]] by setting MARKET_DEFER_EDGE_PP
    #         on Railway; 3.0 is the validated value). Instant rollback = set 0.
    market_defer_edge_pp: float = 0.0

    # Harville place probability (2026-07-24): derive P(top-3) for each runner
    # from the field's WIN probs via bets.harville_horse_top_n, instead of the
    # standalone place model (which was outputting near-uniform values, making
    # the win favourite rank mid-pack to place — Grafton R9). Blend weight:
    #   0.0 = pure trained place model (current behaviour, DEFAULT — off until
    #         backtested per [[feedback_model_release_process]])
    #   1.0 = pure Harville-from-win
    # Set PLACE_HARVILLE_WEIGHT on Railway to enable, as a deliberate release.
    place_harville_weight: float = 0.0

    # Blind-race shrinkage (2026-07-27): when a race has NO market odds at
    # prediction time ("blind" — late-market country meetings like Monday
    # Albury), the form-only model is over-confident (Albury R2: 60% claimed
    # on a horse while the $1.95 market favourite sat ranked 7th at 0.5%).
    # λ blends every win prob toward the field mean, mass- and rank-
    # preserving: p_i ← (1-λ)·p_i + λ·(mass/n). Ordering, top-1 pick and
    # win-rate are untouched; only the probability MAGNITUDES (tier badges,
    # Sharp gates, Harville place) are tempered. 0 = off (default until
    # backtested per feedback_model_release_process). Review
    # /api/admin/backtest/blind-shrinkage then set BLIND_SHRINKAGE on Railway.
    blind_shrinkage: float = 0.0
    # NOTE (2026-07-27): hard win/place display caps were considered after the
    # Albury blind-race incident and deliberately REJECTED — clamping extremes
    # would have masked the canary (a 98% place claim is what exposed the
    # blind-race bug). Instead, extreme claims raise a nightly quality-check
    # alert, and BLIND_SHRINKAGE calibrates the genuinely-blind races.

    # Betfair clients were removed 2026-06-13. RA + OddsPro cover everything
    # the model trains on; the Betfair-derived features (steam_60, steam_30,
    # drift_flag, odds_velocity, late_money, odds_movement_norm) ablated to
    # ~0 or net-harmful on the win model. extra="ignore" means leftover
    # BETFAIR_* env vars on Railway are silently ignored — feel free to
    # delete them from the Railway dashboard at your leisure.

    @property
    def async_database_url(self) -> str:
        """Ensure the URL uses an async driver."""
        url = self.database_url
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


settings = Settings()
