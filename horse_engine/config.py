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
