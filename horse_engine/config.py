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
    # funkyiq.com is the parent brand. Horse-prediction lives on a
    # subdomain so future products (NRL predictions, etc.) get their
    # own subdomain under the same brand.
    app_base_url: str = "https://horse-racing-predictions.funkyiq.com"
    first_admin_email: str = "rusgil@gmail.com"
    member_cap: int = 1000

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
