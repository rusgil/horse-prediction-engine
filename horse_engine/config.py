from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    cron_secret: str = ""
    database_url: str = "sqlite+aiosqlite:///./horse_predictions.db"  # overridden by DATABASE_URL env var on Railway

    tab_base_url: str = "https://api.tab.com.au/v1/tab-info-service"
    tab_jurisdiction: str = "NSW"  # default jurisdiction for single-jurisdiction calls

    # Betfair Exchange API credentials
    betfair_app_key: str = ""
    betfair_username: str = ""
    betfair_password: str = ""
    # Hard kill switch — defaults to disabled. Betfair's identitysso.betfair.com.au
    # has been 403'ing our login attempts (account/IP block), so the stream client
    # was hitting it every 30 seconds with failed auth — exactly the API hammering
    # we forbid in feedback_no_api_hammer.md. Set BETFAIR_ENABLED=true in env vars
    # only when access has been confirmed working.
    betfair_enabled: bool = False

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
