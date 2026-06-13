from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    cron_secret: str = ""
    database_url: str = "sqlite+aiosqlite:///./horse_predictions.db"  # overridden by DATABASE_URL env var on Railway

    tab_base_url: str = "https://api.tab.com.au/v1/tab-info-service"
    tab_jurisdiction: str = "NSW"  # default jurisdiction for single-jurisdiction calls

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
