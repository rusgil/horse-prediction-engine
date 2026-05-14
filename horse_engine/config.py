from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    cron_secret: str = ""
    tab_jurisdiction: str = "NSW"
    database_url: str = "sqlite+aiosqlite:///./horse_predictions.db"

    tab_base_url: str = "https://api.tab.com.au/v1/tab-info-service"
    # Odds API endpoints (public, no key required for TAB odds)
    sportsbet_base_url: str = "https://www.sportsbet.com.au"
    ladbrokes_base_url: str = "https://www.ladbrokes.com.au"

    narrative_model: str = "claude-haiku-4-5-20251001"


settings = Settings()
