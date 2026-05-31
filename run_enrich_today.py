"""
Run today's enrichment locally — replicates what the 6am cron job does.

Usage:
  DATABASE_URL=postgresql://... python run_enrich_today.py

The DATABASE_URL should point to the Railway PostgreSQL database so results
are written to production. Get it from Railway dashboard → Variables.
"""
import asyncio
import logging
import sys
from datetime import date, timedelta
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

_AEST = ZoneInfo("Australia/Sydney")


async def main():
    from horse_engine.models.database import init_db
    from horse_engine.clients.factory import get_tab_client
    from horse_engine.api.main import (
        _enrich_date,
        _seed_results_for_date,
        _load_model,
    )
    from horse_engine.api.database import get_session

    await init_db()

    client = get_tab_client()
    async with get_session() as session:
        model = await _load_model(session)

    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # Pre-warm jockey/trainer cache so parse_race fires zero extra GQL calls
    if hasattr(client, "prefetch_people_for_date"):
        log.info("Pre-fetching jockey/trainer stats for %s ...", today)
        await client.prefetch_people_for_date(today)

    # Enrich today
    log.info("=== Enriching %s ===", today)
    await _enrich_date(today, client, model)

    # Seed results for yesterday and today
    for d in (yesterday, today):
        n = await _seed_results_for_date(d)
        if n:
            log.info("Seeded %d results for %s", n, d)

    log.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
