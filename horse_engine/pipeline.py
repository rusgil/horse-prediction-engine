"""
Full enrichment + prediction pipeline for one race.
Called by API endpoints and cron.
"""
from __future__ import annotations

import logging
from typing import Optional

from horse_engine.models.race import Race
from horse_engine.prediction.engine import predict_race, RunnerPrediction
from horse_engine.prediction.model import HorseModel
from horse_engine.prediction.narrative import generate_race_narratives

log = logging.getLogger(__name__)


async def enrich_and_predict_race(
    race: Race,
    model: HorseModel,
    generate_narratives: bool = True,
) -> tuple[list[RunnerPrediction], dict]:
    """
    Run the full pipeline:
      1. Predict race (enrichment + logistic regression)
      2. Generate Claude narratives for top 6
    Returns (predictions, metadata).
    """
    if not race.runners:
        log.warning("Race %s has no runners", race.race_id)
        return [], {}

    log.info(
        "Predicting %s R%s — %dm %s (%d runners)",
        race.venue, race.race_number, race.distance,
        race.track_condition, len(race.runners),
    )

    predictions = predict_race(race, model)

    if generate_narratives:
        try:
            await generate_race_narratives(
                predictions=predictions,
                race_distance=race.distance,
                track_condition=race.track_condition,
            )
        except Exception as e:
            log.warning("Narrative generation failed for %s: %s", race.race_id, e)

    meta = {
        "race_id": race.race_id,
        "venue": race.venue,
        "distance": race.distance,
        "track_condition": race.track_condition,
        "runners": len(predictions),
        "top_pick": predictions[0].runner.horse_name if predictions else None,
    }

    return predictions, meta


async def enrich_meeting(
    race_date: str,
    venue_code: str,
    model: HorseModel,
) -> list[dict]:
    """Enrich all races at a meeting. Returns summary list."""
    from horse_engine.clients.tab import TABClient
    tab = TABClient()
    meetings = await tab.get_meetings(race_date)
    meeting_meta = next((m for m in meetings if m.get("meetingCode") == venue_code), {})
    venue_name = meeting_meta.get("venueName", venue_code)
    state = meeting_meta.get("location", {}).get("state", "")

    raw_races = await tab.get_meeting_races(race_date, venue_code)
    summaries = []

    for raw in raw_races:
        race_num = raw.get("raceNumber")
        try:
            full = await tab.get_race(race_date, venue_code, race_num)
            if not full:
                continue
            race = tab.parse_race(full, race_date, venue_name, state)
            predictions, meta = await enrich_and_predict_race(race, model)
            summaries.append({**meta, "status": "ok"})
        except Exception as e:
            log.warning("Pipeline failed R%s at %s: %s", race_num, venue_code, e)
            summaries.append({"race_number": race_num, "status": "error", "error": str(e)})

    return summaries
