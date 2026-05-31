"""
Full enrichment + prediction pipeline for one race.
Called by API endpoints and cron.
"""
from __future__ import annotations

import logging
from typing import Optional

from horse_engine.models.race import Race
from horse_engine.prediction.engine import predict_race, RunnerPrediction
from horse_engine.prediction.model import HorseModel, PlaceModel, ExoticModel
from horse_engine.prediction.narrative import generate_race_narratives

log = logging.getLogger(__name__)


async def enrich_and_predict_race(
    race: Race,
    model: HorseModel,
    generate_narratives: bool = True,
    venue_calibration: dict[str, float] | None = None,
    place_model: PlaceModel | None = None,
    exotic_model: ExoticModel | None = None,
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

    predictions = predict_race(race, model, venue_calibration=venue_calibration, place_model=place_model, exotic_model=exotic_model)

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
    venue_slug: str,
    model: HorseModel,
) -> list[dict]:
    """Enrich all races at a meeting. Returns summary list."""
    from horse_engine.clients.factory import get_tab_client
    client = get_tab_client()

    meeting_slug = f"{venue_slug}-{race_date.replace('-', '')}"
    meeting_detail = await client.get_meeting_by_slug(meeting_slug)
    if not meeting_detail:
        log.warning("Meeting not found: %s", meeting_slug)
        return []

    venue_obj = meeting_detail.get("venue") or {}
    venue_name = venue_obj.get("name", venue_slug)
    state = venue_obj.get("state", "")

    raw_events = await client.get_meeting_races(meeting_slug)
    summaries = []

    for raw_event in raw_events:
        race_num = raw_event.get("eventNumber")
        try:
            full_event = await client.get_race(meeting_slug, race_num)
            if not full_event:
                continue
            race = await client.parse_race(full_event, race_date, venue_name, state)
            predictions, meta = await enrich_and_predict_race(race, model)
            summaries.append({**meta, "status": "ok"})
        except Exception as e:
            log.warning("Pipeline failed R%s at %s: %s", race_num, venue_slug, e)
            summaries.append({"race_number": race_num, "status": "error", "error": str(e)})

    return summaries
