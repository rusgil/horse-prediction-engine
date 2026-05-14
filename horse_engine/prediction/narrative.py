"""
Claude Haiku narrative generation for horse race predictions.

Each horse gets a 2–4 sentence analysis in punter-friendly Australian vernacular,
covering: form, key advantage, pedigree angle, and a value verdict.
"""
from __future__ import annotations

import logging
from typing import Optional

import anthropic

from horse_engine.config import settings
from horse_engine.models.enriched import EnrichedRunner

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a sharp Australian horse racing analyst — part form student,
part pedigree expert, part market watcher. You write concise, punter-friendly race summaries
in plain Australian English. Never use jargon without explaining it. Never hedge excessively —
be direct. You sound like the best analyst at the track, not a liability disclaimer.
Two to four sentences maximum per horse."""


def _build_prompt(er: EnrichedRunner, race_distance: int, track_condition: str, flags: list[str]) -> str:
    flag_str = "\n".join(flags) if flags else "No standout flags"
    first_up_note = ""
    if er.is_resuming:
        first_up_note = f"Resuming after {er.spell_weeks or '?'} weeks. "
    if er.is_first_start:
        first_up_note = "Race debutant — no prior starts. "

    return f"""Analyse {er.horse_name} for a {race_distance}m race on {track_condition} going.

KEY STATS:
- Recent form score: {er.form_score:.2f}/1.0
- Win rate at this distance: {er.win_rate_distance:.0%}
- Win rate at this track: {er.win_rate_track:.0%}
- Surface match score: {er.surface_match_score:.2f}/1.0
- Trainer {er.trainer} strike rate: {er.trainer_overall_rate:.0f}% overall, {er.trainer_track_rate:.0f}% at this track
- Jockey {er.jockey} strike rate: {er.jockey_overall_rate:.0f}% overall
- Trainer+Jockey combo rate: {er.trainer_jockey_combo_rate:.0f}%
- Sire: {er.sire_name} (dam's sire: {er.dam_sire_name})
- Distance aptitude from bloodlines: {er.distance_aptitude}
- Pedigree wet track score: {er.pedigree_wet_score:.1f}/10
- Barrier {er.barrier}, speed map position: {er.speed_map_position}
- Market rank: #{er.market_rank}, odds movement: {"steamed" if er.is_steamed else "drifted" if er.is_drifted else "stable"}
- Model win probability: {er.win_rate_career:.0%} career, value overlay: {er.overlay:+.0%}
- {first_up_note}Class change: {"easier" if er.class_change == 1 else "harder" if er.class_change == -1 else "same"}

KEY FLAGS:
{flag_str}

Write 2–4 punter-friendly sentences analysing this runner. Lead with the most compelling angle
(form, pedigree match, trainer/jockey angle, or market signal). Be direct. Australian tone."""


async def generate_narrative(
    er: EnrichedRunner,
    race_distance: int,
    track_condition: str,
    flags: list[str],
    client: Optional[anthropic.AsyncAnthropic] = None,
) -> str:
    if not settings.anthropic_api_key:
        return _fallback_narrative(er, flags)

    _client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    prompt = _build_prompt(er, race_distance, track_condition, flags)

    try:
        msg = await _client.messages.create(
            model=settings.narrative_model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        log.warning("Claude narrative failed for %s: %s", er.horse_name, e)
        return _fallback_narrative(er, flags)


async def generate_race_narratives(
    predictions: list,
    race_distance: int,
    track_condition: str,
) -> None:
    """Populate narrative field on each RunnerPrediction in-place. Top 6 only."""
    if not settings.anthropic_api_key:
        for p in predictions:
            p.narrative = _fallback_narrative(p.enriched, p.key_flags)
        return

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    # Only generate narratives for top 6 model picks (cost control)
    for pred in predictions[:6]:
        pred.narrative = await generate_narrative(
            er=pred.enriched,
            race_distance=race_distance,
            track_condition=track_condition,
            flags=pred.key_flags,
            client=client,
        )


def _fallback_narrative(er: EnrichedRunner, flags: list[str]) -> str:
    """Rule-based fallback when Claude is unavailable."""
    parts = []

    if er.form_score >= 0.7:
        parts.append(f"{er.horse_name} comes in on the back of strong recent form.")
    elif er.form_score <= 0.3:
        parts.append(f"{er.horse_name} has been inconsistent of late and needs to find improvement.")
    else:
        parts.append(f"{er.horse_name} shows solid if unspectacular recent form.")

    if er.distance_aptitude != "mile" and er.pedigree_distance_match >= 0.8:
        parts.append(f"The {er.sire_name} bloodline is a good fit for this {er.distance_aptitude} trip.")

    if er.trainer_jockey_combo_rate >= 25:
        parts.append(f"The {er.trainer}/{er.jockey} combination has been firing at {er.trainer_jockey_combo_rate:.0f}%.")

    if er.is_steamed:
        parts.append("Well supported in the market — worth watching.")
    elif er.is_drifted:
        parts.append("Money has moved away in betting — treat with caution.")

    return " ".join(parts) or f"{er.horse_name} — insufficient data for detailed analysis."
