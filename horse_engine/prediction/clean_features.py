"""
Recompute the BUG-18-contaminated aggregate-rate features in a history row's
``enriched_json`` using only ``HistoricalResultRow`` rows strictly before the
row's race date.

The contamination is in fields produced by ``_inject_accumulated_stats``:
trainer/jockey overall/track/distance/wet rates and horse career/track/distance/
condition rates. Form-history features (which always had the date filter), market
features, pedigree, gear, speed map etc. are read unchanged from ``enriched_json``.

Used by the retrain endpoints, calibration sweeps, and the contamination-audit
admin endpoint so training inputs match what production inference sees after the
BUG-18 fix on 2026-06-11.
"""
from __future__ import annotations

import bisect
import json
import logging
from collections import defaultdict
from typing import Iterable

from horse_engine.models.enriched import EnrichedRunner
from horse_engine.prediction.features import build_feature_vector

log = logging.getLogger(__name__)

# Same thresholds the original _inject_accumulated_stats used.
_MIN_PERSON_SAMPLES = 10
_MIN_HORSE_SAMPLES = 5
_DIST_TOLERANCE_M = 200


class AggregateIndex:
    """Precomputed per-entity sorted history of HistoricalResultRow objects.

    Lookups by (entity_type, entity_name, race_date_prefix) return all rows
    strictly before race_date_prefix in O(log N) via bisect.
    """

    def __init__(self, hr_rows: Iterable):
        self.by_horse: dict[str, list] = defaultdict(list)
        self.by_jockey: dict[str, list] = defaultdict(list)
        self.by_trainer: dict[str, list] = defaultdict(list)
        # Sort once on construction so every entity slice is chronologically ordered.
        for r in sorted(hr_rows, key=lambda x: x.race_id):
            if r.horse_name:
                self.by_horse[r.horse_name.lower()].append(r)
            if r.jockey:
                self.by_jockey[r.jockey].append(r)
            if r.trainer:
                self.by_trainer[r.trainer].append(r)
        # Precompute the race_id keys per entity for bisect_left.
        self._horse_keys = {k: [r.race_id for r in v] for k, v in self.by_horse.items()}
        self._jockey_keys = {k: [r.race_id for r in v] for k, v in self.by_jockey.items()}
        self._trainer_keys = {k: [r.race_id for r in v] for k, v in self.by_trainer.items()}

    def _slice_before(self, by: dict, keys: dict, entity: str, race_date_prefix: str) -> list:
        if not entity:
            return []
        rows = by.get(entity, [])
        if not rows:
            return []
        ks = keys.get(entity, [])
        idx = bisect.bisect_left(ks, race_date_prefix)
        return rows[:idx]

    def horse_history(self, horse_name: str, race_date_prefix: str) -> list:
        return self._slice_before(
            self.by_horse, self._horse_keys, (horse_name or "").lower(), race_date_prefix
        )

    def jockey_history(self, jockey: str, race_date_prefix: str) -> list:
        return self._slice_before(self.by_jockey, self._jockey_keys, jockey or "", race_date_prefix)

    def trainer_history(self, trainer: str, race_date_prefix: str) -> list:
        return self._slice_before(self.by_trainer, self._trainer_keys, trainer or "", race_date_prefix)


def _is_wet(track_condition: str | None) -> bool:
    tc = (track_condition or "").lower()
    return "soft" in tc or "heavy" in tc


def _rate_pct(wins: int, starts: int, threshold: int, default: float = 10.0) -> float:
    return (100.0 * wins / starts) if starts >= threshold else default


def _rate_frac(wins: int, starts: int) -> float:
    """0..1 fraction. No threshold/default — caller decides whether to gate."""
    return (wins / starts) if starts > 0 else 0.0


def _patch_clean_aggregates(er_data: dict, history_row, index: AggregateIndex) -> dict:
    """Mutate er_data in-place, replacing only the BUG-18-contaminated fields.

    Mirrors the field-set _inject_accumulated_stats writes to Runner+Jockey+
    Trainer stats, then the wr_*/rate features engine.enrich_runner derives
    from them. Other features (form, pedigree, market, gear, speed map) come
    straight from the original enriched_json.
    """
    race_date_prefix = history_row.race_id[:10]
    venue_lower = (er_data.get("venue") or getattr(history_row, "venue", None) or "").lower()
    distance = er_data.get("distance") or getattr(history_row, "distance", None) or 0
    dist_lo, dist_hi = distance - _DIST_TOLERANCE_M, distance + _DIST_TOLERANCE_M
    track_condition_cat = (er_data.get("track_condition_category") or "good").lower()
    is_wet_day = track_condition_cat in ("soft", "heavy")

    horse_name = history_row.horse_name or ""
    jockey = history_row.jockey or er_data.get("jockey") or ""
    trainer = history_row.trainer or er_data.get("trainer") or ""

    # ── Horse career / track / distance / condition ──────────────────────────
    horse_before = index.horse_history(horse_name, race_date_prefix)
    h_starts = len(horse_before)
    if h_starts >= _MIN_HORSE_SAMPLES:
        h_wins = sum(1 for r in horse_before if r.position == 1)
        h_places = sum(1 for r in horse_before if r.placed)
        track_starts = sum(1 for r in horse_before if (r.venue or "").lower() == venue_lower)
        track_wins = sum(
            1 for r in horse_before
            if (r.venue or "").lower() == venue_lower and r.position == 1
        )
        dist_starts = sum(
            1 for r in horse_before if r.distance and dist_lo <= r.distance <= dist_hi
        )
        dist_wins = sum(
            1 for r in horse_before
            if r.distance and dist_lo <= r.distance <= dist_hi and r.position == 1
        )
        wet_starts = sum(1 for r in horse_before if _is_wet(r.track_condition))
        wet_wins = sum(
            1 for r in horse_before if _is_wet(r.track_condition) and r.position == 1
        )
        dry_starts = max(0, h_starts - wet_starts)
        dry_wins = max(0, h_wins - wet_wins)

        er_data["win_rate_career"] = _rate_frac(h_wins, h_starts)
        er_data["career_place_rate"] = _rate_frac(h_places, h_starts)
        er_data["win_rate_track"] = _rate_frac(track_wins, track_starts)
        er_data["win_rate_distance"] = _rate_frac(dist_wins, dist_starts)
        # Track+distance combined — same formula as engine.py:wr_track_distance
        # (which uses runner.track_distance_starts/wins; we approximate with the
        # same venue+distance bucket).
        er_data["win_rate_track_distance"] = (
            _rate_frac(
                sum(
                    1 for r in horse_before
                    if (r.venue or "").lower() == venue_lower
                    and r.distance and dist_lo <= r.distance <= dist_hi
                    and r.position == 1
                ),
                sum(
                    1 for r in horse_before
                    if (r.venue or "").lower() == venue_lower
                    and r.distance and dist_lo <= r.distance <= dist_hi
                ),
            )
        )
        # Condition rate uses today's going
        if is_wet_day:
            cond_starts, cond_wins = wet_starts, wet_wins
        else:
            cond_starts, cond_wins = dry_starts, dry_wins
        er_data["win_rate_condition"] = _rate_frac(cond_wins, cond_starts)
        # Surface match in engine.py prefers wr_condition when condition_starts > 0
        if cond_starts > 0:
            er_data["surface_match_score"] = er_data["win_rate_condition"]
        # Wet/dry track records used by going_preference in build_feature_vector
        er_data["wet_track_record"] = _rate_frac(wet_wins, wet_starts)
        er_data["dry_track_record"] = _rate_frac(dry_wins, dry_starts)
    # If below the horse threshold we leave existing er_data values alone —
    # they came from form-history fallbacks in engine.enrich_runner which were
    # already clean (the form-history block always had the date filter).

    # ── Jockey rates (percentages, 0..100 then /100 in build_feature_vector) ─
    jockey_before = index.jockey_history(jockey, race_date_prefix)
    j_starts = len(jockey_before)
    j_wins = sum(1 for r in jockey_before if r.position == 1)
    j_track_starts = sum(1 for r in jockey_before if (r.venue or "").lower() == venue_lower)
    j_track_wins = sum(
        1 for r in jockey_before
        if (r.venue or "").lower() == venue_lower and r.position == 1
    )
    j_dist_starts = sum(
        1 for r in jockey_before if r.distance and dist_lo <= r.distance <= dist_hi
    )
    j_dist_wins = sum(
        1 for r in jockey_before
        if r.distance and dist_lo <= r.distance <= dist_hi and r.position == 1
    )
    er_data["jockey_overall_rate"] = _rate_pct(j_wins, j_starts, _MIN_PERSON_SAMPLES, default=10.0)
    er_data["jockey_track_rate"] = _rate_pct(
        j_track_wins, j_track_starts, _MIN_PERSON_SAMPLES, default=10.0
    )
    er_data["jockey_distance_rate"] = _rate_pct(
        j_dist_wins, j_dist_starts, _MIN_PERSON_SAMPLES, default=10.0
    )
    # jockey_booking_significance is derived from jockey_overall_rate in engine.py
    er_data["jockey_booking_significance"] = max(
        0.0,
        min(1.0, (er_data["jockey_overall_rate"] - 15.0) / 15.0)
        if er_data["jockey_overall_rate"] > 0 else 0.0,
    )

    # ── Trainer rates ────────────────────────────────────────────────────────
    trainer_before = index.trainer_history(trainer, race_date_prefix)
    t_starts = len(trainer_before)
    t_wins = sum(1 for r in trainer_before if r.position == 1)
    t_track_starts = sum(1 for r in trainer_before if (r.venue or "").lower() == venue_lower)
    t_track_wins = sum(
        1 for r in trainer_before
        if (r.venue or "").lower() == venue_lower and r.position == 1
    )
    t_dist_starts = sum(
        1 for r in trainer_before if r.distance and dist_lo <= r.distance <= dist_hi
    )
    t_dist_wins = sum(
        1 for r in trainer_before
        if r.distance and dist_lo <= r.distance <= dist_hi and r.position == 1
    )
    t_wet_starts = sum(1 for r in trainer_before if _is_wet(r.track_condition))
    t_wet_wins = sum(
        1 for r in trainer_before if _is_wet(r.track_condition) and r.position == 1
    )
    er_data["trainer_overall_rate"] = _rate_pct(
        t_wins, t_starts, _MIN_PERSON_SAMPLES, default=10.0
    )
    er_data["trainer_track_rate"] = _rate_pct(
        t_track_wins, t_track_starts, _MIN_PERSON_SAMPLES, default=10.0
    )
    er_data["trainer_distance_rate"] = _rate_pct(
        t_dist_wins, t_dist_starts, _MIN_PERSON_SAMPLES, default=10.0
    )
    er_data["trainer_wet_rate"] = _rate_pct(
        t_wet_wins, t_wet_starts, _MIN_PERSON_SAMPLES, default=10.0
    )
    # stable_form is derived from trainer_overall_rate in engine.py
    er_data["stable_form"] = max(
        0.0,
        min(1.0, (er_data["trainer_overall_rate"] - 10.0) / 20.0)
        if er_data["trainer_overall_rate"] > 0 else 0.0,
    )

    return er_data


def safe_enriched_runner(er_data: dict) -> EnrichedRunner | None:
    """Construct an EnrichedRunner from a dict, surviving null values for
    non-Optional fields in old enriched_json.

    Pre-2026-06-11 enriched_json rows were sometimes written with ``null``
    for fields like ``career_place_rate`` that EnrichedRunner declares as
    ``float = 0.0`` (strict, not Optional). Pydantic rejects None on those.
    By dropping None values before construction, the model defaults kick in
    for those fields — which is the desired behaviour anyway.

    Returns None if construction fails for any other reason.
    """
    try:
        clean = {
            k: v for k, v in er_data.items()
            if k in EnrichedRunner.model_fields and v is not None
        }
        return EnrichedRunner(**clean)
    except Exception as e:
        log.debug("safe_enriched_runner failed: %s", e)
        return None


def recompute_clean_feature_vector(history_row, index: AggregateIndex) -> list[float] | None:
    """Build a feature vector for one history row using BUG-18-fixed aggregates.

    Returns None if anything goes wrong (missing enriched_json, malformed JSON,
    EnrichedRunner validation failure, build_feature_vector error, etc.). Callers
    rely on None as the signal to fall back to the raw enriched_json path —
    NEVER re-raise from this function. If we did, the caller's outer try/except
    would swallow the error WITHOUT firing the fallback (the calibration sweep
    bug found 2026-06-11 13:50 AEST).
    """
    if not history_row.enriched_json:
        return None
    try:
        er_data = json.loads(history_row.enriched_json)
        er_data = _patch_clean_aggregates(er_data, history_row, index)
        er = safe_enriched_runner(er_data)
        if er is None:
            return None
        return build_feature_vector(er)
    except Exception as e:
        log.debug(
            "Clean recompute failed for %s/%s: %s",
            history_row.race_id, history_row.horse_name, e,
        )
        return None


def fallback_feature_vector(history_row) -> list[float] | None:
    """Build a feature vector from raw enriched_json with safe pydantic
    construction. Used as the fallback when recompute_clean_feature_vector
    returns None — replaces the previous ``EnrichedRunner(**json.loads(...))``
    pattern that crashed on pre-fix rows with null career_place_rate.
    """
    if not history_row.enriched_json:
        return None
    try:
        er_data = json.loads(history_row.enriched_json)
        er = safe_enriched_runner(er_data)
        if er is None:
            return None
        return build_feature_vector(er)
    except Exception as e:
        log.debug(
            "Fallback feature vector failed for %s/%s: %s",
            history_row.race_id, history_row.horse_name, e,
        )
        return None


def contamination_diff(history_row, index: AggregateIndex) -> dict:
    """Return a per-field old → new comparison for audit purposes.

    Used by the /api/admin/contamination-audit endpoint to quantify how much
    BUG-18 actually moved each contaminated field on a given history row.
    """
    if not history_row.enriched_json:
        return {"error": "no enriched_json"}
    old = json.loads(history_row.enriched_json)
    new = _patch_clean_aggregates(json.loads(history_row.enriched_json), history_row, index)
    fields = [
        "win_rate_career", "career_place_rate",
        "win_rate_track", "win_rate_distance", "win_rate_track_distance",
        "win_rate_condition", "surface_match_score",
        "wet_track_record", "dry_track_record",
        "jockey_overall_rate", "jockey_track_rate", "jockey_distance_rate",
        "jockey_booking_significance",
        "trainer_overall_rate", "trainer_track_rate", "trainer_distance_rate", "trainer_wet_rate",
        "stable_form",
    ]
    diff = {}
    for f in fields:
        ov = old.get(f)
        nv = new.get(f)
        if ov is None and nv is None:
            continue
        delta = (nv - ov) if (ov is not None and nv is not None) else None
        diff[f] = {"old": ov, "new": nv, "delta": (round(delta, 4) if delta is not None else None)}
    return {
        "race_id": history_row.race_id,
        "horse_name": history_row.horse_name,
        "jockey": history_row.jockey,
        "trainer": history_row.trainer,
        "fields": diff,
    }
