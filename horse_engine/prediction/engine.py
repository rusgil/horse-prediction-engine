"""
Prediction engine — orchestrates enrichment + model for a full race.

For each race:
  1. Extract form features per runner
  2. Enrich with trainer/jockey, pedigree, market, barrier, surface, speed map signals
  3. Build feature vectors
  4. Run model → win/place probabilities
  5. Compute value (overlay = model_prob - market_implied_prob)
  6. Rank runners
  7. Generate key flags
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from horse_engine.enrichers import form as form_enricher
from horse_engine.enrichers.market import compute_market_features
from horse_engine.enrichers.speed_map import build_speed_map, infer_running_style, pace_scenario_score
from horse_engine.enrichers.surface import barrier_score as compute_barrier_score, track_bias_advantage
from horse_engine.models.enriched import EnrichedRunner
from horse_engine.models.race import Race, Runner
from horse_engine.pedigree.sire_profiles import (
    get_distance_aptitude,
    get_sire_profile,
    get_wet_track_pedigree_score,
    parse_condition_category,
)
from horse_engine.prediction.features import build_feature_vector
from horse_engine.prediction.model import HorseModel, PlaceModel, ExoticModel

log = logging.getLogger(__name__)


class RunnerPrediction:
    def __init__(
        self,
        runner: Runner,
        enriched: EnrichedRunner,
        win_prob: float,
        place_prob: float,
        feature_vector: list[float],
    ):
        self.runner = runner
        self.enriched = enriched
        self.win_prob = win_prob
        self.place_prob = place_prob
        self.feature_vector = feature_vector
        self.model_rank: int = 0
        self.place_model_rank: int = 0   # rank by dedicated place model
        self.exotic_model_rank: int = 0  # rank by exotic (trifecta) model
        self.overlay: float = 0.0
        self.value_rating: float = 0.0
        self.key_flags: list[str] = []
        self.narrative: Optional[str] = None


def enrich_runner(
    runner: Runner,
    race: Race,
    market_features: dict,
    speed_map: dict[str, str],
    pace_advantage: dict[str, float],
    field_avg_weight: float,
) -> EnrichedRunner:
    """Convert raw Runner + race context into EnrichedRunner with all features."""
    starts = runner.last_10_starts
    condition_cat = parse_condition_category(race.track_condition)

    # ── Form ─────────────────────────────────────────────────────────────
    fs = form_enricher.weighted_form_score(starts)
    avg_beaten = form_enricher.avg_beaten_margin_last5(starts)
    best_finish = form_enricher.best_finish_at_distance(starts, race.distance)
    days_last = form_enricher.days_since_last_run(starts)
    runs_prep = form_enricher.runs_this_prep(starts)
    class_chg = form_enricher.class_change_flag(starts, race.race_class)
    spell_w = form_enricher.spell_weeks(starts)

    # Career win rate — prefer Punters direct stats (accurate), fall back to form starts
    if runner.career_starts > 0:
        wr_career = runner.career_wins / runner.career_starts
    else:
        wr_career = form_enricher.career_win_rate(runner)

    # Distance win rate — require >= 3 starts to avoid 50%/100% small-sample noise
    if runner.distance_starts >= 3:
        wr_distance = runner.distance_wins / runner.distance_starts
    else:
        wr_distance = form_enricher.win_rate_at_distance(starts, race.distance)

    # Track win rate — prefer track+distance > track > form-based
    if runner.track_distance_starts >= 2:
        wr_track = runner.track_distance_wins / runner.track_distance_starts
    elif runner.track_starts > 0:
        wr_track = runner.track_wins / runner.track_starts
    else:
        wr_track = form_enricher.win_rate_at_track(starts, race.venue)

    wr_class = form_enricher.win_rate_at_class(starts, race.race_class)

    # Track+distance combined (new feature)
    wr_track_distance = (
        runner.track_distance_wins / runner.track_distance_starts
        if runner.track_distance_starts > 0 else 0.0
    )

    # Condition win rate — actual results in today's going
    if runner.condition_starts > 0:
        wr_condition = runner.condition_wins / runner.condition_starts
    else:
        wr_condition = 0.0

    # First-up / second-up win rates
    wr_first_up = (
        runner.first_up_wins / runner.first_up_starts
        if runner.first_up_starts > 0 else 0.0
    )
    wr_second_up = (
        runner.second_up_wins / runner.second_up_starts
        if runner.second_up_starts > 0 else 0.0
    )

    # Surface match — prefer Punters condition stats, fall back to form-based
    if runner.condition_starts > 0:
        surface_match = wr_condition
    else:
        surface_match = form_enricher.surface_match_score(starts, race.track_condition)

    wet_record = form_enricher.wet_track_record(starts)
    dry_record = form_enricher.dry_track_record(starts)

    # ── Barrier / surface ─────────────────────────────────────────────────
    bs = compute_barrier_score(
        venue=race.venue,
        distance=race.distance,
        barrier=runner.barrier,
        rail_position=race.rail_position,
        field_size=len(race.runners),
    )

    # ── Trainer stats ─────────────────────────────────────────────────────
    ts = runner.trainer_stats
    trainer_overall = ts.win_rate_overall if ts else 10.0
    trainer_track = ts.win_rate_track if ts else 10.0
    trainer_dist = ts.win_rate_distance if ts else 10.0
    trainer_first_up = ts.win_rate_first_up if ts else 10.0
    trainer_wet = ts.win_rate_wet if ts else 10.0

    # ── Jockey stats ──────────────────────────────────────────────────────
    js = runner.jockey_stats
    jockey_overall = js.win_rate_overall if js else 10.0
    jockey_track = js.win_rate_track if js else 10.0
    jockey_dist = js.win_rate_distance if js else 10.0
    jockey_combo = js.trainer_jockey_combo_rate if js else 10.0
    jockey_wins_today = js.wins_today if js else 0

    # Barrier-range jockey rate
    barrier = runner.barrier
    if barrier <= 5:
        jockey_barrier = js.win_rate_barrier_low if js else 10.0
    elif barrier <= 10:
        jockey_barrier = js.win_rate_barrier_mid if js else 10.0
    else:
        jockey_barrier = js.win_rate_barrier_wide if js else 10.0

    # ── Pedigree ──────────────────────────────────────────────────────────
    sire_name = runner.pedigree.sire if runner.pedigree else "Unknown"
    dam_sire_name = runner.pedigree.dam_sire if runner.pedigree else "Unknown"
    dist_match = get_distance_aptitude(sire_name, race.distance)
    wet_ped_score = get_wet_track_pedigree_score(sire_name, dam_sire_name, condition_cat)
    sire_p = get_sire_profile(sire_name)
    dist_apt = sire_p["aptitude"] if sire_p else "mile"
    first_up_ped = sire_p["first_up"] if sire_p else 5.0
    dosage = sire_p["dosage_index"] if sire_p else 3.0

    # ── Market ────────────────────────────────────────────────────────────
    mf = market_features.get(runner.horse_name, {})
    mkt_rank = mf.get("market_rank", len(race.runners))
    mkt_implied = mf.get("market_implied_prob", 0.0)
    odds_move = mf.get("odds_movement", 0.0)
    best_odds = mf.get("best_available_odds", runner.fixed_win_odds or 0.0)
    is_steamed = mf.get("is_steamed", False)
    is_drifted = mf.get("is_drifted", False)

    # ── Speed map ─────────────────────────────────────────────────────────
    speed_pos = speed_map.get(runner.horse_name, "midfield")
    pace_adv = pace_advantage.get(runner.horse_name, 0.0)
    tb_adv = track_bias_advantage(speed_pos, race.rail_position, race.venue)

    # ── Gear ──────────────────────────────────────────────────────────────
    gear_changes: list[str] = runner.gear_changes
    gear_score = 0.3 if "blinkers_on" in gear_changes else (-0.1 if "blinkers_off" in gear_changes else 0.0)

    # ── Weight ────────────────────────────────────────────────────────────
    weight_vs_avg = runner.weight - field_avg_weight

    # ── Unique signals ────────────────────────────────────────────────────
    # Stable form: rough proxy = trainer's season win rate vs 10% baseline
    stable_form_score = min((trainer_overall - 10.0) / 20.0, 1.0) if trainer_overall > 0 else 0.0

    # Jockey booking significance: last-minute top-jockey booking
    # Simple proxy: jockey overall rate vs median (15%)
    jockey_booking_sig = min((jockey_overall - 15.0) / 15.0, 1.0) if jockey_overall > 0 else 0.0

    # Age/sex weight factor: mares and younger horses carry less weight on scale
    age_sex_factor = _age_sex_weight_factor(runner.sex, runner.age, race.distance)

    is_resuming = runs_prep == 0 and days_last >= 60

    return EnrichedRunner(
        horse_name=runner.horse_name,
        barrier=barrier,
        tab_number=runner.tab_number,
        jockey=runner.jockey,
        trainer=runner.trainer,
        weight=runner.weight,
        form_score=fs,
        win_rate_career=wr_career,
        win_rate_distance=wr_distance,
        win_rate_track=wr_track,
        win_rate_class=wr_class,
        avg_beaten_margin_last5=avg_beaten,
        best_finish_distance=best_finish,
        days_since_last_run=days_last,
        runs_this_prep=runs_prep,
        class_change=class_chg,
        prizemoney_career=runner.career_wins * 5000,  # rough estimate
        barrier_score=bs,
        weight_vs_field_avg=weight_vs_avg,
        wet_track_record=wet_record,
        dry_track_record=dry_record,
        surface_match_score=surface_match,
        track_condition_category=condition_cat,
        trainer_overall_rate=trainer_overall,
        trainer_track_rate=trainer_track,
        trainer_distance_rate=trainer_dist,
        trainer_first_up_rate=trainer_first_up,
        trainer_wet_rate=trainer_wet,
        trainer_jockey_combo_rate=jockey_combo,
        jockey_overall_rate=jockey_overall,
        jockey_track_rate=jockey_track,
        jockey_distance_rate=jockey_dist,
        jockey_barrier_rate=jockey_barrier,
        jockey_wins_today=jockey_wins_today,
        pedigree_distance_match=dist_match,
        pedigree_wet_score=wet_ped_score,
        pedigree_first_up_score=first_up_ped,
        sire_name=sire_name,
        dam_sire_name=dam_sire_name,
        distance_aptitude=dist_apt,
        market_rank=mkt_rank,
        market_implied_prob=mkt_implied,
        odds_movement=odds_move,
        best_available_odds=best_odds,
        overlay=0.0,  # filled after model
        is_steamed=is_steamed,
        is_drifted=is_drifted,
        speed_map_position=speed_pos,
        speed_map_advantage=pace_adv + tb_adv,
        gear_changes=gear_changes,
        gear_change_score=gear_score,
        is_first_start=len(runner.last_10_starts) == 0,
        is_resuming=is_resuming,
        spell_weeks=spell_w,
        age_sex_weight_factor=age_sex_factor,
        dosage_index=dosage,
        international_form=runner.country not in ("AUS", "NZ"),
        stable_form=max(0.0, min(1.0, stable_form_score)),
        jockey_booking_significance=max(0.0, min(1.0, jockey_booking_sig)),
        track_bias_advantage=tb_adv,
        win_rate_track_distance=wr_track_distance,
        win_rate_condition=wr_condition,
        first_up_win_rate=wr_first_up,
        second_up_win_rate=wr_second_up,
    )


def predict_race(race: Race, model: HorseModel, venue_calibration: dict[str, float] | None = None, place_model: PlaceModel | None = None, exotic_model: ExoticModel | None = None) -> list[RunnerPrediction]:
    """Full prediction pipeline for one race. Returns ranked RunnerPredictions."""
    if not race.runners:
        return []

    # Field context
    valid_weights = [r.weight for r in race.runners if r.weight > 0]
    field_avg_weight = sum(valid_weights) / len(valid_weights) if valid_weights else 58.0

    market_features = compute_market_features(race.runners)
    speed_map = build_speed_map(race.runners, race.distance)
    pace_advantage = pace_scenario_score(speed_map)

    enriched_runners: list[EnrichedRunner] = []
    for runner in race.runners:
        try:
            er = enrich_runner(
                runner=runner,
                race=race,
                market_features=market_features,
                speed_map=speed_map,
                pace_advantage=pace_advantage,
                field_avg_weight=field_avg_weight,
            )
            enriched_runners.append(er)
        except Exception as e:
            log.warning("Enrich failed for %s: %s", runner.horse_name, e)

    from horse_engine.prediction.venue_calibration import apply_venue_calibration
    feature_vectors = [build_feature_vector(er) for er in enriched_runners]
    win_probs, place_probs = model.predict_field(feature_vectors)
    win_probs = apply_venue_calibration(list(win_probs), race.venue, venue_calibration or {})

    predictions: list[RunnerPrediction] = []
    for i, (runner, er) in enumerate(zip(race.runners[:len(enriched_runners)], enriched_runners)):
        wp = win_probs[i] if i < len(win_probs) else 0.0
        pp = place_probs[i] if i < len(place_probs) else 0.0
        overlay = wp - er.market_implied_prob
        er.overlay = round(overlay, 4)

        pred = RunnerPrediction(
            runner=runner,
            enriched=er,
            win_prob=wp,
            place_prob=pp,
            feature_vector=feature_vectors[i],
        )
        pred.overlay = overlay
        pred.key_flags = _generate_flags(er, wp, overlay)
        predictions.append(pred)

    # Rank by win probability
    predictions.sort(key=lambda p: p.win_prob, reverse=True)
    for rank, p in enumerate(predictions, 1):
        p.model_rank = rank
        p.value_rating = _value_rating(p.win_prob, p.enriched.best_available_odds, p.overlay)

    # Run place model if provided — assigns place_model_rank independently
    if place_model is not None:
        place_win_probs, _ = place_model.predict_field(feature_vectors)
        place_ranked = sorted(
            zip(place_win_probs, predictions),
            key=lambda x: x[0],
            reverse=True,
        )
        for place_rank, (_, p) in enumerate(place_ranked, 1):
            p.place_model_rank = place_rank

    # Run exotic model if provided — assigns exotic_model_rank independently
    if exotic_model is not None:
        exotic_scores, _ = exotic_model.predict_field(feature_vectors)
        exotic_ranked = sorted(
            zip(exotic_scores, predictions),
            key=lambda x: x[0],
            reverse=True,
        )
        for exotic_rank, (_, p) in enumerate(exotic_ranked, 1):
            p.exotic_model_rank = exotic_rank

    return predictions


def _generate_flags(er: EnrichedRunner, win_prob: float, overlay: float) -> list[str]:
    flags = []
    if er.is_steamed:
        flags.append("💰 STEAMED — money in")
    if er.is_drifted:
        flags.append("📉 DRIFTED — market moving away")
    if overlay > 0.15:
        flags.append(f"🎯 VALUE OVERLAY +{overlay:.0%}")
    if er.runs_this_prep == 0 and er.pedigree_first_up_score >= 7:
        flags.append("🔥 First-up specialist bloodlines")
    if er.is_resuming and er.trainer_first_up_rate >= 20:
        flags.append(f"✅ Trainer fires {er.trainer_first_up_rate:.0f}% first-up")
    if er.track_condition_category in ("soft", "heavy") and er.pedigree_wet_score >= 7:
        flags.append("🌧 Wet-track pedigree — loves the mud")
    if er.track_condition_category in ("soft", "heavy") and er.wet_track_record >= 0.4:
        flags.append(f"🌧 {er.wet_track_record:.0%} wet track win rate")
    if er.jockey_wins_today >= 2:
        flags.append(f"🔥 Jockey {er.jockey} on fire — {er.jockey_wins_today} wins today")
    if er.trainer_jockey_combo_rate >= 25:
        flags.append(f"🤝 Deadly combo: {er.trainer_jockey_combo_rate:.0f}% strike rate together")
    if er.class_change == 1:
        flags.append("📉 Dropping in class — easier assignment")
    if er.gear_changes and "blinkers_on" in er.gear_changes:
        flags.append("👓 Blinkers on first time — watch for improvement")
    if er.dosage_index > 5.5 and er.win_rate_distance > 0.3:
        flags.append("⚡ Pure speed pedigree — ideal at this distance")
    if er.form_score >= 0.75:
        flags.append("📈 Outstanding recent form")
    if er.barrier_score >= 0.7 and er.speed_map_position in ("leader", "on_pace"):
        flags.append("🚪 Ideal barrier for on-pace runner")
    if er.international_form:
        flags.append("🌏 International form — watch debutants adjust")
    if er.is_first_start:
        flags.append("🐎 Race debutant — unknown quantity")
    return flags[:5]  # cap at 5 most relevant flags


def _value_rating(win_prob: float, best_odds: float, overlay: float) -> float:
    """
    Composite value score: combines probability edge and odds quality.
    >0 = value bet, <0 = no value.
    """
    if not best_odds or best_odds <= 1.0:
        return 0.0
    expected_value = win_prob * best_odds - 1.0
    return round(expected_value + overlay * 0.5, 4)


def _age_sex_weight_factor(sex: str, age: int, distance: int) -> float:
    """
    Weight-for-age scale adjustment.
    Younger horses and mares carry less weight relative to older males.
    Returns an adjustment factor (0 = average, positive = beneficial).
    """
    # Weight-for-age: 3yo vs 5yo at 2000m = ~3kg difference
    age_adj = max(0.0, (5 - min(age, 7)) * 0.5)  # younger = slight positive
    sex_adj = 1.0 if sex in ("M", "F") else 0.0   # mares get weight concession
    return round(age_adj + sex_adj, 2)
