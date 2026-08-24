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
import math
from datetime import date

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
from horse_engine.bets import harville_horse_top_n
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
        # Isotonic input snapshot. NOT the bare softmax: it is captured
        # AFTER venue calibration but BEFORE isotonic and the multiplier
        # stack — i.e. exactly the value the isotonic curve receives at
        # inference time. Fit and apply must use the same quantity; the
        # nightly rebuild fits on this column. (Renamed conceptually from
        # "raw softmax" 2026-07-22 — the old comment was wrong about
        # where the snapshot sits, the plumbing was always consistent.)
        self.win_prob_raw: float = win_prob
        self.place_prob = place_prob
        self.feature_vector = feature_vector
        self.model_rank: int = 0
        self.place_model_rank: int = 0   # rank by dedicated place model
        self.exotic_model_rank: int = 0  # rank by exotic (trifecta) model
        self.overlay: float = 0.0
        self.value_rating: float = 0.0
        self.key_flags: list[str] = []


def _form_figures(starts: list, n: int = 6) -> str:
    """Industry-standard form string, MOST RECENT FIRST.

    Each recent start's finishing position as one char: 1-9 as-is, 10+ (or
    'unplaced') shown as '0' (finished outside the top 9) — the universal AU
    form-guide convention. Orientation is made robust to the two upstream
    orderings: when starts carry dates (the scraped InteractiveForm path) we
    sort newest-first by date; the dateless form-string fallback is documented
    oldest->newest, so we reverse it. Capped at the last `n` runs.
    """
    if not starts:
        return ""
    dated = sum(1 for s in starts if (getattr(s, "date", "") or "").strip())
    if dated >= max(1, len(starts) // 2):
        seq = sorted(starts, key=lambda s: (getattr(s, "date", "") or ""), reverse=True)
    else:
        seq = list(reversed(starts))
    out = []
    for s in seq[:n]:
        p = getattr(s, "position", None)
        if not p or p <= 0:
            continue
        out.append(str(p) if 1 <= p <= 9 else "0")
    return "".join(out)


def enrich_runner(
    runner: Runner,
    race: Race,
    market_features: dict,
    speed_map: dict[str, str],
    pace_advantage: dict[str, float],
    field_avg_weight: float,
) -> EnrichedRunner:
    """Convert raw Runner + race context into EnrichedRunner with all features."""
    # Sort starts newest-first at source. Several form functions ASSUME this
    # order — days_since_last_run reads starts[0].date, weighted_form_score
    # weights by list index (RECENCY_WEIGHTS[i]) — but runner.last_10_starts is
    # not guaranteed sorted, so an unsorted feed produced phantom long layoffs
    # (KOMITO/Sandown-Lakeside 2026-08-02: 221d snapshot vs 39d real) and
    # mis-weighted form scores. Sorting once here fixes the whole pipeline;
    # runs_this_prep's own defensive sort becomes redundant but harmless.
    starts = sorted(
        runner.last_10_starts or [],
        key=lambda s: (s.date or ""),
        reverse=True,
    )
    condition_cat = parse_condition_category(race.track_condition)

    # ── Form ─────────────────────────────────────────────────────────────
    fs = form_enricher.weighted_form_score(starts)
    avg_beaten = form_enricher.avg_beaten_margin_last5(starts)
    best_finish = form_enricher.best_finish_at_distance(starts, race.distance)
    days_last = form_enricher.days_since_last_run(starts)
    # Discount form_score for long layoffs — a 15-month-spell horse with
    # "great recent form" 15 months ago shouldn't look like a 30% pick
    # (BULLETIN BEAU, Pinjarra R5 2026-06-24). The model already gets
    # days_since_last_run as a separate feature but it doesn't override
    # a strong form signal on its own.
    fs = form_enricher.discount_form_for_layoff(fs, days_last)
    # Discount form_score for thin records — a 0.85 form on 2 starts is
    # small-sample noise, not a confident signal (OSAKA CASTLE,
    # ballarat R5 2026-06-12). Applied after the layoff discount so a
    # horse with both gets compounded penalty toward the no-form baseline.
    fs = form_enricher.discount_form_for_thin_record(fs, len(starts) if starts else 0)
    runs_prep = form_enricher.runs_this_prep(starts)
    class_chg = form_enricher.class_change_flag(starts, race.race_class)
    spell_w = form_enricher.spell_weeks(starts)

    # Career win rate — prefer Racing Australia direct stats (accurate), fall back to form starts
    if runner.career_starts > 0:
        wr_career = runner.career_wins / runner.career_starts
    else:
        wr_career = form_enricher.career_win_rate(runner)

    # Career place rate
    if runner.career_starts > 0:
        career_place_rate = runner.career_places / runner.career_starts
    else:
        career_place_rate = 0.0

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

    # Surface match — prefer Racing Australia condition stats, fall back to form-based
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
        career_place_rate=career_place_rate,
        wins_last_10=sum(1 for s in starts if s.position == 1),
        places_last_10=sum(1 for s in starts if s.position is not None and s.position <= 3),
        starts_last_10=len(starts),
        form_figures=_form_figures(starts),
        # Live stream signals — passed through directly if populated
        steam_60=runner.steam_60,
        steam_30=runner.steam_30,
        late_money=runner.late_money,
        drift_flag=runner.drift_flag,
        odds_velocity=runner.odds_velocity,
    )


def predict_race(race: Race, model: HorseModel, venue_calibration: dict[str, float] | None = None, place_model: PlaceModel | None = None, exotic_model: ExoticModel | None = None, output_calibration_curve: list[tuple[float, float]] | None = None, trace: dict | None = None) -> list[RunnerPrediction]:
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
    win_probs_raw, heuristic_place_probs = model.predict_field(feature_vectors)
    win_probs = apply_venue_calibration(list(win_probs_raw), race.venue, venue_calibration or {})
    # Diagnostic trace — instrument every multiplier stage so the admin
    # endpoint can show why any given horse's win_prob is at its final
    # value. Costs one dict-append per horse per stage when enabled; noop
    # when trace is None (production path).
    _venue_mult = (venue_calibration or {}).get(race.venue, 1.0) if venue_calibration else 1.0
    if trace is not None:
        for i, (runner, er) in enumerate(zip(race.runners[:len(win_probs)], enriched_runners)):
            trace[runner.horse_name] = {
                "raw_softmax": round(win_probs_raw[i], 4),
                "after_venue_calibration": round(win_probs[i], 4),
                "venue_multiplier": round(_venue_mult, 4),
                "metadata": {
                    "venue": race.venue,
                    "speed_map_position": getattr(er, "speed_map_position", None),
                    "track_condition_category": getattr(er, "track_condition_category", None),
                    "starts_last_10": getattr(er, "starts_last_10", None),
                    "market_implied_prob": round(getattr(er, "market_implied_prob", 0) or 0, 4),
                    "race_distance": race.distance,
                    "race_track_condition": race.track_condition,
                },
            }

    # Place-prob assembly.
    # PlaceModel.predict_field returns (softmax(raw), softmax(raw × 0.5) × N).
    # Neither element cleanly represents per-horse P(top-3):
    #  - [0] is softmax → sums to 1, individual values can be < win
    #  - [1] is heuristic scaled by N_places — fine for 8+ runner fields
    #    but for fields <5 runners N_places=1 so [1] can be < [0] and
    #    < the same horse's win_prob (violates P(top-3) ≥ P(top-1))
    # Take [1] first (better on medium/large fields), then enforce
    # place ≥ win per horse post-hoc. That's a mathematical axiom for
    # the same horse in the same race — no calibration cost.
    # Shipped 2026-07-12 after pioneer-park (small QLD field) still
    # showed place < win under the fix that used [1] alone.
    if place_model is not None:
        _, place_probs_list = place_model.predict_field(feature_vectors)
    else:
        place_probs_list = list(heuristic_place_probs)

    # Compute exotic scores up-front too (BUG-40 / ARCH-1 sibling fix). The old
    # `zip(scores, predictions)` after `predictions.sort()` paired scores in
    # original-runner-order with predictions in win-rank order — assigning each
    # rank to the wrong horse. Compute scores while indexes still match the
    # predictions list, attach them to each prediction, then sort.
    if exotic_model is not None:
        exotic_scores_list, _ = exotic_model.predict_field(feature_vectors)
    else:
        exotic_scores_list = []

    predictions: list[RunnerPrediction] = []
    for i, (runner, er) in enumerate(zip(race.runners[:len(enriched_runners)], enriched_runners)):
        wp = win_probs[i] if i < len(win_probs) else 0.0
        pp = place_probs_list[i] if i < len(place_probs_list) else 0.0
        overlay = wp - er.market_implied_prob
        er.overlay = round(overlay, 4)

        pred = RunnerPrediction(
            runner=runner,
            enriched=er,
            win_prob=wp,
            place_prob=round(pp, 4),
            feature_vector=feature_vectors[i],
        )
        pred.win_prob_raw = round(wp, 4)  # isotonic-input snapshot: post-venue-calibration, pre-isotonic/multipliers
        pred.overlay = overlay
        pred.key_flags = _generate_flags(er, wp, overlay)
        # Attach exotic score before sort so ranking pairs the right horse.
        pred._exotic_score = exotic_scores_list[i] if i < len(exotic_scores_list) else 0.0
        predictions.append(pred)

    # Output-space isotonic calibration — moved here (2026-07-15) from AFTER
    # the multiplier stack. Rationale: fitting isotonic on POST-multiplier
    # win_probability caused PAV to pool the 22.5-32.5% band into a single
    # 22.6% output (the multipliers demote raw ~40% horses into that band,
    # and those genuinely win less than natural-25% horses — non-monotone).
    # Running isotonic on raw softmax means input→actual is monotone by
    # construction (the model was trained to maximise likelihood on
    # winners), so PAV can't pool. The reactive multipliers then apply
    # their researched corrections to the calibrated probability, which
    # is what they were tuned to work against anyway.
    # NOTE: _load_output_calibration_curve currently returns None (short-
    # circuit shipped 2026-07-15). Once ~30 days of raw data have
    # accumulated in RunnerPredictionHistoryRow.win_prob_raw, the nightly
    # rebuild can fit on that column and the short-circuit reverts to a
    # one-line change.
    if output_calibration_curve:
        from horse_engine.prediction.output_calibration import apply_calibration_curve
        pcts = [p.win_prob * 100.0 for p in predictions]
        calibrated = apply_calibration_curve(pcts, output_calibration_curve)
        for p, c in zip(predictions, calibrated):
            p.win_prob = round(c / 100.0, 4)
    if trace is not None:
        for p in predictions:
            t = trace.get(p.runner.horse_name)
            if t is not None:
                t["after_isotonic"] = p.win_prob
                t["isotonic_active"] = output_calibration_curve is not None

    # Midfield penalty — 90-day winner-vs-loser feature analysis showed
    # rank-1 picks with speed_map_position='midfield' won at 22.7% vs
    # 29-31% for leader/on-pace (n=1,108 midfield over 90d). The raw
    # model treats all pace profiles the same. Apply a 0.85× multiplier
    # to win_prob for midfield horses — pulls them down relative to
    # front-runners without changing rank within an all-midfield field.
    # No renormalisation: the display reflects the corrected confidence.
    for p in predictions:
        if getattr(p.enriched, "speed_map_position", None) == "midfield":
            p.win_prob = round(p.win_prob * 0.85, 4)
            if trace is not None and p.runner.horse_name in trace:
                trace[p.runner.horse_name]["midfield_multiplier"] = 0.85
        elif trace is not None and p.runner.horse_name in trace:
            trace[p.runner.horse_name]["midfield_multiplier"] = 1.0
    if trace is not None:
        for p in predictions:
            t = trace.get(p.runner.horse_name)
            if t is not None:
                t["after_midfield"] = p.win_prob

    # Going calibration — split per-going multipliers (was a unified 0.55
    # for both soft and heavy). 60-day going-calibration backtest on
    # 2026-06-30 (n=85 soft, n=41 heavy) showed soft and heavy diverge
    # by ~0.15 in actual win rate, justifying separate multipliers:
    #   soft  →  7.1% actual win vs good baseline 17.6%  → mult 0.40
    #   heavy →  9.8% actual win vs good baseline 17.6%  → mult 0.55
    # Soft is roughly 2.5× over-confident — slightly more chaotic for
    # the model than heavy. Avg winner SP on soft <20% bin was $14.47
    # (market also flags off-going as chaotic). Multipliers de-confidence
    # every rank in the field — rank order is preserved, just at lower
    # absolute probabilities.
    for p in predictions:
        going = getattr(p.enriched, "track_condition_category", None)
        mult = 1.0
        if going == "soft":
            mult = 0.40
        elif going == "heavy":
            mult = 0.55
        if mult != 1.0:
            p.win_prob = round(p.win_prob * mult, 4)
        if trace is not None and p.runner.horse_name in trace:
            trace[p.runner.horse_name]["going_multiplier"] = mult
    if trace is not None:
        for p in predictions:
            t = trace.get(p.runner.horse_name)
            if t is not None:
                t["after_going"] = p.win_prob

    # Thin-record penalty — the model can't distinguish a lightly-raced
    # horse's genuine form from lucky variance. Even at 3 starts a 1-3-3
    # record could equally be 1-lucky-2 (5-length beaten places we can't
    # see) or 1-genuine-3. The form_score discount in form.py handles the
    # feature-side pull; this multiplier attacks the OUTPUT for whatever
    # pathway drove the confidence (trainer/jockey/distance stacking).
    # Shipped 2026-07-10 after COAL SEAM/mackay R1 went 68.8% → 6th of 8.
    for p in predictions:
        n = getattr(p.enriched, "starts_last_10", None)
        mult = 1.0
        if isinstance(n, (int, float)):
            if n <= 1:
                mult = 0.50
            elif n == 2:
                mult = 0.70
            elif n == 3:
                mult = 0.85
        if mult != 1.0:
            p.win_prob = round(p.win_prob * mult, 4)
        if trace is not None and p.runner.horse_name in trace:
            trace[p.runner.horse_name]["thin_record_multiplier"] = mult
    if trace is not None:
        for p in predictions:
            t = trace.get(p.runner.horse_name)
            if t is not None:
                t["after_thin_record"] = p.win_prob

    # Feature-completeness gate — when critical race-level features are
    # missing (distance=0, going unparseable) or the runner has no market
    # signal, the model is scoring on an incomplete picture and its
    # confidence is not trustworthy. Shrink win_prob so a data-quality
    # miss can't produce a 72% pick.
    # Shipped 2026-07-11 after COEUR VOLANTE/caulfield R8 hit 72.1% with
    # race.distance=0 — every distance-dependent feature (pedigree match,
    # win_rate_distance, distance_aptitude, jockey_distance_rate,
    # trainer_distance_rate) was silently miscomputed on a zero.
    dist_missing = not race.distance or race.distance == 0
    going_missing = not race.track_condition or not race.track_condition.strip()
    for p in predictions:
        missing = 0
        if dist_missing:
            missing += 1
        if going_missing:
            missing += 1
        market_prob = getattr(p.enriched, "market_implied_prob", 0) or 0
        if market_prob <= 0:
            missing += 1
        mult = 1.0
        if missing >= 2:
            mult = 0.60
        elif missing == 1:
            mult = 0.80
        if mult != 1.0:
            p.win_prob = round(p.win_prob * mult, 4)
        if trace is not None and p.runner.horse_name in trace:
            trace[p.runner.horse_name]["feature_completeness_multiplier"] = mult
            trace[p.runner.horse_name]["missing_features_count"] = missing
    if trace is not None:
        for p in predictions:
            t = trace.get(p.runner.horse_name)
            if t is not None:
                t["after_feature_completeness"] = p.win_prob

    # (Output-space isotonic calibration was moved above, before the
    # multipliers. Applying it here — on the already-multiplier-adjusted
    # win_probability — created the plateau failure mode documented in
    # commit 75ee777 and the block up-pipeline.)

    # Market-anchored shrinkage — a large model-vs-market gap is evidence
    # the model is missing information the market has (barrier trials,
    # stable whispers, weight-of-money). Below 25pp overlay we trust the
    # model (that's where value lives). Between 25pp and 55pp we ramp
    # linearly to a 60% market weight. Above 55pp we cap at 60% — the
    # market isn't infallible either.
    # Fixes the sparse-tail gap in the isotonic curve: 70%+ predictions
    # are too rare to calibrate empirically, but the market ALWAYS has a
    # prior on them, so market-anchored shrinkage covers what isotonic
    # can't yet see.
    # Shipped 2026-07-11 after COEUR VOLANTE hit 72.1% against a bookie
    # price of ~17.4% (55pp overlay) — isotonic curve was identity at
    # that band, feature-completeness gate alone only dropped it to ~58%.
    # Benter blend (2026-07-22) — when BENTER_ALPHA/BETA are set, replace
    # the one-sided shrinkage below with a fitted two-sided combination:
    #   score_i = alpha·log(p_model_i) + beta·log(p_market_i) → softmax.
    # Applied MASS-PRESERVINGLY: the field's total probability mass (already
    # deflated by going/thin-record/completeness multipliers) is kept and
    # redistributed by the blend, so absolute tier semantics (>=30% gates,
    # Sharp) are unchanged while within-race ranking gains the market signal.
    # Unlike shrinkage this also pulls the model UP toward strong favourites
    # it under-rates — the documented 10-17pp favourite gap.
    if _apply_benter_blend(predictions, trace):
        pass  # blend supersedes shrinkage
    else:
        _run_market_shrinkage(predictions, trace)

    # Blind-race shrinkage (2026-07-27): when NO runner in the field has market
    # odds (late-market country meetings — the market simply isn't up yet), the
    # form-only model runs uncalibrated and over-confident (Albury R2
    # 2026-07-27: 60% claimed while the $1.95 market favourite sat ranked 7th
    # at 0.5%). When BLIND_SHRINKAGE > 0, blend every win prob toward the
    # field mean — mass-preserving and rank-preserving, so the pick order and
    # win-rate metrics are untouched while the magnitudes that drive tier
    # badges, Sharp gating and Harville place are tempered. Market detection
    # uses best_available_odds only: market_implied_prob defaults to 1/N when
    # odds are missing, so it cannot distinguish blind from sighted.
    from horse_engine.config import settings as _bs_settings
    _bs = float(getattr(_bs_settings, "blind_shrinkage", 0.0) or 0.0)
    if _bs > 0 and len(predictions) >= 2:
        _has_market = any(
            (getattr(p.enriched, "best_available_odds", 0) or 0) > 1.0
            for p in predictions
        )
        if not _has_market:
            _mass = sum(p.win_prob for p in predictions)
            _mean = _mass / len(predictions)
            for p in predictions:
                p.win_prob = round((1.0 - _bs) * p.win_prob + _bs * _mean, 4)
                if trace is not None and p.runner.horse_name in trace:
                    trace[p.runner.horse_name]["after_blind_shrinkage"] = p.win_prob

    # Place probability. The standalone place model was outputting near-uniform
    # values (~3/N for every runner), so its RANKING was noise — the win
    # favourite could rank mid-pack to place (Grafton R9, 2026-07-24). When
    # place_harville_weight > 0 we blend in a Harville P(top-3) derived from the
    # (already sharp) final WIN probs, so place tracks win strength and the
    # ranking is coherent. Weight 0 = pure trained model (default, off until
    # backtested per feedback_model_release_process).
    from horse_engine.config import settings as _pf_settings
    _hw = float(getattr(_pf_settings, "place_harville_weight", 0.0) or 0.0)
    _win_vec = [max(p.win_prob or 0.0, 0.0) for p in predictions] if _hw > 0 else None
    # Enforce P(top-3) ≥ P(top-1) per horse — a mathematical axiom for the same
    # horse in the same race.
    for idx, (p, plc) in enumerate(zip(predictions, place_probs_list)):
        if _hw > 0:
            others = _win_vec[:idx] + _win_vec[idx + 1:]
            harv = harville_horse_top_n(_win_vec[idx], others, 3)
            plc = _hw * harv + (1.0 - _hw) * plc
        if plc < p.win_prob:
            plc = p.win_prob
        p.place_prob = round(min(plc, 0.99), 4)

    # Rank by win probability
    predictions.sort(key=lambda p: p.win_prob, reverse=True)
    for rank, p in enumerate(predictions, 1):
        p.model_rank = rank
        p.value_rating = _value_rating(p.win_prob, p.enriched.best_available_odds, p.overlay)

    # Market-defer guardrail — on a thin win-rank disagreement, defer our rank-1
    # to the market favourite. See settings.market_defer_edge_pp for the why and
    # the Crucible evidence. Off unless the edge threshold is set (>0). Swaps the
    # two horses' win_prob so a re-sort re-ranks cleanly (rank↔prob stays
    # consistent for the sharp/edge gates downstream), rather than leaving a
    # rank-1 whose prob is below rank-2's.
    from horse_engine.config import settings as _defer_settings
    _defer_edge = float(getattr(_defer_settings, "market_defer_edge_pp", 0.0) or 0.0) / 100.0
    if _defer_edge > 0.0 and len(predictions) >= 2:
        _our_top = predictions[0]  # model_rank 1 after the sort above
        _mkt_fav = next((p for p in predictions if getattr(p, "market_rank", 0) == 1), None)
        if (
            _mkt_fav is not None
            and _mkt_fav is not _our_top
            and (_our_top.win_prob - _mkt_fav.win_prob) < _defer_edge
        ):
            _our_top.win_prob, _mkt_fav.win_prob = _mkt_fav.win_prob, _our_top.win_prob
            predictions.sort(key=lambda p: p.win_prob, reverse=True)
            for rank, p in enumerate(predictions, 1):
                p.model_rank = rank
                p.value_rating = _value_rating(p.win_prob, p.enriched.best_available_odds, p.overlay)
            if trace is not None and _mkt_fav.runner.horse_name in trace:
                trace[_mkt_fav.runner.horse_name]["market_defer"] = (
                    f"promoted_to_rank1(edge<{_defer_edge*100:.0f}pp)"
                )

    # R1-3 segment calibration (2026-08-25, Crucible-confirmed OOS on n=2918:
    # early-card rank-1s predicted 21.4% but actually win 30.2%). When enabled
    # (R1_3_SEGMENT_CALIB > 1.0), scale up the FINAL rank-1 on races 1-3 by the
    # conservative OOS-test ratio. Deflation-multiplier style (per-runner scale,
    # no renorm — matches the going/thin-record stages); capped at 0.90 for the
    # display rule. Rank-preserving: only the current top pick is lifted, so the
    # PICK never changes — this corrects the displayed %/tier/Sharp-eligibility,
    # not win rate. Applied last, on the post-defer favourite.
    from horse_engine.config import settings as _seg_settings
    _r13 = float(getattr(_seg_settings, "r1_3_segment_calib", 0.0) or 0.0)
    if _r13 > 1.0 and predictions and getattr(race, "race_number", None) in (1, 2, 3):
        _top = max(predictions, key=lambda p: p.win_prob)
        _boosted = min(round(_top.win_prob * _r13, 4), 0.90)
        if trace is not None and _top.runner.horse_name in trace:
            trace[_top.runner.horse_name]["r1_3_segment_calib"] = f"x{_r13}->{_boosted}"
        _top.win_prob = _boosted

    # Rank by place probability (using the trained place model when provided,
    # else the heuristic carried on each prediction). Sorting predictions by
    # their own place_prob guarantees the ranking matches the horse.
    if place_model is not None:
        place_sorted = sorted(predictions, key=lambda p: p.place_prob, reverse=True)
        for place_rank, p in enumerate(place_sorted, 1):
            p.place_model_rank = place_rank

    # Rank by exotic score — uses the score attached on each prediction so the
    # rank pairs the right horse.
    if exotic_model is not None:
        exotic_sorted = sorted(predictions, key=lambda p: getattr(p, "_exotic_score", 0.0), reverse=True)
        for exotic_rank, p in enumerate(exotic_sorted, 1):
            p.exotic_model_rank = exotic_rank

    return predictions


def _apply_benter_blend(predictions: list, trace: dict | None = None) -> bool:
    """Mass-preserving Benter blend. Returns True if applied.

    Requires BENTER_ALPHA/BENTER_BETA > 0 in settings and market coverage
    on ≥80% of the field (missing-market runners use their own model prob
    as the market proxy so they are neither boosted nor punished).
    Evidence (2026-07-22 offline backtest, 868-race holdout): model-alone
    top-1 21.8%, SP-market 35.0%, blend 35.1%; live-market fit
    alpha=0.14, beta=0.53. See memory phase2-experiments-2026-07-22."""
    from horse_engine.config import settings
    alpha = float(getattr(settings, "benter_alpha", 0.0) or 0.0)
    beta = float(getattr(settings, "benter_beta", 0.0) or 0.0)
    if alpha <= 0.0 and beta <= 0.0:
        return False
    if len(predictions) < 2:
        return False
    markets = [getattr(p.enriched, "market_implied_prob", 0.0) or 0.0 for p in predictions]
    covered = sum(1 for m in markets if m > 0)
    if covered < 0.8 * len(predictions):
        if trace is not None:
            for p in predictions:
                if p.runner.horse_name in trace:
                    trace[p.runner.horse_name]["benter_blend"] = f"skipped_coverage({covered}/{len(predictions)})"
        return False
    mass = sum(p.win_prob for p in predictions)
    if mass <= 0:
        return False
    floor = 1e-4
    p_model = [max(p.win_prob / mass, floor) for p in predictions]
    mkt_sum = sum(m for m in markets if m > 0)
    scores = []
    for pm, mk in zip(p_model, markets):
        mk_n = max(mk / mkt_sum, floor) if (mk > 0 and mkt_sum > 0) else pm
        scores.append(alpha * math.log(pm) + beta * math.log(mk_n))
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    total = sum(exps)
    for p, e in zip(predictions, exps):
        blended = mass * e / total
        if trace is not None and p.runner.horse_name in trace:
            trace[p.runner.horse_name]["benter_blend"] = (
                f"applied(a={alpha},b={beta})"
            )
            trace[p.runner.horse_name]["pre_blend_win_prob"] = p.win_prob
            trace[p.runner.horse_name]["final_win_prob"] = round(blended, 4)
        p.win_prob = round(blended, 4)
    return True


def _run_market_shrinkage(predictions: list, trace: dict | None = None) -> None:
    """Legacy one-sided market-anchored shrinkage (pre-Benter behaviour).
    Active whenever the Benter blend is disabled or skipped for coverage."""
    _SHRINK_START = 0.25   # 25pp overlay — no shrinkage below
    _SHRINK_FULL = 0.55    # 55pp overlay — full shrinkage weight
    _SHRINK_MAX = 0.60     # max weight on market (never fully overrule the model)
    _NULL_MARKET_CAP = 0.40  # ceiling for win_prob when market signal is missing
    for p in predictions:
        market = getattr(p.enriched, "market_implied_prob", 0.0) or 0.0
        shrink_action = "none"
        if market <= 0:
            if p.win_prob > _NULL_MARKET_CAP:
                p.win_prob = round(0.5 * p.win_prob + 0.5 * _NULL_MARKET_CAP, 4)
                shrink_action = "null_market_cap"
        else:
            overlay = p.win_prob - market
            if overlay > _SHRINK_START:
                excess = min(overlay, _SHRINK_FULL) - _SHRINK_START
                weight = _SHRINK_MAX * excess / (_SHRINK_FULL - _SHRINK_START)
                p.win_prob = round((1.0 - weight) * p.win_prob + weight * market, 4)
                shrink_action = f"market_anchor(w={round(weight, 3)})"
        if trace is not None and p.runner.horse_name in trace:
            trace[p.runner.horse_name]["market_shrinkage"] = shrink_action
            trace[p.runner.horse_name]["final_win_prob"] = p.win_prob


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
