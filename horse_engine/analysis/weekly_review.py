"""Weekly review analyser — runs Monday 03:00 AEST.

Reviews the last 7 days of racing (Mon→Sun ending on the Sunday before
the cron fires) and produces structured suggestions. Mirrors the
nightly review's data model — same NightlyReviewRow table, same
suggestion shape, same dashboard — but with detectors tuned for
week-over-week pattern detection rather than single-day drift.

Why not just reuse the nightly analyser with `window_days=7`? The
nightly detectors compare *today* vs the prior 30 days. The weekly
analyser instead compares the *full week* vs the prior 30 days, so the
denominator is large enough to flag genuine trends (a 30-race week
gives enough sample to detect a 10pt drop in Sharp hit rate; a single
day rarely does).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from horse_engine.analysis.nightly_review import (
    Suggestion,
    _index_results,
    _normalize_horse,
    _parse_enriched,
)
from horse_engine.models.database import (
    HistoricalResultRow,
    RunnerPredictionHistoryRow,
)


# ---------- data loading ----------

async def _load_weekly_window(
    session: AsyncSession,
    end_date: date,
    weekly_days: int = 7,
    baseline_days: int = 30,
) -> tuple[list, list, dict]:
    """Loads picks + results for [end - (weekly+baseline), end] and the
    top-3 sum per race (for Sharp eligibility). Returns dedup'd latest
    snapshot per (race, horse)."""
    start = (end_date - timedelta(days=weekly_days + baseline_days)).isoformat()
    end_excl = (end_date + timedelta(days=1)).isoformat()

    pred_rows = (await session.execute(
        select(
            RunnerPredictionHistoryRow.race_id,
            RunnerPredictionHistoryRow.horse_name,
            RunnerPredictionHistoryRow.win_probability,
            RunnerPredictionHistoryRow.model_rank,
            RunnerPredictionHistoryRow.market_rank,
            RunnerPredictionHistoryRow.enriched_json,
            RunnerPredictionHistoryRow.enriched_at,
        )
        .where(RunnerPredictionHistoryRow.race_id >= f"{start}_")
        .where(RunnerPredictionHistoryRow.race_id < f"{end_excl}_")
        .where(RunnerPredictionHistoryRow.cancelled.is_(False)
               | RunnerPredictionHistoryRow.cancelled.is_(None))
        .where(RunnerPredictionHistoryRow.source == "live")
        .order_by(RunnerPredictionHistoryRow.enriched_at.desc())
    )).fetchall()

    seen: set = set()
    latest = []
    top3_seen: set = set()
    top3_by_race: dict[str, float] = {}
    for row in pred_rows:
        key = (row.race_id, _normalize_horse(row.horse_name))
        if key in seen:
            continue
        seen.add(key)
        latest.append(row)
        if row.model_rank in (1, 2, 3) and key not in top3_seen:
            top3_seen.add(key)
            top3_by_race[row.race_id] = top3_by_race.get(row.race_id, 0.0) + float(row.win_probability or 0) * 100

    result_rows = (await session.execute(
        select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
               HistoricalResultRow.position, HistoricalResultRow.starting_price,
               HistoricalResultRow.placed, HistoricalResultRow.winner,
               HistoricalResultRow.field_size)
        .where(HistoricalResultRow.race_id >= f"{start}_")
        .where(HistoricalResultRow.race_id < f"{end_excl}_")
    )).fetchall()

    return latest, result_rows, top3_by_race


def _is_sharp(pick, top3_by_race: dict) -> bool:
    """Edge/Lounge Sharp: rank-1 ≥30% OR top-3 sum ≥60%."""
    rank1_pct = (pick.win_probability or 0) * 100
    t3_sum = top3_by_race.get(pick.race_id, 0.0)
    return rank1_pct >= 30 or t3_sum >= 60


def _split_picks_by_week(picks, end_date: date, weekly_days: int = 7):
    """Returns (this_week_picks, baseline_picks). Both are rank-1 only."""
    rank1 = [p for p in picks if p.model_rank == 1]
    week_start = (end_date - timedelta(days=weekly_days - 1)).isoformat()
    end_iso = end_date.isoformat()
    this_week = [p for p in rank1 if week_start <= p.race_id[:10] <= end_iso]
    baseline = [p for p in rank1 if p.race_id[:10] < week_start]
    return this_week, baseline


def _suggestion_id(end_date: date, pattern_id: str) -> str:
    return f"{pattern_id}__{end_date.isoformat()}"


# ---------- detectors ----------

def _detect_weekly_top1_drift(
    end_date: date, picks: list, results: dict, applied: set, top3_by_race: dict
) -> Optional[Suggestion]:
    """Did rank-1 win rate drop materially this week vs the 30-day baseline?"""
    this_week, baseline = _split_picks_by_week(picks, end_date)
    if len(this_week) < 15 or len(baseline) < 60:
        return None

    def win_pct(rows):
        with_result = [r for p in rows
                       for r in [results.get((p.race_id, _normalize_horse(p.horse_name)))]
                       if r and r["position"] is not None and r["position"] > 0]
        if not with_result:
            return None
        return round(sum(1 for r in with_result if r["winner"]) / len(with_result) * 100, 1)

    wk = win_pct(this_week)
    base = win_pct(baseline)
    if wk is None or base is None:
        return None
    drop = base - wk
    if drop < 5.0:
        return None
    sev = "high" if drop >= 10 else "medium"
    return Suggestion(
        id=_suggestion_id(end_date, "weekly_top1_winrate_drift"),
        pattern_id="weekly_top1_winrate_drift",
        title=f"Top-1 win rate dropped {drop:.1f}pt this week vs 30d baseline",
        severity=sev,
        rationale=(
            f"This week's rank-1 picks won {wk:.1f}% ({len(this_week)} races) vs "
            f"the prior-30d baseline of {base:.1f}%. {drop:.1f}pt drop on a "
            f"{len(this_week)}-race week is large enough that it's likely structural "
            "rather than variance. Investigate which failure modes dominated this week "
            "before changing model weights."
        ),
        evidence=[],
        paste_prompt=(
            f"Top-1 win rate dropped {drop:.1f}pt this week ({wk:.1f}% vs 30d baseline "
            f"{base:.1f}%). Pull /api/admin/model-anomalies?days=7 and report the dominant "
            "failure-mode tags. If one tag is responsible for >40% of the misses, propose "
            "the specific engine.py / form.py change with a backtest plan. If the misses "
            "are spread across tags, hold and re-check next week."
        ),
        code_pointer="horse_engine/prediction/engine.py",
        metric_delta={"baseline_30d_pct": base, "this_week_pct": wk,
                      "this_week_races": len(this_week)},
    )


def _detect_weekly_sharp_drift(
    end_date: date, picks: list, results: dict, applied: set, top3_by_race: dict
) -> Optional[Suggestion]:
    """Did Sharp-niche hit rate drift this week vs the prior 30-day Sharp baseline?"""
    this_week, baseline = _split_picks_by_week(picks, end_date)
    wk_sharp = [p for p in this_week if _is_sharp(p, top3_by_race)]
    base_sharp = [p for p in baseline if _is_sharp(p, top3_by_race)]
    if len(wk_sharp) < 8 or len(base_sharp) < 30:
        return None

    def win_pct(rows):
        wr = [results.get((p.race_id, _normalize_horse(p.horse_name))) for p in rows]
        wr = [r for r in wr if r and r["position"] is not None and r["position"] > 0]
        if not wr:
            return None
        return round(sum(1 for r in wr if r["winner"]) / len(wr) * 100, 1)

    wk_pct = win_pct(wk_sharp)
    base_pct = win_pct(base_sharp)
    if wk_pct is None or base_pct is None:
        return None
    drop = base_pct - wk_pct
    if drop < 5.0:
        return None
    sev = "high" if drop >= 10 else "medium"
    return Suggestion(
        id=_suggestion_id(end_date, "weekly_sharp_winrate_drift"),
        pattern_id="weekly_sharp_winrate_drift",
        title=f"Sharp niche hit rate dropped {drop:.1f}pt this week",
        severity=sev,
        rationale=(
            f"This week's Sharp-qualifying picks won {wk_pct:.1f}% ({len(wk_sharp)} races) "
            f"vs the prior-30d Sharp baseline of {base_pct:.1f}%. Per the hard rule, the "
            "fix is to ADD a tightening gate (not loosen the existing threshold). Look for "
            "a feature slice that disproportionately drove this week's Sharp misses — going "
            "category, distance band, market disagreement, layoff bucket — and propose "
            "excluding that slice from the Sharp filter."
        ),
        evidence=[],
        paste_prompt=(
            f"Sharp hit rate dropped {drop:.1f}pt this week ({wk_pct:.1f}% vs 30d {base_pct:.1f}%). "
            "Pull /api/admin/bets/winner-vs-loser-features?days=7 to find the feature with the "
            "biggest divergence between this week's Sharp winners and Sharp losers. Propose a "
            "new gate (e.g., 'edge_sharp_exclude_<slice>') with a backtest projection: slice "
            "loss rate, share of Sharp picks dropped, projected lift after exclusion. "
            "NEVER suggest loosening the existing threshold."
        ),
        code_pointer="horse_engine/api/main.py (Sharp filter)",
        metric_delta={"baseline_30d_sharp_pct": base_pct, "this_week_sharp_pct": wk_pct,
                      "this_week_sharp_races": len(wk_sharp)},
    )


def _detect_weekly_dominant_failure(
    end_date: date, picks: list, results: dict, applied: set, top3_by_race: dict
) -> Optional[Suggestion]:
    """Surface the dominant failure-mode tag across the week — different
    from the nightly anomaly cluster because the denominator is a full
    week, so a tag that fires 7 times across 7 days is much more credible
    than 7 times in one day."""
    this_week, _ = _split_picks_by_week(picks, end_date)
    tag_losses: dict[str, int] = {}
    examples: dict[str, list] = {}
    for p in this_week:
        r = results.get((p.race_id, _normalize_horse(p.horse_name)))
        if not r or r["position"] is None or r["position"] <= 0:
            continue
        if r["winner"] or r["placed"]:
            continue
        if r["starting_price"] < 6.0:
            continue
        e = _parse_enriched(p)
        days_off = e.get("days_since_last_run")
        form = e.get("form_score")
        tags = []
        if isinstance(days_off, (int, float)) and days_off > 180:
            tags.append("long_layoff")
        if (e.get("starts_last_10") or 0) <= 2:
            tags.append("thin_record")
        if isinstance(form, (int, float)) and form >= 0.75 and isinstance(days_off, (int, float)) and days_off > 120:
            tags.append("stale_form_signal")
        if (p.market_rank or 0) >= 5:
            tags.append("market_disagreed")
        for t in tags:
            tag_losses[t] = tag_losses.get(t, 0) + 1
            examples.setdefault(t, []).append({
                "race_id": p.race_id, "horse_name": p.horse_name,
                "position": r["position"], "starting_price": r["starting_price"],
            })
    if not tag_losses:
        return None
    top_tag, top_count = max(tag_losses.items(), key=lambda x: x[1])
    if top_count < 4:
        return None
    return Suggestion(
        id=_suggestion_id(end_date, f"weekly_dominant_failure_{top_tag}"),
        pattern_id=f"weekly_dominant_failure_{top_tag}",
        title=f"{top_count} losses this week tagged '{top_tag}' (most-common failure)",
        severity="medium",
        rationale=(
            f"Across the week, {top_count} losing rank-1 picks shared the '{top_tag}' tag — "
            "the dominant failure mode. Other tags this week: " +
            ", ".join(f"{k} ({v})" for k, v in sorted(tag_losses.items(), key=lambda x: -x[1])[1:5]) +
            ". A weekly cluster on a single tag is much more credible than a daily one — "
            "the discount/penalty for this input likely needs strengthening."
        ),
        evidence=examples[top_tag][:5],
        paste_prompt=(
            f"This week's dominant rank-1 failure mode was '{top_tag}' ({top_count} losses). "
            "Compare to last week via /api/admin/model-anomalies?days=14 grouped by week. "
            "If '{top_tag}' is consistently dominant for 2+ weeks, propose the specific "
            "engine.py / form.py change. If it's a one-week spike, hold and re-check."
        ),
        code_pointer="horse_engine/prediction/engine.py",
        metric_delta={"dominant_tag": top_tag, "count_this_week": top_count,
                      "all_tags": dict(sorted(tag_losses.items(), key=lambda x: -x[1])[:6])},
    )


def _detect_weekly_going_pattern(
    end_date: date, picks: list, results: dict, applied: set, top3_by_race: dict
) -> Optional[Suggestion]:
    """Compare per-going calibration across the week."""
    this_week, _ = _split_picks_by_week(picks, end_date)
    by_going: dict[str, list] = {"good": [], "firm": [], "soft": [], "heavy": [], "synthetic": []}
    for p in this_week:
        r = results.get((p.race_id, _normalize_horse(p.horse_name)))
        if not r or r["position"] is None or r["position"] <= 0:
            continue
        e = _parse_enriched(p)
        going = e.get("track_condition_category")
        if going not in by_going:
            continue
        by_going[going].append({
            "model_pct": (p.win_probability or 0) * 100,
            "won": r["winner"],
        })

    def ratio(rows):
        if len(rows) < 6:
            return None
        avg_model = sum(x["model_pct"] for x in rows) / len(rows)
        actual = sum(1 for x in rows if x["won"]) / len(rows) * 100
        return round(actual / avg_model, 2) if avg_model > 0 else None

    dry_rows = by_going["good"] + by_going["firm"] + by_going["synthetic"]
    dry = ratio(dry_rows)
    soft = ratio(by_going["soft"])
    heavy = ratio(by_going["heavy"])
    if not dry:
        return None

    flagged = []
    for label, r, n_rows in (("soft", soft, by_going["soft"]),
                             ("heavy", heavy, by_going["heavy"])):
        if r is None:
            continue
        gap = dry - r
        if gap >= 0.25:
            flagged.append({"going": label, "ratio": r, "n_races": len(n_rows),
                            "dry_ratio": dry, "gap": round(gap, 2)})
    if not flagged:
        return None
    return Suggestion(
        id=_suggestion_id(end_date, "weekly_going_calibration_pattern"),
        pattern_id="weekly_going_calibration_pattern",
        title=f"Going calibration off this week on {', '.join(f['going'] for f in flagged)}",
        severity="medium",
        rationale=(
            f"Dry-going Sharp picks calibrated at ratio {dry} this week ("
            f"{len(dry_rows)} races). Wet-going picks diverged: " +
            "; ".join(f"{f['going']} ratio {f['ratio']} ({f['n_races']} races)" for f in flagged) +
            ". A weekly miss on going calibration is the right resolution to act on — daily "
            "samples are too small."
        ),
        evidence=flagged,
        paste_prompt=(
            "Going calibration drift detected over the week. Run "
            "/api/admin/bets/going-calibration?days=60 for the wider window and tell me the "
            "right per-going multiplier. Currently engine.py applies *0.55 for soft/heavy. "
            "If soft and heavy ratios diverge by >0.15, propose splitting them into separate "
            "multipliers; otherwise tighten the single multiplier."
        ),
        code_pointer="horse_engine/prediction/engine.py (going calibration)",
        metric_delta={"flagged_going": flagged, "dry_baseline_ratio": dry},
    )


# Baseline captured 2026-07-10 after shipping the thin-record + midfield
# + going-multiplier changes. See detector below.
_THIN_RECORD_BASELINE_2026_07_10 = {
    "captured_at": "2026-07-10",
    "settled_rank1_picks_7d": 297,
    "overall_win_pct": 23.6,
    "midfield_win_pct": 26.4,       # n=144
    "on_pace_win_pct": 21.7,        # n=83
    "leader_win_pct": 20.0,         # n=70
    "days_off_winner_mean": 148.7,
    "days_off_loser_mean": 145.3,
    "days_off_loser_minus_winner": -3.4,
}


def _detect_thin_record_validation_2026_07_10(
    end_date: date, picks: list, results: dict, applied: set, top3_by_race: dict
) -> Optional[Suggestion]:
    """One-off validation check for the 2026-07-10 thin-record ship.

    Fires exactly once on the first weekly review from 2026-07-14 (the
    first Monday after the ship) onwards. Emits a paste-prompt that walks
    the operator through re-running the winner-vs-loser-features endpoint
    and diffing the midfield + days-off numbers against the baseline
    captured on ship day.

    Uses the applied-history mechanism to avoid re-firing across weeks —
    once the operator marks the suggestion applied (or dismissed), the
    detector goes quiet.
    """
    target_date = date(2026, 7, 14)
    if end_date < target_date:
        return None
    pattern_id = "validation_thin_record_ship_2026_07_10"
    if pattern_id in applied:
        return None
    b = _THIN_RECORD_BASELINE_2026_07_10
    return Suggestion(
        id=_suggestion_id(end_date, pattern_id),
        pattern_id=pattern_id,
        title="Validate thin-record + midfield + going ship (2026-07-10)",
        severity="medium",
        rationale=(
            f"On 2026-07-10 we shipped three model refinements: form.py's "
            "thin-record discount extended to 3-start horses, engine.py's "
            "post-ranking multiplier for starts_last_10 ≤ 3, and split "
            "soft/heavy going multipliers (0.40 / 0.55). The plan was to "
            "re-run /api/admin/bets/winner-vs-loser-features?days=7 a week "
            "later and check that midfield bias and days_off lift have "
            "shrunk toward the baseline. Now's the time."
        ),
        evidence=[b],
        paste_prompt=(
            "Run the thin-record ship validation. Pull "
            "/api/admin/bets/winner-vs-loser-features?days=7 and compare "
            "against the 2026-07-10 baseline:\n"
            f"  overall win% baseline={b['overall_win_pct']}% "
            f"midfield={b['midfield_win_pct']}% (n=144) "
            f"days_off gap={b['days_off_loser_minus_winner']}d\n"
            "\n"
            "Pass criteria:\n"
            "  1. midfield win% should have converged toward 22-24% "
            "(closer to on_pace/leader). If still ≥26% ship a "
            "further-tightened midfield multiplier.\n"
            "  2. days_off (loser − winner) absolute lift ≤ 4 days. If "
            "drifting toward +10 or -10, revisit the discount curve.\n"
            "  3. Going multiplier is on a longer clock (rare off-going "
            "days). Note the sample size and re-check monthly with "
            "?days=60 instead.\n"
            "\n"
            "Post a short pass/fail table. If a metric failed, propose "
            "the specific next change. If everything passed, mark this "
            "suggestion applied so it doesn't fire again."
        ),
        code_pointer="horse_engine/prediction/engine.py + horse_engine/enrichers/form.py",
        metric_delta={"baseline_2026_07_10": b},
    )


WEEKLY_DETECTORS = [
    _detect_weekly_top1_drift,
    _detect_weekly_sharp_drift,
    _detect_weekly_dominant_failure,
    _detect_weekly_going_pattern,
    _detect_thin_record_validation_2026_07_10,
]


# ---------- entrypoint ----------

async def generate_weekly_review(
    session: AsyncSession,
    end_date: date,
    applied_pattern_ids: Optional[list[str]] = None,
) -> dict:
    """Returns {summary_markdown, suggestions, headline_stats}.

    `end_date` is the last day of the week being reviewed. For a Monday
    cron, pass yesterday (Sunday) to review Mon→Sun.
    """
    applied_set = set(applied_pattern_ids or [])
    picks, result_rows, top3_by_race = await _load_weekly_window(session, end_date)
    results = _index_results(result_rows)

    # Headline stats for the week
    this_week, _ = _split_picks_by_week(picks, end_date)
    wins = 0
    placed = 0
    with_result = 0
    sharp_count = 0
    sharp_wins = 0
    for p in this_week:
        r = results.get((p.race_id, _normalize_horse(p.horse_name)))
        if not r or r["position"] is None or r["position"] <= 0:
            continue
        with_result += 1
        if r["winner"]:
            wins += 1
        if r["placed"] or (r["position"] and r["position"] <= 3):
            placed += 1
        if _is_sharp(p, top3_by_race):
            sharp_count += 1
            if r["winner"]:
                sharp_wins += 1

    week_start = (end_date - timedelta(days=6)).isoformat()
    headline = {
        "week_start": week_start,
        "week_end": end_date.isoformat(),
        "rank1_races_with_result": with_result,
        "rank1_wins": wins,
        "rank1_placed": placed,
        "rank1_win_pct": round(wins / with_result * 100, 1) if with_result else None,
        "rank1_place_pct": round(placed / with_result * 100, 1) if with_result else None,
        "sharp_count": sharp_count,
        "sharp_wins": sharp_wins,
        "sharp_win_pct": round(sharp_wins / sharp_count * 100, 1) if sharp_count else None,
    }

    suggestions: list[Suggestion] = []
    for detector in WEEKLY_DETECTORS:
        try:
            s = detector(end_date, picks, results, applied_set, top3_by_race)
            if s is None:
                continue
            if s.pattern_id in applied_set:
                s.severity = "low"
                s.notes = "Repeat of a pattern already applied — review whether the prior fix degraded."
            suggestions.append(s)
        except Exception as e:
            suggestions.append(Suggestion(
                id=_suggestion_id(end_date, f"detector_error_{detector.__name__}"),
                pattern_id=f"detector_error_{detector.__name__}",
                title=f"Weekly detector {detector.__name__} crashed",
                severity="low",
                rationale=f"Exception: {e}",
                paste_prompt=f"Fix the crashed detector {detector.__name__} in horse_engine/analysis/weekly_review.py",
            ))

    rank = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: rank.get(s.severity, 3))

    lines = [
        f"# Weekly review — {week_start} → {end_date.isoformat()}",
        "",
        f"**Rank-1 this week:** {wins}/{with_result} wins "
        f"({headline['rank1_win_pct'] or 0:.1f}%), "
        f"{placed} placed ({headline['rank1_place_pct'] or 0:.1f}%).",
        f"**Sharp niche this week:** {sharp_wins}/{sharp_count} wins "
        f"({headline['sharp_win_pct'] or 0:.1f}%).",
        "",
    ]
    if not suggestions:
        lines.append("_No suggestions this week — all weekly detectors clean._")
    else:
        lines.append(f"**{len(suggestions)} suggestion(s):**")
        lines.append("")
        for s in suggestions:
            lines.append(f"- **[{s.severity.upper()}] {s.title}** (`{s.pattern_id}`)")

    return {
        "summary_markdown": "\n".join(lines),
        "suggestions": [asdict(s) for s in suggestions],
        "headline_stats": headline,
    }
