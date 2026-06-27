"""Nightly review analyser.

Produces a list of structured suggestions from the last N days of
runner/result data. Each suggestion has a stable `id` so the admin
dashboard can track its lifecycle (pending → assessed → applied/dismissed)
across multiple nightly runs.

Design notes:
- Suggestions are *named patterns*, not free-form prose. Each pattern
  has its own detector, threshold, and paste-prompt template. New
  patterns are added by writing a new detector and appending to
  `DETECTORS`.
- Detector output is deterministic given the input — the same data
  produces the same suggestion id. This is critical so the dashboard
  can collapse "we suggested this on three nights running" into a single
  history row.
- A suggestion is only *worth surfacing* if the underlying signal is
  strong enough (each detector defines its own threshold). The analyser
  always runs all detectors, but only suggestions with `worth_surfacing`
  True land in the output.
- `applied_pattern_ids` lets the analyser suppress repeat suggestions
  for patterns already shipped — caller passes in the list from the DB.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from horse_engine.models.database import (
    HistoricalResultRow,
    RunnerPredictionHistoryRow,
)


_COUNTRY_CODE_RE = re.compile(r"\s*\([a-z]{2,4}\)\s*$", re.IGNORECASE)


def _normalize_horse(name: str) -> str:
    return _COUNTRY_CODE_RE.sub("", name or "").lower().strip()


@dataclass
class Suggestion:
    """One structured suggestion emitted by a detector.

    `pattern_id` groups suggestions across nights — the dashboard uses it
    to detect repeats. `id` is unique per (review_date, pattern_id).
    """
    id: str                       # stable: {pattern_id}__{review_date}
    pattern_id: str               # e.g. 'layoff_floor_drift'
    title: str
    severity: str                 # 'high' | 'medium' | 'low' (drives sort order)
    rationale: str                # plain-English why this matters
    evidence: list[dict] = field(default_factory=list)  # supporting rows
    paste_prompt: str = ""        # ready-to-paste instruction for Claude
    code_pointer: str = ""        # file:line of the code most likely to change
    metric_delta: dict = field(default_factory=dict)  # before/after for the dashboard
    status: str = "pending"       # 'pending' | 'assessed' | 'applied' | 'dismissed'
    notes: str = ""
    applied_at: Optional[str] = None
    dismissed_reason: Optional[str] = None

    def to_json(self) -> dict:
        return asdict(self)


# ---------- data loading ----------

async def _load_window(
    session: AsyncSession,
    end_date: date,
    days_back: int,
) -> tuple[list, list]:
    """Returns (latest_pred_per_horse, results_rows) for [end - days_back, end]."""
    start = (end_date - timedelta(days=days_back)).isoformat()
    end_excl = (end_date + timedelta(days=1)).isoformat()

    pred_rows = (await session.execute(
        select(
            RunnerPredictionHistoryRow.race_id,
            RunnerPredictionHistoryRow.horse_name,
            RunnerPredictionHistoryRow.win_probability,
            RunnerPredictionHistoryRow.place_probability,
            RunnerPredictionHistoryRow.model_rank,
            RunnerPredictionHistoryRow.market_rank,
            RunnerPredictionHistoryRow.enriched_json,
            RunnerPredictionHistoryRow.enriched_at,
            RunnerPredictionHistoryRow.scheduled_time,
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
    for row in pred_rows:
        key = (row.race_id, _normalize_horse(row.horse_name))
        if key in seen:
            continue
        seen.add(key)
        latest.append(row)

    result_rows = (await session.execute(
        select(HistoricalResultRow.race_id, HistoricalResultRow.horse_name,
               HistoricalResultRow.position, HistoricalResultRow.starting_price,
               HistoricalResultRow.placed, HistoricalResultRow.winner,
               HistoricalResultRow.field_size)
        .where(HistoricalResultRow.race_id >= f"{start}_")
        .where(HistoricalResultRow.race_id < f"{end_excl}_")
    )).fetchall()

    return latest, result_rows


def _index_results(result_rows: list) -> dict:
    """(race_id, horse_norm) → result dict."""
    out = {}
    for r in result_rows:
        out[(r.race_id, _normalize_horse(r.horse_name or ""))] = {
            "position": r.position,
            "starting_price": float(r.starting_price or 0),
            "placed": bool(r.placed) if r.placed is not None else False,
            "winner": bool(r.winner) if r.winner is not None else False,
            "field_size": r.field_size,
        }
    return out


def _parse_enriched(p) -> dict:
    try:
        return json.loads(p.enriched_json) if p.enriched_json else {}
    except Exception:
        return {}


# ---------- detectors ----------
# Each detector takes (review_date, picks, results_by_horse, applied_pattern_ids)
# and returns Optional[Suggestion]. Returning None means "no signal worth
# surfacing tonight" — that's fine, most nights only fire 0-2 detectors.

def _suggestion_id(review_date: date, pattern_id: str) -> str:
    return f"{pattern_id}__{review_date.isoformat()}"


def _detect_top1_winrate_drift(
    review_date: date, picks: list, results: dict, applied: set
) -> Optional[Suggestion]:
    """Has the top-1 win rate dropped sharply vs the 30-day baseline?"""
    rank1 = [p for p in picks if p.model_rank == 1]
    today_iso = review_date.isoformat()
    cutoff7 = (review_date - timedelta(days=7)).isoformat()
    today_picks = [p for p in rank1 if p.race_id.startswith(today_iso)]
    week_picks = [p for p in rank1 if cutoff7 <= p.race_id[:10] <= today_iso]
    older_picks = [p for p in rank1 if p.race_id[:10] < cutoff7]
    if len(today_picks) < 5 or len(older_picks) < 30:
        return None

    def win_pct(rows) -> Optional[float]:
        with_result = []
        for p in rows:
            r = results.get((p.race_id, _normalize_horse(p.horse_name)))
            if r and r["position"] is not None and r["position"] > 0:
                with_result.append(r["winner"])
        if not with_result:
            return None
        return round(sum(with_result) / len(with_result) * 100, 1)

    today_wr = win_pct(today_picks)
    week_wr = win_pct(week_picks)
    base_wr = win_pct(older_picks)
    if today_wr is None or base_wr is None:
        return None
    drop = base_wr - today_wr
    if drop < 8.0:  # threshold — anything smaller is normal day-to-day variance
        return None

    sev = "high" if drop >= 15 else "medium"
    return Suggestion(
        id=_suggestion_id(review_date, "top1_winrate_drift"),
        pattern_id="top1_winrate_drift",
        title=f"Top-1 win rate dropped {drop:.1f}pt vs 30d baseline",
        severity=sev,
        rationale=(
            f"Today's rank-1 picks won {today_wr:.1f}% ({len(today_picks)} races) vs "
            f"the prior-30d baseline of {base_wr:.1f}%. Week-to-date is {week_wr or 0:.1f}%. "
            f"A {drop:.1f}pt single-day drop on a small sample isn't necessarily structural — "
            "check the anomaly suggestions below first to see if the misses cluster around a "
            "known failure mode (layoff, going, market disagreement) before changing weights."
        ),
        evidence=[],
        paste_prompt=(
            "Investigate today's top-1 win-rate drop. Pull /api/admin/model-anomalies?days=1 "
            "for today's date, find the common failure mode tags, and tell me whether this is "
            "noise on a small sample or a structural issue that needs a model change. If "
            "structural, propose the specific code change with file:line and a backtest plan."
        ),
        code_pointer="horse_engine/prediction/engine.py",
        metric_delta={"baseline_30d_pct": base_wr, "today_pct": today_wr, "week_pct": week_wr},
    )


def _detect_layoff_floor_drift(
    review_date: date, picks: list, results: dict, applied: set
) -> Optional[Suggestion]:
    """Long-layoff (>365d) rank-1 picks losing badly — is the discount strong enough?"""
    rank1_layoff = []
    for p in picks:
        if p.model_rank != 1:
            continue
        e = _parse_enriched(p)
        days_off = e.get("days_since_last_run")
        if not isinstance(days_off, (int, float)) or days_off < 365:
            continue
        r = results.get((p.race_id, _normalize_horse(p.horse_name)))
        if not r or r["position"] is None or r["position"] <= 0:
            continue
        rank1_layoff.append({
            "race_id": p.race_id,
            "horse_name": p.horse_name,
            "days_off": days_off,
            "position": r["position"],
            "field_size": r["field_size"],
            "form_score": e.get("form_score"),
            "model_pct": round((p.win_probability or 0) * 100, 1),
        })

    if len(rank1_layoff) < 3:
        return None
    losses = [x for x in rank1_layoff if (x["position"] or 99) > 3]
    loss_rate = len(losses) / len(rank1_layoff) * 100
    if loss_rate < 80:  # acceptable — they're meant to lose more often than baseline
        return None

    sev = "high" if loss_rate >= 90 and len(rank1_layoff) >= 5 else "medium"
    return Suggestion(
        id=_suggestion_id(review_date, "layoff_floor_drift"),
        pattern_id="layoff_floor_drift",
        title=f"Long-layoff rank-1 picks: {loss_rate:.0f}% miss rate ({len(losses)}/{len(rank1_layoff)})",
        severity=sev,
        rationale=(
            f"Over the last week, {len(rank1_layoff)} rank-1 picks had 365d+ layoffs and "
            f"{len(losses)} of them ({loss_rate:.0f}%) finished outside the placings. The current "
            "extreme-layoff floor at 0.15 (form.py:95) may be too generous — consider tightening "
            "the floor toward 0.10 or pulling LAYOFF_EXTREME_DAYS down from 730 to 547 (18 months)."
        ),
        evidence=rank1_layoff[:5],
        paste_prompt=(
            "The extreme-layoff floor needs review. In horse_engine/enrichers/form.py:94-95 the "
            "current values are LAYOFF_EXTREME_DAYS=730 and LAYOFF_EXTREME_SCORE=0.15. "
            "Run a backtest comparing 4 scenarios: (a) current, (b) floor=0.10, (c) "
            "LAYOFF_EXTREME_DAYS=547, (d) both. Use the rank-1 win-rate and ROI as metrics. "
            "If (d) wins on win-rate, ship it. Otherwise pick the winner and explain why."
        ),
        code_pointer="horse_engine/enrichers/form.py:94",
        metric_delta={"miss_rate_pct": round(loss_rate, 1), "sample_size": len(rank1_layoff)},
    )


def _detect_going_calibration_drift(
    review_date: date, picks: list, results: dict, applied: set
) -> Optional[Suggestion]:
    """Are soft/heavy-track rank-1 picks still over-predicting after the 0.55 calibration?"""
    by_going = {"soft": [], "heavy": [], "good": [], "firm": [], "synthetic": []}
    for p in picks:
        if p.model_rank != 1:
            continue
        e = _parse_enriched(p)
        going = e.get("track_condition_category")
        if going not in by_going:
            continue
        r = results.get((p.race_id, _normalize_horse(p.horse_name)))
        if not r or r["position"] is None or r["position"] <= 0:
            continue
        by_going[going].append({
            "model_pct": (p.win_probability or 0) * 100,
            "won": r["winner"],
        })

    def stats(rows):
        if len(rows) < 8:
            return None
        avg_model = sum(x["model_pct"] for x in rows) / len(rows)
        actual = sum(1 for x in rows if x["won"]) / len(rows) * 100
        return {"n": len(rows), "avg_model_pct": round(avg_model, 1),
                "actual_pct": round(actual, 1),
                "ratio": round(actual / avg_model, 2) if avg_model > 0 else None}

    dry_rows = by_going["good"] + by_going["firm"] + by_going["synthetic"]
    dry = stats(dry_rows)
    soft = stats(by_going["soft"])
    heavy = stats(by_going["heavy"])
    if not dry:
        return None

    flagged = []
    for label, s in (("soft", soft), ("heavy", heavy)):
        if not s or not s["ratio"] or not dry["ratio"]:
            continue
        # If wet ratio is materially below dry ratio, the 0.55 multiplier
        # is too generous — the model is still over-confident on wet tracks.
        gap = dry["ratio"] - s["ratio"]
        if gap >= 0.25:
            flagged.append({"going": label, "stats": s, "dry_ratio": dry["ratio"],
                            "gap": round(gap, 2)})

    if not flagged:
        return None

    return Suggestion(
        id=_suggestion_id(review_date, "going_calibration_drift"),
        pattern_id="going_calibration_drift",
        title=f"Going calibration drift on {', '.join(f['going'] for f in flagged)}",
        severity="medium",
        rationale=(
            f"Dry-going rank-1 picks calibrate close to expected (ratio {dry['ratio']}). "
            f"Wet-going picks are materially below: " +
            "; ".join(f"{f['going']} ratio {f['stats']['ratio']} ({f['stats']['n']} races)" for f in flagged) +
            ". The current 0.55 multiplier in engine.py may need to drop further on heavy, or "
            "be split into separate soft/heavy multipliers if their ratios diverge."
        ),
        evidence=flagged,
        paste_prompt=(
            "The going calibration in horse_engine/prediction/engine.py (the `* 0.55` multiplier "
            "on soft/heavy track_condition_category) needs a refresh. "
            "Run /api/admin/bets/going-calibration?days=60 and tell me the right per-going "
            "multiplier. If soft and heavy diverge by more than 0.15, split them. "
            "Ship the change and re-run the calibration backtest to confirm both ratios land in "
            "[0.85, 1.15]."
        ),
        code_pointer="horse_engine/prediction/engine.py (going calibration)",
        metric_delta={"flagged": flagged, "dry_baseline_ratio": dry["ratio"]},
    )


def _detect_anomaly_cluster(
    review_date: date, picks: list, results: dict, applied: set
) -> Optional[Suggestion]:
    """Today's anomalies cluster around a single failure mode → investigate that mode."""
    today_iso = review_date.isoformat()
    today_rank1 = [p for p in picks if p.model_rank == 1 and p.race_id.startswith(today_iso)]
    if len(today_rank1) < 10:
        return None

    tag_counts: dict[str, int] = {}
    examples: dict[str, list[dict]] = {}
    for p in today_rank1:
        r = results.get((p.race_id, _normalize_horse(p.horse_name)))
        if not r or r["position"] is None or r["position"] <= 0 or r["starting_price"] < 8.0:
            continue
        if r["position"] <= 5:
            continue
        e = _parse_enriched(p)
        days_off = e.get("days_since_last_run")
        form_score = e.get("form_score")
        tags = []
        if isinstance(days_off, (int, float)) and days_off > 180:
            tags.append("long_layoff")
        if (e.get("starts_last_10") or 0) <= 2:
            tags.append("thin_record")
        if isinstance(form_score, (int, float)) and form_score >= 0.75 and isinstance(days_off, (int, float)) and days_off > 120:
            tags.append("stale_form_signal")
        if (p.market_rank or 0) >= 5:
            tags.append("market_disagreed")
        if (e.get("runs_this_prep") or 0) == 0:
            tags.append("first_up_resuming")
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
            examples.setdefault(t, []).append({
                "race_id": p.race_id, "horse_name": p.horse_name,
                "position": r["position"], "starting_price": r["starting_price"],
                "days_off": days_off, "form_score": form_score,
            })

    if not tag_counts:
        return None
    top_tag, top_count = max(tag_counts.items(), key=lambda x: x[1])
    if top_count < 3:
        return None

    return Suggestion(
        id=_suggestion_id(review_date, f"anomaly_cluster_{top_tag}"),
        pattern_id=f"anomaly_cluster_{top_tag}",
        title=f"{top_count} of today's rank-1 misses tagged '{top_tag}'",
        severity="medium",
        rationale=(
            f"{top_count} rank-1 picks today were 8+ SP, finished worse than 5th, and shared the "
            f"'{top_tag}' failure mode. Other clusters today: " +
            ", ".join(f"{k} ({v})" for k, v in sorted(tag_counts.items(), key=lambda x: -x[1])[1:4]) +
            ". If '{top_tag}' is now the dominant failure mode for two consecutive nights, the "
            "discount/penalty for that input needs strengthening."
        ),
        evidence=examples[top_tag][:5],
        paste_prompt=(
            f"Today's biggest rank-1 anomaly cluster was '{top_tag}' ({top_count} picks). "
            "Pull /api/admin/model-anomalies?days=14 and tell me if this tag has been growing as "
            "a share of total anomalies. If yes, propose the specific code change to address it. "
            "If no, this is single-day noise — dismiss."
        ),
        code_pointer="horse_engine/prediction/engine.py",
        metric_delta={"dominant_tag": top_tag, "count_today": top_count,
                      "other_tags": dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:6])},
    )


DETECTORS = [
    _detect_top1_winrate_drift,
    _detect_layoff_floor_drift,
    _detect_going_calibration_drift,
    _detect_anomaly_cluster,
]


# ---------- entrypoint ----------

async def generate_review(
    session: AsyncSession,
    review_date: date,
    applied_pattern_ids: Optional[list[str]] = None,
    window_days: int = 30,
) -> dict:
    """Returns {summary_markdown, suggestions, headline_stats}.

    `applied_pattern_ids` is consulted so we can mark a suggestion as
    'already-shipped' instead of re-suggesting verbatim — the detector
    still runs (so we can compare numbers across nights), but the
    suggestion is decorated with `notes` flagging it as repeat-of-applied
    and dropped to severity=low.
    """
    applied_set = set(applied_pattern_ids or [])
    picks, result_rows = await _load_window(session, review_date, window_days)
    results = _index_results(result_rows)

    # Headline stats — always computed regardless of detectors firing.
    today_iso = review_date.isoformat()
    today_rank1 = [p for p in picks if p.model_rank == 1 and p.race_id.startswith(today_iso)]
    today_with_result = 0
    today_wins = 0
    today_places = 0
    for p in today_rank1:
        r = results.get((p.race_id, _normalize_horse(p.horse_name)))
        if not r or r["position"] is None or r["position"] <= 0:
            continue
        today_with_result += 1
        if r["winner"]:
            today_wins += 1
        if r["placed"] or (r["position"] and r["position"] <= 3):
            today_places += 1

    headline = {
        "review_date": today_iso,
        "rank1_races_with_result": today_with_result,
        "rank1_wins": today_wins,
        "rank1_places": today_places,
        "rank1_win_pct": round(today_wins / today_with_result * 100, 1) if today_with_result else None,
        "rank1_place_pct": round(today_places / today_with_result * 100, 1) if today_with_result else None,
    }

    suggestions: list[Suggestion] = []
    for detector in DETECTORS:
        try:
            s = detector(review_date, picks, results, applied_set)
            if s is None:
                continue
            if s.pattern_id in applied_set:
                s.severity = "low"
                s.notes = "Repeat of a pattern already applied — review whether the prior fix degraded."
            suggestions.append(s)
        except Exception as e:
            # A broken detector must never tank the whole review — log and skip.
            suggestions.append(Suggestion(
                id=_suggestion_id(review_date, f"detector_error_{detector.__name__}"),
                pattern_id=f"detector_error_{detector.__name__}",
                title=f"Detector {detector.__name__} crashed",
                severity="low",
                rationale=f"Exception: {e}",
                paste_prompt=f"Fix the crashed detector {detector.__name__} in horse_engine/analysis/nightly_review.py",
            ))

    # Sort high → medium → low
    rank = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: rank.get(s.severity, 3))

    # Markdown summary the dashboard can render as-is.
    lines = [
        f"# Nightly review — {today_iso}",
        "",
        f"**Rank-1 today:** {today_wins}/{today_with_result} wins "
        f"({headline['rank1_win_pct'] or 0:.1f}%), "
        f"{today_places} placed ({headline['rank1_place_pct'] or 0:.1f}%).",
        "",
    ]
    if not suggestions:
        lines.append("_No suggestions tonight — all detectors clean._")
    else:
        lines.append(f"**{len(suggestions)} suggestion(s):**")
        lines.append("")
        for s in suggestions:
            lines.append(f"- **[{s.severity.upper()}] {s.title}** (`{s.pattern_id}`)")

    return {
        "summary_markdown": "\n".join(lines),
        "suggestions": [s.to_json() for s in suggestions],
        "headline_stats": headline,
    }
