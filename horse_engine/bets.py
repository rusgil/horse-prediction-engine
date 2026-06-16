"""Trifecta bet recommender — paper-trading.

For each race that passes the gate criteria, generate a fixed-shape
basket of $2 flexi-box trifectas. The basket mirrors the manual play
worked out for Longreach R7 (2026-06-15):

  core_top3           top 3 by win model (3-horse box)
  core_top4           top 4 by win model (4-horse box)
  value_runner1       top 2 + horse #5 (3-horse box)
  value_runner2       top 2 + horse #6 (3-horse box)
  no_favourite_hedge  horses 2..5 (4-horse box, banker excluded)

Hit rule (settlement): the actual top-3 finishers must all be present in
box_horses. Payout = (stake / num_permutations) * trifecta_dividend.
"""
from typing import Optional

DEFAULT_STAKE = 2.0
MIN_FIELD_SIZE = 7        # 60d backtest: field=7 hits 48.7%, sweet spot
MAX_FIELD_SIZE = 13       # 14+ runners drops to 22% hit rate — skip
MIN_TOP1_WIN_PCT = 20.0   # below this, the model has nothing to say
# 60d backtest trap zone: rank-1 25-30% hits only 27% (vs 34% at 20-25%
# and 40%+ at 30-40%). Skip races in this band — the model has a weak
# favourite that's been wrong more often than the neighbouring buckets.
TRAP_ZONE_LO = 25.0
TRAP_ZONE_HI = 30.0


def _perms(n: int) -> int:
    """Permutations for an n-horse box = n × (n-1) × (n-2)."""
    return n * (n - 1) * (n - 2) if n >= 3 else 0


def generate_recommendations(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """
    Build the 5-bet basket for a single race.

    Args:
      runners: list of {tab_number, horse_name, win_probability, place_probability,
        model_rank, cancelled}. win_probability is a 0..1 fraction.
      stake: per-box stake (default $2).

    Returns:
      list of dicts: {strategy_label, box_horses, box_horse_names,
      num_permutations, stake_dollars}. Empty list if the race fails
      the gate criteria.
    """
    active = [r for r in runners if not r.get("cancelled")]
    active = [r for r in active if r.get("model_rank") is not None and r.get("tab_number") is not None]
    if not (MIN_FIELD_SIZE <= len(active) <= MAX_FIELD_SIZE):
        return []

    active.sort(key=lambda r: r["model_rank"])
    top1_win_pct = (active[0].get("win_probability") or 0) * 100
    if top1_win_pct < MIN_TOP1_WIN_PCT:
        return []
    if TRAP_ZONE_LO <= top1_win_pct < TRAP_ZONE_HI:
        return []

    def slot(i: int) -> Optional[dict]:
        return active[i] if i < len(active) else None

    t1, t2, t3, t4, t5, t6 = (slot(i) for i in range(6))

    bets: list[dict] = []

    def _add(label: str, horses: list[dict]):
        bets.append({
            "strategy_label": label,
            "box_horses": [h["tab_number"] for h in horses],
            "box_horse_names": [h["horse_name"] for h in horses],
            "num_permutations": _perms(len(horses)),
            "stake_dollars": stake,
        })

    # 1. Top-3 box (the model's "place top 3")
    if t3:
        _add("core_top3", [t1, t2, t3])

    # 2. Top-4 box (wider coverage on the model's strongest 4)
    if t4:
        _add("core_top4", [t1, t2, t3, t4])

    # 3. Top 2 + value runner #1 (next-best horse outside top 4)
    if t5 and t2:
        _add("value_runner1", [t1, t2, t5])

    # 4. Top 2 + value runner #2 (alternative value runner)
    if t6 and t2:
        _add("value_runner2", [t1, t2, t6])

    # 5. No-favourite hedge (top 5 minus the model favourite)
    if t5 and t4 and t3 and t2:
        _add("no_favourite_hedge", [t2, t3, t4, t5])

    # 6. The Sweep — a single 5-horse box of top 5 by win rank. 60-day
    # backtest had this matching the 5-box basket's race hit rate at 1/5th
    # the cost; we generate it alongside the spread so the two strategies
    # can be A/B compared on real dividends as they come in.
    if t5:
        _add("wide_top5", [t1, t2, t3, t4, t5])

    return bets


# Strategy-group taxonomy. Maps each box strategy_label to a top-level
# brand. The Lab UI surfaces stats / picks / ledger entries per group.
STRATEGY_GROUP = {
    "core_top3": "spread",
    "core_top4": "spread",
    "value_runner1": "spread",
    "value_runner2": "spread",
    "no_favourite_hedge": "spread",
    "wide_top5": "sweep",
}
STRATEGY_GROUP_LABELS = {
    "spread": "The Spread",
    "sweep": "The Sweep",
}


# ── Alternative strategies for backtest comparison ───────────────────────
#
# Each generator follows the same shape as generate_recommendations: take
# the list of runners, return a list of bets. Filter rules (min field,
# trap zone, etc.) are applied per-strategy so each can have its own
# selectivity criteria.

def _sorted_active(runners: list[dict], rank_key: str = "model_rank") -> list[dict]:
    active = [r for r in runners if not r.get("cancelled")
              and r.get(rank_key) is not None and r.get("tab_number") is not None]
    active.sort(key=lambda r: r[rank_key])
    return active


def strategy_tight_top3(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Single 3-horse box of the model's top 3 by win rank.
    Tightest, cheapest box ($12/race). Hits when all three favourites
    fill the placings — high precision, lower recall than the 5-box
    basket."""
    active = _sorted_active(runners, "model_rank")
    if not (MIN_FIELD_SIZE <= len(active) <= MAX_FIELD_SIZE):
        return []
    w1 = (active[0].get("win_probability") or 0) * 100
    if w1 < MIN_TOP1_WIN_PCT or (TRAP_ZONE_LO <= w1 < TRAP_ZONE_HI):
        return []
    top3 = active[:3]
    return [{
        "strategy_label": "tight_top3",
        "box_horses": [h["tab_number"] for h in top3],
        "box_horse_names": [h["horse_name"] for h in top3],
        "num_permutations": _perms(3),
        "stake_dollars": stake,
    }]


def strategy_wide_top5(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Single 5-horse box of top 5 by win rank ($120/race at $2 flexi).
    Widest single box — covers more ground but pays out less per hit.
    Tests whether breadth beats the multi-box basket's depth."""
    active = _sorted_active(runners, "model_rank")
    if not (MIN_FIELD_SIZE <= len(active) <= MAX_FIELD_SIZE) or len(active) < 5:
        return []
    w1 = (active[0].get("win_probability") or 0) * 100
    if w1 < MIN_TOP1_WIN_PCT or (TRAP_ZONE_LO <= w1 < TRAP_ZONE_HI):
        return []
    top5 = active[:5]
    return [{
        "strategy_label": "wide_top5",
        "box_horses": [h["tab_number"] for h in top5],
        "box_horse_names": [h["horse_name"] for h in top5],
        "num_permutations": _perms(5),
        "stake_dollars": stake,
    }]


def strategy_place_top4(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """4-horse box ranked by place_model_rank (not win_model_rank).
    The place model is trained specifically for top-3 finishes — should
    align better with trifecta hit conditions than the win model."""
    active = _sorted_active(runners, "place_model_rank")
    if not (MIN_FIELD_SIZE <= len(active) <= MAX_FIELD_SIZE) or len(active) < 4:
        return []
    # Still gate on win-model conviction so we don't bet noise races.
    win_sorted = _sorted_active(runners, "model_rank")
    if win_sorted:
        w1 = (win_sorted[0].get("win_probability") or 0) * 100
        if w1 < MIN_TOP1_WIN_PCT or (TRAP_ZONE_LO <= w1 < TRAP_ZONE_HI):
            return []
    top4 = active[:4]
    return [{
        "strategy_label": "place_top4",
        "box_horses": [h["tab_number"] for h in top4],
        "box_horse_names": [h["horse_name"] for h in top4],
        "num_permutations": _perms(4),
        "stake_dollars": stake,
    }]


def strategy_exotic_top3(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """3-horse box ranked by exotic_model_rank (the dedicated exotic-bet
    model). Cheapest place-aware variant — tests if exotic model's top 3
    differs meaningfully from win-model top 3."""
    active = _sorted_active(runners, "exotic_model_rank")
    if not (MIN_FIELD_SIZE <= len(active) <= MAX_FIELD_SIZE):
        return []
    win_sorted = _sorted_active(runners, "model_rank")
    if win_sorted:
        w1 = (win_sorted[0].get("win_probability") or 0) * 100
        if w1 < MIN_TOP1_WIN_PCT or (TRAP_ZONE_LO <= w1 < TRAP_ZONE_HI):
            return []
    top3 = active[:3]
    return [{
        "strategy_label": "exotic_top3",
        "box_horses": [h["tab_number"] for h in top3],
        "box_horse_names": [h["horse_name"] for h in top3],
        "num_permutations": _perms(3),
        "stake_dollars": stake,
    }]


def strategy_anchor(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """5-horse box of top 5 by win, but ONLY fires when rank-1 win prob
    >= 35%. Concentrates capital on races with a genuinely strong model
    favourite — the backtest showed rank-1 40%+ races hit 45.5%.
    Same box shape as The Sweep, narrower entry criteria."""
    active = _sorted_active(runners, "model_rank")
    if not (MIN_FIELD_SIZE <= len(active) <= MAX_FIELD_SIZE) or len(active) < 5:
        return []
    w1 = (active[0].get("win_probability") or 0) * 100
    if w1 < 35.0:
        return []
    top5 = active[:5]
    return [{
        "strategy_label": "anchor",
        "box_horses": [h["tab_number"] for h in top5],
        "box_horse_names": [h["horse_name"] for h in top5],
        "num_permutations": _perms(5),
        "stake_dollars": stake,
    }]


def strategy_net(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """6-horse box of top 6 by win rank. Wider than The Sweep —
    captures more outsiders at the cost of bigger permutation count
    (120 perms vs 60). Tests whether stretching one box further
    increases race hit rate enough to justify the dilution."""
    active = _sorted_active(runners, "model_rank")
    if not (MIN_FIELD_SIZE <= len(active) <= MAX_FIELD_SIZE) or len(active) < 6:
        return []
    w1 = (active[0].get("win_probability") or 0) * 100
    if w1 < MIN_TOP1_WIN_PCT or (TRAP_ZONE_LO <= w1 < TRAP_ZONE_HI):
        return []
    top6 = active[:6]
    return [{
        "strategy_label": "net",
        "box_horses": [h["tab_number"] for h in top6],
        "box_horse_names": [h["horse_name"] for h in top6],
        "num_permutations": _perms(6),
        "stake_dollars": stake,
    }]


def strategy_blend(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Box the UNION of (top 3 by win) and (top 3 by place). Combines
    two different model signals — the place model picks up consistent
    top-3 finishers that the win model might rank lower. Box size
    varies 3–5 horses depending on overlap between the two rankings."""
    win_active = _sorted_active(runners, "model_rank")
    place_active = _sorted_active(runners, "place_model_rank")
    if not (MIN_FIELD_SIZE <= len(win_active) <= MAX_FIELD_SIZE):
        return []
    w1 = (win_active[0].get("win_probability") or 0) * 100
    if w1 < MIN_TOP1_WIN_PCT or (TRAP_ZONE_LO <= w1 < TRAP_ZONE_HI):
        return []
    if len(win_active) < 3 or len(place_active) < 3:
        return []
    # Preserve order: win-top first, then place-top runners not already in.
    seen_tabs: set = set()
    blend: list[dict] = []
    for r in win_active[:3] + place_active[:3]:
        if r["tab_number"] in seen_tabs:
            continue
        seen_tabs.add(r["tab_number"])
        blend.append(r)
    if len(blend) < 3:
        return []
    return [{
        "strategy_label": "blend",
        "box_horses": [h["tab_number"] for h in blend],
        "box_horse_names": [h["horse_name"] for h in blend],
        "num_permutations": _perms(len(blend)),
        "stake_dollars": stake,
    }]


def strategy_sniper(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Premium-grade 4-horse box. ONLY fires for races where:
      - field size 7–10 (small enough to box tightly)
      - rank-1 win >= 30% (model has strong conviction)
      - top-3 sum >= 60% (concentrated probability mass)
    Targets the highest-hit-rate buckets from the 60-day backtest. Low
    volume but should have the best precision per dollar."""
    active = _sorted_active(runners, "model_rank")
    if not (MIN_FIELD_SIZE <= len(active) <= 10) or len(active) < 4:
        return []
    w1 = (active[0].get("win_probability") or 0) * 100
    if w1 < 30.0:
        return []
    top3_sum = sum((active[i].get("win_probability") or 0) * 100 for i in range(3))
    if top3_sum < 60.0:
        return []
    top4 = active[:4]
    return [{
        "strategy_label": "sniper",
        "box_horses": [h["tab_number"] for h in top4],
        "box_horse_names": [h["horse_name"] for h in top4],
        "num_permutations": _perms(4),
        "stake_dollars": stake,
    }]


def strategy_pocket(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """4-horse box, ONLY fires for fields 7–8. The backtest showed
    field=7 hits 48.7% and field 8-9 hits 32.5% — this strategy lives
    entirely in the sweet spot. Tighter than The Sweep, broader than
    a 3-horse box, restricted to the high-hit-rate field range."""
    active = _sorted_active(runners, "model_rank")
    if not (7 <= len(active) <= 8):
        return []
    w1 = (active[0].get("win_probability") or 0) * 100
    if w1 < MIN_TOP1_WIN_PCT or (TRAP_ZONE_LO <= w1 < TRAP_ZONE_HI):
        return []
    top4 = active[:4]
    return [{
        "strategy_label": "pocket",
        "box_horses": [h["tab_number"] for h in top4],
        "box_horse_names": [h["horse_name"] for h in top4],
        "num_permutations": _perms(4),
        "stake_dollars": stake,
    }]


def strategy_small_field_only(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Single 3-horse box of top 3 by win — ONLY for races with field ≤ 9
    (where the backtest showed 32-48% race-level hit rates). Bets very
    sparingly but only on the strongest setups."""
    active = _sorted_active(runners, "model_rank")
    if not (MIN_FIELD_SIZE <= len(active) <= 9):
        return []
    w1 = (active[0].get("win_probability") or 0) * 100
    if w1 < MIN_TOP1_WIN_PCT or (TRAP_ZONE_LO <= w1 < TRAP_ZONE_HI):
        return []
    top3 = active[:3]
    return [{
        "strategy_label": "small_field_top3",
        "box_horses": [h["tab_number"] for h in top3],
        "box_horse_names": [h["horse_name"] for h in top3],
        "num_permutations": _perms(3),
        "stake_dollars": stake,
    }]


# Registry — keys map to a function (runners, stake) -> list[bets].
# Used by the backtest endpoint to evaluate strategies side-by-side.
STRATEGY_REGISTRY = {
    "baseline_5box": generate_recommendations,
    "tight_top3": strategy_tight_top3,
    "wide_top5": strategy_wide_top5,
    "place_top4": strategy_place_top4,
    "exotic_top3": strategy_exotic_top3,
    "small_field_top3": strategy_small_field_only,
    "anchor": strategy_anchor,
    "net": strategy_net,
    "blend": strategy_blend,
    "sniper": strategy_sniper,
    "pocket": strategy_pocket,
}


def is_trifecta_hit(box_horses: list[int], actual_top3: list[int]) -> bool:
    """Top-3 finishers must all be in the box (order doesn't matter)."""
    if len(actual_top3) < 3:
        return False
    box_set = set(box_horses)
    return all(t in box_set for t in actual_top3[:3])


def compute_payout(stake: float, num_permutations: int, dividend: float, is_hit: bool) -> tuple[float, float]:
    """Return (payout, pnl). Flexi-bet payout = (stake / perms) * dividend."""
    if not is_hit or num_permutations <= 0 or not dividend:
        return 0.0, -stake
    payout = round((stake / num_permutations) * dividend, 2)
    return payout, round(payout - stake, 2)
