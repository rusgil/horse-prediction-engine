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

# Metro meeting filter — box trifectas only get profitable when the
# dividend pool is big enough. Country/provincial trifectas average
# $50-130 (below The Spread's 4-horse-box breakeven of ~$80); metro
# Saturday meetings typically clear $200+. Filtering here means we
# stake $10/race only on the venues where the math works.
METRO_VENUES: set[str] = {
    # NSW
    "randwick", "royal-randwick", "rosehill", "rosehill-gardens",
    "canterbury-park", "warwick-farm",
    # VIC
    "caulfield", "caulfield-heath", "flemington", "moonee-valley",
    "sandown", "sandown-lakeside", "sandown-hillside", "the-valley",
    # QLD
    "doomben", "eagle-farm",
    # SA
    "morphettville", "morphettville-parks",
    # WA
    "ascot", "belmont",
}


def is_metro_venue(venue_code: str) -> bool:
    """True if the given venue slug is a registered metro track."""
    return (venue_code or "").lower() in METRO_VENUES

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
    # Trap-zone filter (25-30%) removed 2026-06-17 — it was based on a
    # noisy 60d sample and was rejecting otherwise-fine races (rank-1
    # 28% sits right in the band), creating inconsistency where older
    # bets exist but new strategies (Trio/Quad/Stack) wouldn't generate.

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

    # 6. The Trio — a single 3-horse box of the model's top 3 by win.
    # Tightest box (6 perms); biggest per-hit payout (33¢/perm at $2
    # flexi). Lower hit rate but the most profitable per-hit at typical
    # AU country dividends ($50-150 range).
    if t3:
        _add("trio_only", [t1, t2, t3])

    # 7. The Quad — a single 4-horse box of the model's top 4 by win.
    # Middle-ground (24 perms, 8¢/perm). Broader coverage than Trio
    # at half the per-hit payout. Real-data ROI between Trio and
    # Sweep at typical AU country dividends.
    if t4:
        _add("quad_only", [t1, t2, t3, t4])

    # 8. The Stack — 5 different 3-horse trifecta boxes per race.
    # 270d portfolio sim: 15.7% race hit rate, +8.6% ROI at $200 avg
    # dividend, +$19.5k total profit (highest absolute dollars of any
    # portfolio tested at metro divs). 5 boxes × $2 = $10/race stake.
    # Each box targets a different miss pattern.
    if t5 and t4 and t3:
        _add("stack_top3",   [t1, t2, t3])  # all favourites land
        _add("stack_skip3",  [t1, t2, t4])  # rank-3 misses
        _add("stack_skip2",  [t1, t3, t4])  # rank-2 misses
        _add("stack_no_fav", [t2, t3, t4])  # favourite collapses
        _add("stack_value",  [t1, t2, t5])  # value runner places

    return bets


# Strategy-group taxonomy. Maps each box strategy_label to a top-level
# brand. The Lab UI surfaces stats / picks / ledger entries per group.
STRATEGY_GROUP = {
    "core_top3": "spread",
    "core_top4": "spread",
    "value_runner1": "spread",
    "value_runner2": "spread",
    "no_favourite_hedge": "spread",
    # Legacy single-box wides — kept in the map so historical rows still
    # report correctly under their original group, but no new rows are
    # generated. The recommender now emits trio_only / quad_only instead.
    "wide_top5": "sweep",
    "wide_top6": "net",
    "trio_only": "trio",
    "quad_only": "quad",
    # The Stack — 5 × 3-horse trifecta boxes per race.
    "stack_top3":   "stack",
    "stack_skip3":  "stack",
    "stack_skip2":  "stack",
    "stack_no_fav": "stack",
    "stack_value":  "stack",
}
STRATEGY_GROUP_LABELS = {
    "spread": "The Spread",
    "sweep": "The Sweep",
    "net":   "The Net",
    "trio":  "The Trio",
    "quad":  "The Quad",
    "stack": "The Stack",
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


def is_trifecta_hit(box_horses: list[int], actual_top3: list[int]) -> bool:
    """Top-3 finishers must all be in the box (order doesn't matter)."""
    if len(actual_top3) < 3:
        return False
    box_set = set(box_horses)
    return all(t in box_set for t in actual_top3[:3])


def is_first_four_hit(box_horses: list[int], actual_top4: list[int]) -> bool:
    """Top-4 finishers must all be in the box (order doesn't matter)."""
    if len(actual_top4) < 4:
        return False
    box_set = set(box_horses)
    return all(t in box_set for t in actual_top4[:4])


def is_hit(box_horses: list[int], actual: list[int], bet_type: str) -> bool:
    """Dispatch hit check by bet type. Used in settlement + backtest."""
    if bet_type == "first_four":
        return is_first_four_hit(box_horses, actual)
    return is_trifecta_hit(box_horses, actual)


def is_standout_hit(bet: dict, actual: list[int]) -> bool:
    """Standout bets: certain horses must finish in specific positions,
    other positions filled from a box. Encoded on the bet dict as:
      bet['banker_tabs']: dict {1: tab, 2: tab, ...} — required positions
      bet['box_horses']: list — horses that fill non-banker positions
    Trifecta standout checks against actual[:3]; first-four against
    actual[:4]."""
    bet_type = bet.get("bet_type", "trifecta")
    needed_positions = 4 if bet_type == "first_four" else 3
    if len(actual) < needed_positions:
        return False
    actual_n = actual[:needed_positions]
    bankers = bet.get("banker_tabs") or {}
    # Each banker must finish at the locked position.
    for pos, required_tab in bankers.items():
        if actual_n[int(pos) - 1] != required_tab:
            return False
    # Remaining positions must be filled by horses in the box.
    box_set = set(bet.get("box_horses") or [])
    remaining_actual = [actual_n[i] for i in range(needed_positions)
                        if (i + 1) not in bankers]
    return all(t in box_set for t in remaining_actual)


# ─── Premium-precision strategies (target: >40% hit rate) ─────────────────
# All apply tight selection filters so the box only fires on races where
# the model is most confident. Lower volume than the wider strategies but
# the goal here is hit rate, not coverage.

def _by(runners: list[dict], key: str) -> list[dict]:
    return sorted(
        [r for r in runners
         if not r.get("cancelled")
         and r.get(key) is not None
         and r.get("tab_number") is not None],
        key=lambda r: r[key],
    )


def strategy_surgeon(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """3-horse trifecta box of the model's top 3 by PLACE rank.
    Only fires when top-3 place probabilities sum to ≥ 60% — the place
    model is high-conviction. Uses the place-trained model since
    trifecta hit = top-3 finish (which is what the place model is for).
    """
    place_sorted = _by(runners, "place_model_rank")
    if len(place_sorted) < 3 or not (MIN_FIELD_SIZE <= len(place_sorted) <= MAX_FIELD_SIZE):
        return []
    top3 = place_sorted[:3]
    place_sum = sum((r.get("place_probability") or 0) * 100 for r in top3)
    if place_sum < 60.0:
        return []
    return [{
        "strategy_label": "surgeon",
        "bet_type": "trifecta",
        "box_horses": [h["tab_number"] for h in top3],
        "box_horse_names": [h["horse_name"] for h in top3],
        "num_permutations": _perms(3),
        "stake_dollars": stake,
    }]


def strategy_first4_place(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """4-horse FIRST FOUR box of top 4 by place rank. Top-4 place sum ≥ 70%.
    First four pays better per perm than trifecta on similar dividends —
    smaller pool, fewer winning tickets."""
    place_sorted = _by(runners, "place_model_rank")
    if len(place_sorted) < 4 or not (MIN_FIELD_SIZE <= len(place_sorted) <= MAX_FIELD_SIZE):
        return []
    top4 = place_sorted[:4]
    place_sum = sum((r.get("place_probability") or 0) * 100 for r in top4)
    if place_sum < 70.0:
        return []
    return [{
        "strategy_label": "first4_place",
        "bet_type": "first_four",
        "box_horses": [h["tab_number"] for h in top4],
        "box_horse_names": [h["horse_name"] for h in top4],
        "num_permutations": _perms(4) * (len(top4) - 3),  # 4×3×2×1 = 24 perms for first four
        "stake_dollars": stake,
    }]


def strategy_hybrid(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """3-horse trifecta box from a 50/50 blended win+place ranking.
    Captures horses the place model loves that the win model under-rates.
    Rank-1 win ≥ 30% to ensure a real favourite anchors the box."""
    active = [r for r in runners
              if not r.get("cancelled")
              and r.get("tab_number") is not None
              and r.get("win_probability") is not None
              and r.get("place_probability") is not None]
    if not (MIN_FIELD_SIZE <= len(active) <= MAX_FIELD_SIZE):
        return []
    win_sorted = sorted(active, key=lambda r: -(r["win_probability"] or 0))
    if (win_sorted[0]["win_probability"] or 0) * 100 < 30.0:
        return []
    # Blended score: 0.5 × win_prob + 0.5 × place_prob
    blended = sorted(active, key=lambda r: -(0.5 * (r["win_probability"] or 0)
                                              + 0.5 * (r["place_probability"] or 0)))
    top3 = blended[:3]
    return [{
        "strategy_label": "hybrid",
        "bet_type": "trifecta",
        "box_horses": [h["tab_number"] for h in top3],
        "box_horse_names": [h["horse_name"] for h in top3],
        "num_permutations": _perms(3),
        "stake_dollars": stake,
    }]


def strategy_tight(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """3-horse trifecta box of model's top 3 by win, ONLY for small
    fields (7-8) where top-3 win sum is concentrated (≥ 65%). Tight
    + concentrated = the model's strongest signal."""
    win_sorted = _by(runners, "model_rank")
    if not (7 <= len(win_sorted) <= 8):
        return []
    if len(win_sorted) < 3:
        return []
    top3 = win_sorted[:3]
    win_sum = sum((r.get("win_probability") or 0) * 100 for r in top3)
    if win_sum < 65.0:
        return []
    return [{
        "strategy_label": "tight",
        "bet_type": "trifecta",
        "box_horses": [h["tab_number"] for h in top3],
        "box_horse_names": [h["horse_name"] for h in top3],
        "num_permutations": _perms(3),
        "stake_dollars": stake,
    }]


def strategy_premium_first4(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """4-horse first four box of top 4 by win. Premium gate:
    rank-1 ≥ 35% AND top-3 win sum ≥ 70%. Only fires on races where the
    model has high conviction across the placings."""
    win_sorted = _by(runners, "model_rank")
    if len(win_sorted) < 4 or not (MIN_FIELD_SIZE <= len(win_sorted) <= MAX_FIELD_SIZE):
        return []
    w1 = (win_sorted[0].get("win_probability") or 0) * 100
    if w1 < 35.0:
        return []
    top3_sum = sum((win_sorted[i].get("win_probability") or 0) * 100 for i in range(3))
    if top3_sum < 70.0:
        return []
    top4 = win_sorted[:4]
    return [{
        "strategy_label": "premium_first4",
        "bet_type": "first_four",
        "box_horses": [h["tab_number"] for h in top4],
        "box_horse_names": [h["horse_name"] for h in top4],
        "num_permutations": 24,  # P(4,4) = 24 perms for first four
        "stake_dollars": stake,
    }]


# ─── Standout strategies (banker + box, position-anchored) ───────────────

def strategy_banker_win(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Rank-1 horse MUST finish 1st. 2nd/3rd box from {rank 2, 3, 4}.
    Fires only when rank-1 conviction >= 40%. 1 × 3 × 2 = 6 effective
    perms — same as a 3-horse box, but the structural constraint is
    'banker wins'."""
    win_sorted = _by(runners, "model_rank")
    if len(win_sorted) < 4 or not (MIN_FIELD_SIZE <= len(win_sorted) <= MAX_FIELD_SIZE):
        return []
    w1 = (win_sorted[0].get("win_probability") or 0) * 100
    if w1 < 40.0:
        return []
    banker = win_sorted[0]
    box = win_sorted[1:4]
    return [{
        "strategy_label": "banker_win",
        "bet_type": "trifecta",
        "banker_tabs": {1: banker["tab_number"]},
        "box_horses": [h["tab_number"] for h in box],
        "box_horse_names": [banker["horse_name"]] + [h["horse_name"] for h in box],
        "num_permutations": 6,  # 1 × 3 × 2
        "stake_dollars": stake,
    }]


def strategy_banker_win_wide(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Rank-1 must win. 2nd/3rd box from {rank 2..6} (5 horses). Wider
    backup pool reduces the conditional-miss risk; perms = 1 × 5 × 4 = 20.
    Fires when rank-1 >= 35%."""
    win_sorted = _by(runners, "model_rank")
    if len(win_sorted) < 6 or not (MIN_FIELD_SIZE <= len(win_sorted) <= MAX_FIELD_SIZE):
        return []
    w1 = (win_sorted[0].get("win_probability") or 0) * 100
    if w1 < 35.0:
        return []
    banker = win_sorted[0]
    box = win_sorted[1:6]
    return [{
        "strategy_label": "banker_win_wide",
        "bet_type": "trifecta",
        "banker_tabs": {1: banker["tab_number"]},
        "box_horses": [h["tab_number"] for h in box],
        "box_horse_names": [banker["horse_name"]] + [h["horse_name"] for h in box],
        "num_permutations": 20,
        "stake_dollars": stake,
    }]


def strategy_banker_place(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Rank-1 must finish IN top 3 (any position), with other 2 spots
    from {rank 2..5}. Looser banker constraint than 'must win'. 4-horse
    box centred on rank-1, modelled as a regular 4-horse trifecta box
    here since rank-1 always being in the box makes the 'banker_place'
    notion identical to box inclusion. Perms 24."""
    win_sorted = _by(runners, "model_rank")
    if len(win_sorted) < 5 or not (MIN_FIELD_SIZE <= len(win_sorted) <= MAX_FIELD_SIZE):
        return []
    w1 = (win_sorted[0].get("win_probability") or 0) * 100
    if w1 < 30.0:
        return []
    top4 = win_sorted[:4]
    return [{
        "strategy_label": "banker_place",
        "bet_type": "trifecta",
        "box_horses": [h["tab_number"] for h in top4],
        "box_horse_names": [h["horse_name"] for h in top4],
        "num_permutations": 24,
        "stake_dollars": stake,
    }]


def strategy_dual_banker(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """Both rank-1 AND rank-2 must finish in top 3 (any positions).
    Third position from {rank 3, 4, 5}. Effectively a 5-horse trifecta
    box that ONLY counts as a hit when both top model picks are in the
    placings. Modelled as standout requires a non-standard hit check;
    represented here as 5-horse box gated by both-anchored result —
    the backtest framework will need a custom check for this."""
    win_sorted = _by(runners, "model_rank")
    if len(win_sorted) < 5 or not (MIN_FIELD_SIZE <= len(win_sorted) <= MAX_FIELD_SIZE):
        return []
    w1 = (win_sorted[0].get("win_probability") or 0) * 100
    w2 = (win_sorted[1].get("win_probability") or 0) * 100
    if w1 < 30.0 or w2 < 15.0:
        return []
    top5 = win_sorted[:5]
    return [{
        "strategy_label": "dual_banker",
        "bet_type": "trifecta",
        "required_in_top3": [top5[0]["tab_number"], top5[1]["tab_number"]],
        "box_horses": [h["tab_number"] for h in top5],
        "box_horse_names": [h["horse_name"] for h in top5],
        "num_permutations": 60,  # P(5,3)
        "stake_dollars": stake,
    }]


def strategy_first4_standout(runners: list[dict], stake: float = DEFAULT_STAKE) -> list[dict]:
    """First four with rank-1 banker for 1st. 2nd/3rd/4th from {rank 2,
    3, 4, 5} box. Fires on rank-1 >= 40%. 1 × 4 × 3 × 2 = 24 perms."""
    win_sorted = _by(runners, "model_rank")
    if len(win_sorted) < 5 or not (MIN_FIELD_SIZE <= len(win_sorted) <= MAX_FIELD_SIZE):
        return []
    w1 = (win_sorted[0].get("win_probability") or 0) * 100
    if w1 < 40.0:
        return []
    banker = win_sorted[0]
    box = win_sorted[1:5]
    return [{
        "strategy_label": "first4_standout",
        "bet_type": "first_four",
        "banker_tabs": {1: banker["tab_number"]},
        "box_horses": [h["tab_number"] for h in box],
        "box_horse_names": [banker["horse_name"]] + [h["horse_name"] for h in box],
        "num_permutations": 24,
        "stake_dollars": stake,
    }]


def compute_payout(stake: float, num_permutations: int, dividend: float, is_hit: bool) -> tuple[float, float]:
    """Return (payout, pnl). Flexi-bet payout = (stake / perms) * dividend."""
    if not is_hit or num_permutations <= 0 or not dividend:
        return 0.0, -stake
    payout = round((stake / num_permutations) * dividend, 2)
    return payout, round(payout - stake, 2)


# Registry — must be at end of file so all strategy functions above are
# bound by the time this dict is built (Python evaluates module body
# top-to-bottom; referencing a function before its def raises NameError).
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
    # Premium-precision contenders (target: >40% hit rate)
    "surgeon": strategy_surgeon,
    "first4_place": strategy_first4_place,
    "hybrid": strategy_hybrid,
    "tight": strategy_tight,
    "premium_first4": strategy_premium_first4,
    # Standout-style bets (position-anchored bankers)
    "banker_win":       strategy_banker_win,
    "banker_win_wide":  strategy_banker_win_wide,
    "banker_place":     strategy_banker_place,
    "dual_banker":      strategy_dual_banker,
    "first4_standout":  strategy_first4_standout,
}
