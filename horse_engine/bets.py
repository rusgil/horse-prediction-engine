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

    return bets


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
