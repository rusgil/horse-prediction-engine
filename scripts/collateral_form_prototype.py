"""
Collateral-form strength prototype (Phase 0).

Builds a TIME-CORRECT horse-strength rating from result history and backtests
whether it predicts winners — especially in thin-market (country) races where
our win pick is currently a coin-flip.

WHY ELO (not as-of PageRank) FOR PHASE 0
-----------------------------------------
Elo processed in strict chronological order is time-correct *by construction*:
a horse's rating GOING INTO a race depends only on races that already happened.
It also captures the two things a flat rate table can't:
  • collateral form  — beating strong horses lifts you more than beating weak ones,
                       and strength chains transitively through the update sequence.
  • recency          — a layoff regresses your rating back toward the mean.
No graph database required; this reads straight from Postgres via the app's models.

WHAT IT MEASURES
----------------
For every historical race, the pre-race Elo of each runner (leakage-free), then:
  • Elo-favourite win rate      vs  market-favourite win rate  vs  random baseline
  • the same, split metro / country  (the thin-market hypothesis)
  • "disagreement" races where Elo's pick != the market's pick — does Elo find
    winners the market misses?
  • marginal signal: within each market rank, do Elo-strong runners outperform
    Elo-weak ones? (i.e. does Elo add anything *beyond* the price?)

RUN
---
From the repo root, with the same env as the app (DATABASE_URL set), e.g.:
    python -m scripts.collateral_form_prototype
    railway run python -m scripts.collateral_form_prototype
Optional: --since 2026-01-01  --min-games 3  --csv /tmp/collateral.csv

This is a READ-ONLY analysis. It writes nothing to the database.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sys
from collections import defaultdict

# Allow both `python -m scripts.collateral_form_prototype` and direct execution.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select  # noqa: E402

from horse_engine.models.database import SessionLocal, HistoricalResultRow  # noqa: E402

try:
    from horse_engine.bets import is_metro_venue  # noqa: E402
except Exception:  # pragma: no cover - defensive
    def is_metro_venue(_v: str) -> bool:  # type: ignore
        return False

# ── Elo parameters ──────────────────────────────────────────────────────────
BASE = 1500.0
K_NEW, K_MID, K_SETTLED = 40.0, 30.0, 24.0          # K decays as a horse races more
LAYOFF_HALFLIFE_DAYS = 365.0                          # deviation from BASE halves per year off
MIN_FINISHERS = 4                                     # ignore tiny/dead fields


def parse_rid(rid: str):
    """race_id = '{YYYY-MM-DD}_{venue}_R{n}'  ->  (date, venue, race_number)."""
    parts = (rid or "").split("_")
    if len(parts) < 2:
        return None, None, None
    date = parts[0]
    if parts[-1].startswith("R"):
        try:
            rnum = int(parts[-1][1:])
        except ValueError:
            rnum = None
        venue = "_".join(parts[1:-1])
    else:
        rnum = None
        venue = "_".join(parts[1:])
    return date, venue, rnum


def days_between(d1: str, d2: str) -> float:
    """Crude day delta from two ISO date strings; robust to bad input."""
    try:
        import datetime as _dt
        a = _dt.date.fromisoformat(d1)
        b = _dt.date.fromisoformat(d2)
        return abs((b - a).days)
    except Exception:
        return 0.0


def k_factor(games: int) -> float:
    if games < 5:
        return K_NEW
    if games < 15:
        return K_MID
    return K_SETTLED


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%  (n={d})" if d else "   —   (n=0)"


async def load_rows(since: str | None):
    cols = (
        HistoricalResultRow.race_id,
        HistoricalResultRow.horse_name,
        HistoricalResultRow.position,
        HistoricalResultRow.beaten_margin,
        HistoricalResultRow.winner,
        HistoricalResultRow.starting_price,
        HistoricalResultRow.venue,
        HistoricalResultRow.field_size,
    )
    async with SessionLocal() as s:
        res = await s.execute(select(*cols))
        rows = res.all()
    out = []
    for race_id, horse, pos, margin, winner, sp, venue_col, fsize in rows:
        date, venue_rid, rnum = parse_rid(race_id)
        if not date:
            continue
        if since and date < since:
            continue
        venue = (venue_col or venue_rid or "").lower()
        out.append({
            "race_id": race_id, "date": date, "venue": venue, "rnum": rnum or 0,
            "horse": (horse or "").strip().lower(),
            "pos": pos, "margin": margin, "winner": bool(winner),
            "sp": sp, "field_size": fsize,
        })
    return out


def run_elo(rows, min_games: int):
    """Chronological Elo; returns per-runner records with leakage-free pre-race ratings."""
    races = defaultdict(list)
    for r in rows:
        races[r["race_id"]].append(r)
    order = sorted(races.keys(), key=lambda rid: (races[rid][0]["date"], races[rid][0]["venue"], races[rid][0]["rnum"]))

    rating: dict[str, float] = {}
    games: dict[str, int] = defaultdict(int)
    last_date: dict[str, str] = {}
    records = []

    for rid in order:
        field = races[rid]
        finishers = [r for r in field if isinstance(r["pos"], int) and r["pos"] > 0 and r["horse"]]
        if len(finishers) < 2:
            continue

        # 1) SNAPSHOT pre-race ratings (with layoff regression) — before any update.
        pre = {}
        for r in finishers:
            h = r["horse"]
            cur = rating.get(h, BASE)
            if h in last_date:
                off = days_between(last_date[h], r["date"])
                if off > 0:
                    cur = BASE + (cur - BASE) * (0.5 ** (off / LAYOFF_HALFLIFE_DAYS))
            pre[h] = cur

        # elo rank (1 = strongest) and market rank (1 = shortest price)
        elo_sorted = sorted(finishers, key=lambda r: -pre[r["horse"]])
        elo_rank = {r["horse"]: i + 1 for i, r in enumerate(elo_sorted)}
        priced = [r for r in finishers if r["sp"] and r["sp"] > 1.0]
        mkt_sorted = sorted(priced, key=lambda r: r["sp"])
        mkt_rank = {r["horse"]: i + 1 for i, r in enumerate(mkt_sorted)}

        metro = is_metro_venue(field[0]["venue"])
        for r in finishers:
            h = r["horse"]
            records.append({
                "race_id": rid, "date": r["date"], "venue": r["venue"], "metro": metro,
                "n_finishers": len(finishers), "horse": h,
                "pre_elo": round(pre[h], 1), "elo_rank": elo_rank[h],
                "elo_games": games[h], "market_rank": mkt_rank.get(h),
                "pos": r["pos"], "winner": r["winner"], "sp": r["sp"],
            })

        # 2) UPDATE Elo with pairwise results (dead-heats -> 0.5).
        delta = defaultdict(float)
        for i in range(len(finishers)):
            ri = finishers[i]; hi = ri["horse"]; Ri = pre[hi]
            acc = 0.0
            for j in range(len(finishers)):
                if i == j:
                    continue
                rj = finishers[j]; Rj = pre[rj["horse"]]
                if ri["pos"] < rj["pos"]:
                    s = 1.0
                elif ri["pos"] > rj["pos"]:
                    s = 0.0
                else:
                    s = 0.5
                exp = 1.0 / (1.0 + 10 ** ((Rj - Ri) / 400.0))
                acc += (s - exp)
            delta[hi] = k_factor(games[hi]) * acc / (len(finishers) - 1)
        for r in finishers:
            h = r["horse"]
            rating[h] = pre[h] + delta[h]
            games[h] += 1
            last_date[h] = r["date"]

    return records


def analyse(records, min_games: int):
    by_race = defaultdict(list)
    for r in records:
        by_race[r["race_id"]].append(r)

    def top(recs, key):
        pool = [r for r in recs if r.get(key)]
        return min(pool, key=lambda r: r[key]) if pool else None

    # Aggregators
    agg = defaultdict(lambda: [0, 0])  # label -> [wins, races]
    elo_bucket = defaultdict(lambda: [0, 0])
    disagree = {"elo": [0, 0], "mkt": [0, 0]}
    # marginal signal: (market_rank_bucket, elo_strong) -> [wins, runners]
    marginal = defaultdict(lambda: [0, 0])
    rand_sum = 0.0
    rand_n = 0

    for rid, recs in by_race.items():
        if len(recs) < MIN_FINISHERS:
            continue
        seg = "metro" if recs[0]["metro"] else "country"
        rand_sum += 1.0 / len(recs)
        rand_n += 1

        elo_pick = top(recs, "elo_rank")
        seasoned = elo_pick and elo_pick["elo_games"] >= min_games
        mkt_pick = top(recs, "market_rank")

        for label, pick, gate in (
            ("ELO top-pick (all)", elo_pick, True),
            (f"ELO top-pick (>={min_games} prior runs)", elo_pick, seasoned),
            ("MARKET favourite", mkt_pick, True),
        ):
            if pick and gate:
                agg[label][1] += 1
                agg[label][0] += 1 if pick["winner"] else 0
                agg[label + " | " + seg][1] += 1
                agg[label + " | " + seg][0] += 1 if pick["winner"] else 0

        # bucket win rate by elo rank
        for r in recs:
            b = r["elo_rank"] if r["elo_rank"] <= 3 else 4
            elo_bucket[b][1] += 1
            elo_bucket[b][0] += 1 if r["winner"] else 0

        # disagreement races (seasoned elo pick only, to be fair)
        if seasoned and mkt_pick and elo_pick["horse"] != mkt_pick["horse"]:
            disagree["elo"][1] += 1
            disagree["elo"][0] += 1 if elo_pick["winner"] else 0
            disagree["mkt"][1] += 1
            disagree["mkt"][0] += 1 if mkt_pick["winner"] else 0

        # marginal signal
        third = max(1, len(recs) // 3)
        for r in recs:
            if not r["market_rank"]:
                continue
            mb = r["market_rank"] if r["market_rank"] <= 3 else 4
            strong = "elo-strong" if r["elo_rank"] <= third else "elo-weak"
            key = (mb, strong)
            marginal[key][1] += 1
            marginal[key][0] += 1 if r["winner"] else 0

    return agg, elo_bucket, disagree, marginal, (rand_sum / rand_n if rand_n else 0.0), rand_n


def report(agg, elo_bucket, disagree, marginal, rand_base, rand_n, min_games):
    print("\n" + "=" * 68)
    print("  COLLATERAL-FORM (ELO) PROTOTYPE — BACKTEST")
    print("=" * 68)
    print(f"  Scored races: {rand_n:,}   |   random top-pick baseline: {100*rand_base:.1f}%")

    print("\n  TOP-PICK HIT RATE (did the rank-1 selection win?)")
    for label in ["ELO top-pick (all)",
                  f"ELO top-pick (>={min_games} prior runs)",
                  "MARKET favourite"]:
        w, n = agg[label]
        print(f"    {label:<34} {pct(w, n)}")
        for seg in ("metro", "country"):
            w2, n2 = agg[label + " | " + seg]
            print(f"        └ {seg:<8} {pct(w2, n2)}")

    print("\n  WIN RATE BY ELO RANK (every runner)")
    for b in (1, 2, 3, 4):
        w, n = elo_bucket[b]
        lbl = f"elo_rank {b}" if b <= 3 else "elo_rank 4+"
        print(f"    {lbl:<12} {pct(w, n)}")

    print("\n  DISAGREEMENT RACES (Elo pick ≠ market favourite)")
    we, ne = disagree["elo"]; wm, nm = disagree["mkt"]
    print(f"    Elo's pick     {pct(we, ne)}")
    print(f"    Market's pick  {pct(wm, nm)}")
    print("    (Elo > Market here = Elo finds winners the price misses.)")

    print("\n  MARGINAL SIGNAL — win rate within each market rank, Elo-strong vs Elo-weak")
    print("    (if strong > weak inside the same market rank, Elo adds signal beyond price)")
    for mb in (1, 2, 3, 4):
        s = marginal[(mb, "elo-strong")]; w = marginal[(mb, "elo-weak")]
        lbl = f"market_rank {mb}" if mb <= 3 else "market_rank 4+"
        print(f"    {lbl:<15} strong {pct(*s)}   |   weak {pct(*w)}")
    print("=" * 68 + "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ISO date lower bound, e.g. 2026-01-01")
    ap.add_argument("--min-games", type=int, default=3, help="prior runs before trusting an Elo pick")
    ap.add_argument("--csv", default=None, help="optional path to dump per-runner records")
    args = ap.parse_args()

    print("Loading result history…", flush=True)
    rows = await load_rows(args.since)
    print(f"  {len(rows):,} runner-rows loaded"
          + (f" (since {args.since})" if args.since else ""), flush=True)
    if not rows:
        print("No rows — is DATABASE_URL pointed at the results DB?")
        return

    print("Building time-correct Elo…", flush=True)
    records = run_elo(rows, args.min_games)
    print(f"  {len(records):,} scored runner-records", flush=True)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            wtr.writeheader()
            wtr.writerows(records)
        print(f"  wrote {args.csv}", flush=True)

    agg, elo_bucket, disagree, marginal, rand_base, rand_n = analyse(records, args.min_games)
    report(agg, elo_bucket, disagree, marginal, rand_base, rand_n, args.min_games)


if __name__ == "__main__":
    asyncio.run(main())
