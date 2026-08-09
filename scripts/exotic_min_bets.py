"""
Minimum-bets analysis — how many top combos do you actually need? (offline, strategy only)

For every qualifying race we rank EVERY exotic ordering by the Harville joint
probability. When the bet hits, we record the RANK of the winning combo in that
sorted list. That distribution answers "is 50 overkill — would 15/20/30 do?":
  - if winners cluster in the top ~20, betting 50 just wastes stake.
Then we sweep N (number of top combos backed, $1 each) and report ROI at each N.

Trifecta gate: Sharp + clear-tri (rank3-rank4 win% > 5pts) + top pick odds > X.
First-4  gate: Sharp + clear-ff  (rank4-rank5 win% > 5pts) + top pick odds > X.

Run: railway run bash -c 'DATABASE_URL="$DATABASE_PUBLIC_URL" PYTHONPATH=. python3 scripts/exotic_min_bets.py'
"""
import asyncio
import re
from itertools import permutations
from collections import defaultdict

from sqlalchemy import text
from horse_engine.models.database import SessionLocal

GAP = 0.05
POOL = 8
N_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60]

_COUNTRY = re.compile(r"\s*\((?:[A-Z]{2,3})\)\s*$")
_APOS = {ord(c): None for c in "'’‘`"}


def norm(name):
    s = _COUNTRY.sub("", name or "").translate(_APOS).lower().strip()
    return re.sub(r"\s+", " ", s)


def h_top3(p1, p2, p3):
    if min(p1, p2, p3) <= 0:
        return 0.0
    d1 = 1 - p1
    if d1 <= 0:
        return 0.0
    d2 = d1 - p2
    if d2 <= 0:
        return 0.0
    return p1 * (p2 / d1) * (p3 / d2)


def h_top4(p1, p2, p3, p4):
    if min(p1, p2, p3, p4) <= 0:
        return 0.0
    d1 = 1 - p1
    if d1 <= 0:
        return 0.0
    d2 = d1 - p2
    if d2 <= 0:
        return 0.0
    d3 = d2 - p3
    if d3 <= 0:
        return 0.0
    return p1 * (p2 / d1) * (p3 / d2) * (p4 / d3)


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1))))
    return sorted_vals[i]


async def load():
    async with SessionLocal() as s:
        drows = (await s.execute(text(
            "SELECT race_id, trifecta, first_four FROM race_exotic_dividends "
            "WHERE trifecta IS NOT NULL OR first_four IS NOT NULL"))).fetchall()
        div = {r[0]: {"tri": r[1], "ff": r[2]} for r in drows}
        ids = list(div.keys())
        pr = (await s.execute(text(
            "SELECT race_id, horse_name, win_probability, model_rank, is_sharp, best_available_odds "
            "FROM runner_prediction_history WHERE race_id = ANY(:ids) "
            "AND (source='live' OR source IS NULL) AND (cancelled IS FALSE OR cancelled IS NULL) "
            "ORDER BY race_id, model_rank NULLS LAST"), {"ids": ids})).fetchall()
        preds, ranked, meta, seen = defaultdict(dict), defaultdict(list), {}, set()
        for rid, hn, wp, rank, sharp, odds in pr:
            n = norm(hn)
            if wp is None or (rid, n) in seen:
                continue
            seen.add((rid, n))
            preds[rid][n] = float(wp)
            ranked[rid].append(n)
            if rank == 1:
                meta[rid] = {"sharp": bool(sharp), "odds1": odds}
        rr = (await s.execute(text(
            "SELECT race_id, horse_name, position FROM historical_results "
            "WHERE race_id = ANY(:ids) AND position IS NOT NULL"), {"ids": ids})).fetchall()
        order = defaultdict(dict)
        for rid, hn, pos in rr:
            order[rid][int(pos)] = norm(hn)
    return div, ids, preds, ranked, meta, order


def analyse(kind, legs, div_key, gap_lo, gap_hi, odds_min, data):
    div, ids, preds, ranked, meta, order = data
    fn = h_top4 if legs == 4 else h_top3
    ranks = []           # rank of winning combo when it's within the top-(pool perms)
    races = []           # (sorted_combos, actual, dividend) for qualifying races
    for rid in ids:
        wp, rk, m, o = preds.get(rid, {}), ranked.get(rid, []), meta.get(rid, {}), order.get(rid, {})
        if len(rk) <= gap_hi or not m:
            continue
        tot = sum(wp.values())
        if tot <= 0:
            continue
        wpn = {k: v / tot for k, v in wp.items()}
        gap = wpn[rk[gap_lo]] - wpn[rk[gap_hi]]
        odds1 = m.get("odds1")
        actual = tuple(o.get(i) for i in range(1, legs + 1))
        dv = div[rid][div_key]
        if not (m.get("sharp") and gap > GAP and odds1 and odds1 > odds_min and all(actual) and dv):
            continue
        combos = sorted(((c, fn(*[wpn[x] for x in c])) for c in permutations(rk[:POOL], legs)),
                        key=lambda x: -x[1])
        clist = [c for c, _ in combos]
        races.append((clist, actual, dv))
        if actual in clist:
            ranks.append(clist.index(actual) + 1)  # 1-based rank of the winning combo
    return ranks, races


def report(label, ranks, races):
    print("\n" + "=" * 92)
    print(f"{label}   ({len(races)} qualifying races, {len(ranks)} of them hit within the top-{POOL} perms)")
    print("=" * 92)
    if ranks:
        rs = sorted(ranks)
        print(f"  winning-combo RANK when it hits:  min {rs[0]}  median {pct(rs,50)}  "
              f"75th {pct(rs,75)}  90th {pct(rs,90)}  max {rs[-1]}")
        print(f"  all winning ranks: {rs}")
    else:
        print("  no hits in sample.")
    print(f"\n  {'N combos':>9} {'races':>6} {'hits':>5} {'hit%':>6} {'staked':>8} {'return':>8} "
          f"{'ROI':>8} {'ROI-exMax':>9}")
    for N in N_GRID:
        st = rt = big = 0.0
        h = 0
        for clist, actual, dv in races:
            topN = set(clist[:N])
            st += min(N, len(clist))
            if actual in topN:
                rt += dv; h += 1; big = max(big, dv)
        if not st:
            continue
        net = rt - st
        roi = net / st * 100
        roi_ex = (net - big) / st * 100
        flag = "  <== +EV" if roi > 0 and roi_ex > 0 else ("  (outlier)" if roi > 0 else "")
        print(f"  {N:>9} {len(races):6d} {h:5d} {h/len(races)*100:5.1f}% {st:8.0f} {rt:8.0f} "
              f"{roi:+7.1f}% {roi_ex:+8.1f}%{flag}")


async def main():
    data = await load()
    print("Loaded. POOL(top-N runners considered) =", POOL)
    # TRIFECTA — clear-tri = rank3-rank4 gap; legs 3
    for om in (3.0, 2.0):
        ranks, races = analyse("tri", 3, "tri", 2, 3, om, data)
        report(f"TRIFECTA  · Sharp + clear-tri(r3-r4>5pts) + odds>${om:.0f}", ranks, races)
    # FIRST FOUR — clear-ff = rank4-rank5 gap; legs 4
    for om in (3.0, 2.0):
        ranks, races = analyse("ff", 4, "ff", 3, 4, om, data)
        report(f"FIRST FOUR · Sharp + clear-ff(r4-r5>5pts) + odds>${om:.0f}", ranks, races)
    print("\n  RANK distribution tells you the minimum N: if winners never rank beyond, say, 30,")
    print("  then N=30 captures every hit for less stake than N=50. Small n => noisy. Offline only.")


asyncio.run(main())
