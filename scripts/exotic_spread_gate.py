"""
Exotic overlay / spread gate — OFFLINE backtest (strategy only, nothing goes live).

Tests whether the model's per-runner win probabilities can drive a profitable
First-Four / Trifecta *spread* play (the "~$50 book for a big payout" idea) against
the REAL tote dividends captured in race_exotic_dividends.

For every settled race with a dividend we:
  1. read the model's pre-race win-probability vector (runner_prediction_history),
  2. read the actual finishing order (historical_results, positions 1..4),
  3. read the real First-Four / Trifecta dividend (per $1),
  4. cover a set of ordered combos with $1 each (a "structure"),
  5. bank the $1 dividend iff the actual order is one of our combos.

We report, per structure and per pre-race GATE, the realised:
    races bet · staked · returned · ROI · hit-rate · EV/race · biggest single hit
plus the model's Harville coverage probability (its own estimate of hitting).

Harville joint == Plackett-Luce with win-probs as strengths (remove-and-renormalise).

Run:  railway run bash -c 'DATABASE_URL="$DATABASE_PUBLIC_URL" PYTHONPATH=. python3 scripts/exotic_spread_gate.py'
"""
import asyncio
import re
from itertools import permutations
from collections import defaultdict

from sqlalchemy import text
from horse_engine.models.database import SessionLocal

# ── takeout (from horse_engine/bets.py) ─────────────────────────────────────
TAB_TRIFECTA_TAKEOUT = 0.145
TAB_FIRST_FOUR_TAKEOUT = 0.20

# ── name normalisation (mirrors main._normalize_horse) ──────────────────────
_COUNTRY = re.compile(r"\s*\((?:[A-Z]{2,3})\)\s*$")
_APOS = {ord(c): None for c in "'’‘`"}


def norm(name: str) -> str:
    s = _COUNTRY.sub("", name or "").translate(_APOS).lower().strip()
    return re.sub(r"\s+", " ", s)


# ── Harville / Plackett-Luce joint (from bets.py) ───────────────────────────
def h_top3(p1, p2, p3):
    if min(p1, p2, p3) <= 0:
        return 0.0
    d1 = 1.0 - p1
    if d1 <= 0:
        return 0.0
    d2 = d1 - p2
    if d2 <= 0:
        return 0.0
    return p1 * (p2 / d1) * (p3 / d2)


def h_top4(p1, p2, p3, p4):
    if min(p1, p2, p3, p4) <= 0:
        return 0.0
    d1 = 1.0 - p1
    if d1 <= 0:
        return 0.0
    d2 = d1 - p2
    if d2 <= 0:
        return 0.0
    d3 = d2 - p3
    if d3 <= 0:
        return 0.0
    return p1 * (p2 / d1) * (p3 / d2) * (p4 / d3)


# ── structures: ranked names -> set of ordered combos ───────────────────────
# Each structure returns a list of ordered tuples (the exact finishing orders it
# covers). We stake $1 on each. len(combos) == dollar cost.
def ff_box(k):
    return lambda r: list(permutations(r[:k], 4)) if len(r) >= k else []


def ff_anchor1(k):  # rank-1 wins; ranks 2..k fill 2/3/4
    def f(r):
        if len(r) < k:
            return []
        return [(r[0],) + rest for rest in permutations(r[1:k], 3)]
    return f


def ff_banker2(k):  # ranks 1,2 fill top-2 either order; ranks 3..k fill 3/4
    def f(r):
        if len(r) < k:
            return []
        out = []
        for top in permutations(r[:2], 2):
            for tail in permutations(r[2:k], 2):
                out.append(top + tail)
        return out
    return f


def tri_box(k):
    return lambda r: list(permutations(r[:k], 3)) if len(r) >= k else []


def tri_anchor1(k):  # rank-1 wins; ranks 2..k fill 2/3
    def f(r):
        if len(r) < k:
            return []
        return [(r[0],) + rest for rest in permutations(r[1:k], 2)]
    return f


def tri_banker2(k):  # ranks 1,2 top-2 either order; ranks 3..k fill 3rd
    def f(r):
        if len(r) < k:
            return []
        out = []
        for top in permutations(r[:2], 2):
            for third in r[2:k]:
                out.append(top + (third,))
        return out
    return f


FF_STRUCTS = {
    "ff_box4        (24)": ff_box(4),
    "ff_box5       (120)": ff_box(5),
    "ff_anchor1_f5  (24)": ff_anchor1(5),
    "ff_anchor1_f6  (60)": ff_anchor1(6),
    "ff_anchor1_f7 (120)": ff_anchor1(7),
    "ff_banker2_f6  (24)": ff_banker2(6),
    "ff_banker2_f7  (40)": ff_banker2(7),
}
TRI_STRUCTS = {
    "tri_box3        (6)": tri_box(3),
    "tri_box4       (24)": tri_box(4),
    "tri_box5       (60)": tri_box(5),
    "tri_anchor1_f5 (12)": tri_anchor1(5),
    "tri_anchor1_f6 (20)": tri_anchor1(6),
    "tri_banker2_f6  (8)": tri_banker2(6),
}


def cover_prob(combos, wp, legs):
    """Model's Harville probability that the result is one of our combos."""
    fn = h_top4 if legs == 4 else h_top3
    tot = 0.0
    for c in combos:
        ps = [wp.get(n, 0.0) for n in c]
        if all(ps):
            tot += fn(*ps)
    return tot


async def main():
    async with SessionLocal() as s:
        div_rows = (await s.execute(text(
            "SELECT race_id, trifecta, first_four FROM race_exotic_dividends "
            "WHERE trifecta IS NOT NULL OR first_four IS NOT NULL"))).fetchall()
        race_ids = [r[0] for r in div_rows]
        div = {r[0]: {"tri": r[1], "ff": r[2]} for r in div_rows}
        print(f"dividend races: {len(race_ids)}")

        # model win-prob vectors (pre-race snapshot, live source only)
        preds = defaultdict(dict)  # race_id -> {norm_name: win_prob}
        rank_names = defaultdict(list)  # race_id -> [name by model_rank]
        pr = (await s.execute(text(
            "SELECT race_id, horse_name, win_probability, model_rank "
            "FROM runner_prediction_history "
            "WHERE race_id = ANY(:ids) AND (source = 'live' OR source IS NULL) "
            "AND (cancelled IS FALSE OR cancelled IS NULL) "
            "ORDER BY race_id, model_rank NULLS LAST"), {"ids": race_ids})).fetchall()
        seen = set()
        for rid, hn, wp, rank in pr:
            n = norm(hn)
            if (rid, n) in seen or wp is None:
                continue
            seen.add((rid, n))
            preds[rid][n] = float(wp)
            rank_names[rid].append(n)

        # actual finishing order
        res = (await s.execute(text(
            "SELECT race_id, horse_name, position FROM historical_results "
            "WHERE race_id = ANY(:ids) AND position IS NOT NULL"),
            {"ids": race_ids})).fetchall()
        order = defaultdict(dict)  # race_id -> {position: norm_name}
        for rid, hn, pos in res:
            order[rid][int(pos)] = norm(hn)

    # build per-race records
    races = []
    for rid in race_ids:
        wp = preds.get(rid, {})
        ranked = rank_names.get(rid, [])
        o = order.get(rid, {})
        if not wp or not ranked:
            continue
        tot = sum(wp.values())
        if tot <= 0:
            continue
        wpn = {k: v / tot for k, v in wp.items()}  # renormalise to sum 1
        actual_ff = tuple(o.get(i) for i in (1, 2, 3, 4))
        actual_tri = tuple(o.get(i) for i in (1, 2, 3))
        r1 = wpn.get(ranked[0], 0.0)
        venue = rid.split("_")[1] if "_" in rid else ""
        races.append({
            "rid": rid, "wpn": wpn, "ranked": ranked,
            "ff": actual_ff if all(actual_ff) else None,
            "tri": actual_tri if all(actual_tri) else None,
            "div_ff": div[rid]["ff"], "div_tri": div[rid]["tri"],
            "field": len(ranked), "r1": r1, "venue": venue,
        })

    print(f"races with model + results: {len(races)}")

    # ── gates (pre-race computable) ─────────────────────────────────────────
    gates = {
        "all":            lambda r: True,
        "field<=10":      lambda r: r["field"] <= 10,
        "field<=8":       lambda r: r["field"] <= 8,
        "rank1>=30%":     lambda r: r["r1"] >= 0.30,
        "rank1>=40%":     lambda r: r["r1"] >= 0.40,
        "cover>=median":  None,  # filled per-structure below
    }

    def run(structs, legs, div_key, actual_key, takeout):
        print("\n" + "=" * 100)
        print(f"{'FIRST FOUR' if legs == 4 else 'TRIFECTA'}  (takeout {takeout:.0%}, $1/combo)")
        print("=" * 100)
        header = f"{'structure':22} {'gate':14} {'bets':>5} {'staked':>8} {'return':>9} " \
                 f"{'ROI':>8} {'ROI-exMax':>9} {'hit%':>6} {'EV/race':>8} {'cover%':>7} {'biggest':>9}"
        for label, fn in structs.items():
            # precompute per-race coverage + outcome
            recs = []
            for r in races:
                dv = r[div_key]
                actual = r[actual_key]
                if dv is None or actual is None:
                    continue
                combos = fn(r["ranked"])
                if not combos:
                    continue
                cset = set(combos)
                cost = float(len(combos))
                ret = dv if actual in cset else 0.0
                cp = cover_prob(combos, r["wpn"], legs)
                recs.append({"cost": cost, "ret": ret, "cover": cp, "r": r})
            if not recs:
                continue
            cov_sorted = sorted(x["cover"] for x in recs)
            cov_med = cov_sorted[len(cov_sorted) // 2] if cov_sorted else 0.0
            print("-" * 100)
            print(header)
            for gname, gfn in gates.items():
                if gname == "cover>=median":
                    sub = [x for x in recs if x["cover"] >= cov_med]
                else:
                    sub = [x for x in recs if gfn(x["r"])]
                if not sub:
                    continue
                staked = sum(x["cost"] for x in sub)
                returned = sum(x["ret"] for x in sub)
                hits = sum(1 for x in sub if x["ret"] > 0)
                roi = (returned - staked) / staked if staked else 0.0
                ev = (returned - staked) / len(sub)
                hitpct = hits / len(sub) * 100
                avgcov = sum(x["cover"] for x in sub) / len(sub) * 100
                biggest = max((x["ret"] for x in sub), default=0.0)
                # robustness: ROI if the single luckiest payout is removed
                roi_exmax = (returned - biggest - staked) / staked if staked else 0.0
                flag = "  <== +EV" if roi > 0 and roi_exmax > 0 else ("  (outlier)" if roi > 0 else "")
                print(f"{label:22} {gname:14} {len(sub):5d} {staked:8.0f} {returned:9.0f} "
                      f"{roi*100:7.1f}% {roi_exmax*100:8.1f}% {hitpct:5.1f}% {ev:8.2f} {avgcov:6.1f}% {biggest:9.0f}{flag}")

    run(FF_STRUCTS, 4, "div_ff", "ff", TAB_FIRST_FOUR_TAKEOUT)
    run(TRI_STRUCTS, 3, "div_tri", "tri", TAB_TRIFECTA_TAKEOUT)
    print("\n(ROI>0 => the structure/gate returned more than it staked across the sample.")
    print(" 'cover%' = model's own Harville hit-probability estimate. Strategy only — nothing live.)")


asyncio.run(main())
