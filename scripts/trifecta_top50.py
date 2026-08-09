"""
Top-50 straight-trifecta spread — OFFLINE backtest (strategy only, nothing live).

Gate a race on:  Sharp race  AND  clear trifecta (rank3_win% - rank4_win% > 5pts)
                 AND our top pick's win odds > $3.
For each gate race, rank EVERY trifecta ordering by the Harville joint probability
and take the top 50. Stake $1 on each ($50 book). If the actual 1-2-3 finishing
order is one of the 50, collect the real tote TRIFECTA dividend (per $1).

Harville trifecta joint (== Plackett-Luce with win-probs as strengths):
    P(a 1st, b 2nd, c 3rd) = p_a * p_b/(1-p_a) * p_c/(1-p_a-p_b)

Run: railway run bash -c 'DATABASE_URL="$DATABASE_PUBLIC_URL" PYTHONPATH=. python3 scripts/trifecta_top50.py'
"""
import asyncio
import re
from itertools import permutations
from collections import defaultdict

from sqlalchemy import text
from horse_engine.models.database import SessionLocal

GAP = 0.05        # rank3 - rank4 win-prob gap (5 points)
ODDS_MIN = 3.0    # our top pick's win odds must exceed this
N_COMBOS = 50     # top-N trifecta orderings to back
POOL = 8          # consider orderings among the top-8 model runners

_COUNTRY = re.compile(r"\s*\((?:[A-Z]{2,3})\)\s*$")
_APOS = {ord(c): None for c in "'’‘`"}


def norm(name):
    s = _COUNTRY.sub("", name or "").translate(_APOS).lower().strip()
    return re.sub(r"\s+", " ", s)


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


def top_trifectas(ranked, wpn, n=N_COMBOS, pool=POOL):
    cand = ranked[:pool]
    combos = [(c, h_top3(wpn[c[0]], wpn[c[1]], wpn[c[2]])) for c in permutations(cand, 3)]
    combos.sort(key=lambda x: -x[1])
    return combos[:n]


async def main():
    async with SessionLocal() as s:
        div = {r[0]: r[1] for r in (await s.execute(text(
            "SELECT race_id, trifecta FROM race_exotic_dividends WHERE trifecta IS NOT NULL"))).fetchall()}
        ids = list(div.keys())

        preds = defaultdict(dict)       # rid -> {name: winprob}
        ranked = defaultdict(list)      # rid -> [name by model_rank]
        tab = defaultdict(dict)         # rid -> {name: tab_number}
        meta = defaultdict(dict)        # rid -> {is_sharp, odds1, venue, rno}
        pr = (await s.execute(text(
            "SELECT race_id, horse_name, win_probability, model_rank, tab_number, "
            "       is_sharp, best_available_odds, venue, race_number "
            "FROM runner_prediction_history "
            "WHERE race_id = ANY(:ids) AND (source='live' OR source IS NULL) "
            "  AND (cancelled IS FALSE OR cancelled IS NULL) "
            "ORDER BY race_id, model_rank NULLS LAST"), {"ids": ids})).fetchall()
        seen = set()
        for rid, hn, wp, rank, tn, sharp, odds, ven, rno in pr:
            n = norm(hn)
            if wp is None or (rid, n) in seen:
                continue
            seen.add((rid, n))
            preds[rid][n] = float(wp)
            ranked[rid].append(n)
            tab[rid][n] = tn
            if rank == 1:
                meta[rid] = {"sharp": bool(sharp), "odds1": odds, "venue": ven or "", "rno": rno}

        rows = (await s.execute(text(
            "SELECT race_id, horse_name, position FROM historical_results "
            "WHERE race_id = ANY(:ids) AND position IS NOT NULL"), {"ids": ids})).fetchall()
        order = defaultdict(dict)
        for rid, hn, pos in rows:
            order[rid][int(pos)] = norm(hn)

    # build all scoreable races (model + settled top-3 + dividend), tag gate fields
    allr = []
    for rid in ids:
        wp, rk, m, o = preds.get(rid, {}), ranked.get(rid, []), meta.get(rid, {}), order.get(rid, {})
        if len(rk) < 4 or not m:
            continue
        tot = sum(wp.values())
        if tot <= 0:
            continue
        wpn = {k: v / tot for k, v in wp.items()}
        actual = tuple(o.get(i) for i in (1, 2, 3))
        if not all(actual):
            continue  # need a settled top-3 to score
        allr.append({
            "rid": rid, "wpn": wpn, "ranked": rk, "tab": tab[rid], "meta": m,
            "actual": actual, "div": div[rid],
            "gap": wpn[rk[2]] - wpn[rk[3]], "odds1": m.get("odds1"), "sharp": bool(m.get("sharp")),
        })

    def gate_tight(r):
        return r["sharp"] and r["gap"] > GAP and (r["odds1"] and r["odds1"] > ODDS_MIN)

    qual = [r for r in allr if gate_tight(r)]
    print(f"trifecta-dividend races: {len(ids)}   |   scoreable: {len(allr)}   |   "
          f"qualifying (sharp + clear-tri + odds>${ODDS_MIN}): {len(qual)}")
    if not qual:
        print("no qualifying races.")
        return

    def lbl(r, name):
        t = r["tab"].get(name)
        return str(t) if t else name

    # ── EXAMPLE: most-recent qualifying race, full 50-combo list ────────────
    ex = sorted(qual, key=lambda r: r["rid"])[-1]
    combos = top_trifectas(ex["ranked"], ex["wpn"])
    covered = {c for c, _ in combos}
    hit = ex["actual"] in covered
    m = ex["meta"]
    print("\n" + "=" * 78)
    print(f"EXAMPLE RACE  {ex['rid']}   {m['venue']} R{m['rno']}   field {len(ex['ranked'])}")
    print(f"  Sharp: yes   top-pick odds: ${ex['odds1']:.2f}   rank3-rank4 gap: {ex['gap']*100:.1f}pts")
    print(f"  actual 1-2-3 (tab): {'-'.join(lbl(ex, n) for n in ex['actual'])}"
          f"   real trifecta dividend: ${ex['div']:.2f}")
    print("=" * 78)
    print(f"  {'#':>3} {'combo':>10} {'P(model)':>9} {'cumP':>7}")
    cum = 0.0
    for i, (c, p) in enumerate(combos, 1):
        cum += p
        star = "  <== WINNER" if c == ex["actual"] else ""
        print(f"  {i:>3} {'-'.join(lbl(ex, n) for n in c):>10} {p*100:8.2f}% {cum*100:6.1f}%{star}")
    print(f"\n  staked ${len(combos)}   |   {'HIT +$%.0f  => race P&L $%+.0f' % (ex['div'], ex['div']-len(combos)) if hit else 'MISS  => race P&L $%+.0f' % (-len(combos))}")

    # ── BACKTEST across all qualifying races ────────────────────────────────
    n = len(qual)
    staked = ret = hits = biggest = 0.0
    per_combo_cost = 0
    hit_divs = []
    for r in qual:
        combos = top_trifectas(r["ranked"], r["wpn"])
        cset = {c for c, _ in combos}
        staked += len(combos)
        per_combo_cost += len(combos)
        if r["actual"] in cset:
            ret += r["div"]
            hits += 1
            biggest = max(biggest, r["div"])
            hit_divs.append(r["div"])
    net = ret - staked
    roi = net / staked * 100 if staked else 0
    net_exmax = net + 0  # for exmax remove biggest
    print("\n" + "=" * 78)
    print(f"BACKTEST — top-{N_COMBOS} straight trifectas, $1 each, on all {n} qualifying races")
    print("=" * 78)
    print(f"  races:            {n}")
    print(f"  avg combos/race:  {per_combo_cost / n:.0f}   (staked ${staked:.0f} total)")
    print(f"  hits:             {hits}  ({hits/n*100:.1f}% of races)")
    print(f"  total returned:   ${ret:.0f}")
    print(f"  NET P&L:          ${net:+.0f}")
    print(f"  ROI:              {roi:+.1f}%")
    print(f"  biggest hit:      ${biggest:.0f}")
    if hits:
        print(f"  hit dividends:    {', '.join('$%.0f' % d for d in sorted(hit_divs, reverse=True))}")
        print(f"  NET ex-biggest:   ${net - biggest:+.0f}  (ROI {(net-biggest)/staked*100:+.1f}%)")
    print("\n  (straight trifectas — exact 1-2-3 order. Dividend is per $1. Offline; nothing live.)")

    # ── SENSITIVITY: same top-50 strategy under relaxed gates (bigger samples) ─
    def bt(races):
        st = rt = big = 0.0
        h = 0
        for r in races:
            combos = top_trifectas(r["ranked"], r["wpn"])
            st += len(combos)
            if r["actual"] in {c for c, _ in combos}:
                rt += r["div"]; h += 1; big = max(big, r["div"])
        net = rt - st
        return len(races), h, st, rt, net, (net / st * 100 if st else 0), \
            ((net - big) / st * 100 if st else 0), big

    gates = [
        ("all scoreable",                 lambda r: True),
        ("clear-tri (gap>5pts)",          lambda r: r["gap"] > GAP),
        ("clear-tri + odds>$3",           lambda r: r["gap"] > GAP and r["odds1"] and r["odds1"] > ODDS_MIN),
        ("clear-tri + sharp",             lambda r: r["gap"] > GAP and r["sharp"]),
        ("sharp only",                    lambda r: r["sharp"]),
        ("odds>$3 only",                  lambda r: r["odds1"] and r["odds1"] > ODDS_MIN),
        ("TIGHT (sharp+tri+odds>$3)",     gate_tight),
    ]
    print("\n" + "=" * 92)
    print(f"SENSITIVITY — top-{N_COMBOS} straight trifectas, $1 each, under relaxed gates")
    print("=" * 92)
    print(f"  {'gate':30} {'races':>6} {'hits':>5} {'hit%':>6} {'staked':>8} {'return':>8} {'net':>8} {'ROI':>8} {'ROI-exMax':>9}")
    for name, g in gates:
        sub = [r for r in allr if g(r)]
        if not sub:
            continue
        nr, h, st, rt, net, roi, roi_ex, big = bt(sub)
        flag = "  <== +EV" if roi > 0 and roi_ex > 0 else ("  (outlier)" if roi > 0 else "")
        print(f"  {name:30} {nr:6d} {h:5d} {h/nr*100:5.1f}% {st:8.0f} {rt:8.0f} {net:+8.0f} "
              f"{roi:+7.1f}% {roi_ex:+8.1f}%{flag}")
    print("\n  ROI-exMax = ROI with the single biggest dividend removed (variance check).")
    print("  Small n => not significant; watch whether ROI survives ex-max and holds as n grows.")

    # ── "HIT BIG" reality check: are the big dividends even reachable? ───────
    def buckets(divs):
        b = [0, 0, 0, 0, 0]  # <50, 50-100, 100-500, 500-1k, >=1k
        for d in divs:
            b[0 if d < 50 else 1 if d < 100 else 2 if d < 500 else 3 if d < 1000 else 4] += 1
        return b

    hitd, missd = [], []
    for r in allr:
        cs = {c for c, _ in top_trifectas(r["ranked"], r["wpn"])}
        (hitd if r["actual"] in cs else missd).append(r["div"])
    print("\n" + "=" * 78)
    print("DOES IT HIT BIG?  trifecta dividend distribution, top-50 on all scoreable races")
    print("=" * 78)
    print(f"  {'bucket':>10} {'HITS(covered)':>14} {'MISSES':>9}")
    names = ["<$50", "$50-100", "$100-500", "$500-1k", ">=$1k"]
    hb, mb = buckets(hitd), buckets(missd)
    for i, nm in enumerate(names):
        print(f"  {nm:>10} {hb[i]:>14} {mb[i]:>9}")
    alld = sorted([r["div"] for r in allr], reverse=True)
    covered_big = [r["div"] for r in allr
                   if r["actual"] in {c for c, _ in top_trifectas(r["ranked"], r["wpn"])} and r["div"] >= 500]
    print(f"\n  biggest trifecta dividend in dataset: ${alld[0]:.0f}   (top-5: {', '.join('$%.0f'%d for d in alld[:5])})")
    print(f"  our top-50 HITS >= $500: {len(covered_big)}   (biggest we actually covered: "
          f"${max(covered_big) if covered_big else 0:.0f})")
    print(f"  avg dividend WHEN WE HIT: ${sum(hitd)/len(hitd):.0f}   vs WHEN WE MISS: ${sum(missd)/len(missd):.0f}")

    # ── "play for the big one": anchor top-pick to WIN, spread the placings ──
    def anchor_wide(r, k):
        a = r["ranked"][0]
        return [(a, x, y) for x, y in permutations(r["ranked"][1:k], 2)]

    print("\n" + "=" * 92)
    print("PLAY-FOR-BIG — anchor our top pick to WIN, box ranks 2..k for 2nd/3rd (gate: odds>$3)")
    print("=" * 92)
    print(f"  {'structure':22} {'races':>6} {'hits':>5} {'hit%':>6} {'combos':>7} {'staked':>8} "
          f"{'return':>8} {'ROI':>7} {'ROI-exMax':>9} {'biggest':>8}")
    val = [r for r in allr if r["odds1"] and r["odds1"] > ODDS_MIN]
    for k in (6, 8, 10, 14):
        st = rt = big = 0.0
        h = 0
        combos_total = 0
        for r in val:
            combos = anchor_wide(r, k)
            if not combos:
                continue
            combos_total += len(combos)
            st += len(combos)
            if r["actual"] in set(combos):
                rt += r["div"]; h += 1; big = max(big, r["div"])
        if not st:
            continue
        net = rt - st
        roi = net / st * 100
        roi_ex = (net - big) / st * 100
        cr = combos_total / len(val)
        flag = "  <== +EV" if roi > 0 and roi_ex > 0 else ("  (outlier)" if roi > 0 else "")
        print(f"  {'anchor1+box2-'+str(k):22} {len(val):6d} {h:5d} {h/len(val)*100:5.1f}% {cr:7.0f} "
              f"{st:8.0f} {rt:8.0f} {roi:+6.1f}% {roi_ex:+8.1f}% {big:8.0f}{flag}")
    print("\n  (anchor structure only wins when OUR PICK wins; then a longshot 2nd/3rd = big dividend caught.)")


asyncio.run(main())
