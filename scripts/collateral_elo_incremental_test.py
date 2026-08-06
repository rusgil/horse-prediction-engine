"""
Collateral-Elo incremental-value test (the go/no-go gate).

Question: does a collateral-form Elo strength add predictive lift ON TOP OF
the model's existing win probability? The model's win_probability already
summarises all 41 features, so if Elo adds nothing beyond it, it's not worth
a retrain. If it does, that's the green light for refine -> backtest -> release.

Method (no retrain, no production changes):
  1. Build the same time-correct Elo as the prototype; take each runner's
     within-race Elo z-score (strength vs today's field).
  2. Join to the model's own live win_probability (RunnerPredictionHistoryRow,
     source="live") and the actual outcome.
  3. Time-split (earlier races train, later races holdout — no look-ahead).
  4. Fit two logistic models on train via IRLS:
        base: y ~ logit(model_prob)
        aug : y ~ logit(model_prob) + elo_z
  5. On holdout compare: top-1 win rate (rank by score), log-loss, and the
     elo_z coefficient + approx significance. Split metro/country + favourite.

READ-ONLY. Run where the DB is reachable, e.g.:
  railway run bash -c 'DATABASE_URL="$DATABASE_PUBLIC_URL" python3 -m scripts.collateral_elo_incremental_test'
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select  # noqa: E402
from horse_engine.models.database import SessionLocal, HistoricalResultRow, RunnerPredictionHistoryRow  # noqa: E402
from scripts.collateral_form_prototype import load_rows, run_elo, is_metro_venue  # noqa: E402


def _norm(name: str) -> str:
    return (name or "").strip().lower()


async def load_model_probs(since: str | None):
    """(race_id, norm_horse) -> live model win_probability."""
    async with SessionLocal() as s:
        res = await s.execute(
            select(
                RunnerPredictionHistoryRow.race_id,
                RunnerPredictionHistoryRow.horse_name,
                RunnerPredictionHistoryRow.win_probability,
            )
            .where(RunnerPredictionHistoryRow.source == "live")
            .where(RunnerPredictionHistoryRow.model_rank.isnot(None))
            .where(RunnerPredictionHistoryRow.cancelled.is_(False) | RunnerPredictionHistoryRow.cancelled.is_(None))
        )
        rows = res.all()
    out = {}
    for rid, horse, wp in rows:
        if wp is None:
            continue
        if since and (rid.split("_")[0] < since):
            continue
        out[(rid, _norm(horse))] = float(wp)
    return out


# ── pure-stdlib logistic regression (IRLS / Newton) ──────────────────────────
def _solve(A, b):
    """Gaussian elimination with partial pivoting for a small KxK system."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            M[c][c] = 1e-12
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_logistic(X, y, iters=30, l2=1e-3):
    n, K = len(X), len(X[0])
    beta = [0.0] * K
    XtWX = None
    for _ in range(iters):
        XtWX = [[0.0] * K for _ in range(K)]
        grad = [0.0] * K
        for xi, yi in zip(X, y):
            eta = sum(beta[k] * xi[k] for k in range(K))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))
            w = max(p * (1.0 - p), 1e-6)
            r = yi - p
            for a in range(K):
                grad[a] += xi[a] * r
                xa_w = xi[a] * w
                for b_ in range(K):
                    XtWX[a][b_] += xa_w * xi[b_]
        for a in range(1, K):        # L2 on non-intercept terms
            XtWX[a][a] += l2
        delta = _solve(XtWX, grad)
        for k in range(K):
            beta[k] += delta[k]
        if max(abs(d) for d in delta) < 1e-7:
            break
    # approx standard errors from inverse Hessian diagonal
    se = [None] * K
    try:
        inv = [[1.0 if i == j else 0.0 for j in range(K)] for i in range(K)]
        cols = [_solve(XtWX, [inv[r][c] for r in range(K)]) for c in range(K)]
        se = [math.sqrt(abs(cols[c][c])) for c in range(K)]
    except Exception:
        pass
    return beta, se


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def logloss(rows, beta):
    tot = 0.0
    for xi, yi in rows:
        p = min(max(sigmoid(sum(beta[k] * xi[k] for k in range(len(beta)))), 1e-12), 1 - 1e-12)
        tot += -(yi * math.log(p) + (1 - yi) * math.log(1 - p))
    return tot / len(rows)


def top1_rate(by_race, beta):
    wins = races = 0
    for rid, runners in by_race.items():
        if len(runners) < 2:
            continue
        races += 1
        best = max(runners, key=lambda t: sum(beta[k] * t["x"][k] for k in range(len(beta))))
        wins += 1 if best["y"] == 1 else 0
    return wins, races


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    args = ap.parse_args()

    print("Loading results + model probs…", flush=True)
    rows = await load_rows(args.since)
    probmap = await load_model_probs(args.since)
    print(f"  {len(rows):,} result rows | {len(probmap):,} live model probs", flush=True)

    print("Building time-correct Elo…", flush=True)
    records = run_elo(rows, min_games=0)

    # within-race Elo z-score, joined to model prob + outcome
    by_race_elo = defaultdict(list)
    for r in records:
        by_race_elo[r["race_id"]].append(r)

    samples = []  # dict: race_id, date, metro, logit_p, elo_z, y, model_prob, model_fav
    for rid, runs in by_race_elo.items():
        joined = [(r, probmap.get((rid, r["horse"]))) for r in runs]
        joined = [(r, p) for (r, p) in joined if p is not None]
        if len(joined) < 2:
            continue
        elos = [r["pre_elo"] for (r, _) in joined]
        m = sum(elos) / len(elos)
        var = sum((e - m) ** 2 for e in elos) / len(elos)
        sd = math.sqrt(var) or 1.0
        pmax = max(p for (_, p) in joined)
        for (r, p) in joined:
            pc = min(max(p, 1e-4), 1 - 1e-4)
            samples.append({
                "race_id": rid, "date": r["date"], "metro": r["metro"],
                "logit_p": math.log(pc / (1 - pc)),
                "elo_z": (r["pre_elo"] - m) / sd,
                "y": 1 if r["winner"] else 0,
                "model_fav": (p == pmax),
            })
    print(f"  {len(samples):,} joined runner samples across "
          f"{len({s['race_id'] for s in samples}):,} races", flush=True)

    # time split
    dates = sorted({s["date"] for s in samples})
    cut = dates[int(len(dates) * (1 - args.holdout_frac))]
    train = [s for s in samples if s["date"] < cut]
    holdout = [s for s in samples if s["date"] >= cut]
    print(f"  train {len(train):,} / holdout {len(holdout):,} (split at {cut})", flush=True)

    # standardize features on train
    def stats(key):
        v = [s[key] for s in train]
        mu = sum(v) / len(v)
        sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
        return mu, sd
    mu_lp, sd_lp = stats("logit_p")
    mu_ez, sd_ez = stats("elo_z")

    def feats(s, aug):
        lp = (s["logit_p"] - mu_lp) / sd_lp
        ez = (s["elo_z"] - mu_ez) / sd_ez
        return [1.0, lp, ez] if aug else [1.0, lp]

    Xtr_b = [feats(s, False) for s in train]; ytr = [s["y"] for s in train]
    Xtr_a = [feats(s, True) for s in train]
    beta_b, _ = fit_logistic(Xtr_b, ytr)
    beta_a, se_a = fit_logistic(Xtr_a, ytr)

    ll_b = logloss([(feats(s, False), s["y"]) for s in holdout], beta_b)
    ll_a = logloss([(feats(s, True), s["y"]) for s in holdout], beta_a)

    def build_by_race(subset, aug):
        d = defaultdict(list)
        for s in subset:
            d[s["race_id"]].append({"x": feats(s, aug), "y": s["y"], "metro": s["metro"], "fav": s["model_fav"]})
        return d

    hr_b = build_by_race(holdout, False)
    hr_a = build_by_race(holdout, True)
    wb, nb = top1_rate(hr_b, beta_b)
    wa, na = top1_rate(hr_a, beta_a)

    def seg(subset, metro):
        sub = [s for s in subset if s["metro"] == metro]
        b = build_by_race(sub, False); a = build_by_race(sub, True)
        return top1_rate(b, beta_b), top1_rate(a, beta_a)

    (mwb, mnb), (mwa, mna) = seg(holdout, True)
    (cwb, cnb), (cwa, cna) = seg(holdout, False)

    coef_ez = beta_a[2]; se_ez = se_a[2] if se_a[2] else float("nan")
    z_ez = coef_ez / se_ez if se_ez and not math.isnan(se_ez) else float("nan")

    def pct(w, n):
        return f"{100*w/n:5.2f}% (n={n})" if n else "  — (n=0)"

    print("\n" + "=" * 66)
    print("  COLLATERAL-ELO INCREMENTAL-VALUE TEST (on top of the model)")
    print("=" * 66)
    print(f"  Holdout: {na:,} races, {len(holdout):,} runners (from {cut})")
    print("\n  TOP-1 WIN RATE (rank each race by the model's score)")
    print(f"    base  (model prob only)      {pct(wb, nb)}")
    print(f"    aug   (+ collateral_elo)     {pct(wa, na)}")
    print(f"    lift                          {100*(wa/na - wb/nb):+.2f} pp" if nb and na else "")
    print("      metro   base %s | aug %s" % (pct(mwb, mnb), pct(mwa, mna)))
    print("      country base %s | aug %s" % (pct(cwb, cnb), pct(cwa, cna)))
    print("\n  HOLDOUT LOG-LOSS (lower = better)")
    print(f"    base {ll_b:.5f}   aug {ll_a:.5f}   Δ {ll_a - ll_b:+.5f}")
    print("\n  collateral_elo COEFFICIENT (standardized)")
    print(f"    coef {coef_ez:+.4f}   se {se_ez:.4f}   z {z_ez:+.2f}"
          + ("   (>|2| ~ significant)" if not math.isnan(z_ez) else ""))
    print("=" * 66 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
