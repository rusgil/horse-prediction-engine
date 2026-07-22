# Phase 2 Handoff — Findings & Changes (2026-07-22)

Session handoff document. Covers the expert-panel model review, the offline
experiments that followed, every code change shipped, and the open work
queue. Written for continuation by another session/model with zero context.

Companion artifacts:
- Full panel review: https://claude.ai/code/artifact/01aa844b-eaee-4bf6-8538-edfe1335ae0f
- Memory: `phase2_experiments_2026-07-22.md`, `code_review_bugs.md` (BUG-42/43 + FIX-AE)

---

## 1. Headline findings (evidence, not opinion)

### 1.1 The market is excluded from the win model on stale evidence — the #1 accuracy lever
- All market features (`market_rank_norm`, `market_implied_prob`, `odds_movement_norm`)
  are **win-masked** (`features.py:79-85`) per a 2026-06-12 ablation.
- That ablation predates the composite-odds fix `74b605e` (2026-07-16). Before the fix,
  `market_implied_prob` was literally `1/N` for every horse — the ablation tested noise.
- Documented symptom matches: model under-rates market favourites by 10-17pp
  (memory `retrain_snapshot_2026-07-16`).

### 1.2 Offline backtest (180-day export, walk-forward, no test-on-train)
Data: `runner_prediction_history` (source=live, not cancelled/contaminated) joined to
`historical_results`; 2,433 races; date-split 70/30; scripts in session scratchpad
(`experiment_benter.py`, `experiment_live_market.py`).

| Selector (868-race holdout) | Top-1 win % | Top-1 place % | Log loss |
|---|---|---|---|
| Current model (renormalised) | 21.8 ±2.7 | 52.1 ±3.3 | 2.094 |
| SP market alone | 35.0 ±3.2 | 68.8 ±3.1 | 1.744 |
| Benter blend (fitted α≈0.0, β=1.14) | 35.1 | 68.8 | 1.737 |

- On the 591 top-pick disagreements: market's pick won 197, model's won 82.
- **Interpretation discipline:** 35% is the SP-favourite benchmark, not a model
  achievement — SP doesn't exist at prediction time. Live-market version
  (post-fix clean rows only, n=85, leave-one-day-out): model 21.2% / live market
  25.9% / blend 27.1%, fitted **α=0.14, β=0.53** — against the weaker early market
  the model DOES carry real weight. Realistic near-term production: 22% → 26-27%,
  climbing toward low-30s as odds coverage improves.

### 1.3 Value/disagreement analysis — where paying odds actually live
Flat $1 at SP on all runners by overlay bucket (renormalised model prob − SP prob):
every bucket negative; overlay 8-15pp is the worst (-44.6% ROI). Backing the model's
rank-1 when it disagrees with the market: 13.5% win, -16.4% ROI overall, but by SP band:

| Disagreement pick SP | n | Win % | ROI |
|---|---|---|---|
| <$4 | 233 | 27.0 | -10.3% |
| **$4-8** | **688** | **17.7** | **-2.9%** (≈break-even; likely positive at BAO) |
| $8-15 | 433 | 8.1 | -18.9% |
| $15+ | 344 | 2.6 | -44.7% |

→ The value seam is $4-8 disagreements at BAO. $15+ "overlays" are model error.
Caveat: measured on contaminated-market-era model outputs; expect better post-fix.

### 1.4 Probability semantics are broken by the multiplier stack
- Going (×0.40 soft / ×0.55 heavy), midfield ×0.85, thin-record ×0.50-0.85,
  completeness ×0.60-0.80 apply WITHOUT renormalisation (`engine.py`, comment says
  deliberate). On a soft track the field sums to ~0.40.
- Downstream consumers treat these as probabilities: tier gates (≥46/36/30), Sharp
  filter, overlay, Harville exotic math, EV. Off-going races are structurally locked
  out of tiers; exotic EV computed from non-probabilities.
- Recommended fix (NOT yet implemented): per-going temperature on raw scores
  pre-softmax; renormalise after runner-specific multipliers.
- Going-bucket check (n small): soft rank-1 predicted 13.5% vs actual 7.7% (n=26) —
  the ×0.40 may be *insufficient* at rank-1, not excessive; good/firm predicted 19.0%
  vs actual 22.4% (under-confident). Re-check with more data before changing.

### 1.5 Candidate backtests were in-sample (BUG-42 — FIXED)
All four retrain paths (win/place/exotic endpoints + nightly exotic cron) trained
through today, then `_backtest_weight_candidate(holdout_days=7)` scored on the last
7 days — a subset of the training window. Promote/reject was partly memorisation.

### 1.6 History snapshots froze odds-less state (BUG-43 — FIXED)
`_snapshot_prerace_predictions` ran ONCE at 9am AEST. Afternoon races were frozen
with morning (odds-less) enrichment → **71% of clean post-2026-07-16 races had a
flat/missing market** in history (213 of 298). This starved market features, the
blend, and the clean-market retrain accumulation.

### 1.7 `win_prob_raw` is not the raw softmax (docs bug, plumbing was correct)
The snapshot at `engine.py` is taken AFTER venue calibration, exactly where isotonic
applies at inference — fit/apply were always consistent; only comments lied. Comments
fixed; do NOT "fix" the snapshot position without moving isotonic.

### 1.8 `RacePredictionRow` is a dead table
Nothing constructs it anywhere in the codebase — only legacy reads remain
(`main.py:7990`, `main.py:15831` region). Any new feature reading it will silently
get zero rows. Live race-card data rides on `RunnerPredictionRow` (+ enriched_json).

---

## 2. Code shipped (all pushed to main)

| Commit | What |
|---|---|
| `ac46e67` | (pre-review) `win_prob_raw` carried into `RunnerPredictionHistoryRow` via `save_race_predictions` — isotonic sample accumulation fix |
| `7574d06` | Phase-2 core: Benter blend layer; BUG-42 retrain holdout clip (`_candidate_train_end()`, `_CANDIDATE_HOLDOUT_DAYS=7`, all 4 paths); BUG-43 hourly T-2h snapshots (CronTrigger hour="9-19", `_SNAP_WINDOW=2h`); Premium/Lock odds cap $3-$10; Wilson 95% CIs on `/api/track-record`; `win_prob_raw` comment fix; dashboard Phase-2 improvement chart; perf window 30→60d |
| `812c7e1` | Chart: 22 Jul marker labeled in heading + on-canvas |
| `653a5b9` | Conditions indicator v1: `/api/conditions-today` + header badge + per-day going dots on chart; `offgoing_pct` per day in `/api/performance` |
| `7942d98` | Conditions v2: meeting-level (venue chips, worst-seen classification) |
| `55dbe76` | Conditions v3: read live `RunnerPredictionRow` (RacePredictionRow dead-table fix) |

### The Benter blend (engine.py `_apply_benter_blend`)
- `score_i = α·log(p_model_i) + β·log(p_market_i)` → softmax within race.
- **Mass-preserving**: total field probability (post-multipliers) is kept and
  redistributed — absolute tier/Sharp semantics unchanged, ranking gains the market.
- Gated: `settings.benter_alpha/benter_beta` (env `BENTER_ALPHA`/`BENTER_BETA`),
  both 0.0 = disabled → legacy `_run_market_shrinkage` fallback. Also falls back
  when market coverage < 80% of field (missing-market runners use own model prob
  as proxy). Smoke-tested: mass preserved, default-off, coverage guard.
- **RELEASE ACTION (human, pending): set `BENTER_ALPHA=0.14`, `BENTER_BETA=0.53`
  on Railway.** Fit is thin (n=85) — refit in ~2 weeks on accumulated
  full-coverage data (see §4).

### Uncommitted working-tree state (deliberate)
- `horse_engine/api/main.py`: 4 Harville hunks in `get_edge_picks` (~line 4761-4860)
  — replace naive place-prob product with Harville box probabilities + est dividends.
  Kept out of every commit via `git apply --cached -R /tmp/harville.patch` pattern.
  Ship AFTER adding Stern/Henery discounting (§4).
- `infra/ra-proxy/README.md`, `infra/ra-proxy/main.tf` — pre-existing notes
  (LetsEncrypt rate-limit), not session work.

---

## 3. Monitoring now in place

- **Dashboard** (Win & Placement tab): "Phase-2 Improvement Tracker" — daily rank-1
  win% + place% from 2026-07-08, dashed pre-change baselines, orange 22-Jul marker,
  per-day going dots (green all Good/Firm, amber some Soft/Heavy, red heavy).
  Header badge: today's meeting-level conditions (`/api/conditions-today`).
- **Cloud routine** `trig_01QJTFwdRQbSfvgzvLevqjfU`: one-shot 2026-07-22T23:10Z —
  measures follow-up id=5 (isotonic sample count), projects rate to 10k, checks
  track-record drift vs session baselines (Hot 44.9/78.3 n=69, High 45.9/72.1 n=111,
  Strong 31.5/64.8 n=162).
- **Follow-up id=20** (2026-07-23, target ≥500): win_prob_raw accumulation check.
- Expected trajectory: snapshot market coverage jumps from ~29% within 1-2 days of
  BUG-43 (hourly T-2h snapshots); win% line climbs from ~22% toward high-20s over
  the first blended week IF the env vars are set.

## 4. Open work queue (priority order)

1. **Set BENTER_ALPHA=0.14 / BENTER_BETA=0.53 in Railway** (human release action).
2. Verify after 1-2 race days: blend engagement (trace `benter_blend: applied` vs
   `skipped_coverage`), snapshot market coverage %, chart movement.
3. **Follow-up queue hygiene**: re-arm id=5 (`UPDATE weekly_review_followups SET
   measured_at=NULL, measured_value=NULL, verdict=NULL WHERE id=5;` — it was consumed
   by a manual measure-now); id=12 already deleted (mislabeled duplicate); amend id=7
   (2026-07-30) — its `POST /api/retrain?days=14` alone CANNOT fix favourite
   under-rating while market features are win-masked; pair with re-ablation.
4. **Stern/Henery discounting** in `bets.py` Harville functions (λ2≈0.81, λ3≈0.65
   literature priors; later fit on own trifecta dividend history), then commit the
   Harville edge-picks hunks on top.
5. **Market-feature re-ablation** once `clean_market_history_count` ≥ 5000 (with
   BUG-43 fixed this accumulates fast) — decide unmask vs keep-blend-only.
6. **Refit α/β (~2 weeks)** on full-coverage clean data; expect β to firm up.
7. Going temperature (replace ×0.40/0.55 deflation) — blocked on more off-going
   sample; current evidence ambiguous (§1.4).
8. Remaining review roadmap: EB shrinkage of small-sample rate features, rolling
   30/90-day trainer/jockey form, full-field odds snapshots → own steam features,
   reliability table in weekly review, RAS data deal (email pending).

## 5. Hard rules / gotchas for the next session

- **Model release process**: refine → backtest → human promote. Nothing unattended
  touches live weights/curves/blend params. (memory `feedback_model_release_process`)
- **Win rate is the north star**; ROI is downstream of the bet-selection layer.
  User: "the model needs to predict; edge/premium concepts find ROI".
- **No API hammering** (memory `feedback_no_api_hammer`).
- **Sharp gates: tighten or hold, never loosen.**
- **PRE-READ** memory `code_review_bugs.md` before touching main.py / database.py /
  engine.py — FIX-A..AE lock-in patterns, incl. new FIX-AE (train-window clip).
- Selective staging: keep the Harville hunks out of commits until #4 above ships
  (`/tmp/harville.patch` may be gone next session — regenerate from
  `git diff horse_engine/api/main.py`, hunks in the 4400-5100 line region).
- DB access from local machine: direct writes are permission-blocked; read-only
  exports work via `railway run --service Postgres -- <python> <script>` with the
  URL taken from env inside the script (never materialise credentials to disk).
- The system-reference artifact (2e9f418e…) is stale: proxy is Hetzner Falkenstein
  (not DigitalOcean), tier bands now ≥46/36-45/30-35, isotonic disabled via
  `_load_output_calibration_curve` short-circuit. Needs a v0.2 pass.
