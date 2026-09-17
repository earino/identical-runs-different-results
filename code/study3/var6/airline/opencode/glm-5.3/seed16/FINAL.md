# FINAL.md — Airline delay AUC benchmark

## 1. Problem

Binary classification of whether a flight is delayed ≥15 minutes (`dep_delayed_15min`,
positive class `Y`) from the 2005 100k-row training slice. Scored by AUC on the
held-out 2006 100k evaluation slice (the harness additionally scores an unseen
1M-row 2006 slice2 elsewhere). Single XGBoost model in `train.py`, trained within
4 threads / 6 GB RAM, inference via `predict_proba(df)`.

## 2. Result

- **Final official eval AUC: 0.7441** (experiment #10, commit `fc170d1`),
  vs. 0.7141 baseline (+0.030).
- `./validate.sh`: CONTRACT OK, eval AUC via `predict_proba` = 0.7441, run time 46.7s.
- Seed stability of the final config: 0.7441 / 0.7436 (seeds 42/1) — gains are
  above the ±0.001 seed-noise scale.

### Top-5 official configurations

| # | AUC | Configuration |
|---|--------|----------------|
| 1 | 0.7441 | Final: `Origin×hour` + `route×hour` categorical interactions (native, freq-ordered levels), mixed encoding, recency weights; XGB hist: ne800, d10, lr .03, mcw5, colsample_bytree .8, colsample_bynode .3, bin512 |
| 2 | 0.7318 | Same minus `origin_hour` cat, d4, colsample_bynode .5 |
| 3 | 0.7294 | Same at d4, col .6, no bynode, ne800 lr .03 |
| 4 | 0.7285 | + `route_hour` cat + linear recency weights, ne500 d4 lr .05 col .6 |
| 5 | 0.7216 | Mixed encoding (numeric Month/DayofMonth/DayOfWeek via `str.slice(2)`, native cats) + hour/minute, 200@4 lr .05 mcw5 col .8 bin512 |

## 3. What produced the gains (chronological)

1. **Mixed encoding**: parse `Month`/`DayofMonth`/`DayOfWeek` to numeric
   (`str.slice(2)`), keep carrier/origin/dest as *native categoricals* — ordinal
   label codes and one-hots were both worse. (+0.0075)
2. **Time features**: `hour = DepTime//100`, `minute = DepTime%100`. (+part of 1)
3. **Interaction categoricals** built only from train-fitted frequency-ordered
   levels: `route_hour = Origin_Dest_(hour)` (~33k levels) and
   `origin_hour = Origin_(hour)`; no target statistics anywhere. (+0.007, +0.012 combined)
4. **Recency sample weights** `w = 0.5 + month/11`: linear upweight of later 2005
   months against the 2006 shift; beat step/exp variants. (+0.002)
5. **Slow boosted regularization**: lr .03 × ne800. (+0.001)
6. **Randomized deep boosting** (the decisive step, +0.012 together):
   `colsample_bynode` 0.3–0.5 enables depth 7→10 without memorizing 2005
   noise; depth alone at d8 without bynode had earlier *hurt* (0.7085).

## 4. What failed (all measured on eval)

- All target encodings (plain/smoothed/hierarchical, per-cat or per-key); a
  target-encoded route-hour catastrophically overfit (0.67).
- Sin/cos cyclic features; coarse 2h/3h hour buckets; month-interaction cats;
  `Dest×hour` and `Carrier×hour` cats (only Origin× and Route× helped).
- Deep trees without per-node sampling; `lossguide`/max_leaves; subsample;
  monotone constraints; recency-specialist blends; correlated-model ensembles
  (+0.0002, and 2×48s models risk the 120s wall limit).
- **xgboost pathological-slow regimes** (measured hangs, avoided): d≥11 with
  high-card cats, ne≥1200 @ lr≤.02, max_bin 1024, colsample_bytree 1.0,
  colsample_bynode ≥.35 at d10 — each confirmed ≥2 hangs in fresh processes;
  the container's 4-CPU cgroup quota amplifies OpenMP spin-wait storms into
  multi-minute fits. The final config sits inside the reliably-fast envelope
  (7/7 successful runs incl. official + validate).

## 5. Robustness notes

- All preprocessing is fitted on `data/train.csv` only (`prepare()` is
  self-contained); eval/holdout pass through the same categories, unseen
  levels become NaN and are handled by XGBoost natively.
- No eval labels were ever touched for training decisions; the loop only read
  official AUCs from `run_experiment.sh` output (10 official experiments, all
  `ok`, 0 reverted-at-end; every non-improving commit was reset).
- Stopped before budget exhaustion with reason: every remaining candidate was
  measured as a wash (|Δ|<.0005: mcw, gamma, bynode .25–.4, col .7–1.0, bin256,
  lr/ne within safe range, seeds) or a confirmed pathological hang; further
  runs would be near-duplicate noise-chasing, which the brief explicitly forbids.
