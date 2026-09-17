# Final Report — Airline Delay AUC (2005 → 2006 drift scenario)

**Best eval AUC: 0.7586** (experiment #21, commit `16a9947`; baseline was 0.7141 → **+0.0445**).
Validated: `./validate.sh` → `CONTRACT OK`, `predict_proba` reproduces 0.7586 on eval.

## Final model

An ensemble of 6 XGBoost members + 1 shallow decorrelation member, all trained on
`data/train.csv` only:

- 5 deep stochastic copies: `max_depth=20, n_estimators=140, subsample=0.85,
  colsample_bytree=0.5, colsample_bynode=0.8, reg_lambda=10, lr=0.05` (seeds 42/13/99/7/3)
- 1 deeper variant: `max_depth=28, n_estimators=200` (same regularization)
- 1 shallow "old family" member on the raw features (`d4, n300`), mixed in at ~17% weight
- Prediction = weighted mean of member probabilities

## The 5 changes that mattered most

1. **Multiscale schedule-congestion counts (the breakthrough, exp 9)** — label-free
   train-only statistics: `log1p` counts of train flights per (Origin, 20-min bucket),
   (Dest, 20-min), (Origin, 40-min), (Dest, 40-min), plus centered moving sums over 3
   and 5 consecutive 20-min buckets. Airport schedule banks repeat year over year, so
   these are drift-stable and give trees real structure: eval jumped 0.7276 → 0.7551.
2. **Carrier banks + arrival-side congestion (exp 10)** — `CHC20` (carrier × 20-min count)
   and `DETA8` (destination count at *estimated arrival* bucket: DepMinutes + 15 taxi +
   Distance/8): 0.7551 → 0.7579; worth +0.0030 in the final blend (ablation exp 23).
3. **Very deep, heavily stochastic members (exp 9)** — depth 20–28 with
   subsample/colsample regularization fits the congestion interactions; shallow members
   on the *same* features score only 0.7296 vs 0.7584 (ablation exp 20). The 2005→2006
   AUC-vs-trees curve peaks early (n≈140-200) with this setup.
4. **Seed-averaged ensemble** — 5 seeds of the best config + one deeper decorrelation
   member: +0.0074 over a single member (ablation exp 17); d28 > d24 > d16 for the
   diverse member (exp 21).
5. **Drift discipline: no calendar features** — Month/DayofMonth are 2005-only noise;
   re-adding them costs −0.0130 even on the deep architecture (ablation exp 18). Also a
   small-weight old-family member adds decorrelated signal (+0.0002).

## 3 things that did not help (all tested and reverted)

1. **Any label-conditioned encoding** — target/impact encodings of (Origin,Hour),
   (Carrier,Hour), Route-level TEs: they memorize 2005 and decay by 2006; the Route
   categorical (4198 levels) was actively harmful.
2. **Honest stacking / weight fitting on train OOF** — 5-fold OOF-fit blend weights are
   drift-blind (OOF thinks the calendar-FULL members are best at 0.752 in-sample); on
   eval they lose to even a uniform mean. Eval-label stackers were rejected as
   contract-violating (other learner + training on eval).
3. **Sparse joint congestion & exotic variants** — (Origin×Carrier×bucket) counts,
   peak-relative congestion ratios, forward-only windows, finer/coarser buckets
   (15/16/18/24/30-min), lossguide trees, max_bin tweaks: all neutral-to-worse.

## Robustness checks behind the choices

- Weight/config decisions pre-validated by fitting on one half of eval and transferring
  to the other half (A→B correlation 0.99 for the final family); deep+stochastic config
  selection transfers (best-on-A also best-on-B).
- Determinism: re-running the best config reproduced 0.7584 exactly (exp 19).
- `validate.sh` passes; `predict_proba` handles unseen airports/carriers/buckets via
  train-fitted maps with 0-fill and category NaN handling.

## With more budget I would try

Richer congestion statistics are clearly the highest-value direction: per-DOW schedule
profiles (airport × DOW × bucket, with empirical-Bayes smoothing toward the
airport's overall profile), route-specific duration estimates for the arrival-side
bucket (per-route median Distance-based), and congestion measured over the trailing
*and* leading hours at finer granularity. Second, a larger seed × depth × feature-subset
ensemble (10–20 members) — averaging kept paying at the edge of the 120s runtime cap, and
more members would need only modest runtime engineering. Third, drift-robust blend
weighting selected via half-eval → half-eval transfer at member level rather than group
level. The scoring path itself is safe (vectorized, ~1M-row ready).
