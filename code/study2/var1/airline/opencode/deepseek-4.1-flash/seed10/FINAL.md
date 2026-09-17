# FINAL — airline delay AUC

**Best Eval AUC: 0.7448** (experiment #35, commit `100158c`; baseline 0.7141, +0.0307).

Final model: `xgboost.XGBClassifier`, `tree_method=hist`, native categoricals,
`max_depth=8`, `learning_rate=0.02`, `n_estimators=1000`, `subsample=0.8`,
`colsample_bytree=0.7`, `min_child_weight=50`, `reg_lambda=30`, averaged over 3 seeds.
Contract validated: `./validate.sh` prints `CONTRACT OK` (predict_proba reproduces all
feature engineering on a raw DataFrame with the target column removed).

## Changes that mattered most

1. **Route-hour frequency + temporal neighbors** (`RouteHour_freq`, prev/next hour).
   Single biggest jump: 0.7293 → 0.7426. Counting scheduled flights on the same
   Origin–Dest route in the same (and adjacent) departure hour captures shuttle/bank
   structure that categorical route identity could not.
2. **Origin/destination congestion windows**: counts of departures per Origin-hour at
   offsets h, h±1, h±2, h±3 and Dest-hour at h, h±1, h±2, plus window sums and
   counts relative to total airport volume. Took 0.7206 → 0.7293 and reflects delay
   cascades.
3. **Stable frequency/congestion aggregations** of carrier / origin / dest / route
   / origin-day / dest-day / carrier-hour / carrier-day. Generalize across the
   2005→2006 time split where target statistics did not.
4. **Calendar features** derived from the `c-<n>` columns: `DayOfYear` and
   `IsWeekend` (verified DayOfWeek 1=Mon…7=Sun against known 2005 dates).
5. **Model capacity/regularization**: shallow+few trees underfit, unregularized deep
   ensembles overfit the time shift. Deep trees (depth 8) with low lr and strong
   `min_child_weight`/`reg_lambda`, averaged over 3 seeds, gave the final gain.

## What did not help

- **Route identity as a categorical** (4.2k levels): 0.7056/0.7044 — overfits and does
  not transfer across years.
- **OOF target encoding** of carrier/origin/dest delay propensity: 0.7183 — per-airport
  delay rates shift between 2005 and 2006; unsafe under the time split.
- **Holiday-distance / yearly cyclical features** and **lossguide / huge unregularized
  tree counts**: neutral-to-worse (0.7197–0.7204, 0.7186, 0.708).

## With more budget

I would push the winning idea further: build route–hour *relative* features (share of
the route's daily flights in that hour, rank of the hour), add carrier–route–hour and
neighboring-route competition counts, and test a proper time-ordered validation split
instead of selecting on `eval.csv`, since differences below ~0.001 AUC are within
sampling noise on 100k rows. I would also try a larger, more diverse seed ensemble
(6–8 members with varied depth/colsample) and monotone/regularization tuning on the
route-hour features, which were by far the richest source of signal.
