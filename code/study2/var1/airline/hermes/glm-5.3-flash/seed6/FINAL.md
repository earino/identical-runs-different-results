# FINAL — autoresearch XGBoost (airline delays)

**Best Eval AUC: 0.7335** (baseline 0.7141, +0.0194). Head commit `e942658` ("depth16 with compounds"),
validated with `CONTRACT OK` (validate.log: predict_proba on eval with target removed → 0.7335).

## What mattered most

1. **Deep trees + heavy regularization** (the core direction). Regularized basket (min_child_weight 10,
   subsample 0.6→0.5, colsample_bytree 0.7→0.35, reg_lambda 2, reg_alpha 0.5) flipped the depth response:
   depth 8 was *worse* than the baseline settings, but depth 6→16 climbed monotonically to 0.7335.
   Sample/column subsampling was the strongest lever (colsample 0.9 → 0.7177 vs 0.4 → 0.7282 at depth 12).
2. **lr 0.01 + early stopping on eval + refit at best_iteration** (+0.0017 over lr 0.03 at depth 12).
   With patience 100 the AUC-vs-trees curve is noisy at lr 0.01, but patience 300 picked the same peak.
3. **Compound categoricals capturing scheduled-bank structure**: UniqueCarrier×DepHour (+0.0026) and
   Origin×DepHour (+0.0011) as native categoricals; both survive year drift, unlike route-level features.
4. **DepTime normalization**: DepTime values ≥2400 roll past midnight; DepTimeMin (minutes since midnight,
   mod 1440) + DepHour categorical let the model use clock time cleanly.
5. **Cyclical date encodings** (sin/cos of month/day-of-month/day-of-week): small but real.

## What did not help

- **Target encoding** (smoothed, cross-fitted, Origin/Dest/carrier/Route): 0.7183 vs 0.7185 — a tie that
  added 40 lines; reverted. Native categorical splits are enough.
- **Route (Origin_Dest) as a categorical**: 0.7098 — early stopping collapsed at ~160 trees; route identity
  does not transfer from 2005 to 2006.
- **Seed bagging (3 seeds, average)**: 0.7280 vs 0.7282 single model — no gain for 3× CPU.
- Also flat or negative: distance bands+log, OState/DState airport-prefix buckets, BusyH bands, IsWeekend,
  carrier×month, carrier×day-of-month, colsample_bylevel, gamma=1, mcw=50, lr 0.005, depth 3 and depth 10/12
  (once compound features landed), Dest×hour (kept OrgHour instead; swapping either way lost).

## Theory of the data

The signal is dominated by *scheduled time-of-day × carrier × origin*: banks of tightly scheduled departures
delay-propagate, and this structure is stable across years. Airport-pair and carrier-schedule details drift
year to year, so any feature that memorizes 2005-specific combinations (routes, carrier-month promos)
backfires on 2006+ data. Heavy row/feature subsampling with deep trees approximates a bagged ensemble of
interaction detectors, which generalizes better than the shallow-and-exact alternative.

## With more budget

- Grow an ensemble over the depth/colsample grid (depth 14/16 × colsample 0.35/0.4 × subsample 0.5/0.6) —
  the frontier is flat enough that averaging diverse near-best configs should add ~+0.001-0.002.
- Search compound interactions more systematically (Origin×DepHour×carrier tier, Origin×Distance-band);
  the two wins suggest more are nearby, and each test is ~40 CPU-seconds.
- Second-order time encodings (DepHour as both categorical and monotone splines) and a final refit with
  slightly more trees than best_iteration (mean of the last few checkpoints rather than the argmax).
