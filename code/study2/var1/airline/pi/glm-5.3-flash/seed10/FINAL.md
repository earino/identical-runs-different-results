# Final report — airline delay (XGBoost, autoresearch harness)

**Best Eval AUC: 0.7357** (baseline 0.7141; +0.022). Final model: average of 24 diverse XGBoost
classifiers over engineered features; validated (`CONTRACT OK`), AUC reproduced via `predict_proba`
with the target column removed.

## Changes that mattered most

1. **Carrier × hour interaction categorical** (exp 7/9): the single biggest feature win (+0.011 alone).
   Delay propensity is driven by carrier-specific schedules within the day; `hour_dow` (hour × day-of-week)
   adds a further robust scheduling-pattern signal.
2. **Row-local time-of-day features** (exp 4): `dep_hour`, `dep_min` from the hhmm `DepTime` (with midnight
   wrap for 2400–2620 codes), plus int-coding of the `c-<n>` Month/DayofMonth/DayOfWeek strings (+0.006).
   `dep_min` proved surprisingly valuable (dropping it cost −0.007).
3. **Ensembling** (exp 21–34): averaging diverse members (depth 4–8, lr 0.03–0.15, mcw 10–100,
   subsample/colsample/gamma/reg variations, lossguide member) took 0.7300 → 0.7357. More members helped
   with diminishing returns (3: 0.7335 → 10: 0.7348 → 24: 0.7357; 28–32 ≈ 24).
4. **Leaf regularization `min_child_weight=20`** (exp 11): enabled slightly more trees (200 vs 100) without
   overfitting the 2005→2006 shift (0.7292 → 0.7300).
5. **Train-fitted frequency encodings** for carrier/origin/dest/route (exp 8, +0.0006, confirmed useful by
   ablation exp 35) and `max_bin=512` (exp 33, +0.0001).

## What did not help

- **High-cardinality categoricals**: `route` (Origin_Dest, ~4200 levels) as a categorical cost −0.015;
  `dep_time` hhmm categorical (~1100 levels) cost −0.028; `origin_hour` neutral. Tiny groups → memorized noise.
- **Target encoding** (OOF, smoothed m=30): −0.0025 for carrier/origin/dest/route; exactly neutral (−0.0001)
  when restricted to low-cardinality keys (carrier/hour/dow/month). XGBoost's native categorical partitioning
  already captures these effects.
- **Capacity without regularization**: depth 8 / 300–400 trees / lr 0.08 consistently worse (0.6977–0.7164);
  holiday-window flags & day-of-year features (−0.003); subsample/colsample 0.8 (−0.003); row-bagging at 80%
  (wash); per-member early stopping on a 15% train split (−0.004, stops too early); carrier_dow (−0.002).

## Theory

The dataset's signal is dominated by *scheduled* delay patterns: time-of-day, carrier scheduling,
weekday, and season. Train (2005) and eval/holdout (2006) are time-separated, so models that fit 2005
noise (deep trees, many trees, high-cardinality splits) degrade sharply on later data. Robust, coarse
scheduling features + moderately regularized trees + averaging generalize best.

## With more budget

- Larger/wider ensemble (~50–100 members) and bagged members with higher subsample fractions;
- A small stacked meta-learner or per-member weights from time-split CV *within* train (avoiding eval leakage);
- More interaction search (origin × hour-dow, carrier × route frequency) using monotone/regularized variants;
- finer DepTime handling (scheduled vs. red-eye wrap-around buckets) and distance-scaled hourly patterns.
