# FINAL — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7664** (experiment #20, commit `1454586`)
Baseline (30 trees, raw features): 0.7141 → final: **0.7664** (+0.0523).

## Changes that mattered most

1. **Traffic-frequency / congestion features (largest gain, ~+0.02 in total).**
   Flight counts joined from training data only, keyed by origin / destination / route /
   carrier combined with the scheduled 30-minute departure bucket, plus the counts of the
   neighbouring buckets (origin & route ±1…±4, dest & carrier ±1…±3). These encode local
   airport/route congestion at the moment of departure. The 30-minute resolution and the
   adjacent-window counts each added ~+0.006–0.009 AUC.
2. **Time-of-day and calendar features.** Splitting `DepTime` into hour/minute/fractional
   time + cyclic sin/cos, plus day-of-year and weekend indicators, beat using the raw
   `hhmm` integer.
3. **Deep, strongly L1-regularised trees.** `max_depth=15`, `reg_alpha=3`, `colsample_bytree=0.3`,
   `learning_rate=0.03`, 500 rounds. Because train (2005) and eval (2006) are separated by a
   year, shallow models underfit and unregularised deep models overfit; deep trees with heavy
   L1 and low column sampling generalised best. `max_depth` 5→15 was worth ~+0.008.
4. **Seed averaging.** Averaging 4 model fits (same config, different seeds) is a small but
   deterministic gain (~+0.0005) and reduces variance.
5. **Network-diversity & carrier-route frequencies.** Distinct carriers per route / routes per
   origin / carriers per origin, congestion *shares* (`route_bucket / origin_bucket`), and
   carrier×origin / carrier×destination flight counts.

## Things that did NOT help (and were reverted)

- **Route as a categorical feature** — very high cardinality, badly overfit across the year
  split (0.716 → 0.701).
- **Out-of-fold target encoding** of carrier/origin/dest/route-hour delay rates — hurt
  (0.7469 → 0.7454); the native categorical splits plus count features already capture it,
  and historical delay rates transfer poorly across years.
- **Cumulative daily congestion** and **plain/cumulative count encodings**, holiday/peak
  flags, destination/postal-month counts — all neutral or negative.
- **A more diverse depth/colsample ensemble** tied the single-config seed average exactly,
  so the simpler model was kept.

## What I would try with more budget

The features are almost entirely *train-statistics* features (frequencies), and the model
is selected on a 100k 2006-slice. The obvious next steps are (i) a proper **cross-year
validation** scheme (e.g. hold out part of 2005 by time) so model selection is not tuned on
`eval.csv`, and (ii) **target-aware historical delay rates with time decay** estimated on
2005 and blended with congestion, which is the one signal we found that did not transfer as
coded. Beyond that: pseudo-labelling the (unlabelled) holdout-style rows or training on
larger slices, a small stacking ensemble over depth/feature-subset models, and a systematic
search over bucket width and window radius (we only tested a handful and found 30 min / ±4
best). Given the year shift, I would also explicitly test feature stability between 2005 and
2006 to drop anything that drifts.
