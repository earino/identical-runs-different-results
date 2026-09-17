# FINAL — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7520** (experiment #38, commit `385a901`), up from the 0.7141 baseline.
Final model: 20-model XGBoost ensemble (`hist`, native categoricals) over a feature set built in
`prepare()` from calendar/time fields, categorical carrier/airport features, schedule-congestion
counts, rolling "bank" counts, congestion ratios, and out-of-fold target encodings.

## Changes that mattered most

1. **Target encoding of time-conditioned categorical keys (biggest lever, +0.015).** OOF
   (5-fold) smoothed target mean for `origin`, `carrier`, `dest`, `route` and their
   half/quarter-hour variants (`origin_half`, `carrier_half`, `route_half`, `origin_hour`,
   `origin_10`, `carrier_quarter`, `origin_quarter`, `carrier_origin_half`, ...). Encoding is fit
   on training data only; `prepare()` applies the full-train map to unseen rows. Smoothing k=50.
2. **Schedule-congestion counts at fine time resolution (+0.011 total).** Frequency of co-scheduled
   flights per `Origin`/`Dest`/`Carrier` at hour -> half-hour -> quarter-hour -> 10-min granularity.
   Fine resolution captured airport "banks" and was dramatically better than hourly.
3. **Rolling bank counts (±15..120 min windows) and congestion ratios (+0.005).** Counts of
   departures/arrivals at the origin within a window of the exact scheduled minute, plus ratios
   such as short/long-window bank, origin/dest, and origin/arrival (turnaround pressure).
4. **Shallow-tree diversity ensemble (+0.003).** 20 models, depths 2–7, varied subsample/colsample
   and seeds, probabilities averaged. Depth 9 and high tree counts overfit the 2005→2006 shift.
5. **Hour as a categorical feature + `dep_minute` (+0.001).** Delay probability rises monotonically
   through the day; explicit hour bins helped the shallower trees.

## Things that did NOT help

- **`route` (Origin_Dest) as an XGBoost categorical** consistently *hurt* (0.7008–0.7071 vs 0.7141),
  as did structural network features (route mean distance, carrier counts).
- **Deep trees / many boosting rounds** (depth 9, 400 trees) overfit the temporal split badly.
- **Frequency-only encodings and early stopping** were neutral once the time-conditioned counts and
  target encodings were present (and target encoding without the count features was also neutral).

## Generalization note / what I would try with more budget

The train (2005) / eval (2006) split makes temporal drift the central risk: raw target-rate features
depend on 2005 behavior, so I preferred leak-free structural counts and OOF encoding, and used a
moderate smoothing constant. With more budget I would (a) tune per-key smoothing by cardinality and
validate on a *time-forward* internal split rather than random folds, (b) add inbound-aircraft
turnaround estimates using route-average flight duration, (c) test a stacking meta-learner over the
OOF base predictions (XGBoost meta-model), and (d) average over multiple OOF folds/seeds to reduce
target-encoding variance.
