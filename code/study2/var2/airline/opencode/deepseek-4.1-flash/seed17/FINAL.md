# Final report

**Best Eval AUC: 0.7434** (experiment #19, commit `9eacb60`; validated).

## Changes that mattered most

1. **Time-of-day features.** Adding `dep_hour` (as a categorical), `dep_minute`, and
   `dep_time` (minutes since midnight) was the first big jump (0.7141 -> 0.7252). Departure
   timing is the strongest single signal in this dataset.
2. **Origin/destination x hour interactions.** `origin_hour` and `dest_hour` categoricals
   capture airport-specific congestion patterns through the day and were the largest feature
   gain (+~0.008 over the pre-interaction model, confirmed by within-train CV).
3. **L1 regularization (`reg_alpha=5`).** A robust regularization win (~+0.008) that also
   stabilizes the year-to-year shift between 2005 training and 2006 eval/holdout.
4. **Depth-diverse XGBoost ensemble.** Averaging six-to-eight `XGBClassifier`s with
   `max_depths` 5..12, `learning_rate=0.05`, `n_estimators=400`, and `colsample_bynode=0.9`.
   Variance reduction across capacities is more robust than any single tuned depth.
5. **Aggregate/network features.** Frequency counts for `Origin`, `Dest`, `UniqueCarrier`,
   `carrier_origin`, `carrier_dest`, route and carrier-route; carrier network size and origin
   diversity; `day_of_year`; and `Distance * dep_hour`. Each added small, consistent gains.

## Things that did not help

1. **Target encoding** (in-sample, out-of-fold, leave-one-out) of high-cardinality keys:
   no gain, and prone to a subtle leakage bug that briefly produced a spurious 0.7584.
2. **Route (`Origin_Dest`) categorical and calendar interactions** (`month_hour`, `dow_hour`,
   `origin_dow`, `origin_month`): hurt, because these memorise 2005-specific structure that
   does not transfer to 2006.
3. **Stochastic subsampling and constraints:** `subsample<1`, `colsample_bytree<1`, and
   monotone constraints all reduced AUC. A `sklearn` HistGradientBoosting blend and
   rank-averaging the ensemble also gave no improvement over plain probability averaging.

## What I would try with more budget

The ensemble is now the bottleneck: training takes ~115s against a 120s limit, so most
remaining ideas could not be tested within the timeout. With more budget I would first cut
cost (per-depth tree counts, fewer deep trees, or training the ensemble in a single boosted
process) to open room for the features that looked genuinely promising but were too slow:
the `carrier x distance-band` interaction scored +0.0012 in single-model screening before it
timed out. Beyond that, I would set up proper time-based validation (train on earlier months,
validate on later) to distinguish real signal from eval noise, and explore seed/feature-subset
bagging plus mild early stopping to make the ensemble more robust to the distribution shift.
