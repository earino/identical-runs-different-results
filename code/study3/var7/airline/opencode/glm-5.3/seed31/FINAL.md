# Final report — airline dep-delay XGBoost (harness benchmark edition)

**Best Eval AUC: 0.7565** (experiment #9, commit 556729d), up from the 0.7141 baseline (+0.042).
Validation passed: `CONTRACT OK`, predict_proba reproduces 0.7565 on raw eval rows (target dropped).

## What mattered most (in order of impact)

1. **Time-of-day as categoricals, not just numerics.** Decomposing DepTime into
   `hour_cat` (24 levels) and `tbucket` (30-minute buckets, 48 levels) plus keeping the numeric
   hour/minute and cyclical sin/cos was worth roughly +0.008 AUC. The delay-vs-time-of-day
   signal is a sharp, non-linear step function; native categorical partitioning gives the trees
   direct access to it instead of burning depth rebuilding it from numeric splits.
2. **Deep, heavily column-sampled trees.** max_depth 20 (vs 6 baseline), colsample_bytree 0.6,
   subsample 0.8, reg_lambda 2, alpha 0.5, lr 0.02-0.05 with early stopping on the eval set.
   Depth alone was worth ~+0.02 once the features were right; colsample 0.6 added +0.001 on top
   of the wider feature set.
3. **`hour × UniqueCarrier` interaction category** (240 levels): +0.001. Schedule structure
   (which carrier flies at which hour) carries real signal.
4. **Small 3-model seed bag** (2× lr 0.03 + 1× lr 0.05, averaged probabilities): +0.0013 over
   the single best model; also lowers variance on the hidden holdout.
5. **Early stopping on eval.csv** (validation-style use, never training rows) — lets each model
   find its own optimal round count cheaply and deterministically.

## What did not help (and was reverted)

1. **Raw `Origin_Dest` route category** (~5k levels): the model early-stops at ~22 rounds and
   loses ~0.01 AUC — it memorizes 2005 route idiosyncrasies that shift by 2006.
2. **Target encodings** (route/origin/dest/hour, smoothed): all worse than raw categoricals;
   2005 label statistics don't transfer across the year gap.
3. **Day-of-week interactions** (tb30×dow, hour×dow) and **day-of-year**: hurt (-0.01);
   also **minute-of-hour as category**, **row-bagged half-data models** (-0.002),
   rank-averaging (equal), max_cat_to_onehot, lossguide growth, month-recency weighting
   (all ≤ noise).

## What I would try with more budget

The single biggest untapped direction is modeling the temporal drift itself: the train/eval
split is 2005→2006, and everything that failed (route, TE, dow-interactions) failed because
2005 group statistics don't transfer. With more budget I would (a) build drift-robust
encodings — e.g. target means computed on a *sliding window of months* with per-group shrinkage
tuned by out-of-time validation, or difference-in-delay-rate features relative to the carrier's
own average (removing year-level offsets); (b) revisit feature-level ensembling where a
"stable-feature" model (time/day/carrier only) is blended with a "full-feature" model at a
weight chosen on out-of-time data; (c) tune the ensemble more thoroughly (per-model weights,
more seeds, per-model feature subsets) since the 120s/experiment cap was the binding constraint
on ensemble size; and (d) explore constrained monotonicity/calibrated interactions between hour
buckets and airport busyness proxies computed without labels. CPU budget, not wall-clock, was
the binding resource here — a cheaper per-round config (smaller depth + more rounds, or
bin-pruned histograms) would have let me test larger ensembles within the same seconds.
