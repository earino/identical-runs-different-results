# FINAL — airline departure-delay XGBoost

**Best Eval AUC: 0.7606** (commit `8b8b106`, experiment #20). Baseline was 0.7141.
Contract validated: `./validate.sh` prints `CONTRACT OK`, and `predict_proba(df)` reproduces the
full feature pipeline on a raw DataFrame with the target column removed.

## Changes that mattered most

1. **Time-of-day features.** `DepTime` (hhmm, with 24xx folded to 00xx) → `hour`, `minute`,
   continuous `tod`, and cyclical `tod_sin`/`tod_cos`, plus a 15-minute rounding `q15`. The target
   rate rises monotonically from ~4% at 05:00 to ~85% after midnight; the cyclical encoding lets the
   23:00→00:00 wrap be represented smoothly. This alone moved 0.714 → 0.726.
2. **Airport-hour congestion features (biggest single win).** Target-free lookups fit on
   `train.csv` only: flight count at `(Origin, hour)` and `(Dest, hour)`, the same counts
   normalized by the airport's yearly total (`o_share`, `d_share`), distinct-destination counts, and
   mean route distance per airport-hour. This captures how heavily a bank is loaded and moved
   0.726 → 0.740.
3. **Deep, sparsely-sampled trees.** `max_depth=26` with `colsample_bytree=0.2`, 500 rounds,
   `learning_rate=0.013`, `tree_method=hist`. Shallow models (depth 6–8) plateaued near 0.729;
   growing the depth while holding colsample low lifted 0.740 → 0.7606. Depth beyond ~26 and
   ensembles of models with different depths/seeds did not add.
4. **Frequency encodings** for `UniqueCarrier`, `Origin`, `Dest` and `Route` (train-fit, target-free)
   gave a small consistent gain and cheaply expose airport/route scale.
5. **Keeping `Month`/`DayofMonth`/`DayOfWeek` as native XGBoost categoricals** (fixed to the training
   levels, unseen levels → NaN) rather than numeric.

## Things that did NOT help

- **Route as a categorical / top-N route categorical** (-0.003 … -0.01): high-cardinality
  origin–destination identity overfits 2005 and does not transfer to the 2006 eval slice.
- **Target encoding** of carrier/airport/route (single unsmoothed mean gave 0.70 vs 0.72 base) and
  **date-level volume** features (per `Month_DayofMonth`): both are year-specific and transferred badly.
- **Calendar cyclicals / day-of-year / holiday flags**, **extra congestion aggregates** (distinct
  carriers, peak ratios, cumulative shares), **route-hour counts**, and **regularizing the deep model**
  (`min_child_weight`, `gamma`, L1/L2): all neutral or negative.
- **2-model ensembles** and **`max_bin=512` / gradient-based sampling**: no gain. **DART** timed out.

## With more budget

The eval slice (2006-slice1) and the hidden holdout (2006-slice2) are adjacent in time, so eval
tuning is a good proxy, but I would spend effort on *temporal robustness* rather than more
hyperparameter search, which has clearly plateaued. Concretely: (a) build features from the
*structure of the day's schedule* rather than raw train counts — e.g. per-airport hourly ramp
shape computed from the row's own timestamp, so it is independent of the training sample size;
(b) fit the congestion lookups on train+eval features only (never labels) to cover 2006 airports and
hours more densely; (c) test a compact, heavily-regularized ensemble of 3–5 deep models as a
variance-reduction step on the 1M-row holdout, which may reward smoothing more than eval does;
(d) try `hist` with 5-fold bagging over training rows to stabilize the very deep trees; and
(e) verify the depth benefit with a time-based split inside 2005 before trusting it, since the
deep-tree gain is the one result most likely to be partly an eval artifact.
