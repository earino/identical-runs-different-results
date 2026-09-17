# Final Report — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7191** (commit `3cd802d`), up from the 0.7141 baseline. Validated: `CONTRACT OK`,
`predict_proba` reproduces 0.7191 on `data/eval.csv` with the target column removed.

## What the final model is

An average of 12 shallow, diverse XGBoost classifiers (`tree_method="hist"`, native categoricals),
plus four count-based "congestion" features fitted on `data/train.csv` only:

- `f_origin_hour`, `f_carrier_hour`, `f_dest_hour`: number of 2005 departures sharing the row's
  (airport/carrier, hour-of-day); captures time-of-day traffic load.
- `f_carrier_origin`: number of 2005 departures by that carrier from that origin (hub structure).

All feature engineering lives inside `prepare()`, which `predict_proba()` calls, so it applies
identically to the hidden holdout.

## Changes that mattered most

1. **Shallow trees.** Depth 3 (0.7160) beat depth 6 (0.7141) and depth 2 (0.7140). The 2005→2006
   temporal shift punishes capacity; train AUC ~0.79 vs eval 0.715 showed the baseline already
   overfit. Depth 3 was the bias/variance sweet spot.
2. **Ensembling diverse shallow models.** A 12-model average across depths 2–5, learning rates
   0.05–0.1, and `lossguide` vs `depthwise` growth lifted 0.7160 → 0.7183. Variance reduction is
   robust under distribution shift.
3. **Hour-conditional traffic counts.** `f_origin_hour` (+0.0007), `f_carrier_hour` (+0.0005),
   `f_dest_hour` (+0.0001) and `f_carrier_origin` (+0.0002) each added a little. These are
   structural congestion signals that stay meaningful across years.
4. **Keeping `DepTime` numeric.** Feature importance showed `DepTime` dominating (0.42); trees split
   it directly, so derived hour/minute features were redundant.

## Things that did NOT help

1. **High-cardinality identity features.** `route` as a native categorical (0.7014) and target
   encoding of carrier/origin/dest/route (0.7047) both *hurt badly* — group identities shift
   between 2005 and 2006, so the model overfits them.
2. **More capacity / early stopping.** 400–600 trees, lower learning rates, heavy regularization,
   DART, and internal-validation early stopping (which chose 333 rounds) all scored at or below the
   shallow baseline. Train-internal validation is a poor proxy for the time-separated eval.
3. **More features or more ensemble members.** A 19-model ensemble (0.7180 vs 0.7183), row bagging
   (0.7176), `colsample_bynode` (tie), dropping `DayofMonth` (tie), and share-normalized congestion
   variants (0.7157–0.7161) gave no reliable gain — differences here are within the ~0.0026 AUC
   standard error of a 100k-row eval.

## What I would try with more budget

The 0.718–0.719 plateau is roughly 1 standard error wide, so the honest bottleneck is *reliable
validation*, not feature ideas. I would build a temporally ordered validation split inside 2005
(e.g. last 20% of the year by a real timestamp) and select/average on that, since the eval set is
too noisy and too seasonally shifted to rank configs finer than ~0.003. Beyond that: pseudo-label
or domain-adaptation using the 2006 eval *features* (no labels) to correct the 2005→2006 covariate
shift, and calibrated per-month ensembles to track the strong `Month` effect (importance 0.15)
that clearly changes year to year.
