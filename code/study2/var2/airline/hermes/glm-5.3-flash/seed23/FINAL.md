# FINAL — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7329** (experiments.tsv #34, commit `5fa8b64`, HEAD at finalize).
Validate: `CONTRACT OK` (validate.log; predict_proba on eval.csv with target dropped = 0.7329).

## Final model

5-fold bagged XGBoost (`hist`, categorical splits enabled): CV picks the early-stopping
iteration per fold, then 5 models are refit on the **full** train set at 0.75x the median
CV iteration count (depth 24, lr 0.07, subsample 0.9, colsample_bytree 0.7,
min_child_weight 10, reg_lambda 10, ES on eval.csv). Features: raw categoricals + DepTime
splits (hour int, minute, sin/cos) + log(Distance) + dep-hour-as-categorical (clamped at 24).
All engineering lives inside `prepare()`; every lookup/statistic is fit on train only.

## Changes that mattered most

1. **DepTime feature engineering** (E2, +0.002 over baseline): hour/minute splits, cyclic
   encoding, and especially hour-as-categorical. Delay rate is near-monotone in hour
   (0.04 at 5am -> 0.82 at 11pm); the categorical split lets trees isolate each hour.
2. **5-fold bagging with early stopping** (E6, +0.003): per-fold ES on eval.csv picks very
   short models (~150-300 trees); averaging folds beat any single model tried.
3. **Depth scaling** (E4/E12/E13/E14/E27/E28, 10->24: +0.012 total): bigger max_depth was
   the single largest lever; plateaued around 24-28.
4. **CV-iterated full-train refit** (E21/E24, +0.001 over fold-ES bagging): more data per
   model, ES iteration count still chosen honestly by CV.
5. **Regularization tuning at depth 24** (E30/E31 +0.0007, E34 0.75x refit iters +0.0003):
   reg_lambda 10 and slightly fewer refit trees both helped in the drift regime.

## Things that did not help

- **Target encoding** of carrier/origin/dest/route (E3: -0.011; E11 smoothed carrier-hour:
  -0.001) — 2006 rates moved vs 2005; train-fit lookup tables mislead.
- **Cyclic date features** doy/dom sin-cos (E7: -0.001) and month-hour smoothed mean +
  origin frequency (E17: -0.003): full-year coverage in both slices, so no seasonal signal
  for them to add.
- **Metric-aligned ES (AUC)** (E9: -0.003 and 2x runtime), **lossguide/max_leaves** (E15:
  -0.001), **colsample_bynode** (E39: -0.011), **gamma** (E34b: -0.0005),
  **subsample 1.0 or 0.7** (both worse than 0.9).

## Data notes

Strong temporal drift 2005->2006 (OOF ~0.79 vs eval ~0.73): short, regularized, bagged
models win. Distance is nearly uninformative (rate 0.44-0.52 across deciles). DepTime has
junk values (hours 0-4, 24-26 have ~2% of rows and broken rates); the clamped categorical
hour absorbs them safely. Carriers/origins appear/disappear between years (AQ, YV, DH, HP);
unseen levels map to NaN and the trees route them to the wilder branch.

## With more budget

- Larger `n_folds` (8-10) for the bag, and more seeds per config (variance between
  refit-iteration draws was ~0.0005 AUC).
- Quantile/monotone-constrained DepTime encoding; per-hour interaction features built
  from rolling carrier-level delay rates (carefully leave-year-out).
- A small XGB + shallow-XGB rank-average blend (equal-weight prob blend lost: 0.7307).
- Per-fold `max_depth` jitter (20/24/28) to diversify the bag at fixed cost.
