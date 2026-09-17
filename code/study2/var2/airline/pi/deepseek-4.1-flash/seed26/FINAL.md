# Final report — airline delay prediction (XGBoost, AUC)

**Best Eval AUC: 0.7457** (experiment #40, commit `d79d989`) — up from the baseline 0.7141 (+0.0316).
Budget used: 40/40 experiments, ~85 min wall clock, 5024/18000 CPU-seconds.

## What mattered most

1. **Drop the calendar features (`Month`, `DayofMonth`, `DayOfWeek`).** Their per-category delay
   rates shift strongly between 2005 (train) and 2006 (eval) — e.g. January 0.541 → 0.444. Keeping
   them in the native-categorical/deep-tree model caused catastrophic overfitting (adding DOW/DOM
   back dropped AUC from 0.7425 to 0.7062). Removing them was the first real gain (0.7141 → 0.7156).
2. **Give the model enough capacity.** With the unstable features gone the baseline was badly
   underfit. Increasing trees (30 → 2000) and especially tree depth (6 → 30) moved AUC from 0.7156
   to 0.7425. The signal is dominated by fine-grained interactions between `DepTime`, `Origin`,
   `Dest`, `UniqueCarrier` and `Distance`, which only deep trees capture.
3. **Keep `DepTime` at full granularity.** Collapsing it to hour-of-day was the single worst change
   (0.7426 → 0.6949); the raw hhmm value is highly informative.
4. **Increase `max_bin` in the hist tree method.** `DepTime` spans 1–2620; with the default 256 bins
   its resolution is coarse. `max_bin=2048` was worth +0.003 over the default at the final config
   (0.7431 → 0.7457).
5. **Low learning rate + many trees** (`lr=0.03`, 800 trees, `max_depth=30`) gave a small but
   consistent gain over `lr=0.1`/300 trees.

## What did not help

- **High-cardinality `route` (`Origin_Dest`) as a categorical or as target encoding** — 0.7056 /
  0.7373. Deep trees already model the airport interaction natively; explicit routes just overfit.
- **Regularization** (`min_child_weight=20`, `reg_lambda=5`, `subsample=0.8`, `colsample=0.8`,
  `lossguide` leaves). Every form of constraining the trees reduced eval AUC; the deep unregularized
  model was consistently the best on the held-out year.
- **Calendar features in any form** (native categorical, numeric ordinal, or hour target encoding
  beyond the plain hour rate), and **seed ensembling** (3-seed average was exactly equal to the
  single model — deep trees are already stable across seeds).

## With more budget

The ceiling here is set by the available columns: there is no weather, aircraft, or arrival-time
information, so most of the residual error is irreducible. I would (a) sweep the `max_bin` /
`max_depth` / `lr` / `n_estimators` interaction more finely around the current optimum, since
`max_bin` was still improving at the last data point; (b) try domain-adaptive sample weighting to
correct the 2005→2006 covariate shift (the calendar ablation shows how much year-specific structure
hurts); (c) build a small ensemble of the best deep model trained on different feature subsets /
`max_bin` values, which is more likely to transfer to the 1M-row 2006-slice2 hidden holdout than
further single-model tuning on the 100k eval slice.
