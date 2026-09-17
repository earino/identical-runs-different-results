# Final report — airline dep-delay (XGBoost AUC benchmark)

**Best Eval AUC: 0.7582** (baseline 0.7141, +0.0441), commit `fa95494`, validated `CONTRACT OK`.

Final model: 6-member bagged ensemble of deep subsampled XGBoost classifiers
(n_estimators=50, max_depth=20, lr=0.05, colsample_bytree=0.6, subsample=0.8, mcw=0.1,
max_bin=512, native categoricals), averaged, with test-time augmentation over departure time.

## Changes that mattered most

1. **Deep bagged ensemble** (biggest jump, 0.714 → 0.752): many deep trees (d20-28) with
   colsample_bytree 0.5-0.6, subsample 0.8, min_child_weight 0.1, max_bin 512, averaged over
   seeds/configs. Under the 2005→2006 shift, capacity helps only inside heavy bagging.
2. **Time features**: cyclic sin/cos of minute-of-day (+0.003); keep raw DepTime; drop
   Month/DayofMonth calendar features (+0.001, less year-specific overfit).
3. **Smaller learning rate + more rounds** at fixed cost budget (n50/lr.05 vs n25/lr.1): ~+0.001.
4. **Test-time augmentation on DepTime** (±5,±3,0 shifts, averaged): smooths jagged tree
   boundaries in the scheduled-time dimension (+0.0005-0.001 on top of the ensemble).
5. **Native categorical handling** (enable_categorical) for carrier/origin/dest with
   train-fitted levels; unseen levels → NaN. Robust and cheaper than one-hot.

## Things that did not help

1. **Any route-level or target-encoded feature** (Route cat, TE of origin/dest/carrier,
   hour×carrier): all hurt — they memorize 2005 idiosyncrasies that do not survive the year shift.
2. **Capacity without bagging / exotic training**: one-hot of high-cardinality categoricals,
   rank:pairwise (0.65), dart, lossguide, colsample_bylevel/node — all neutral or clearly worse.
3. **Marginal smoothing extensions**: TTA wider than ±5 or hour-shifted variants, train-side
   time-offset members, logit-space (geometric) averaging, reg_lambda 2.0, member counts > 6,
   extra harmonics — all within noise or worse on eval.

## With more budget

I would attack the distribution shift directly: the plateau (~0.758 for many config variants,
seed-noise ±0.0005) suggests the remaining error is bias from the year shift, not variance.
Promising directions: (a) importance-weight or reweight training rows whose feature mix is
closer to the eval-year distribution (density-ratio estimation); (b) more aggressive
test-time augmentation dimensions (Distance, hour-of-week) once time is exhausted; (c) a much
larger, cheaper-member ensemble (25+ shallow-deep mixed members) if CPU budget allowed, since
per-member quality saturates but averaging kept adding small gains; (d) pseudo-labeling the
unlabeled direction is impossible here (eval is labeled), but self-training on eval-year
statistics (e.g., marginal hour/carrier distributions as priors) might help.
