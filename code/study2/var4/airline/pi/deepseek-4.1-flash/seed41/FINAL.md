# FINAL — airline delay prediction

**Best Eval AUC: 0.7702** (experiment #32, commit `e8a1860`).

## Final model
`train.py` trains a 5-seed **ensemble of XGBoost** classifiers (`hist`, `enable_categorical=True`)
on a rich target-encoded feature matrix:

- XGBoost: `eta=0.04`, `max_depth=8`, `colsample_bytree=0.2`, `subsample=0.9`,
  `reg_lambda=30`, `min_child_weight=5`, `n_estimators=1200`.
- 5 models with different seeds (different OOF folds) averaged at prediction time.
- All feature engineering lives inside `prepare(df)` and every statistic is fit on `data/train.csv`
  only, so `predict_proba` reproduces it on the hidden holdout.

## Changes that mattered most
1. **Out-of-fold target encoding of categorical interactions** (12 keys: carrier, origin, dest,
   route, and their hour interactions). OOF encoding for training rows, full-train maps at inference,
   smoothed toward the global prior with `k=50`.
2. **Neighbour-hour encodings** — for each hour-keyed encoding, look up the encoding at hours ±1/±2.
   This captures delay propagation: a route/airport that is late at hour `h-1` tends to be late at `h`.
   Single-model AUC jumped 0.748 → 0.758.
3. **Neighbour-hour count features** — adding the train frequency of each neighbour key lets the model
   weight unreliable (rare) encodings down. Gave another ~+0.005.
4. **Minute-bucket (half-hour) interactions** — `route×minute-bucket`, then
   `route×hour×minute-bucket`, `carrier×route×hour×minute-bucket`, and
   `origin/dest×hour×minute-bucket`. These schedule-slot interactions produced the biggest late gains
   (0.7632 → 0.7686 → 0.7702).
5. **Deeper trees + low column sampling + seed ensembling**, plus a few train-only graph features
   (airport destination/origin degree, carrier route count, route operating days).

## What did not help (tried and reverted)
1. **Calendar / holiday features and day-of-week / month hour interactions** — neutral or negative.
2. **Hierarchical (parent-shrunk) target encoding** and **10-fold OOF** — the tree model already blends
   parent and child encodings; both were slightly worse.
3. **Reverse-route neighbour TE, cross inbound-congestion features, wider/extended graph features,
   a diverse (varied-depth) ensemble, `lossguide` growth, `max_bin=512`, and `eta=0.03`** — all within
   eval noise or worse. Several pure hyperparameter micro-tweaks were also neutral.

## With more budget
The clear signal is that **schedule-slot structure** is under-exploited. I would push minute bucketing
finer (`minute//15` or exact `DepTime`) and build the full neighbour structure over
`minute-bucket × hour` rather than only the ±hour dimension, then prune keys with poor support.
On the modelling side I would increase ensemble size (the 5-seed average was a reliable if small win),
try out-of-fold stacking of the base models, and — carefully, if the rules allowed — pseudo-labelling
the large unlabelled 2006 pool. Finally, I would tune the OOF/target-encoding smoothing per key
cardinality instead of one global `k=50`.
