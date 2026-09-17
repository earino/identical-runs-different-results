# Final Report

**Best Eval AUC: 0.7462** (commit `6dd1577`, experiment #39; baseline was 0.7141).

## Setup

XGBoost (`xgboost.XGBClassifier`, hist, native categoricals, `max_cat_threshold=32`, `max_bin=512`)
with **2×10-fold CV bagging**: 20 members (lr 0.1, depth 6, n_estimators≤600, early stopping on the
held-out fold), predictions averaged. Validation AUC comes from `data/eval.csv` (2006 slice); the hidden
holdout is later/other rows, so all feature engineering lives inside `prepare(df)` (contract-compliant).

## Changes that mattered most

1. **CV-bagging ensemble** (5→10 folds, then 2 repeats × 10 folds, lr 0.1 members): 0.7274 → 0.7426. The
   single biggest lever — variance reduction over fold models beat every hyperparameter tweak.
2. **carrier × departure-hour interaction categorical**: 0.7169 → 0.7274. Schedule exposure per carrier is
   the dominant delay signal; the model cannot compose it cheaply from separate carrier/hour splits.
3. **DepTime engineering**: minutes-since-midnight, cyclical sin/cos, hour categorical, then 30-min and
   15-min slot categoricals (finer time granularity kept helping: +0.0018, then +0.0015).
4. **Longer training with early stopping** (600–1200 trees, lr 0.03–0.05 single model; later lr 0.1
   members): 0.7150 → 0.7166.
5. **Native categorical tuning**: `max_cat_threshold=32` (+0.0014) and `max_bin=512` (+0.0001).

## Things that did not help

1. **Route (Origin→Dest) categorical** and **origin×hour / dest×hour interactions** — too sparse at 100k
   rows; splits wasted on noise (0.7025–0.7123).
2. **Target encoding** (OOF, smoothed) of carrier/origin/dest — duplicated what native categorical splits
   already learn; hurt (0.7092). Frequency encoding also neutral-to-negative.
3. **Deeper/narrower members and member regularization** — depth 8 (0.7079), depth 5 (0.7347),
   subsample/colsample 0.8 (0.7270), min_child_weight 10 (0.7411), patience 100 (no change).

## With more budget

I would try: (a) heterogeneous bagging that fits the time limit (mix of lr 0.1/0.03 members, feature-subset
members) rather than the timeout-prone lr-0.05 members; (b) two-level stacking on the bag's OOF predictions;
(c) a systematic slot-granularity × carrier interaction sweep (15-min carrier×slot at 0.7384 suggests the
level count needs pruning/smoothing, e.g. Kneser-Ney-style backoff to the hour level); (d) monotonic
constraints on time-of-day to bias toward the physically expected delay cascade.
