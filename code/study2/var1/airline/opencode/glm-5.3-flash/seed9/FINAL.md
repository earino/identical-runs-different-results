# Final Report — airline / XGBoost (autoresearch harness edition)

**Best Eval AUC: 0.7421** (baseline 0.7141, +0.0280). Best commit: `2db6c1b` (HEAD), validated `CONTRACT OK`.

## Final model

Ensemble of 3 identical-config XGBoost classifiers (only seeds differ: 42/7/3), averaged:
`n_estimators=300, learning_rate=0.05, max_depth=20, colsample_bytree=0.2, colsample_bynode=0.85`,
hist tree method, native categorical support. Features: raw DepTime/Distance/log-dist, native
categoricals (Month, DayOfMonth, DayOfWeek, UniqueCarrier, Origin, Dest, dep_hour), cyclic
hour sin/cos, flight-count features, and 17 smoothed out-of-fold target encodings (airports,
carriers, routes, hour/airport/carrier/month interactions, distance-decile interactions).
TE maps are fitted on train only; train rows use K=5 OOF encodings; `predict_proba()` applies
full-train maps inside `prepare()`, so the hidden holdout is handled leak-free.

## Changes that mattered most

1. **Heavy column subsampling + deep trees** (exp 16-18, 0.7226 → 0.7360): with many correlated
   TE features, `colsample_bytree=0.2` + `colsample_bynode` decorrelates splits and unlocks deep
   trees (d12 → d16 → d20 each added ~+0.002). This was the single biggest lever (+0.013).
2. **OOF target encoding of high-cardinality categoricals** (exp 7-8, 0.7171 → 0.7197): Origin,
   Dest, Route, carrier combos, hour combos, with per-column smoothing.
3. **Feature-set hygiene under colsampling** (exp 28, 0.7363 → 0.7417): pruning the weak low-card
   TEs (te_Month/te_DayOfWeek/te_DayOfMonth) that duplicated native categoricals was worth +0.004;
   only *weak* features dilute colsubsampled trees.
4. **Distance-decile interaction TEs** (exp 22, +0.0013): DistDecile, Carrier×DistDecile,
   Hour×DistDecile, using train-quantile bin edges.
5. **Tuned shallow-to-deep regime + 3-member seed bagging** (exp 6, 12: 0.7141 → 0.7205): depth 4
   (later 20), 150-300 trees, lr 0.05; averaging 3 seeds gave a robust +0.0006.

## Things that did not help

1. **Deeper/wider models without colsampling** (exp 3-5): depth 6-8 with early stopping + refit
   scored 0.7116-0.7122, below the 30-tree baseline — the 2006 eval punishes overfitting fast.
2. **`hhmm` as a native categorical** (~1200 levels, exp 10): 0.7126, catastrophic memorization.
3. **Interaction TEs at low smoothing** (exp 9: OriginMonth/OriginDOW/CarrierHour → 0.7191;
   exp 23: Origin/Month×DistDecile → 0.7350) and **TE smoothing scale** (×0.5/×2: no change).
4. **Feature duplication & K=10 OOF** (exp 30, 14): duplicating strong TEs 0.7389, K=10 0.7203/0.7413.
5. **DART member, row subsampling, min_child_weight, reg_lambda, gamma, cat_smooth, max_bin=128,
   depth-diverse members, lossguide**: all neutral or worse.

## With more budget

I would (a) grid bytree/bynode jointly at d20-d24 with the pruned feature set (the 0.8→0.85→0.9
bynode sweep suggests the optimum is flat and slightly higher with more features pruned),
(b) revisit interaction TEs one family at a time under the *pruned* feature set (exp 19/23 were
tested against unpruned sets), (c) try 5-fold-averaged OOF-TE *ensembles* (one model per fold's
encoders) instead of a single OOF matrix, and (d) tune per-feature TE smoothing (Origin/Dest/Route
individually) — all small, but the plateau at ~0.742 looks like a feature-quality limit, not a
capacity one.
