# FINAL — airline delay (AUC) experiment report

**Best Eval AUC: 0.7444** (git `9ad95e7`, experiment #39 of 40; baseline was 0.7141, +0.0303).
Wall clock used: ~29 min of 230. CPU used: ~5,400 s of 18,000. Experiments: 40/40 (budget exhausted).
`./validate.sh` → `CONTRACT OK` (eval AUC reproduced at 0.7444 through `predict_proba` with the target column removed).

## Final model

`XGBClassifier`, `grow_policy="lossguide"`, `max_leaves=512`, `n_estimators=700`, `learning_rate=0.02`,
`subsample=0.7`, `colsample_bytree=0.6`, `min_child_weight=2`, `tree_method="hist"`,
`enable_categorical=True`; 3 identical models over seeds {42, 7, 2024}, probabilities averaged.

## The 5 changes that mattered most

1. **Decomposing `DepTime` hhmm into a real time-of-day (`hour + minute/60`, sin/cos, minute, missing flag)**
   plus numerics parsed out of the `c-<n>` calendar strings. Cheapest and largest single feature win
   (baseline 0.7141 → 0.7224 for the same tree settings).
2. **Refusing XGBoost categorical splits for medium/high-cardinality identities.** `Origin`/`Dest` (~300
   levels) and `Origin_Dest` (~4000 levels) as categoricals cost ~0.010 AUC (0.7255 with counts only vs
   0.7141/0.7062 with categoricals); regularizing them with `cat_smooth=100, cat_l2=50, max_cat_threshold=32`
   did not save them (0.7075). They are now represented by frequency counts instead.
3. **`lossguide` growth with many leaves and almost no `min_child_weight`.** The depthwise `max_depth=6`
   configuration was *underfitting*: 0.7255 → 0.7299 (48 → 96 leaves) → 0.7312 (`min_child_weight` 10 → 5)
   → 0.7318 (`min_child_weight` 2). `max_depth=4` (0.7210) and `max_depth=8` (0.7245) both lost to
   lossguide. Capacity, not regularization, was the binding constraint.
4. **Row/column subsampling at 0.7/0.6** (0.7348 → 0.7378). Sampling randomness was worth more than any
   explicit L1/L2/gamma penalty I tried.
5. **Congestion-count features, fit on `train` only**: flights per (Origin, 3-hour bucket), (Dest, bucket),
   (Carrier, bucket), (Origin_Dest, bucket) and (Origin_Dest, day-of-week). Worth nothing at low capacity
   (0.7235 → 0.7226 at `max_leaves=256`, depthwise) and +0.0041 once the trees were big enough to use them
   (0.7393 → 0.7434). Averaging 3 seeds of the same config added a further +0.0015 (0.7378 → 0.7393).

## The 3 things that did not help

1. **Out-of-fold target encoding** (smoothed mean of `y` for carrier/origin/dest/route/route×bucket keys,
   5-fold OOF for the training matrix, full-train statistics for inference). Slightly *worse* than plain
   counts at low capacity (0.7219 vs 0.7235) and again at high capacity (0.7411 vs 0.7434). The signal in
   these identities is volume/congestion, not the identity's own delay rate — which is also consistent
   with the year shift: 2005 delay rates do not transfer cleanly to 2006.
2. **Raw categorical handling of identities** (route, `carrier×dest`, `origin×hour`, `dest×hour`). Every
   variant lost 0.010–0.017 AUC even with `max_cat_to_onehot=1` and strong categorical regularization.
3. **Explicit regularization and small refinements**: `reg_lambda=3 / reg_alpha=0.5 / gamma=2` (0.7245),
   `min_child_weight=1` (0.7317 vs 0.7318 at 2), exact-hour instead of 3-hour count buckets plus extra
   day-of-week count keys (0.7411/0.7428 vs 0.7434), and ratio/share versions of the count features
   (0.7416). More trees at a lower learning rate also plateaued (lr 0.015 × 2500 trees = 0.7347 vs
   lr 0.02 × 1500 = 0.7348).

## What I would try with more budget

The capacity curve was still rising when the budget ran out (leaves 96 → 192 → 256 → 384 → 512 gave
0.7299 → 0.7329 → … → 0.7444, and the last run before that needed 3 seeds × 512 leaves to stay inside the
120 s limit), so the first thing to do is push capacity under the time cap — fewer, larger trees, or
`max_leaves` well above 512 with an explicit early-stopping-by-wall-clock rule — since the ceiling here is
compute, not the feature set. Second, the evaluation is a genuine year shift (train 2005, eval/holdout
2006), so I would add shift-aware features: per-month rather than per-identity delay statistics with
recency weighting, and a model that predicts the *change* in delay propensity between calendar periods
rather than the level. Third, I would revisit route-level information properly — route frequency helped
only through counts, but a route encoding estimated on an adjacent time window (rather than out-of-fold on
the same window) should behave better than the OOF target encoding that failed here. Finally, with the
remaining time I would grow the ensemble in the direction that was actually productive (seed averaging,
not config mixing — mixed-config averaging was neutral at low capacity, 0.7252 vs 0.7255, and slightly
negative at high capacity, 0.7437 vs 0.7444) and calibrate per-month offsets, which AUC ignores but which
usually stabilizes the ranking across a shifted holdout.
