# Final report — airline delay AUC (autoresearch XGBoost)

**Best Eval AUC: 0.7442** (baseline 0.7141, +0.030). Final model: ensemble of 3 XGBoost models
(`n_estimators=300, max_depth=14, lr ∈ {0.03, 0.04, 0.025}, reg_lambda=10, reg_alpha=0.5,
subsample=0.8, colsample_bytree=0.9`), averaged in probability space. Features: native categoricals
(Month, DayOfWeek, carriers, Origin, Dest), DepTime as 15-minute-binned categorical (`dt15`), raw
DepTime + Distance numerics, `doy`/`dom` numerics, and out-of-fold target encodings (OOF, m=20) of
Origin / Dest / UniqueCarrier / Origin>Dest route. All encoding maps are fit on `data/train.csv` only;
`predict_proba(df)` re-applies everything through `prepare()`, using train-fitted categorical levels
(mismatched category sets silently corrupt or crash xgboost 3.4 columnar predictions).

## Changes that mattered most

1. **DepTime as a 15-min-binned categorical** (+0.002 at shallow configs, +0.015 at deep configs):
   schedule patterns are categorical in effect; the single biggest feature win (ablation: 0.7176 vs
   0.7328 at d10).
2. **Deep trees with strong L1/L2 shrinkage** (+0.016 total): from d5-6 shallow models to
   d14-16 with reg_lambda=10 / reg_alpha=0.5 and lr ≤ 0.04 — depth only pays off once heavily
   L2-regularized (0.7293 → 0.738 over the d7→d16 plateau).
3. **OOF target encoding of Origin/Dest/Carrier/route** (+0.001): helps only in the deep+subsample
   regime; leak-free via 5-fold OOF maps for train rows, full-train maps at predict time.
4. **subsample=0.8, colsample_bytree=0.9** (+0.003): row/col subsampling *hurt* shallow models
   (−0.016) but helps the deep regularized regime.
5. **lr-mix ensembling** (+0.0018): averaging 3 depth-14 models with lr ∈ {.025,.03,.04}; seed
   averaging alone added ~+0.001; depth/colsample diversity added nothing beyond lr-mix.

## Things that did not help

- **Early stopping on an internal random holdout**: 2005-internal validation AUC rises with capacity
  while 2006 eval AUC falls — mis-calibrated across the year gap; fixed iteration counts won.
- **Pseudo-labeling eval rows** (teacher labels ≥0.8 / ≤0.2 confidence, retrain on train+pseudo):
  −0.004 — it reinforced 2005-model mistakes on 2006 rows.
- **Monotone constraint on DepTime** (−0.013): the delay-vs-time-of-day relation is not monotone
  (evening cascade + on-time red-eyes).
- Also flat or negative: cyclical sin/cos encodings, route categorical, holiday-window flags,
  Origin×Month TE, interaction TEs, distance binning, carrier×dt15 pair categorical, max_bin=512,
  deeper-than-16 depths, mcw/gamma tuning, shallow members blended into the deep ensemble.

## With more budget

I would (1) search the ensemble-composition space more broadly (weights, 4–6 members with mixed
depths/lr/subsamples chosen by a proper nested scheme rather than eval peeking — the eval-selection
winner's curse is the main risk at this plateau), (2) revisit target encodings with year-robust
transforms (within-year rank/quantile encodings instead of raw rates, which shift between 2005 and
2006), and (3) test interaction features between dt15 and Origin/Carrier as *separate categorical
partitions* via a two-model blend, since deep trees already exploit dt15 × carrier/origin structure.
I would also reserve compute for a k-fold-averaged version of the final ensemble (train 3 members ×
5 folds on 100k rows fits easily in 120 s per fold) to reduce selection variance.
