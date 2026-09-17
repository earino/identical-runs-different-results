# Final report — airline delay XGBoost

**Best Eval AUC: 0.7324** (commit `fb96906`, validated: `CONTRACT OK`, predict_proba reproduces 0.7324).

## Final model
3-seed bag (seeds 42/7/13) of `XGBClassifier`: hist tree method, `enable_categorical`, depth 6,
lr 0.008, 3000 trees with early stopping (100 rounds) on eval.csv, `min_child_weight=10`,
`reg_lambda=5`, `subsample=0.9`, `colsample_bytree=1.0`, **`colsample_bynode=0.3`**.
Features (all engineered inside `prepare()`): native categoricals for Month/DayOfMonth/DayOfWeek/
UniqueCarrier/Origin/Dest, raw DepTime, `dep_hour`, cyclical sin/cos of time-of-day,
`hour_cat` (24 levels), `carrier_hour` (UniqueCarrier × hour interaction).

## Changes that mattered most (baseline 0.7141 → 0.7324)
1. **Early stopping on eval + many trees** (exp 2, +0.001): 30 trees → up to a few hundred with lr 0.05.
2. **Time-of-day features** (exp 4, +0.0013): `dep_hour` + cyclical sin/cos, with past-midnight
   DepTime (2400–26xx) normalized. Cyclical encoding matters (dropping it cost −0.003).
3. **carrier × hour interaction categorical** (exp 7, +0.0067): by far the biggest single win —
   delay propensity is carrier-specific and strongly hour-dependent.
4. **`colsample_bynode=0.3`** (exp 21–23, +0.006 over bytree-only): per-node feature subsampling
   decorrelates trees far more than per-tree sampling on this ~13-feature set.
5. **3-seed bag** (exp 34, +0.0006) and leaf regularization (mcw 10, λ 5, exp 30, +0.0001);
   lr decay 0.05 → 0.008 (exp 12–14, +0.001).

## What did not help
- **High-cardinality interaction/route categoricals**: Origin×Dest route, Origin×hour, Dest×hour,
  DayOfWeek×hour, Carrier×month — all worse; they dilute splits and overfit 100k rows.
- **Target + frequency encoding** of carrier/Origin/Dest (OOF, smoothed): 0.7156 vs 0.7165 without;
  redundant with native categorical splits.
- **Depth/lossguide**: depth 8 or 7 and lossguide growth all worse than depth 6; likewise
  interaction constraints (collapsed to 0.7100), max_bin 512, gamma 1, cat_smooth, seed jitter,
  deeper subsampling (sub 0.7).

## With more budget
I would (a) build a proper time-ordered CV split inside the 2005 training slice to tune
early stopping without touching eval.csv, (b) explore smoothed target encoding of carrier_hour /
origin_hour used *alongside* native categoricals rather than instead of them, (c) try a larger
bag (5–8 seeds, possible if per-model trees are capped), and (d) probe monotone constraints on
the cyclical/hour block and per-carrier calibrations. All keep/discard decisions here were made
on eval.csv, so the final margin over ~0.726 single-model is modest; the bynode-sampling and
interaction-feature gains look the most robust for the hidden 2006 holdout.
