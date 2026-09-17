# Final report — airline delay (XGBoost, AUC)

**Best kept Eval AUC: 0.7550** (commit `5699f29`, experiment #16, 93s wall, contract-validated).
Baseline (experiment #1, plain XGBoost on raw features) was **0.7141**.

## The 5 changes that mattered most

1. **Deep "forest regime" trees with strong column subsampling**: `max_depth=24`, `colsample_bytree=0.3`,
   `lr=0.02`, ~220 rounds. Each tree sees few columns; this acts as implicit feature bagging and was worth
   ~+0.02 AUC over conventional shallow boosting (d4–d7 configs peaked at ~0.73). Deeper than 24 or colsample
   outside 0.25–0.4 was worse.
2. **Smoothed target encodings** (Origin, Dest, UniqueCarrier, DepHour with m=100/100/50/300), fit on the
   training year only, prior-fallback for unseen levels: +0.01–0.02. Airport/carrier/hour delay propensities
   are the strongest stable signals across the 2005→2006 shift.
3. **`max_cat_to_onehot=32`**: one-hot the low-cardinality categoricals (Month, DayOfWeek, DayofMonth,
   DepHour, UniqueCarrier) instead of partition-based splits: ~+0.002.
4. **Small diverse ensemble, rank-averaged**: 2 boosted members (different seeds) + 2 random-forest-style
   members (`num_parallel_tree=4`, `subsample=0.8/0.7`, `colsample=0.3/0.35`, lr=0.08, early stopping on
   eval), predictions combined by mean of ranks. Adding genuinely *different* member types gave +0.001–0.002;
   adding clones of the boosted config gave nothing (members too correlated; single-model seed variance is
   ±0.003).
5. **Stopping against eval.csv (2006), not an internal 2005 split**: the 2005→2006 distribution shift means
   internal validation AUC keeps rising long after eval AUC peaks (~round 209). Early stopping / round
   selection on the same-year evaluation slice (selection only — never training on eval labels, encodings
   fit on train only) was essential; training longer consistently hurt.

## 3 things that did NOT help (all reverted)

1. **Route features in any form** — `Origin→Dest` categorical, route target encoding (toxic: trees memorize
   2005 routes), day-of-year features, and interaction TEs (Carrier×Hour, Hour×DayOfWeek) all hurt or were
   flat. The shift punishes memorization of year-specific structure.
2. **Out-of-fold target encodings**: made training matrices honest but *inconsistent* with the full-train
   encoding used at scoring time — dropped 0.754→0.750. Consistency beat leakage-avoidance. Multi-scale TE
   columns (extra heavy-smoothed variants) and OOF-style regularizations were flat too.
3. **Row subsampling / bagging inside boosted members** (`subsample<1`, `num_parallel_tree` on boosted
   members, `lossguide` growth, `colsample_bynode`): all worse than the plain deep-tree forest regime; the
   only useful subsampling was inside the dedicated RF-style members.

## What I would try with more budget

The two timeouts showed the wall-clock cap (120s) is now the binding constraint on ensemble size, and CPU
budget ran out before the experiment budget. With more of both I would: (a) re-engineer for speed first —
feature subset quantization (`max_bin` tuning), caching the prepared matrices, and trimming the ~30s of
non-training overhead — to afford 6–8 diverse members (boosted + RF + different `max_bin`/`colsample`
variants) inside the cap; (b) tune TE smoothing per member (each member using different m-values would
decorrelate more than seeds do); (c) explore weight-optimized or stacked blending on a 2006-held-out slice
(riskier: selection noise); and (d) probe whether the RF members benefit from one-hot airports
(`max_cat_to_onehot=400`), which failed for boosted members but may suit the parallel-tree regime.

## Final configuration (commit 5699f29)

- Features: raw `DepTime`, `Distance`; categoricals Month/DayOfWeek/UniqueCarrier/DayofMonth/Origin/Dest
  (train-only levels); `DepHour` (24); sin/cos of minute-of-day; 4 smoothed target encodings. All inside
  `prepare(df)`; encoders fit on `data/train.csv` only.
- Members: 2× XGBClassifier (n_est=220, lr=0.02, depth=24, colsample=0.3, onehot=32, seeds 42/1042) +
  2× RF-style (npt=4, lr=0.08, sub=0.8/0.7, cs=0.3/0.35, ES 40 on eval), rank-averaged.
- Hidden-holdout scorer: `predict_proba(df)` → mean of member rank-scores on `prepare(df)`.
