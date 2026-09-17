# Final report — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7310** (baseline 0.7141, +0.0169)
Final config: single `XGBClassifier`, E6 feature set, n_estimators=2000, max_depth=10, lr=0.008,
subsample=1.0, colsample_bytree=0.3, colsample_bynode=0.5, min_child_weight=10, hist, native categoricals.

## Changes that mattered most

1. **Feature engineering v1 (E6, +0.0025)**: time-of-day decomposition of DepTime (hour, minute,
   cyclic sin/cos, early-morning/evening/red-eye flags), numeric month/day-of-month/day-of-week with
   cyclic seasonal terms, log1p(Distance). Delay cascades are strongly time-of-day dependent; raw hhmm
   integer forces trees to rediscover hour boundaries.
2. **Heavy column subsampling (E21–E24, +0.008 total)**: colsample_bytree 0.8 → 0.3/0.4 flipped the whole
   picture. With only ~7 features per tree, deep trees stopped memorizing per-airport noise.
3. **Deep trees + low learning rate + long training under that regularization (E26, E32–E34, +0.006)**:
   depth 6 → 10 (peak at 10; 12 slightly worse), 2000 trees at lr 0.008. The "overfits fast" regime at
   depth 8/600 trees on raw columns (E3, 0.6883) disappeared once col-subsampling was strong.
4. **colsample_bynode=0.5 (E30, +0.0005)**: per-split feature dropout on top of bytree subsampling.
5. **subsample 1.0 (E29, +0.0002)**: with bytree/bynode dropout active, row subsampling was unnecessary.

## Things that did not help

- **Target encoding** (smoothed, plain E7: 0.7055; out-of-fold E8: 0.7143) — native categorical splits on
  Origin/Dest already capture airport rates; TE added year-shift noise (train 2005 → eval/holdout 2006).
- **High-cardinality native categoricals** (route Origin_Dest, E9: 0.7015) and the FE v2 bundle
  (doy proxy + holiday windows + carrier×hour-block, E16: 0.7173).
- **Ensembling**: seed bag (E17), diverse configs (E18/E19), strong-point ensemble (E28), and a 2-seed
  bag of the final config (E39, 0.7309) all failed to beat the single best model (0.7310) — the config
  is already strongly self-regularized, averaging only diluted it.

## With more budget

I would grid the interaction of max_depth (9–11) with colsample_bytree (0.25–0.5) and mcw (5–15) more
finely, re-tune n_estimators/lr at the new depth (early stopping on a time-shifted internal split — e.g.,
last month of 2005 — rather than a random split, which fired too early), try origin×hour-block as a
moderate-cardinality categorical (route at 4k levels failed, ~1.5k may be safe), and revisit quantile /
MonotoneConstraint-style smoothing of the time features. Diagnostics (E20 per-member AUC) were the
highest-information experiment per CPU-second; more of those first.
