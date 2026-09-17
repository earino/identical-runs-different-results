# FINAL — airline delay XGBoost (harness benchmark, scenario 2)

Best Eval AUC: **0.7539** (baseline: 0.7141, +0.0398). Verified with `./validate.sh` → CONTRACT OK
(predict_proba path reproduces the full feature pipeline on unseen data, target column dropped).

## Final model

2-member XGBoost ensemble (hist/lossguide, 768 leaves, lr 0.03, ≤2400 trees, es 300; seeds 42/7,
colsample_bytree 0.6/0.75, colsample_bynode 0.8, subsample 0.8, reg_lambda 5, reg_alpha 1,
min_child_weight 10, max_bin 512, 4 threads) over ~20 engineered features.

## Changes that mattered most

1. **Time-of-day granularity as categoricals** (+0.005 over cyclic-only): dep_hour (24), half-hour (48),
   15-minute (96) departure bins as native categoricals. The single most robust feature direction —
   each finer step gained until 5-minute bins (288 levels) oversparsified and lost 0.006.
2. **Lossguide growth + leaf budget** (+0.004 over depth-wise d10): max_leaves 512→768 under strong
   regularization; wide-shallow trees suit one-hot categorical splits.
3. **Strong regularization enabling capacity** (+0.005): colsample_bytree 0.6 + colsample_bynode 0.8 +
   reg_lambda 5 + reg_alpha 1 turned the lossguide capacity from overfit-prone into +0.005.
4. **Congestion counts** (+0.001): log flights per (Origin, hour) and (Dest, hour) from train only,
   plus origin/dest traffic counts and per-carrier median hour/distance offsets.
5. **2-member ensemble** (+0.0002–0.0004 across seeds/colsample/subsample diversity); 3 members added
   nothing beyond noise.

## What did not help (all reverted)

- **OOF target encoding** (carrier/origin/dest/route, smooth 20–50): consistently below plain native
  categoricals at 100k rows (0.7109–0.7253 vs 0.7263 contemporaneous) — TE noise outweighed rank signal.
- **High-cardinality interaction categoricals**: route (~28k levels) crashed AUC to 0.7106; hour×dow
  (168) and carrier×quarter (1.9k) also net-negative; schedule structure is already captured by the
  per-carrier median-hour offset.
- **More cyclic encodings**: month sin/cos, dow sin/cos, weekend flag, redeye/rush flags — all ≤ +0.0002
  or harmful; the binned categoricals already encode the same non-monotone structure.

## With more budget

- Bagged ensemble of 4–6 members with per-member feature subsets (route/TE held out for members that
  can use them with heavy shrinkage) — the 2→3 member step suggested diminishing returns for plain
  seed bagging, but feature-subset bagging is untested.
- Per-origin or per-carrier ordinal rankings of delay propensity (rank-transformed, smoothed) as a
  leakage-safe substitute for target encoding.
- A proper CV-based hyperparameter sweep of eta/lambda/leaves on a 4-fold inner split (all tuning so
  far was eval-guided single shots), and DepTime=2400+ wrap handling as separate weekend/next-day flags.
