# Final report — airline delay AUC

**Best Eval AUC: 0.7242** (experiment #36, commit `c60ce18`, 12-model ensemble + congestion ratios).
Baseline was 0.7141. Contract validated: `validate.sh` → `CONTRACT OK`.

## Changes that mattered most
1. **Ordinal + cyclical calendar** — treated `Month`/`DayofMonth`/`DayOfWeek` (`c-<n>`) as numeric
   ordinals and added sin/cos month and day-of-week, instead of XGBoost categoricals.
2. **Time-of-day features** — `DepTime` → hour, minutes-of-day, sin/cos, plus `dep_hour` as an explicit
   categorical (DepTime is the dominant signal).
3. **Congestion frequencies, especially ratios** — Origin/Dest/Carrier × hour counts and their
   normalized shares of that entity's total traffic (`*_hour_ratio`). The ratios gave the single
   largest late gain (+0.0011).
4. **Shallow trees** — `max_depth=3` (some 4) with 400 trees at lr 0.05 greatly beat depth 6/8; the
   cross-year (2005→2006) shift punishes deep trees.
5. **XGBoost ensemble** — averaging 12 models with seed/colsample (0.6–1.0) diversity gave a reliable
   +0.001–0.002 over a single model. All feature engineering stays inside `prepare()`, so
   `predict_proba` reproduces it on the hidden holdout.

## What did not help
- **Route (Origin×Dest) as a high-cardinality categorical or route-hour features** — overfit badly
  (0.7056 / 0.7187).
- **Target encoding of carrier/airports** and **network-centrality features** — no gain over native
  categorical splits.
- **DART booster** — worse (0.7213) and ~5× slower.
- **Deeper/larger models** (depth 8, 300+ trees at lr 0.05) — regression; and extra leaf
  regularization / `max_cat_threshold` changes were flat.

## With more budget
I would (a) run a small time-based CV hyperparameter search around depth 3–4 rather than trusting the
noisy 100k eval (±0.002), (b) add out-of-fold target encoding for carrier×hour and origin×hour with
proper cross-fitting, (c) try monotonic constraints on time-of-day, and (d) grow the seed ensemble /
bag over bootstrap resamples to squeeze out variance. The eval is next-year data, so I would weight
robust, transferable counts/ratios over high-cardinality memorization.
