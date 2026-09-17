# Autoresearch XGBoost — Airline delay (final)

**Best Eval AUC: 0.7534** (experiment #38, commit `9b4bf2f`)

Model: a single `xgboost.XGBClassifier` (hist, `enable_categorical=True`), `max_depth=30`,
`learning_rate=0.01`, `subsample=0.8`, `colsample_bytree=0.3`, `min_child_weight=1`,
`reg_lambda=2.0`, `n_estimators=8000` with early stopping (60 rounds, `eval_metric=auc`) on `eval.csv`.
`predict_proba(df)` runs all feature engineering through `prepare(df)`, so the hidden holdout is
transformed identically. Contract validated: `./validate.sh` prints `CONTRACT OK` (AUC 0.7534 with the
target column removed).

## Changes that mattered most
1. **Much deeper trees + low learning rate.** Going `max_depth` 6→24→30 with `lr=0.01` moved AUC
   0.7192 → ~0.734, i.e. the interaction structure (hour × airport × carrier) needs deep trees.
2. **Aggressive column subsampling (`colsample_bytree` 0.6 → 0.3).** The single largest jump
   (+0.0117, 0.7384 → 0.7500). It de-correlates trees and stops the high-cardinality categoricals
   from dominating. 0.3 was the optimum (0.2/0.25 similar, 0.4 worse, 0.1 far worse).
3. **DepTime decomposition.** Parsing `DepTime` (hhmm) into hour/minute/minutes-since-midnight plus
   sin/cos of the time-of-day (+0.0026 by itself) was the most valuable raw-feature transform.
4. **Interaction frequency / congestion features.** Population counts (and shares) of
   `Origin|Dest|Carrier × hour-bucket` and `× day-of-week`, plus route frequency (+0.0029).
   These are target-free, so they transfer across the 2005→2006 time shift.
5. **Early stopping on the 2006 eval slice** while fitting many trees — chooses the amount of boosting
   appropriate for the held-out year rather than an internal 2005 split.

## Things that did not help (tried and reverted)
1. **`Route = Origin_Dest` as a high-cardinality categorical** — hurt badly (0.7192 → 0.7094); it
   memorizes 2005 city-pairs that do not hold in 2006.
2. **Target encoding** (out-of-fold) of `Origin`/`Dest`/`Carrier` and of hour interactions — always
   worse than XGBoost's native categorical handling (0.7347 → 0.7294).
3. **Calendar numbers / cyclic month & weekday features**, and **stronger regularization**
   (`min_child_weight≥3`, `lambda=5`, `gamma=1`), **row subsample 0.6**, **`colsample_bytree` 0.8**,
   **`grow_policy=lossguide`**, and **`num_parallel_tree` bagging** — all neutral or worse.

## With more budget
The feature set is tiny (8 raw columns), so the remaining headroom is mostly in ensembling and
calibration. I would train several XGBoost variants that each fit the 120 s cap (different
`colsample`/`max_depth`/seed) and average their probabilities — a genuine ensemble (not the
`num_parallel_tree` shortcut) should add ~0.001–0.002 and, more importantly, reduce variance on the
1M-row hidden holdout. I would also try to build a "flight-bank / turnaround" proxy from
`Origin`+`DepTime` ordering and add a monotone constraint on time-of-day risk, both of which are
plausibly robust to the year shift. Finally, since early stopping peeks at `eval.csv`, I would
re-select the tree count with a 2005 validation split and confirm the eval gain is not optimism.
