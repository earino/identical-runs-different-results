# FINAL — airline delay (dep_delayed_15min) XGBoost

**Best Eval AUC: 0.7608** (experiment #40, commit `dd49d11`), up from the 0.7141 baseline.
Contract validated: `./validate.sh` → `CONTRACT OK`, and `predict_proba(df)` — called on `data/eval.csv`
with the target column removed — reproduces 0.7608, so all feature engineering lives inside `prepare()`.

## Changes that mattered most

1. **Out-of-fold target encoding + count encoding on `(route × time-of-day bucket)` keys.** The single
   biggest lever. Key = `Origin_Dest` crossed with `DepTime` binned to hour / 30 / 15 / 10 minutes, plus
   origin-, dest- and carrier-crossed variants, each smoothed toward the prior
   (`p = (sum_y + k·prior)/(n + k)`, k = 10–100). Train rows use 5-fold out-of-fold statistics; eval/holdout
   rows use full-train statistics. Every key also gets a `log1p(count)` feature so the model can learn how
   much to trust a sparse cell. `route_hour` alone: 0.7252 → 0.7359; 30-minute buckets: → 0.7507.
2. **Heavy regularization in place of raw capacity.** 30 unregularized trees scored 0.7141; 300 of them
   scored 0.7123. What worked was shallow + slow + heavily constrained (depth 4–7, lr 0.02–0.03,
   `min_child_weight` 40, `reg_lambda` 10, `subsample` 0.8, `colsample_bytree` 0.6): 0.7201 by experiment 9.
   The 2005→2006 shift punishes anything that memorizes.
3. **Dropping the raw high-cardinality categoricals.** Once the target encodings existed, leaving `Origin`,
   `Dest` and `UniqueCarrier` as native XGBoost categoricals only hurt — removing them took 0.7553 → 0.7580
   (−0.0027 when present). Low-cardinality `Month`/`DayofMonth`/`DayOfWeek` categoricals stayed.
4. **A randomized ensemble of XGBoost models.** 9 models over randomized depth (4–7), `min_child_weight`
   (20–60), `reg_lambda` (5–20) and sampling rates, averaged on probability. Best single model 0.7595 →
   0.7597 (3 models) → 0.7603 (5 models) → 0.7608 (9 models). Small, but the direction was monotone.
5. **`max_bin=512`**, which lets the continuous smoothed-encoding features split more precisely: 0.7597 → 0.7601.

## What did not help

1. **`Route` as a native XGBoost categorical** (~5.6k levels): 0.7146 → 0.7055. It memorizes 2005 route
   idiosyncrasies that do not survive into 2006. Smoothing-based encodings of the same information are fine.
2. **Calendar/date encodings** — day-of-year (numeric + cyclic), `date` (= month/day) target encoding and
   `month × DayOfWeek`: 0.7597 → 0.7542. Weather/event effects tied to specific 2005 dates don't transfer to
   the 2006 eval year at all.
3. **Micro-tuning of regularization and learning rate**: `min_child_weight` 15/80, `reg_lambda` 3/30,
   `colsample_bytree` 0.5/0.8, TE smoothing ×3, 10-fold instead of 5-fold OOF, 3000×0.01 vs 1500×0.02, and
   `max_bin=1024` were all flat-to-worse. Day-of-week interaction encodings and traffic-share ratio features
   were also flat. The configuration is at a local optimum; what was left was feature families, not knobs.

## With more budget

The remaining headroom looks small — every change after ~0.7560 moved the metric by ≤0.001, which is inside
the noise of a 100k-row eval set. I would spend it three ways. First, replace the flat smoothed encodings with
a proper *hierarchical* encoding (5-minute bucket with its 15-minute parent as prior, and so on up to the
route), so that fine time resolution borrows strength from coarse cells instead of being shrunk to the global
prior; 5-minute granularity is where the raw data still has unused structure. Second, address the train/predict
distribution mismatch in the encodings directly: fit them with out-of-time folds inside 2005 (by `Month`)
rather than random folds, so the model sees encodings degraded the same way the 2006 rows are, instead of the
random-fold OOF being essentially noise-free. Third, extend the ensemble past probability averaging — bag over
the encoding-seed itself (several OOF matrices with different fold splits) and use `booster="dart"`, which was
never tried because its cost grew past the 120s per-experiment limit. Beyond that the ceiling is the feature
set: with only scheduled `DepTime` and no `ArrTime`, tail number or weather, ~0.76 AUC looks close to the
information limit for this slice.
