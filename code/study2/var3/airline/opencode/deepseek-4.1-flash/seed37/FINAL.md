# Final report — airline departure-delay XGBoost

**Best Eval AUC: 0.7329** (3-seed XGBoost ensemble, commit `dc7e2c2`).
Baseline was 0.7141. Budget fully used: 40/40 experiments, ~5.8k of 18k CPU-seconds.

## Contract
`./validate.sh` → `CONTRACT OK`, eval AUC via `predict_proba` (target removed) = 0.7329.
All feature engineering lives inside `prepare(df)`; every mapping is fitted on `data/train.csv` only.

## Changes that mattered most
1. **Ordinal calendar + DepTime decomposition** (0.7141 → 0.7177). Month/DayofMonth/DayOfWeek
   arrive as `c-<n>` strings; converting to ordinal and parsing `DepTime` hhmm into hour, minute and
   continuous minutes-since-midnight was the single most valuable feature change.
2. **`reg_alpha=5` (L1 on leaf weights)** (0.7283 → 0.7315, the largest single jump). With a
   time-shifted eval, sparsity on leaf weights clearly helped generalization; `alpha=20` over-regularized
   (0.7242) and `alpha=3` matched `alpha=5`.
3. **Strong sampling/weight regularization**: `subsample=0.5`, `colsample_bytree=0.4`,
   `min_child_weight=5`, `reg_lambda=20`. Pushing these up monotonically raised eval AUC
   (e.g. mcw 30 → 10 → 5 kept improving).
4. **Deeper trees once regularized**: with the regularization above, increasing `max_depth` from 6 to 12
   helped (0.7254 → 0.7281); depth 14 added nothing. `lossguide`/`max_leaves` timed out.
5. **3-seed averaging** (`[42, 7, 2024]`, 2500 trees, lr 0.012): +0.0009 over a single model; 5 seeds
   gave no further gain, and a depth-diverse ensemble matched the seed ensemble.

## Things that did not help
- **Categorical interaction features** (`Route = Origin_Dest`) and **frequency encodings** of
  Origin/Dest/Carrier/Route — both consistently hurt under the 2005→2006 time shift.
- **Target encoding** (smoothed, O/D/carrier/route) — large drop (0.7175); leaking/past-year delay
  propensities do not transfer.
- **Cyclic calendar features** (sin/cos month & weekday, day-of-year), `max_cat_threshold=256`,
  raw-`DepTime` ablation, and `subsample=0.4` — all neutral-to-negative.

## What I would try with more budget
The plateau at ~0.733 suggests the raw feature set is nearly exhausted. I would invest in
(i) careful out-of-fold target/statistic encodings for airport and carrier that are validated
strictly on future rows (walk-forward by day within 2005) rather than the whole-year mapping I tried,
(ii) explicit congestion/seasonality aggregates computed over a longer history (e.g. per Origin×hour×month
delay base rates) with heavy smoothing, and (iii) a proper future-split validation harness so that
keep/discard decisions better track the hidden 2006-holdout instead of the 100k eval slice. On the model
side, a bagged set of 8–10 seeds would likely add another ~0.0005 robustness at the cost of runtime.
