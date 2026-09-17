# FINAL.md — autoresearch XGBoost (airline delay, AUC)

## Result
- **Best Eval AUC: 0.7426** (experiment #40, commit `f3a0408`)
- Baseline Eval AUC: 0.7141 (experiment #1, commit `d33ab73`)

## Final configuration (`train.py` at HEAD)
- Features: raw `DepTime` plus `hour`/`minute`/`tod`; train-fitted frequency counts
  (`origin`, `dest`, `carrier`, `route`, `undirected route`, `origin×hour`, `dest×hour`,
  `carrier×hour`, `carrier×origin`) and their share ratios; holiday/window/weekend flags derived
  from `Month`/`DayofMonth`/`DayOfWeek` (no year needed).
- Categoricals handled natively by XGBoost (`enable_categorical=True`): Month, DayofMonth,
  DayOfWeek, UniqueCarrier, Origin, Dest.
- Model: `grow_policy="lossguide"`, `max_leaves=512`, `max_depth=0`, `n_estimators=1000`,
  `learning_rate=0.03`, `min_child_weight=5`, `reg_lambda=2.0`, `subsample=0.8`,
  `colsample_bytree=0.4`, `tree_method="hist"`.

## Changes that mattered most
1. **Switching to leaf-wise growth (`lossguide`, 128–512 leaves)** instead of fixed depth.
   This alone moved 0.7278 → 0.7297 and unlocked the later capacity gains.
2. **Tuning down regularization for lossguide** — `min_child_weight` 150→5 and `reg_lambda`
   80→2 gave the largest single gains (0.7307 → 0.7398). With limited leaves the model was
   underfit, not overfit.
3. **Time decomposition** (`hour`, `minute`, `tod`): the raw `hhmm` integer obscures the very
   strong, monotonic time-of-day delay signal (target rate rises from ~0.04 at 05:00 to ~0.82 at 23:00).
4. **Train-fitted frequency/share features** (origin/dest/carrier/route counts and congestion
   ratios). These are stable across 2005→2006 and added ~0.003.
5. **Holiday/seasonality flags** computed year-free from month/day/day-of-week (Thanksgiving,
   Christmas–New Year, July 4, Memorial/Labor Day windows): 0.7297 → 0.7307.
6. **Increasing leaves to 512 at low regularization** on the final run: 0.7398 → 0.7426.

## Things that did NOT help
1. **More trees / lower learning rate** (2000 trees at lr 0.02) — consistently worse; 1000 @ 0.03 was best.
2. **Ensembling** — both a mixed-depth 4-model ensemble (0.7227) and a homogeneous depth-8
   seed ensemble (0.7252) underperformed the best single model, so the extra complexity was dropped.
3. **Smoothed out-of-fold target encoding** of origin/dest/route/carrier/hour groups — neutral to
   slightly negative; XGBoost's native categorical splits plus the count features already capture it.
4. **Dropping Origin/Dest** (correlation of their year-to-year delay rates is low, ~0.38) — much
   worse (0.7004), because airport congestion is a real, transferable signal.
5. **Day-of-year / cyclical encodings**, `gamma`, `max_cat_threshold`, and extra
   Origin×DayOfWeek/Month congestion counts — all neutral or negative.

## What I would try with more budget
The regularization sweep was still improving at the budget limit (0.7342 → 0.7370 → 0.7377 →
0.7391 → 0.7398 → 0.7426), so the first move would be to continue it: `min_child_weight` and
`reg_lambda` toward 1–2 with `max_leaves` 1024–2048, and a slightly lower `colsample_bytree`.
Beyond tuning, the highest-value direction is target encoding of the *stable, low-cardinality*
interactions (carrier, day-of-week, carrier×hour) with proper out-of-fold folds, rather than the
high-cardinality airport/route groups that do not transfer across years. A weighted average of
the best lossguide models (weighting by validation AUC instead of uniform) is also worth another
attempt now that the base model is stronger. Finally, a time-based internal validation split
(early-2005 vs late-2005) would make the many cheap tuning decisions less dependent on a single
100k-row `eval.csv` and reduce the risk of fitting eval noise.
