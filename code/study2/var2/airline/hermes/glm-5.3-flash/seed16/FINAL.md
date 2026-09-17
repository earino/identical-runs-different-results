# FINAL — autoresearch XGBoost (airline)

Best Eval AUC: **0.7377** (baseline 0.7141, +0.0236). HEAD = `3aeae4f DepHour categorical`.

## Changes that mattered most

1. **Real hyperparameters + native categorical splits** (exp 4, 0.7141 → 0.7319): max_depth 9,
   min_child_weight 20, subsample 0.8, colsample 0.6, lr 0.05, ~1200 trees with early stopping
   (patience 50) on the last 20k rows of 2005 as internal validation, refit on all train with the
   best iteration count. Categoricals handled natively by XGBoost (enable_categorical) instead of
   the baseline's label-index proxy categories.
2. **Feature engineering of DepTime/Distance** (in exp 4/13): DepHour (hh + mm/60), cyclic
   sin/cos, raw hhmm, log-distance, short-haul and time-of-day bucket flags.
3. **Frequency-count features for Origin/Dest/carrier + RouteFreq** (exp 13, 0.7319 → 0.7342):
   value counts fitted on train, mapped in prepare(); unseen levels become NaN.
4. **Diverse parallel ensemble** (exp 27, 0.7342 → 0.7355; plus 33/34 tie-breakers): 9 XGBoost
   members, depths 6–13, colsample 0.4–0.7, subsample 0.7–0.9, fitted concurrently with
   ThreadPoolExecutor (2 threads each) and rank-averaged (rank-average beats prob-average slightly).
5. **Derived categoricals: DistCat (distance buckets) + hour harmonics** (exp 39, → 0.7374) and
   **DepHourCat (hour as a 24-level native categorical)** (exp 40, → 0.7377). Discrete bucket
   representations of the two numeric columns were the only feature family that still moved eval
   AUC late in the budget.

## Things that did not help

1. **Target encoding** in every variant — leave-one-out (0.6537, catastrophic: ES collapsed to 12
   trees because LOO injects −y-correlated noise into train features) and clean 5-fold OOF TE
   (0.7286). Native categorical splits already capture per-level effects.
2. **Volume/interaction count features**: pairwise freq (carrier/origin/dest/route × hour) scored
   0.7200 — schedule-volume patterns did not transfer from 2005 to 2006, despite val AUC jumping to
   0.81 (in-domain overfit). DOW×hour-block crossed cat was neutral (0.7352) and too slow.
3. **Tree-count and learning-rate moves**: 2000 trees (0.7330), lr 0.03/patience 100 (0.7342),
   12 members × lr 0.07 × 800 trees (0.7355, tie). Also no gain: row-bagged 80% members (0.7324),
   lossguide/64-leaf trees (0.7350), heavier leaf regularization (0.7327), Route as a 4198-level
   native categorical (0.7330), DayOfWeek re-encoded as string category (0.7312).

## With more budget

- A **50-fold-expanding-window OOF target/rank encoding** with year-aware folds, or better: per-hour
  and per-carrier historical delay-rate tables from *external-year* structure (train on 2005 Q1–Q3,
  encode with 2005 Q4 → capture the 2005→2006 drift directly). Every naive leak-free encoding I tried
  underperformed native categoricals, but a drift-aware one has not been tried.
- **Quantile-binned DepTime** (bins learned from train quantiles) as another categorical, plus
  interaction DistCat×DepHourCat as a crossed categorical (both helped individually).
- **Deeper diversity**: ensemble members trained on different feature subsets (column-bagged at the
  feature-family level: one member without freq features, one without harmonics, etc.).
- Longer patience/lower lr only if wall clock allows; the 120s cap was the binding constraint on
  ensemble size (9 members ≈ 100s at lr 0.05).

## Notes for the scorer

- All feature engineering lives inside `prepare(df)`; fitted statistics (categorical levels, freq
  maps, best iteration count, model weights) are module state fitted on training data only.
- Validation: `./validate.sh` → `CONTRACT OK`, eval AUC via `predict_proba` (target dropped) = 0.7377.
