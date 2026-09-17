# Final Report — airline delay (AUC)

**Best Eval AUC: 0.7352** (40/40 experiments used, single XGBoost model).
Baseline was 0.7141; the kept best `train.py` is commit `8a70eb4`.

## Changes that mattered most

1. **Calendar ordinals + time-of-day.** Parsing `Month` / `DayofMonth` / `DayOfWeek`
   from `c-<n>` strings into integers (instead of treating them as unordered
   categoricals) and adding `dep_hour`: 0.7141 → 0.7204. Imposing the ordinal /
   cyclic structure generalized across the 2005→2006 shift much better than
   letting the tree split arbitrary category subsets.
2. **Deep trees under strong regularization.** With `min_child_weight=30`,
   `reg_lambda=20`, `subsample=0.8`, `colsample_bytree=0.5`, increasing
   `max_depth` from 5 to 15–20 improved monotonically (0.7228 → 0.7321). The
   signal is interaction-heavy (time-of-day × airport × carrier); shallow trees
   underfit.
3. **Focused smoothed target encoding** of `UniqueCarrier`, `Origin`, `Dest`
   (smoothed P(delay), width 100, fit on train only): 0.7306 → 0.7320.
4. **Frequency features** for origin / dest / carrier / route (train-relative
   popularity): 0.7228 → 0.7235 — a stable, year-shift-robust signal.
5. **Regularization rebalance** at the end: `reg_alpha=5`, lowering `reg_lambda`
   to 5, and `min_child_weight=20`: 0.7321 → 0.7352.

## Things that did NOT help (reverted)

- **High-cardinality interactions**: `Route` as a categorical (0.7104), route /
  hour-combination target encodings (0.7229), and broad target encoding
  (0.7062) all overfit the 2005→2006 shift badly.
- **A 5-seed XGBoost ensemble** tied the single best model (0.7319 vs 0.7320),
  so error is dominated by distribution shift, not seed variance.
- **More trees** (800 vs 400) hurt (0.7295); **dropping raw `DepTime`** hurt
  (0.7263); `lossguide` growth tied (0.7320); temporal target rates hurt (0.7295).

## With more budget

The deciding factor is the 2005→2006 temporal shift, not model capacity. I would
build a **time-ordered validation split inside 2005** to select hyperparameters
for forward robustness instead of trusting a 100k eval slice, use **out-of-fold
target encoding** with the fold structure to remove its optimistic bias, try
**importance weighting / domain adaptation** between the 2005 and 2006 feature
distributions, and add **holiday indicators** and airport-level congestion
statistics. A stacking ensemble of the deep model with a deliberately shallower,
differently-encoded model may also recover the decorrelation the seed ensemble
lacked.
