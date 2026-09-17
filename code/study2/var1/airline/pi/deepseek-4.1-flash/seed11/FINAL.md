# Final report — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7370** (experiment #40, commit `799e2cb`). Baseline was 0.7141.

## Setup that won
A bagged ensemble of 20 `XGBClassifier` models (`n_estimators=600`, `max_depth=5`,
`learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`,
`reg_lambda=2.0`, `max_cat_to_onehot=16`, `tree_method="hist"`, `enable_categorical=True`),
averaging `predict_proba`. All feature engineering lives inside `prepare(df)`, so the
contract's `predict_proba` reproduces it on unseen rows.

## The 3–5 changes that mattered most
1. **`UniqueCarrier` × half-hour-of-day categorical interaction** (`carrier_hour`), built from
   `DepTime`. This was the single largest gain: +0.0072 at the hour granularity and +0.0075 more
   when refined to 30-minute buckets (0.7189 → 0.7340). Carrier delay behavior is strongly
   time-of-day dependent and stable across years.
2. **Bagged shallow ensemble instead of one deeper model.** A 20-seed average of depth-4 models
   gave +0.0016 over the single model (0.7145 → 0.7161), and the average was robust to the
   2005→2006 distribution shift that punished individual high-capacity models.
3. **Day-of-year seasonality** (`doy`, `doy_sin`, `doy_cos`) added +0.0014 (0.7161 → 0.7175);
   month+day interactions are otherwise hard for shallow trees to form.
4. **Lower learning rate + more trees at depth 5** once the interaction existed: lr 0.05/400 trees
   (0.7340) → lr 0.03/600 trees at depth 5 (0.7368), and `min_child_weight=5` finished at 0.7370.
5. **`UniqueCarrier` × distance-bin interaction** (small, +0.0004) and `max_cat_to_onehot=16`
   to one-hot the low-cardinality calendar/carrier fields.

## What did NOT help
- **Extra model capacity without structure**: 300 trees at lr 0.05 (0.7119), depth 6 (0.7343),
  40 seeds vs 20 (no change), and internal early stopping that picked 382 trees on a 2005
  validation split but generalized worse to 2006 (0.7108). The year shift strongly favors
  simple, well-regularized models.
- **High-cardinality geographic interactions**: `route` (Origin×Dest) repeatedly collapsed the
  score (0.7024, 0.7041); `origin_hour` (0.7212) and `origin_dist` (0.7362) also hurt.
- **Target/frequency encodings** of carrier/origin/dest: neutral-to-negative (+0.0001 or worse),
  so they were dropped for simplicity. So were plain `hour`/`tod`, `carrier_dow`/`carrier_mon`,
  and 15-minute buckets (too fine, 0.7308).

## What I would try with more budget
The dominant signal is the carrier × time-of-day schedule effect; the natural next step is richer
*time-of-day × carrier* structure, e.g. a hierarchical or target-encoded carrier-time surface, or
a decomposition of `DepTime` into hour-of-day plus a per-carrier "delay accumulation curve". I
would also explore destination-airport-specific time effects with heavy smoothing (target encoding
with out-of-fold maps) rather than raw high-cardinality categoricals, and try stacking a
second-level XGBoost on out-of-fold predictions of the base ensemble. Finally, since the
2005→2006 shift punishes capacity, an explicit domain-adaptation/sample-weighting scheme keyed on
distribution-stable features could let a larger model be used safely.
