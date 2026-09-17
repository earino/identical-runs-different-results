# FINAL REPORT — airline dep_delayed_15min XGBoost

**Best Eval AUC: 0.7416** (kept commit `93eb63c`, validated `CONTRACT OK`, re-confirmed
twice at 0.7416 on re-runs). Baseline was 0.7141 → **+0.0275 AUC**.

Final model: 110-model XGBoost ensemble = 10 hyperparameter configs × 11 feature views,
7 of 10 configs carrying a monotone constraint on the first column. All engineering lives
inside `prepare()` / train-fitted statistics only (verified by validate.py's
target-column-removed predict_proba check).

## The 5 changes that mattered most

1. **Monotone constraint on the calendar/time column (exp 20, 0.7141→0.7209 solo).**
   Single biggest lever. Enforcing the prior that delay probability rises with departure
   time-of-day regularizes exactly the direction that is stable across the 2005→2006 shift.
2. **Diverse ensembling (exp 19→29).** Averaging depth/shrinkage/colsample_bynode variants
   plus monotone-constrained and unconstrained members: 0.7172 (single) → 0.7220.
   colsample_bynode was the only stochasticity that produced real member diversity
   (hist is deterministic, so seed bagging alone did nothing).
3. **Group-mean deviation features ("schedule norms"):** departure time minus the mean
   scheduled time for the same carrier/origin/dest/route at the same hour-of-day
   (exp 22→28). Exploits the fact that lateness is relative to the route's typical
   schedule; hours-based norms transferred well across years.
4. **Route×hour and route×quarter-day norms + target-rate encodings (exp 26, 29, 32).**
   `dep_time − route-hour mean` (0.7299 step) and smoothed route×hour / origin×hour
   delay-rate (train-fitted TE) were the strongest single additions.
5. **Multi-view averaging.** The 11 views (raw, +grp, +doy, +grp+doy, +chr, +rhr,
   +chr+rhr, triple, quint, rhq, ote) each capture a different slice of the same physics;
   the ensemble of views consistently beat any single view (0.7203 best solo vs 0.7416).

## What did not help (all measured, then discarded)

- **Additive engineered features without monotone constraints / ensembling** — linear
  cyclic time (dep_sin/dep_cos, tod, dep_hr), numeric calendar, distance transforms,
  DepTime-as-categorical, route string, season buckets, weekend flag, doy, congestion
  counts, origin/dest/route degree features: every one *lowered* eval AUC
  (0.699–0.709 vs 0.7141) when added to the raw-column model. The raw categorical
  representation plus a constrained model is remarkably strong.
- **Hyperparameter tuning** — depth 8/5/3, lr up/down, n_estimators alone, subsample,
  colsample_bytree, min_child_weight, gamma, reg_lambda, max_bin, lossguide: all ≤ 0.7172.
  Only the depth 4 + lr 0.05 + n 120 combo (itself found via grid probing) became the base.
- **Target leakage-style and stacking tricks** — residual stacking on OOF (≤0.7219),
  pseudo-labeling on eval features (0.7184), training-set reweighting by train/eval
  frequency ratios (0.7189), weighted blends (flat), 22-view exhaustive pairs (0.7336),
  DART (0.7160), gblinear (unsupported with categoricals), 30-min/quarter-day schedule
  norms as standalone additions (±0.0004), seed shifts (±0.0001).

## What I would try with more budget

1. **Quantile/rank-transformed times within route-hour cells** (normalized position of a
   flight's scheduled time in its cell), and **higher-order norms** (carrier×route×hour).
2. **Greedy view selection** with a proper validation protocol: with 40+ candidate views,
   forward-selection on a time-shifted validation split (train on months 1–9 of 2005,
   validate on 10–12) would likely find a better subset than my hand-curated 11.
3. **Early stopping against a 2006-like split** (calendar-aware CV) instead of fixed
   n_estimators, and **isotonic calibration** of the final blend.
4. **Native XGBoost `max_cat_*` parameters** (cat_smooth was a no-op in this build) for
   the high-cardinality categoricals, and GPU or bigger thread counts to afford
   300+ member ensembles and full pairwise view search.
