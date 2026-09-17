# Final Report — airline delay XGBoost

**Best Eval AUC: 0.7306** (HEAD = `53bfb97`, validated `CONTRACT OK`, AUC reproduced via `predict_proba` on a target-dropped frame).

## Progression
| step | change | AUC |
|---|---|---|
| baseline | 30 trees, raw string cats | 0.7141 |
| time features | cyclical dep time (dep_min/sin/cos), parsed Month/DOM/DOW ints, log distance | 0.7178 |
| regularized forest | 400 trees, lr .06, min_child_weight 20, subsample/colsample .8 | 0.7211 |
| ensembling | 3-seed mean → 5 diverse → 13-member diverse | 0.7215 → 0.7224 → 0.7239 |
| **L1 regularization** | reg_alpha=1.0 on all members | 0.7263 |
| **reg_alpha=3.0** | same ensemble, stronger L1 | **0.7306** |

## Final model
13-member XGBoost ensemble (hist, enable_categorical) over features
{dep_min, dep_sin, dep_cos, distance_log, Month, DayofMonth, DayOfWeek, DepTime,
Distance, UniqueCarrier, Origin, Dest}, all engineered inside `prepare()` with
categorical levels fitted on train only. Members: 3×(d6/400/lr.06),
2×(d8/500/lr.06/λ2), 2×(d6/800/lr.03), 2×(d6/600/sub.6/col.5),
2×(d8/800/lr.03), 2×(d6/800/sub.5/col.4); mean of predicted probabilities.

## Changes that mattered most
1. **Cyclical time-of-day features** from DepTime (dep_min % 1440, sin/cos): +0.004 alone; DepTime wraps past 2400 and raw hhmm misrepresents the midnight boundary.
2. **Regularized bigger forest** (400 trees @ lr .06 + min_child_weight 20 + row/col subsampling): +0.003 over the tiny baseline forest.
3. **reg_alpha (L1) = 3.0 across the ensemble**: +0.0067 total (0.7239→0.7306) — the single biggest gain. L1 pruning of weak splits suits this redundant/noisy categorical-heavy feature set.
4. **Ensemble diversity** (seeds × depth × learning-rate × subsample axes, 13 members): +0.0028 cumulative; diversity axes beat hyperparameter micro-tuning.
5. Proper integer parsing of `c-<n>` date columns and log1p(Distance).

## What did not help
- **Target encoding** of carrier/origin/dest (any smoothing; smoothed and cross-fit variants): consistently ≤ baseline — XGBoost's native categorical splits already capture the same signal.
- Hand-built interaction/binned features (month×hour, dow×hour, distance bins, schedule-tightness flags, airport traffic counts, max_bin=512, distance quantile bins): all neutral-to-negative; depth-6 trees find these crosses themselves.
- Early stopping on a time-ordered tail split (loses 20% of training data and stopped at ~500 trees while full-data 400 was already better), lossguided grow policy, median aggregation, cat_smooth, categorical-encoding of Month/DOW (int ordering is better), dropping raw DepTime, 17-member ensembles (dilution).

## With more budget
Sweep reg_alpha per-member (the 1.0→3.0 jump suggests the optimum may be beyond 3.0, possibly per-config), couple L1 with lower learning rates / more trees per member, and explore a stacked two-level combination (logistic on member outputs is not allowed — only XGBoost learners — so a shallow XGBoost meta-learner over member predictions, cross-fitted). Also worth one more attempt at cross-fitted TE now that strong L1 is in place, since L1 may suppress the leakage-driven noise that sank it before.
