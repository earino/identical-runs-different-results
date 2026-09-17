# FINAL — airline dep_delayed_15min, XGBoost autoresearch

Best Eval AUC: **0.7258** (baseline 0.7141), reached at commit 16a2d5f ("exp34: 24-member diverse ensemble"),
verified by `./validate.sh` → `CONTRACT OK`, AUC via `predict_proba` on the target-dropped eval frame: 0.7258.

## The changes that mattered most

1. **Departure-time decomposition** (DepHour, DepMinute, DepMins, DepSin/DepCos, DistPerMin): +0.0026 over
   baseline in one step — time-of-day is the single strongest signal for departure delay.
2. **Shallow trees (depth 4) + moderate round count (~400 @ lr 0.1)** instead of depth 6: depth 4 was the
   sweet spot on this 100k-row, time-shifted dataset; depth 5-8 always scored worse.
3. **Feature/level subsampling (colsample_bytree 0.6, colsample_bylevel 0.8, max_bin 512)**: ~+0.0010,
   cumulatively the difference between 0.7175 and 0.7200 territory.
4. **Seed- and hyperparameter-diverse bagged ensemble of 24 XGBoost models** (varied colsample_bytree
   0.40-0.70, depth 3/4/5, different seeds), averaged probabilities: the single biggest jump, +0.0054 over
   the best single model (0.7200 -> 0.7254 and up to 0.7258 with 24 members).
5. **DayOfWeek/Carrier x time-of-day-block interaction categoricals**: +0.0002 kept (0.7173 -> 0.7175), small
   but robust (block-level, not row-level, so it generalizes across the year shift).

## Three things that did NOT help

1. **Target encoding (plain and out-of-fold), including route/hour variants**: 0.6977-0.7138, always worse
   than no TE. Delay rates per carrier/airport shift too much between 2005 (train) and 2006 (eval/holdout);
   target-derived encodings bake in the shift.
2. **Route/airport/carrier aggregate features from train stats** (route counts, mean distances): 0.7146 vs
   0.7164 baseline-at-the-time — redundant with the raw categoricals the trees already split on.
3. **Row subsampling (subsample 0.8) and heavier L2/min_child_weight regularization**: 0.7109-0.7155.
   Consistently hurt at this sample size; also 800+ rounds overfit (0.7165 vs 0.7198 at 400).

## What I would try with more budget

The model is variance-limited, not feature-limited: every gain after the first hour came from averaging more
diverse XGBoost members, and member count scaling (7 -> 12 -> 18 -> 24) was still yielding small gains
(+0.0003-0.0006 per step) when the experiment budget ran out. With more budget I would (a) scale the ensemble
to 100+ members using the CPU headroom (~1,400 of 18,000 CPU-s used; each 24-member run costs ~90-150 CPU-s),
(b) add per-member bootstrap resampling of the 100k training rows (a diversity axis I only tried as a global
subsample, never as bagging within the ensemble), (c) try one-hot-free ordinal encoding of Month/DayofMonth
as integers rather than c-strings to give trees natural split order, and (d) tune learning rate down to ~0.05
with proportionally more rounds per member, since the ensemble average favors lower-variance members. I would
NOT revisit target encoding — it failed three separate ways (plain, OOF, per-route), and the mechanism of
failure (train/eval year shift) is structural, not fixable by smoothing.
