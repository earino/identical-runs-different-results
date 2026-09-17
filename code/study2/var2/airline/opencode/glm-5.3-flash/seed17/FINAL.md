# FINAL REPORT

**Best Eval AUC: 0.7339** (commit 18ed2c3, experiment #35; validated `CONTRACT OK`, predict_proba reproduces 0.7339 with the target column removed).

## Changes that mattered most

1. **carrier×hour categorical interaction** (exp 33, +0.0068 → 0.7307): carrier-specific hourly delay
   patterns (schedules, hub banks) are stable from 2005 to 2006, unlike route-level noise. The single
   biggest win of the run.
2. **colsample_bytree tuning on the richer feature set** (exp 35, → 0.7339): once carrier×hour and
   dow×hour were in, relaxing feature bagging from 0.3 to 0.4 helped. Earlier, colsample 0.3 + many
   shallow trees was the key that made depth 4→6 and lr 0.1→0.02 viable (exp 20–29, 0.7202→0.7231).
3. **Slow learning rate with many trees** (exp 17–18, 29): lr 0.1 → 0.03 → 0.02 with 400→1800 trees at
   depth 4–6 steadily gained (+0.004 combined) once colsample controlled variance.
4. **dow×hour interaction** (exp 32, +0.0006): same year-stable-pattern logic as carrier×hour.
5. **Cheap derived features** (exp 12, +0.0016): int-parsed Month/DayofMonth/DayOfWeek, dep_hour =
   (DepTime//100)%24, dep_min — with route deliberately excluded.

## Things that did NOT help

1. **Target encoding** of carrier/origin/dest/route (exp 15, 0.7090): 2005 delay rates don't transfer.
2. **Route (Origin_Dest) categorical** and **carrier×dow×hour 3-way** (exp 11/37): high-cardinality
   pair/triple features memorize year-specific noise.
3. **Seed ensembling** (exp 16): hist XGBoost at subsample=1.0 is near-deterministic; zero diversity.
   Likewise subsample 0.8 (exp 25) and min_child_weight 10 (exp 40) — row/leaf regularization hurt.

## Theory of the data

The 2005→2006 time shift dominates: in-year validation AUC (~0.75) massively overstates next-year AUC,
and early stopping on an in-year split selects ~360 trees when ~100–1800 regularized trees is what
actually transfers. Everything that worked reduced variance (colsample, shallow-ish trees, slow lr) or
added *entity×time-of-day* structure (carrier×hour) that persists across years.

## With more budget

- Origin×hour / Dest×hour as *smoothed numeric* (not raw categorical) airport-hour congestion indices.
- Quantile/binned DepTime and Distance features; per-carrier hour-of-day target-free priors.
- A small bagged ensemble across colsample ∈ {0.3, 0.4} and depth ∈ {5, 6} (real diversity, unlike seeds).
- Careful per-feature-group colsample (more weight on the year-stable interactions).
