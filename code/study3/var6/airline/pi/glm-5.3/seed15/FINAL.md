# FINAL — autoresearch XGBoost (airline delay, AUC)

**Best Eval AUC: 0.7527** (baseline 0.7141, +0.039) — commit `437c725`, experiment #40 of 40.
Contract verified: `./validate.sh` → `CONTRACT OK` (train.py runs in ~112s, `predict_proba` reproduces 0.7527).

## Final architecture

12-member XGBoost ensemble, prediction = mean of member probabilities. All members: `binary:logistic`,
hist, lr 0.03, ~100 boosting rounds, min_child_weight 1, reg_lambda 1, subsample 1.0, max_depth 18,
colsample_bytree 0.4 (variants 0.5), 4 threads. Features (built in `prepare()`, encoders fit on train only):
Month/DayofMonth/DayOfWeek parsed to ints from `c-<n>` strings, hour, minute, DepTime, Distance,
categoricals (UniqueCarrier/Origin/Dest), smoothed target encodings (empirical-Bayes, m=100) of
origin/dest/carrier; group variants add te_hour+te_dow ("time"), route/airport counts ("vol"),
and minute-of-day ("mod").

## The 5 changes that mattered most

1. **Very deep trees + heavy column subsampling + very few rounds** (max_depth 18, colsample_bytree 0.4,
   100 rounds): the single biggest lever. Under colsample 0.4, scaling depth 6→18 kept helping under the
   2005→2006 shift (member AUC 0.726 → 0.748). The regime behaves like a small forest of deep conditional
   kernels; regularization attempts (min_child_weight, reg_lambda, fewer bins) all hurt it.
2. **Ensembling 12 decorrelated members** (seeds × colsample 0.4/0.5 × FE-group variants): mean adds
   +0.004–0.005 over the best single member.
3. **Smoothed target encoding (EB, m=100) of Origin/Dest/Carrier** — structural delay propensity that
   transfers across years.
4. **Minute-of-day feature** (`hour*60+minute`, replaces the discontinuous hhmm integer with a clean
   time axis): +0.002 on members, final ensemble 0.7520 → 0.7527. Found on the last experiment.
5. **Fitting rounds to the deep regime** (~100 rounds; more rounds hurt: 120→0.740, 140→0.743) and
   integer date parsing + hour/minute features.

## What did not help (all reverted)

- Route-pair categoricals, interaction target encodings (carrier×hour, origin×route), month TE —
  all hurt under the year shift.
- Early stopping on a 2005 validation split: internal val AUC keeps climbing while 2006 AUC declines —
  the year shift makes 2005-based stopping actively harmful.
- Monotone constraints, recency (time-decay) instance weights at depth, high learning rates (0.1+),
  subsample < 1 at depth 18, depth 20+, rounds > 100, min_child_weight > 1, reg_lambda > 1,
  max_bin 128, lossguide growth, colsample_bynode, num_parallel_tree (RF mode), DART (neutral at depth).
- Feature-drop members (no Distance / no DepTime / no TE): each is much weaker (0.72–0.74) and drags the mean.
- Blending rule: mean-of-probabilities = mean-logit = mean-rank = median. Weighted/dropped-member means
  all ≤ simple mean.

## With more budget

The sharp optimum at (depth 18, colsample 0.4, ~100 rounds) suggests the winning model is a
boosted micro-forest. I would (a) add minute-of-day to *all* members and re-run the member lottery,
exploring more "clean" time encodings (sin/cos of day-minute, departure banks); (b) average 20+ members
of the best config to cancel seed-lottery variance on the hidden holdout (individual seeds vary ±0.004);
(c) tune an explicit random-forest-mode xgboost (num_parallel_tree with lr scaled by 1/ntree, per-node
feature sampling) at depth 18; (d) out-of-fold stacking of the member predictions with a tiny logistic
calibrator; (e) verify the deep-forest regime transfers by simulating year-shifts within 2005
(train Jan–Oct → validate Nov–Dec) to reduce reliance on the 100k eval slice.
