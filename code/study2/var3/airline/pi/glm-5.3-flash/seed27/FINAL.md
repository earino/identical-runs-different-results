# Final Report — airline delay XGBoost (autoresearch harness)

**Best Eval AUC: 0.7289** (40/40 experiments used; baseline was 0.7141).

Final model: an ensemble of **7 XGBoost models** (depths 6/7/7/8/8/9/9, colsample 0.8/0.8/0.6/0.8/0.6/0.8/0.6,
tree counts 1560/1440/1440/1320/1320/1200/1200 at lr 0.05, max_bin 512, subsample 0.85), each trained on the
full 100k-row training set, pooled by **logit-mean**. All feature engineering lives inside `prepare(df)`
(shared once per predict call), so `predict_proba` reproduces the training-time transform on unseen data
(validated: CONTRACT OK, eval AUC via `predict_proba` = 0.7289 with the target column removed).

## Changes that mattered most

1. **Real feature engineering from the raw strings** (exp 2, 0.7141→0.7208): Month/DayofMonth/DayOfWeek
   parsed to ints, DepTime decomposed into hour + cyclical sin/cos of minutes-since-midnight, Distance +
   log1p. The time-of-day features are the strongest signal (evening flights cascade delays).
2. **Full-data refit at an early-stopping-tuned tree count** (exp 5, →0.7234): fit with ES on a 90/10 split,
   then retrain on all 100k rows with ~1.1× the best iteration. Uses 10% more data than ES-fit alone.
3. **Ensembling with diversity** (exp 11/12/14/35, →0.7289): seed bagging (+0.0017), then depth diversity
   (+0.0020), then a 7th member with a new colsample axis (+0.0004). The single most reliable lever.
4. **Low-cardinality target encoding + counts** (exp 4/5, ~+0.0026 with refit): smoothed TE (m=30) for
   Origin/Dest/UniqueCarrier plus frequency counts — the only TE family that transferred to 2006.
5. **Logit-mean pooling + max_bin 512** (exp 16, +0.0010) and **safe target-free aggregates**
   (avg distance by carrier/origin/dest, distance ratio, weekend flag; exp 18, +0.0001).

## What did not help (all reverted)

1. **High-cardinality / interaction target encodings** (route, hour, carrier×hour, carrier×month; exp 3/8)
   and even their **out-of-fold** version (exp 9): val AUC inside 2005 reached 0.794 while eval fell to
   ~0.719 — these features encode year-specific patterns that don't survive the 2005→2006 shift.
2. **Capacity/regularization knobs**: depth 10 + min_child_weight 10 (exp 6), lr 0.03 with 3000 trees
   (exp 7), gamma 0.5 (exp 24) and 0.2 (exp 40), subsample 0.75 (exp 38), trees ×1.15 (exp 32) — all
   slightly worse; the gap to holdout is distribution shift, not variance.
3. **Categorical re-encodings**: month/dow/hour as native categoricals (exp 15), route as a native
   categorical (exp 17 — timed out), rank-mean pooling (exp 31), row-bagged members on 85% subsets
   (exp 25), month/dom/dow TE (exp 29, clearly harmful).

## With more budget

I would (a) build a proper nested-CV evaluation *inside 2005* to tune member weights without touching
eval.csv, (b) try per-member feature-subset ensembling more aggressively (one raw-only member tied, so
more granular subsets might do better), (c) probe `grow_policy=lossguide` and monotonic constraints on the
time features, and (d) revisit leakage-safe interactions (e.g., origin×hour as native cat in a faster
one-hot form). I would also test whether averaging two full 6-member ensembles trained with different
global seeds beats single-ensemble logit-mean pooling.
