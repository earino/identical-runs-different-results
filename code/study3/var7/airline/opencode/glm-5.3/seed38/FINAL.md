# Final report — airline dep_delayed_15min, AUC on hidden 2006 holdout

**Best Eval AUC: 0.7484** (experiments 1–17; baseline 0.7141). HEAD = `2932144`, validated
(`CONTRACT OK`, predict_proba path reproduces 0.7484 on eval with the target column removed).
17 experiments used; budget closed out on the 18,000 CPU-second limit.

## The changes that mattered most (0.7141 → 0.7484)

1. **Slow, shallow boosting on raw + cyclic time features** (exp 1→7): 1500 rounds, depth 4,
   lr 0.05 with explicit `dep_min` + sin/cos of day-time replaced the 30-tree depth-6 baseline.
   Within-2005 validation misled (2005→2006 shift), so robust low-variance configurations were
   chosen instead of early stopping (exp 2–4 all reverted).
2. **Carrier × time-of-day interaction as a native categorical** (exp 7→8→13→16):
   `UniqueCarrier@(dep_min // 24min)` — granularity tuned 60→30→20→24 minutes; 24-min bins
   (60 levels) was the sweet spot between specificity and year-transfer robustness.
3. **Time-of-day × distance interaction** (exp 9→14): `hh@(Distance octile)` (half-hour × 8
   distance quantile bins, train-fitted levels; unseen combos → missing).
4. **Feature pruning + column-subsampled seed bag** (exp 10→11): dropping route
   count/mean-distance and dow×hour columns that leaked 2005-specific noise; a bag of
   XGBClassifier seeds with colsample_bytree=0.6 (0.7418).
5. **Capacity×diversity rebalance** (exp 14→15): depth 5 + colsample_bytree 0.5 + distance
   octiles (0.7473), then bag of 7 seeds (0.7476) and 24-min carrier bins (0.7484).

## Things that did not help (all reverted or screened out)

- **Target/impact encoding** of Origin/Dest/route (exp 6): non-stationary across years — hurt badly.
- **Route/airport-level categorical interactions** (route cat, Origin@hour, Dest@hour, month@hour,
  carrier@dow, dow@halfhour): too sparse or 2005-specific; every variant was worse.
- **Early stopping on a 2005 holdout** and any single-model capacity increase (depth ≥6, higher lr):
  consistently overfit the 2005 split and transferred worse to 2006. DART, rank:pairwise,
  row subsampling, reg_lambda/mcw/max_bin tweaks, 10/15-minute interaction bins, and
  fold-based or mixed-config bags all screened flat or worse; a 9-member bag tied the 7-member bag.

## With more budget

I would attack the 2005→2006 covariate shift directly: fit a light domain-classifier
(2005 vs 2006) and reweight or importance-sample 2005 to look like 2006 before boosting, so the
interaction bins recalibrate to the deployment year. I would also try smoothing the carrier×time
and time×distance categorical statistics with hierarchical shrinkage (leaf → carrier/hour →
global) as numeric features alongside the raw cats, and a properly time-blocked CV
(month-ahead folds) to make keep/discard decisions without burning eval-slice trust.
Finally, a larger diverse bag (alternating depth 4/5, colsample 0.4–0.6) is worth one more pass —
the seed-bag trend was still (barely) positive at 9 members, and CPU, not insight, ended it.
