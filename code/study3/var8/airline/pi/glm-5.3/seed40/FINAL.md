# FINAL — airline dep-delay (XGBoost, 2005 train → 2006 eval/holdout)

**Best Eval AUC: 0.7552** (baseline: 0.7141). Final model validated (`CONTRACT OK`): a 6-seed
bagged XGBoost ensemble (`XGBClassifier`, hist, `enable_categorical`), 120 trees, depth 20,
lr 0.03, subsample 0.8, colsample 0.5, averaging predict_proba across seeds.

## The changes that mattered most
1. **hour-of-day as a native categorical** (DepTime // 100) instead of only raw hhmm ints
   (+0.004). Delay rate is strongly non-monotonic in hour (morning low, evening high) and
   categorical splits capture arbitrary hour groupings.
2. **carrier × hour interaction as a lumped categorical** (levels with <50 train flights →
   NaN) (+0.006 over hour-cat alone). This was the single biggest feature win; lumping away
   rare late-night carrier combos added another +0.001 by removing 2005-specific noise.
3. **origin × hour interaction, lumped (≥50 train flights)** (+0.002).
4. **Deep trees + heavy shrinkage + subsampling instead of many shallow trees**: depth 16–20,
   lr 0.03, subsample 0.8, colsample 0.5, n_estimators ~120–250. The 2005→2006 time shift
   makes boosting "long" overfit year-specific noise: eval AUC fell monotonically with more
   trees in the shallow regime, while deep+subsampled+few trees kept improving (0.727 → 0.753).
5. **Seed-bagging the ensemble** (3→6 seeds) and **dropping day-of-month** (calendar noise):
   small but consistent; averaging over seeds transfers to the hidden holdout as pure
   variance reduction.

## What did NOT help (all reverted)
- **Route (Origin+Dest) categorical** — catastrophic (−0.01): 4198 levels of 2005-specific
  route noise. **dest×hour** and **hour×dow**, **carrier×month**, **carrier×origin**,
  **origin×4h-block**, **carrier×30-min-slot** interactions: all flat or worse.
- **Target encodings** (Origin/Dest/carrier/route/hour, smoothed or not) — worse on 2006;
  also DART boosting, lossguide ensemble mix, per-model hyperparameter jitter,
  reg_lambda/min_child_weight increases, max_bin changes, sin/cos and day-of-year features,
  c-string dates as categoricals (int parsing is better), and early stopping on a 2005
  internal split (it happily drives tree count deep into time-shift overfitting).

## What I would try with more budget
The dominant obstacle is the 2005→2006 distribution shift: every lever that reduces
year-specific variance won. With more budget I would (a) build a proper uncertainty harness
around the tree-count/depth plateau (bootstrap the eval split to pick configs that are best
in expectation rather than on point estimate), (b) systematically mine interaction
categoricals of the form (stable entity) × (time-of-day bucket) with count-based lumping
thresholds chosen by held-out-year stability rather than eval AUC, (c) try a two-level
stacked XGBoost whose meta-learner combines members trained on feature-disjoint views
(hour-family vs route-family), and (d) quantify per-feature transfer with a 2005-internal
temporal split (last months of 2005 as pseudo-2006) to select features that are stable
across time before touching the real eval set.
