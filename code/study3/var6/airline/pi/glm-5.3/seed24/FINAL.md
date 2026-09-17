# FINAL — airline delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7638** (experiments #24/#25, commit 7c7b254; baseline was 0.7141, +0.0497 total).
Final model: 6-member XGBoost ensemble = 3 feature-set variants × 2 seeds, mean of probabilities.
Each member: `XGBClassifier(n_estimators=5000, learning_rate=0.03, max_depth=10, min_child_weight=3,
subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc", early_stopping_rounds=120,
tree_method="hist", enable_categorical=True)`. Runtime ≈ 105 s; `validate.sh` prints `CONTRACT OK`.

## The 5 changes that mattered most

1. **Out-of-fold target encoding of categorical interactions** (instead of native categoricals):
   carrier/origin/dest/route plus hour-bucket interactions (carrier×hour, origin×hour, route×hour,
   dow×hour, carrier×dow, …), m-smoothed, OOF-fitted on 4 folds per seed. 0.7141 → ~0.745.
2. **Time-of-day feature family**: hour/minute/tod numerics + 30-minute-bucket interaction TEs
   (route×hm, carrier×hm, origin×hm, dow×hm), train-frequency counts/congestion features for the same
   keys, and tod-deviation-from-group-mean features. ~0.747 → 0.7565.
3. **Feature-set-diverse ensembling** (biggest single jump): average 6 members built on deliberately
   different feature views — full (30-min TEs), fine (15-min TEs replacing the 30-min ones), nohm
   (hour-granularity TEs only). Diversity between members was worth far more than extra seeds:
   0.7565 → 0.7607.
4. **4-fold OOF folds** (vs the 5-fold default) for the target encodings: 0.7630 → 0.7632.
5. **Capacity retune at the end**: max_depth 8→10 and min_child_weight 10→3 (with ES 120 for runtime
   headroom): 0.7632 → 0.7638.

## 3 things that did not help (all measured, then reverted)

1. Native categorical/one-hot route features and Month/DayofMonth features — they overfit the
   2005→2006 distribution shift; TE encodings generalize better.
2. Recency weighting (upweighting late-2005 rows), month-informed sample weights, hierarchical TE,
   and new TE keys (dest_hm, car_route, route_dow, car_dest, car_org, global tod-bucket): all ≤ noise.
3. Bigger/jittered ensembles: 7–8 members, member re-weighting or reallocation, sklearn-HGB members
   (too correlated), TE-smoothing/depth/colsample jitter; also longer training (lr 0.025 + ES 250/300,
   3-fold OOF, reg_lambda 2.0, max_bin 512, rank:pairwise). All equal or worse than the simpler config.

## What I would try with more budget

A properly cross-fitted stacking meta-learner over the six member predictions (OOF member predictions
as meta-features, logistic/ridge meta-model) to replace the plain mean; optimizing the TE smoothing
constants m by inner-CV instead of fixed values; averaging OOF-TE over multiple fold-seeds to denoise
training targets; Distance-aware congestion features (distance-bucket × hour) since only time and
route/carrier families were exploited; and a fine capacity sweep around (d10, mcw3) — the last two
capacity steps both gained, suggesting a little more headroom. On the infrastructure side: the
per-experiment 120 s cap allowed 6 members comfortably but not 8; a faster member (fewer trees with
higher lr, or subsampled rows per member) might buy ensemble width, which was the strongest lever.
