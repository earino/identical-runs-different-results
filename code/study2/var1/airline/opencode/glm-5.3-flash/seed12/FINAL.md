# Final report — airline / XGBoost binary classifier

**Best Eval AUC: 0.7257** (baseline 0.7141 → +0.0116). Final config: 9-member XGBoost bag,
`n_estimators=400, max_depth=9, learning_rate=0.02, subsample=0.9, colsample_bytree=0.6,
min_child_weight=20, reg_lambda=5`, native categorical handling for Month/DayofMonth/DayOfWeek/
UniqueCarrier/Origin/Dest, plus engineered numeric features. `./validate.sh`: CONTRACT OK.

## Changes that mattered most

1. **Bagged ensemble of seeds** (5→9 members, averaged probabilities): +0.0025 early, kept improving
   robustness; the single biggest structural lever at the baseline config.
2. **In-experiment hyperparameter search on an internal 70/30 split** (selected by CV, not eval):
   moved from depth-6/30-round to depth-9/lr-0.02/400-round with strong row/column subsampling and
   light L2 — worth +0.004 total across three refinement rounds (0.7167 → 0.7201 → 0.7204 → 0.7213).
3. **Dual numeric + categorical representation of the time columns**: hour, minute, late-night flag,
   numeric month/dom/dow, day-of-year, log-distance as *extra* columns beside the raw DepTime /
   native categoricals (+0.0024 combined: 0.7213 → 0.7233 → 0.7245). Trees get cheap ordered splits
   on hour and date without spending depth parsing `hhmm`.
4. **Traffic-volume features** (log flights per Origin and per Dest, fit on train only): +0.0005 —
   shift-robust because they carry no target information.
5. **Final regularization nudge**: subsample 0.8 → 0.85 → 0.9 (+0.0004) — more data per tree helped
   once depth-9 + bagging controlled variance.

## Things that did not help

- **Target-encoding features** (smoothed delay rates for carrier/origin/dest/route/hour): 0.7066 —
  2005 rates do not transfer cleanly to 2006 and duplicate what native categoricals learn.
- **Route features in any form** (Origin>Dest categorical with rare-capping, route volume): always
  negative; routes are too sparse/unstable across years.
- **Feature-view diversity bags and shallow-pocket blends** (dropping Origin/Dest per member, mixing
  depth-6 members into the deep pocket): all below the homogeneous deep bag.
- Also negative: cyclical sin/cos encodings, hour×dow cross categorical, hour/minute as replacement
  (instead of addition) for raw DepTime, K-fold bagging, colsample 1.0, max_bin 128, lossguide,
  deeper (d10/d11) or more rounds (n=600) at fixed lr, rank-mean aggregation (exactly tied prob-mean).

## Key insight

The eval set is one year ahead of train (2005 → 2006), so the binding constraint is covariate/
concept shift, not model capacity. Everything that fit 2005-specific level combinations harder
(target encodings, route features, deeper-or-longer without subsampling) lost AUC; everything that
added stable structure (bagged variance reduction, interactions via depth-9 trees, redundant
representations of the same signal, target-free aggregates) gained. Internal CV ranked configs
correctly only up to ~±0.001; beyond that the year shift dominates and only robust choices paid off.

## With more budget

- Stacking: 5-fold OOF predictions of the deep bag into a shallow XGBoost meta-learner (untried —
  the only structural idea left with real upside; expensive at ~90-120 s per experiment).
- A broader randomized search around the depth-9/10 pocket with `grow_policy=lossguide` and
  interaction-aware column sampling, scored by repeated internal splits to cut selection noise.
- Larger member count (15-25) with disjoint seeds/subsample streams, and per-member feature-bagging
  at the column-group level (time block vs. airport block vs. distance block).
