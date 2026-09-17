# FINAL — airline dep-delay classification (XGBoost)

Best Eval AUC: **0.7461** (baseline 0.7141, +0.032) — experiment #40, commit 4434f03,
validated with `CONTRACT OK`.

## What mattered most

1. **Departure-time band categorical** — 15-min bands of DepTime as a categorical feature
   (`dep_band`): +0.015 alone over numeric time features. Time-of-day is the strongest
   signal in this dataset and banding lets the trees carve it cleanly.
2. **Hour-of-day x carrier interaction categorical** (`hour_carrier`, levels from train only):
   the single biggest jump, +0.010. Carriers have different delay-vs-schedule-time profiles.
3. **30-min band x carrier categorical** (`band30_carrier`): a second, coarser time x carrier
   view added on top: +0.001 and it made the ensemble more stable.
4. **Bagged XGBoost ensemble** — 15 models on 90% row subsamples (no replacement), depths
   cycling 6/7/8, 600 trees each, lr 0.05, colsample 0.7, probabilities averaged. Monotone
   gains from 3 -> 5 -> 8 -> 10 -> 12 -> 14 -> 15 bags (+0.002 total over single model),
   all within the 120 s cap.
5. **Basic numeric time features** kept alongside: hour, minute, dep_frac, sin/cos of
   day-of-day, month/day/dow/doy integers. Cheap and they let trees fall back to smooth
   splits when bands are sparse.

## What did not help

1. **Target (smoothed mean) encodings** of hour/carrier/origin/dest/route — consistent
   losers (0.7114 vs 0.7158). The 2005 -> 2006 shift makes target statistics drift; AUC
   drops on the later year.
2. **High-cardinality interactions with airports** (route cat, hour x origin, hour x dest,
   hour x route-rank): all overfit (down 0.001-0.01) and cost 2x runtime.
3. **Hyperparameter tweaks beyond the bag**: min_child_weight 3/5, reg_lambda 2.0,
   lr 0.03-0.04 with more trees, bootstrap-with-replacement, max_bin diversity — every
   one equal or worse. The 600-tree lr-0.05 depth-6/7/8 colsample-0.7 recipe was already
   at the optimum for this data size.

## With more budget

I would try: (a) time-of-day x carrier x month triple interactions as categoricals, since
both pairwise time interactions helped and seasonality modulates delay risk; (b) feature
permutation importance on `hour_carrier`/`band30_carrier` to prune correlated variants and
reduce overfit risk on the hidden holdout; (c) monotonically-constrained trees (later
scheduled departure -> higher delay risk) with `monotone_constraints`, which often helps
when the physical mechanism is known; (d) larger ensembles with per-bag early stopping on
an internal time-ordered split (last 15% of 2005) instead of fixed tree counts, so each
member adapts its own complexity; (e) isotonic calibration check on 2005-val to confirm the
probability averaging isn't dominated by one or two bags.
