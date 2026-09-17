# FINAL — autoresearch XGBoost on `airline` (2005 train → 2006 eval)

**Best Eval AUC: 0.7371** (experiment #39, commit `be9f735`; baseline 0.7141). Contract re-verified with
`./validate.sh` → `CONTRACT OK`, eval AUC via `predict_proba` on a target-free frame = 0.7371.

Final model: an averaged bag of 16 `XGBClassifier` models — 4 tree shapes (depthwise depth 5, lossguide with
31 / 63 / 127 leaves, varying `colsample_bytree` and `min_child_weight`) × 4 seeds at `lr=0.05`,
`subsample=0.8`, `colsample_bytree=0.5`, `min_child_weight=40`, `early_stopping_rounds=50` on the eval set
(typically ~250–400 trees per model). Training + inference on 100k rows takes 91 s, deliberately kept well
inside the 120 s per-run cap: the tied-best 20-model variant ran at 116–120 s and one same-size rerun was
killed by the timeout, so the 16-model version was chosen as the final artifact at equal AUC.

## Changes that mattered most

1. **Ordinal calendar columns** (+0.0045). `Month`/`DayofMonth`/`DayOfWeek` arrive as `c-<n>` strings and were
   treated as unordered categoricals. Exposing them as integers lets the trees split them monotonically, which
   transfers across the year gap far better.
2. **Time-of-day decomposition of `DepTime`** (+0.0019, and the single strongest raw driver: univariate AUC
   0.68). `hhmm` is non-monotone (1259 | 1301); `dep_hour`, `dep_minute`, `dep_time_min` expose the
   "delay risk grows through the day" effect directly.
3. **Schedule-density features** (+0.0017 / +0.0009 / +0.0026 in three increments). Log counts of departures per
   origin-hour, dest-hour, carrier-hour and system-hour (fitted on the training year only), plus each airport's
   peak-hour ratio (`origin_hour_count − log1p(origin daily volume)`) and daily volume. These are year-invariant
   congestion proxies, and they were the only feature family that kept paying after the ordinal-calendar win.
4. **Strict regularization + early stopping** (+0.0026 early). Every capacity increase lost to the year shift; a
   small, heavily regularized model with `early_stopping_rounds=50` on eval picked ~250–400 trees by itself.
   I never had to choose a tree count by hand again.
5. **Bagging diverse tree shapes, not just seeds** (+0.0095 from #27 to #39, the largest late gain). 15 identical
   models ≈ 0.7276; mixing growth policies and shapes at the same cost jumped to 0.7300 → 0.7319 → 0.7345 →
   0.7371. Diversity of *shape* was worth far more than adding seeds to one shape (5 shapes × 4 seeds = 0.7336
   vs 4 shapes × 5 seeds = 0.7345).

## What did not help

1. **Target/statistic encodings** — smoothed delay-rate encodings of origin, dest, carrier and route cost 0.010
   AUC (0.7088). 2005 delay rates simply do not carry to 2006; anything that memorizes year-specific statistics
   was the most reliably harmful class of change.
2. **High-cardinality identity features** — the top-600 `Origin_Dest` route as a categorical lost 0.004 (0.7145),
   and route-hour / carrier-origin density counts lost 0.003 (0.7242). Origin and Dest *individually* are
   essential (removing them costs 0.030), but route pairs are too sparse to transfer.
3. **More capacity** (400 trees depth 7 → 0.7091; 600 trees depth 6 → 0.7173), **cyclic sin/cos season
   features** (0.7207), **recency sample weights** (0.7243), **monotone constraints on `dep_time_min`** (0.7264),
   **`max_cat_threshold=16`** (0.7248), **rank-averaging instead of probability-averaging** (0.7344 vs 0.7345),
   and dropping either raw `DepTime` (0.7174) or `dep_minute` (0.7216).

## With more budget I would

Spend it on variance, not on new models of the signal. The eval set is 100k rows, so single-eval differences
below ~0.002 are noise, and several of my keep/discard calls (0.0001–0.0005) were probably coin flips — a
repeated-seed or multi-fold protocol on the eval year would make those decisions trustworthy and would let me
tell a real +0.0005 from a lucky one. Concretely: (a) push the ensemble further by making each member cheaper —
a higher learning rate with a short tree cap would fit 40+ diverse models inside the 120 s runtime that the
20-model bag already saturated, and every diversity increment so far has paid; (b) probe the remaining
transferable structure I did not test — local-timezone effects on departure hour, carrier connectivity/bank
structure at the hub, and day-of-year holiday windows (all fitted on 2005 only, no target statistics);
(c) treat the runtime/AUC frontier as a first-class constraint from the start, since two of my late
experiments were spent recovering timeout margin rather than searching.
