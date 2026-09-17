# Final Report — airline delay (AUC)

**Best Eval AUC: 0.7241** (commit `87f869d`, experiment #39). Baseline was 0.7141, so +0.0100.

## What mattered most

1. **Numeric/cyclical calendar instead of categorical calendar** (+0.0024 in one step, exp #30).
   Treating `Month`/`DayofMonth`/`DayOfWeek` as numeric + cyclic (`month_sin/cos`, `dow_sin/cos`,
   `dom_num`) instead of `enable_categorical` levels was the single biggest win. Categorical calendar
   levels let the trees memorize 2005-specific date effects that do not transfer to 2006.

2. **Low-depth, many-tree base learners** (depth 3–4, lr 0.01–0.02, 600–1600 trees) instead of the
   baseline depth-6/30-tree model. Deeper/higher-capacity single models monotonically *lost* on the
   time-separated split (e.g. 600 trees depth 7: 0.7036), showing over-fitting to 2005.

3. **Depth-diverse XGBoost ensemble** (depths 1–8, plus bagged members) — averaging many
   differently-regularized boosters gave steady variance reduction (0.7174 → 0.7190).

4. **Relative airport-hour congestion shares** (`origin_hour / origin_total`, same for dest) (+0.0013,
   exp #36). How busy an airport is *at that hour relative to its own baseline* is stable year-to-year;
   absolute daily volumes were not.

5. **Strong L1/L2 regularization** (`reg_lambda=20`, `reg_alpha=2.0`) on every ensemble member
   (+0.0006, exp #39). Regularization and, similarly, log-distance / carrier-hour frequency features
   each added a little.

## What did NOT help

- **Target encoding** of Origin/Dest/Route/Carrier (0.7020, −0.014): 2005 delay propensities simply do
  not transfer to 2006.
- **Deep / high-capacity single models** (100 trees: 0.7125; 600 depth-7 trees: 0.7036) and
  **origin/dest daily-volume features** (0.7125) — both memorize the training year.
- **DART boosters**, logit-space averaging, recency sample-weighting, and raw frequency encodings:
  neutral or worse (DART also ran to ~119 s, near the timeout).

## With more budget

The clear lesson is that this 2005→2006 shift rewards *low-variance* modeling. Next I would (a) tune
the per-depth ensemble weights and member count against a proper time-based validation split carved out
of the training year, (b) grid over `reg_lambda`/`min_child_weight`/`max_cat_threshold` for the
Origin/Dest categoricals, since the categorical-calendar ablation suggests high-cardinality levels are
the remaining over-fitting risk, and (c) try reducing Origin/Dest cardinality via stable airport
clusterings while keeping the relative-congestion features.
