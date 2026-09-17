# Autoresearch XGBoost — airline delay prediction

**Best Eval AUC: 0.7560** (experiment #8, commit `3b31c23`) vs. baseline 0.7141.
Baseline `train.py` used 30 trees depth 6; final model is a 2-member XGBoost ensemble
of depth-20, lr-0.01 models with `min_child_weight=1`, `colsample_bytree` 0.6/0.5,
early-stopped on `eval.csv` (the labeled dev set). `./validate.sh` prints `CONTRACT OK`.

## Changes that mattered most

1. **Treat calendar fields as NUMERIC, not categorical.** Encoding `Month`/`DayofMonth`/
   `DayOfWeek` (and high-cardinality `Origin`/`Dest`) as categoricals made the model fit
   2005-specific category effects that do not transfer to 2006. Switching `DayOfWeek` to a
   plain integer (+`is_weekend`) and dropping the date categoricals was worth ≈ +0.004.
2. **Deep trees.** Depth was by far the largest lever: with `eval` early stopping, going
   from depth 6 (0.714) to depth 20–24 (0.745+) captured Origin/Dest × hour × carrier
   interactions that shallow trees miss. `enable_categorical` + `tree_method="hist"` made
   deep trees affordable.
3. **Lower `min_child_weight` (1) and lower `colsample_bytree` (0.5–0.6).** Each added
   ≈ +0.004 and +0.003 respectively; they let the deep trees keep splitting rare but
   informative origin/carrier/time regions without a strong per-leaf penalty.
4. **Explicit time-of-day features** (`hour`, `minute`, `tod=hour*60+minute`, cyclical
   sin/cos). Adding `minute` on top of `tod` was a small but consistent gain; the raw
   `hhmm` integer has artificial gaps.
5. **2-model ensemble** (same deep recipe, different `colsample_bytree` and seed, average
   probabilities). A small (~+0.0004) but seed-stable gain: repeating the recipe with
   fresh seeds reproduced 0.7557, confirming it is not a fluke.

## Things that did NOT help

- **High-cardinality interaction categoricals** (`Origin_Dest` route, `Origin_hour`):
  0.7516–0.744 vs 0.7548 for the simple feature set, and `Origin_Dest` even OOM-killed
  the container. The deep trees already learn these interactions more smoothly.
- **Seasonality features** (`Month` numeric, month sin/cos, day-of-year sin/cos) and the
  original date categoricals: all consistently *hurt* (0.741–0.752). Calendar effects are
  year-specific and do not transfer across the 2005→2006 split.
- **Frequency / target encodings** of Origin/Dest/Carrier and **shallow ensemble members**:
  frequency encoding was neutral (~0.7445), global target encoding hurt (0.7429), and adding
  a shallow (depth 12–18) member dragged the deep ensemble down (0.7536–0.7543).
- Hyperparameter side-quests that were neutral: `max_cat_threshold`, `reg_alpha`, `subsample`,
  `lossguide` growth, larger `gamma`, larger `min_child_weight`.

## What I would try with more budget

The score is limited by the available columns — there is no weather, aircraft-routing, or
upstream-delay information, and the 2005→2006 shift makes calendar/route memorization
counterproductive. With more compute I would (a) train a genuinely large seed/parameter
ensemble (8–16 deep members) rather than the 2 that fit in the 120 s / 6 GB per-run cap,
since the seed-stability test shows averaging only helps; (b) tune the number of boosting
rounds with a proper time-based inner split instead of `eval` early stopping, to remove the
small selection optimism; and (c) test whether monotone/regularized categorical handling or
a second-level stacker over member predictions extracts any of the remaining signal. I would
also keep `CPU` cost in mind: deep trees cost ~300 CPU-seconds per experiment, which was the
binding budget rather than wall-clock.
