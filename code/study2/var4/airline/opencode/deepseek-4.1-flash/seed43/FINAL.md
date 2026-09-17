# Final report — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7390** (commit `2a698de`, "depth20 n100"), vs. **0.7141** baseline.
40/40 experiments used; contract validated (`CONTRACT OK`).

## What mattered most

1. **Dropping year-specific calendar features (−0.0 +0.0044).** Removing `Month` and then
   `DayofMonth` categoricals was the single biggest win (0.7177 → 0.7213 → 0.7221). The train
   slice is 2005 and eval/holdout are 2006: month/day seasonality is a year-specific artifact that
   the model happily memorizes but does not transfer. `DayOfWeek` *did* transfer (dropping it cost
   −0.003), so weekday effects are stable while calendar-position effects are not.
2. **Regularization before capacity.** Early on, extra trees hurt (30 → 400 trees lost 0.006).
   Adding `min_child_weight=10`, `subsample/colsample=0.8`, depth 4–5 and later `reg_lambda=5`
   turned the direction around and made capacity usable.
3. **Very deep trees with few boosting rounds.** Once regularized, depth was the dominant lever:
   0.7234 (d5, n250) → 0.7308 (d12) → 0.7359 (d14) → 0.7379 (d16) → 0.7386 (d18) → **0.7390
   (d20, n100)**. Deep trees capture the strong high-order interactions (carrier × airport ×
   time-of-day) that shallow trees miss; the optimum moved from ~400 rounds at depth 6 to ~100
   rounds at depth 20.
4. **Bagged ensemble of 10 seeds** (subsample 0.8 makes seeds stochastic) for small but consistent
   variance reduction: +0.0004 over a single model at the equivalent config.
5. **Frequency encodings** of carrier / origin / dest / route (`*_freq`, fit on train only): small
   robust gain that survives the year shift, unlike target encodings.

## What did not help

- **Target encoding** of carrier/origin/dest/route (smoothed, train-only): −0.013 (0.7041).
  Airport delay base rates shift between 2005 and 2006, so the model latches onto stale signal.
- **Route as a categorical** (~4.2k levels, only 92% present in eval): −0.010, and dropping the
  `Origin`/`Dest` identity categoricals entirely also lost 0.003 — identity is useful but the raw
  route level overfits.
- **Early stopping on an internal random split**: selected 257 trees and scored 0.7123. Because the
  split is same-year, internal validation cannot see the 2005→2006 shift and systematically
  over-selects capacity.
- (Also neutral/negative: raw `DepTime` removal, `is_weekend`, `max_cat_threshold=16`,
  subsample 0.7, unlimited depth `max_depth=0`.)

## With more budget

The clear remaining direction is the depth/rounds trade-off: the optimum was still moving at the
budget edge (depth 20, n=100), and unlimited depth with 80 rounds scored 0.7388 — essentially tied.
A proper sweep of `max_depth ∈ {20, 24, 0}` × `n_estimators ∈ {60..120}` × `min_child_weight ∈
{5,10,20}` with a larger ensemble would likely find another few thousandths. More fundamentally, the
year shift is the binding constraint: I would invest in shift-robust feature construction (e.g.
relative frequency within airport/hour, interactions engineered to be stable across years) and in a
leave-one-year-out style validation harness rather than random splits, since every random-split
signal in this task was misleading. Finally, stacking the deep-tree ensemble with a shallow
regularized model (which was better at low capacity) could hedge against the holdout drawing a
different capacity optimum.
