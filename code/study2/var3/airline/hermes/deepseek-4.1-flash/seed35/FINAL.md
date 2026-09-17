# FINAL — airline departure-delay (AUC) autoresearch run

**Best Eval AUC: 0.7253** (commit `6030976`, "destination hub/breadth features"), vs **0.7141** for the
committed baseline — a gain of **+0.0112** on the time-separated 2005→2006 split.
Contract re-checked after the last experiment: `./validate.sh` → `CONTRACT OK`, and `predict_proba`
(target column removed) reproduces 0.7253. Budget used: 40/40 experiments, 5,201 of 18,000 CPU-seconds,
~35 minutes of the 230-minute wall clock.

## What mattered most (in order of impact)

1. **Ordinal re-parameterisation of the `c-<n>` calendar columns** (+~0.003). `Month`, `DayofMonth` and
   `DayOfWeek` arrive as strings, so the baseline handed them to XGBoost as *categoricals* whose splits are
   partitions, not intervals. Exposing numeric ordinals (`month_num`, `dom_num`, `dow_num`, `day_of_year`,
   `is_weekend`) lets one split express "summer vs winter" or "Mon–Thu vs Fri–Sun".
2. **Dropping those same three categorical columns** (+0.0017) — the ordinals cover them, and the partition
   splits over 12×31×7 interleaved levels were pure overfitting. Together (1) and (2) are the single biggest
   move in the run: 0.7199 → 0.7231.
3. **Regularised capacity with early stopping** (+0.0028 over baseline, the first real gain). `learning_rate=0.03`
   with `early_stopping_rounds=50` on eval, `min_child_weight=5`, `subsample≈0.9`, `colsample_bytree≈0.8`,
   `reg_lambda=5`. The learning curve is flat after ~100–250 trees at that rate, so raw capacity was never the
   binding constraint — variance was.
4. **Congestion/hub structure fitted on train only** (+0.0019 total). Share of scheduled departures per
   `Origin|hour` and `UniqueCarrier|hour` (+0.0011), then `hub_carrier_share` / `hub_origin_share` /
   `hub_dest_carrier_share` / `carrier_origin_ndest` (+0.0007). These encode "how busy is this airport for this
   airline at this time" as a single ordinal split instead of needing deep interactions.
5. **A diverse bag of 8 XGBoost models** (+0.0010 in total over the best single model): different seeds, depths
   5–11, `min_child_weight` 5–150, `colsample` 0.6–1.0, `learning_rate` 0.02–0.03, each with its own early
   stopping, plus `num_parallel_tree=2`. Averaging probabilities is the main variance-reduction lever left once
   the feature set saturates; the members that helped most were the *deep but heavily regularised* ones
   (depth 8–11 with `min_child_weight` 30–150), which is why the last few experiments keep nudging this mix.

## What did not help (reverted)

- **Target encoding** (leave-one-out smoothed means for origin/dest/carrier/route/hour): 0.5986 — the LOO
  train-time encoding and the full-train encoding at eval time are different distributions, and the model
  buys into the leak.
- **Route/city-pair as a categorical feature**: 0.7102 (and the frequency family built on it, 0.7185). 4,198
  levels with many rare pairs is exactly the high-variance structure that fails to transfer across years —
  the same reason dropping the calendar partitions helped.
- **Raw depth / capacity increases** without extra regularisation: depth 7–8 on the baseline features (0.7167),
  400 trees (0.7134), 8/10-member bags at the wrong depth mix, `max_cat_to_onehot=32`, `colsample_bynode=0.5`,
  and month/day-of-week sin-cos seasonality all landed at or below the incumbent.

## What I would try with more budget

The eval set (100k) and the hidden holdout (1m) are both balanced random samples of the same year, so eval AUC
is an unusually faithful proxy here and I would keep optimising against it. The curve says the remaining
headroom is in *new information*, not in tuning: I would pursue aircraft-turn/lagged-delay structure (the same
tail number's earlier leg is the single strongest real-world predictor of a departure delay, but this schema
has no flight id, so the best proxy would be a per-`Origin|UniqueCarrier|hour` delay-propensity estimate fitted
*out-of-fold* across years — with the LOO protocol replaced by k-fold, since plain LOO encoding failed badly
here). Second, I would try weighting the bag instead of averaging it equally, and stacking the member outputs
with a small constrained XGBoost meta-model (`max_depth=2`, heavy regularisation). Third, I would sample
2005-vs-2006 schedule drift explicitly — e.g. airport-pair share features restricted to pairs present in both
years — to make the structural features even more year-robust. Everything else I would expect to be noise at
the ±0.001 level where this run has been living since experiment 30.
