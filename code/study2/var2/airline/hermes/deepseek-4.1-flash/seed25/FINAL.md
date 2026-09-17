# FINAL — autoresearch XGBoost on the airline delay task

**Best Eval AUC: 0.7294** (experiment #40, commit `3677dad`). Baseline: 0.7141 (#1, commit `0d9add1`).
40/40 experiments used, ~215 minutes of the wall clock left unused, 1806/18000 CPU-seconds of Python compute.

`./validate.sh` prints `CONTRACT OK` and the validator's own held-out check reproduces **0.7294** through
`predict_proba()` on a frame with the target column removed.

## What the final `train.py` does

* `prepare(df)` is the single feature path used by both training and `predict_proba`; every statistic
  (categorical levels, calendar constants) is either fixed arithmetic or fitted on `data/train.csv` only.
* Features: departure clock time (`dep_min`, `dep_hour`, cyclic sin/cos, missing flag) replacing the raw
  non-monotone `hhmm`; ordinal calendar features (`month_num`, `dow_num`, `dom_num`, `is_weekend`) plus a
  fixed holiday-window flag; estimated scheduled arrival time (`arr_min`, `arr_hour`, sin/cos, `is_night_arr`,
  `est_block_h`, `is_next_day_arr`) built from `dep_min` + a distance-based block time; `Distance`;
  `UniqueCarrier`, `Origin`, `Dest` as XGBoost categoricals.
* Model: bag of 8 XGBoost classifiers (seed / depth 5–7 / subsample 0.6–0.9 / colsample 0.4–0.6 /
  reg_lambda 20–100 diversity), `n_estimators=1200`, `learning_rate=0.03`, `min_child_weight=50`,
  `max_cat_threshold=16`, probed averaged.

## Changes that mattered most

1. **Pruning noisy categorical features** (+0.0040, #28). Dropping the raw `c-<n>` Month/DayOfWeek/DayofMonth
   categorical columns (superseded by the ordinal + holiday features) was the single largest jump,
   0.7198 → 0.7238. Pruning `holiday_prox`/`is_peak_day` later added a further +0.0003.
2. **Regularization of the categorical/tree structure**: `colsample_bytree=0.5` (+0.0012, #17),
   `reg_lambda=50` (+0.0005, #16), `max_cat_threshold=16` (+0.0008, #25), initial
   `min_child_weight=50 / subsample=0.8 / colsample=0.8` (+0.0008, #4).
3. **Bagging 8 diverse models** (#23, and the whole config after it): early stopping on a 2005-internal
   holdout scored 0.7541 while eval scored 0.7123 — a large year-to-year shift — so variance reduction
   across decorrelated regularized models was worth more than any single fit (+0.0010 at the time, and it
   made every later measurement less noisy).
4. **Clock-time features replacing raw `DepTime`** (#6, +0.0005). `hhmm` is non-monotone (2359 wraps to
   0000); the derived cyclic terms turned out to be load-bearing (removing them cost 0.0105, #30).
5. **Estimated arrival-time features** (#38/#39, +0.0018 combined): departure clock + distance-based block
   time exposes destination-side congestion and overnight arrivals that departure time alone cannot express.
6. **Dropping the monotone constraint once features were pruned** (#31, +0.0020) together with restoring
   capacity (#33/#34: 200 → 350 → 600 trees, +0.0012; final `lr=0.03`, 1200 trees, +0.0002). The constraint
   helped (+0.0003) while the noisy feature set was in place, but blocked the night-flight signal afterwards.

## What did not help

1. **Raw capacity before regularization**: 400 trees / depth 8 gave 0.7118 vs the 30-tree baseline's 0.7141
   (#3), and depth 4 underfit at 0.7144 (#12). Early stopping tuned on 2005-internal data overfit the year
   shift (0.7123, #10).
2. **High-cardinality interaction and traffic features**: explicit `route`, `carrier|hour`, `origin|hour`
   categoricals (0.7163 vs 0.7169, #15), congestion/traffic-share encodings (0.7150 vs 0.7154, #9), and
   frequency encodings inside the first feature pack (0.7056, #2) all hurt — per-category memorization does
   not transfer from 2005 to 2006.
3. **Tighter regularization past the optimum**: `gamma=2.0` (0.7188 vs 0.7190, #24), `max_cat_threshold=8`
   (0.7192 vs 0.7198, #26), `colsample=0.3` (0.7162 vs 0.7186, #18), `subsample=0.6` (0.7176 vs 0.7186,
   #19), `min_child_weight=150` (0.7168 vs 0.7186, #22), and a 16-model bag (0.7196 vs 0.7198, #27 — no gain
   for 2x cost).

## What I would try with more budget

The dominant fact about this task is the 2005 → 2006 shift: an internal 2005 holdout is ~0.04 AUC easier than
the 2006 eval set, so every capacity or memorization knob has to be judged on cross-year robustness, and the
0.0005-level differences that drove most keep/discard calls on a 100k eval sample are at the edge of noise.
With more budget I would (a) build a proper cross-year validation scheme — hold out whole months of 2005 in a
rolling fashion and only trust changes that win in several folds — instead of single-eval decisions;
(b) replace the hand-written holiday windows with recency-weighted, smoothed target encodings for
carrier/airport fitted on training data only, which are the natural next information source but too
shift-sensitive to tune reliably at this budget; (c) run a feature-selection sweep over the remaining
numeric columns (`dom_num`, `is_weekend`, `dep_time_missing`, `log_dist`) and over the block-time constant,
since pruning repeatedly outperformed adding; (d) ensemble across feature subsets as well as hyperparameters
rather than only within one subset; and (e) test monotone constraints applied only outside the 00:00–05:00
window, which would keep the robust daytime shape without blocking the night-flight effect that made the
global constraint harmful.
