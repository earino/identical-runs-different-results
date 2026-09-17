# Final report — autoresearch XGBoost (airline)

## Result

- **Best Eval AUC: 0.7466** (experiment #39, commit `0624155`, single XGBoost model)
- Baseline: 0.7141 (experiment #1) → **+0.0325 AUC**
- Budget used: 40/40 experiments, ~342 / 18,000 Python CPU-seconds, ~8 / 230 minutes wall clock.
- `./validate.sh` prints `CONTRACT OK`; `predict_proba(df)` reproduces all feature engineering from raw rows.

## The 3–5 changes that mattered most

1. **Out-of-fold target encoding of composite entity×time keys (huge, +0.03 total).**
   The single largest lever was smoothed, 5-fold out-of-fold mean-target encoding of interaction keys:
   `Route×Hour`, `Route×30/15/5/1 min`, `Origin×Hour`, `Dest×Hour`, `Origin/Dest×15/5 min`,
   `Carrier×Hour/Time`, `DayOfWeek×Hour`. Finer time resolution kept helping:
   0.7192 (hour) → 0.7285 (Route×Hour) → 0.7369 (Route×30m) → 0.7407 (×15m) → 0.7414 (×5m),
   and Origin/Dest×time lifted 0.7414 → 0.7455. Fold encodings are computed on train only; the
   full-train maps are stored and reused by `prepare()` for unseen rows. Smoothing `k=20` was best.
2. **Calendar / cyclical features (+0.0015).** A synthetic day-of-year from Month+DayofMonth plus
   `sin/cos` of day-of-year and time-of-day, and an `is_weekend` flag. `Month` was the #2 feature by
   gain, so making seasonality smooth and interpolable helped shallow trees.
3. **Shallow, regularized trees (depth 3, 200 estimators, lr 0.1).** The 2005→2006 shift punishes
   capacity: depth 4–6, 300–400 trees and subsampling all reduced eval AUC. Depth 3 with 200 trees
   was consistently the best operating point (baseline depth 6 / 30 trees = 0.7141).
4. **Combining native XGBoost categorical splits with target encoding.** Native `enable_categorical`
   partitions alone gave 0.7160; target encoding alone (no native cats) collapsed to 0.7107. Keeping
   *both* was necessary — they capture complementary structure.
5. **Explicit `dep_hour` / `dep_minutes` numeric features** on top of raw `DepTime` (+0.0005), which
   made the time-of-day TE buckets and the tree splits align.

## The 3 things that did NOT help

1. **More model capacity.** Depth 4/5/6, 300–400 trees, `subsample`/`colsample=0.8`,
   `min_child_weight=5`, `max_cat_threshold=256` — all equal or worse. The year shift is the binding
   constraint, not the model.
2. **Seasonal and structural composite encodings.** `Origin×Month`, `Dest×Month`, `Carrier×Month`,
   and `Carrier×Route` target encodings all hurt (e.g. 0.7180 vs 0.7192). Seasonality is already
   carried by the calendar features; adding it per entity mostly added noise.
3. **Higher Fourier harmonics and finer smoothing.** 2nd/3rd day-of-year and time-of-day harmonics
   overfit (0.7166 vs 0.7171). Lowering the TE smoothing to `k=5` also hurt (0.7406 vs 0.7414).
   An ensemble of 3–5 same-config seeds gave *exactly* the same eval AUC (0.7466), so it was dropped
   for simplicity.

## What I would try with more budget

The dominant factor is clearly within-route/within-airport time-of-day delay structure, so I would
push further along that axis: (a) time-decayed or recency-weighted target encoding so 2005 statistics
are blended toward the inference period rather than uniformly averaged over the year; (b) a
hierarchical/shrunk encoding (route→origin→global) instead of independent maps per key, which should
be more stable for sparse route×minute cells; (c) frequency/congestion features (flights per
route-hour, origin-hour throughput) as a target-free complement to the mean encodings; (d) test
whether training on 2005 + a pseudo-labelled or unlabelled slice of the later period closes the
remaining year gap; and (e) a low-learning-rate depth-3 model with many more trees plus proper
early stopping on a time-based holdout, which might safely extract a little more than the fixed
200-tree configuration. Given eval (2006 slice 1) and the hidden holdout (2006 slice 2) are the same
year, these TE-based gains should transfer, but a time-decay scheme would be the natural guard
against over-fitting the 2005 delay regime.
