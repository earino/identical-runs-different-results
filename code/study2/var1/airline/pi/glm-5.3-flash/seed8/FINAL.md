# Final Report — airline delay (XGBoost, scenario 2)

**Best Eval AUC: 0.7343** (`Eval AUC: 0.7343`, reproduced by `validate.sh` → `CONTRACT OK`).
Shipped config: one-hot + engineered time features + carrier×hour block, 5-fold bagged ensemble of
XGBoost models (500 trees, depth 9, lr 0.1, subsample 0.8, colsample 0.8, max_bin 128, seed-diverse
members), chunked `predict_proba` (100k-row chunks, bounded ~3.3 GB peak).

## Deliberate safety choice
Between the top-scoring config (7-fold × 550 trees, eval 0.7361, ~118 s local, ~155 s estimated when the
hidden scorer re-runs `train.py` and predicts 1M holdout rows) and this one (0.7343, ~70 s local, ~100 s
estimated hidden), I shipped the latter: a killed hidden run scores as a failure, and the local gap
(0.0018) is ~2-3× the measured eval noise (same config, seed 42 vs 44: 0.7343 vs 0.7341; run-to-run is
deterministic). Chunked prediction is also what keeps a 1M-row frame under the 6 GB memory cap
(unchunked `prepare()` on 1M rows OOM-killed a 6 GB container during testing).

## What mattered most
1. **One-hot encoding instead of XGBoost native categoricals** (+0.010 eval): native categorical
   partitions overfit 2005-specific level structure; one-hot with more trees generalizes across the
   2005→2006 year shift.
2. **Bagging into a 5-fold seed-diverse ensemble** (+0.004 over a single model at equal trees): members
   see 80% of train each, different `random_state` per member.
3. **carrier×departure-hour one-hot interaction** (+0.002): delay patterns are carrier-specific by time
   of day; stable across years.
4. **Feature pruning** (+0.0015 total): dropping numeric `month`/`dow`/`is_weekend` (one-hots already
   carry them) and numeric `day` (non-monotonic noise) reduced dilution under colsample; each removal
   matched or beat the fuller set.
5. **Capacity under regularization** (600→550 trees/depth 9/subsample-colsample 0.8, max_bin 128):
   deeper/looser single models overfit the year shift; bagged depth-9 with stochastic subsampling did not.

## What did not help
1. **Route (origin–dest) as a categorical** (-0.010): 4.2k levels, mostly tiny cells — overfits even
   in-year; also fine-grained interactions (dow×hour, origin×dow, day-of-month one-hot) all lost.
2. **Target encodings and frequency encodings** of carrier/origin/dest (OOF, smoothed α=20-50): neutral
   to slightly negative on top of both native-categorical and one-hot layouts.
3. **Hyperparameter micro-tuning around the optimum** (gamma 2, min_child_weight 10, max_bin 512,
   lr 0.08×700, subsample/colsample 0.7 or 0.9, 10×300-trees bag, full-data no-fold bag, lossguide):
   all within ±0.0006 noise or worse; several cost timeouts.

## With more budget
I would (a) verify with a leakage-free 2006-slice validation whether the tree-count trend
(2000→3850 trees: 0.7339→0.7361) holds out-of-year or flattens, and if it holds, squeeze more total
trees into the runtime envelope by vectorizing the one-hot `prepare()` (numpy indexing instead of
`get_dummies`, ~10 s saved on 1M rows); (b) try one-hot interactions restricted to *stable* blocks only
(carrier×month, hour×month) instead of the high-cardinality ones that failed; (c) replace uniform bag
weights with OOF-AUC-weighted averaging; (d) test a two-level stack (average of a fold-bag and a
seed-bag) at reduced tree counts to fit the runtime cap.
