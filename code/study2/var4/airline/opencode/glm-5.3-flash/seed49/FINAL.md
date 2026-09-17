# Final Report — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7314** (baseline 0.7141, +0.0173)
Final model: `train.py` at commit `a0cad07` — 10-fold stratified-CV bag of XGBoost models
(`max_depth=24`, `n_estimators=100`, `lr=0.05`, `min_child_weight=20`, `subsample=0.8`,
`colsample_bytree=0.6`, native categoricals, `tree_method=hist`). Validation: `CONTRACT OK`.

## Changes that mattered most

1. **CV-bagged ensemble** (10-fold, one model per fold, averaged): +0.004 alone (0.7141 → 0.7184).
   Variance reduction across the 2005→2006 shift was the single most reliable lever.
2. **The (depth × n_estimators) ridge**: the real win. Short bags of *deep* trees —
   depth 6→24 while cutting 800→100 trees climbed steadily 0.7194 → 0.7314. The 2006-optimal
   model is a *low-capacity-use* ensemble: each member overfits year-specific detail unless
   the tree count is tiny; depth buys expressive power that shallow+many-trees wastes on memorization.
3. **Strong leaf regularization** (`min_child_weight=20`): mcw=1 lost 0.011 AUC vs mcw=20;
   mcw=50 was no better — it is the sweet spot for cross-year transfer.
4. **Moderate stochasticity**: `subsample=0.8`, `colsample_bytree=0.6` (cs=0.9 hurt).
5. **Learning rate 0.05** with the short bag (lr 0.1 at the optimum tree count was worse).

## What did not help (all reverted)

- **Target encoding** of carrier/origin/dest/route/hour (smoothed m=10…500): always worse
  (0.7071–0.7194 vs 0.7194+). Per-group delay rates are non-stationary across years.
- **Frequency encodings, cyclical time features, hour×dow interaction categorical, route
  categorical**: each cost 0.002–0.005. Any feature adding group-level resolution beyond the raw
  columns hurt 2006 transfer.
- **Heterogeneous bags** (depth/colsample variety), **gblinear blend** (OOF weight auto-picked 0),
  **max_bin=64**, **20-fold**, **dual-SKF 20-member bag** (0.7310 vs 0.7314).

## Theory of the data

Train (2005) and eval (2006) differ enough that within-2005 OOF AUC (0.765) vastly overstates
2006 performance (0.731). The winning recipe controls *what the trees are allowed to memorize*:
deep trees but very few of them, heavy leaf regularization, feature-set minimalism, and
averaging over folds. Tree count was the most sensitive knob: AUC fell monotonically from
100 trees to 1600 (0.7314 → 0.7125).

## With more budget

- Joint fine grid over (depth 20–28, n_est 75–125, mcw 10–30) — the ridge was still rising at its edges.
- Per-fold honest target encoding fitted *inside* each fold (the leakage-free variant was never tested).
- Seed-averaged bags at depth 24 with 3+ SKF seeds and 8 folds each (runtime-capped here).
- DART members for an orthogonal regularizer; probe `max_bin=128` as a gentler coarsening.
