# Final report — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7392** (baseline 0.7141, +0.0251) — commit `a6e5793`, experiment #40.
Final model: ensemble of 3 XGBoost classifiers (depth 12/10/12, colsample 0.7/0.5/0.9, subsample 0.7/0.8/0.9,
lr 0.015/0.02/0.03, max_bin 512, min_child_weight 20/20/10, reg_alpha 4.0), each up to 8000 trees with
early stopping (200 rounds) on eval.csv, predictions probability-averaged.

## What mattered most

1. **Date/time feature engineering** (+0.0117 single-model): parsing `c-<n>` columns to ints, DepTime →
   hour/minute + sin/cos of time-of-day, log-distance. Time-of-day is the dominant signal (delay rate
   ranges 0.04→0.98 across hours; scheduled times >2400 are almost always delayed).
2. **Strong L1 regularization `reg_alpha`** (+0.0056 in the endgame): 0.5→1.0→2.0→4.0 improved eval AUC
   monotonically (0.7351→0.7358→0.7370→0.7392). Shrunk leaf weights generalize across the 2005→2006 shift.
3. **Hyperparameter depth/lr rebuild** (+0.0025): depth 10→12, lr 0.05→0.02, min_child_weight 20,
   subsample/colsample sampling, early stopping (best_iteration ~500, vs 62 at first attempt).
4. **Ensembling 3 full-data members** (+0.0012): seed + config diversity (subsample diversity helped,
   +0.0004); max_bin 512 (+0.0001).
5. **Keeping "obviously redundant" features**: an ablation dropping dep_min and dow sin/cos cost −0.0046.

## What did not help (all reverted)

- **Target encoding** (OOF and full-train, carrier/origin/dest/route/hour/month/dow): −0.004…−0.009.
  Airport/carrier delay rates drift between 2005 and 2006; 2005 statistics actively mislead.
- **High-cardinality native-categorical combos** (route, origin×hour, dest×month, …): −0.014. Same drift story.
- **Rank-encoded airport delay rates** (drift-robust variant of TE): −0.0007. Even ranks don't transfer.
- Frequency (log-count) encoding, holiday-proximity features, sin/cos cross-products, hour/dom/dow as
  native categoricals, colsample_bynode, gamma, CV-bagged (80%-subset) ensembles, lighter 5-member
  ensembles, logit-averaging: all neutral-to-negative.

## Theory

This dataset is a balanced 50/50 sample with a hard year shift (train 2005 → eval/holdout 2006).
Anything that memorizes *level-specific* statistics from 2005 (target encodings, high-cardinality
categorical splits on route/airport-hour combos) overfits the shift; XGBoost's native categorical
*partitioning* on Origin/Dest survives because it learns coarse "hub vs spoke" splits. Smooth calendar
features + heavy L1 shrinkage give the best cross-year transfer. With more budget I would sweep
reg_alpha per member (2–8), pair it with even lower learning rates (0.005–0.01), try depth 8–9 with
alpha 4, and add 2–3 more diverse members if wall-clock allowed.
