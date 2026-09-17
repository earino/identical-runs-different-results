# Final report

Best Eval AUC: **0.7393** (baseline 0.7141, +0.0252). Final commit: 2eebd23 ("colsample 0.2").
Validation: `CONTRACT OK`, predict_proba reproduces 0.7393 on eval.csv with the target column removed.

## Final model

XGBoost (hist, 4 threads) on ~32 features: raw time parts (month/dom/dow/deph/depm/minutes),
cyclical sin/cos encodings of month/dow/dep_time, Distance + log1p, log-frequency of
carrier/origin/dest/route, native categoricals for carrier/origin/dest/route, and 9 smoothed
out-of-fold target encodings (carrier, origin, dest, route, dep_h, and the interactions
dest|dep_h, carrier|dep_h, origin|dep_h, route|dep_h; smoothing m=100 on a global prior).
Hyperparameters: depth 7, lr 0.03, subsample 0.85, colsample_bytree 0.2, min_child_weight 2,
reg_lambda 384, early stopping on eval (best_iter ~1500-2000). All fitting happens on train only
(TE maps, freq, category levels, OOF encodings); predict_proba rebuilds every feature inside
prepare(df), so it transfers to the hidden holdout.

## Changes that mattered most

1. **Feature engineering + smoothed OOF target encodings** (exp 3, +0.0043 over baseline):
   cyclical time features and TE of the identity and hour-interaction keys were the single
   biggest lever. Route|dep_h proved surprisingly valuable (removing it cost -0.005).
2. **Strong L2 regularization** (exp 21-28, 1.0 -> 384, +0.0085 cumulative): the 2005->2006
   distribution shift punishes sharp leaves; eval AUC rose monotonically with reg_lambda
   until plateauing at ~384.
3. **colsample_bytree 0.8 -> 0.2** (exp 32-35, +0.0101): interacts synergistically with heavy
   lambda — each column subsample force-chooses weaker splits, which regularizes in the same
   direction. 0.15 and 0.3 were both worse, 0.2 is the optimum.
4. **Capacity/EA tuning**: depth 7 > 6 > 9, lr 0.03 > 0.05 > 0.02, ES window 150.

## Things that did not help

1. **More interaction TEs** (exp 8, 10, 13, 14): seasonal keys (origin|month etc.), low-card
   time keys (dow, dom, dep_h|dow), carrier|origin, carrier|haul — every TE addition beyond the
   core 9 lost 0.002-0.006. OOF noise outweighs the extra signal.
2. **Seed ensembles** (exp 12: 3-seed 0.7205 vs 0.7207; exp 39: 2-seed 0.7390 vs 0.7393):
   ties at multiples of the compute cost; dropped for simplicity.
3. **CV-based early stopping + training on 100% of train** (exp 18: 0.7184 vs 0.7207):
   iteration counts picked by 2005-CV underfit the shifted 2006 distribution; ES-on-eval
   transferred better despite using eval for one hyperparameter.

## With more budget

The lambda x colsample ridge was still paying at the budget line — a proper 2-D grid around
(lambda 256-768, colsample 0.15-0.3) plus lr 0.02-0.05 at the optimum would likely find another
+0.002-0.004. I would then try bagged XGBoosts on bootstrap resamples (diversity beyond seeds),
a wider TE key set selected per-key by OOF AUC contribution rather than wholesale, and DepTime
 missing-value handling (the hhmm > 2400 rows) as an explicit category. Finally, a larger OOF
 ensemble (bag the TE fold models themselves) is the standard next step if wall-clock allows.
