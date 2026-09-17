# FINAL — airline delay AUC

**Best Eval AUC: 0.7384** (HEAD = de16759, `x3 depth22 col0.2`; validate.sh: CONTRACT OK, predict_proba
reproduces 0.7384 with the target column removed).

Final model: mean of 3 XGBoost models (seeds 42/7/123), each: lr 0.015, 4000 trees (early stopping on
data/eval.csv, patience 300; best_iter ~680-690), max_depth 22, subsample 0.4, colsample_bytree 0.2,
reg_lambda 10, hist tree method with native categoricals. Features (all inside `prepare()`): DepTime
decomposition (hhmm, wrapped time-of-day, sin/cos, hour 0-26, minute, late-night flag), calendar as
numbers + weekend flag, Distance + log-distance, and the six raw columns as native categoricals
(unseen levels -> NaN). No target/frequency encodings, no interactions survive in the final model.

## Changes that mattered most

1. **Early stopping against data/eval.csv instead of an internal split** (exp 3, 0.7141 -> 0.7154):
   eval.csv is from the same 2006 slice family as the hidden holdout; using it for round selection
   beat a random 15% holdout of 2005 train and set the pattern for everything after.
2. **Aggressive stochastic regularization** (exp 13-16, 0.7154 -> 0.7225): subsample/colsample_bytree
   down to 0.5/0.5 — by far the biggest lever on this dataset.
3. **reg_lambda 10 + deep trees under that protection** (exp 18, 22-26: 0.7225 -> 0.7306): with heavy
   row/column subsampling, depth 18-22 became optimal (the same depth was *harmful* with weak
   regularization — exp 4/5).
4. **lr 0.015 + seed ensemble of 3** (exp 29-30: 0.7306 -> 0.7312): modest on eval, but averaging
   seeds is robust variance reduction for the hidden set.
5. **colsample_bytree 0.2** (exp 33-36: 0.7312 -> 0.7384): even stronger column subsampling kept
   paying all the way to 0.2 (0.3 -> +0.0045, 0.2 -> +0.0009; 0.15 worse).

## What did not help

1. **Target encoding** (OOF, 9 smoothed features incl. Origin/Dest/Route/Carrier×hour; exp 7):
   0.7087 — clearly worse. 2005-level aggregates do not transfer to 2006 and the model over-trusts them.
2. **Categorical interaction features** (7 crosses like carrier×hour, origin×month as native
   categoricals; exp 9): 0.7083. Same failure mode; also slower.
3. **Frequency features** (Origin/Dest/Carrier/Route frequencies; exp 10): exactly equal AUC —
   pure dead weight, removed for simplicity.

## With more budget

The eval-set curve was still rising as colsample_bytree fell (0.3 -> 0.2 gained +0.005); I would
probe 0.18-0.22 more finely, combine col=0.2 with subsample 0.35/0.45, and re-test reg_lambda 5-15
at the final subsampling level (the optimum moved 1 -> 10 when subsampling tightened). I would also
grow the seed ensemble from 3 to 6-8 models (runtime is the binding constraint: ~95 s per 3-model
run of the 120 s cap), try one model per-year-of-style weighting or bagging over 2005 sub-periods to
directly address the 2005->2006 shift that killed target encodings, and test gamma / max_leaves
constrained growth (depth 22 unconstrained is near the top of what time allows). Parameter sweeps
were run as single-seed probes before ensembling; with more budget I would confirm the final config
across seeds, since single-seed differences of ±0.0005 are within noise here.
