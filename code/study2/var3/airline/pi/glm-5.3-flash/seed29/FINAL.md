# Final report — airline delay XGBoost

**Best Eval AUC: 0.7255** (baseline: 0.7141, +0.0114). Final commit `d7d161e`, validated (`CONTRACT OK`).

## Final model
9 XGBoost members → actually 8 members (the 9th was dropped as no-gain), probability-averaged.
Each member: depth 4–8, subsample 0.6–0.9, colsample_bytree 0.5–0.9, **colsample_bynode 0.15–0.5**,
gamma 0.2–3, lr 0.02–0.1, 250–900 trees. Features: raw categoricals (native categorical splits) +
smoothed out-of-fold target encodings (carrier k=50, Origin/Dest k=200, per-member KFold variants) +
time-of-day features (hour, minute, cyclic sin/cos, night flag) + log-distance.

## Changes that mattered most
1. **`colsample_bynode` (per-split feature bagging) = 0.35** — the single biggest win (+0.004 over three
   steps 0.7→0.5→0.35, exps 25–27). Classic colsample_bytree alone was far weaker.
2. **Diversity ensemble instead of a single tuned model** (exp 15): 5–8 members with different
   depth/subsample/lr/regularization, averaged — +0.0016 over the best single member and much more stable.
3. **Smoothed target encodings for carrier/Origin/Dest** (k=50/200/200) + regularized 300-tree model
   (exp 8/10): +0.0046 over the raw-categorical baseline.
4. **Per-member OOF-TE fold variants** (exp 24): each ensemble member sees target encodings from a
   different KFold split — small but consistent gain, and reduces encoding self-leak.
5. **Moderate regularization set** (subsample 0.7, colsample_bytree 0.7, mcw 5–10, gamma) with lr 0.03
   and 500–800 trees: the year shift (2005 train → 2006 eval) punishes overfitting hard; depth 8 or
   500+ unregularized trees all scored worse than the baseline.

## Things that did NOT help
- **Self-training / pseudo-labels on eval.csv** (exp 22): 0.7162, clearly worse.
- **Covariate-shift density-ratio weighting** via a train-vs-eval domain classifier (exp 30): 0.7239, worse.
- **Interaction target encodings** (route, carrier×hour, origin×hour; exp 9): 0.7117, much worse — leak +
  year-instability of high-cardinality interactions.
- Plain scaling (500–800 trees, depth 8, exps 2–4), hour/date TEs (exps 18/20), max_bin=64 (exp 19),
  rank-averaging (tie, exp 23), subsample 0.6 (exp 32), dropping raw airport categories (exp 21).

## Theory of the data
The dominant obstacle is the **2005→2006 temporal shift**: random train/val AUC runs ~0.04 above the 2006
eval AUC, and every attempt to fit 2005 harder (more trees, depth, interactions) loses on 2006. What
transfers across years are coarse, smoothed aggregate propensities (carrier/airport delay rates,
hour-of-day effect) and heavily bagged shallow trees. With more budget I would next try: (a) tuning the
per-member `colsample_bynode` grid more finely with more members, (b) stacking the ensemble with a
regularized logistic meta-learner on OOF member predictions, and (c) year-robust encodings computed as
*within-carrier/airport deviations* from the global mean (differencing removes year-level rate drift).
