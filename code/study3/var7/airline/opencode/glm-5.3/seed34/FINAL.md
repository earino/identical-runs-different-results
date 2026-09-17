# Final Report

**Best Eval AUC: 0.7444** (experiment #10, commit 4423f9d; re-validated, `CONTRACT OK`).

Final model: a single large XGBoost (hist, depth 12, 2000 trees, lr 0.015, reg_alpha 5,
min_child_weight 1, colsample_bytree 0.7) on engineered features including smoothed
out-of-fold target encodings and label-free congestion counts.

## Trajectory

| # | Commit | Change | Eval AUC |
|---|--------|--------|----------|
| 1 | 86ab46d | Baseline XGBoost (30 trees, d6) | 0.7141 |
| 4 | 7da848a | Time/date FE (tod, doy sin/cos, minute) | 0.7191 |
| 5 | f311883 | L1 regime (alpha 10, d8) + harmonics | 0.7285 |
| 6 | d5ab09c | 6-model ensemble + OOF target encodings | 0.7348 |
| 7 | 2cbea55 | 10-member ensemble (TE + harm6 variants) | 0.7354 |
| 8 | 62c3dfb | Congestion counts (half-hour bins), all members | 0.7406 |
| 9 | b5596a7 | Single big model (TE, harm6, d12, n2000, mcw1) | 0.7424 |
| 10 | 4423f9d | harm9 + reg_alpha 5 | **0.7444** |
| 11 | bcbca49 | n2500 lr0.012 (tie, reverted to faster #10) | 0.7444 |

Experiments 2-3 (early stopping on an internal split; route-concat categorical) were worse and reverted.

## The 5 changes that mattered most

1. **Regularized capacity over defaults**: strong L1 (reg_alpha 5-10) unlocks deep trees
   (d9-d14) and many slow rounds (2000-3000, lr 0.01-0.015), plus colsample_bytree 0.7.
   The final model is one such large slow learner (better than ensembling smaller ones).
2. **Label-free congestion counts** (+0.005): flights per origin/dest per half-hour bin,
   per origin/dest per hour, and per dow-hour, computed on train only. They encode
   schedule congestion that transfers 2005→2006, unlike label-fitted features.
3. **Smoothed OOF target encodings** (+0.004-0.006 with L1): carrier/origin/dest/route
   delay rates with count smoothing; train rows get 5-fold out-of-fold values, predict rows
   get full-train maps. Plain (leaky) TE overfits and hurts; under L1 it adds signal.
4. **Time decomposition FE** (+0.005): tod harmonics (sin/cos k<=9), hour categorical,
   minute-of-hour, day-of-year cyclic, distance x hour interaction.
5. **Ensembling as a stepping stone** (+0.005 cumulatively): equal-weight multi-config
   averaging revealed the big-model optimum (ensemble members individually improved as
   they got bigger); the final single model matches or beats the 10-model ensemble at
   half the inference cost.

## 3 things that did NOT help

- **Target-encoder-style categorical concat** (route string categorical) and naive
  full-map TE on train rows: both leak/overfit and lost ~0.009 AUC.
- **Bagging/seed diversity**: row subsample, seed replicas of the same config, one-hot
  encoding (≈ native categoricals) — all neutral or worse.
- **Covariate-shift sample reweighting** (train-vs-eval density ratio via domain
  classifier, unlabeled rows only) and dart boosters: solos worse, ensembles unchanged.
  Extra count features (quarter-hour bins, route/carrier congestion, ratios, logs) also
  added nothing over the half-hour set.

## What I would try with more budget

The 2005→2006 year shift is the core difficulty: label-heavy structure (TE, route
categoricals) overfits, while label-free structure (congestion, schedule slots)
transfers. With more budget I would push the label-free direction harder: learn
per-airport/quarter-hour "slot delay profiles" by pooling multiple airports via
clustering or hierarchical smoothing; build origin-dest congestion interacted with
seasonality (month x slot); and test a two-level model where a small model's OOF
prediction of congestion-features-only is fed as a feature. I would also tune the
ensemble of 2-3 big models (side studies showed 0.7443-0.7446 for d12/d14 pairs) with a
120s-safe runtime, and probe lower learning rates (0.008-0.01, n3000+) with alpha 3-5,
since every decrease in lr so far was flat-to-positive.
