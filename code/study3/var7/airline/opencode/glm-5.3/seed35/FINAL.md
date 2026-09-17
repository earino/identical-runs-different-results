# Final report — airline XGBoost (autoresearch harness benchmark)

**Best Eval AUC: 0.7408** (baseline: 0.7141, +0.0267). Best commit: `f364b2e`
("randomized hyperparam jitter across members"); validation prints `CONTRACT OK`
and reproduces 0.7408 through `predict_proba()` on a target-less DataFrame.

## What mattered most (in order of impact)

1. **Smoothed, out-of-fold target encoding** of high-cardinality keys — carrier,
   origin, dest, route, and dep-time bins (15-min, 5-min, exact minute) — fitted on
   train only, applied inside `prepare()`. Biggest single jump once combined with
   capacity (0.7147 -> 0.7232).
2. **Physically-motivated interaction TEs** — origin×hour, dest×hour and especially
   carrier×hour (top feature by gain, 0.155), plus coarse 2-hour-period variants.
   Stable airport/carrier time-of-day delay structure transfers across years.
3. **Bagged XGBoost ensemble** (6-8 members, subsample/colsample 0.8, averaged
   probabilities): +0.006 over the single model, and it unlocked depth — depth
   6 -> 10 kept helping only under bagging (0.7346 -> 0.7390).
4. **Per-member decorrelation**: each member gets its own OOF fold assignment for the
   TE features (and its own hyperparameter draw in the final config) — decorrelating
   the TE noise and the trees themselves gave the last +0.001.
5. **Capacity**: 30 rounds/lr 0.1 -> 450-500 rounds/lr ~0.04, depth 10. Required the
   low-variance TE features to pay off; on raw features more capacity only overfit.

## What did NOT help

1. **Route as a raw 5000-level categorical** — actively harmful (-0.008 on eval);
   ~20 rows/level memorizes 2005 route noise. Route only helps as a smoothed TE.
2. **Calendar-day TEs** (month×day-of-month, carrier×month, origin×month): the
   worst result of the run (-0.0085). 2005's calendar-day rates don't transfer to
   2006 (weekday alignment shifts) — a clean leakage-trap lesson.
3. **Micro-regularization tweaks**: min_child_weight=5 (-0.0008), TE alpha 10
   (-0.0009), subsample 0.7 (-0.0009), heterogeneous depth mixes, wider
   jitter (d9-11/550r), and 10 lighter members — all neutral-to-worse; leaf
   regularization is not the bottleneck on this data.

## With more budget

I would attack the two structural limits: (a) member count is capped by the 120s
wall-clock limit (~8 members at depth 10), so I'd profile and speed up the feature
build (polars, cached TE frame) to fit 12-16 members, and try full
random-hyperparameter-search ensembles (like the winning jitter, but larger);
(b) the TE quality itself — hierarchical/empirical-Bayes shrinkage (partial pooling
of origin×hour toward origin and toward hour, rather than a flat global mean),
and time-aware OOF folds (split by month within 2005) that mimic the 2005->2006
shift so the encodings are optimized for year-level generalization. Finally I'd
verify robustness with repeated seeds and a month-blocked CV on train before
trusting any further eval.csv gains of <0.001.

Final `train.py` layout: 8-member jittered bagged XGBoost (depth 10, 400-500
rounds, lr 0.035-0.045, per-member OOF target encodings, per-member
hyperparameters), averaged in `predict_proba()`.
