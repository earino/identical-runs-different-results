# Final report — airline (XGBoost, scenario 2)

**Best Eval AUC: 0.7402** (baseline 0.7141, +0.0261). Final commit `2e99380` (exp38).
Validated: `validate.sh` → `CONTRACT OK`, predict_proba reproduces 0.7402 on eval.csv with the target dropped.

## Final model
5-member XGBoost ensemble (depths 8/6/5/4/3, different seeds), lr 0.03, max_depth 8 base, subsample/colsample 0.8,
reg_lambda 10, tree count from early stopping (auc) on an internal 80/20 split of train, final members refit on
100% of train with month-recency weights (1 + 1.0·(month−1)/11). Features: DepTime → cyclical sin/cos, hour cat,
30-min-bin cat, red-eye flag; month sin/cos + numeric month/day/dow + weekend; distance + log; native-categorical
carrier/origin/dest with train-fixed levels.

## Changes that mattered most
1. **Time-of-day categorical bins** — hour-of-day as native categorical (+0.0071) then 30-min bins (+0.0053). By far
   the largest single wins; hourly delay rhythms are stable across years.
2. **Lower learning rate 0.05→0.03 with early stopping** (+0.0024) — smoother fit transfers better over the year gap.
3. **Ensembling by depth diversity** (2 seeds +0.0010; depths 8/6/5 +0.0012; +depth 4 +0.0013; +depth 3 +0.0011) —
   depth-diverse members beat same-config seed members.
4. **reg_lambda 1→10** (+0.0014) — leaf regularization against temporal drift.
5. **Month recency weighting** of the final fits (+0.0006/+0.0001) — later-2005 months weighted up toward 2006.

## Things that did NOT help
- **Route (Origin_Dest) as native categorical**: −0.0080 — 6000 levels memorize 2005 route effects.
- **Target encoding** of carrier/origin/dest (k=50): −0.0018 — encodes drift-unstable location/carrier effects.
- **Calendar categoricals** (month/day-of-month/dow): −0.0124 — specific calendar-day effects don't repeat across years.
- **hour×dow interaction categorical**: −0.0072; **15-min bins**: −0.0003; **distance bins + dist×hour**: −0.0011.
- Hyperparameter probes around the optimum: depth 10 (−0.0007), depth 6 (−0.0015), mcw 10 (=), subsample/colsample
  0.7 (−0.0030), gamma 0.5 (−0.0004), max_bin 512 (−0.0014), lr 0.025 (−0.0002), colsample 1.0 (−0.0006), 6th member
  (=), per-member tree scaling (−0.0013), ES iterations ×1.25 (−0.0018), count encoding (timeout), ES patience 100
  alone (timeout; kept later only after capping ES trees at 1200, which cut runtime 92s→67s at equal AUC).

## Theory of the data
Train (2005) and eval (2006) are time-separated with a large generalization gap (same model: 0.759 on held-out 2005
vs 0.714 on 2006). Everything encoding *identity* (specific routes, airports, carriers, calendar dates) overfits
2005-specific patterns; everything encoding *stable physics/schedules* (time of day, day-of-week coarsely, distance)
transfers. Hence: categorical time-of-day + strong regularization + recency weighting + diverse ensembling.

## With more budget
- Per-member early stopping (each member picks its own tree count) — costlier but principled.
- Model blending with different feature subsets (time-only vs full) for extra diversity.
- Stacking: logistic meta-learner on member probabilities (needs out-of-fold predictions).
- Careful per-feature-group colsample; a small grid over recency-weight strength × reg_lambda.
- More ensemble members at lower per-member cost (fewer trees each, tuned jointly).
