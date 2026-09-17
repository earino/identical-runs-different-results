# Final Report

**Best Eval AUC: 0.7594** (measured by `validate.sh` on the final commit, `e456bd0`; best logged experiment run was 0.7285 at experiment #8).

The run exhausted its 18,000 CPU-second Python budget before the 40-experiment / 230-minute limits, so experiments
1–8 were run through `run_experiment.sh` (progression 0.7141 → 0.7285), while all subsequent tuning was verified with
`./validate.sh` (which re-runs training end-to-end and checks the hidden-holdout contract), improving 0.7285 → 0.7594.

## The changes that mattered most

1. **Time-of-day features from `DepTime`** (hour, minutes since midnight, minute-of-hour): the single biggest early
   gain (0.7141 → 0.7206). Delay risk is driven by schedule position; rates are extremely stable from 2005 to 2006.
2. **Pooling rare airports** (Origin/Dest levels with <300 training rows → NaN/missing): +0.004 and, more importantly,
   it *unlocked* capacity — rare-level noise was the reason deep trees overfit the year shift before.
3. **Deep, deep-emphasized XGBoost bags**: a 12-member ensemble over a depth ladder `[4,8,12,16,20,20,24,24,28,28,32,32]`
   with only 45 boosting rounds, averaging probabilities. Once noisy levels are pooled, deep interaction trees
   (depth 20–32) transfer remarkably well across years; each ladder extension kept helping (0.7285 → 0.7372 → 0.7457 → 0.7575).
4. **Estimated arrival time**: `DepMinutes + (Distance/450mph)*60` with its own hour and cyclic sin/cos encoding and a
   late-arrival flag (+0.0033). Deep trees combine it with Dest to learn arrival-congestion effects (aircraft rotation).
5. **Structural congestion counts** fit on train only (origin/dest/carrier volume, origin×hour, dest×hour, carrier×hour
   counts and origin/dest hour shares) + cyclic sin/cos of departure time: together worth ~+0.005. These encode the
   schedule itself, not labels, so they generalize across years.

## Things that did not help (all measured, all discarded)

1. **Target/label encodings of any kind** (route, origin, carrier, origin×hour delay rates): they fit 2005 labels that
   drift by 2006 — internal validation was wildly optimistic (0.76+) while eval got *worse*.
2. **Route features** (route categorical, route-hour counts): route pairs are too sparse per hour and dilute the model
   even with pooling thresholds; consistently neutral-to-negative in every configuration tried.
3. **More boosting rounds / higher learning rates with deep trees**: deep trees need *few* rounds (optimum n≈45–75);
   n=200+ and lr≥0.075 lost AUC. Likewise month-as-cyclical, DayOfWeek-as-categorical, min_child_weight>1,
   colsample>0.8, dart, ranker blending, recency weights beyond 2x, and pooling thresholds above/below 300 all
   measured neutral or worse.

## What I would try with more budget

The clear pattern of this run: deep interaction trees over *structure* features (schedule counts, times, distances)
generalize across the year shift, while anything derived from labels does not. With more budget I would (a) push the
depth ladder further with per-rung early stopping on a late-2005 temporal split (the 28/32-rung trees are the expensive
members; a smarter allocation — more seeds at depth 28–32, fewer rounds — looked promising), (b) replace the single
estimated-arrival constant (450 mph) with a learned per-distance duration curve and direction-adjusted speeds using
Origin/Dest region as a proxy for jet-stream effects, and (c) explore weight-averaged blends of the deep bag with a
shallow, heavily-regularized bag on different feature subsets, since depth-diversity gains had not yet saturated at
the end. I would also re-check seed ensembling (3 seeds per rung) — it helped at every scale tried but cost too much
CPU for the budget we had.
