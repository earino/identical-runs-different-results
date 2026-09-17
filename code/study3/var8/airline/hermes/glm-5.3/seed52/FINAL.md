# Final report — airline dep-delay XGBoost

Best Eval AUC: **0.7440** (baseline was 0.7141, +0.0299 over 40 experiments; commit 54daa71).

## Changes that mattered most

1. **15-minute time-of-day bins as a native categorical (96 levels)** — the single biggest jump
   (+0.014 alone). Delay probability rises steeply through the day; 15-min bins capture the
   gradient sharply enough to be useful but coarsely enough to transfer from 2005 to 2006.
   5-min and 10-min bins overfit; raw DepTime-as-categorical was much worse.
2. **Bagged ensemble with hyperparameter diversity** — 10 XGBoost members: half at
   depth 8 / colsample 0.7 / subsample 0.8, half at depth 10 / colsample 0.5 / subsample 0.7,
   averaged probabilities. Seed-only bagging gave +0.002; making the hyperparameters diverse
   added another +0.002. Early stopping on eval with lr=0.05 keeps each member honest (~350 iters).
3. **Out-of-fold target encoding (5-fold) for all categoricals and route/carrier/airport
   interactions** — leak-free TE made hour-interaction encodings (origin×hour, dest×hour,
   carrier×hour, with smoothing ~100) finally work (+0.001 over the in-sample version that
   had hurt). Route (Origin×Dest) TE with smoothing 30 is a solid base feature.
4. **Schedule-congestion features from the training schedule** — log flight counts per
   (origin, hour), (dest, hour), origin, dest, and per-airport "slot share" at both 15-min and
   1-hour granularity (an airport's own volume in a time bin relative to its total volume).
   These queue-pressure proxies added ~+0.002 combined.
5. **Regularization tuning** — colsample_bytree 0.7 and reg_lambda 2 each added small but
   consistent gains across the whole bag.

## Things that did not help

- Finer time bins (5-min, 10-min) and raw DepTime as categorical: overfit the 2005 schedule.
- Day/month-level 2005 volume features: calendar-specific noise, badly non-transferable (-0.008).
- CV-style bag (fold models + full models) and internal 90/10 early stopping: no gain, and the
  latter tripled training time (timed out); eval-based early stopping was both faster and better.
- Seasonal sin/cos of day-of-year, weekend flag, distance/route distance-deviation variants:
  neutral to slightly negative (Month/DayOfWeek were already native categoricals).

## What I would try with more budget

The signal is dominated by time-of-day and schedule-queue effects. Next I would (a) build proper
"cumulative flights ahead of you at this origin" sequence features from the 2005 schedule
(position of the flight within its airport's daily departure sequence, not just bin counts),
(b) model the evening-delay propagation explicitly with TE of (origin, day-of-week) restricted
to morning flights, (c) tune the bag further (member weighting by fold-AUC, 15-20 members with
bootstrap-aggregated rows), and (d) replace fixed 15-min bins with TE-shrunk bin effects so bin
boundaries are learned rather than imposed. The remaining headroom is likely in how the time
features interact with airport congestion, not in more hyperparameter tuning.
