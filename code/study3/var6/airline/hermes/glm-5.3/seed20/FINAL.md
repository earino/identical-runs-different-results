# FINAL — airline dep_delayed_15min, XGBoost

## Best Eval AUC: 0.7392 (baseline 0.7141, +0.0251)

Final model: 5-bag XGBoost ensemble (seed-varying 90/10 ES splits), depth 10, lr 0.03,
subsample 0.6, colsample_bytree 0.6, colsample_bylevel 0.7, colsample_bynode 0.8,
reg_lambda 4.0, reg_alpha 1.0, min_child_weight 1, early stopping 60.

## Changes that mattered most

1. **Departure-order features within (Origin, date)** (exps 15-16, 0.7236 -> 0.7284):
   rank-percentile of the flight's DepTime among all departures at its origin that day,
   span position between the day's first/last departure, and their products with
   time-of-day. Captures delay propagation through the day at an airport; +0.005 alone.
2. **Heavy regularization** (exps 24-25, 0.7287 -> 0.7354): min_child_weight 1,
   subsample 0.6, colsample_bytree 0.6 + colsample_bylevel 0.7, reg_lambda 4,
   reg_alpha 1 (+ max_cat_to_onehot 50). The 2005->2006 shift punishes low-bias models;
   this was worth +0.007 in two steps.
3. **colsample_bynode 0.8** (exp 40, 0.7360 -> 0.7392): extra node-level feature
   sampling decorrelated the 5-bag ensemble; biggest single step after regularization.
4. **Bagged ensemble of 5 XGB models** (exp 8, +0.002) averaging seeds/splits; the
   single-model -> 5-model move established the base recipe everything else built on.
5. **Time features + arrival-side ranks**: true minute-of-day with sin/cos (exp 7,
   +0.002) and estimated-arrival rank within (Dest, date) using distance-derived
   travel time (exp 37, +0.0005).

## Things that did NOT help

1. **Target encoding (both naive and out-of-fold, smoothed)**: 0.6759 / 0.6919 —
   catastrophic. Label-dependent encodings do not transfer across the 2005->2006 shift.
2. **Frequency/count encodings and label-free schedule profiles** (exp 5, 36): no gain
   (0.7125 / 0.7352); raw counts from a 100k sample are noise for the holdout.
3. **Seasonal/holiday distance features** (exp 34): 0.7353, no gain over 0.7355.
   Also neutral: carrier x dow/month interactions, route-date/origin-hour ranks,
   departure-gap and turnaround-gap features, diverse per-bag hyperparameters.

## What I would try with more budget

The two dominant signals are within-day departure order at the origin and heavy
regularization against the year shift. I would push three directions:
(a) a proper flight-chain reconstruction — per (carrier, origin), link departures to
plausible incoming aircraft via estimated arrival times and tail-number-style matching
to get true turnaround slack, instead of crude rank proxies;
(b) two-stage stacking with OOF XGB predictions as features to a second XGB layer,
which the compute budget never allowed;
(c) a small random/grid search around the regularization cliff (lambda 4-8, bynode
0.7-0.9, subsample 0.5-0.7) since three of the last four gains all came from that
region, and calibrating the number of bags (5 -> 8) at lower per-model cost.
I would also validate feature families against a simulated density shift (train on
sparse subsample, evaluate on dense frame) — the raw rank features passed that test,
which is why they stayed.

## Log

40/40 experiments used; wall clock 161 min remaining at exhaustion; CPU 12970/18000 s.
All feature engineering lives in prepare()/_base_features(); encoders and statistics
are fit on data/train.csv only. validate.sh prints CONTRACT OK.
