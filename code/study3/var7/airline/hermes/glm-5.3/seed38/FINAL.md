# FINAL — airline dep-delay XGBoost (autoresearch benchmark)

Best Eval AUC: **0.7429** (baseline 0.7141, +0.029). Validated: `CONTRACT OK`, predict_proba path reproduces 0.7429.

## The 3-5 changes that mattered most

1. **Engineered time features** (DepHour/Minute, cyclic sin/cos, Night/Evening flags, HourxDow grid): 0.7141 -> 0.7174.
2. **Regularization for the 2005->2006 time shift** (depth 4-5, subsample 0.7, colsample_bytree 0.4-0.6, min_child_weight 10, reg_lambda 10): -> 0.7241. Any capacity added without this reg hurt eval.
3. **Schedule-congestion counts** (flights per origin/dest x 15-min slot, per airport-hour, per route-slot, shares of daily traffic): -> 0.7377. The single richest feature family; structure of the schedule generalizes across years.
4. **Carrier-rotation counts** (carrier x origin/dest/route x 15-min slot — proxy for aircraft turnaround chains): -> 0.7417, and the final inbound-bank pressure (origin ops 30-90 min before departure): -> 0.7429.
5. **Bagged XGBoost ensemble** (8 depth-wise members with varied seed/depth/colsample, 85% row bags, plus 2 lossguide members; recency-weighted samples): -> ~+0.001 on top.

## 3 things that did not help

1. **Target encodings fitted on 2005 delay rates** (route/origin/dest/hour smoothed TEs): 2005 delay rates drift by 2006 — consistently at or below baseline features.
2. **More trees / low LR with early stopping on a 2005-tail split**: early stopping picked ~1800 rounds and overfit (0.7186); fixed 300 rounds x lr 0.07 stayed best.
3. **Cascade/CDF proxies, shrinkage formulas, 5-min slot granularity, sqrt counts, rank-averaging, monotone-style tweaks**: all neutral or worse; the plain 15-min counts already carried the signal.

## What I would try with more budget

Recover per-flight schedule context: with tail numbers or the full 2005+2006 schedule I would build true aircraft chains (previous flight's delay is the dominant predictor of the next), and per-airport rolling departure-bank utilization at 10-minute resolution computed from the *evaluation-year schedule itself* (allowed if the holdout frame can be passed as a whole). I would also try grouping counts by season (month x slot) to absorb seasonal schedule changes, a two-stage model (delay-rate model per airport-day + flight-level residual), and calibration of member weights on a 2005-holdback month. CPU budget was never binding (1.9k/18k used) — the experiment count was; with more runs I would grid the slot-window width for bank pressure (only tried 2/4/6) and the recency-weight slope jointly.

Run summary: 40/40 experiments, 0 crashes/ooms at final config, all in ~24s each, ~1922 CPU-s total.
