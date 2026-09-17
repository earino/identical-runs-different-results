# Final report — airline delay XGBoost (autoresearch)

Best Eval AUC: **0.7440** (baseline 0.7141, +0.0299 over 40 experiments; best commit `5472e42`).

## The changes that mattered most

1. **Carrier x time-of-day interaction categoricals** (biggest single gain, ~+0.015 total).
   `UniqueCarrier x 30-min departure slot` was the sweet spot: hour granularity (carrier x hour) gave
   +0.0119, refining to 30-min slots added +0.0047, but 15-min slots and Origin/Dest x time variants
   overfit and were dropped.
2. **Heterogeneous ensemble of 13 XGBoost models** (+0.0026, then +0.0007 scaling 5 -> 14 members):
   varied max_depth (6-10), subsample (0.7-0.9), colsample_bytree (0.6-0.9), min_child_weight,
   reg_lambda, plus 4 lossguide members (max_leaves 48-128). Probability-mean averaging.
3. **colsample_bynode=0.8** on every member (+0.0005) — cheap extra decorrelation that also
   complemented per-tree column sampling.
4. **Scaling trees per member** 200 -> 250 -> 300 -> 350 -> 380 (+0.0017, +0.0010, +0.0006, +0.0003):
   reliable monotone gains until the 120s wall killed 400 trees. 380 trees at lr 0.05 is the edge.
5. **Minute-of-day continuous clock** (+0.0014 over raw hhmm) and **route aggregates** (route flight
   count, mean distance, circular departure-time deviation from the route norm; +0.0002-0.0004 each).
   Also fixing month/dom/dow to real numerics (the "c-4" strings were silently NaN in the numeric
   features) plus seasonal sin/cos.

## What did NOT help

1. **Target/mean encoding** — tried twice (plain smoothed TE, and TE of the strongest interaction
   carrier x slot30): both *lost* ~0.001. The 2005->2006 shift means group delay rates drift; trees
   with categorical splits transfer better than train-fitted rates.
2. **Early stopping on a random train slice** (2000 trees, best_iter=161): -0.0011. The validation
   slice is 2005 data, so it selects for year-specific noise, not for 2006 generalization.
3. **Date-keyed features**: airport congestion counts (flights per airport-hour keyed by exact date),
   carrier x day-of-week, hour x day-of-week — all hurt (-0.002 to -0.007). Anything keyed to a
   specific calendar date/weekday pattern of 2005 does not transfer to 2006. Finer 15-min time slots
   also hurt (too sparse per carrier). DART members: -0.0008 and 2x slower.

## What I would try with more budget

The model is compute-bound, not idea-bound: every tree increase helped until the 120s timeout, and
dropping the weakest member freed enough time for +20 trees. With more time per experiment I would
(a) shrink the ensemble to 8-10 members and push each to 800-1200 trees at lr 0.02-0.03 (lower lr
with more rounds has more capacity headroom than 380 x 0.05), (b) explore leave-one-out/count
encodings with *per-year* stability weighting to fix the TE transfer problem, (c) add origin x carrier
and route x slot interactions pruned by holdout validation, and (d) tune max_bin and
categorical one_hot depth for the interaction features, which are the dominant signal and likely
still under-split at depth 6-9.

Best model: 13 members, 380 trees, lr 0.05, colsample_bynode 0.8, minute-of-day + carrier x slot30
interactions + route stats + seasonal encodings. Validated: CONTRACT OK, 0.7440 on eval.csv via
`predict_proba`.
