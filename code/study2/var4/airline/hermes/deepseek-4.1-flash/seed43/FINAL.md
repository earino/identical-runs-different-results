# FINAL — autoresearch XGBoost (airline delay, 2005 train / 2006 eval)

**Best Eval AUC: 0.7465** (baseline 0.7141, +0.0324). Kept commit: `fa53c9d` "2400 trees lr0.007 depth10" (HEAD).
`./validate.sh` -> `[validate] eval AUC via predict_proba (target column removed): 0.7465` / `[validate] CONTRACT OK`.
Budget used: 40/40 experiments, ~15 min wall clock, 2078 of 18000 CPU-seconds.

## What mattered most

1. **Fine-grained time-of-day as an unordered categorical** (the single biggest lever, +0.017 across three runs).
   `DepTime` was split into 15-minute bucket categories (96 levels) plus an hour category and a red-eye flag
   (`DepTime >= 2300 or < 500`), so the model can learn an arbitrary per-bucket delay profile instead of
   stacking binary thresholds on a numeric clock. 24 levels 0.7232 -> 48 levels 0.7312 -> 96-144 levels 0.7329;
   288 levels (5-min) and raw DepTime as a category (1341 levels) both overfit and regressed.
2. **Regularization, then capacity** — the two only work together. `min_child_weight=20, reg_lambda=5,
   subsample=0.8, colsample=0.7` gave +0.002 over unregularized; with that in place the depth ladder was
   monotone upward (6 -> 0.7452 at depth 8 -> 0.7462 at depth 10) and 1600 trees at `lr=0.01`, then 2400 trees at
   `lr=0.007`, added the last +0.0003. Earlier, unregularized depth 7 with 2000 trees was *worse* than 30 trees.
3. **Carrier x time-of-day interaction categorical** (`UniqueCarrier` x 30-min bucket, ~960 levels) +0.0053,
   the one entity interaction that transferred; it encodes carrier-specific daily delay waves.
4. **Schedule-bank features fitted on train only** (+0.0027): for each Origin and UniqueCarrier, the share of its
   departures in the flight's 30-min bucket and the cumulative share already departed before that bucket
   (a congestion / "how deep into the operating day" proxy, no target involved).
5. **Raw frequency counts** of Origin/Dest/carrier/route (log1p) + keeping raw `DepTime` alongside the engineered
   time features: small but positive (+0.0006), and they hold the baseline's information while the new features
   add to it.

## What did not help (all reverted)

- **High-cardinality memorization of 2005-specific identifiers**: `Origin_Dest` route as a categorical (-0.0070),
  target-encoded delay rates for carrier/origin/dest/route (out-of-fold in training, -0.0013), `DepTime` as a raw
  categorical (-0.015), carrier x origin (-0.005), day-of-week x time (-0.005), month x time (-0.011). Because
  train is 2005 and eval/holdout are 2006, anything that memorizes individual entities or calendars transfers
  badly; only *time-structured* signal (daily schedule shape) transfers.
- **Pure capacity without regularization** and **extra regularization at fixed depth** (`min_child_weight=100,
  reg_lambda=20` at depth 8 was -0.0011; depth 5 was -0.0009).
- **Model averaging**: a 3-seed ensemble of the best config, and a 2-config (depth 10 + depth 8) blend, were both
  flat (0.7416 and 0.7464 vs 0.7465 single) at double the training cost — this model is already low-variance.
- **Dropping the coarse redundant time features** (numeric `tod`, `hour`, `minute`, `hour_cat`, 30-min buckets;
  -0.0008) — kept them, since they are nearly free and the loss, though small, was consistent.
- Dest/route bank shares (-0.0012) and origin x time interaction (flat).

## What I would try with more budget

The remaining gap looks like distribution shift rather than model capacity: train AUC is ~0.98 while eval AUC is
0.7465, and every attempt to encode 2005-after-the-fact structure (target encodings, entity pairs, calendar
interactions) lost ground. I would attack the shift directly: (a) fit the final model on a *time-ordered* slice
of 2005 with explicit recency weighting, or ensemble models trained on disjoint 2005 quarters, so the model
sees the most recent regime; (b) build "2005 vs 2006-normalized" features — e.g. per-airport share/cumulative
schedule features at finer temporal resolution (quarterly), which are already the most transferable family;
(c) explore the schedule-density direction further with leave-one-day-out validation inside 2005 to get a
selection signal that is not eval.csv, since the tiny eval differences (<0.001) below which I discarded changes
may hide real gains; (d) try `grow_policy="lossguide"` with `max_leaves` and `max_bin` tuning, which is a
different bias/variance shape than fixed depth; and (e) revisit ensembling only after a genuinely different
model family (different feature subsets or objectives) exists, since seed/config averaging of near-identical
trees added nothing.
