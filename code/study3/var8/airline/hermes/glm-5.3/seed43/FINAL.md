# Final report — airline delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7612** (baseline: 0.7141, +0.0471), reached in 40 experiments.
Validated: `./validate.sh` prints `CONTRACT OK`; `predict_proba` is chunk-invariant
(max prob diff 0.0 between full-frame and 1000-row chunks).

## Final recipe (train.py)

1. Smoothed target encoding (TE, smoothing 50) of categorical keys fitted on train only:
   carrier, origin, dest, route, route x hour, origin x hour, dest x hour,
   carrier x 250-mile distance bucket, origin x 15-min slot, route x 15-min slot,
   route x minute-of-hour, carrier x 15-min slot.
2. Out-of-fold (5-fold) TE for the training matrix; full-train TE maps at predict time.
3. Hierarchical TE shrinkage: each fine key's TE blended 50/50 with its parent
   (route_moh -> route_m15 -> route_hour; carrier_m15 -> carrier; origin_m15 -> origin_hour;
   dest_hour -> route_hour) — denoises sparse high-cardinality buckets.
4. TE support counts (log1p) for the interaction keys (confidence features).
5. Train-only aggregates: route/origin avg distance & avg DepTime, route/origin/dest/carrier
   counts, origin/dest diversity, distance vs route average.
6. Numeric date/time features (month/dom/dow, minute-of-day + sin/cos, minute-of-hour).
7. Recency sample weights (late-2005 months weighted up: 0.25 + 0.75 * m/12).
8. Bagged ensemble of 32 XGB models (16 fold-split seeds x 2 model seeds), averaged
   probabilities; each model: n_estimators=360, depth 8, min_child_weight 10,
   lr 0.066, colsample 0.6, subsample 0.85, hist.

## The changes that mattered most

1. **Target encoding of categorical keys** (carrier/origin/dest/route): 0.7141 -> 0.7162.
2. **Route x hour-of-day TE interaction** (+0.0095 in one step, 0.7192 -> 0.7287) and the
   origin/dest x hour variants that followed — time-of-day x airport is the core signal.
3. **Fine-grained time buckets**: origin/route/carrier x 15-minute TE (+0.0106) and
   minute-of-hour features (+0.0033) — schedule granularity carries real signal.
4. **Train-only aggregate features** (+0.0046): counts, avg distance/deptime, diversity.
5. **Bagged ensembling + hierarchical TE shrinkage + recency weights** (the last ~+0.007
   together): variance reduction that should transfer to the 1M-row hidden holdout.

## Things that did not help (measured, then reverted)

1. Early stopping on an internal 2005 split — the split is a poor proxy for the 2006
   shift (0.7083/0.7094 vs 0.7141); fixed 30-480-tree budgets were better.
2. Native XGBoost categoricals alongside TE (0.7081) — trees waste splits on raw
   high-cardinality columns when TE values are present.
3. Calendar keys that don't transfer across the year boundary: day-of-month/route x dom
   (catastrophic: 0.6151), route x month, month x hour, day-of-year, month x dow.
4. 5- or 10-minute buckets (too sparse), month-block fold CV for TE, dart boosting,
   lossguide growth, max_bin tuning, per-key smoothing schemes, rank/logit averaging,
   extra ensemble members beyond 32 — all flat or worse on eval.

## With more budget

I would (a) model delay propagation explicitly: the biggest missing signal is the
previous flight's actual delay, which is unavailable here, but same-day earlier
departures at the same airport could be proxied with iterative prediction on the
holdout's own scheduled time order (transductive, so out of contract — instead I'd try
route x consecutive-flight scheduled-gap features); (b) tune the hierarchy blend per key
with cross-year (train on Jan-Jun 2005, validate on Jul-Dec) selection to make keep/drop
decisions less eval-dependent; (c) stack a second-level XGB on member predictions with
OOF internal validation; (d) search smoothing jointly with the hierarchy weights (only
the blend was tuned); (e) probe whether the hidden holdout's later 2006 months favor
stronger recency weighting (the eval window is only 2006-slice-1).
