# FINAL — airline delay XGBoost (scenario 2)

**Best Eval AUC: 0.7213** (baseline: 0.7141, +0.0072). Final commit: `dc19c9b` (5-model blend).

## What mattered most

1. **DepTime decomposition** (hour + minute numeric, cyclic sin/cos of time-of-day) — the single
   most valuable feature change (+~0.002 on its own configurations). Scheduled departure time is
   the dominant physical driver of delay; the raw hhmm integer encodes it poorly for tree splits.
2. **Shallow trees: max_depth 3** with 200 trees at lr 0.1 (vs baseline depth 6 / 30 trees).
   Consistent monotone win 0.7184 vs 0.7141. Deep trees memorize 2005-specific carrier/airport
   patterns that do not transfer to 2006; shallow trees capture smooth time/season structure.
3. **Ensembling diverse encodings of the same categoricals**: averaging XGBoost models trained
   on *native categorical* features vs *ordinal-encoded* features (+ a colsample-0.7 variant and
   two extra seeds) gained +0.003 over the best single model. The two encodings make different
   split errors, and their average is consistently better than either alone.
4. **Hour × DayOfWeek interaction categorical** (+0.0002 single-model) — delay depends on the
   joint hour/day pattern (e.g. Friday evening peaks), which neither feature splits alone.
5. **Tuning n_estimators** per member (200-300 trees) at lr 0.1 in the blend.

## What did not help

1. **Target encoding** of Origin/Dest/Carrier/Route (train-fitted, smoothed): 0.6995-0.6999 —
   clearly harmful, the 2005 mean-delay per airport is not the 2006 mean-delay per airport.
2. **Route (Origin→Dest) categorical**: 0.7056-0.7087 in every configuration tried. 4199 levels
   over 100k rows is mostly noise, and it overfits.
3. **Early stopping on a random 10% tail of train** (refit at best iteration): 0.7110. The
   in-year ES holdout selects too many trees for the year-shift problem.
4. **Frequency features** (Origin/Dest/Carrier/Route counts): 0.7181, no gain over the same
   model without them.
5. **Bagging** (5 bootstrap resamples, 100 trees each): 0.7155 — worse than one full-data model;
   row-subsampling drops signal, while *encoding* diversity (exp 32-38) adds it.

## With more budget

- Rank-average (logit-average) instead of arithmetic mean in the blend, and per-member weights
  tuned on a time-aware split (train on Jan-Aug 2005, validate Sep-Dec 2005).
- Quantile-binned DepTime (e.g. 15-minute buckets) as an alternative to hour/minute, and
  DepTime×Origin interactions for hub-specific wave structures.
- A per-carrier calibration layer (isotonic per UniqueCarrier) on top of the blend.
- Gains were flattening near 0.721; the eval-year gap (2005→2006) is the binding constraint,
  so the biggest upside would be any feature that is *stable across years* (calendar, schedule
  structure) rather than entity-level statistics.
