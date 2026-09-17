# Final report — airline delay prediction (XGBoost)

## Result

**Best Eval AUC: 0.7526** (baseline 0.7141, +0.0385). Validated: `./validate.sh` prints
`CONTRACT OK` and reproduces 0.7526 via `predict_proba` on the target-stripped eval frame.
Best commit: `8c72f32` (4-model XGBoost ensemble, depths 1/6/11/16).

## Changes that mattered most

1. **Time-of-day features.** `hour`, `minute`, `tod`, and `sin/cos(tod)` were the single largest
   signal; raw `DepTime` (hhmm) carries delay rate rising monotonically from ~4% at 5am to ~83% at
   midnight. Replacing the integer with derived/cyclical time lifted AUC from ~0.7167 to ~0.7179.
2. **`carrier × hour` interaction** (`UniqueCarrier_hour`, native categorical). The strongest single
   feature (+0.009): each carrier has its own delay profile across the day (scheduling/banks).
3. **`hour × distance_bin` interaction** (10 train-quantile distance bins). Flight length changes the
   time-of-day delay profile (+0.006); equal-frequency bins beat fixed edges.
4. **One-hot vs native categorical for low-cardinality columns** (`max_cat_to_onehot=32` on Month,
   DayOfMonth, DayOfWeek, UniqueCarrier) — more stable across the 2005→2006 time split (+0.0014).
5. **Depth-diverse XGBoost ensemble** (depths 1/6/11/16, averaging probabilities). Averaging shallow
   and deep trees reduced variance and gained ~+0.004 over the best single model; ensembling beats any
   individual depth, and deeper models are worthwhile only as ensemble members.

## Changes that did not help (reverted)

- **Route / high-cardinality interactions** (`Origin_Dest`, `carrier_dest`, `carrier_origin`,
  `carrier×dayofmonth`, `origin×month`): all hurt, often badly (0.70–0.73). Route-level delay patterns
  do not transfer across the year-split.
- **Target encoding** of Origin/Dest/Carrier/route (full-train or as a complement to native
  categoricals) was neutral-to-negative versus native categorical handling.
- **Row/feature subsampling** and extra regularization (`min_child_weight`, `reg_lambda`,
  `colsample_bytree`) reduced both single-model and ensemble AUC; full-data, unregularized trees
  generalized best. Cyclical month/day-of-week and 8/12/16-bin distance variants were also neutral.

## What I would try with more budget

The model is near a plateau around 0.75 and gains are increasingly within eval noise. With more
compute I would (a) search the ensemble composition more systematically (per-member tree counts,
weighted averaging, rank-averaging, and a larger set of depths) rather than by hand; (b) tune the
categorical split hyperparameters (`max_cat_threshold`, one-hot cutoff) jointly with depth under a
proper internal time-aware validation instead of selecting on eval; (c) test richer stable temporal
aggregates — e.g. holiday indicators and departure-time-within-the-day normalized by that
route/carrier's schedule — while explicitly avoiding features fit on the input frame; and (d) confirm
on a larger held-out 2006 slice that the interaction features (`carrier×hour`, `hour×distance`)
transfer, since they are the main source of the gain and the main generalization risk.
