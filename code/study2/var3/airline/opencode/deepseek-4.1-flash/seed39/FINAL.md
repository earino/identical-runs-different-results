# Final report — airline departure-delay classifier

**Best Eval AUC: 0.7483** (commit `85fe553`, validated via `predict_proba` with the target column removed).

## What mattered most

1. **Ordinal date features instead of one-hot/categorical.** Month, DayofMonth, DayOfWeek as
   `c-<n>` strings were being treated as native XGBoost categoricals. Parsing them to integers
   (+0.0020) let trees split on ordered thresholds and transferred across the 2005→2006 gap.
2. **Time-of-day engineering.** Replacing the raw `hhmm` integer with `dep_min`, `dep_hour` and
   sin/cos features, and *dropping* the raw `DepTime` column (+0.0019), was the single largest
   feature block. Feature importance shows departure time dominates (~55% of gain).
3. **Frequency / congestion encodings** fitted on train only: per-route, per-(carrier,route) and
   per-(origin,hour) counts (+~0.002 cumulative). Faithful airport/route sizes are stable across years.
4. **Lossguide trees with `max_leaves` up to 8192, `max_bin=1024`, `min_child_weight=1`,
   `subsample=0.9`, `reg_alpha=1`.** This deep, low-bias recipe was worth +0.015 over the
   depthwise baseline and was by far the biggest model-side lever.
5. **Averaging 3 diverse XGBoost models** (leaves 4096/8192/2048 at slightly different lr/seeds)
   adds ~+0.001 and is the most robust part of the final model for the hidden holdout.

## What did not help

- **Native high-cardinality categoricals for route** (Origin_Dest): overfit badly at low capacity
  (0.7088) and timed out at high capacity.
- **Target encoding** (both in-sample smoothed and 5-fold OOF) for route/origin/dest/carrier:
  always slightly worse — delay propensities drift between 2005 and 2006.
- **Adding many sparse congestion features at once** (dest/carrier/route × hour): diluted the model
  and consistently lowered eval AUC.
- **More capacity via depthwise trees** (depth 7–14) plateaued ~0.726; lossguide leaves were strictly better.

## With more budget

The eval AUC is now well into the range where 100k-row bootstrap noise (~±0.002) dominates, so
further gains need either a larger evaluation signal or genuinely new information. I would try
(a) bagging many more lossguide seeds with a proper time-aware internal validation split to pick
leaf count per model, (b) two-way interaction encodings between carrier and time-of-day and between
origin and destination that are explicitly regularized, and (c) a stacked/averaged ensemble spanning
a wider `max_leaves`–`learning_rate` grid, keeping the contract's `prepare()` path so the hidden
holdout is scored through the exact same feature pipeline.

**Note on budget:** the 18,000 CPU-second Python limit was reached while screening hyperparameters;
the final validated run (and the kept model) completed before the limit and `validate.sh` reports
`CONTRACT OK`.
