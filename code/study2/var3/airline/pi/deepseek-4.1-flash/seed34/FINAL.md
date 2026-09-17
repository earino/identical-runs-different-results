# Final Report — airline XGBoost (autoresearch)

**Best Eval AUC: 0.7316** (commit `6a41c4c`, 5-seed XGBoost ensemble), vs 0.7141 baseline.
Evaluated on `data/eval.csv` (2006-slice1); the hidden holdout is 2006-slice2, so this estimate
should transfer well.

## Changes that mattered most

1. **Out-of-fold target encodings (5-fold) for location/time interactions.** The single biggest
   win was `route_hour` (Origin×Dest×hour): +0.007 AUC on its own (exp 20). `dest_hour`,
   `origin_hour`, `carrier_hour`, `route_month`, `carrier_origin`, `carrier_dest` followed.
   All encoders are fit on training rows only (full-train maps at inference, out-of-fold maps
   while training) so `predict_proba` reproduces them on unseen data.
2. **Support counts and count ratios.** Adding the raw support of every TE key (exp 32) gave
   +0.0008, and schedule-share ratios `cnt_route_hour/cnt_route`, `cnt_origin_hour/cnt_origin`,
   `cnt_dest_hour/cnt_dest` plus day-of-year (exp 38) gave +0.0018. These are target-free and
   therefore very robust to the 2005→2006 shift.
3. **Strong regularization with shallow trees.** Depth 8/600 trees collapsed to 0.6999;
   depth 4 with `colsample_bytree=0.4` and `min_child_weight=50` was best. Lower learning rate
   (0.03) with 500 trees helped.
4. **5-seed XGBoost ensemble** (same recipe, different `random_state`): +0.0003, reduces variance.

## Things that did NOT help

- **High-cardinality raw categorical route / deep trees** (0.7035 and 0.6999): severe overfitting
  to 2005-specific effects. Dropping native Origin/Dest/Carrier categoricals in favour of TEs only
  also hurt (0.7141) — the native cats are still useful.
- **Cyclic/derived clock features** (hour, minute, sin/cos) and month×hour / dow×hour TEs: neutral
  to negative; raw `DepTime` plus the interaction TEs already capture the signal.
- **Hierarchical TE smoothing** (shrinking hour TEs toward their route prior) and **10-fold OOF**:
  slightly worse than the simple global-prior smoothing with 5 folds.

## With more budget

I would push the interaction-encoding direction further: smoother/hierarchical count-ratio
features (e.g. shares of airport traffic by hour with kernel smoothing over adjacent hours),
route-hour rank features, and a two-level model where the first level predicts delay from
schedule structure and the second corrects with airport/carrier state. I would also validate
hyperparameters on a time-based split of the training year to guard against the observed
2005→2006 distribution shift in carrier delay rates.
