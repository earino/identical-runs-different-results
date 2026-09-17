# Final report — airline departure-delay AUC (XGBoost)

**Best Eval AUC: 0.7470** (experiment #36, commit `71bcd6f`), up from the 0.7141 baseline.
Contract validated: `CONTRACT OK`.

## What mattered most

1. **Removing year-specific calendar features.** The single biggest structural win was dropping
   `Month` and `DayofMonth` (0.7153 -> 0.7221). A within-2005 holdout scored 0.76 while 2006 eval
   scored only 0.71, showing heavy 2005-specific overfitting; calendar dummies memorise 2005 weather
   and do not transfer. `DayOfWeek` was kept because it does transfer.
2. **Out-of-fold smoothed target encodings of stable groupings.** Encoding means of `Origin`, `Dest`,
   `UniqueCarrier`, `DayOfWeek`, and especially their interactions with scheduled departure hour.
   `route x hour` gave the largest single feature jump (0.7289 -> 0.7341); `UniqueCarrier x route`
   added another +0.0027. Training-frame encodings are computed out-of-fold (5-fold) to avoid leakage.
3. **Dropping native high-cardinality categoricals once the encodings existed.** With TE features in
   place, removing native `Origin`/`Dest`/`UniqueCarrier` categoricals improved AUC (0.7383 -> 0.7404)
   and simplified the model.
4. **Group-support (count) features.** Adding the training count of each TE group lets the model
   discount low-support encodings; this was the second-largest feature gain (+0.0035 plus +0.0007).
5. **A 16-model XGBoost ensemble** over decorrelated seeds/depths/subsampling for variance reduction
   across the distribution shift (+0.003 over a single model).

## What did not help

- **`route` as a native categorical** (0.7024) and frequency encoding — the 4198-level interaction
  overfits badly; it only helps when smoothed/target-encoded together with hour.
- **Cyclic day-of-year seasonality** (sin/cos) and stronger L2/leaf-weight regularization — both were
  neutral or slightly negative.
- **Very high-order target encodings** (4-way route x dow x hour) and minute-level encodings — too
  sparse, they added noise. More ensemble members (18) and logit-vs-probability averaging were neutral.

## With more budget

I would push on (a) proper **stacking**: train a small XGBoost meta-learner on out-of-fold predictions
of the base ensemble, which was infeasible here under the 120 s/experiment cap; (b) **per-group adaptive
smoothing** rather than one global pseudocount, and encoding richer statistics (e.g. variance and
support-aware shrinkage) for the high-cardinality route/carrier-hour groups; (c) **schedule-level
features** such as departure-minute patterns and turnaround proxies; and (d) a principled search over
the TE feature set (forward selection using a time-based validation split) to prune the ~20 encodings
to the subset that genuinely transfers 2005 -> 2006.
