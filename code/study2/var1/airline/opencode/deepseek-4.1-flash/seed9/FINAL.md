# Final report — airline delay (XGBoost, AUC)

**Best Eval AUC: 0.7426** (commit `5610ba3`, experiment #39). Baseline was 0.7141.
Metric: AUC on `data/eval.csv` (2006), model trained on `data/train.csv` (2005).

## Changes that mattered most

1. **Calendar/time-of-day feature engineering.** Parsed the `c-<n>` string columns into numeric
   ordinals (Month/DayOfWeek/DayofMonth) and derived `hour`, `minute` and `tod` (`hour*60+minute`)
   from the integer `DepTime`. Numeric ordinals let shallow trees split season/weekday sensibly and
   `tod` was by far the highest-gain feature (gain ~419 vs ~78 for carrier).
2. **Categorical time encodings and low-cardinality interactions.** Treating time as a categorical
   (`hour_cat`, then a 15-minute `tod_bin`) and adding `carrier_hour` and `dow_hour` interactions
   gave the single biggest jump (0.7261 -> 0.7307). These are low-cardinality (<= ~500 levels) and
   prove stable across the 2005->2006 shift.
3. **Depth-diverse XGBoost ensemble.** Averaging probabilities over one model per depth in `[2..14]`
   times 3 seeds (39 models) was worth ~+0.006 over the shallow-only ensemble. Depth diversity
   matters more than seed diversity; deeper members kept adding signal up to the wall-clock limit.
4. **Strong regularization for the single-model regime.** Because eval is a later year, capacity had
   to stay small: 100 trees, `max_depth=4`, `learning_rate=0.1`. 250+ trees or `max_depth=6` alone
   consistently lost to the more regularized model.

## Things that did not help

- **High-cardinality route/airport interactions** (`Route` = Origin_Dest, `origin_hour`, `dest_hour`,
  `month_hour`, `dow_tod`, `carrier_tod`): all overfit the 2005 slice and lost 0.005–0.02 AUC.
- **Out-of-fold target encoding** of UniqueCarrier/Origin/Dest: 0.7209 vs 0.7213 baseline — the native
  XGBoost categorical splits already captured this.
- **Cyclic sin/cos calendar features and monotone constraints** on time-of-day: cyclic features cost
  ~0.002; the monotone constraint on `tod` was much worse (0.7112), confirming the delay/time
  relationship is non-monotone (which is exactly why `tod_bin` as a categorical helped).

## What I would try with more budget

The depth trend was still positive when the clock ran out (2–13: 0.7423, 2–14: 0.7426; 2–15 timed out
at 121 s). The clear next steps are to make the deep ensemble fit the time budget: lower
`n_estimators`/subsample for the deepest members, or bag subsets of depths across runs, so more
seeds and deeper trees can be averaged. Beyond that, time-aware (expanding-window) target statistics
for high-cardinality interactions, and calibration/stacking of the depth-diverse members, are the
most promising directions. Any gain must be checked against the fact that only the hidden 2006-slice2
holdout matters, so I favored low-cardinality, stable features over eval-specific ones.
