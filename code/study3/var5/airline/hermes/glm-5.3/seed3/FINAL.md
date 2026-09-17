# FINAL — autoresearch XGBoost (airline delay)

**Best Eval AUC: 0.7432** (baseline was 0.7141, +0.0291). Final config: experiment #33, commit `a9c5d2d`
(ne300 + lr0.1 + reg_lambda 8 + reg_alpha 4 + gamma 0.1 + colsample 0.55, ens of 12 seed/depth-diverse
XGBClassifiers on time-bucketed interaction features).

## The 5 changes that mattered most

1. **Time-bucketed entity interactions (biggest single win, +~0.008 total).** Concatenating
   Origin and UniqueCarrier with the departure-time bucket as extra categoricals —
   `Origin_hour`, `Carrier_hour`, plus 30-minute and 20-minute variants (`ATL_1345`, `UA_1330` style keys).
   These let trees split directly on "this airport around this time of day", capturing
   schedule/congestion effects no numeric feature could express. Levels pinned from train only.
2. **Seed/depth ensemble of 12 XGBoost models (+~0.003).** Averaging probabilities over
   seeds [42,1,7,13,...] × depths [3..8]. Halves seed noise (±0.001 single-model) and adds
   a small consistent gain; cheap because each model is only 300 trees.
3. **Strong regularization tuned for the 2005→2006 time shift (+~0.006 over default).**
   reg_alpha 4, reg_lambda 8, gamma 0.1, colsample_bytree 0.55. The eval year is different from
   the train year, so anything that can memorize 2005 hurts in 2006; the sweep consistently
   favored stronger L1/L2 and smaller column fractions.
4. **Dropping the Month/DayofMonth categorical encodings (+~0.002).** Keeping them only as
   integers removed a date-memorization path; the DayOfWeek categorical stayed (weekday effects transfer).
5. **DepTime decomposition + distance interactions (+~0.0007 over raw).** minutes-of-day, sin/cos
   day-cycle, hour, dist×cos, log/sqrt distance, and mean-departure-time-relative features
   per Origin/Carrier (fit on train only).

## 3 things that did NOT help

1. **Target encoding of any kind** (carrier/origin/dest/route delay rates, smoothed with
   priors, hour×dow rates): −0.005 to −0.03. 2005 delay-rate rankings simply do not transfer
   to 2006; raw category splits generalize better than rate features do.
2. **Deeper trees / more capacity without regularization** (depth 8, 400-500 trees at default
   reg): −0.002 to −0.03. Every overfitting handle was punished by the year shift.
3. **Count/congestion features** (flights-per-day, origin×hour counts): catastrophic (−0.04) —
   they encode the 2005 sampling pattern, not transferable structure. DART, early stopping on a
   time split, bagging, one-hot small cats, dest×hour buckets, and ens16 were all neutral-to-worse.

## What I would try with more budget

The feature axis that kept paying was finer, more diverse interaction of *who/where* with
*when*; the model axis is saturated. Next I would (a) add a second model family by
hyper-parameter space (e.g. very shallow depth-2 stumps at 1500 trees, averaged in) to widen
the ensemble's hypothesis space; (b) replace the single fixed bucket set {60,30,20} with
learned, per-origin adaptive time windows (quantile-based on train counts); (c) exploit
recency — refit a final model with mild sample weights toward late-2005 to better match the
2006 distribution; and (d) test `hist` multi-strategy bagging where each of the 12 ensemble
members sees a different 80% column subset including a different *bucket granularity subset*,
since the 20/30/60-minute features are partially redundant and per-member diversity there
showed a small positive signal in screening.

*Budget state at finalize: 33/40 experiments used; 18001/18000 CPU-s (Python budget
exhausted — the interpreter now refuses to start); ~139 min wall clock remaining. Finalized
per program.md because the CPU budget is spent, which is the binding constraint.*
