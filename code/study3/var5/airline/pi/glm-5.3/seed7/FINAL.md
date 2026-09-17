# Final Report

**Best Eval AUC: 0.7579** (experiment #20, commit 20998c9) — up from the 0.7141 baseline (+0.0438).

## What mattered most

1. **Schedule-density features (the breakthrough, ~+0.017).** For each Origin / Dest / Carrier /
   Origin×Carrier / route, precompute the sorted array of scheduled departure minutes from the
   train slice; for each flight emit the fraction of its group's daily departures scheduled within a
   time window around (before / after) its own departure time. A flight's position inside its
   airport's "departure bank" is the dominant, year-stable delay predictor. Before-side windows
   (the upstream queue the flight inherits) beat symmetric windows; the after side and the
   symmetric variants turned out redundant and were pruned.
2. **Route-level schedule shape (~+0.010).** The same densities computed per Origin>Dest route.
   Route *identity* categoricals and route outcome statistics badly hurt transfer, but the route's
   *schedule shape* (banks) is physical and transfers perfectly — the sharpest persistent-vs-
   year-specific distinction of the whole run.
3. **Exponential-decay queues (~+0.0014, plus engineering).** Soft-kernel version of the
   before-density: sum of exp(−gap/τ) over earlier scheduled departures (τ ∈ 30/60/120 min),
   computed via prefix sums of exp(v/τ) per group. Replacing the O(groups×rows) per-group
   `np.where` loops with one factorize+argsort pass made the whole pipeline ~4.5× faster
   (96 s → 14 s), which is what allowed the extra features to fit the 120 s limit.
4. **Dropping the Month categorical (~+0.0017 early).** Month identity is a 2005 artifact; the
   day-of-year plus sin/cos harmonics carry the seasonality without memorizing the year.
5. **Time-of-day physics + capacity retuning.** DepTime/hour/minute_of_hour, per-airport
   time-of-day ECDFs, evening up-weighting (hour ≥ 15 × 2), and a 5-member XGB ensemble
   (d7, n160, lr .05, mcw 60, colsample .7, α=1) re-tuned after every major feature change —
   the optimal capacity grew each time the feature set got richer.

## What did not help

1. **Year-specific identity/statistics:** route categoricals, target/route encodings, monthly
   flight counts, hashed routes — all hurt transfer (trained on 2005, evaluated on 2006).
2. **Pairwise-ranking objective (`rank:pairwise`):** unsupported kwargs in the sklearn API;
   native `xgb.train` segfaults with one giant query group.
3. **Per-hour score recalibration** (disastrous, −0.04: the hour offsets are the signal itself),
   arrival-bank density at estimated arrival time, carrier queue-share ratios, wider route
   windows, second-order doy harmonics, heterogeneous/larger ensembles, random subspaces,
   row subsampling, and density ratios — all neutral or worse.

## With more budget

I would keep mining the schedule-shape family: per-day-of-week and per-season variants of the
decay queues (currently too noisy at 100 k rows, but a 1 M-row train slice would make them
viable), two-dimensional decay queues (origin × destination time-of-day), a learned soft
attention over each group's departure list instead of fixed kernels, and a small stacked
meta-model over members with different τ feature subsets. A second direction is modeling the
*arrival* side properly: reconstructed scheduled arrival times with arrival-bank densities,
which only failed here because arrival-time reconstruction was too crude. Finally, capacity
has re-tuned upward after every feature addition and never plateaued — with headroom from the
fast grouping, deeper trees (d8+) with more rounds deserve another pass.
