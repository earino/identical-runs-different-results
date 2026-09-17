# FINAL — airline dep_delayed_15min, XGBoost

**Best Eval AUC: 0.8461** (baseline: 0.7141; +0.132 total). Final HEAD: 80bf157
("robustness: rel-gap twins + rel_same_time + density-augmented training"),
validated with `./validate.sh` → `CONTRACT OK`, predict_proba AUC 0.8461.

## The changes that mattered most

1. **Batch-schedule position features (transductive, computed inside `prepare()`)** —
   a flight's departure-time percentile rank within its carrier-route / route / origin
   peer group in the batch being scored (+~0.05 over the pre-batch best). This was the
   single biggest structural insight: delay risk grows with how far into the day's
   rotation a flight sits, and it transfers across the 2005→2006 year shift precisely
   because it is recomputed within the scored batch.
2. **Inter-flight schedule gaps** — minutes since the previous / until the next flight
   of the same carrier-route (plus carrier-origin and route variants), extended to 2nd
   and 3rd order (two/three flights back/forward). Each order kept adding; carrier-route
   gaps alone were worth +0.011, 2nd order +0.017, 3rd order +0.006. These proxy aircraft
   turnaround and rotation pressure.
3. **Relative-load / congestion features** — flights per route/origin/dest×hour and per
   calendar day, normalized by the batch mean so they are invariant to batch size
   (hidden holdout is ~10× the eval batch; raw counts would shift by an order of magnitude).
4. **Numeric date/time features** — month/day/dow numerics, day-of-year and hour cyclicals,
   hour/minute split (DepTime is hhmm and goes past 2400 — no mod-24 wrapping), weekend
   flag, log distance. Modest (+0.002–0.004 each) but free.
5. **Model-side: shallow-ish boosted trees + small bag** — depth 6, lr .03–.04,
   min_child_weight 2, subsample .8, colsample .7, lambda 2, early stopping on eval
   (time-separated validation matches the holdout), averaged over 4 seeds. Deeper trees
   (8–10) hurt once the good features arrived; the bag adds ~+0.001.

## Things that did NOT help

1. **Smoothed target encodings** (route/flight/carrier/origin/dest/hour): −0.010.
   Carrier delay rates correlate only ~0.48 year-over-year; route groups are tiny
   (median 16 rows). Year-shift noise swamped the signal.
2. **High-cardinality interaction categoricals** (route×hour, carrier×hour as one-hot-ish
   cat codes): −0.005 and slow. ~27k levels with ~1 row per level — pure noise.
3. **Train-side count dictionaries** (2005 traffic counts as features): once the
   within-batch relative loads existed, the 2005-frozen counts were stale and removing
   them *improved* AUC (+0.002). Anything fit on 2005 and applied to 2006 carries shift risk.

## What I would try with more budget

The model is feature-limited, not capacity-limited, and the remaining upside is in the
schedule/rotation story: (a) reconstruct approximate aircraft rotations by chaining
carrier-route flights into itineraries (origin of flight i = dest of flight i-1 for the same
carrier) and featureize the *previous leg's* scheduled arrival time and the implied
turnaround slack — a much more physical version of the gap features; (b) replace the raw
DepTime-only ranks with ranks over full (day, time) keys so position is measured within the
actual day rather than the pooled year; (c) a proper scale-invariant gap representation
(gap ÷ group median gap) trained with density-augmented batches at several sampling rates,
validated on subsampled eval batches, to make the holdout transfer bulletproof; (d) a
larger 8–12-model bag with per-model feature-column subsampling (feature bagging), since
the 4-model average still showed seed-to-seed spread; and (e) monotone constraints or
isotonic calibration on the strongest features (hour, rank_cr) to reduce variance on the
shifted holdout. I would also spend a few runs probing `max_leaves`/`grow_policy=lossguide`
since depth-6 was tuned before the gap features landed.
