# FINAL — airline dep-delay XGBoost (autoresearch harness)

**Best Eval AUC: 0.7441** (baseline: 0.7141, +0.0300). HEAD = exp27 (commit a54d582),
validated with `./validate.sh` → `CONTRACT OK` (predict_proba reproduces 0.7441 with
the target column dropped, as the hidden-holdout scorer will call it).

28 experiments logged; CPU-seconds budget (18,000) was the binding constraint
(wall clock had ~2h left, 12 experiment slots left). All kept steps are in
`git log` on branch `experiment`.

## The changes that mattered most

1. **Fix the DepTime representation (hhmm → real time).** `DepTime` is an hhmm integer
   (1159 and 1200 are 1 minute apart, not 41). Converting to true minutes-since-midnight
   plus a 15-min-bucket categorical (levels fit on train only) was the single biggest
   jump: 0.7207 → 0.7292. The raw hhmm integer stayed in as a (now secondary) feature.
2. **Schedule-pressure count encodings (target-agnostic).** Flights-per-(origin,
   15-min block), per-(origin, hour), and per 15-min block, all counted on train only:
   0.7292 → 0.7329. Congestion is the physical mechanism behind departure delay, and
   counts generalize across years unlike target encodings.
3. **Heterogeneous XGBoost ensemble.** 6 members averaging predictions: two feature
   variants (v0 base, v1 adds dest/carrier/route pressure), per-model feature drops
   (Distance / SeasonSin,Cos / Dest / freq encodings), different seeds, all d10
   n600 lr0.025 sub0.8 col0.8. Diversity from *feature-variant mixing* was worth
   ~+0.004 over plain seed ensembling (0.7355 → 0.7393 at the time).
4. **Depth up once features were rich.** The baseline overfit at depth 6, but after the
   minutes/bucket/pressure features, depth 8 → 10 with recency weights gained
   0.7393 → 0.7438. More signal per row changed the capacity calculus.
5. **Linear recency weighting.** 2005 training rows weighted by month
   (ramp 0.5→1.0, steeper 0.25→1.0 on some members) because late-2005 rows are closer
   in time to the 2006 evaluation: worth ~+0.002, robust in direction across probes.

## Things that did NOT help (all reverted)

1. **Target encodings (smoothed route/origin/dest delay rates).** Catastrophic on the
   2005→2006 shift: route target-encode as categorical dropped AUC to 0.583; smoothed
   TE variants ~0.705-0.708. They memorize 2005-specific delay rates that do not
   transfer. Count encodings (used here) are the target-agnostic sibling that does work.
2. **Route as a 4.2k-level categorical.** Same failure mode: the model latches onto
   route-level 2005 noise (0.583-0.70). Origin/Dest as separate categoricals are enough.
3. **Monotone constraints on time/pressure features** (0.7369 vs 0.7438), holiday-distance
   features, day-of-week×hour and hour×carrier interactions, 5-min buckets (too fine),
   row-bagging members, early stopping on a same-year holdout (stops way too early:
   183 rounds → 0.7109 — the 2005 holdout cannot see the year shift), and adding a 6th
   depth-6 member (neutral). Quadratic recency ramps also hurt (mild linear is right).

## What I would try with more budget

The two highest-value unknowns are (a) whether the hidden holdout's 2006-slice2
distribution rewards the same tradeoffs as 2006-slice1, and (b) richer schedule
structure. With more budget I would: cross-validate every keep/discard decision on
*time-blocked* folds (train Jan–Oct 2005, validate Nov–Dec 2005) instead of the single
2006 slice, to stop overfitting eval.csv; engineer per-airport scheduled-departure
*density curves* (smoothed origin×hour flight mass, and the rank of each flight within
its airport-day departure sequence — queue-position features); try a two-stage residual
model where a coarse hour/origin model's residual is fit by a second XGBoost; and grid
the recency-ramp slope per ensemble member jointly with depth, since those two
interacted (steep ramps fit shallower models best). CPU was the binding limit, so I
would also switch the ensemble to `xgboost.train` with early stopping on a *weighted*
time-blocked fold to spend the same CPU on fewer, better-shaped trees.
