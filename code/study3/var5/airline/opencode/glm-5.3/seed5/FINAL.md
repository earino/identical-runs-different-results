# FINAL

**Best Eval AUC: 0.7646** (experiment #13, commit 16914f4; baseline was 0.7141, +0.0505).

## The changes that mattered most

1. **Drift-robust training regime** (0.7141 → 0.7384): drop calendar features, keep only
   stable schedule features (DepTime/TOD/Hour/Distance + UniqueCarrier/Origin/Dest as
   native categoricals), fixed round count (no early stopping — random-split validation
   lies under the 2005→2006 time shift), deep trees (d16-20) + `reg_alpha=1.0` + `max_bin>=512`.
2. **Dest categorical restored** in the deep+L1 regime (→ 0.7483) and **label-free traffic-count
   features** fit on train only: origin/dest hourly departure counts and shares, route counts,
   hour share, arrival-hour congestion (AHcnt) and its share (AHfrac), carrier-origin counts (→ ~0.7552).
3. **Feature-subset ensemble members** (numeric-only, cats-heavy): +0.002 over full-feature-only members.
4. **Shallow long-boosted members** (d3 lr.05 n1200, d2 lr.05 n1500, d3-numeric, d1-stump lr.02 n3000):
   the single biggest ensemble jump (+0.002-0.003); they transfer far better under the time shift and
   decorrelate from the deep members.
5. **Logit-space averaging** of member probabilities (+0.001) and **circular time features**
   (SinTod/CosTod, SinArr/CosArr for estimated arrival time) (+0.002 combined) → 0.7646.

## Things that did not help

1. **Any calendar feature** — Month/Day/Dow numerics, year harmonics (−0.019!), seasons, DOW flags,
   holiday windows: 2005-specific seasonal bins memorize noise that does not repeat in 2006.
2. **Target encodings and interaction categoricals** — OOF-smoothed TE per hour/origin/carrier,
   route/origin-hour/dest-hour/carrier-origin cats, rolling congestion features: neutral or harmful.
3. **Capacity/regularization tricks that reduce variance on the wrong axis** — early stopping,
   subsample/colsample/bagging, min_child_weight>1, one-hot encoding, seed averaging (hist is
   deterministic), rank:pairwise objective, gblinear/dart members, median/trimmed/rank aggregation,
   shallow-weighted averaging (deep members are essential).

## With more budget

I would build a drift-aware validation scheme (train on 2005-H1, validate on 2005-H2 as a
pseudo-2006 shift) so that keep/discard decisions stop relying on a single 100k-row eval slice;
mine more schedule-structure features that repeat year-over-year (route network overlap,
carrier fleet/season changes); engineer dedicated smooth features for the shallow members
(which generalize best); try out-of-fold stacking with the shallow members' predictions as
meta-features; and grid the shallow-member family more systematically (depth 1-4 × lr ×
max_bin × feature subsets), since that family produced the largest, most reproducible gains.

## Contract check

`./validate.sh` prints `CONTRACT OK` (eval AUC via `predict_proba` on unseen rows: 0.7646).
