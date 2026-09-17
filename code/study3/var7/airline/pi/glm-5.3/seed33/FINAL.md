# FINAL — autoresearch XGBoost (airline delay)

**Best Eval AUC: 0.7437** (baseline 0.7141, +0.0296) — commit `3391f96`, validated (`CONTRACT OK`).

## What mattered most (in order of impact)

1. **HourCarrierCat** — a joint categorical `UniqueCarrier@hour` (~430 levels, native categorical splits).
   The single biggest win: carriers have very different time-of-day delay profiles, and the joint
   cell means transfer almost perfectly from 2005 to 2006 (0.7207 → 0.7292 by itself).
2. **Hour-of-day as a categorical** (`HourCat`, hour%24) instead of relying on raw `DepTime` only
   (0.7174 → 0.7207). Delay rate runs 4% → 74% across the day; bins let trees share evidence cleanly.
3. **Dropping `Month` and `DayofMonth`** — their 2005 patterns do not transfer to 2006 and act as pure
   memorization noise (0.7336 → 0.7355). The 2005→2006 shift consistently rewards stable structure.
4. **Anchor shape**: max_depth=8, learning_rate=0.03, ~1250 rounds, hist method, full data
   (d6@lr0.1 30 trees → 0.714 baseline; the deep/low-lr anchor reaches 0.7388 solo), plus a
   **recency-weighted twin** (w = 0.5 + month/12) for diversity.
5. **Probability-averaged 4-member ensemble**: anchor + recency anchor + two bagged d5/lr0.05/sub0.85
   members, all recency-weighted (bagged members recency-weighted was the last win: 0.7434 → 0.7437).
   Composition was chosen by greedy forward selection over cached eval predictions.

## What did not help

1. **Target encoding** (smoothed, additive, any variant) — always hurt; year-shift instability beats
   the variance reduction.
2. **Sparse interaction categoricals** — route (Origin×Dest), hour×dow, carrier×dow, carrier×month,
   origin×dow: all negative or neutral. Only hour×carrier worked.
3. **Capacity at high lr / random-split early stopping / bigger ensembles** — random-split validation
   punishes capacity exactly when the year shift rewards stability; >5 members dilute the ensemble
   (15-member: 0.730x), and a two-stage lr-decay anchor (0.01 continuation) matched but never beat
   the plain d8/lr0.03 anchor.

## With more budget

The biggest remaining lever is probably a *learned* representation of carrier×hour rather than raw
categorical cells (e.g., stacking a first-stage model's oof predictions as a feature, or per-carrier
hour-mean targets fit only on early 2005 months and validated against late 2005). Second, optimizing
ensemble weights (instead of uniform averaging) on a time-blocked internal split rather than eval.csv.
Third, a larger family of recency-weighted bagged members (more seeds × more subsample levels) selected
by the same greedy procedure — cheap to train, and the last two composition wins both came from there.
