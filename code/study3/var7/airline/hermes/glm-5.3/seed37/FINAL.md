# autoresearch XGBoost — final report

Best Eval AUC: **0.7359** (baseline 0.7141, +0.0218), reached at experiment #40
(commit 073cadd, HEAD of branch `experiment`). Validation: `CONTRACT OK`
(predict_proba on a raw frame with the target removed reproduces the score).

## The changes that mattered most

1. **Departure-time quantile banks as a native categorical feature (80 bins).**
   The single biggest win (+0.006 in one step, and it unlocked depth scaling later).
   Delay probability is a fine-grained, non-monotonic function of scheduled time
   (0.04 at 5am, 0.83 at 11pm, with a dip at 4am); a categorical bank lets shallow
   trees carve that curve without spending depth on linear thresholding.
2. **Tree depth 5→6→7 (with the dt_bank feature present).** Depth 4→5→6→7 gave
   +0.0033/+0.0015/+0.0007, but only *after* dt_bank existed — before it, depth 5
   hurt (0.7164). The categorical gave the trees the structure to spend capacity on.
3. **Slow learning rate + regularization + 5-seed bag.** lr 0.02 × 1200 rounds
   beat both fast (0.06) and slower (0.01); d4/mcw20/sub0.7/col0.7/λ2 + averaging
   5 seed-varied XGB models added ~+0.002 total and stabilized eval results.
4. **Schedule-volume features (train-fit lookups).** (Origin, hour), (Dest, hour),
   (Carrier, hour) scheduled counts — absolute, relative to the airport's mean
   (size-invariant), and interacted with time-of-day. Standalone AUC 0.55 on both
   years; ~+0.001 in-model. Fit on train only, so the scale holds on any frame.
5. **Time/season FE: hour cyclic, minofday, day-of-year cyclic, weekend flag,
   dist×hour.** Modest but each transfer-checked across the 2005→2006 shift.

## What did not help (reverted)

- **Target encoding of any kind.** Light smoothing (m=20) with hour-granular keys
  collapsed AUC to 0.6745; even heavily smoothed route/origin/dest TE (m=500-1000)
  scored 0.7131 vs 0.7209 without it. 2005 route rates memorize noise that 2006
  does not repeat, and the TE column distorts splits around the raw cats.
- **More capacity without structure.** 3000 rounds × lr 0.05 with early stopping
  (0.7129), depth 5 before dt_bank (0.7164), and heterogeneous 5-variant ensemble
  (0.7178) all underperformed the simple bag.
- **Recency sample weights** (upweighting later 2005 months): 0.7180 vs 0.7182.
  Month identity effects visibly do not transfer between the two years.
- **Monotone constraints on hour** (0.7177): the 4am dip makes the true relation
  non-monotonic; the constraint cost accuracy.
- **160-bin dt_bank** (0.7290 vs 0.7304 at 80 bins): bins finer than the signal
  start to memorize.
- **Banked congestion quantiles** (0.7303 ≈ 0.7304): no gain, dropped for simplicity.

## With more budget

I would (a) grid the dt_bank granularity jointly with depth and min_child_weight,
since those interact — the bin count and the depth are effectively one hyperparameter
pair; (b) replace the single 80-bin bank with per-airport time-of-day curves
(clustered, train-fit) so busy hubs get their own rhythm instead of one global bank;
(c) add an XGB rank/pairwise objective variant to the bag for ensemble diversity;
(d) probe whether eval and the hidden holdout respond differently to depth 7+ by
cross-checking a 2005-internal time split, since the depth gains may partly be
eval-specific; and (e) try 8-10 seed bags at depth 6-7, where variance reduction
still looked under-exploited when the experiment count ran out.
