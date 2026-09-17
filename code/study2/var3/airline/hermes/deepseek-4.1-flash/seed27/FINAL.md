# FINAL — airline dep_delayed_15min (XGBoost, 2005 -> 2006 time shift)

**Best Eval AUC: 0.7669** (commit `58e86b1`, 30 experiments run, baseline `59e2cd0` = 0.7141, so **+0.0528**).
Contract check: `./validate.sh` -> `CONTRACT OK`, eval AUC trough `predict_proba()` with the target column
removed = 0.7669, run time 70 s (limit 120 s).

Final model: single `xgboost.XGBClassifier`, `n_estimators=600, max_depth=20, learning_rate=0.01,
min_child_weight=1, reg_lambda=2, subsample=0.8, colsample_bytree=0.25, tree_method="hist",
enable_categorical=True`. Every feature is produced inside `prepare()`, and every statistic (category levels,
density counts, queue-position lists) is fitted on the training rows only, so `predict_proba()` reproduces the
exact transformation on unseen rows.

## Changes that mattered most

1. **Schedule-density ("congestion") features** — counts of flights sharing the same airport-hour, dest-hour,
   route-hour, carrier-airport and carrier-hour, taken from the training rows. Biggest single jump:
   0.7287 -> 0.7421, and it kept paying off later (see 3 and 4). These are pure schedule statistics with no
   target involved, so they do not drift when the year changes, unlike delay-rate encodings.
2. **Deep, strongly feature-subsampled trees on top of those features** — `max_depth=20, min_child_weight=1,
   colsample_bytree=0.25, subsample=0.8, reg_lambda=2`. On the original features more capacity clearly
   *hurt* (30 trees beat 100 beat 400: 0.7141/0.7117/0.7045); once the stable density features existed the
   ordering flipped and depth 16-20 plus tiny column subsampling was worth ~0.006-0.008.
3. **Queue-position (rank) features** — for each row, how many training flights with the same
   origin-hour / dest-hour / route-hour / carrier-origin-hour are scheduled at or before it (a proxy for the
   departure queue in front of the flight). +0.0035 (0.7590).
4. **Sliding-window density** — counts of training flights within 30/60/120 min of the scheduled time at the
   same origin / dest / route, which smooths the arbitrary hour-bucket boundary. +0.0025 -> 0.7620, plus a
   second batch of scales and a carrier-hour rank -> 0.7636.
5. **Calendar parsing + clock features** — `c-<n>` -> integer for Month/DayofMonth/DayOfWeek, and
   hour/minute/minutes-since-midnight from `DepTime` (handling post-midnight codes like 2620). +0.0034 at
   baseline capacity; hour-of-day is by far the strongest single feature (52 % of splits' gain).

## Things that did NOT help

1. **Route / origin / dest target encodings and route as a high-cardinality categorical** — `Route`
   (Origin_Dest, ~7k levels) cost 0.008 and smoothed target encodings of Origin/Dest/Carrier/Route cost
   0.010. Delay rates are year-specific; the 2006 evaluation punishes anything that memorises 2005 rates.
2. **Extra capacity on the raw feature set** — 100 and 400 trees at depth 6-8 without density features fell
   to 0.7117 / 0.7045 vs 0.7141 for 30 trees; internal 2005 validation rose while eval fell, i.e. classic
   year-shift overfitting. Also `grow_policy="lossguide"`, month-recency sample weighting, and pooling rare
   categories were neutral-to-worse.
3. **Piling on correlated density variants** — dominance ratios (carrier share of an airport-hour, route
   share of the hour) and a second batch of 30/180/240-minute window scales both *lowered* eval AUC
   (0.7642 / 0.7640 vs 0.7669): with `colsample_bytree=0.25` the extra near-duplicate columns crowd out the
   informative ones. `min_child_weight=0` (0.7666 at 111 s) and `max_bin=512` (0.7667) were within noise and
   not kept.

## What I would try with more budget

The whole gain came from *schedule structure* rather than from delay-rate statistics, which suggests the
remaining headroom is in even richer schedule context: a full "flight bank" reconstruction (sorted
departure lists per airport with arrival/departure balance and the carrier's inbound leg feeding this
aircraft, i.e. true propagation chains), rolling aircraft-rotation features (previous scheduled leg of the
same tail number — not available here, but derivable if a tail-number column existed), and explicit
hub-connection exposure (share of the airport-hour's departures connecting to a bank). On the modelling
side, the natural next step is a bagged ensemble of 5-10 deep models with different seeds and column
subsets: two-seed averaging was neutral on eval but should reduce seed variance on the 1M-row hidden
holdout, and the earlier seed sweeps showed ±0.0008 spread. Finally, this configuration sits at
`colsample_bytree=0.25`, `max_depth=20`, `min_child_weight=1` — all three were at the edge of what I
tested, so a proper randomised search over that corner (with time-based internal validation to guard
against the year shift) is the obvious next move.
