# FINAL — airline delay (dep_delayed_15min) XGBoost, 40/40 experiments

**Best Eval AUC: 0.7408** (baseline 0.7141; +0.0267). Best commit: the final `train.py`
(5-model XGBoost ensemble, 600 trees, lr 0.028). `./validate.sh` prints `CONTRACT OK` and
`predict_proba` on the raw eval frame (target column dropped) reproduces 0.7408.

## What mattered most

1. **Dropping the drift-only calendar columns (Month, DayofMonth)** — +0.0056 (0.7141 -> 0.7197).
   Train is 2005, eval/holdout is 2006. Per-month delay rates correlate only r=0.59 between the
   years and are essentially flat in 2006 (0.444-0.556), so those columns only give the trees room
   to fit 2005-specific seasonality. This was the single most robust change.
2. **Explicit carrier x time-of-day crosses** — +0.0089 in total (0.7207 -> 0.7254 -> 0.7406).
   `UniqueCarrier` and clock-of-day are the two most year-stable signals, and their interaction is
   something shallow trees only partly discover, so it is materialized as a categorical cross.
   Resolution mattered a lot: hour (480 levels) helped, adding a half-hour grid (960 levels) helped
   a lot more, quarter-hour (1920) was too sparse and lost ground.
3. **A 5-model ensemble instead of one model** — +0.0022 (0.7312 -> 0.7334). Members differ in
   depth, colsample and seed, then probabilities are averaged. Cheap decorrelation with no extra
   capacity per member.
4. **Capacity up, then heavily regularized and subsampled** — depth 4 -> 5 -> 6 with
   `min_child_weight=20`, `reg_lambda=5`, `colsample_bytree=0.6`, `subsample=0.9`,
   `n_estimators=600`, `lr=0.028` (0.7162 -> 0.7408 across the chain). Both `depth=3` and
   `colsample_bytree=0.4` were worse.
5. **Clock features replacing raw DepTime** — +0.0007 (0.7197 -> 0.7204), plus traffic-volume
   (log-frequency) encodings for origin/dest/carrier/route — +0.0003. Small individually but
   drift-free: volume and time-of-day transfer across the year gap far better than target rates do.

## What did not help

1. **Target encoding** (smoothed, k=50, for carrier/origin/dest/route, fitted on train only):
   0.7101 vs 0.7197 — a large loss. The 2005 target rate for a category is a poor estimate of its
   2006 rate, and the model trusts it; XGBoost's own categorical partitioning handled those columns
   better (removing Origin/Dest identity entirely cost 0.0149, so the identity carries real signal
   even though its target rate does not transfer).
2. **High-cardinality route and airport x hour crosses** (`Origin_Dest`: 0.7045;
   `Origin x hour` / `Dest x hour`: 0.7200; origin/dest x coarse hour-bin: 0.7239). Thousands of
   sparse cells turn into pure overfitting. Only dense crosses (carrier x time) paid off.
3. **Tighter categorical splitting / more trees**: `max_cat_threshold=16` cost 0.0051, and 700
   trees cost 0.0033 (150 trees also cost 0.0023) — the optimum is narrow at ~300-600 trees with
   `colsample_bytree=0.6`. Growing the ensemble from 5 to 9 members bought nothing, and
   `subsample=0.7`, `max_depth=7` and `min_child_weight=50` were all neutral to slightly negative.

## With more budget

The dominant error source is the 2005 -> 2006 distribution shift, not underfitting, so I would
attack it directly rather than add capacity. Concretely: (a) make the clock x carrier crosses
*smoothed numeric* features instead of categoricals, e.g. each (carrier, half-hour) cell scored by a
shrunken rate computed out-of-fold on training rows only, so dense and sparse cells are handled on
one scale; (b) search the time-grid resolution properly (10/20/30-minute grids, and per-carrier
grids) since half-hour won by a wide margin and the optimum was not bracketed; (c) add
flight-frequency-at-departure-hour numbers (schedule congestion) which transfers across years;
(d) weight training rows by recency proxy (later 2005 months) if any ordering can be recovered;
(e) validate candidate changes on a held-out slice of the training year *ordered by month* rather
than only on eval.csv, to tell robust gains from eval-specific ones before committing to them.
