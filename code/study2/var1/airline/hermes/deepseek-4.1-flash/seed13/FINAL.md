# FINAL — airline departure-delay (dep_delayed_15min), XGBoost

**Best Eval AUC: 0.7498** (experiment #40, commit `f83a103`), vs **0.7141** for the committed baseline
(+0.0357). Metric: ROC-AUC on `data/eval.csv` (2006 slice), trained on `data/train.csv` (2005 slice).
`./validate.sh` prints `CONTRACT OK` and reproduces 0.7498 through `predict_proba` with the target column
removed, i.e. all feature engineering lives inside the `prepare*()` path used for unseen rows.

Budget used: 40/40 experiments, ~165 min wall clock, 12.7k/18k CPU-seconds of Python compute.

## Final model

12 XGBoost members averaged, over two feature views:

* **Native view (7 members)** — the raw columns as XGBoost categoricals (`Month`, `DayofMonth`, `DayOfWeek`,
  `UniqueCarrier`, `Origin`, `Dest`), plus a `carrier_x_hour` categorical interaction, plus numeric time
  features (hour, minute, minute-of-day, sin/cos of hour and minute-of-day, day-of-week, month, day-of-month,
  weekend flag, log-distance). Tree structures span `max_depth` 4/8/9/10 with `colsample_bylevel` 0.6-0.7.
* **Target-encoded view (6 members)** — the same numeric block plus 11 smoothed, out-of-fold target encodings
  (carrier, origin, dest, route, carrier×hour, carrier×origin, origin×hour, dest×hour, carrier×dow,
  route×dow, carrier×route) with 20-fold OOF values for training rows and full-train maps for inference
  (smoothing `k=500` towards the prior). Members run 800 trees at `lr=0.03`.

Shared regularization: `reg_lambda=3`, `colsample_bylevel≈0.7-0.8`. The ratio of native to TE members matters —
3 native + 5 TE scored 0.7469 while 6 native + 5 TE scored 0.7493, so the native view is not redundant with the
target-encoded view and both must be represented.

## Changes that mattered most

1. **`carrier_x_hour` categorical interaction** (0.7141 → 0.7257 single model). Carrier identity and time of
   day interact strongly; nearly every *other* interaction I tried (origin/dest × time, route, carrier ×
   origin/dest/month, carrier × day-of-week) hurt, so this one was the only interaction worth keeping.
2. **Time-of-day feature block** (0.7164 → 0.7183). `dep_min` alone is worth ~0.005 AUC — departures are
   scheduled on a 5-minute grid and the minute values carry real delay signal — and the sin/cos encodings add
   a further ~0.0014.
3. **Averaging structurally different XGBoost models** (0.7385 → 0.7400 with 3 members). Structural
   diversity (depth 4 through 10, lossguide, different seeds) is worth more than any single hyperparameter.
4. **A dedicated target-encoded feature view as ensemble members** (0.7440 → 0.7460, then 0.7498 as the view
   was enriched). This was the single biggest late gain. Notably, *appending* TE columns to the native
   categorical matrix always hurt (exp #33, 0.7490 vs 0.7496) — the value comes from having models that see
   only the encoded view, not from the columns themselves.
5. **Smoothed, high-fold out-of-fold encoding** (`k=500`, 20 folds; 0.7471 → 0.7496). Heavier smoothing and
   more folds both helped, consistent with a noisy year-to-year target rate.

## What did not help

1. **Most extra feature engineering.** Grid-alignment/night flags (#11, 0.7407), extra native interactions
   (`origin_blk`, `carrier_dow`, `route`, `dom_hour`), distance bins, and frequency encodings all lost
   0.002-0.004. Ablating `mon`/`dom` was also a wash (#15), so nothing was removed.
2. **More capacity or weaker regularization.** Depth 8 / lr 0.03 / 800 trees scored 0.7040 at the start, and
   `subsample=0.8` (#18, 0.7389), `gamma=0.5` (#17, no change) and looser `max_cat_threshold` (#10, 0.7417)
   all failed. The 2005→2006 shift rewards parsimony: the winning direction was shallower trees plus heavier
   regularization, not more.
3. **DART boosting.** Two attempts (#12, #14) both hit the 120 s experiment timeout even at 250 trees, so DART
   could not be evaluated within budget. `rank:pairwise` (probed off-budget) scored 0.57 without query groups.
   Adding *more* members also stops paying at ~12: 8 members of mixed quality (#13, 0.7431) and 13 members
   (#29, 0.7489) were both worse than their sparser predecessors.

## What I would try with more budget

The plateau from experiment #24 onward is real: 17 consecutive experiments moved the metric by less than
0.0005, which is inside the noise band for a 100k-row AUC (~0.002 SE), so most of the late decisions were
selection on noise. With more budget I would attack the year shift directly rather than keep tuning: (a)
weight training rows by recency — I never tested month-dependent sample weights, and the eval/holdout slices
are strictly later than 2005; (b) an explicit adversarial or importance-weighting scheme to find which 2005
patterns the 2006 slices disagree with, then drop those columns; (c) get DART under the time limit by
shrinking the member schedule (e.g. 150 trees at lr 0.12 on 2 members) since it is the one structurally
different booster I could not measure; (d) replace the fixed member schedule with per-member early stopping
on an internal train split, which I never had budget for; and (e) a proper nested selection protocol — pick
the ensemble by cross-validation inside `train.csv` only and touch `eval.csv` once — to stop paying for
noise-fitting at the top of the leaderboard.
