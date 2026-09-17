# FINAL — airline delay prediction (XGBoost), 40/40 experiments

**Best Eval AUC: 0.7613** (commit `ffd20b3`), vs **0.7141** for the baseline → **+0.0472**.
Baseline `train.py` prints 0.7613 too, and `./validate.sh` prints `CONTRACT OK` with the same
0.7613 reproduced through `predict_proba(df)` on eval with the target column removed.

Budget used: 40/40 experiments, ~2 838 of 18 000 CPU-seconds, ~26 minutes of the 230 available
(every experiment after the first feature pass ran in 4–50 s, far under the 120 s cap).

## The 5 changes that mattered most

1. **Schedule-congestion features (the big one, +0.0227).** For each flight, count the other flights
   that the *same frame* schedules at the same airport in the same hour of the same day — for both
   origin and destination — plus per-airport daily totals and their shares. This captures how loaded
   the departure/arrival slot actually is, which is the dominant delay driver available in this
   schema. 0.7212 → 0.7439 (exp 7/8/9).
   Because the scoring frame's sampling density is not the training density (100 k rows here vs
   1 m rows for the hidden holdout), counts are rescaled to the training rows-per-calendar-day
   (`_scale = TRAIN_DENSITY / frame_density`) so the feature estimates the *true* flights per
   airport-hour rather than "how often this row was duplicated".
2. **Cumulative daily airport load + carrier-hour congestion (+0.0013).** Number of flights already
   scheduled earlier that day at the origin/destination airport (delays accumulate through the day),
   and the carrier's own departures from that airport in that hour (bank structure). 0.7439 → 0.7452
   (exp 13).
3. **Capacity, re-tuned once the features were strong (+0.0121).** With raw features extra capacity
   *hurt* (2005 → 2006 shift), but with congestion features the model needed depth: depth 6→8→10→12
   and `min_child_weight` 100→30→10→3 gave monotone gains, 0.7439 → 0.7560 (exp 15–20). Plateau at
   `mcw=3`; `mcw=1` was flat.
4. **A small diverse ensemble (+0.0041).** Four XGBoost models (depths 10/12/14/12, different seeds)
   with `subsample=0.5`, `colsample_bytree=0.4`, `reg_lambda=5`, 300 trees at lr 0.03, probabilities
   averaged: 0.7560 → 0.7601 (exp 23, 24, 28, 29, 30).
5. **Dropping the destination *identity* and encoding it numerically instead (+0.0012).** Removing the
   high-cardinality `Dest` categorical (the model kept memorising 2005-specific airport identities)
   gained +0.0009; adding smoothed 5-fold out-of-fold target encoding for `dest`/`route` on top gained
   another +0.0003 → 0.7613 (exp 36, 38). `Origin` and `UniqueCarrier` categoricals were kept.

Supporting: parsing `c-<n>` columns to numerics, `DepTime` → hour/minute/cyclical/red-eye,
`Distance`/`log_distance`, carrier/origin/dest/route frequencies, and — early on — a small heavily
regularised model (depth 6, `mcw=100`, `sub=0.7`, `col=0.6`) that lifted the raw-feature ceiling from
0.7141 to 0.7168 (exp 4).

## 3 things that did not help

1. **More capacity or early stopping on a 2005 holdout (before the congestion features).** An internal
   random 2005 holdout improved monotonically with trees (0.7428 → 0.7679 at 30 → 1200 trees) while
   2006 eval AUC fell (0.7141 → 0.7054). The year-to-year shift is large; internal 2005 validation
   confidently picks the wrong model. Final model therefore uses no early stopping.
2. **Target encoding at the weak-feature stage** (origin/dest/carrier/route): 0.7210 vs 0.7212, dropped.
   Likewise neutral/negative: route-as-categorical, 3-hour and 3-day smoothed congestion windows,
   position-in-day ranks, route frequency counts, day-of-year/seasonality terms, relative day-load
   ratios, rank-normalised congestion counts, row bagging per member, `grow_policy=lossguide`,
   widening the ensemble to 6 members, and `n_estimators=700` (0.7533).
3. **Micro-tuning after the plateau.** `mcw=1` (0.7559), 6-member ensembles (0.7599/0.7585), and
   lossguide (0.7598) were all within noise of the kept configurations but more complex, so they were
   reverted.

## Robustness note (hidden holdout)

`predict_proba` was checked against the documented density difference: the same 30 000 eval rows score
AUC 0.7423 alone and 0.7421 when embedded in a 10× denser frame, and individual probabilities move by
at most 0.04 — the congestion rescaling makes the ranking essentially density-invariant. All fitted
tables (category levels, frequencies, target-encoding maps, `TRAIN_DENSITY`) are estimated on
`data/train.csv` only and applied inside `prepare()`, which `predict_proba` calls on the raw frame.

## What I would try with more budget

The remaining headroom is mostly in making the congestion signal cleaner and in selection rather than
in more capacity. First, quantile/rank-normalise every congestion count *within the scoring frame*, so
the feature distribution matches between training and a 10× denser holdout by construction rather than
by rescaling (a first attempt was neutral on eval at equal cost, so it needs a better validation
protocol to judge). Second, replace the single-level ensemble by stacking: train 3–4 base XGBoost
models with out-of-fold predictions on train and a shallow XGBoost meta-learner on top. Third, build a
validation split that actually mimics the shift — train on 2005 months 1–9, validate on months 10–12 —
so hyperparameters are chosen by transfer rather than by internal fit; nearly every wrong turn in this
run came from trusting a random 2005 holdout. Fourth, chase rotation/chain features (same
carrier+origin aircraft sequences within a day) and holiday-calendar flags, which the current feature
set cannot express, and try sample weights that emphasise high-congestion rows.
