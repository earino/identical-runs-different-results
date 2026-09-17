# Final report — airline delay AUC (2005 -> 2006 transfer)

**Best Eval AUC: 0.7304** (experiment #37, commit `0080507`; baseline was 0.7141, so +0.0163).
Budget used: 40/40 experiments, ~44 minutes of the 230 available, 2,641 of 18,000 CPU-seconds.
`./validate.sh` prints `CONTRACT OK` and reproduces 0.7304 through `predict_proba(df)` with the
target column removed, so every engineered feature is reproduced on unseen rows.

The kept configuration: an averaged ensemble of 30 XGBoost classifiers
(`depths 4-8 x 6 seeds`, 200 trees, lr 0.05, subsample 0.8, **colsample_bytree 0.4**, max_bin 512,
hist, native categoricals) over ~21 engineered features, all built inside `prepare(df)` from
statistics fitted on `data/train.csv` only.

## Changes that mattered most

1. **Representation of the raw columns (+0.0027).** `Month`/`DayofMonth`/`DayOfWeek` are `c-<n>`
   strings: they were ordinalised so trees can split them by threshold; `DepTime` (hhmm) was turned
   into `dep_hour` + `dep_min` minutes-since-midnight, with the `24xx-26xx` after-midnight values
   wrapped; `Distance` got a `log1p` twin. Keep one or the other of each pair is nearly free.
2. **Train-only smoothed target encodings of categorical keys (+0.0022, and they dominate the model).**
   Smoothed means of the label for `Origin`, `Dest`, `UniqueCarrier`, `route`, `origin x hour` and
   `dest x hour`, with `alpha=60`. `te_orig_hour` and `te_dest_hour` are by far the top features
   (0.22 and 0.15 gain-normalised importance) — airport-by-hour delay propensity is the signal.
   Training rows use 5-fold out-of-fold values so the encoders never see their own label, while
   `predict_proba` uses the full-train maps.
3. **Airport schedule-share congestion features (+0.0008).** For each airport, the share of its
   departures that fall in the flight's hour and in a ±1h window, from a train-only hourly profile:
   scale-free, and it transfers across years because it describes schedule structure, not delays.
4. **Ensembling (+0.0014 on its own).** 1 -> 5 identical-seed models (+0.0010), then members mixing
   depths 5-7, then 21, then 30 members spanning depths 4-8 (0.7262 -> 0.7276 with colsample at 0.8).
5. **Aggressive feature subsampling (+0.0028, the single biggest late win).** `colsample_bytree`
   0.8 -> 0.6 -> 0.4 gave 0.7276 -> 0.7292 -> 0.7304. With only 100k training rows from one year and
   a year-separated eval, decorrelating the trees is the cheapest regularisation available;
   `0.2` was too far (0.7294) and so was `subsample=0.6` (0.7300).

## Things that did not help (reverted)

- **More capacity / longer boosting.** 400 trees at depth 7 (0.7074), 800 trees at lr 0.03 (0.7174),
  400 trees lr 0.05 (0.7253), 200 -> 400 with the final feature set. The 2005 -> 2006 shift punishes
  memorisation hard; 200 trees at lr 0.05 with depth 6 is the sweet spot.
- **Explicit calendar interactions** `day_of_year` and `is_weekend` (0.7230 vs 0.7241 at the time),
  and **categorical twins** of Month/DayOfWeek/hour (0.7243) — native one-vs-rest splits on top of
  the ordinal columns added noise rather than signal.
- **Extra keys and coarser buckets**: carrier x airport interactions (0.7261), 2-hour/daypart
  airport buckets (0.7259), nationwide minute-level load features (0.7259), (airport-hour) minus
  (airport) deviation features (0.7257), hierarchical TE that shrinks an airport-hour cell toward
  its airport instead of the global prior (equal at 0.7262/0.7260, so discarded on the simplicity
  tie-break), 20-fold instead of 5-fold out-of-fold encodings (0.7257), `min_child_weight` 5 -> 25
  (0.7255) and depth 5 everywhere (0.7251). All of these landed inside the ±0.0005 seed-noise band.
- **Dropping features was not free either**: removing the count features entirely cost 0.0003, so
  `cnt_Origin`/`cnt_Dest`/`cnt_route` were kept while the redundant ones were pruned.

## What I would try with more budget

The feature set is saturated: everything after experiment #21 moved the score by at most ±0.0005
except the colsample sweep, and my keep/discard calls at that resolution are close to coin flips.
The one direction that still looked live when the budget ran out was decorrelation — 0.4 was the
best colsample tested on a 5-point grid but the response surface between 0.3 and 0.5 is unexplored,
and per-member colsample/lr diversity inside the ensemble is untested. Beyond that I would spend the
budget on a proper stacking layer: out-of-fold predictions from the 30 members fed to a small
second-level XGBoost together with `te_orig_hour`/`dep_min`, which is still XGBoost-only and would
let the model weight each member by competence instead of averaging uniformly. I would also check
whether a balanced-subsample-aware calibration matters for the hidden holdout, and re-tune
`TE_ALPHA` jointly with the choice of keys now that colsample has changed the effective feature
count per tree — all of it judged against a multi-seed evaluation, because single-run differences
below 0.001 are not measurable at this scale.
