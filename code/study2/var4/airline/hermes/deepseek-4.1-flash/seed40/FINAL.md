# Final report — autoresearch XGBoost on the airline task

**Best Eval AUC: 0.7419** (experiment #39, commit `d69f1cf`; baseline was 0.7141, so +0.0278).
Metric is on `data/eval.csv` (2006-slice1, 100k rows) as printed by `python train.py` / re-checked by
`./validate.sh` (`CONTRACT OK`, `predict_proba` on the target-stripped frame returns the same 0.7419).

Budget used: 40/40 experiments, ~55 min of run time (171 min wall clock remained, so the experiment
count was the binding limit, not the clock or the 18,000 CPU-second allowance: 11,324 s used).

## Final configuration

`train.py` builds everything inside `prepare(df)` with statistics fixed from `data/train.csv` only:
departure time decomposed into `hour`, `minute`, `tmin` (minutes since midnight) plus the daily Fourier
basis `sin/cos(2*pi*k*tmin/1440)` for k=1..12, raw `DayOfWeek`, `Distance`/`log1p(Distance)`, and native
XGBoost categoricals for `UniqueCarrier`, `Origin`, `Dest` and `route = Origin_Dest`.
The model is an average of 3 `XGBClassifier`s (seeds 42/7/2024), depth 8, lr 0.03, 2000 rounds,
subsample 0.8, `colsample_bytree` 0.6, `min_child_weight` 5, `tree_method="hist"`,
`enable_categorical=True`. Runtime ~85 s, well inside the 120 s cap and the 6 GB memory cap.

## Changes that mattered most

1. **Daily Fourier harmonics of the scheduled departure time (k=1..12): +0.0169.** 0.7224 -> 0.7393 in
   three steps (exp31 k<=3, exp32 k<=6, exp33 k<=12). Axis-aligned splits on `hour`/`tmin` approximate a
   smooth, steep delay-probability curve poorly; the harmonic basis gives the trees that curve directly.
   Extending past k=12 (exp35) or adding a weekly cycle (exp34) did not add anything.
2. **Dropping the calendar features: +0.0046 total.** Removing `Month`+`DayofMonth` (exp20, 0.7194),
   then the month cyclicals (exp21, 0.7198), then the day-of-week cyclicals (exp22, 0.7207) was a
   consistent, simplifying win — 2005 month/season delay levels do not repeat in 2006.
3. **Keeping raw `DayOfWeek` after removing its cyclicals: +0.0013** (exp26, 0.7220). Weekday is a stable
   categorical-style effect, but the smooth sin/cos encoding of it is the wrong inductive bias.
4. **3-seed averaging: +0.0018** (exp13, 0.7157 -> 0.7175), and it held up on every later feature set.
5. **`colsample_bytree` 0.6: +0.0020** (exp39, 0.7399 -> 0.7419). With 24 correlated harmonic columns,
   sampling fewer features per tree is the regularization that fits.

## Things that did not help (all reverted)

1. **Out-of-fold target encoding** of carrier/origin/dest/route/carrier-hour (exp4, 0.7148 vs 0.7152) and
   the related count/volume features (exp15, +0.0001, later confirmed worthless and dropped in exp25).
   2005 delay rates simply do not carry over to 2006.
2. **All year/season/week-scale features**: month cyclicals, day-of-year, December-holiday flag (exp9,
   0.7126) and hour-of-week harmonics (exp34, 0.7358) each cost AUC.
3. **More capacity or more members instead of better features**: lr 0.05 with 3.5k rounds (0.7130),
   depth 10 (0.7390), `num_parallel_tree=4` bagging (0.7170), 4 members at n=1500-1800 (0.7166/0.7212/
   0.7387). Down-weighting old months (recency weights, exp17, 0.7155) and looser categorical split
   parameters (exp18, 0.7167) also hurt. Early stopping against an internal 10% split was unreliable:
   it peaked at 140-2500 rounds depending on regularization while eval kept improving, so a fixed round
   count was used.

## What I would try with more budget

The harmonic result says the remaining signal is concentrated in the *shape* of the departure-time
response and its interactions, so I would push on that axis rather than on raw columns: per-carrier and
per-airport time-of-day curves (carrier x harmonic products, or a low-rank factorisation of the
carrier/airport x time-of-day delay surface), a spline or kernel-smoothed `tmin` representation compared
side-by-side with the Fourier basis, and joint tuning of the round count, learning rate and
`colsample_bytree` now that the feature set is stable (the colsample win was the last one found and was
not re-tuned against n). I would also revisit cross-validated blending — averaging the 3 seeds in
*probability* space was better than rank or parameter diversity here, but out-of-fold stacking weights
over a few structurally different members (depth 6/8, harmonic subsets) is the natural next step, and I
would re-check whether the harmonic basis still helps once 2006-slice2-sized data (1M rows) is available
for training, since that is where the final score is computed.
