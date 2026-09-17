# FINAL — autoresearch XGBoost on `airline` (dep_delayed_15min)

## Result

| | Eval AUC |
|---|---|
| baseline (`train.py` as shipped, 30 trees / depth 6 / lr 0.1) | **0.7141** |
| final (`HEAD`, experiment 40) | **0.7531** |

40/40 experiments used, ~190 minutes of wall clock, 6.8k of 18k CPU-seconds. `./validate.sh` prints
`CONTRACT OK`, and re-scores 0.7531 through `predict_proba(df)` with the target column removed, so all
feature engineering is reproducible on unseen rows.

Final model: 3-member XGBoost ensemble (`lossguide`, `max_leaves=256`, `lr=0.015`, `n_estimators=1000`,
`min_child_weight=50`, `subsample=0.8`, `colsample_bytree=0.35`, `reg_lambda=20`, `max_bin=512`);
members 0/1 are seed variants and share the configuration. Runtime ≈ 80 s, memory well under the 6 GB cap.

## Changes that mattered most

1. **Decode the coded columns and the clock (+0.0034).** `Month` / `DayofMonth` / `DayOfWeek` ship as ordinal
   strings (`c-7`) and were being treated as unordered categoricals; `DepTime` is an hhmm integer. Decoding to
   integers and splitting it into `dep_hour` / `dep_minute` / `dep_minofday` was the first real gain
   (0.7141 → 0.7175), and it is where nearly all the signal is: minutes-of-day alone scores 0.68 AUC.
2. **Regularize hard and grow `lossguide` (+0.0154 over the decoded-feature model).** The 30-tree baseline was
   not underfit but *over*-regularized in the wrong way: 400 trees at lr 0.05 scored 0.7154, i.e. worse. What
   worked was few deep lossguide trees per round with heavy shrinkage: `max_leaves=256`, `min_child_weight=50`,
   `reg_lambda=20`, `colsample_bytree=0.35`, lr 0.02, hundreds of rounds → 0.7329. Column subsampling turned out
   to be the single most sensitive knob (0.65 → 0.7447, 0.35 → 0.7481 at the final feature set).
3. **Timetable-density ("congestion") features — the biggest single win (+0.015).** `log1p` counts, computed on
   training rows only, of how many flights share the same route / origin / destination / carrier, and above all
   how many share the *same 15-minute departure slot* at the same origin or destination, plus smoothed 45/75-minute
   window sums. These are target-free, and that is exactly why they work where target encodings fail: airport
   schedules are stable across the 2005 → 2006 shift, so the feature transfers even though the year does not.
4. **Neighbouring and arrival slots (+0.005).** The immediately preceding and following 15-minute slots at the
   origin (asymmetric queues), the slot-share features (`min15 − hour` counts), and congestion at the destination
   around the *estimated arrival* slot (departure + `Distance`/7.5 min) — the latter was the last accepted change
   (0.7500 → 0.7531).
5. **Seed-averaged ensemble at a lower learning rate (+0.0015).** Three members at `n_estimators=1000`,
   `lr=0.015`. Small on eval, but averaging is variance reduction rather than eval fitting, so it should hold up
   on the hidden holdout.

## What did not help

1. **High-cardinality identity features.** Adding `Origin_Dest` as a native categorical cost 0.010; out-of-fold
   smoothed target encodings of carrier/origin/dest cost 0.002. Per-category delayed-rates do transfer somewhat
   (Spearman 0.53–0.68 on train → eval), but the tree over-commits to 2005-specific identities.
2. **Calendar × time-of-day interactions.** Explicit `DayOfWeek×hour` / `Month×hour` / `DayofMonth×hour`
   categorical levels dropped 0.7329 → 0.7141, and busy-day density (Month-DayofMonth flight counts, a holiday
   proxy) dropped it to 0.7358. Time-of-year effects do not line up between the two years.
3. **Finer or coarser congestion granularity beyond 15 minutes.** 5-minute buckets, exact-minute buckets and
   carrier-level slot counts were all neutral to negative; so was replacing the "unseen key → 0" convention with a
   hierarchical fallback to the hour-level count (−0.011), and so were one-hot carrier, `colsample_bylevel`,
   `n_estimators=1500`/`lr=0.01`, and a deliberately diverse-config ensemble (0.7511 vs 0.7515).

## Caveats

Keep/discard decisions were made on `data/eval.csv`, so anything below ~0.001 AUC is noise (100k rows) and I
treated it as such: every accepted change is either a mechanism-driven feature or variance reduction, and the
last few accepted deltas were 0.0009–0.0015. Roughly 3 of the 40 experiments were wasted on crashes and 3 on
120 s timeouts from over-long probe sweeps.

## With more budget I would try

A proper internal time split (early-2005 → late-2005) for hyperparameter selection instead of spending eval
observations, so that eval stays a clean check; richer queueing-style schedule features (each airport's slot
occupancy relative to its own daily peak, departure-rate slopes through the morning bank structure) built the same
target-free way; monotone constraints on minutes-of-day and congestion to cut variance; and genuine propagation
features, which this schema blocks — no date, tail number or actual departure time is available, so aircraft
rotations and inbound-late chains cannot be reconstructed. Given that time-of-day plus 15-minute slot congestion
accounts for most of the 0.753, the remaining headroom is probably in exactly those propagation signals rather
than in further hyperparameter search.
