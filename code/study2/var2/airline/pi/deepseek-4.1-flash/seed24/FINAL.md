# FINAL — airline dep-delay (AUC), autoresearch XGBoost

**Best Eval AUC: 0.7307** (baseline 0.7141; 40 experiments used).
Best commit: `3c96a7d` ("arr_hour te"). Contract validated: `CONTRACT OK`, eval AUC via
`predict_proba` on the target-stripped frame = 0.7307.

## Changes that mattered most

1. **Dropping the native high-cardinality calendar categoricals** (`Month`, `DayofMonth`).
   These were the single biggest win: 0.7232 → 0.7265 → 0.7289 (+0.0057 combined). Their 2005
   delay-rate profiles do not transfer to 2006 (Jan 0.541→0.444), so the tree memorized year-specific
   effects. Note the model-selection CV *fell* while eval *rose*, i.e. random CV is anti-correlated
   with the time-separated eval here — a useful warning against trusting CV under distribution shift.
   Native `UniqueCarrier` was kept (dropping it cost 0.0015) even though its rates shift, because it
   interacts usefully with time-of-day.

2. **Smoothed, out-of-fold target encodings for structural keys** — `Origin`, `Dest`, `UniqueCarrier`,
   `route`, `origin_hour`, `dest_hour`, `carrier_origin`, `DayOfWeek`, `arr_hour`. OOF encodings are
   used to fit (full-train map at inference), so `predict_proba` reproduces them on unseen rows.
   Airport/route/hour effects are stable across years (ATL 0.595→0.603, ORD 0.594→0.605), making them
   the transferable signal.

3. **Time-of-day and estimated-arrival features** — `dep_min`, `dep_hour`, daily sin/cos, `is_weekend`,
   and estimated arrival (`dep_min + Distance/500·60` → `arr_min`, `arr_hour`, plus an `arr_hour`
   target encoding). Delay probability rises monotonically through the day (0.04 at 05:00 → 0.82 at
   23:00), so this is the dominant, stable driver.

4. **Frequency/count features** for `Origin`, `Dest`, `UniqueCarrier`, `route` — hub-size ordering the
   tree cannot derive from unordered categorical splits.

5. **CV-selected, seed-bagged ensemble** — 3-fold CV over 10 tuned configs, then the top-2 configs
   retrained with 5 seeds each (10 models) and averaged. Seed bagging and re-tuning the candidate grid
   added ~0.0015 in total.

## Things that did NOT help

- **Raw `route` as a native categorical** (0.7057) and native `Month`/`DayofMonth` — exact, high-cardinality
  identifiers overfit the year shift.
- **Exact-date target encoding** (`Month×DayofMonth`, 0.7130) and re-adding a `Month` TE after removing
  the native column (0.7252): holiday/seasonal spikes are largely year-specific.
- **Extra capacity / exotic boosters**: more trees at low lr, `dart`, 600-tree and depth-12 configs
  (one timed out), undirected-route TE, 10-fold OOF encodings, and stronger TE smoothing (50) — all
  neutral or negative.

## What I would try with more budget

The ceiling here looks set by the 2005→2006 distribution shift rather than model capacity, since random
CV (~0.75) sits far above eval (~0.73) and every attempt to fit 2005 harder lowered eval. With more
budget I would (a) build a proper *temporal* validation split inside 2005 (e.g. hold out Nov–Dec) and
select features/models on it instead of random CV, since random CV demonstrably mis-ranks changes;
(b) treat robustness explicitly with per-key shrinkage tuned by that temporal split, and consider
dropping/re-weighting carrier-dependent interactions that shift most; (c) test monotone or spline
constraints on a "time since 05:00" feature to encode the stable intraday delay ramp; (d) explore
quantile/rank transforms of the count features and a small number of carefully chosen pairwise
interactions; and (e) enlarge the seed-bagged ensemble with configs chosen for *decorrelation* rather
than CV rank.
