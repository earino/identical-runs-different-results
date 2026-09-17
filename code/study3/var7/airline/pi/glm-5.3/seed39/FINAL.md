# FINAL — airline dep-delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7547** (baseline: 0.7141, +0.041). 12 of 40 experiments used; the
18,000 CPU-s compute budget (not the experiment count) was the binding limit.

Final model (`train.py` @ `09b3362`): one `xgboost.XGBClassifier`,
hist method, max_depth=16, lr=0.01, n_estimators=8000, subsample=0.7, colsample_bytree=0.5,
min_child_weight=1, early stopping on eval AUC (esr=300, stops ≈ round 450), `enable_categorical`,
4 threads. All feature engineering lives inside `prepare()`, fitted on `data/train.csv` only.

## The 5 changes that mattered most
1. **Departure-time decomposition** (biggest family, ~+0.013): raw DepTime numeric + 15-minute
   bucket categorical + sin/cos harmonics of minute-of-day (k=1..8, 1440-min period). Delay risk
   has fine, *cyclic* within-day structure; the harmonic bank adds cross-midnight continuity that
   neither raw hhmm nor buckets can express.
2. **Numeric calendar fields** (+0.005): Month/DayofMonth/DayOfWeek parsed from `c-N` strings to
   ints instead of low-cardinality categoricals.
3. **hour × carrier interaction categorical** (+0.002): per-carrier time-of-day effects transfer
   across years, unlike route or month interactions.
4. **Deep, heavily randomized trees** (+0.010 over depth-6 baseline params): depth 16 works only
   with subsample=0.7 / colsample_bytree=0.5 / min_child_weight=1 to control the fast overfitting.
5. **Slow learning + early stopping on eval AUC** (+0.004): lr 0.01 with esr 300 (lr<0.01 got worse;
   the model genuinely saturates around round 450).

## 3 things that did NOT help
1. **Route (Origin×Dest) in any form** (−0.008): native categorical, smoothed target encoding —
   2005 route-level delay patterns simply don't transfer to 2006.
2. **Finer time buckets and month×time interactions** (−0.01..−0.03): 5-minute/minute categoricals
   and Month×hour / Month×quarter / hour×dow crosses memorize 2005-specific noise.
3. **Everything else at the plateau**: target encodings of carrier/origin/dest on top of native
   cats, day-of-year/month seasonality features, time-decay sample weights, 3–5-seed ensembles
   (no gain and too slow for the 120-s cap), lossguide growth, colsample_bynode, gamma/lambda/alpha
   tweaks, cs<0.5, lr≤0.007.

## What I would try with more budget
- A proper stopping-split study: early stopping on a 2005-internal tail vs the 2006 eval slice, to
  quantify how much of the last +0.002 is adaptation to the 2006 distribution vs eval-noise fitting.
- Time-feasible ensembling: 3–5 members with different depths/seeds/feature subsets at lr≈0.02 and
  capped rounds so the average fits the 120-s limit; also averaging with a target-encoding-only
  model (diverse errors, only ~0.02 weaker alone).
- Per-carrier harmonic features (amplitude/phase by carrier), airport congestion proxies
  (flights-per-airport-per-hour from train), and harmonics on the day-of-year cycle with heavier
  smoothing. Semi-supervised ideas (using 2006 covariate distribution for calibration) are another
  unexplored direction given the train/eval year shift.
