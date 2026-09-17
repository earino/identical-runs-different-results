# Final report — airline dep_delayed_15min (XGBoost autoresearch)

Best Eval AUC: **0.7543** (baseline 0.7141, +0.0402). Validated: `CONTRACT OK`,
`predict_proba` reproduces 0.7543 on a raw DataFrame without the target column.

Final model: 3-seed ensemble of `xgboost.XGBClassifier` (lossguide, max_leaves=1536,
lr=0.05, ~140-160 trees per seed via early stopping on eval, min_child_weight=3,
colsample_bytree=0.5, colsample_bynode=0.5, alpha=4, reg_lambda=2), probabilities averaged.

## The 5 changes that mattered most

1. **Dates as numbers only** (exp15, +0.010): the raw `c-<n>` Month/DayofMonth/DayOfWeek
   categoricals let the model memorize 2005's calendar; numeric values generalize to 2006.
2. **No month signal at all** (exp17, +0.004): dropping month_num/m_sin/m_cos on top — any
   month pattern learned from a single year is calendar-specific noise. dom_num/dow_num stay.
3. **Lossguide growth, max_leaves=1536** (exp18, +0.007 over depth-9 depthwise): wide shallow-ish
   trees fit the hour-of-day smoothness better than depthwise.
4. **Seed ensemble** (exp16, +0.001-0.004 at each stage): averaging 3 seeds' probabilities;
   robust, and 3 seeds cost less wall time than 4 (the 120s cap).
5. **DepTime decomposition + interactions** (exp3/19): hour, minute, hm ordinal, cyclical
   sin/cos, hour_cat, plus carrier_hour and origin_hour interaction categoricals.

## Things that did NOT help (all tested, all reverted)

1. **Route / Origin-Dest pair categorical**: -0.005 (exp4/6) — 2005 route idiosyncrasies.
2. **Target encoding** (smoothed, for Origin/Dest/Carrier/hour, and hour-interaction TEs):
   -0.001 to -0.004 every time — native categorical splits + L1 already regularize better here.
3. **Season/daypart/winter features, doy, dom_hour, dow_hour cats**: all negative; anything
   calendar-flavored overfits the single training year.

## With more budget

I would (a) hunt for a delay-propagation feature: within (Origin, date, hour) the count/order
of earlier departures is a congestion signal — needs careful train-only statistics and a
time-aware backoff; (b) tune the ensemble weights / add a 4th seed within the CPU cap by
trimming trees per seed; (c) probe max_bin and hist-precision settings for the high-cardinality
interaction cats; (d) test "eval-informed" early stopping replaced by a 2005 holdout month to
avoid any eval leakage in tree count (currently ES uses eval.csv, which is allowed but could
slightly overfit the eval year); (e) two-stage calibration of the averaged probabilities.
