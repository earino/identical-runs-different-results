# FINAL — airline dep_delayed_15min (XGBoost autoresearch)

**Best Eval AUC: 0.7539** (baseline 0.7141, +0.0398), experiment #40 (commit bfb73d0), validated
(`CONTRACT OK`, predict_proba reproduces 0.7539 with the target column removed).

## Final configuration

15-member bagged XGBoost ensemble; each member: 45 trees, depth 24, lr 0.1, colsample_bytree 0.6,
full rows, random column subset (75% of features) per member; members averaged in log-odds space.
Features: numeric Month/DayofMonth/DayOfWeek (parsed from c-strings), raw DepTime/Distance plus
DepHour, MinuteOfDay, sin/cos hour, and interactions hour*distance, dow*hour.

## The 5 changes that mattered most

1. **Bagged ensemble of column-subset XGB models** (exp14, 0.7178 -> 0.7266): 15 members on random
   75% column subsets, averaged. Variance reduction against the 2005->2006 shift; single largest jump.
2. **Deep trees inside the ensemble** (exp16-19: depth 8 -> 16 -> 20): 0.7266 -> 0.7499. Deep trees
   were catastrophic for a single model (300 trees at depth 6 dropped AUC to 0.708), but bagging
   absorbs their variance, so depth 24 members ended up +0.02 over depth 8 members.
3. **Full-row members + colsample 0.6** (exp22-23): 0.7503 -> 0.7526. Each member on all rows with
   per-tree column subsampling beats row subsampling here; 0.6 is the sweet spot (0.5 and 0.8 worse).
4. **Numeric calendar features** (exp10-11): parsing c-4/c-10/c-7 strings into ints and dropping the
   raw string versions: 0.7145 -> 0.7177 (pre-ensemble baseline). One range split replaces ~30
   per-level splits.
5. **Time-of-day features + compact interactions** (exp7, exp30): DepHour/MinuteOfDay/sin+cos gave
   +0.0004 on the baseline; hour*distance and dow*hour interactions added +0.0009 on the ensemble.

## What did NOT help

1. **More capacity on a single model**: 300 trees dropped AUC 0.7141 -> 0.7083; early stopping on eval
   capped it at ~75 iters and still scored lower. The time shift punishes single-model capacity hard.
2. **Route (Origin_Dest pair) categorical**: -0.009 alone (exp8) and -0.010 inside the ensemble (exp25).
   2005 route-level delay rates don't transfer to 2006.
3. **Smoothed target-rate features** (carrier/origin/dest/hour delay rates, prior 20): 0.7126 vs
   0.7145. Same leakage-into-shift problem, even smoothed.
4. Also flat or worse: min_child_weight=10 (-0.009), hyperparameter jitter (-0.001), rank averaging
   (-0.0006), more members (25/30), more col-subset (0.85), lr 0.15, season/quarter, extra
   interactions (month*hour, hour_block, log_dist).

## What I would try with more budget

The single biggest open lever is **temporal weighting / drift adaptation**: train is 2005, eval and
holdout are 2006, and everything that memorized 2005 (routes, target rates) hurt. I would fit the
ensemble with sample weights increasing through 2005 (recent months weighted up), or add a "months
since start" numeric feature so trees can build time-conditional splits. Second, a proper internal
time-blocked CV (last 2 months of 2005 as validation) to tune member hyperparameters without
touching eval.csv — every keep/discard decision here was made on eval, so tiny gains (+0.0001-0.0004)
may be noise. Third, carrier-level features that are drift-robust: number of flights per
carrier/origin (volume), average distance per route bucket, rather than delay rates. Finally,
giving each ensemble member a different *feature* transform (some with target rates, some without)
to increase functional diversity, and averaging 30+ members if runtime allows.
