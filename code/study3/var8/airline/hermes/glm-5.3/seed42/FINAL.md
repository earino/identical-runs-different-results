# FINAL — airline dep-delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7451** (baseline 0.7141, +0.0310 over 37 experiments)
Final config: 5-seed bagged ensemble of `xgboost.train` boosters, d16 / mcw3 / lr.05 / ss.9 / cs.3 /
reg_lambda 1.5, fixed 300 rounds each, rank-averaged. Validated: `CONTRACT OK`, predict_proba on raw
target-less DataFrame reproduces 0.7451.

## The 5 changes that mattered most

1. **Native-API fixed-round bagged ensemble** (exp23, +0.009): shared DMatrix + `xgb.train` with a
   FIXED 300-400 rounds instead of sklearn-API early-stopping-on-2006. Early stopping on the 2006 eval
   slice quit at ~50-150 rounds and was systematically underfitting; fixed rounds chosen by 2005-only CV
   (CV AUC kept rising to 550-650 rounds) transferred robustly. Single biggest jump of the run.
2. **colsample_bytree=0.3** (exp23, part of the same jump): after doubling the feature count, offline
   CV showed strong column subsampling dominates (CV 0.7716 -> 0.7898 region). Counter-intuitive vs the
   baseline's 0.7-1.0 habit.
3. **Fine-grained smoothed time-of-day delay-rate priors** (exp27/28, +0.002 each): train-only smoothed
   (empirical-Bayes) delay rates for 30-min, 15-min, and 10-min DepTime bins. The hourly delay pattern
   is the single most year-stable signal (2005->2006 rate corr 0.96); finer bins + heavy smoothing
   (k=100-200) let even d16 members exploit it (CV 0.7942 -> 0.8007 with d16).
4. **Volume/congestion features** (exp17, +0.002): train-derived flight counts per origin, dest, and
   origin x hour, log-scaled. Target-free, zero leakage risk, high year-stability (log-vol corr 0.90-0.98).
5. **Rank-averaging + deep members tuned on the CURRENT feature set** (exp16, exp35/36): averaging
   argsort-ranks instead of probabilities is calibration-free (safer under 2006 drift), and re-running
   member-param CV after each feature change (mcw 5->3, reg_lambda 1.5) kept the ensemble at its
   time-constrained optimum. Offline CV on train-only folds made this cheap: 3-fold CV on 2005 was run
   ~30 configs for zero experiment cost.

## 3 things that did not help

1. **Route features in ANY form** (exp3, exp5): route categorical or target-encoded, -0.010 AUC.
   Route-level delay rates barely transfer 2005->2006 (corr 0.57); 100k rows can't support ~5k routes.
   Same for dest priors and origin/dow/month/dow-rate priors later — every prior whose key had unstable
   marginal rates hurt, despite decent interaction stability (month x hour corr 0.96 but month marginal
   0.59 -> mh_rate hurt, exp26).
2. **Early stopping on the 2006 eval slice** (baseline paradigm, exp10 CV evidence): it optimizes
   "fit to 2006-slice1" which is NOT the objective and underfits badly (~145 vs ~600 CV-optimal rounds).
   Also killed several runs: members kept hitting the tree cap and blowing the 120s limit (exp13, 18,
   20, 22, 31, 37 - six timeouts, all from capacity/time mismatches).
3. **More members / heterogeneous depth mixes / member row-subsampling** beyond 5 deep members
   (exp15, 24, 29, 31, 33, 34, 37): 7->10 same-config members gave +0.0005; mixing d12+d16 after the
   feature upgrades was worse than pure d16; ss<0.9 hurt single-member CV. The 120s cap is the binding
   constraint: 5x d16n300 with mcw3 is the measured optimum of "members x rounds x depth" under it.

## What I would try with more budget

The score is feature-limited, not model-limited: CV (2005) sits at ~0.80 while eval (2006) is 0.745 —
the gap is year drift. With more budget I'd attack drift directly: (a) time-aware validation inside
train.py (last-3-months-of-2005 as internal val) to pick rounds without touching eval.csv at all,
since eval-based choices are the overfitting risk here; (b) per-feature drift screening: compute each
candidate feature's 2005-vs-2006 stability (as I did for priors) and keep only stable ones, applied
systematically to interactions like carrier x hour x distance; (c) seasonal alignment: month effects
differ by year, so a month-agnostic "day-of-year" cyclic encoding or matching by weather-season proxies
might recover the month signal that raw Month carries but cannot transfer; (d) spend leftover wall time
(which is plentiful - only CPU was scarce) on more seeds for the rank-average, since each added
decorrelated member is nearly free AUC once its config is fixed. Finally, a holdout-simulated
"2005-train / late-2005-test" harness would have let me test all of this offline at full fidelity
without spending a single scored experiment.
