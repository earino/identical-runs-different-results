# FINAL — airline dep-delay XGBoost autoresearch

**Best Eval AUC: 0.7345** (baseline 0.7141, +0.0204). Best commit: `80d55cd` (experiment #27).
Contract validated: `CONTRACT OK` (predict_proba reproduces the score on raw frames).

## The 5 changes that mattered most

1. **Cyclic time-of-day features with a full harmonic ladder (biggest single win, ~+0.006 total).**
   angle = 2*pi*(hour*60+minute)/1440; sin/cos for harmonics k=1..8. Each pair up to k=8 helped
   (k=9,10 hurt). This gives the trees a smooth, drift-stable encoding of the strong
   "delay rate rises through the day" effect, far better than raw DepTime alone.
2. **Diverse XGBoost ensemble + many trees (~+0.013 over single best).** 12 members
   diversified by max_depth (5-8) x colsample_bytree (0.5-0.8), n_estimators=1100, lr=0.05,
   max_bin=512, averaged probabilities. Averaging cancels the year-specific memorization that
   makes any single deep model worse on 2006 data.
3. **Right capacity regime for drift (~+0.002 early, enabled everything else).** train=2005,
   eval=2006: within-2005 validation AUC *rises* with capacity while 2006 AUC *falls*.
   Shallow members in an ensemble tolerate far more capacity than a single model.
4. **hour + minute + logdist + night features (~+0.001).** Decompose DepTime into hour,
   minute; log(Distance); flag departures <=4:00 or >=24:00.
5. **Fit every categorical level list on train only.** eval/holdout contain unseen carriers
   (AQ, YV) and routes; restricting categories to train levels prevents both crashes and
   silent leakage.

## 3 things that did NOT help

1. **Target encoding in every form** (carrier/origin/dest/route/hour, smoothed or raw,
   plain or in interactions): best case neutral, worst catastrophic (route x hour TE: 0.679).
   Carrier delay rates drift heavily year-to-year, so memorized rates generalize badly.
2. **Route (Origin_Dest) as one categorical** at any capacity (0.707-0.712): ~3000 levels,
   and 385 routes in eval are unseen in train; the split budget drowns the stable
   Origin/Dest effects.
3. **Early stopping on a within-2005 split, rank:pairwise objective, monotone constraints,
   subsample/row bagging, cyclic month/dow harmonics, dropping the carrier feature,
   one-hot-ing, lossguide.** All neutral or harmful on 2006 eval; within-2005 early stopping
   actively selected worse configs because 2005 validation does not see the drift.

## What I would try with more budget

The ensemble is CPU-bound, not idea-bound: members are CPU-heavy (112s of the 120s cap at
nest1100 x12) while the screen showed deeper members (d7/8) still adding AUC. I would
(a) switch members to fewer, deeper trees at lower learning rate with `max_leaves` control,
(b) test a two-level stack: one model family on the harmonic features only, another on the
categoricals only, then blend their ranks — the two families make different errors,
(c) fit a final calibration/rank blend weight on 2005 folds while monitoring the 2006 gap,
(d) probe robustness of the harmonic ladder against a 2007-style shift by holding out the
last two months of the training year, and (e) try polars/numpy-native data prep to cut the
~10s of pandas overhead and buy one more member per run.
