# Final report — airline delay XGBoost (autoresearch)

**Best Eval AUC: 0.7424** (commit `0c73301`, experiments #40/40 used; validate.sh: CONTRACT OK).

Final recipe: features = DayOfWeek, UniqueCarrier, Origin, Dest (native categoricals fit on 2005 levels)
+ raw DepTime + Distance; single XGBClassifier, tree_method=hist, max_depth=32, learning_rate=0.03,
subsample=0.8, colsample_bytree=0.8, max_bin=1024, early stopping (patience 100) on a random 10% of train
(stops at ~119 rounds); all engineering inside `prepare()`.

## Changes that mattered most
1. **Removing drift-carrying features**: dropping DayofMonth (0.7154→0.7171) and Month (→0.7208).
   Calendar-anchored delay rates of 2005 do not transfer to 2006; the biggest conceptual win.
2. **Deep trees**: depth 6→24 (→0.7383 at the time) then 32 — deep trees capture the stable
   interactions (DepTime × Origin/Dest/Carrier) that shallow trees cannot reach within the ES budget.
   min_child_weight must stay 1: mcw=10 cost −0.0085.
3. **Early stopping + capacity**: fixed 30 trees → ES-selected rounds (0.7141→0.7154) — the entry ticket.
4. **max_bin 1024** (256→512→1024 gave ~+0.002 total): finer histogram resolution for DepTime.
5. **lr 0.03 with patience 100** (lr 0.05→0.03: +0.002); lr 0.02/0.04 and patience 200 all worse.

## What did not help (reverted)
- Any 2005-specific statistics as features: smoothed target encodings, count encodings, Route
  (Origin_Dest) categorical, origin-level seasonality sin/cos — hurt by 0.003–0.017.
- Derived time features (hour/minute/tod/sin-cos): neutral — raw DepTime already carries the signal.
- All ensembling: 10-seed bagging, 5-fold CV bagging, 3-member hyper-diverse ensemble — 0 gain or worse.
- Strong regularization (depth 4/mcw 50/λ10), lossguide growth, colsample_bynode, subsample/colsample 1.0,
  full-data retrain at fixed rounds, time-based validation split, recency weighting — all worse.

## Caveat
Experiments 37–40 (es patience 200, subsample 0.9, lr 0.04, diverse ensemble) were accidentally run on
top of the "drop Distance" commit, so their results are confounded by the missing Distance feature
(Distance turned out to be an important stable route proxy). They were re-reverted; HEAD = best commit.

## With more budget
The three confounded probes (patience 200, subsample 0.9, lr 0.04) deserve clean re-runs on the correct
base. Beyond that I would attack the time-shift directly: build a within-2005 rolling-origin CV (train on
months 1..k, validate on k+1..k+2) so that model selection measures forward transfer instead of same-year
random validation; engineer shift-stable statistics (shrunk Origin×hour congestion profiles with heavy
smoothing, route-level distance/physical features instead of 2005 target rates); and do a finer sweep of
the depth/bin/rounding frontier (depth 28–40, max_bin 1536, per-feature binning). If eval noise allows,
a small ensemble of the best config across stopping-split seeds would hedge the single-split ES choice
on the hidden holdout.
