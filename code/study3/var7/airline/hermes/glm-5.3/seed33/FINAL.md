# Final report — airline dep-delay XGBoost (autoresearch benchmark)

## Result

- **Best Eval AUC: 0.7414** (experiments.tsv #23, commit `6a8a01c`), vs baseline 0.7141 → **+0.0273**.
- Final validation: `./validate.sh` prints `CONTRACT OK`; `predict_proba` on eval (target dropped) reproduces 0.7414.
- Budget at stop: 13 experiments and ~140 wall minutes remained, but Python CPU compute was nearly
  exhausted (17,517 / 18,000 CPU-s; each further run costs ~350 CPU-s, and the interpreter refuses to
  start past 18,000). Stopping with headroom for a clean finalize was deliberate.

## The 5 changes that mattered most

1. **Depth-spread ensemble of XGBoost models** (members at max_depth 1..10, averaged): the single
   biggest structural win. Depth 4-ish trees transfer best across the 2005→2006 shift, and averaging
   over a depth range beats any single depth (+0.005 over tuned single model).
2. **Schedule-density count features**: log1p(count) of train-set flights per (carrier/origin/dest,
   time block) at hour, 30-min, 15-min and 5-min granularity, all computed on train.csv only inside
   `prepare()`. Going from hourly to 15-min granularity alone was worth +0.0027; the hierarchy of
   granularities is complementary (removing the coarse levels cost -0.0034).
3. **Route-level features**: (Origin_Dest) counts, route × time-block counts, and train-mean
   scheduled DepTime/Distance per route (+0.0014 from route×block counts alone).
4. **Feature bagging of ensemble members** (`colsample_bytree=0.2/0.3`, 20 members): decorrelating
   members via column subsampling was worth about +0.005 at fixed depth mix.
5. **Month-recency sample weights** (linear 0.5→1.0 over the 2005 year): upweighting later months,
   which are distributionally closer to 2006, added +0.0004 and never hurt.

## 3 things that did not help

1. **Target encoding** (smoothed carrier/origin/dest/route delay rates) — consistently *hurt* by
   ~0.02 AUC; the year shift breaks per-level rates. Same for route-as-a-single-categorical (4k
   levels) and explicit entity×hour crossed categoricals.
2. **Early stopping on an internal time split** (train on Jan–Nov, stop on Dec): December is
   unrepresentative; it stopped at 45 trees and underfit. Fixed n_estimators on the full 100k did better.
3. **Row subsampling, depth-weighted averaging, min_child_weight, quadratic recency, dow×block
   counts, day-of-year weights, 30 members at ne400** — all equal or worse than the kept config.

## With more budget I would try

The two levers that kept paying were (a) finer schedule resolution and (b) more decorrelated members.
I would push (a) toward exact-minute (carrier, route, minute) counts and per-(flight-identity) delay
statistics computed *within* 2005 via out-of-fold encoding to avoid leakage; and (b) toward a much
larger member zoo (50+ models over depth × colsample × random feature-subset hard-bagging), which the
120 s cap prevented. I would also probe calibration-and-rank-averaging (AUC is rank-based; my quick
rank-average test was promising but untested at the final feature set), and a two-stage model where
a first-stage model's out-of-fold predictions become a feature. Finally, cross-year validation of
every feature family (fit on Jan–Jun, validate on Jul–Dec 2005) would give a second, hidden-set-like
signal to keep/discard against, guarding against eval.csv overfitting of the small +0.0005 decisions.
