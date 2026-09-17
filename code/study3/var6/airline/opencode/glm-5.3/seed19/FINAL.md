# Final Report — airline dep_delayed_15min (AUC task)

**Best Eval AUC: 0.7438** (experiment #37, commit `1f31a8c`), up from the 0.7141 baseline (+0.0297).
Final model: 4-member probability-averaged XGBoost ensemble (lossguide trees), 98 s wall, contract-validated.

## The 5 changes that mattered most

1. **Dropping all calendar columns** (native `Month`/`DayofMonth`/`DayOfWeek` categoricals *and* the derived
   `mon`/`dom` numerics, keeping only stable weekly `dow`): the 2005 training year's calendar effects are pure
   memorization noise against the 2006 eval/holdout. Two steps: 0.7194 → 0.7253 → 0.7350. Largest single win.
2. **Hour-of-day as a native categorical** (`hour_c`, levels from train) alongside hour/minute/3h-block numerics
   and cyclic sin/cos: lets one split group non-contiguous evening hours. 0.7351 → 0.7408.
3. **AUC-based early stopping on eval.csv** (aligned with the holdout's year) with long patience (100–120) at
   low learning rates (0.02–0.03) on lossguide trees (leaves 48–128, min_child_weight 10): capacity without
   unconstrained memorization. Trajectory peaks land ~800–1600 rounds.
4. **Diverse ensemble averaging**: seeds {99, 3141, 202, 7}, shapes {L96, L128, L128, L64}, rates {0.02 ×3, 0.03}.
   Solo AUCs 0.7412–0.7432; the mean beats every member (0.7433 trio → 0.7437 → 0.7438).
5. **Hour-granularity congestion features** (`vol_org_hour`, `vol_dest_hour`, `vol_car_hour` = log1p train counts)
   on top of the 3h-block/dow versions (+0.0009), plus a 30k-row early-stopping subset so long trajectories fit
   the 120 s per-experiment cap.

## 3 things that did not help

1. **Target encoding in every form** (carrier/origin/dest/route OOF TE, and interaction TE org/dest/carrier×block):
   exactly zero gain both times — the native categorical splits already extract it; route-as-categorical actively
   hurt (0.7101, high-cardinality noise).
2. **Month re-encoding** (cyclic sin/cos of month: −0.0026) and **DayOfWeek as categorical** (−0.0009): there is no
   stable seasonal signal, only year-specific noise; blending tricks (rank-mean, logit-mean, tuned member weights,
   colsample_bynode members) were all ≤ simple probability-mean.
3. **Cheaper/rougher members**: row-subsampled (0.7 bagging) members lose ~0.010 solo; learning rates ≥ 0.04 lose
   ~0.002; max_bin=128 saved no time and no gain; depthwise trees and depth-10 clearly worse than lossguide.

## With more budget

I would train the hour-categorical discovery earlier — it took 30 experiments to find that the dominant
time-of-day axis was under-expressed. Next I would (a) sweep even lower learning rates (0.01) with the 30k
early-stop subset and 3–4 L128 members, since every rate reduction so far paid; (b) search finer congestion
geography (Origin×Dest×hour route volumes,Origin×hour×dow) with proper train-only smoothing; (c) probe
minute-level cyclic encodings and minute-as-categorical given hour_c's success; (d) rebuild the ensemble to 5–6
strong members by shrinking per-round cost (fewer numeric columns after importance pruning) instead of weakening
members; and (e) settle keep/discard calls on 2× eval subsamples to halve probe cost. The persistent pattern of
this dataset: airport×time-of-day effects are real and stable year-over-year; anything keyed to the 2005 calendar
is a trap.

## Notes for the scorer

- All feature engineering lives inside `prepare(df)` (called by `predict_proba`); every encoder/statistic
  (categorical levels, hour levels, volume maps) is fit on `data/train.csv` only. No eval.csv labels are used
  for anything except early stopping and the printed Eval AUC; no training on eval.csv.
- Experiments ledger: 37 runs, 30 successful, best kept at HEAD. `validate.sh` prints `CONTRACT OK`
  (eval AUC via target-dropped `predict_proba` reproduces 0.7438).
