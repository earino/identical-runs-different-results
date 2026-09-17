# autoresearch XGBoost — airline (scenario 2) — final report

**Best Eval AUC: 0.7514** (experiment #38, commit `2b99126`), up from the 0.7141 baseline.
The final `train.py` passes `validate.sh` (`CONTRACT OK`); `predict_proba` reproduces 0.7514
on eval.csv with the target column removed, so the hidden 2006 holdout should see the same pipeline.

## What mattered most (in order of discovery/impact)

1. **Bagged ensemble of deep XGBoost members with per-member randomization.** Single models
   overfit the 2005→2006 time shift very fast (30 trees 0.7171, 300 trees 0.7063). The fix that
   dominates everything else: average 36 members, each trained on a random 80% row sample with
   `colsample_bytree=0.5` and a different seed. colsample diversity alone was worth +0.0035
   (0.7186→0.7221) and unlocked everything after it.
2. **Depth scaling *inside* the ensemble.** The same depth that overfits as a single model is the
   main capacity source that averaging can exploit: d6→0.7245, d10→0.7328, d14→0.7419, d20→0.7486,
   d24→0.7514. Peak at d24 (d26 regressed).
3. **Light numeric feature engineering**: parse `c-<n>` strings to ints (Month/DayofMonth/DayOfWeek),
   decompose `DepTime` into Hour and minutes-of-day, add `log1p(Distance)` (+0.003 over raw-string baseline).
4. **Frequency (popularity) encoding** of Origin, Dest and Origin_Dest route: log train-counts as
   numeric columns, no label leakage (+0.0022, 0.7488→0.7510).
5. **Members-vs-rounds rebalance**: averaging beats boosting rounds here — 60→30→25 rounds per
   member at equal or better AUC, spending the saved time on more members (30→36).

## What did not help

- **Target encoding (OOF, smoothed) for carrier/origin/dest/route** — neutral for a single model
  (0.7176 vs 0.7177) and *harmful* in the deep ensemble (0.7455 vs 0.7486). The trees already learn
  the same signal from native categoricals + freq counts.
- **Route as a native categorical** (Origin_Dest string, ~2-3k levels): spurious splits on small
  groups (0.7088 at 30 trees vs 0.7171 without).
- **Seasonality/cyclicals**: sin/cos of hour/month/dow were dead weight, and `Yday`
  (month*31+day) was the single worst feature (-0.006): deep members memorized 2005-specific
  seasonal patterns that don't transfer to 2006.
- **Leaf/histogram regularization against the shift**: `min_child_weight=10` -0.009 (tiny deep
  leaves are where the signal lives), `max_bin=64` -0.005, `colsample_bylevel`, per-member
  `subsample=0.7`, `bag_frac=0.6`, more members with too few rounds (50×15: 0.7499).

## What I would try with more budget

The reliable axis on this task is averaging many cheap, deep, randomized members; the frontier was
set by the 120 s/run cap, not by diminishing returns on members. I would (a) push the member-count /
rounds / depth Pareto with a faster loop (e.g. `max_bin=128` for cheaper members) toward 60-100
members; (b) add *structural* member diversity — members trained on different feature subsets
(e.g. with/without freq encodings, hour-as-categorical vs numeric), and mixed-depth bags of equal-AUC
members; (c) build year-robust features instead of year-specific ones: carrier×hour and route×hour
aggregate delay patterns with heavy smoothing and minimum-support floors, exploiting that hour-of-day
physics transfers across years even when the calendar does not; (d) use a proper cross-year proxy
validation (e.g. month-blocked CV inside 2005) to select features, since eval-based selection at
+0.0003 granularity risks overfitting the 100k eval slice; and (e) re-test lightly-smoothed route-level
target/rate features with support floors, which was the strongest untested signal family.
