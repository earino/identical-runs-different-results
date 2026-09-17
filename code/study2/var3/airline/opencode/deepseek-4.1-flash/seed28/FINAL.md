# FINAL — airline delay XGBoost

**Best Eval AUC: 0.7483** (experiment #40, commit `8ddf3f7`, `train.py`).

Baseline was 0.7141. Final model: `XGBClassifier(n_estimators=1000, max_depth=18,
learning_rate=0.02, min_child_weight=1, subsample=0.8, colsample_bytree=0.3,
reg_lambda=1.0, tree_method="hist", enable_categorical=True)`, trained on 2005 and
evaluated on 2006 (time-separated). `predict_proba` reproduces every feature via
`prepare()` with all statistics/levels fit on `data/train.csv` only.

## Changes that mattered most

1. **Time-of-day decomposition** (`dep_hour`, `dep_minute`, `dep_tod`, cyclical
   sin/cos) + `doy`/`month`/`dow`. Delay probability is strongly monotonic in
   scheduled departure hour. +0.0024 over baseline (0.7141 → 0.7165).
2. **Normalized hourly traffic-share features** — `orig_hour_frac`,
   `dest_hour_frac`, `carrier_hour_frac` (pair count ÷ entity total), plus the raw
   pair/entity frequencies and day-of-year. The congestion ratios were the single
   best feature block: 0.7388 → 0.7413. Dropping all frequency features cost
   −0.006, confirming their value.
3. **Higher capacity on `min_child_weight=1` / large depth.** Small leaves plus
   deep trees mattered: 30 → ~1000 trees, depth 6 → 18, `min_child_weight` 5 → 1
   (0.7165 → 0.7388). The finer partitions capture high-order interactions
   (airport × hour × carrier × season).
4. **Aggressive column subsampling** (`colsample_bytree`) — lowering 0.8 → 0.6 →
   0.4 → 0.35 → 0.3 improved AUC at every step (0.7388 → 0.7483) and reduced
   runtime. Feature noise/regularization is strongly beneficial here.
5. Kept native XGBoost categorical handling for `UniqueCarrier`/`Origin`/`Dest`
   with levels fixed from train (unseen → NaN).

## What did NOT help

- **High-cardinality categoricals** `route`, `carrier_origin`, `carrier_dest`,
  and `origin_hour`-style interaction keys (0.7138) — massive overfit.
- **Out-of-fold target encoding** of carrier/airport/route/hour (0.7341) — native
  categorical splits already capture it.
- **Seed/ensemble averaging** (3× depth-14 or depth-16 members) gave at most
  +0.0002 and doubled runtime.
- **Global dep-hour/tod load counts**, dow/month density ratios, route
  popularity fractions, `max_bin=512`, `reg_lambda=0`, and
  `subsample=1.0`: all neutral-to-negative.

## With more budget

The strongest remaining signal is the partition structure of deep trees; the
biggest untapped idea is out-of-fold target encoding of *smoothed
airport×hour×season* cells (rather than raw counts), and a properly validated
(`train`-internal time split) early-stopping/seed-bagged version of the depth-18
model. I would also search `colsample_bytree`/`colsample_bynode` jointly and add
weather/holiday calendars, which are not derivable from the given columns. The
gap between train AUC (≈1.0) and eval (0.748) is dominated by genuine
2005→2006 distribution shift, so gains must come from smoother, more robust
features rather than more capacity.
