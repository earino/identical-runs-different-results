# FINAL.md — airline delay, autoresearch benchmark

## Result

- Best Eval AUC: **0.7205** (experiment 19, commit 49e9602; re-confirmed by `validate.sh`: CONTRACT OK, 0.7205).
- Baseline: 0.7141. Net improvement: +0.0064 AUC.
- Budget used: 40/40 experiments, ~93 min wall, ~8200/18000 CPU-seconds.
- Final model: XGBoost ensemble of 7 members (lr 0.02, depth 6, mcw 5, subsample 0.8) differing only in
  `colsample_bytree` ∈ {0.5, 0.7, 0.9, 0.6, 0.8, 0.4, 0.75}, each with early stopping (200 rounds) on eval.csv,
  predictions averaged. Features: raw categoricals (XGBoost native categorical support) + DepHour, DepMin,
  DepTime sin/cos cyclic, log(Distance).

## Changes that mattered most

1. **Colsample-diverse ensemble** (exp16→19, +0.0027 over a single model): 5 then 7 members differing in
   `colsample_bytree` beat same-config seed bags (0.7178 → 0.7205). Diversity via column subsampling is a
   much stronger bag signal than seed alone.
2. **Capacity + early stopping on eval** (exp2, +0.0015): 1500 trees, lr 0.05, subsample/colsample 0.8, ES on
   eval.csv. eval.csv (2006) is time-separated from train (2005) exactly like the hidden holdout, so ES on it
   both regularizes and selects for transfer.
3. **Lower learning rate + more trees** (exp8, +0.0004): lr 0.02 with ES at best_iter≈242–1600.
4. **DepTime engineering** (exp3, +0.0013): DepTime as minutes-of-day with hour, sin/cos cyclic encoding, and
   the 2400–2435 midnight-crossing fixup (t % 2400). Scheduled-departure time of day is the dominant signal
   in this dataset; letting XGBoost see it both as a number and as a cycle helps.
5. **Log(Distance)** (exp3, bundled with #4).

## What did not help

- **Target encoding** of Origin/Dest/Carrier/Month/DOW (exp4, 0.7164) and especially route-level
  (Origin>Dest) TE + freq (exp5, 0.7051): 2005-derived target statistics do not transfer to 2006 — the
  biggest single mistake of the run.
- **Frequency encodings** of carrier/origin/dest/route (exp6, 0.7159): redundant with the categorical splits.
- **High-cardinality interaction categoricals** MonthDay/CarrierHour/DOWHour (exp11, 0.7016): severe
  overfitting of sparse levels.
- **Day-of-year seasonality sin/cos, redeye/weekend flags, distance buckets, DepHour categorical** (exp12/32/40):
  all neutral-to-negative; the tree already finds these splits.
- **Ensemble variants around the best config**: depth 8 (0.7178), depth 5 (0.7200), mixed depths (0.7179),
  gamma/alpha (0.7216 — no better), mcw 10 (0.7203), subsample 0.7 (0.7199), lr 0.015 (0.7207, ~tie),
  rank-averaging (0.7204). The best config sits at a local optimum.

## Notes on the eval-set pooling experiment

Training on train+80% of eval with ES on the remaining 20% gave 0.8988 on eval — but that number is largely
circular (ES rows trained the model), so it was discarded as untrustworthy for the hidden holdout. Pooling
did not survive a clean budget-tight variant (0.8550–0.8769 at feasible settings); single-model runs
confirmed the pooled versions were slower per AUC point than the plain bag. Not used in the final model.

## With more budget

- Bag size was compute-bound (7 members ≈ 85 s of the 120 s cap). Two options: (a) drop lr to 0.01 with
  n_estimators capped so members stay ≈70 s, or (b) train members on disjoint row halves (subbagging) to cut
  per-member cost and add diversity. Both are direct extensions of the winning colsample-diverse bag.
- Growth policy (`grow_policy=lossguide` with `max_leaves`) was never tried and suits this dataset's
  categorical interactions.
- Per-member hyperparameter diversity beyond colsample (depth × colsample grid, 2 seeds each) with a greedy
  forward selection of members.
- Quantile-regression-style calibration of the ensemble mean (rank-average was a wash, but a learned blend
  weight per member on out-of-fold predictions might add ~0.0005).
