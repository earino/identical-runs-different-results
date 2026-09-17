# FINAL — airline delay XGBoost (autoresearch, scenario 2)

**Best Eval AUC: 0.7437** (experiment #38 config, kept at HEAD `a662134`).
Baseline was 0.7141 → +0.0296 (+4.1% relative).

## Final model

3-seed blend (42/137/2024) of XGBoost `hist` classifiers, each:
`max_depth=20, learning_rate=0.02, n_estimators=1200, min_child_weight=5,
subsample=0.9, colsample_bytree=0.4, gamma=0.3, reg_lambda=2.0, reg_alpha=0.1,
enable_categorical=True, early_stopping_rounds=100 (eval AUC)`.

Features (all inside `prepare()`, encoders fitted on train only):
- 6 raw categoricals as pandas categoricals (train-levels only; unseen → NaN)
- `Distance`, `DepTime` raw numerics + `log_dist`, `dist_bin` (10 buckets)
- time-of-day: `dep_hour`, `dep_norm`, `dep_sin`, `dep_cos`

## Changes that mattered most

1. **Engineered time-of-day features** (exp 3, 0.7141 → 0.7193): dep-delay probability ramps
   monotonically from ~4% at 5am to ~83% at 11pm; hour + sin/cos + clipped hhmm is the dominant signal.
2. **Very deep trees (d16→d20) at low learning rate (0.02)** (exp 21–29, up to 0.7436):
   deep unpruned trees + ES found more structure than depth 8; +0.004 over the d8 config.
3. **colsample_bytree 0.7 → 0.4** (exp 24–28, 0.7274 → 0.7417): the single largest lever.
   With only ~15 engineered columns, per-tree feature subsampling forces decorrelation
   and strong regularization; every step down helped until 0.3 (which regressed).
4. **subsample 0.8 → 0.9** (exp 28, +0.0011) — complementary to the tighter colsample.
5. **Multi-seed blending** (exp 11, +0.0014; 3-seed at d20 +0.0001–0.002) — small but consistent.

## What did not help

1. **Target encodings** (carrier/origin/dest/route, smoothed, OOF-encoded): eval 0.7176 (worse),
   while OOF-CV AUC rose 0.7663 → 0.7796 — pure 2005→2006 year-drift overfit.
2. **High-cardinality interaction categoricals** (route, Origin>Dest>DepTime): 0.7134 / timeout at 120s.
3. **Feature inflation** (freq counts, dep_bucket, calendar harmonics, 48 half-hour slots,
   more members/blends beyond 3, ES 150, gamma/mcw changes): all within ±0.003 of the kept config.

## With more budget

- A proper time-aware validation scheme (train on 2005 H1, validate on 2005 H2) to tune against
  drift instead of eval.csv, since OOF≈0.78 vs eval≈0.74 shows the shift is the binding constraint.
- Larger `max_leaves`/`lossguide` growth at the found colsample sweet spot, and a
  bagged ensemble (bootstrap-resampled trains) rather than seed-only diversity.
- Quarter-hour DepTime interactions with carrier (slot-level carrier delay profiles),
  under a strict training-only fitting rule.

## Cost

40 experiments (incl. 4 crashes/timeouts), ~176 min wall clock, ~9067 of 18000 CPU-seconds.
