# Final report — airline delay AUC

**Best Eval AUC: 0.7379** (final commit `620d9df`; baseline 0.7141, +0.0238).

Final model: bag of 3 XGBoost `lossguide` classifiers (seeds 42–44), each `n_estimators=2000, lr=0.015,
max_leaves=160, min_child_weight=20, subsample=0.85, colsample_bytree=0.4, reg_lambda=1, reg_alpha=4,
gamma=2, max_bin=1024, max_depth=0`, probabilities averaged. Features: int calendar (month/dow/dom),
sin/cos cyclical encodings of month/dow/departure-time, dep_minutes/dep_hour (red-eye DepTime>2400
wrapped mod 24h), distance + log1p(distance), carrier/origin/dest as pandas categoricals (unseen levels
→ NaN). All engineering inside `prepare()`; encoders fit on train only.

## Changes that mattered most
1. **Lossguide growth** (`max_depth=0, max_leaves=96→160`) instead of depthwise depth-10: +0.0026 — asymmetric
   trees capture hour×carrier×airport conjunctive delay rules far more efficiently.
2. **Column subsampling + strong split regularization**: colsample_bytree 0.6→0.4 (+0.0021), then
   `reg_alpha=4` (+0.0022) and `gamma=2` (+0.0015) — the time-separated 2005→2006 split rewards heavy
   regularization; alpha/gamma synergized.
3. **Cyclical time features** (sin/cos of month, day-of-week, time-of-day): +0.0024 vs dropping them.
4. **max_bin 512→1024**: +0.0010 (finer split points for dense numeric features).
5. **Bagging 3 seeds** (+0.0004) and lower lr with more trees (lr 0.02×1500 → 0.015×2000: +0.0014).

## What did not help
- **Out-of-fold target encoding** of carrier/origin/dest/route (0.7173 vs 0.7184): 2005-level delay rates
  drift by 2006; noisy encoded features distracted splits.
- **Route categorical / interaction categoricals** (carrier×hour, dow×hour): 0.7067 / 0.7204 — ~20k-level
  sparse categories overfit.
- **dayofyear** (month*31+dom): 0.7303 — redundant, deceptive axis. Frequency (count) encodings and
  ensembling with a shallow member were neutral-to-negative; early stopping per member (0.7329) lost to
  fixed 2000 trees (0.7340).

## With more budget
Deeper OOF-free alternatives to level statistics: smoothed *global-prior-only* encodings per airport pair
under heavy shrinkage; per-member feature-subset diversity in the bag; a 4–5 member bag at max_bin=1024
(now affordable at ~60s/model); Optuna-style search over (leaves, mcw, cs, alpha, gamma) jointly rather
than coordinate-wise; and a time-ordered internal validation scheme to select tree count without touching
eval.csv.
