# Final report — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7324** (commit `e27ca36`, experiment #36; baseline was 0.7141).

## Setup

- Data: train = 100k rows from 2005, eval = 100k rows from 2006 (time-separated; the year shift is the
  dominant difficulty: same-year cross-validation reaches ~0.74, cross-year ~0.71 with default features).
- Final model: ensemble of 5 XGBoost classifiers (`n_estimators=1100, lr=0.04, max_depth=9,
  min_child_weight=5, subsample=0.9, colsample_bytree=0.8, reg_lambda=2`), native categorical handling,
  monotone-increasing constraint on the time-of-day feature.
- All feature engineering lives in `prepare(df)`; target encodings/count statistics are fitted on the
  training data only (out-of-fold encodings for training rows, full-train maps for unseen data).

## What mattered most

1. **Removing year-drifting features** — dropping `Month` and `DayOfMonth` (feature relationships that
   changed between 2005 and 2006) gave the single largest jump (+0.0037). The 2005 month pattern actively
   misled predictions on 2006.
2. **Hub-congestion count features** — log flight volume per (Origin, hour), (Dest, hour), (Origin, DOW),
   (Carrier, hour), symmetric-corridor route volume, computed from the training year (+0.0036). Stable,
   physical drivers of delay that transfer across years.
3. **Stable interaction target encodings** — smoothed TE of Origin/Dest/Carrier/DOW × hour-bucket,
   distance-bin × hour-bucket, plus main-effect TE for Origin/Dest/Carrier/Route with count encodings.
   Cross-year stability analysis (cell-rate correlation 2005 vs 2006) motivated exactly these pairs.
4. **Monotone time-of-day** — encoding scheduled departure as "minutes since 5am" with a monotone-increasing
   constraint captures the huge, stable diurnal delay curve (4% delay at 5am → 83% at midnight) without
   memorizing noisy small cells.
5. **Ensembling + moderate LR** — 5-seed average (+0.0015) and lr 0.04/1100 trees instead of 0.1/550 (+0.0016).

## What did not help

1. Deeper/longer training early on (3000 trees, depth 8, early stopping): internal 2005 AUC 0.754 but eval
   0.711 — the year shift punishes capacity fitted to 2005-specific structure.
2. Interaction TEs on a deep model (E6) and finer 3-hour buckets (E23), heavier TE smoothing (E24),
   hourly Origin/Dest×hour TE rates (E32/33), extra Dest×DOW/Route×hour volumes (E38): all neutral.
3. Diverse-depth ensembles (E27), coarse season feature (E28), forbidding Origin×Dest tree interactions
   (E29), stronger leaf regularization after the feature fixes (E35): no gain.

## With more budget

I would (a) run a proper random/hyperband search over depth/LR/regularization around the current point,
(b) try per-member feature-subset bagging and rank-averaging, (c) revisit drift with importance weighting
of 2005 rows toward the 2006 covariate distribution, and (d) test finer congestion features
(carrier×origin banks, 15-minute slots) — the congestion family was the last thing that moved the needle.
