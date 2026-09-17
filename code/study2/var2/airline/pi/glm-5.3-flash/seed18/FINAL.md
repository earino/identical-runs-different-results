# Final Report — airline delay XGBoost

**Best Eval AUC: 0.7324** (baseline 0.7141 → +0.0183). Final config: commit `88f1a96` (exp37),
validated with `CONTRACT OK` (predict_proba reproduces all features on raw frames, target dropped).

## Final model
Ensemble of 9 diverse XGBoost members (depth 5–8, varied subsample/colsample/seed), each trained twice —
at 0.4×R and 0.7×R rounds (R from one early-stopping fit on a stratified 10% val split) — 18 models,
probability-averaged. Every member sees a seeded random **60% subset of feature columns** (feature bagging).

## Changes that mattered most
1. **Per-member feature bagging (keep=0.6)** — +0.008 over the same ensemble without it (0.7246→0.7324).
   Per-member column subsets decorrelate members far more than per-tree colsample_bytree.
2. **Ensembling with dual round caps (0.4R + 0.7R)** — +0.003. Early stopping on 2005 data overshoots
   for 2006; blending a smoother (0.4R) and sharper (0.7R) view regularizes the year shift.
3. **Frequency encodings** (Origin/Dest/Carrier/route/dep_hour/carrier_hour counts) + time features
   (dep_hour categorical, cyclic tod sin/cos, day-of-year, log-distance) — +0.002 total over raw categoricals.
4. **Capacity + early stopping** (3000 trees @ lr 0.05, best-iteration refit on full data) — +0.001 over
   the 30-tree baseline.
5. Right-sized hyperparameters: depth 6 reference, light regularization (exp8's heavy reg and depth-9/10
   members at full rounds both hurt).

## What did not help
- **Target encoding in any form** (raw, smoothed, OOF 5-fold): −0.005 to −0.009. Airport/carrier delay
  rates drift too much between 2005 and 2006; self-leak even with OOF folds.
- **Covariate-shift importance weighting** (cross-fitted train-vs-eval discriminator weights): −0.002.
- **More members** (12/15, seed-bagged), structural diversity (lossguide, max_bins, lr variants),
  extra count features (carrier×origin, origin×dow, seasonal counts), feature pruning, log1p counts,
  rank-averaging: all neutral or slightly negative.

## With more budget
- Tune the feature-bagging fraction per member type and combine it with row bagging;
- stack members' OOF predictions with a small XGBoost meta-learner;
- search caps finer (0.3–0.6) with per-member cap scaling by depth;
- revisit smoothed TE with year-robust shrinkage (rank/quantile targets instead of raw rates).
