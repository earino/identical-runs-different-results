# Final report — airline delay XGBoost

**Best Eval AUC: 0.7230** (baseline 0.7141, +0.0089). Final model: ensemble of 32 XGBoost classifiers
(8 depths 3–10 x 4 seeds) with lr/n_estimators/colsample_bynode diversity, averaged probabilities.
Features: raw DepTime/Distance + Month/DOM/DOW/Carrier/Origin/Dest as categoricals + out-of-fold
target encodings (Origin, Dest, Carrier, route) + frequency counts + time-of-day features.

## Changes that mattered most
1. **colsample_bynode=0.25 ensemble regime** (+0.003 over 0.7183 → 0.7212): feature subsampling per
   split, swept 0.15–0.5; extreme sparsity decorrelates trees strongly. Final: per-model mix 0.2/0.25/0.3/0.35.
2. **Hyperparameter-diverse ensemble, proba-averaged** (0.7141 → 0.7230 cumulative): 32 models over
   depths 3–10 x 4 seeds with learning-rate (0.05/0.1/0.15/0.2) and n_estimators (20/30/40/50) cycles.
   Depth mix alone gave +0.0015; lr and n_estimators diversity each +0.0005–0.001.
3. **OOF target encoding** of Origin/Dest/Carrier/route (smoothing m=30, 5-fold, leak-free train matrix,
   full-train maps for inference): +0.0003 at baseline capacity; essential compact signal for high-card cats.
4. **Time-of-day features + frequency counts + dep>=2400 flag** (+0.0002): hour, tod sin/cos, freq of
   Origin/Dest/Carrier/route/hour.
5. **Staying small**: every capacity increase (more trees per model, deeper, less regularization) hurt —
   train is 2005, eval/holdout 2006; cross-year drift punishes complexity. 30 trees/depth<=8 per model is
   the sweet spot; ensembling beats enlarging.

## Things that did not help
- Interaction target encodings (carrier x hour, origin x dow, origin x month, m=50): 0.7087 (−0.006).
- Dropping Origin/Dest categorical columns in favor of TE only: 0.7171 (−0.006).
- Early stopping on a random internal split (2000 trees): 0.7057 — val split is same-year, cannot see drift.
- Rank-averaging instead of probability-averaging: identical AUC. TE smoothing m=15 vs 30: neutral.
- Seed-only replication at fixed depth (ens24), hp-grid soup (24 models), subsample/bytree/reg_lambda
  diversity, max_bin=512, n_estimators scaling to 64: all neutral or slightly worse.

## With more budget
- Stacking: XGB meta-model on 5-fold OOF ensemble predictions + base features (drift risk, untested).
- OOF-AUC-weighted ensemble instead of uniform averaging.
- Origin/Dest geo enrichment (state/region from airport code prefixes) and route-level distance-ratio features.
- Careful per-feature TE smoothing sweep (m per column by cardinality) and month-TE for seasonality.
