# Final Report

**Best Eval AUC: 0.7268** (baseline 0.7141, +0.0127). Final commit: `c063ddd`, `validate.sh` → `CONTRACT OK`.

## Final architecture

Equal... weighted blend of XGBoost members, all trained full-data on 2005 (`train.csv`), averaged at the
probability level with ascent-tuned member weights `[cs3bp=0, d4g=0.5, forest=1.25, cs3dt5=1.5, d4gp=0.75]`:

- **cs3bp** (w=0 after ascent): depth-4, lr 0.05, 400 trees, colsample_bytree 0.3, on COLS_PROD (base + smoothed train-fit delay-rate encodings + DepTime×Distance/doy products)
- **d4g** (w=0.5): depth-4, lr 0.03, 800 trees, gamma 2, on COLS (base + rates)
- **forest** (w=1.25): single-iteration extra-trees-style XGBoost — 20 parallel trees, lr 1.0, subsample 0.8, colsample_bynode 0.8, depth 4
- **cs3dt5** (w=1.5): cs3 config on COLS_PROD + 5-min DepTime bucket (`dt5`)
- **d4gp** (w=0.75): d4g config on COLS_PROD

Every member is a 5-seed average (seeds offset per member), 20 XGBClassifier models total, ~107 s wall.
All feature engineering lives in `prepare(df)`; rate tables (smoothed by K=75/100 toward the 2005 prior),
categorical levels and the dt5 level list are fit on train only, so `predict_proba(df)` reproduces the
pipeline on any raw frame (verified by `validate.py` with the target column removed).

## What mattered most

1. **Smoothed delay-rate encodings** of Month/carrier/Origin/Dest/departure-block (train-fit, K≈75–100): +~0.006 over raw categoricals; the single biggest feature win.
2. **Cyclical time features** (dep_min sin/cos, day-of-year sin/cos, 3-hour buckets) with shallow depth-4 trees: +~0.004; letting trees see near-continuous departure time beats hour-only buckets.
3. **Probability-level blending of structurally different members** (greedy shallow booster + slow high-gamma booster + extreme-randomized forest-style model): +~0.003 over the best single model (0.7231 → 0.7261 before seeding/weights).
4. **Seed averaging (5 seeds/member) + coordinate-ascent member weights**: +~0.0007 (0.7261 → 0.7268); ascent correctly down-weighted the correlated cs3bp/categorical-heavy members in favor of the diverse ones.
5. **DepTime treated as raw noisy integers** (plus log-distance and time×distance products) rather than wrapped mod-2400.

## What did not help

1. **Pair/interaction features** (Route, Origin×Month, carrier×hour rate encodings, pairwise rate K=300): consistently −0.01 to −0.02 — 2005→2006 shift makes pair statistics overfit; single-feature rates transfer, pairs don't.
2. **OOF stacking** (LogisticRegression / small XGB meta-learners on member out-of-fold predictions): 0.7228/0.7219 vs 0.7259 equal weights — meta-weights fit on 2005 don't transfer; equal blending won every time.
3. **More folds / recency weighting / frequency encodings / holiday flags / deeper trees / rank-averaging / DART**: all neutral-to-negative; 10-fold ≈ 5-fold, recency weights hurt, freq encodings −0.001, depth 6–8 −0.005.

## With more budget

I would attack the year-shift problem directly: build the rate encodings with hierarchical backoff
(airport rate shrunk toward its carrier-share-weighted expectation, or toward month rates) and validate
them on a *time-ordered* 2005→2006-style split rather than random folds, since random-split CV on 2005
(0.749) wildly overestimates 2006 performance (0.70–0.73) and misleads early-stopping decisions. I would
also try pairwise-rank (AUC-aligned) objectives as blend members with isotonic calibration to
probabilities, and an OOF-stacked blend where the meta-learner is fit on 2005 and sanity-checked for
coefficient stability across folds before acceptance.
