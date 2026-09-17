# Final Report — airline XGBoost AUC

## Best result
- **Eval AUC: 0.7418** (experiment #22, commit `12ee80c`), validated: `CONTRACT OK`.
- Model: ensemble of two deep, low-learning-rate XGBoost classifiers over native categoricals plus
  smoothed out-of-fold target encodings, frequency encodings and structural network features.

## Changes that mattered most
1. **Deep trees with strong column subsampling.** Moving from depth-5/6 to depth 14–22 with
   `colsample_bytree≈0.30`, low `learning_rate≈0.012`, `subsample=0.8` raised single-model AUC from
   ~0.719 to ~0.731–0.740. Deep trees capture the high-order interactions (origin × hour × carrier)
   that shallow models miss; strong column subsampling keeps them from overfitting.
2. **OOF target encodings for origin×hour, carrier×hour and dest×hour (smoothing k=50).** Each added
   ~+0.002 once deep trees could exploit them. OOF encoding prevents the model from over-trusting its
   own labels; inference uses the full training-set map, so it generalizes to the hidden holdout.
3. **Frequency encodings for those same hour keys** (`freq_origin_hour`, `freq_carrier_hour`,
   `freq_dest_hour`) added ~+0.002. Flight-volume features are very stable across years (year-over-year
   correlation ≈0.99) and therefore transfer well to the 2006 holdout.
4. **Structural network features** (origin out-degree, dest in-degree, carriers per origin,
   origin×carrier share, route frequency) added ~+0.0015 and are essentially drift-free.
5. **Small deep ensemble (2 models, distinct seeds) with `reg_alpha=0.5`.** Averaging a depth-22 and a
   depth-18 model was both slightly more accurate and cheap enough to run in ~90 s.

## What did not help
1. **Shallow, high-colsample models** (depth 5–7, `colsample_bytree=0.8`) — the original baseline,
   plateauing near 0.719.
2. **Many extra target-encoding keys** (origin×month, dest×month, carrier×month, route, origin×dow) —
   all neutral or negative (route TE was clearly harmful); frequency/structural encodings were the
   robust substitutes.
3. **Large ensembles and seed-averaging beyond 2 members** — a 5-model / 3-model deep ensemble was no
   more accurate and risked the 120 s per-experiment timeout (one 5-model run timed out).
4. **Alternative growth policies** (`grow_policy=lossguide`, `max_leaves` up to 2048) were no better
   than plain depthwise growth.

## With more budget
The dominant issue is temporal drift between 2005 training data and 2006 evaluation/holdout. The
biggest remaining opportunity is to make the model more drift-robust rather than to add capacity:
periodic/rolling-window validation inside `train.py`, drift-aware sample weighting (down-weight old
months), and monotone or stabilized encodings (clipping/shrinking target-encoding maps toward the
prior more aggressively). Beyond that, a modest 3–4 model deep ensemble with a runtime budget for
~110 s, plus a careful search over `max_depth ∈ {18, 20, 22}` and `colsample ∈ {0.25, 0.28, 0.30}`,
could plausibly add another ~0.001–0.002 AUC without attacking the drift problem directly.
