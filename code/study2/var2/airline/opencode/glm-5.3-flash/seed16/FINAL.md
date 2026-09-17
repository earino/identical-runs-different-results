# Final report — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7316** (baseline 0.7141, +0.0175). Final model: single `XGBClassifier`,
`n_estimators=4400, learning_rate=0.05, max_depth=7, max_bin=64, colsample_bynode=0.3`,
`tree_method="hist", enable_categorical=True`, trained on all 100k rows of `data/train.csv`.
Runtime ~65s (well inside the 120s cap); `validate.sh` → `CONTRACT OK`.

## Changes that mattered most

1. **Categorical interaction features** (Origin+Dest route, UniqueCarrier/Origin/Dest × Hour,
   cats constrained to train-fitted levels so unseen 2006 levels map to NaN): flipped the
   capacity regime — after this, deeper trees (depth 7) and many more trees started to help
   (0.7160 → 0.7233 with depth 7, lr 0.05, 3200–4400 trees).
2. **`colsample_bynode` decorrelation sweep** (1.0 → 0.9 → … → 0.3): the single biggest win,
   0.7240 → 0.7316. Re-sampling 30% of features at every split regularizes without shrinking
   data per tree (unlike `subsample`/`colsample_bytree`, which both hurt) and made training
   faster as a side effect.
3. **Learning-rate / tree-count scaling**: lr 0.05 with 1600→4400 trees gave steady gains
   (0.7214 → 0.7240); `max_bin=64` bought tree count within the time cap.
4. **Cyclical departure-time features** (hour, minute, sin/cos of time-of-day): modest but
   real; ablating them cost time without helping.
5. **Selection discipline**: 2005-holdout AUC rises with capacity while 2006 AUC falls, so all
   keep/discard decisions used eval-2006 plateaus, not in-sample valid splits.

## Things that did not help

- **Target encoding** (smoothed in-sample and proper 5-fold out-of-fold): both hurt
  (0.7148 / 0.7186 vs 0.7162 at the time); 2005-estimated rates don't transfer well.
- **Row/bag regularization**: `subsample` 0.7 (0.7006), `colsample_bytree` 0.7 (0.7134),
  seasonality combos Origin/Dest×Month (0.7114), carrier×DayOfWeek (0.7210) — all worse.
- **2-model ensemble** of n=2000 members (0.7213) and **lossguide** growth (0.7177): member
  quality dominated; a single big model beat both under the 120s cap.

## With more budget

Continue the `colsample_bynode` descent below 0.3 and jointly re-scale tree count (runtime
fell to 65s, so 5000+ trees fit); revisit a *deep* ensemble (2×n4400 needs ~130s — only
feasible with faster bins or a larger cap); retry OOF target encoding at the current capacity
level (it may only pay off once the model is strong); explore per-airport delay-prior features
from train-only statistics with heavier smoothing.
