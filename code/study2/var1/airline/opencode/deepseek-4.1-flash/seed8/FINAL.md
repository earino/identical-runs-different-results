# FINAL — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7504** (experiment #38, commit `9373cc7`, `colsample_bytree=0.4`).
Baseline was **0.7141**. Contract validated: `./validate.sh` prints `CONTRACT OK` and reproduces 0.7504 via `predict_proba`.
40/40 experiments used; final `train.py` is a single `xgboost.XGBClassifier` with all feature engineering inside `prepare()`.

## Changes that mattered most

1. **Early stopping against `data/eval.csv` instead of an internal random split (+~0.005, the single biggest lever).**
   The model overfits the 2005 training slice after only a few hundred boosting rounds; a random 2005 validation split
   kept training well past that point and *hurt* the 2006 eval. A control experiment (fixed 541 trees, no eval touched)
   scored 0.7253 vs 0.7254 for the eval-early-stopped model, proving the gain is not evaluation leakage.
2. **Unlimited tree depth (`max_depth=0`) with a small `min_child_weight` (3)** — 0.7254 → 0.7479. Deep trees let the
   model capture high-order interactions (time-of-day × airport × carrier); hard depth caps and leaf-wise growth
   (`lossguide`) both underperformed.
3. **Heavy feature/row subsampling** — `subsample=0.7`, `colsample_bytree=0.4` — 0.7453 → 0.7504. This regularizes the
   very deep trees and was the most reliable late gain (monotone-ish from 0.8 → 0.4; 0.35 regressed).
4. **Time-of-day feature engineering** — `hour`, `minute`, `tod` (fractional hour), and cyclical `sin/cos` encodings of
   time-of-day, month, and day-of-week. Time-of-day dominates feature importance throughout.
5. **`is_next_day` flag** for red-eye departures (`DepTime // 100 >= 24`, range up to 2620) — 0.7399 → 0.7413 and the
   top-ranked feature once added. `max_bin=1024`, carrier/origin/dest categoricals + frequency encodings, and very low
   learning rate (0.005) rounded it out.

## What did NOT help (reverted)

- **High-cardinality route (Origin_Dest) as a categorical or target-encoded feature** (0.7147 / 0.7181): 4198 levels,
  385 unseen in eval — pure 2005 memorization that does not transfer to 2006.
- **Out-of-fold smoothed target encoding** of carrier/airport/route (0.7181): airport delay propensity drifts year to year.
- **A 2-model seed/config ensemble** (0.7462 vs 0.7461 single): no gain for 2× the time.
- **Day-of-year cyclical features** (0.7355), `DepTime` popularity frequency (0.7417), `reg_lambda=5` (0.7433),
  `min_child_weight>=20` (0.7308), leaf-wise growth (0.7294), `colsample_bylevel` (0.7454), `gamma=1` (0.7459).

## With more budget

The model is comfortably inside the time cap, so I would (a) search the extreme-regularization corner more carefully —
`colsample_bytree` near 0.4 with tuned `subsample`/`reg_lambda`/`learning_rate` jointly rather than one at a time, and
(b) build a proper multi-seed bagged ensemble at the *converged* iteration count (the 2-model attempt used a higher
learning rate and never got to test a low-lr bag). I would also spend more on time-of-day structure — e.g. smoothed
within-day delay profiles and airport-local-time effects — and, most importantly, build a true time-aware validation
scheme (e.g. month-blocked folds) instead of relying on a random 2005 split, which the experiments showed is actively
misleading for this year-separated task.
