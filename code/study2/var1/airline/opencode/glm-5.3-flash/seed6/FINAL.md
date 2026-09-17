# Final Report — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7477** (baseline 0.7141). Final `train.py` = 2-member XGBoost bag,
depth 20, lr 0.03/0.04, subsample 0.8, colsample_bytree 0.7, colsample_bynode 0.9,
logloss early stopping on an 80/20 train split, refit on 100% train; features: parsed
c-<n> columns, DepTime hour/min + cyclical sin/cos + hour-as-categorical, native
categoricals for carrier/origin/dest. Runtime ~82s (safety margin vs the 120s cap;
the 1M-row holdout prediction path was also optimized via a shared DMatrix).

*(The single best eval AUC observed was 0.7482 with a 3-member bag, but at 114s/120s
runtime it risked a scoring timeout; the kept model gives up 0.0005 for that safety.)*

## Changes that mattered most
1. **DepTime feature engineering** (hour, minute, cyclical sin/cos, numeric c-cols):
   0.7141 → 0.7255 — the largest single step.
2. **Deep trees with heavy subsampling**: depth 8→16→20 (with subsample/colsample):
   0.7255 → 0.7450. Depth 16–20 dominates; the optimum subsampling shifted with depth.
3. **dep_hour as native categorical** (replacing reliance on numeric hour splits):
   0.7455 → 0.7482/0.7477 (+0.0027).
4. **Seed/learning-rate bagging** (averaging 2–3 XGBoost members): +0.001–0.002,
   consistent across configs.
5. **Low learning rate (0.03) + logloss early stopping + refit on full train**:
   +0.0016 over lr 0.05, and more stable than AUC-based stopping.

## What did not help
1. **Target/frequency encoding** (OOF, smoothed) of carrier/origin/dest/route: +0.0003
   over native categoricals — dropped for simplicity; **interaction TEs** (carrier×hour,
   origin×month, carrier×origin) actively hurt (-0.003).
2. **AUC-metric early stopping**: probes 966 trees vs 291 (logloss) but evals at
   0.7358 — within-2005 validation is biased toward more trees than the 2005→2006
   shift allows. Same reason 1.2× refit trees lost.
3. **Structural/regularization variants**: lossguide growth (-0.008), min_child_weight
   10 (-0.007), reg_lambda 5 (timeout via longer early-stop), route-as-categorical
   (-0.02, severe overfit), max_bin 512 (neutral, slower), dropping sin/cos (-0.003).

## With more budget
- A 4–5 member bag with lr ∈ [0.03, 0.05] and per-member feature/row subsampling,
  once runtime is tamed (smaller probe, shared iteration counts).
- Careful low-smoothing TEs of *temporal* interactions (hour×day-of-week, origin×month)
  with year-robust validation (e.g., month-held-out folds) rather than random folds.
- A nested validation scheme that mimics the year shift (fit on 2005-H1, validate on
  2005-H2) to pick tree counts; the random-split early stop was the clearest miscalibration.
