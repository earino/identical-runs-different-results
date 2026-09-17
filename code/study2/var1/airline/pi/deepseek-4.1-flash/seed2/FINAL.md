# FINAL — autoresearch XGBoost (airline delay)

**Best Eval AUC: 0.7437** (experiment #39, commit `9a2600d`), up from the 0.7141 baseline.
Contract validated: `./validate.sh` printed `CONTRACT OK` (AUC via `predict_proba` on the target-stripped
frame: 0.7437).

## Final model

An ensemble of four `xgboost.XGBClassifier` models over a single `prepare(df)` feature pipeline:

- members `(depth, trees, lr)`: `(12, 600, 0.05)`, `(14, 500, 0.05)`, `(16, 350, 0.05)`, `(14, 500, 0.05)`
  (seeds `42..45`), predictions averaged.
- `tree_method="hist"`, `enable_categorical=True`, `max_bin=512`, `max_cat_to_onehot=512`,
  `max_cat_threshold=256`.

Features (all built inside `prepare`, with every mapping fit on `train` only):

- Numeric date parts parsed from the `c-<n>` strings: `Month`, `DayofMonth`, `DayOfWeek`; raw
  `DepTime`, `Distance` retained.
- `DepHour`, `DepMinutes`, `LogDistance`.
- Frequency (traffic) encodings: `Origin`, `Dest`, `Carrier`, `Route`, `Origin×DepHour`,
  `Carrier×DepHour` — log1p counts computed from train.
- Native categoricals: `UniqueCarrier`, `Origin`, `Dest`, `DayOfWeek`.

`Month` and `DayofMonth` are deliberately **not** used as features.

## Changes that mattered most

1. **Dropping Month/DayofMonth (−0.004 -> +0.004 swing).** Train is 2005, eval 2006. Seasonal
   target rates transfer poorly (Month's univariate eval AUC 0.52 vs 0.57 self), and removing these
   features lifted eval from 0.7141 to 0.7180 and stopped the model from fitting year-specific seasonality.
2. **Capacity: deeper trees / more boosting.** With the overfitting seasonal features gone, eval rose
   monotonically with model capacity: 30 -> 800 -> 2000 trees and depth 6 -> 8 -> 10 -> 12 -> 14 took
   the score from 0.718 to ~0.734. The task was underfit-limited, not variance-limited.
3. **Categorical one-hot (`max_cat_to_onehot=512`)** for Origin/Dest (282 levels each) plus
   `max_cat_threshold=256`: 0.7411 -> 0.7436. Partition splits on high-cardinality airports were
   unstable across the year gap; one-hot was both better and ~4× faster.
4. **Frequency/congestion features** (Origin×DepHour and Carrier×DepHour traffic counts): 0.7355 -> 0.7367.
   Stable, target-independent, and they transfer across years.
5. **`max_bin=512`**: 0.7367 -> 0.7379, finer numeric splits for `DepMinutes` and the frequency features.
6. **Depth-diverse seed ensemble** (4 members): a small, robust variance reduction on top of the above.

## Things that did NOT help

- **Target encoding** of carrier/origin/dest/route/DOW (0.7269 vs 0.7343): in-fold leakage made the model
  over-rely on it; native categoricals already captured the signal.
- **High-cardinality `Route` / `CarrierRoute` categoricals** (0.7115, and 0.7170 in the ensemble): strong
  overfitting across the year gap.
- **Explicit regularization** (`subsample`/`colsample_bytree`/`min_child_weight` at depth 12, 0.7251): the
  model was underfit, so regularization only hurt.
- **Dest×Hour / Route×Hour frequencies** (0.7288) and Carrier×Origin/DOW×Hour frequencies (no change).
- **Internal early stopping** (0.7107): the internal 2005 split selected a model that overfit 2005 relative
  to 2006; fixed high-capacity models with eval-driven choices did better.
- Depths beyond 16, `max_bin=1024`, and larger tree counts: all within noise (< 0.0002).

## What I would try with more budget

The score is dominated by departure time-of-day; the remaining headroom is in cross-year-stable structure.
I would (a) replace the hand-rolled frequency features with out-of-fold target/count encodings per
(Origin, DepHour) and (Route, DepHour) to capture airport-time delay propensity without leakage;
(b) test a low-rank / embedding-style encoding of airports and routes instead of one-hot, which may beat
one-hot while generalizing to the unseen airports present in eval (~7–8); and (c) build a proper
time-based validation split inside 2005 (train on early months, validate on late months) to select depth,
`max_bin` and the number of ensemble members without touching eval, since several of my keep/discard calls
rested on < 0.001 AUC differences that are within eval noise. Finally, a stacked/blended ensemble with a
small logistic meta-learner over the member probabilities (still XGBoost-only learners) is the most likely
next robust gain.
