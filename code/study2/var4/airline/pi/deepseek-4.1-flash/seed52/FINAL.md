# FINAL — airline delay (XGBoost), experiment budget exhausted

**Best Eval AUC: 0.7350** (experiment #40, commit `4166989`; baseline was 0.7141).
Contract check: `./validate.sh` → `CONTRACT OK` (train.py reproducibly prints `Eval AUC: 0.7350` and
`predict_proba(df)` applies the full feature path to a raw DataFrame with the target removed).

## What mattered most (in order of impact)

1. **Within-dataset schedule-share features** (`share_{base}_{hour|dow|month}` and their reverse,
   plus route composition within origin/dest). These are scale-free conditional distributions of the
   flight schedule computed *inside* `prepare()` on whatever DataFrame is passed in. They encode
   airport/carrier bank structure and congestion, which is stable across the 2005→2006 year shift.
   This single direction moved AUC 0.7188 → 0.7320, by far the largest gain.
2. **Regularized, moderately shallow trees + averaging over a diverse ensemble.** A single deep model
   overfits the year shift (0.7091); min_child_weight-regularized depth-4..8 configs with
   subsample/colsample diversity averaged over 6 seeds × 5 configs gave a consistent, robust boost.
3. **Day-of-year and holiday features** (`doy`, sin/cos, holiday windows) added a small but real gain
   (0.7180 → 0.7188) by exposing ordered seasonality that the Month/DayOfMonth categoricals bake out.
4. **Out-of-fold target encodings of the stable airport/carrier-hour delay propensities** (with
   smoothing α=50 and 5-fold OOF for training rows, full-train map at inference). Added on top of the
   share features this gave +0.0010 (0.7339 → 0.7349).
5. **Ensemble size** (4 → 6 seeds) as a cheap final variance reduction (0.7349 → 0.7350).

## What did *not* help (reverted)

- **Raw route / high-cardinality categorical splits** and target-encoded route/origin/dest features:
  airport-level delay propensities correlate only ~0.26–0.38 between 2005 and 2006, so they overfit.
  Dropping Origin/Dest entirely was catastrophic (−0.017), but encoding their *schedule structure*
  rather than their historical delay rate was the winning move.
- **Stacking with a meta-XGBoost on out-of-fold base predictions** (0.7152) and **DART / lossguide
  variants**: no gain, and DART blew the 120 s limit.
- **Explicit time-of-day features** (hour, minutes, sin/cos), frequency encodings, joint-probability /
  lift / airport-network-size features, and categorical-split tuning (`max_cat_threshold`,
  `max_cat_to_onehot`): all neutral-to-negative — XGBoost already extracts that signal from the raw
  `DepTime`/`Distance` columns.

## With more budget

The clear next step is to push the "schedule structure" idea further: characterize each flight by its
position in the bank (time to the previous/next departure of the same origin, expected connection
volume from the inbound banks) rather than by simple hour shares, and mine joint
origin×destination×carrier×hour schedule densities with proper smoothing. I would also run a nested
(model-selection inside train) hyperparameter search instead of selecting on `eval.csv`, since the
last ~10 experiments were within ±0.001 of each other and I could not reliably distinguish real gains
from eval noise. Finally, a calibration/ranking objective or a second-level blend weighted by
validation folds might squeeze out another ~0.001, but the dominant remaining uncertainty is the
2009-style year shift, which no amount of feature engineering on this schema removes.
