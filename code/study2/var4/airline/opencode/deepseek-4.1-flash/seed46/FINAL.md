# Final report — airline delay XGBoost

**Best Eval AUC: 0.7291** (commit `4f18b6b`, "ensemble: extend bag to 8 seeds"). Baseline was
0.7141, so the run improved eval AUC by +0.0150. 40/40 experiments used; `./validate.sh` printed
`CONTRACT OK` (AUC 0.7291 via `predict_proba` on `data/eval.csv`).

## Changes that mattered most

1. **Numeric / cyclical calendar features instead of categorical calendar.** Parsing `c-<n>` to
   numbers and adding `doy`, `doy_sin/cos`, `dow_sin/cos` (while dropping the categorical
   `Month/DayofMonth/DayOfWeek`) was worth ~+0.005 alone (0.7150 -> 0.7198 with the same model).
2. **Airport × hour crosses (`origin_hour`, `dest_hour`).** By far the largest single gain:
   0.7233 -> 0.7284 (+0.0051). Airport-at-hour local delay/congestion is the dominant
   transferable signal; it also generalized to 2006 despite the year shift.
3. **Ensembling many XGBoost models** averaged probability-space predictions: bagging over
   depths 3–9 with different seeds/subsample/colsample added ~+0.003 cumulatively over the best
   single model (0.7150 -> ~0.718 with 8 members, -> 0.729 with the full bag).
4. **Strong regularization on the deep members** (`min_child_weight` 10–50, `reg_lambda` 3–10,
   `subsample`/`colsample_bytree` 0.5–0.7, low `learning_rate`). The eval is time-separated
   (2005 -> 2006) and the model overfits quickly, so capacity without regularization is harmful.
5. **Carrier crosses (`carrier_hour`, `carrier_dow`)** added a smaller but consistent gain
   (0.7221 -> 0.7229).

## Things that did NOT help

1. **Target encoding** of Origin / Dest / Route / carrier (smoothing alpha=20). It leaked into the
   training fit and, more importantly, the 2005 delay propensities did not transfer to 2006:
   0.7012, a large regression. Frequency encoding was neutral.
2. **High-cardinality route categoricals** (`Origin_Dest`, `route_hour`), and `carrier_month` /
   `origin_dow` / `dest_dow` crosses. Month-conditioned features were especially bad (0.7143),
   consistent with the month seasonality shifting between 2005 and 2006.
3. **Time-of-day features** (`hour`, `minute`, `tod`, cyclical tod): consistently neutral-to-slightly
   negative alongside the raw `DepTime` integer, which already carries the signal.

## What I would try with more budget

The eval and the hidden holdout are separated in time, so the main enemy is distribution shift
(month/carrier effects move between years). I would pursue features that are structural rather than
year-specific: airport-hour congestion (already the big win) could be extended with delay
*propagation* features — e.g. cumulative prior departures and recent upstream delay at the same
origin/destination — built from the raw rows rather than target statistics. I would also try
training the final ensemble with a time-based validation split inside `train.csv` (rather than
selecting on `eval.csv`) to reduce eval overfitting, and I would prune the ensemble to the handful
of strongly-regularized deep members plus the low-depth variants, since bagging beyond ~38 members
had clearly plateaued (0.7291 at 8 seeds, no gain at 10).
