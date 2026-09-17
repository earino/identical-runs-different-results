# Final Report — airline delay XGBoost

**Best Eval AUC: 0.7329** (experiment #40, commit `2d61548`); baseline 0.7141 (+0.0188).
Evaluated on `data/eval.csv` (2006); hidden holdout is 2006 slice 2, so all choices favored
features/structures that are stable across the 2005→2006 time split rather than eval-specific fits.
The final `train.py` passes `./validate.sh` (`CONTRACT OK`, AUC 0.7329 via `predict_proba` with the
target column removed).

## Changes that mattered most

1. **Time-of-day categorical encoding.** Decomposing `DepTime` into hour/minute/`dep_min` lifted
   0.7141 → 0.7160, but the decisive jump came from treating departure time as discrete buckets:
   `hour_cat` (+0.0018) and especially `dep15_cat` (15-minute buckets, +0.0044). XGBoost's
   categorical partition splits capture non-smooth per-bucket time-of-day fixed effects far more
   efficiently than monotone splits on continuous `dep_min`. Final features keep `hour`,
   `dep_min`, 15-minute and distance categoricals, day-of-year seasonality, and an hour×weekend flag.
2. **Frequency / congestion features.** Counts and traffic-share ratios built from train only —
   carrier, origin, dest, route, and their (entity × hour) combinations. Origin-hour congestion
   frequency alone was +0.0022; hour traffic-share ratios added +0.0014. This is the main
   *structural* signal (airport/carrier/route busyness) and it transfers cleanly across years.
3. **Ensemble of diverse XGBoost models.** Probability-averaging 5 → 8 → 16 → 24 models spanning
   depths 5–8 and subsample/colsample settings gave a steady +0.001–0.002 with no single-model
   tuning risk (0.7162 → 0.7316 on the same features). Rank averaging was tried and was neutral.
4. **`max_cat_to_onehot=32`.** One-hot encoding low-cardinality categoricals (Month, DayOfWeek,
   hour, hour×weekend) while partition-splitting the rest gave a final +0.0012 → 0.7329.
5. **Seasonality + distance buckets.** Day-of-year sin/cos and 100-mile distance categoricals each
   added ~+0.0005.

## What did not help

- **Target encoding** (smoothed carrier/origin/dest/route target means): −0.010 AUC. Delay
  propensities for specific routes/airports are unstable across the 2005→2006 split, unlike
  frequency/busyness features.
- **High-cardinality route as a categorical** (−0.015) and **exact-minute categorical** (−0.014):
  both overfit 2005-specific idiosyncrasies. Finer than ~15-minute buckets degraded monotonically
  (5-min 0.7273, 30-min 0.7294 vs 15-min 0.7311).
- **More raw tree capacity / heavy regularization**: 600 trees (0.7121), depth-7 300-tree
  (0.7170), and gamma/reg_lambda/min_child_weight increases were all flat or worse; the ensemble,
  not individual complexity, was the productive axis. A DART ensemble timed out at 120 s.

## With more budget

I would move from eval-guided model selection to a proper year-aware validation protocol: build an
internal 2005-holdout or rolling time split inside `train.py` and optimize it directly, then reserve
`eval.csv` only for a final sanity check, reducing the risk of shipping an eval-specific 0.7329.
The most promising concrete directions are (a) a level-2 XGBoost stacker trained on out-of-fold
predictions from the 24 base models (with the same fast fit), (b) neighborhood/traffic features —
e.g. flights per origin per 15-minute window and time-since-previous-departure-style cascade
proxies — since total congestion was the strongest non-time signal, and (c) larger bagged
ensembles of 15-minute-bucketed models, which were still improving when the experiment budget ended.
