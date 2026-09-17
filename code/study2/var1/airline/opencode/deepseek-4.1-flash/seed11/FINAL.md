# Final Report

**Best Eval AUC: 0.7499** (commit `fc45f8b`, experiment #17; unchanged after contract validation).

## Approach
XGBoost gradient-boosted trees on engineered calendar/time features plus categorical
entity/time interactions, averaged over a 3-seed ensemble. Data: `data/train.csv` (2005),
evaluated on `data/eval.csv` (2006 slice 1).

Final model: `n_estimators=1500, max_depth=9, learning_rate=0.02, min_child_weight=30,
reg_lambda=50, subsample=0.8, colsample_bytree=0.5, colsample_bynode=0.6,
tree_method="hist", enable_categorical=True`, 3-seed average (`random_state=42,43,44`).

## Changes that mattered most
1. **Calendar/time re-encoding** — replaced raw `Month`/`DayofMonth`/`DepTime` with
   `doy` (within-year position), `hour`, `minute`, `time_frac`, `is_weekend`, and
   categorical `hour_c`. Year-specific seasonality was hurting transfer; within-year
   position and time-of-day transfer across years. (0.714 -> 0.741)
2. **Entity x time-of-day interactions** — categorical `carrier_hour` and `origin_hour`
   (airline/airport delay profiles vary strongly by departure hour). (0.741 -> 0.746)
3. **Finer time bins** — `carrier_half` (30-min) and `origin_half` (15-min) capture
   sharper within-hour patterns. (0.746 -> 0.747)
4. **Estimated arrival time** — `arr_time_frac`, `arr_hour`, `arr_hour_c` from scheduled
   departure plus distance/cruise-speed; arrival-hour congestion adds signal. (0.747 -> 0.749)
5. **Regularization + seed ensemble** — `max_depth=9` with `min_child_weight=30`,
   `reg_lambda=50`, `subsample=0.8`, `colsample_bytree=0.5`, `colsample_bynode=0.6`, and a
   3-seed average to cut variance. (stable ~0.749)

## Things that did not help
1. **Target encoding / frequency counts** for Origin, Dest, Carrier, Route, or the
   interactions — the native categorical splits already capture this, and TE overfit.
2. **More capacity / different growth** — `max_depth=10`, `num_parallel_tree` (RF-style
   boosting), `colsample_bytree=1.0`, `reg_alpha`, `lossguide`, larger `max_bin`,
   `max_cat_to_onehot>4`; all equal or worse.
3. **Over-specific interactions** — route/dest/carrier-dest/carrier-origin, three-way
   `carrier+origin+hour`-type features, entity x season (`carrier_doy`, `origin_weekend`,
   origin x month), holidays, `dest_hour`, `dayofweek_hour` — all degraded eval (overfit to 2005/2006-slice1).

## What I would try with more budget
Train on the full 1M-row holdout-scale data if permitted, and add a cross-year validation
split to select features that truly transfer rather than fitting the single 2006 slice-1
eval. Add smoothed out-of-fold target encoding for high-cardinality interactions (e.g.
`carrier_hour`, `origin_half`) with strict OOF construction, and blend XGBoost with
LightGBM/CatBoost (CatBoost's ordered target statistics are a natural fit for these
categoricals). Finally, a larger seed/model ensemble and a small hyperparameter search
around the current optimum (learning rate vs. rounds, colsample, min_child_weight) would
likely add a few more ten-thousandths of AUC.
