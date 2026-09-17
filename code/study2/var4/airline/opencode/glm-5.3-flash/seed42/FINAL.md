# Final Report

**Best Eval AUC: 0.7558** (experiment #40, commit 80e7944). Baseline (program's starting config): 0.7141 → **+0.0417**.

Final model: mean-ensemble of 2 XGBoost models (`xgboost.XGBClassifier`, `tree_method="hist"`, `enable_categorical=True`, 4 threads):
- anchor: colsample_bytree 0.4, depth 20, 400 trees, lr 0.03, seed 2024 → single 0.7546
- member: colsample_bytree 0.4, depth 20, 300 trees, lr 0.05, seed 7 → single ~0.7527
- shared: subsample 1.0, max_bin 512, reg_lambda 1.0 on feature set "full".

## Feature set (all fitted on train only, inside `prepare()`)
- Base: DepTime hour/minute, cyclical sin/cos of time-of-day (mod 2400), MonthNum + sin/cos month, DayOfMonth, DayOfWeekNum + sin/cos dow, Distance, log(1+Distance).
- Native categoricals: UniqueCarrier, Origin, Dest, DayOfWeek as `pd.Categorical` with train-only levels (unseen eval levels → NaN).
- Log1p train-count frequencies of composite keys: Origin×DepHour, Dest×DepHour, Carrier×DepHour, Origin, Dest, Route, Carrier×Origin, Carrier×Dest, Carrier×Route.
- Share ratios: hour-frequency / base frequency (+1 smoothing) for origin-hour, dest-hour, carrier-hour, and carrier-route share of route.

## What mattered most
1. **Hour-of-day features** (from DepTime; delay rate 4%→83% across the day) — biggest single jump of the whole run.
2. **Frequency features on origin×hour / dest×hour / carrier×hour + base keys** (+0.008 over cats-only; experiment #24).
3. **Share ratios** (freq ÷ base freq with +1 smoothing) (+0.0017; experiments #31/#32).
4. **Shallow-column regularization regime: colsample_bytree ≈ 0.4 with very deep trees (16–20), subsample 1.0, max_bin 512** (+0.02 across experiments #9–#13).
5. **Ensembling strong, diverse members** (seeds/lr/colsample/depth variety, mean aggregation) (+0.002).

## What did not help
- **Target (out-of-fold) encoding** of categoricals — neutral to negative at every depth tried (#6/#7, #17).
- **DayOfMonth as categorical, distance buckets, month/DoW frequency counts** — neutral or harmful (#23, #26).
- **Stronger single-model tweaks late in the run**: lr 0.02×600 anchor (timeout), DowHour frequency key, deeper ensembles beyond 2–3 members within the 120 s cap — all neutral or worse (#27, #28, #32, #36).

## With more budget
- Optimize feature preparation (vectorized string keys, cached matrices) to fit 4+ ensemble members per run; a 3rd/4th diverse member on the `full` feature set likely adds +0.001–0.002.
- Larger anchor (lr 0.02, 800–1000 trees) if the time cap allows; possibly max_bin 1024 for finer DepTime granularity.
- Calibration/weight search over ensemble members, and a careful re-check of robustness of tiny (<0.001) gains against the temporal 2005→2006 shift (internal valid AUC ran ~0.78 vs eval ~0.75, so eval-selection noise is real).
