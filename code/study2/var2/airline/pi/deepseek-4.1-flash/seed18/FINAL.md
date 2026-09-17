# Final report — autoresearch XGBoost (airline)

## Result

- **Best Eval AUC: 0.7236** (commit `0efc6ac`, experiment #39).
- Contract validated: `./validate.sh` → `CONTRACT OK` (predict_proba reproduces 0.7236 on `eval.csv` with the target column removed).
- Model: an ensemble of 8 `xgboost.XGBClassifier` models (hist tree method, `enable_categorical=True`), probability-averaged. Base members are depth-4 / ~200–400 trees / lr 0.04–0.05 with subsample+colsample regularization; diversity comes from depth-3/5/6 and a lossguide member.
- Start point: baseline = 0.7141. Net gain = **+0.0095 AUC**.
- Budget: 40/40 experiments, ~1341 CPU-s of 18000, ~13 min wall clock of 230.

## Changes that mattered most

1. **`day_of_year` (+0.0020).** Month/DayofMonth/DayOfWeek were only categorical. Adding the absolute
   position in the year lets shallow trees fit the seasonal delay curve and holiday spikes (Thanksgiving,
   Christmas/New Year, July 4) without deep Month×Day interactions. This was the single largest jump.
2. **Congestion / frequency features and their ratios (cumulative ≈ +0.0025).** Counts of every
   Origin/Dest/route/carrier, then **Origin×hour, Dest×hour, Carrier×hour counts**, then the *shares*
   (`origin_hour_freq / origin_freq`, etc.). These capture airport peak-hour congestion and hub/route
   concentration, are target-free, and transfer across years. The share/ratio versions clearly beat raw counts.
3. **Conditional `carrier-at-airport-hour` share (+0.0005, final experiment).** `count(origin,carrier,hour) /
   count(origin,hour)` and its Dest analogue — how dominant a carrier is at that airport in that hour.
4. **Multi-model XGBoost ensemble (+0.0013 over the best single model).** Averaging 8 diverse, moderately
   regularized models was a reliable, robust gain.
5. **Shallow + regularized single model (0.7141 → 0.7167).** Depth 4, 200 trees, lr 0.05, subsample 0.8,
   colsample 0.8 beat both the 30-tree baseline and every higher-capacity variant.

## Things that did not help (reverted)

1. **Target encoding** of carrier/origin/dest/route, both full-train and out-of-fold: −0.0006 to −0.0100.
   2005→2006 target-statistic drift plus leakage made it strictly worse than XGBoost's own categorical splits.
2. **High-cardinality `route` as a raw categorical**: −0.0156 (massive overfit; dropping Origin/Dest entirely
   was even worse, −0.014, so airports are essential — just not as a 3000-level categorical).
3. **More capacity**: depth 5/6 with 300–500 trees, lower lr with more trees, `max_bin=512`, DART — all
   neutral to worse. The problem is distribution shift, not insufficient fit.
4. **Explicit `dep_tod`/`is_weekend`/`dow_n`, national hour shares, month×airport frequencies, route-hour
   conditional shares** — all neutral or negative once the count/share features were present.

## What I would try with more budget

The signal is nearly exhausted: gains after the feature work were ≤0.0005 and every structural change beyond
shallow trees regressed, which points at a low signal ceiling for these 8 columns. With more budget I would
(a) search the count/share feature space more systematically (e.g. lag-free rolling shares of route traffic by
month, since 2005→2006 traffic shifts are the main source of drift), (b) build a stacked ensemble whose
meta-learner is fit only on an internal 2005 time-split rather than on `eval.csv`, and (c) invest in
regularization/feature selection on the airport-categorical representation (e.g. one-hot for small airports,
merged rare levels) instead of adding more capacity. Any further tuning should be judged on a held-out slice
of 2005, since the last few eval improvements were within run-to-run selection noise.
