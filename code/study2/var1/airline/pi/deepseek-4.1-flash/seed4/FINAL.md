# Final report — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7366** (experiment #39, commit `9de0119`, 35-model XGBoost ensemble).
Baseline was 0.7141, so the loop added **+0.0225 AUC**. All 40 experiments were used.

## What mattered most

1. **Time-of-day feature engineering (largest single lever, ~+0.011).**
   `DepTime` (hhmm) was decomposed into `dep_hour`, `dep_min`, `dep_frac`, `is_night`, and —
   crucially — **categorical time bins**: hour (24 levels, +0.0031), 30-min (48 levels, +0.0056) and
   15-min (96 levels, +0.0027). Discrete time bins clearly beat the raw integer/continuous encoding.

2. **Diverse bagged ensemble of XGBoost models (~+0.004).**
   A single model plateaus near 0.7165; averaging 5 same-config models gave +0.0026, and growing to
   35 deliberately diverse members (depths 3–8, lr 0.03–0.05, subsample 0.5–0.9, colsample 0.4–0.9,
   varied `min_child_weight`, plus `grow_policy="lossguide"` members) reached 0.7366.

3. **Congestion / frequency-count features (~+0.003).**
   Counts of training rows per interaction — `origin×time`, `dest×time`, `carrier×time`, `route×time`,
   `carrier×origin`, `carrier×dest` — plus their ratios (`frac_*`) and plain entity counts
   (`cnt_carrier/origin/dest/route`). These are label-free, year-stable traffic/congestion proxies and
   transferred well from 2005 to 2006.

4. **Moderate capacity per model.** Depth 5 with `n_estimators=400`, `lr=0.05` was the sweet spot;
   ensembling then recovered any variance.

## What did NOT help (reverted)

1. **Raw high-cardinality categoricals**: `Route` (Origin_Dest) as an XGBoost categorical dropped AUC
   to 0.7029; one-hot encoding Origin/Dest dropped it to 0.7192 and was ~4× slower.
2. **Out-of-fold target encoding** of carrier/origin/dest/route (and route/time ratios): 0.7157 vs
   0.7165 at the time — the categorical handling plus count features already captured this signal, and
   the target drift across the 2005/2006 boundary hurts.
3. **Extra capacity / noisy features**: 500–900 trees at depth 7, 5-minute time bins, day-of-year
   cyclic features, weekend×time interactions, network-diversity features, and `max_cat_threshold=256`
   were all flat or worse.

## What I would try with more budget

The biggest remaining opportunity is **stacking**: train a second-level XGBoost on out-of-fold
predictions of the base ensemble (never on eval labels), which typically adds a few thousandths but
needs ~5× the compute we had per run. Beyond that: (a) a finer search around the 10–20 minute time-bin
resolution since the response is sharply peaked there; (b) per-airport/per-carrier temporal smoothing
of the congestion ratios rather than raw sample counts; (c) tuning XGBoost's categorical split
parameters (`max_cat_to_onehot`, `max_cat_threshold`) jointly rather than globally; and (d) a
calibrated blend where base models are weighted by out-of-fold performance instead of uniformly.
