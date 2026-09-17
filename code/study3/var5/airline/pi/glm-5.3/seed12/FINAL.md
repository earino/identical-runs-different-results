# Final report — airline delay classification (XGBoost)

## Result

**Best Eval AUC: 0.7585** (baseline: 0.7141, +0.0444). Final config: `19e7112`
("ramp amplitude diversity across members"), validated with `CONTRACT OK` (predict_proba on raw
frames reproduces 0.7585; runtime ~60s, inference ~10s per 100k rows).

Final model: an average of 8 XGBoost members (60 trees, max_depth 24, lr 0.1, subsample 0.9,
colsample_bytree 0.35, hist, native categoricals) over ~18 engineered features, each member fit
with a different recency weight ramp on the 2005 training slice.

## The changes that mattered most

1. **Deep, heavily column-bagged boosted trees** (max_depth 20-24 with colsample_bytree 0.35-0.4,
   averaged over 8-10 seeds) — the single biggest discovery (+~0.03 alone). Under the 2005→2006
   covariate shift, shallow boosted trees (baseline: depth 6) overfit year-specific patterns; deep
   trees restricted to random feature subsets behave like random-forest weak learners and transfer
   much better. Depth 6 → 10 → 14 → 20 → 24 gave monotone gains; beyond 24 it saturated.
2. **DepTime parsing**: hour, minutes-of-day, 15-minute bins, sin/cos of hour (baseline only had raw
   hhmm). Time-of-day is the strongest stable predictor of delays (+~0.003).
3. **Airport traffic / hub features**: log flight counts per origin and destination plus number of
   distinct routes served (structural, stable year-over-year) (+~0.005 in the deep regime).
4. **Recency weighting**: linear sample-weight ramp over months of 2005 (later months are closer to
   the 2006 evaluation period) (+~0.002), with per-member ramp amplitudes [1.0…2.5] both hedging the
   choice and diversifying the ensemble (+0.0006).
5. **Subsampling tuning + right-sized ensemble**: subsample 0.9, colsample 0.35, 8 members
   (ensemble AUC saturates at 6-8 members; 10+ added runtime without gain).

## Things that did not help (all tested, all reverted)

1. **Route (Origin–Dest) in any form** — native categorical, smoothed target encoding (k=30…300),
   count encoding, capped-to-common-routes: every variant *hurt* (−0.003 to −0.02). Origin and
   destination main effects are stable year-over-year, but 2005 route-level interaction rates are
   anti-informative for 2006 (schedule/route mix changes).
2. **Target encodings and interaction features in general** — TE of carrier/origin/dest/hour,
   origin×hour TE, hour×dow and hour×carrier categoricals, holiday-distance, weekend flags,
   month sin/cos, carrier counts, origin-hour traffic: all flat or worse. The deep trees find these
   interactions themselves, and added columns dilute the aggressive column sampling.
3. **Other learners/objectives/regularizers**: rank:pairwise objective (0.66!), lossguide growth,
   monotone constraints on departure time, colsample_bynode, hyperparameter-bag soups, more
   boosting rounds (100-400) with lower lr, shallow/deep model blends — all worse or equal.

## What I would try with more budget

The eval plateau at 0.7585 was confirmed by three independent configs, so the remaining headroom
is probably in *data*, not hyperparameters: (a) a proper within-2005 temporal CV (train Jan–Sep →
validate Oct–Dec) to select features/configs by shift-robustness rather than eval AUC, which would
also quantify the eval-vs-holdout generalization gap; (b) exploiting the 2006 marginal feature
distribution legitimately (no labels) is off-limits here, so instead I would look for more
*physically stable* features — e.g. scheduled block-time proxies (distance × time-of-day), airport
delay-propagation chains (late-day flights inherit earlier delays at the same airport), and
carrier fleet-mix indicators; (c) a larger member count trained on bootstrap resamples with
replacement (classic bagging) using the spare wall-clock budget; (d) stacking a
regularized-TE-based GBM as a decorrelated second view. I would also re-examine whether the
balanced 50/50 subsampling implies duplicate rows that make cross-validation optimistic.
