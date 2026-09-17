# Final Report — Airline Departure Delay (XGBoost, autonomous run)

**Best Eval AUC: 0.7633** (baseline 0.7141, +0.049). Final commit `c69b7a7`, `validate.sh` → `CONTRACT OK`.
Budget used: 19 of 40 experiment slots, ~180 of 230 minutes, CPU at ~18000/18000 s.

## Final model
Ensemble of 7 XGBoost models, equal-weight average of probabilities, all statistics fit on `data/train.csv` only:
- 3 "TE" members (seeds 42/1/2, d8, lr 0.05, 800 trees, subsample/colsample 0.8, **recency-weighted**: weight ramps linearly 1→2 over 2005 months) over smoothed (K=30) **out-of-fold target encodings** of ~25 categorical structures × time, plus log traffic counts, plus numeric time/distance features.
- 2 "raw" members: native categoricals (carrier/origin/dest/dow) + numeric, depths 8 and 6 (hybrid d6 member additionally gets 5 key TEs).
- 2 "MoE" members: the TE feature set fit separately on morning (DepTime ≤ 12:00) and afternoon flights, each row scored by its segment's expert.

## Changes that mattered most
1. **Out-of-fold target encoding** (smoothed, K=30) of categorical structures — especially interactions with time-of-day: Origin/Dest/Carrier/Route × hour, later refined to **× 30-min and 20-min blocks of day** (RouteTod30/RouteTod20 were each worth ~+0.002 alone). OOF (not full-train) TE on the training rows was essential to avoid leakage-driven overfit.
2. **Dropping 2005-calendar features** (Month, DayofMonth) everywhere — the single biggest early jump (+0.007): calendar effects were one year's noise and did not generalize to 2006.
3. **Traffic-count features** (log1p of flights per Origin/Route/Carrier × hour and × 30-min block): +0.006 total. Airport/route/carrier congestion at the right time granularity is a real, generalizing driver.
4. **Within-hour minute-block TEs** (Min10/Min5): a small but exactly year-replicating pattern (+0.003).
5. **Diverse ensemble around the TE family**: native-categorical members, seed averaging, recency weighting of the 2005 sample, and the am/pm mixture-of-experts member (+0.005 combined over any single family).

## Things that did not help (all screened, then discarded)
1. **Native high-cardinality categoricals as tree features** (Route as a category: solo 0.709, ensemble −0.006) and, conversely, adding all TEs to the categorical members — each family should keep its own feature view.
2. **Learned stacking / weight tuning**: OOF-fitted logistic or NNLS blenders were much worse (0.752) than the equal-weight average; extra ensemble members beyond the final 6–7 prediction vectors diluted rather than added.
3. **Calendar and fine-calendar variants**: month/day-of-month TEs, quarter/seasonal encodings, DayOfWeek interactions (Origin×dow, Dest×dow), exact-minute TE — all flat to clearly negative.
4. Also negative/neutral: hierarchical TE shrinkage, per-cardinality adaptive K, K re-tunes (20–100), diff/ratio features, monotone constraints, lossguide, lr/depth/mcw/colsample sweeps once the ensemble was in place, 20-fold OOF TE, extra count features (dow-level traffic), Distance-bin TE (Distance is constant per route).

## Notes for reproducibility
- Tree counts fixed at 800 for all members (the train-only CV validation curves had been pegged at that plateau for several runs; hard-coding cut runtime 3× at equal AUC).
- Eval-vs-holdout caution: keep/discard decisions used `data/eval.csv`; only changes ≥ +0.001 were adopted as mechanisms (smaller margins were accepted only when they replicated in `train.py` itself, e.g. +0.0006 for the final TE additions).
- Two experiment slots were lost to crashes (XGBoost 3.x `eval_metric` API change; a pandas truth-value bug) — both fixed immediately and re-run.
